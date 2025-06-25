// Copyright 2024 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.
// You may obtain a copy of the license at
//
//     http://www.apache.org/licenses/license-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the license is distributed on an "as is" basis,
// without warranties or conditions of any kind, either express or implied.
// See the license for the specific language governing permissions and
// limitations under the license.

use crate::icosphere::make_weights;
use crate::{
    make_icosphere, make_vertices, table::PaddedTable, IcoSphere, SphericalCoord, Vector3, Vertex,
};
use anyhow::{bail, Result};
use core::f64::consts::PI;
use get_size::GetSize;
use itertools::Itertools;
use nalgebra::Matrix3;
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, OnceLock};

/// A 4D icotable where each vertex holds an icotable of floats
pub type IcoTable4D = IcoTable2D<IcoTable2D<f64>>;

/// A 6D table for relative twobody orientations, R → 𝜔 → (𝜃𝜑) → (𝜃𝜑)
///
/// The first two dimensions are radial distances and dihedral angles.
/// The last two dimensions are polar and azimuthal angles represented via icospheres.
/// The final `f64` data is stored at vertices of the deepest icospheres.
pub type Table6D = PaddedTable<PaddedTable<IcoTable4D>>;

/// Represents vertex indices defining the three corners of a face
pub type Face = [usize; 3];

/// Icosphere table
///
/// This is used to store data on the vertices of an icosphere.
/// It includes barycentric interpolation and nearest face search.
///
/// https://en.wikipedia.org/wiki/Geodesic_polyhedron
/// 12 vertices will always have 5 neighbors; the rest will have 6.
///
/// Vertex positions and neighbor information are available through a shared pointer.
/// To enable concurrent write access, data is wrapped with `OnceLock` which allows
/// setting data only once.
#[derive(Clone, GetSize)]
pub struct IcoTable2D<T: Clone + GetSize> {
    /// Reference counted pointer to vertex positions and neighbours
    /// We want only *one* copy of this, hence the ref. counted, thread-safe pointer
    /// Each vertex corresponds to a position on the unit-sphere and can be converted
    /// to a spherical coordinate (θ, φ), i.e. a 2D coordinate.
    #[get_size(size = 8)]
    vertices: Arc<Vec<Vertex>>,
    /// Vertex information (position, data, neighbors)
    #[get_size(size_fn = oncelock_size_helper)]
    data: Vec<OnceLock<T>>,
}

fn oncelock_size_helper<T: GetSize>(value: &Vec<OnceLock<T>>) -> usize {
    value.get_size() + std::mem::size_of::<T>() * value.len()
}

impl<T: Clone + GetSize> IcoTable2D<T> {
    /// Iterator over vertices `(positions, neighbors)`
    pub fn iter_vertices(&self) -> impl Iterator<Item = &Vertex> {
        self.vertices.iter()
    }
    /// Get i'th vertex position (normalized)
    pub fn get_normalized_pos(&self, index: usize) -> Vector3 {
        self.vertices[index].pos.normalize()
    }
    /// Get i'th data or `None`` if uninitialized
    pub fn get_data(&self, index: usize) -> &OnceLock<T> {
        &self.data[index]
    }
    /// Get i'th neighbors
    pub fn get_neighbors(&self, index: usize) -> &[u16] {
        &self.vertices[index].neighbors
    }
    /// Get i'th vertex normalized position; neighborlist; and data
    pub fn get(&self, index: usize) -> (&Vector3, &[u16], &OnceLock<T>) {
        (
            &self.vertices[index].pos,
            &self.vertices[index].neighbors,
            self.get_data(index),
        )
    }
    /// Iterate over vertex positions
    pub fn iter_positions(&self) -> impl Iterator<Item = &Vector3> {
        self.iter_vertices().map(|v| &v.pos)
    }
    /// Iterate over vertex positions; neighborlists; and data
    pub fn iter(&self) -> impl Iterator<Item = (&Vector3, &[u16], &OnceLock<T>)> {
        (0..self.data.len()).map(move |i| self.get(i))
    }

    /// Generate table based on an existing vertices pointer and optionally set default data
    pub fn from_vertices(vertices: Arc<Vec<Vertex>>, data: Option<T>) -> Self {
        let num_vertices = vertices.len();
        let data = data.map(|d| OnceLock::from(d));
        Self {
            vertices,
            data: vec![data.unwrap_or_default(); num_vertices],
        }
    }

    /// Generate table based on an existing subdivided icosaedron
    fn from_icosphere_without_data(icosphere: &IcoSphere) -> Self {
        if log::log_enabled!(log::Level::Debug) {
            vmd_draw(Path::new("icosphere.vmd"), icosphere, "green", Some(10.0)).unwrap();
        }
        Self {
            vertices: Arc::new(make_vertices(icosphere)),
            data: vec![OnceLock::default(); icosphere.raw_points().len()],
        }
    }

    /// Generate table based on an existing subdivided icosaedron
    pub fn from_icosphere(icosphere: &IcoSphere, default_data: T) -> Self {
        let table = Self::from_icosphere_without_data(icosphere);
        table.set_vertex_data(|_, _| default_data.clone()).unwrap();
        table
    }

    pub fn angle_resolution(&self) -> f64 {
        let n_points = self.data.len();
        (4.0 * std::f64::consts::PI / n_points as f64).sqrt()
    }

    /// Number of vertices in the table
    pub fn len(&self) -> usize {
        self.vertices.len()
    }

    /// Check if the table is empty, i.e. has no vertices
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Set data associated with each vertex using a generator function
    /// The function takes the index of the vertex and its position
    /// Due to the `OnceLock` wrap, this can be done only once!
    pub fn set_vertex_data(&self, f: impl Fn(usize, &Vector3) -> T) -> anyhow::Result<()> {
        if self.data.iter().any(|v| v.get().is_some()) {
            bail!("Data already set for some vertices")
        }
        self.iter().enumerate().try_for_each(|(i, (pos, _, data))| {
            let value = f(i, pos);
            if data.set(value).is_err() {
                bail!("Data already set for vertex {}", i);
            }
            Ok(())
        })
    }

    /// Discard data associated with each vertex
    ///
    /// After this call, `set_vertex_data` can be called again.
    pub fn clear_vertex_data(&mut self) {
        for data in self.data.iter_mut() {
            *data = OnceLock::default();
        }
    }

    /// Get data associated with each vertex
    pub fn vertex_data(&self) -> impl Iterator<Item = &T> {
        self.data.iter().map(|v| v.get().unwrap())
    }

    /// Transform vertex positions using a function
    /// TODO: Better to use Mutex or RwLock for this? We normally do not want to
    /// touch the vertices and this is used for testing interpolation schemes only
    pub fn transform_vertex_positions(&mut self, f: impl Fn(&Vector3) -> Vector3) {
        let new_vertices = self
            .iter_vertices()
            .map(|v| {
                let pos = f(&v.pos);
                let neighbors = v.neighbors.clone();
                Vertex { pos, neighbors }
            })
            .collect_vec();
        self.vertices = Arc::new(new_vertices);
    }

    /// Get projected barycentric coordinate for an arbitrary point
    ///
    /// See "Real-Time Collision Detection" by Christer Ericson (p141-142)
    pub fn barycentric(&self, p: &Vector3, face: &Face) -> Vector3 {
        let (a, b, c) = self.face_positions(face);
        // Check if P in vertex region outside A
        let ab = b - a;
        let ac = c - a;
        let ap = p - a;
        let d1 = ab.dot(&ap);
        let d2 = ac.dot(&ap);
        if d1 <= 0.0 && d2 <= 0.0 {
            return Vector3::new(1.0, 0.0, 0.0);
        }
        // Check if P in vertex region outside B
        let bp = p - b;
        let d3 = ab.dot(&bp);
        let d4 = ac.dot(&bp);
        if d3 >= 0.0 && d4 <= d3 {
            return Vector3::new(0.0, 1.0, 0.0);
        }
        // Check if P in edge region of AB, if so return projection of P onto AB
        let vc = d1 * d4 - d3 * d2;
        if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
            let v = d1 / (d1 - d3);
            return Vector3::new(1.0 - v, v, 0.0);
        }
        // Check if P in vertex region outside C
        let cp = p - c;
        let d5 = ab.dot(&cp);
        let d6 = ac.dot(&cp);
        if d6 >= 0.0 && d5 <= d6 {
            return Vector3::new(0.0, 0.0, 1.0);
        }
        // Check if P in edge region of AC, if so return projection of P onto AC
        let vb = d5 * d2 - d1 * d6;
        if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
            let w = d2 / (d2 - d6);
            return Vector3::new(1.0 - w, 0.0, w);
        }
        // Check if P in edge region of BC, if so return projection of P onto BC
        let va = d3 * d6 - d5 * d4;
        if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
            let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            return Vector3::new(0.0, 1.0 - w, w);
        }
        // P inside face region.
        let denom = 1.0 / (va + vb + vc);
        let v = vb * denom;
        let w = vc * denom;
        Vector3::new(1.0 - v - w, v, w)
    }

    /// Get barycentric coordinate for an arbitrary point on a face
    ///
    /// - Assume that `point` is on the plane defined by the face, i.e. no projection is done
    /// - https://en.wikipedia.org/wiki/Barycentric_coordinate_system
    /// - http://realtimecollisiondetection.net/
    /// - https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
    pub fn naive_barycentric(&self, p: &Vector3, face: &Face) -> Vector3 {
        let (a, b, c) = self.face_positions(face);
        let ab = b - a;
        let ac = c - a;
        let ap = p - a;
        let d00 = ab.dot(&ab);
        let d01 = ab.dot(&ac);
        let d11 = ac.dot(&ac);
        let d20 = ap.dot(&ab);
        let d21 = ap.dot(&ac);
        let denom = d00 * d11 - d01 * d01;
        let v = (d11 * d20 - d01 * d21) / denom;
        let w = (d00 * d21 - d01 * d20) / denom;
        let u = 1.0 - v - w;
        Vector3::new(u, v, w)
    }

    /// Get the three vertices of a face
    pub fn face_positions(&self, face: &Face) -> (Vector3, Vector3, Vector3) {
        (
            self.get_normalized_pos(face[0]),
            self.get_normalized_pos(face[1]),
            self.get_normalized_pos(face[2]),
        )
    }

    /// Find nearest vertex to a given point
    ///
    /// This is brute force and has O(n) complexity. This
    /// should be updated with a more efficient algorithm that
    /// uses angular information to narrow down the search.
    ///
    /// See:
    /// - https://stackoverflow.com/questions/11947813/subdivided-icosahedron-how-to-find-the-nearest-vertex-to-an-arbitrary-point
    /// - Binary Space Partitioning: https://en.wikipedia.org/wiki/Binary_space_partitioning
    pub fn nearest_vertex(&self, point: &Vector3) -> usize {
        self.iter_vertices()
            .map(|v| (v.pos.normalize() - point).norm_squared())
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0
    }

    /// Find nearest face to a given point
    ///
    /// The first nearest point is O(n) whereafter neighbor information
    /// is used to find the 2nd and 3rd nearest points which are guaranteed
    /// to define a face.
    pub fn nearest_face(&self, point: &Vector3) -> Face {
        let point = point.normalize();
        let nearest = self.nearest_vertex(&point);
        let face: Face = self
            .get_neighbors(nearest)
            .iter()
            .cloned()
            .map(|i| {
                (
                    i,
                    (self.get_normalized_pos(i as usize) - point).norm_squared(),
                )
            })
            .sorted_by(|a, b| a.1.partial_cmp(&b.1).unwrap()) // sort ascending
            .map(|(i, _)| i as usize) // keep only indices
            .take(2) // take two next nearest distances
            .collect_tuple()
            .map(|(a, b)| [a, b, nearest]) // append nearest
            .expect("Face requires exactly three indices")
            .iter()
            .copied()
            .sorted_unstable() // we want sorted indices
            .collect_vec() // collect into array
            .try_into()
            .unwrap();

        assert_eq!(face.iter().unique().count(), 3);
        face
    }
}

impl std::fmt::Display for IcoTable2D<f64> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "# x y z θ φ data")?;
        for (pos, _, data) in self.iter() {
            let spherical = SphericalCoord::from_cartesian(*pos);
            writeln!(
                f,
                "{} {} {} {} {} {:?}",
                pos.x,
                pos.y,
                pos.z,
                spherical.theta(),
                spherical.phi(),
                data
            )?;
        }
        Ok(())
    }
}

impl IcoTable4D {
    /// Get flat iterator that runs over all pairs of (&pos_a, &pos_b, &OnceLockf64)
    pub fn flat_iter(&self) -> impl Iterator<Item = (&Vector3, &Vector3, &OnceLock<f64>)> {
        self.iter().flat_map(|(pos_a, _, data_a)| {
            data_a
                .get()
                .unwrap()
                .iter()
                .map(move |(pos_b, _, data_b)| (pos_a, pos_b, data_b))
        })
    }
    /// Generate table based on a minimum number of vertices on the subdivided icosaedron
    pub fn from_min_points(min_points: usize, default_data: IcoTable2D<f64>) -> Result<Self> {
        let icosphere = make_icosphere(min_points)?;
        let vertices = Arc::new(make_vertices(&icosphere));
        Ok(Self::from_vertices(vertices, Some(default_data)))
    }

    /// Interpolate data between two faces
    pub fn interpolate(
        &self,
        face_a: &Face,
        face_b: &Face,
        bary_a: &Vector3,
        bary_b: &Vector3,
    ) -> f64 {
        let data_ab = Matrix3::<f64>::from_fn(|i, j| {
            *self
                .get_data(face_a[i])
                .get()
                .unwrap()
                .get_data(face_b[j])
                .get()
                .unwrap()
        });
        (bary_a.transpose() * data_ab * bary_b).to_scalar()
    }
}

impl Table6D {
    pub fn from_resolution(r_min: f64, r_max: f64, dr: f64, angle_resolution: f64) -> Result<Self> {
        // Generate icosphere with at least n vertices
        let n_points = (4.0 * PI / angle_resolution.powi(2)).round() as usize;
        let icosphere = make_icosphere(n_points)?;
        let weights = make_weights(&icosphere);

        // Scale vertices by their relative weights. This is just a simple way to store the
        // weights in the vertices, so that we can use them later
        // NOTE: Normalize vertices before use in geometric operations!
        let mut vertices = make_vertices(&icosphere);
        vertices
            .iter_mut()
            .zip(weights)
            .for_each(|(v, w)| v.pos *= w); // scale vertices by weights

        // Vertex positions and neighbors are shared across all tables w. thread-safe smart pointer.
        // Oncelocked data on the innermost table1 is left uninitialized and should be set later
        let vertices = Arc::new(vertices);
        let table_b = IcoTable2D::<f64>::from_vertices(vertices.clone(), None); // B: 𝜃 and 𝜑

        // We have a new angular resolution, depending on number of subdivisions
        let angle_resolution = table_b.angle_resolution();
        log::info!("Actual angle resolution = {:.2} radians", angle_resolution);

        let table_a = IcoTable4D::from_vertices(vertices, Some(table_b)); // A: 𝜃 and 𝜑
        let table_omega = PaddedTable::<IcoTable4D>::new(0.0, 2.0 * PI, angle_resolution, table_a); // 𝜔
        Ok(Self::new(r_min, r_max, dr, table_omega)) // R
    }
    /// Get remaining 4D space (icotables) at (r, omega)
    pub fn get_icospheres(&self, r: f64, omega: f64) -> Result<&IcoTable4D> {
        self.get(r)?.get(omega)
    }

    /// Write 5D angular space and data to a stream, i.e for a single radial distance.
    /// The column format is `r 𝜔 𝜃1 𝜑1 𝜃2 𝜑2 data`.
    pub fn stream_angular_space(&self, r: f64, stream: &mut impl Write) -> Result<()> {
        writeln!(stream, "# r 𝜔 𝜃1 𝜑1 𝜃2 𝜑2 data")?;
        for (omega, angles) in self.get(r)?.iter() {
            for (vertex1, vertex2, data) in angles.flat_iter() {
                if let Some(data) = data.get() {
                    let (s1, s2) = (
                        SphericalCoord::from_cartesian(vertex1.normalize()),
                        SphericalCoord::from_cartesian(vertex2.normalize()),
                    );
                    writeln!(
                        stream,
                        "{:.2} {:.3} {:.3} {:.3} {:.3} {:.3} {:.4e}",
                        r,
                        omega,
                        s1.theta(),
                        s1.phi(),
                        s2.theta(),
                        s2.phi(),
                        data
                    )?;
                }
            }
        }
        Ok(())
    }
}

/// Draw icosahedron to a Visual Molecular Dynamics (VMD) TCL script
/// Visialize with: `vmd -e script.vmd`
pub(crate) fn vmd_draw(
    path: &Path,
    icosphere: &IcoSphere,
    color: &str,
    scale: Option<f32>,
) -> anyhow::Result<()> {
    let num_faces = icosphere.get_all_indices().len() / 3;
    let path = path.with_extension(format!("faces{}.vmd", num_faces));
    let mut stream = std::fs::File::create(path)?;
    icosphere.get_all_indices().chunks(3).try_for_each(|c| {
        let scale = scale.unwrap_or(1.0);
        let a = icosphere.raw_points()[c[0] as usize] * scale;
        let b = icosphere.raw_points()[c[1] as usize] * scale;
        let c = icosphere.raw_points()[c[2] as usize] * scale;
        writeln!(stream, "draw color {}", color)?;
        writeln!(
            stream,
            "draw triangle {{{:.3} {:.3} {:.3}}} {{{:.3} {:.3} {:.3}}} {{{:.3} {:.3} {:.3}}}",
            a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z
        )
    })?;
    Ok(())
}

/// Get list of all faces from an icosphere
fn _get_faces_from_icosphere(icosphere: IcoSphere) -> Vec<Face> {
    icosphere
        .get_all_indices()
        .chunks(3)
        .map(|c| {
            let v = vec![c[0] as usize, c[1] as usize, c[2] as usize];
            v.try_into().unwrap()
        })
        .collect_vec()
}

impl IcoTable2D<f64> {
    /// Get data for a point on the surface using barycentric interpolation
    /// https://en.wikipedia.org/wiki/Barycentric_coordinate_system#Interpolation_on_a_triangular_unstructured_grid
    pub fn interpolate(&self, point: &Vector3) -> f64 {
        let face = self.nearest_face(point);
        let bary = self.barycentric(point, &face);
        bary[0] * self.data[face[0]].get().unwrap()
            + bary[1] * self.data[face[1]].get().unwrap()
            + bary[2] * self.data[face[2]].get().unwrap()
    }
    /// Generate table based on a minimum number of vertices on the subdivided icosaedron
    ///
    /// Vertex data is left empty and can/should be set later
    pub fn from_min_points(min_points: usize) -> Result<Self> {
        let icosphere = make_icosphere(min_points)?;
        Ok(Self::from_icosphere_without_data(&icosphere))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::make_icosphere;
    use approx::assert_relative_eq;

    #[test]
    fn test_icosphere_table() {
        let icosphere = make_icosphere(3).unwrap();
        let icotable = IcoTable2D::<f64>::from_icosphere(&icosphere, 0.0);
        assert_eq!(icotable.data.len(), 12);

        let point = icotable.get_normalized_pos(0);

        assert_relative_eq!(point.x, 0.0);
        assert_relative_eq!(point.y, 1.0);
        assert_relative_eq!(point.z, 0.0);

        // find nearest vertex and face to vertex 0
        let face = icotable.nearest_face(&point);
        let bary = icotable.barycentric(&point, &face);
        assert_eq!(face, [0, 2, 5]);
        assert_relative_eq!(bary[0], 1.0);
        assert_relative_eq!(bary[1], 0.0);
        assert_relative_eq!(bary[2], 0.0);

        // Nearest face to slightly displaced vertex 0
        let point = (icotable.get_normalized_pos(0) + Vector3::new(1e-3, 0.0, 0.0)).normalize();
        let face = icotable.nearest_face(&point);
        let bary = icotable.barycentric(&point, &face);
        assert_eq!(face, [0, 1, 5]);
        assert_relative_eq!(bary[0], 0.9991907334103153, epsilon = 1e-6);
        assert_relative_eq!(bary[1], 0.000809266589684687, epsilon = 1e-6);
        assert_relative_eq!(bary[2], 0.0);

        // find nearest vertex and face to vertex 2
        let point = icotable.get_normalized_pos(2);
        let face = icotable.nearest_face(&point);
        let bary = icotable.barycentric(&point, &face);
        assert_eq!(face, [0, 1, 2]);
        assert_relative_eq!(bary[0], 0.0);
        assert_relative_eq!(bary[1], 0.0);
        assert_relative_eq!(bary[2], 1.0);

        // Midpoint on edge between vertices 0 and 2
        let point = point + (icotable.get_normalized_pos(0) - point) * 0.5;
        let bary = icotable.barycentric(&point, &face);
        assert_relative_eq!(bary[0], 0.5);
        assert_relative_eq!(bary[1], 0.0);
        assert_relative_eq!(bary[2], 0.5);
    }

    #[test]
    fn test_icosphere_interpolate() {
        let icosphere = make_icosphere(3).unwrap();
        let icotable = IcoTable2D::<f64>::from_icosphere_without_data(&icosphere);
        icotable.set_vertex_data(|i, _| i as f64 + 1.0).unwrap();

        let point = Vector3::new(0.5, 0.5, 0.5).normalize();
        let data = icotable.interpolate(&point);
        assert_relative_eq!(data, 2.59977558757542, epsilon = 1e-6);

        let point = Vector3::new(0.5, 1.0, -2.0).normalize();
        let data = icotable.interpolate(&point);
        assert_relative_eq!(data, 6.062167441678067, epsilon = 1e-6);
    }

    #[test]
    fn test_face_face_interpolation() {
        let n_points = 12;
        let icosphere = make_icosphere(n_points).unwrap();
        let icotable = IcoTable2D::<f64>::from_icosphere_without_data(&icosphere);
        icotable.set_vertex_data(|i, _| i as f64).unwrap();
        let icotable_of_spheres = IcoTable4D::from_min_points(n_points, icotable).unwrap();

        let face_a = [0, 1, 2];
        let face_b = [0, 1, 2];

        // corner 1
        let bary_a = Vector3::new(1.0, 0.0, 0.0);
        let bary_b = Vector3::new(1.0, 0.0, 0.0);
        let data = icotable_of_spheres.interpolate(&face_a, &face_b, &bary_a, &bary_b);
        assert_relative_eq!(data, 0.0);

        // corner 2
        let bary_a = Vector3::new(0.0, 1.0, 0.0);
        let bary_b = Vector3::new(0.0, 1.0, 0.0);
        let data = icotable_of_spheres.interpolate(&face_a, &face_b, &bary_a, &bary_b);
        assert_relative_eq!(data, 1.0);

        // corner 3
        let bary_a = Vector3::new(0.0, 0.0, 1.0);
        let bary_b = Vector3::new(0.0, 0.0, 1.0);
        let data = icotable_of_spheres.interpolate(&face_a, &face_b, &bary_a, &bary_b);
        assert_relative_eq!(data, 2.0);

        // center
        let bary_a = Vector3::new(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
        let bary_b = Vector3::new(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
        let data = icotable_of_spheres.interpolate(&face_a, &face_b, &bary_a, &bary_b);
        assert_relative_eq!(data, (0.0 + 1.0 + 2.0) / 3_f64);
    }

    #[test]
    fn test_table_of_spheres() {
        let icotable = IcoTable2D::<f64>::from_min_points(42).unwrap();
        let icotable_of_spheres = IcoTable4D::from_min_points(42, icotable).unwrap();
        assert_eq!(icotable_of_spheres.data.len(), 42);
        assert_eq!(icotable_of_spheres.data[0].get().unwrap().data.len(), 42);
    }
}
