use super::{anglescan::*, table::PaddedTable};
use anyhow::Result;
use core::f64::consts::PI;
use get_size::GetSize;
use hexasphere::{shapes::IcoSphereBase, AdjacencyBuilder, Subdivided};
use itertools::Itertools;
use nalgebra::Matrix3;
use std::io::Write;
use std::sync::OnceLock;

/// A icotable where each vertex holds an icotable of floats
pub type IcoTableOfSpheres = IcoTable<IcoTable<f64>>;

/// A 6D table for relative twobody orientations, R → 𝜔 → (𝜃𝜑) → (𝜃𝜑)
///
/// The first two dimensions are radial distances and dihedral angles.
/// The last two dimensions are polar and azimuthal angles represented via icospheres.
/// The final `f64` data is stored at vertices of the deepest icospheres.
pub type Table6D = PaddedTable<PaddedTable<IcoTableOfSpheres>>;

impl Table6D {
    pub fn from_resolution(r_min: f64, r_max: f64, dr: f64, angle_resolution: f64) -> Result<Self> {
        let n_points = (4.0 * PI / angle_resolution.powi(2)).round() as usize;
        let table1 = IcoTable::<f64>::from_min_points(n_points)?; // B: 𝜃 and 𝜑
                                                                  // update angular resolution according to icosphere
        let angle_resolution = table1.angle_resolution();
        let n_points = table1.vertices.len();
        log::info!("Actual angle resolution = {:.2} radians", angle_resolution);
        let table2 = IcoTableOfSpheres::from_min_points(n_points, table1)?; // A: 𝜃 and 𝜑
        let table3 = PaddedTable::<IcoTableOfSpheres>::new(0.0, 2.0 * PI, angle_resolution, table2); // 𝜔
        Ok(Self::new(r_min, r_max, dr, table3)) // R
    }
}

/// Represents indices of a face
pub type Face = [u16; 3];

/// Struct representing a vertex in the icosphere
///
/// Interior mutability of vertex associated data is enabled using `std::sync::OnceLock`.
#[derive(Clone, GetSize)]
pub struct Vertex<T: Clone + GetSize> {
    /// 3D coordinates of the vertex on a unit sphere
    #[get_size(size = 24)]
    pub pos: Vector3,
    /// Data associated with the vertex
    #[get_size(size_fn = oncelock_size_helper)]
    pub data: OnceLock<T>,
    /// Indices of neighboring vertices
    pub neighbors: Vec<u16>,
}

fn oncelock_size_helper<T: GetSize>(value: &OnceLock<T>) -> usize {
    std::mem::size_of::<OnceLock<T>>() + value.get().map(|v| v.get_heap_size()).unwrap_or(0)
}

impl<T: Clone + GetSize> Vertex<T> {
    /// Construct a new vertex where data is *locked* to fixed value
    pub fn with_fixed_data(pos: Vector3, data: T, neighbors: Vec<u16>) -> Self {
        let vertex = Self::without_data(pos, neighbors);
        let _ = vertex.data.set(data);
        vertex
    }

    /// Construct a new vertex; write-once data is left empty and can/should be set later
    pub fn without_data(pos: Vector3, neighbors: Vec<u16>) -> Self {
        assert!(matches!(neighbors.len(), 5 | 6));
        Self {
            pos,
            data: OnceLock::<T>::new(),
            neighbors,
        }
    }
}

/// Icosphere table
///
/// This is used to store data on the vertices of an icosphere.
/// It includes barycentric interpolation and nearest face search.
///
/// https://en.wikipedia.org/wiki/Geodesic_polyhedron
/// 12 vertices will always have 5 neighbors; the rest will have 6.
#[derive(Clone, GetSize)]
pub struct IcoTable<T: Clone + GetSize> {
    /// Vertex information (position, data, neighbors)
    pub vertices: Vec<Vertex<T>>,
}

impl<T: Clone + GetSize> IcoTable<T> {
    /// Generate table based on an existing subdivided icosaedron
    pub fn from_icosphere_without_data(icosphere: Subdivided<(), IcoSphereBase>) -> Self {
        let indices = icosphere.get_all_indices();
        let mut builder = AdjacencyBuilder::new(icosphere.raw_points().len());
        builder.add_indices(indices.as_slice());
        let neighbors = builder.finish().iter().map(|i| i.to_vec()).collect_vec();
        let vertex_positions: Vec<Vector3> = icosphere
            .raw_points()
            .iter()
            .map(|p| Vector3::new(p.x as f64, p.y as f64, p.z as f64))
            .collect();

        let vertices = (0..vertex_positions.len())
            .map(|i| {
                Vertex::without_data(
                    vertex_positions[i],
                    neighbors[i].iter().map(|i| *i as u16).collect_vec(),
                )
            })
            .collect();

        Self { vertices }
    }

    /// Generate table based on an existing subdivided icosaedron
    pub fn from_icosphere(icosphere: Subdivided<(), IcoSphereBase>, default_data: T) -> Self {
        let table = Self::from_icosphere_without_data(icosphere);
        table.set_vertex_data(|_, _| default_data.clone());
        table
    }

    pub fn angle_resolution(&self) -> f64 {
        let n_points = self.vertices.len();
        (4.0 * std::f64::consts::PI / n_points as f64).sqrt()
    }

    /// Number of vertices in the table
    pub fn len(&self) -> usize {
        self.vertices.len()
    }

    /// Check if the table is empty, i.e. has no vertices
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }

    /// Set data associated with each vertex using a generator function
    /// The function takes the index of the vertex and its position
    /// Due to the `OnceLock` wrap, this can be done only once!
    pub fn set_vertex_data(&self, f: impl Fn(usize, &Vector3) -> T) {
        self.vertices.iter().enumerate().for_each(|(i, v)| {
            assert!(v.data.get().is_none());
            let value = f(i, &v.pos);
            let result = v.data.set(value); // why can we not use unwrap here?!
            assert!(result.is_ok());
        });
    }

    /// Discard data associated with each vertex
    ///
    /// After this call, `set_vertex_data` can be called again.
    pub fn clear_vertex_data(&mut self) {
        for vertex in self.vertices.iter_mut() {
            vertex.data = OnceLock::new();
        }
    }

    /// Get data associated with each vertex
    pub fn vertex_data(&self) -> impl Iterator<Item = &T> {
        self.vertices.iter().map(|v| v.data.get().unwrap())
    }

    /// Transform vertex positions using a function
    pub fn transform_vertex_positions(&mut self, f: impl Fn(&Vector3) -> Vector3) {
        self.vertices.iter_mut().for_each(|v| v.pos = f(&v.pos));
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
    pub fn face_positions(&self, face: &Face) -> (&Vector3, &Vector3, &Vector3) {
        let a = &self.vertices[face[0] as usize].pos;
        let b = &self.vertices[face[1] as usize].pos;
        let c = &self.vertices[face[2] as usize].pos;
        (a, b, c)
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
        self.vertices
            .iter()
            .map(|v| (v.pos - point).norm_squared())
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
        let face: Face = self.vertices[nearest]
            .neighbors // neighbors to nearest
            .iter()
            .cloned()
            .map(|i| (i, (self.vertices[i as usize].pos - point).norm_squared()))
            .sorted_by(|a, b| a.1.partial_cmp(&b.1).unwrap()) // sort ascending
            .map(|(i, _)| i) // keep only indices
            .take(2) // take two next nearest distances
            .collect_tuple()
            .map(|(a, b)| [a, b, nearest as u16]) // append nearest
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

    // /// Save a VMD script to illustrate the icosphere
    // pub fn save_vmd(&self, path: impl AsRef<Path>, scale: Option<f64>) -> Result<()> {
    //     let mut file = std::fs::File::create(path)?;
    //     let s = scale.unwrap_or(1.0);
    //     writeln!(file, "draw delete all")?;
    //     for face in &self.faces {
    //         let a = &self.vertices[face[0]].pos.scale(s);
    //         let b = &self.vertices[face[1]].pos.scale(s);
    //         let c = &self.vertices[face[2]].pos.scale(s);
    //         let color = "red";
    //         vmd_draw_triangle(&mut file, a, b, c, color)?;
    //     }
    //     Ok(())
    // }
}

impl std::fmt::Display for IcoTable<f64> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "# x y z θ φ data")?;
        for vertex in self.vertices.iter() {
            let (_r, theta, phi) = crate::to_spherical(&vertex.pos);
            writeln!(
                f,
                "{} {} {} {} {} {}",
                vertex.pos.x,
                vertex.pos.y,
                vertex.pos.z,
                theta,
                phi,
                vertex.data.get().unwrap()
            )?;
        }
        Ok(())
    }
}

/// Get list of all faces from an icosphere
fn _get_faces_from_icosphere<T>(icosphere: Subdivided<T, IcoSphereBase>) -> Vec<Face> {
    icosphere
        .get_all_indices()
        .chunks(3)
        .map(|c| {
            let v = vec![c[0] as u16, c[1] as u16, c[2] as u16];
            v.try_into().unwrap()
        })
        .collect_vec()
}

impl IcoTable<f64> {
    /// Get data for a point on the surface using barycentric interpolation
    /// https://en.wikipedia.org/wiki/Barycentric_coordinate_system#Interpolation_on_a_triangular_unstructured_grid
    pub fn interpolate(&self, point: &Vector3) -> f64 {
        let face = self.nearest_face(point);
        let bary = self.barycentric(point, &face);
        bary[0] * self.vertices[face[0] as usize].data.get().unwrap()
            + bary[1] * self.vertices[face[1] as usize].data.get().unwrap()
            + bary[2] * self.vertices[face[2] as usize].data.get().unwrap()
    }
    /// Generate table based on a minimum number of vertices on the subdivided icosaedron
    ///
    /// Vertex data is left empty and can/should be set later
    pub fn from_min_points(min_points: usize) -> Result<Self> {
        let icosphere = make_icosphere(min_points)?;
        Ok(Self::from_icosphere_without_data(icosphere))
    }
}
impl IcoTableOfSpheres {
    /// Generate table based on a minimum number of vertices on the subdivided icosaedron
    pub fn from_min_points(min_points: usize, default_data: IcoTable<f64>) -> Result<Self> {
        let icosphere = make_icosphere(min_points)?;
        Ok(Self::from_icosphere(icosphere, default_data))
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
            *self.vertices[face_a[i] as usize]
                .data
                .get()
                .unwrap()
                .vertices[face_b[j] as usize]
                .data
                .get()
                .unwrap()
        });
        (bary_a.transpose() * data_ab * bary_b).to_scalar()
    }
}

/// Draw a triangle in VMD format
fn _vmd_draw_triangle(
    stream: &mut impl Write,
    a: &Vector3,
    b: &Vector3,
    c: &Vector3,
    color: &str,
) -> Result<()> {
    writeln!(stream, "draw color {}", color)?;
    writeln!(
        stream,
        "draw triangle {{{:.3} {:.3} {:.3}}} {{{:.3} {:.3} {:.3}}} {{{:.3} {:.3} {:.3}}}",
        a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anglescan::make_icosphere;
    use approx::assert_relative_eq;

    #[test]
    fn test_icosphere_table() {
        let icosphere = make_icosphere(3).unwrap();
        let icotable = IcoTable::<f64>::from_icosphere(icosphere, 0.0);
        assert_eq!(icotable.vertices.len(), 12);

        let point = icotable.vertices[0].pos;

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
        let point = (icotable.vertices[0].pos + Vector3::new(1e-3, 0.0, 0.0)).normalize();
        let face = icotable.nearest_face(&point);
        let bary = icotable.barycentric(&point, &face);
        assert_eq!(face, [0, 1, 5]);
        assert_relative_eq!(bary[0], 0.9991907334103153);
        assert_relative_eq!(bary[1], 0.000809266589684687);
        assert_relative_eq!(bary[2], 0.0);

        // find nearest vertex and face to vertex 2
        let point = icotable.vertices[2].pos;
        let face = icotable.nearest_face(&point);
        let bary = icotable.barycentric(&point, &face);
        assert_eq!(face, [0, 1, 2]);
        assert_relative_eq!(bary[0], 0.0);
        assert_relative_eq!(bary[1], 0.0);
        assert_relative_eq!(bary[2], 1.0);

        // Midpoint on edge between vertices 0 and 2
        let point = point + (icotable.vertices[0].pos - point) * 0.5;
        let bary = icotable.barycentric(&point, &face);
        assert_relative_eq!(bary[0], 0.5);
        assert_relative_eq!(bary[1], 0.0);
        assert_relative_eq!(bary[2], 0.5);
    }

    #[test]
    fn test_face_face_interpolation() {
        let n_points = 12;
        let icosphere = make_icosphere(n_points).unwrap();
        let icotable = IcoTable::<f64>::from_icosphere_without_data(icosphere);
        icotable.set_vertex_data(|i, _| i as f64);
        let icotable_of_spheres = IcoTableOfSpheres::from_min_points(n_points, icotable).unwrap();

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
        let icotable = IcoTable::<f64>::from_min_points(42).unwrap();
        let icotable_of_spheres = IcoTableOfSpheres::from_min_points(42, icotable).unwrap();
        assert_eq!(icotable_of_spheres.vertices.len(), 42);
        assert_eq!(
            icotable_of_spheres.vertices[0]
                .data
                .get()
                .unwrap()
                .vertices
                .len(),
            42
        );
    }
}
