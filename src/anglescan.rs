#[cfg(test)]
extern crate approx;
use crate::{
    energy::{self, PairMatrix},
    report::report_pmf,
    structure::Structure,
    Sample,
};
use anyhow::{Context, Result};
#[cfg(test)]
use approx::assert_relative_eq;
use flate2::{write::GzEncoder, Compression};
use hexasphere::{
    shapes::{IcoSphere, IcoSphereBase},
    Subdivided,
};
use indicatif::ParallelProgressIterator;
use iter_num_tools::arange;
use itertools::Itertools;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::{
    f64::consts::PI,
    fmt::Display,
    io::Write,
    path::{Path, PathBuf},
};

pub type Vector3 = nalgebra::Vector3<f64>;
pub type UnitQuaternion = nalgebra::UnitQuaternion<f64>;

/// Struct to exhaust all possible 6D relative orientations between two rigid bodies.
///
/// Fibonacci sphere points are used to generate rotations around the z-axis.
#[derive(Debug)]
pub struct TwobodyAngles {
    /// Rotations of the first body
    pub q1: Vec<UnitQuaternion>,
    /// Rotations of the second body
    pub q2: Vec<UnitQuaternion>,
    /// Rotations around connecting z-azis (0..2π)
    pub dihedrals: Vec<UnitQuaternion>,
}

impl Display for TwobodyAngles {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let n1 = self.q1.len();
        let n2 = self.q2.len();
        let n3 = self.dihedrals.len();
        write!(f, "{} x {} x {} = {} poses 💃🕺", n1, n2, n3, n1 * n2 * n3)
    }
}

impl TwobodyAngles {
    /// Generates a set of quaternions for a rigid body scan
    ///
    /// # Arguments
    /// angle_resolution: f64 - the resolution of the scan in radians
    pub fn from_resolution(angle_resolution: f64) -> Result<Self> {
        assert!(angle_resolution > 0.0,);

        let n_points = (4.0 * PI / angle_resolution.powi(2)).round() as usize;
        let mut points = make_icosphere_vertices(n_points)?;
        let angle_resolution = (4.0 * PI / points.len() as f64).sqrt();
        log::info!(
            "Requested {} points on a sphere; got {} -> new resolution = {:.2}",
            n_points,
            points.len(),
            angle_resolution
        );

        // Ensure that icosphere points are not *exactly* on the z-axis to
        // enable trouble-free alignment below; see `rotation_between()` docs
        let v = nalgebra::UnitVector3::new_normalize(Vector3::new(0.0005, 0.0005, 1.0));
        let q_bias = UnitQuaternion::from_axis_angle(&v, 0.0001);
        points.iter_mut().for_each(|p| *p = q_bias * (*p));

        // Rotation operations via unit quaternions
        let to_zaxis = |p| UnitQuaternion::rotation_between(p, &Vector3::z_axis()).unwrap();
        let to_neg_zaxis = |p| UnitQuaternion::rotation_between(p, &-Vector3::z_axis()).unwrap();
        let around_z = |angle| UnitQuaternion::from_axis_angle(&Vector3::z_axis(), angle);

        let q1 = points.iter().map(to_neg_zaxis).collect();
        let q2 = points.iter().map(to_zaxis).collect();

        let dihedrals = arange(0.0..2.0 * PI, angle_resolution)
            .map(around_z)
            .collect();

        Ok(Self { q1, q2, dihedrals })
    }

    /// Generates a set of quaternions for a rigid body scan.
    /// Each returned unit quaternion pair can be used to rotate two rigid bodies
    /// to exhaustively scan all possible relative orientations.
    pub fn iter(&self) -> impl Iterator<Item = (UnitQuaternion, UnitQuaternion)> + '_ {
        let dihedral_x_q2 = self
            .dihedrals
            .iter()
            .cartesian_product(self.q2.iter())
            .map(|(&d, &q2)| d * q2);
        self.q1.iter().cloned().cartesian_product(dihedral_x_q2)
    }
    /// Total length of the iterator
    pub fn len(&self) -> usize {
        self.q1.len() * self.q2.len() * self.dihedrals.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Opens a gz compressed file for writing
    fn open_compressed_file(outfile: impl AsRef<Path>) -> Result<GzEncoder<std::fs::File>> {
        Ok(GzEncoder::new(
            std::fs::File::create(outfile)?,
            Compression::default(),
        ))
    }

    /// Scan over all angles and write to a file
    ///
    /// This does the following:
    /// - Rotates the COM vector, r by q1
    /// - Rotates the second body by q2
    /// - Translates the second body by r
    /// - Calculates the energy between the two structures
    /// - Writes the distance and energy to a buffered file
    /// - Sum energies and partition function and return as a `Sample`
    ///
    /// # Arguments:
    /// - `ref_a: &Structure` - reference structure A
    /// - `ref_b: &Structure` - reference structure B
    /// - `pair_matrix: &PairMatrix` - pair matrix of twobody energies
    /// - `r: &Vector3` - distance vector between the two structures
    /// - `temperature: f64` - temperature in K
    pub fn sample_all_angles(
        &self,
        ref_a: &Structure,
        ref_b: &Structure,
        pair_matrix: &PairMatrix,
        r: &Vector3,
        temperature: f64,
    ) -> Result<Sample> {
        let mut file = Self::open_compressed_file(format!("R_{:.1}.dat.gz", r.norm()))?;
        let sample = self // Scan over angles
            .iter()
            .map(|(q1, q2)| {
                let (a, b) = Self::transform_structures(ref_a, ref_b, &q1, &q2, r);
                let energy = pair_matrix.sum_energy(&a, &b);
                let com = b.mass_center();
                writeln!(
                    file,
                    "{:.3} {:.3} {:.3} {:.3} {:?}",
                    energy,
                    com.x,
                    com.y,
                    com.z,
                    q2.axis_angle().unwrap()
                )
                .unwrap();
                Sample::new(energy, temperature)
            })
            .sum::<Sample>();
        Ok(sample)
    }

    /// Transform the two reference structures by the given quaternions and distance vector
    ///
    /// This only transforms the second reference structure by translating and rotating it,
    /// while the first reference structure is left unchanged.
    fn transform_structures(
        ref_a: &Structure,
        ref_b: &Structure,
        q1: &UnitQuaternion,
        q2: &UnitQuaternion,
        r: &Vector3, // mass center separation = (0,0,r)
    ) -> (Structure, Structure) {
        let a = ref_a.clone();
        let mut b = ref_b.clone();
        b.pos = ref_b
            .pos
            .iter()
            .map(|pos| q2.transform_vector(pos) + q1.transform_vector(r))
            .collect();
        (a, b)
    }
}

/// Generates n points uniformly distributed on a unit sphere
///
/// Related information:
/// - <https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere>
/// - <https://en.wikipedia.org/wiki/Geodesic_polyhedron>
/// - c++: <https://github.com/caosdoar/spheres>
pub fn make_fibonacci_sphere(n_points: usize) -> Vec<Vector3> {
    assert!(n_points > 1, "n_points must be greater than 1");
    let phi = PI * (3.0 - (5.0f64).sqrt()); // golden angle in radians
    let make_ith_point = |i: usize| -> Vector3 {
        let mut p = Vector3::zeros();
        p.y = 1.0 - 2.0 * (i as f64 / (n_points - 1) as f64); // y goes from 1 to -1
        let radius = (1.0 - p.y * p.y).sqrt(); // radius at y
        let theta = phi * i as f64; // golden angle increment
        p.x = theta.cos() * radius;
        p.z = theta.sin() * radius;
        p.normalize()
    };
    (0..n_points).map(make_ith_point).collect()
}

/// Make icosphere with at least `min_points` surface points (vertices).
///
/// This is done by iteratively subdividing the faces of an icosahedron
/// until at least `min_points` vertices are achieved.
/// The number of vertices on the icosphere is _N_ = 10 × (_n_divisions_ + 1)² + 2
/// whereby 0, 1, 2, ... subdivisions give 12, 42, 92, ... vertices, respectively.
///
///
/// ## Further reading
///
/// - <https://en.wikipedia.org/wiki/Loop_subdivision_surface>
/// - <https://danielsieger.com/blog/2021/03/27/generating-spheres.html>
/// - <https://danielsieger.com/blog/2021/01/03/generating-platonic-solids.html>
///
/// ![Image](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f7/Loop_Subdivision_Icosahedron.svg/300px-Loop_Subdivision_Icosahedron.svg.png)
///
pub fn make_icosphere(min_points: usize) -> Result<Subdivided<(), IcoSphereBase>> {
    let points_per_division = |n_div: usize| 10 * (n_div + 1) * (n_div + 1) + 2;
    let n_points = (0..200).map(points_per_division);

    // Number of divisions to achieve at least `min_points` vertices
    let n_divisions = n_points
        .enumerate()
        .find(|(_, n)| *n >= min_points)
        .map(|(n_div, _)| n_div)
        .context("too many vertices")?;

    Ok(IcoSphere::new(n_divisions, |_| ()))
}

/// Make icosphere vertices as 3D vectors
///
/// ## Examples
/// ~~~
/// use duello::anglescan;
/// let vertices = anglescan::make_icosphere_vertices(20).unwrap();
/// assert_eq!(vertices.len(), 42);
/// ~~~
pub fn make_icosphere_vertices(min_points: usize) -> Result<Vec<Vector3>> {
    let points = make_icosphere(min_points)?
        .raw_points()
        .iter()
        .map(|p| Vector3::new(p.x as f64, p.y as f64, p.z as f64))
        .collect();
    Ok(points)
}

pub fn do_anglescan(
    distances: Vec<f64>,
    angle_resolution: f64,
    ref_a: Structure,
    ref_b: Structure,
    pair_matrix: energy::PairMatrix,
    temperature: &f64,
    pmf_file: &PathBuf,
) -> std::result::Result<(), anyhow::Error> {
    let scan = TwobodyAngles::from_resolution(angle_resolution).unwrap();
    info!("{} per distance", scan);
    let com_scan = distances
        .par_iter()
        .progress_count(distances.len() as u64)
        .map(|r| {
            let r_vec = Vector3::new(0.0, 0.0, *r);
            let sample = scan
                .sample_all_angles(&ref_a, &ref_b, &pair_matrix, &r_vec, *temperature)
                .unwrap();
            (r_vec, sample)
        })
        .collect::<Vec<_>>();

    let masses = (ref_a.total_mass(), ref_b.total_mass());

    report_pmf(com_scan.as_slice(), pmf_file, Some(masses));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_icosphere() {
        let points = make_icosphere_vertices(1).unwrap();
        assert_eq!(points.len(), 12);
        let points = make_icosphere_vertices(10).unwrap();
        assert_eq!(points.len(), 12);
        let points = make_icosphere_vertices(13).unwrap();
        assert_eq!(points.len(), 42);
        let points = make_icosphere_vertices(42).unwrap();
        assert_eq!(points.len(), 42);
        let points = make_icosphere_vertices(43).unwrap();
        assert_eq!(points.len(), 92);
        let _ = make_icosphere_vertices(400003).is_err();

        let samples = 1000;
        let points = make_icosphere_vertices(samples).unwrap();
        let mut center: Vector3 = Vector3::zeros();
        assert_eq!(points.len(), 1002);
        for point in points {
            assert_relative_eq!((point.norm() - 1.0).abs(), 0.0, epsilon = 1e-6);
            center += point;
        }
        assert_relative_eq!(center.norm(), 0.0, epsilon = 1e-1);
    }

    #[test]
    fn test_twobody_angles() {
        use std::f64::consts::FRAC_1_SQRT_2;
        let twobody_angles = TwobodyAngles::from_resolution(1.1).unwrap();
        let n = twobody_angles.q1.len() * twobody_angles.q2.len() * twobody_angles.dihedrals.len();
        assert_eq!(n, 1008);
        assert_eq!(twobody_angles.len(), n);
        assert_eq!(twobody_angles.iter().count(), n);

        let pairs = twobody_angles.iter().collect::<Vec<_>>();
        assert_relative_eq!(pairs[0].0.coords.x, -FRAC_1_SQRT_2, epsilon = 1e-5);
        assert_relative_eq!(pairs[0].0.coords.y, 0.0, epsilon = 1e-4);
        assert_relative_eq!(pairs[0].0.coords.z, 0.0);
        assert_relative_eq!(pairs[0].0.coords.w, FRAC_1_SQRT_2, epsilon = 1e-5);
        assert_relative_eq!(pairs[0].1.coords.x, FRAC_1_SQRT_2, epsilon = 1e-5);
        assert_relative_eq!(pairs[0].1.coords.y, 0.0, epsilon = 1e-4);
        assert_relative_eq!(pairs[0].1.coords.z, 0.0, epsilon = 1e-4);
        assert_relative_eq!(pairs[0].1.coords.w, FRAC_1_SQRT_2, epsilon = 1e-5);
        assert_relative_eq!(pairs[n - 1].0.coords.x, FRAC_1_SQRT_2, epsilon = 1e-5);
        assert_relative_eq!(pairs[n - 1].0.coords.y, 0.0, epsilon = 1e-4);
        assert_relative_eq!(pairs[n - 1].0.coords.z, 0.0, epsilon = 1e-4);
        assert_relative_eq!(pairs[n - 1].0.coords.w, FRAC_1_SQRT_2, epsilon = 1e-5);
        assert_relative_eq!(pairs[n - 1].1.coords.x, 0.705299, epsilon = 1e-5);
        assert_relative_eq!(pairs[n - 1].1.coords.y, -0.050523, epsilon = 1e-5);
        assert_relative_eq!(pairs[n - 1].1.coords.z, 0.050594, epsilon = 1e-5);
        assert_relative_eq!(pairs[n - 1].1.coords.w, -0.705294, epsilon = 1e-5);
        println!("{}", twobody_angles);
    }

    #[test]
    fn test_fibonacci_sphere() {
        let samples = 1000;
        let points_on_sphere = make_fibonacci_sphere(samples);
        let mut center: Vector3 = Vector3::zeros();
        assert_eq!(points_on_sphere.len(), samples);
        for point in points_on_sphere {
            assert_relative_eq!((point.norm() - 1.0).abs(), 0.0, epsilon = 1e-10);
            center += point;
        }
        assert_relative_eq!(center.norm(), 0.0, epsilon = 1e-1);
    }
}
