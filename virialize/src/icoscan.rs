use crate::{
    energy::{self},
    icotable::Table6D,
    report::report_pmf,
    structure::Structure,
    Sample,
};
use get_size::GetSize;
use indicatif::ParallelProgressIterator;
use itertools::Itertools;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::{f64::consts::PI, path::PathBuf};

pub type Vector3 = nalgebra::Vector3<f64>;
pub type UnitQuaternion = nalgebra::UnitQuaternion<f64>;

#[allow(clippy::too_many_arguments)]
pub fn do_icoscan(
    rmin: f64,
    rmax: f64,
    dr: f64,
    angle_resolution: f64,
    ref_a: Structure,
    ref_b: Structure,
    pair_matrix: energy::PairMatrix,
    temperature: &f64,
    pmf_file: &PathBuf,
) -> std::result::Result<(), anyhow::Error> {
    let distances = iter_num_tools::arange(rmin..rmax, dr).collect_vec();
    let table = Table6D::from_resolution(rmin, rmax, dr, angle_resolution)?;
    let n_points = table.get(rmin).unwrap().get(0.0).unwrap().len();
    let angle_resolution = (4.0 * PI / n_points as f64).sqrt();
    let dihedral_angles = iter_num_tools::arange(0.0..2.0 * PI, angle_resolution).collect_vec();

    let total = distances.len() * dihedral_angles.len() * n_points * n_points;

    info!(
        "6D table: 𝑅({}) x 𝜔({}) x 𝜃𝜑({}) x 𝜃𝜑({}) = {} poses 💃🕺 ({:.1} MB)",
        distances.len(),
        dihedral_angles.len(),
        n_points,
        n_points,
        total,
        table.get_heap_size() as f64 / 1e6
    );

    use nalgebra::UnitVector3;

    // Rotation operations via unit quaternions
    let zaxis = UnitVector3::new_normalize(Vector3::new(0.0005, 0.0005, 1.0));
    let to_neg_zaxis = |p| UnitQuaternion::rotation_between(p, &-zaxis).unwrap();
    let around_z = |angle| UnitQuaternion::from_axis_angle(&zaxis, angle);

    // Calculate energy of all two-body poses for given mass center separation and dihedral angle
    let calc_energy = |r: f64, omega: f64| {
        let r_vec = Vector3::new(0.0, 0.0, r);
        let a = table.get(r).unwrap().get(omega).unwrap();
        for vertex_a in a.vertices.iter() {
            for vertex_b in vertex_a.data.get().unwrap().vertices.iter() {
                let q1 = to_neg_zaxis(&vertex_b.pos);
                let q2 = around_z(omega);
                let q3 = UnitQuaternion::rotation_between(&zaxis, &vertex_a.pos).unwrap();
                let mut mol_b = ref_b.clone(); // initially at origin
                mol_b.transform(|pos| (q1 * q2).transform_vector(&pos));
                mol_b.transform(|pos| q3.transform_vector(&(pos + r_vec)));
                let energy = pair_matrix.sum_energy(&ref_a, &mol_b);
                vertex_b.data.set(energy).unwrap();
            }
        }
    };

    // Pair all mass center separations (r) and dihedral angles (omega)
    let r_and_omega = distances
        .iter()
        .copied()
        .cartesian_product(dihedral_angles.iter().copied())
        .collect_vec();

    // Populate 6D table with inter-particle energies (multi-threaded)
    r_and_omega
        .par_iter()
        .progress_count(r_and_omega.len() as u64)
        .for_each(|(r, omega)| {
            calc_energy(*r, *omega);
        });

    // Calculate partition function
    let mut samples: Vec<(Vector3, Sample)> = Vec::default();
    for r in &distances {
        let mut partition_func = Sample::default();
        for omega in &dihedral_angles {
            let r_and_omega = table.get(*r).unwrap().get(*omega).unwrap();
            for vertex_a in r_and_omega.vertices.iter() {
                for vertex_b in vertex_a.data.get().unwrap().vertices.iter() {
                    let energy = vertex_b.data.get().unwrap();
                    partition_func = partition_func + Sample::new(*energy, *temperature);
                }
            }
        }
        samples.push((Vector3::new(0.0, 0.0, *r), partition_func));
    }

    let masses = (ref_a.total_mass(), ref_b.total_mass());

    report_pmf(samples.as_slice(), pmf_file, Some(masses));
    Ok(())
}
