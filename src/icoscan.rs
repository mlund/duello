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

use crate::{
    backend::{EnergyBackend, PoseParams},
    icotable::Table6D,
    report::report_pmf,
    structure::Structure,
    Sample, UnitQuaternion, Vector3,
};
use faunus::aux::open_compressed;
use get_size::GetSize;
use indicatif::{ParallelProgressIterator, ProgressIterator};
use iter_num_tools::arange;
use itertools::Itertools;
use nalgebra::UnitVector3;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::{
    f64::consts::PI,
    io::{BufWriter, Write},
    path::PathBuf,
};
use xdrfile::{Frame, Trajectory, XTCTrajectory};

/// Orient two reference structures to given 6D point and return the two structures
///
/// Structure A is kept fixed at origin while structure B is rotated and translated
/// to the given 6D point (r, omega, 2 x vertex positions). The given reference structures
/// are assumed to be centered at the origin.
pub fn orient_structures(
    r: f64,
    omega: f64,
    vertex_i: Vector3,
    vertex_j: Vector3,
    ref_a: &Structure,
    ref_b: &Structure,
) -> (Structure, Structure) {
    let vertex_i = vertex_i.normalize();
    let vertex_j = vertex_j.normalize();
    let r_vec = Vector3::new(0.0, 0.0, r);
    // Z axis cannot be *exactly* parallel to r_vec; see nalgebra::rotation_between
    let zaxis = UnitVector3::new_normalize(Vector3::new(0.0005, 0.0005, 1.0));
    let to_neg_zaxis = |p| UnitQuaternion::rotation_between(p, &-zaxis).unwrap();
    let around_z = |angle| UnitQuaternion::from_axis_angle(&zaxis, angle);
    let q1 = to_neg_zaxis(&vertex_j);
    let q2 = around_z(omega);
    let q3 = UnitQuaternion::rotation_between(&zaxis, &vertex_i).unwrap();
    let mut mol_b = ref_b.clone(); // initially at origin
    mol_b.transform(|pos| (q1 * q2).transform_vector(&pos));
    mol_b.transform(|pos| q3.transform_vector(&(pos + r_vec)));
    (ref_a.clone(), mol_b)
}

#[allow(clippy::too_many_arguments)]
pub fn do_icoscan<B: EnergyBackend>(
    rmin: f64,
    rmax: f64,
    dr: f64,
    angle_resolution: f64,
    backend: &B,
    temperature: &f64,
    pmf_file: &PathBuf,
    disktable: &Option<PathBuf>,
    xtcfile: &Option<PathBuf>,
) -> std::result::Result<(), anyhow::Error> {
    let ref_a = backend.ref_a();
    let ref_b = backend.ref_b();
    let table = Table6D::from_resolution(rmin, rmax, dr, angle_resolution)?;
    let n_vertices = table.get(rmin)?.get(0.0)?.len();
    let angle_resolution = (4.0 * PI / n_vertices as f64).sqrt();
    let dihedral_angles = arange(0.0..2.0 * PI, angle_resolution).collect_vec();
    let distances = arange(rmin..rmax, dr).collect_vec();
    let n_total = distances.len() * dihedral_angles.len() * n_vertices * n_vertices;

    info!(
        "6D table: 𝑅({}) x 𝜔({}) x 𝜃𝜑({}) x 𝜃𝜑({}) = {} poses 💃🕺 ({:.1} MB)",
        distances.len(),
        dihedral_angles.len(),
        n_vertices,
        n_vertices,
        n_total,
        table.get_size() as f64 / f64::powi(1024.0, 2)
    );

    // Pair all mass center separations (r) and dihedral angles (omega)
    let r_and_omega = distances
        .iter()
        .copied()
        .cartesian_product(dihedral_angles.iter().copied())
        .collect_vec();

    // Populate 6D table with inter-particle energies
    if backend.prefers_batch() {
        // Batched processing (for GPU): batch by distance for progress while keeping large batches
        for r in distances.iter().progress_count(distances.len() as u64) {
            // Collect all poses for this distance (all omega and vertex combinations)
            let mut poses: Vec<PoseParams> = Vec::new();
            let mut data_refs: Vec<_> = Vec::new();

            for omega in &dihedral_angles {
                table
                    .get_icospheres(*r, *omega)
                    .expect("invalid (r, omega) value")
                    .flat_iter()
                    .for_each(|(pos_a, pos_b, data_b)| {
                        poses.push(PoseParams {
                            r: *r,
                            omega: *omega,
                            vertex_i: *pos_a,
                            vertex_j: *pos_b,
                        });
                        data_refs.push(data_b);
                    });
            }

            // Compute all energies for this distance in one GPU batch
            let energies = backend.compute_energies(&poses);

            // Store results
            for (data_b, energy) in data_refs.into_iter().zip(energies.into_iter()) {
                data_b.set(energy).expect("Energy already calculated");
            }
        }
    } else {
        // Per-pose processing with rayon parallelization (for CPU)
        info!(
            "Computing {} (r,omega) pairs using CPU backend...",
            r_and_omega.len()
        );
        let calc_energy = |r: f64, omega: f64| {
            table
                .get_icospheres(r, omega)
                .expect("invalid (r, omega) value")
                .flat_iter()
                .for_each(|(pos_a, pos_b, data_b)| {
                    let pose = PoseParams {
                        r,
                        omega,
                        vertex_i: *pos_a,
                        vertex_j: *pos_b,
                    };
                    let energy = backend.compute_energy(&pose);
                    data_b.set(energy).expect("Energy already calculated");
                });
        };

        r_and_omega
            .par_iter()
            .progress_count(r_and_omega.len() as u64)
            .for_each(|(r, omega)| {
                calc_energy(*r, *omega);
            });
        info!("Finished computing energies");
    }

    // Write oriented structures to trajectory file
    if let Some(xtcfile) = xtcfile {
        info!("Writing trajectory file {}", xtcfile.display());
        let mut traj = XTCTrajectory::open_write(xtcfile)?;
        let mut energy_file =
            BufWriter::new(open_compressed(&xtcfile.with_extension("energy.dat.gz"))?);
        writeln!(energy_file, "# Energy (kJ/mol)").expect("Failed to write header");
        let mut frame_cnt: usize = 0;
        let mut frame = Frame::new();
        let n = r_and_omega.len();

        // Create new XTC frame from two structures and append to trajectory
        let mut write_frame = |oriented_a: &Structure, oriented_b: &Structure, energy| {
            frame.step = frame_cnt;
            frame.time = frame_cnt as f32;
            frame_cnt += 1;
            frame.coords = oriented_a
                .pos
                .iter()
                .chain(oriented_b.pos.iter())
                .map(|&p| [p.x as f32 * 0.1, p.y as f32 * 0.1, p.z as f32 * 0.1]) // angstrom to nm
                .collect();
            traj.write(&frame).expect("Failed to write XTC frame");
            writeln!(energy_file, "{energy:.6}").expect("Failed to write energy to file");
        };

        r_and_omega
            .into_iter()
            .progress_count(n as u64)
            .for_each(|(r, omega)| {
                table
                    .get_icospheres(r, omega) // remaining 4D
                    .expect("invalid (r, omega) value")
                    .flat_iter()
                    .for_each(|(pos_a, pos_b, _data_b)| {
                        let (oriented_a, oriented_b) =
                            orient_structures(r, omega, *pos_a, *pos_b, ref_a, ref_b);
                        write_frame(&oriented_a, &oriented_b, _data_b.get().unwrap());
                    });
            });
        info!("Wrote {frame_cnt} frames to trajectory file");
    }

    // Partition function contribution for single (r, omega) point
    // i.e. averaged over 4D angular space
    let calc_partition_func = |r: f64, omega: f64| {
        table.get_icospheres(r, omega).unwrap().flat_iter().fold(
            Sample::default(),
            |sum, (vertex_i, vertex_j, data_b)| {
                let degeneracy = vertex_i.norm() * vertex_j.norm();
                let energy = data_b.get().unwrap(); // kJ/mol
                sum + Sample::new(*energy, *temperature, degeneracy)
            },
        )
    };

    // Save table to disk
    if let Some(savetable) = disktable {
        log::info!("Saving 6D table to {}", savetable.display());
        let mut stream = BufWriter::new(open_compressed(savetable)?);
        for r in distances.iter().progress_count(distances.len() as u64) {
            table.stream_angular_space(*r, &mut stream)?;
        }
    }

    // Calculate partition function as function of r only
    let mut samples: Vec<(Vector3, Sample)> = Vec::default();
    for r in &distances {
        let partition_func: Sample = dihedral_angles
            .iter()
            .map(|omega| calc_partition_func(*r, *omega))
            .sum();
        log::debug!(
            "r={:.1}: exp_energy={:.4e}, mean_energy={:.4e}, free_energy={:.4e}",
            r,
            partition_func.exp_energy(),
            partition_func.mean_energy(),
            partition_func.free_energy()
        );
        samples.push((Vector3::new(0.0, 0.0, *r), partition_func));
    }

    let masses = (ref_a.total_mass(), ref_b.total_mass());

    report_pmf(samples.as_slice(), pmf_file, Some(masses))?;
    Ok(())
}
