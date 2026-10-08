// Copyright 2023-2024 Mikael Lund
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

//! Gromacs XTC output of scanned poses with a CSV sidecar.
//!
//! The two files are written in lockstep by [`TrajectoryWriter`], so a frame
//! selected in a viewer identifies exactly one CSV row and thus one energy.

use crate::icoscan::{oriented_positions, Pose};
use crate::structure::Structure;
use crate::{Table6DAdaptive, UnitQuaternion, Vector3};
use anyhow::Result;
use faunus::topology::io::ANGSTROM_TO_NM;
use faunus::topology::AtomKind;
use flate2::{write::GzEncoder, Compression};
use icotable::{AngularMesh, MeshLevel, SlabResolution};
use molly::{Frame, XTCWriter};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

const CSV_HEADER: &str = "frame,r_angstrom,omega_rad,n_div,vi,vj,resolution,energy_kjmol";

/// Gromacs default, i.e. 0.001 nm coordinate resolution.
const XTC_PRECISION: f32 = 1000.0;

/// Enough to absorb molly's handful of small writes per frame.
const IO_BUFFER: usize = 1 << 20;

/// The three files making up one trajectory.
pub struct TrajectoryPaths {
    pub xtc: PathBuf,
    pub csv: PathBuf,
    pub topology: PathBuf,
}

impl TrajectoryPaths {
    /// Derive the sidecar paths from the trajectory path.
    ///
    /// Suffixes are appended rather than replacing the extension, so that no
    /// argument can derive onto itself and have one file overwrite another.
    pub fn derive(xtc: &Path) -> Self {
        let stem = if xtc.extension().is_some_and(|e| e.eq_ignore_ascii_case("xtc")) {
            xtc.with_extension("")
        } else {
            xtc.to_path_buf()
        };
        let with_suffix = |suffix: &str| {
            let mut name = stem.clone().into_os_string();
            name.push(suffix);
            PathBuf::from(name)
        };
        Self {
            xtc: xtc.to_path_buf(),
            csv: with_suffix(".energy.csv.gz"),
            topology: with_suffix(".xyz"),
        }
    }
}

/// Writes poses as XTC frames plus one CSV row each.
///
/// Generic over both sinks so that tests can round-trip through memory; see
/// [`TrajectoryWriter::create`] for the file-backed pair.
pub struct TrajectoryWriter<X: Write, E: Write> {
    xtc: XTCWriter<X>,
    csv: E,
    frames: u32,
    /// Reused across frames; refilling beats reallocating tens of millions of times.
    frame: Frame,
    row: String,
}

impl<X: Write, E: Write> TrajectoryWriter<X, E> {
    pub fn new(xtc: X, mut csv: E) -> Result<Self> {
        writeln!(csv, "{CSV_HEADER}")?;
        Ok(Self {
            xtc: XTCWriter::new(xtc),
            csv,
            frames: 0,
            frame: Frame {
                precision: XTC_PRECISION,
                ..Default::default()
            },
            row: String::new(),
        })
    }

    /// Append one frame and its matching CSV row.
    ///
    /// `positions` are the already oriented coordinates of both molecules,
    /// molecule A first.
    pub fn write_pose(
        &mut self,
        positions: impl Iterator<Item = Vector3>,
        pose: &Pose,
        energy: f64,
    ) -> Result<()> {
        self.frame.step = self.frames;
        self.frame.time = self.frames as f32;
        self.frame.positions.clear();
        self.frame.positions.extend(positions.flat_map(|p| {
            [
                (p.x * ANGSTROM_TO_NM) as f32,
                (p.y * ANGSTROM_TO_NM) as f32,
                (p.z * ANGSTROM_TO_NM) as f32,
            ]
        }));
        self.xtc.write_frame(&self.frame)?;

        // One write_all per row: GzEncoder deflates on every call, and a bare
        // writeln! would issue one per literal and per argument.
        use std::fmt::Write as _;
        self.row.clear();
        let resolution = if pose.isotropic { "scalar" } else { "mesh" };
        writeln!(
            self.row,
            "{},{:.3},{:.6},{},{},{},{},{:.6e}",
            self.frames, pose.r, pose.omega, pose.n_div, pose.vi, pose.vj, resolution, energy
        )
        .expect("formatting into a String cannot fail");
        self.csv.write_all(self.row.as_bytes())?;

        self.frames += 1;
        Ok(())
    }

    /// Set a cubic box of `side` ångström for all later frames.
    ///
    /// Poses are not wrapped into it; it exists because tools such as
    /// `gmx trjconv -pbc` reject an all-zero box.
    fn set_cubic_box(&mut self, side: f64) {
        let side = (side * ANGSTROM_TO_NM) as f32;
        self.frame.boxvec = [side, 0.0, 0.0, 0.0, side, 0.0, 0.0, 0.0, side];
    }
}

/// The file-backed pair, as produced by [`TrajectoryPaths`].
pub type FileTrajectoryWriter = TrajectoryWriter<BufWriter<File>, GzEncoder<BufWriter<File>>>;

impl FileTrajectoryWriter {
    pub fn create(paths: &TrajectoryPaths) -> Result<Self> {
        info!(
            "Writing trajectory {} and energies {}",
            paths.xtc.display(),
            paths.csv.display()
        );
        let buffered = |path: &Path| -> Result<BufWriter<File>> {
            Ok(BufWriter::with_capacity(IO_BUFFER, File::create(path)?))
        };
        // Fast deflate: the rows are highly repetitive and the volume is large.
        let csv = GzEncoder::new(buffered(&paths.csv)?, Compression::fast());
        Self::new(buffered(&paths.xtc)?, csv)
    }

    /// Flush both sinks, completing the gzip stream.
    ///
    /// Explicit rather than left to `Drop`, which cannot report failure.
    pub fn finish(self) -> Result<()> {
        let frames = self.frames;
        self.xtc.file.into_inner()?.sync_all()?;
        self.csv.finish()?.into_inner()?.sync_all()?;
        info!("Wrote {frames} frames to trajectory");
        Ok(())
    }
}

/// Append every stored pose of `table`, one frame and one CSV row each.
///
/// Reads the finished table rather than the raw scan, so the trajectory shows
/// the same coarse-grained energies the PMF and B₂ are derived from. Repulsive
/// slabs store no energy and are skipped.
pub fn write_table_poses(
    writer: &mut TrajectoryWriter<impl Write, impl Write>,
    table: &Table6DAdaptive<f32>,
    omega_step: f64,
    ref_a: &Structure,
    ref_b: &Structure,
) -> Result<()> {
    let slice = SliceContext {
        table,
        omega_step,
        ref_a,
        ref_b,
    };
    writer.set_cubic_box(box_side(table, ref_a, ref_b));
    log_estimate(table, ref_a.pos.len() + ref_b.pos.len());

    let mut skipped = 0;
    for ri in 0..table.n_r {
        // Scalar-only and repulsive R-bins record no mesh level, so no pose is defined.
        let Some((energies, level)) = table.energies_at_r(ri) else {
            debug!(
                "R={:.1} Å: no stored orientations, no frames written",
                r_value(table, ri)
            );
            skipped += 1;
            continue;
        };
        slice.write_slice(writer, ri, &energies, level)?;
    }
    if skipped > 0 {
        info!(
            "{skipped} of {} R-bins had no stored orientations (scalar or repulsive)",
            table.n_r
        );
    }
    Ok(())
}

/// Write the atom names XTC cannot carry, positioned at the trajectory's first
/// pose so that the file is a real frame rather than invented geometry.
pub fn write_topology(
    paths: &TrajectoryPaths,
    table: &Table6DAdaptive<f32>,
    omega_step: f64,
    ref_a: &Structure,
    ref_b: &Structure,
    atom_kinds: &[AtomKind],
) -> Result<()> {
    let Some((q_b, separation)) = first_stored_orientation(table, omega_step) else {
        anyhow::bail!("No stored poses: nothing to write a trajectory for");
    };
    info!("Writing trajectory topology {}", paths.topology.display());
    let mut merged = ref_a.clone();
    let mut mol_b = ref_b.clone();
    mol_b.transform(|pos| q_b.transform_vector(&pos) + separation);
    merged = merged + mol_b;
    merged.to_xyz(&mut File::create(&paths.topology)?, atom_kinds)
}

/// Orientation of the first pose [`write_table_poses`] emits, found with the
/// same predicate so the two cannot disagree.
fn first_stored_orientation(
    table: &Table6DAdaptive<f32>,
    omega_step: f64,
) -> Option<(UnitQuaternion, Vector3)> {
    for ri in 0..table.n_r {
        // Skip, not bail: leading R-bins are routinely all repulsive.
        let Some((energies, level)) = table.energies_at_r(ri) else {
            continue;
        };
        let n_omega = table.n_omega;
        let found = (0..level.len() * level.len() * n_omega)
            .find(|&k| energies[k].is_finite())
            .map(|k| (k / n_omega / level.len(), k / n_omega % level.len(), k % n_omega));
        if let Some((vi, vj, oi)) = found {
            let (_, q_b, separation) = icotable::orient(
                r_value(table, ri),
                oi as f64 * omega_step,
                &level.direction(vi),
                &level.direction(vj),
            );
            return Some((q_b, separation));
        }
    }
    None
}

/// The table plus what a slice needs to turn an index triplet into coordinates.
struct SliceContext<'a> {
    table: &'a Table6DAdaptive<f32>,
    omega_step: f64,
    ref_a: &'a Structure,
    ref_b: &'a Structure,
}

impl SliceContext<'_> {
    /// Emit one radial slice, laid out by [`Table6DAdaptive::energies_at_r`] as
    /// vertex i, then vertex j, then ω.
    fn write_slice(
        &self,
        writer: &mut TrajectoryWriter<impl Write, impl Write>,
        ri: usize,
        energies: &[f64],
        level: &MeshLevel,
    ) -> Result<()> {
        let n_v = level.len();
        let n_omega = self.table.n_omega;
        let r = r_value(self.table, ri);
        // Both tables are shared by every vertex pair in the slice.
        let q_omega: Vec<UnitQuaternion> = (0..n_omega)
            .map(|oi| {
                UnitQuaternion::from_axis_angle(&Vector3::z_axis(), oi as f64 * self.omega_step)
            })
            .collect();
        let isotropic: Vec<bool> = (0..n_omega)
            .map(|oi| {
                matches!(
                    self.table.slab_res[ri * n_omega + oi],
                    SlabResolution::Scalar(_)
                )
            })
            .collect();

        for vi in 0..n_v {
            let dir_i = level.direction(vi);
            for vj in 0..n_v {
                // q_b(ω) = q_b(0)·q₂(ω), since q₂(0) is the identity -- so the
                // four square roots inside `orient` are paid once per pair.
                let (_, q_b, separation) =
                    icotable::orient(r, 0.0, &dir_i, &level.direction(vj));
                let row = &energies[(vi * n_v + vj) * n_omega..][..n_omega];
                for (oi, &energy) in row.iter().enumerate() {
                    if !energy.is_finite() {
                        continue;
                    }
                    let pose = Pose {
                        r,
                        omega: oi as f64 * self.omega_step,
                        vi,
                        vj,
                        n_div: level.n_div,
                        isotropic: isotropic[oi],
                    };
                    let positions =
                        oriented_positions(q_b * q_omega[oi], separation, self.ref_a, self.ref_b);
                    writer.write_pose(positions, &pose, energy)?;
                }
            }
        }
        Ok(())
    }
}

/// Separation of radial bin `ri`; the table stores only the range and step.
fn r_value(table: &Table6DAdaptive<f32>, ri: usize) -> f64 {
    table.rmin + ri as f64 * table.dr
}

/// Cube side enclosing the widest pose, in ångström.
fn box_side(table: &Table6DAdaptive<f32>, ref_a: &Structure, ref_b: &Structure) -> f64 {
    let radius = |s: &Structure| s.pos.iter().map(nalgebra::Vector3::norm).fold(0.0, f64::max);
    table.rmax + 2.0 * (radius(ref_a) + radius(ref_b))
}

/// Log the frame count and rough file size before writing, since a full scan can
/// emit tens of millions of frames.
fn log_estimate(table: &Table6DAdaptive<f32>, n_atoms: usize) {
    let frames: usize = (0..table.n_r)
        .map(|ri| {
            let Some(level) = table.mesh_level_at_r(ri) else {
                return 0;
            };
            // Repulsive slabs are the only ones holding no energy.
            let stored = (0..table.n_omega)
                .filter(|oi| {
                    !matches!(
                        table.slab_res[ri * table.n_omega + oi],
                        SlabResolution::Repulsive
                    )
                })
                .count();
            stored * level.len() * level.len()
        })
        .sum();
    // XTC compresses to roughly 6 bytes per atom at this precision.
    let megabytes = (frames * n_atoms * 6) as f64 / 1e6;
    info!("Trajectory: {frames} frames, roughly {megabytes:.0} MB");
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    use icotable::{AdaptiveBuilder, Subdivision};

    fn atom(pos: Vector3) -> crate::structure::Structure {
        crate::structure::Structure {
            pos: vec![pos],
            masses: vec![1.0],
            charges: vec![0.0],
            radii: vec![1.0],
            atom_ids: vec![0],
        }
    }

    /// A table whose first slab is driven repulsive, the rest attractive and
    /// angularly varying so it stays a Mesh rather than collapsing to Scalar.
    fn planted_table() -> (Table6DAdaptive<f32>, f64, usize) {
        let subdivision = Subdivision::Geodesic;
        let omega_step = subdivision.angular_resolution(0);
        let mut builder = AdaptiveBuilder::with_subdivision(
            subdivision, 10.0, 12.0, 1.0, omega_step, 0, 1e-12, 1.0,
        );
        let n_v = builder.current_n_vertices();
        for ri in 0..builder.n_r() {
            for oi in 0..builder.n_omega() {
                let energies: Vec<f64> = (0..n_v * n_v)
                    .map(|k| {
                        if ri == 0 && oi == 0 {
                            1e6
                        } else {
                            -1.0 - 0.01 * k as f64
                        }
                    })
                    .collect();
                builder.set_slab(ri, oi, &energies);
            }
            builder.finish_r_slice(ri);
        }
        (builder.build(), omega_step, n_v)
    }

    /// Repulsive slabs store no energy, so they contribute no frames -- and the
    /// CSV must stay row-per-frame regardless.
    #[test]
    fn repulsive_slabs_emit_no_frames() {
        let (table, omega_step, n_v) = planted_table();
        let (a, b) = (atom(Vector3::zeros()), atom(Vector3::zeros()));
        let mut xtc = Vec::new();
        let mut csv = Vec::new();

        {
            let mut writer = TrajectoryWriter::new(&mut xtc, &mut csv).unwrap();
            write_table_poses(&mut writer, &table, omega_step, &a, &b).unwrap();
        }

        let expected = (table.n_r * table.n_omega - 1) * n_v * n_v;
        let frames = molly::XTCReader::new(&xtc[..]).read_all_frames().unwrap();
        assert_eq!(frames.len(), expected);

        let text = String::from_utf8(csv).unwrap();
        let rows: Vec<&str> = text.lines().skip(1).collect();
        assert_eq!(rows.len(), expected);

        // Nothing from the repulsive slab at (r=10, omega=0).
        assert!(!rows.iter().any(|row| {
            let f: Vec<&str> = row.split(',').collect();
            f[1].parse::<f64>().unwrap() == 10.0 && f[2].parse::<f64>().unwrap() == 0.0
        }));
        // Energies survive as stored: f32, angularly varying, attractive.
        let fields: Vec<&str> = rows[0].split(',').collect();
        let energy: f64 = fields[7].parse().unwrap();
        assert!(energy < 0.0, "expected attractive energy, got {energy}");
        // Vertex indices are meaningless without the level they belong to.
        assert_eq!(fields[3], "0");
        assert_eq!(fields[6], "mesh");
    }

    /// A pose whose orientation is irrelevant; only the CSV columns are read.
    fn pose(r: f64, omega: f64, vi: usize, vj: usize) -> Pose {
        Pose {
            r,
            omega,
            vi,
            vj,
            n_div: 0,
            isotropic: false,
        }
    }

    /// `write_slice` hoists `orient` out of the ω loop via q_b(ω) = q_b(0)·q₂(ω).
    /// Were that identity to break, every frame would be rotated wrongly while
    /// the energies stayed right -- silent, and invisible in the CSV.
    #[test]
    fn hoisted_omega_rotation_matches_orient() {
        let dir_i = Vector3::new(0.3, -0.5, 0.81).normalize();
        let dir_j = Vector3::new(-0.7, 0.2, 0.68).normalize();
        let probe = Vector3::new(1.0, 2.0, 3.0);
        let (_, q_b0, separation0) = icotable::orient(31.0, 0.0, &dir_i, &dir_j);

        for oi in 1..8 {
            let omega = oi as f64 * 0.3;
            let (_, q_b, separation) = icotable::orient(31.0, omega, &dir_i, &dir_j);
            let hoisted = q_b0 * UnitQuaternion::from_axis_angle(&Vector3::z_axis(), omega);
            let (direct, rebuilt) = (q_b.transform_vector(&probe), hoisted.transform_vector(&probe));
            for k in 0..3 {
                assert_relative_eq!(direct[k], rebuilt[k], epsilon = 1e-12);
                // Separation depends on r and dir_i only, hence hoisted too.
                assert_relative_eq!(separation[k], separation0[k], epsilon = 1e-12);
            }
        }
    }

    /// A frame in a viewer must map to exactly one CSV row, so that a pose can
    /// be affiliated with its energy.
    #[test]
    fn xtc_frames_couple_to_csv_rows() {
        let positions = [Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 2.0, 0.0)];
        let mut xtc = Vec::new();
        let mut csv = Vec::new();

        {
            let mut writer = TrajectoryWriter::new(&mut xtc, &mut csv).unwrap();
            writer
                .write_pose(positions.iter().copied(), &pose(30.0, 0.25, 3, 7), -1.5)
                .unwrap();
            writer
                .write_pose(positions.iter().copied(), &pose(30.5, 0.5, 1, 2), 2.25)
                .unwrap();
        }

        let frames = molly::XTCReader::new(&xtc[..]).read_all_frames().unwrap();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].natoms(), 2);
        // Positions are ångström on the way in, nanometre on the way out.
        assert_relative_eq!(frames[0].positions[0], 0.1, epsilon = 1e-3);
        assert_relative_eq!(frames[0].positions[4], 0.2, epsilon = 1e-3);
        // The frame index is the coupling key between the two files.
        assert_eq!(frames[1].step, 1);

        let csv = String::from_utf8(csv).unwrap();
        let rows: Vec<&str> = csv.lines().collect();
        assert_eq!(rows[0], CSV_HEADER);
        assert_eq!(rows.len(), 3);

        let fields: Vec<&str> = rows[2].split(',').collect();
        assert_eq!(fields[0], "1");
        assert_relative_eq!(fields[1].parse::<f64>().unwrap(), 30.5);
        assert_relative_eq!(fields[2].parse::<f64>().unwrap(), 0.5, epsilon = 1e-6);
        assert_eq!((fields[3], fields[4], fields[5]), ("0", "1", "2"));
        assert_eq!(fields[6], "mesh");
        assert_relative_eq!(fields[7].parse::<f64>().unwrap(), 2.25);
    }
}
