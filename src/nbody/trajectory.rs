//! Many-body trajectory: positions and orientations only.
//!
//! Bodies are rigid, so poses plus the reference beads in the header give every
//! bead coordinate; offline analysis can apply any form factors or hydration
//! model without rerunning MC.
//!
//! Layout, all little-endian:
//! - magic `DUELLO-TRAJ\0`, then `u32` format version
//! - `u32` header length, then the header as UTF-8 JSON
//! - fixed-size records: `u32` copy, `u64` sweep, `f64` box length, then per
//!   body `f64` x, y, z, qw, qx, qy, qz
//!
//! f64, not f32: rounding could move a pair sampled just outside contact
//! inside it, so offline energies would see an overlap the run never had.
//! Fixed records let readers such as numpy map frames without parsing.

use super::{Frame, Pose};
use crate::archive::{CgBody, ScanSettings};
use crate::{UnitQuaternion, Vector3};
use anyhow::{ensure, Result};
use nalgebra::Quaternion;
use serde::{Deserialize, Serialize};
use std::io::{ErrorKind, Read, Write};

/// Run description stored once at the start of the file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrajectoryHeader {
    pub n_bodies: usize,
    /// Protein concentration in mol/L (the salt is in `settings.molarity`).
    pub protein_molarity: f64,
    pub sweeps_per_frame: u64,
    pub equilibration_sweeps: u64,
    /// False if equilibration hit its sweep cap without converging.
    pub equilibrated: bool,
    /// Beads in the body frame. All bodies are this one species: runs
    /// accept only homo-dimer scans.
    pub reference: CgBody,
    /// The scan the table came from, including its temperature.
    pub settings: ScanSettings,
}

const MAGIC: &[u8; 12] = b"DUELLO-TRAJ\0";
/// Bumped on any layout change, so old readers fail rather than misread.
const FORMAT_VERSION: u32 = 1;
/// A header is a few kB; a larger length means a damaged or foreign file.
const MAX_HEADER_BYTES: u32 = 64 << 20;
/// Far above any run; a larger count means a damaged or foreign header.
pub const MAX_BODIES: usize = 1 << 20;

/// copy, sweep, box, then seven f64 per body.
fn record_size(n_bodies: usize) -> usize {
    4 + 8 + 8 + 56 * n_bodies
}

pub struct TrajectoryWriter<W: Write> {
    sink: W,
    n_bodies: usize,
    /// Reused for every record, as the size is fixed.
    record: Vec<u8>,
    /// Set after a failed write, which may have left a partial record; a
    /// later record would land misaligned and corrupt every frame after it.
    failed: bool,
}

impl<W: Write> TrajectoryWriter<W> {
    pub fn new(header: &TrajectoryHeader, mut sink: W) -> Result<Self> {
        ensure!(
            header.n_bodies <= MAX_BODIES,
            "too many bodies for a trajectory"
        );
        let json = serde_json::to_vec(header)?;
        sink.write_all(MAGIC)?;
        sink.write_all(&FORMAT_VERSION.to_le_bytes())?;
        sink.write_all(&u32::try_from(json.len())?.to_le_bytes())?;
        sink.write_all(&json)?;
        Ok(Self {
            sink,
            n_bodies: header.n_bodies,
            record: Vec::with_capacity(record_size(header.n_bodies)),
            failed: false,
        })
    }

    /// Writes one record in a single call, so an interrupted run leaves at
    /// most one partial record, which readers skip.
    pub fn append(&mut self, frame: &Frame) -> Result<()> {
        ensure!(
            !self.failed,
            "an earlier write failed; the trajectory ends there"
        );
        ensure!(
            frame.poses.len() == self.n_bodies,
            "frame has {} bodies, header says {}",
            frame.poses.len(),
            self.n_bodies
        );
        self.encode(frame)?;
        let written = self.sink.write_all(&self.record);
        self.failed = written.is_err();
        Ok(written?)
    }

    fn encode(&mut self, frame: &Frame) -> Result<()> {
        let record = &mut self.record;
        record.clear();
        record.extend(u32::try_from(frame.copy)?.to_le_bytes());
        record.extend(frame.sweep.to_le_bytes());
        record.extend(frame.box_length.to_le_bytes());
        for pose in &frame.poses {
            let (p, q) = (pose.position, pose.orientation.quaternion());
            for value in [p.x, p.y, p.z, q.w, q.i, q.j, q.k] {
                record.extend(value.to_le_bytes());
            }
        }
        Ok(())
    }

    pub fn into_inner(self) -> W {
        self.sink
    }

    /// What has been written so far, e.g. to offer a download mid-run.
    pub fn sink(&self) -> &W {
        &self.sink
    }

    /// Bytes each `append` adds, for callers enforcing a size cap.
    pub fn record_bytes(&self) -> usize {
        record_size(self.n_bodies)
    }
}

/// Streams frames one at a time, so trajectories larger than memory work.
/// Iteration stops at a trailing partial record, as left by a stopped run.
pub struct TrajectoryReader<R: Read> {
    header: TrajectoryHeader,
    source: R,
    record: Vec<u8>,
}

impl<R: Read> TrajectoryReader<R> {
    pub fn new(mut source: R) -> Result<Self> {
        let header = read_header(&mut source)?;
        let record = vec![0u8; record_size(header.n_bodies)];
        Ok(Self {
            header,
            source,
            record,
        })
    }

    pub fn header(&self) -> &TrajectoryHeader {
        &self.header
    }
}

impl<R: Read> Iterator for TrajectoryReader<R> {
    type Item = Result<Frame>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.source.read_exact(&mut self.record) {
            Ok(()) => Some(Ok(decode_record(&self.record, self.header.n_bodies))),
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => None,
            Err(e) => Some(Err(e.into())),
        }
    }
}

fn read_header(source: &mut impl Read) -> Result<TrajectoryHeader> {
    let mut magic = [0u8; 12];
    source.read_exact(&mut magic)?;
    ensure!(&magic == MAGIC, "not a duello trajectory");
    let version = read_u32(source)?;
    ensure!(
        version == FORMAT_VERSION,
        "trajectory format version {version} is not supported (expected {FORMAT_VERSION})"
    );
    let length = read_u32(source)?;
    ensure!(length <= MAX_HEADER_BYTES, "trajectory header too large");
    let mut json = vec![0u8; length as usize];
    source.read_exact(&mut json)?;
    let header: TrajectoryHeader = serde_json::from_slice(&json)?;
    ensure!(
        header.n_bodies <= MAX_BODIES,
        "implausible body count {}",
        header.n_bodies
    );
    Ok(header)
}

fn read_u32(source: &mut impl Read) -> Result<u32> {
    let mut bytes = [0u8; 4];
    source.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn decode_record(record: &[u8], n_bodies: usize) -> Frame {
    let f64_at = |offset: usize| f64::from_le_bytes(record[offset..offset + 8].try_into().unwrap());
    let pose = |body: usize| {
        let v = |k: usize| f64_at(20 + 56 * body + 8 * k);
        Pose {
            position: Vector3::new(v(0), v(1), v(2)),
            orientation: UnitQuaternion::new_unchecked(Quaternion::new(v(3), v(4), v(5), v(6))),
        }
    };
    Frame {
        copy: u32::from_le_bytes(record[0..4].try_into().unwrap()) as usize,
        sweep: u64::from_le_bytes(record[4..12].try_into().unwrap()),
        box_length: f64_at(12),
        poses: (0..n_bodies).map(pose).collect(),
    }
}

/// Lab-frame bead positions of one body: R + q·r for each reference bead.
pub fn expand(reference: &CgBody, pose: &Pose) -> Vec<Vector3> {
    reference
        .positions
        .iter()
        .map(|&r| pose.position + pose.orientation * Vector3::from(r))
        .collect()
}

pub struct XtcSummary {
    pub frames: usize,
    /// First written frame as XYZ (Å); viewers need it as topology.
    pub topology_xyz: String,
}

/// Expand a trajectory into bead coordinates as XTC (nm), for tools such as
/// pripps, VMD or MDAnalysis. Copies are independent runs, so viewing
/// needs `copy` set; scattering may average over all.
pub fn to_xtc(trajectory: impl Read, xtc: impl Write, copy: Option<usize>) -> Result<XtcSummary> {
    let reader = TrajectoryReader::new(trajectory)?;
    let reference = reader.header().reference.clone();
    let mut writer = molly::XTCWriter::new(xtc);
    let mut topology_xyz = None;
    let mut frames = 0;
    for frame in reader {
        let frame = frame?;
        if copy.is_some_and(|c| c != frame.copy) {
            continue;
        }
        let beads: Vec<Vector3> = frame
            .poses
            .iter()
            .flat_map(|p| expand(&reference, p))
            .collect();
        topology_xyz.get_or_insert_with(|| xyz(&reference.kinds, &beads));
        writer.write_frame(&xtc_frame(&frame, &beads))?;
        frames += 1;
    }
    let topology_xyz = topology_xyz.ok_or_else(|| anyhow::anyhow!("no frames to expand"))?;
    Ok(XtcSummary {
        frames,
        topology_xyz,
    })
}

fn xtc_frame(frame: &Frame, beads: &[Vector3]) -> molly::Frame {
    use faunus::topology::io::ANGSTROM_TO_NM;
    let side = (frame.box_length * ANGSTROM_TO_NM) as f32;
    molly::Frame {
        step: frame.sweep as u32,
        time: frame.sweep as f32,
        boxvec: [side, 0.0, 0.0, 0.0, side, 0.0, 0.0, 0.0, side],
        precision: 1000.0,
        positions: beads
            .iter()
            .flat_map(|p| [p.x, p.y, p.z].map(|x| (x * ANGSTROM_TO_NM) as f32))
            .collect(),
    }
}

/// Kinds repeat per body, as all bodies are the reference species.
fn xyz(kinds: &[String], beads: &[Vector3]) -> String {
    let mut xyz = format!("{}\nGenerated by Duello\n", beads.len());
    for (kind, p) in kinds.iter().cycle().zip(beads) {
        xyz += &format!("{kind} {:.3} {:.3} {:.3}\n", p.x, p.y, p.z);
    }
    xyz
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(n_bodies: usize) -> TrajectoryHeader {
        TrajectoryHeader {
            n_bodies,
            protein_molarity: 0.005,
            sweeps_per_frame: 700,
            equilibration_sweeps: 5000,
            equilibrated: true,
            reference: CgBody {
                positions: vec![[1.0, 0.0, 0.0]],
                masses: vec![110.0],
                charges: vec![1.0],
                radii: vec![3.0],
                kinds: vec!["LYS".into()],
            },
            settings: ScanSettings {
                ph: 7.0,
                model: "calvados3".into(),
                single_bead: true,
                rmin: 20.0,
                rmax: 60.0,
                dr: 0.5,
                molarity: 0.05,
                temperature: 298.15,
                max_ndiv: 2,
                gradient_threshold: 0.5,
                cutoff: 30.0,
                grid_size: 200,
                homo_dimer: true,
            },
        }
    }

    /// 0.1 and 404.98 are not exact in f32, so the round trip shows the
    /// sampler's f64 state survives unchanged.
    fn frame(copy: usize, sweep: u64) -> Frame {
        let pose = |x: f64| Pose {
            position: Vector3::new(x, 0.1, 100.25),
            orientation: UnitQuaternion::from_euler_angles(0.1, -0.7, 2.3),
        };
        Frame {
            copy,
            sweep,
            box_length: 404.98,
            poses: vec![pose(1.0), pose(-3.75)],
        }
    }

    fn write(frames: &[Frame]) -> Vec<u8> {
        let mut writer = TrajectoryWriter::new(&header(2), Vec::new()).unwrap();
        frames.iter().for_each(|f| writer.append(f).unwrap());
        writer.into_inner()
    }

    fn read_all(bytes: &[u8]) -> (TrajectoryHeader, Vec<Frame>) {
        let reader = TrajectoryReader::new(bytes).unwrap();
        let header = reader.header().clone();
        (header, reader.collect::<Result<_>>().unwrap())
    }

    #[test]
    fn round_trip_is_exact() {
        let frames = [frame(0, 700), frame(3, 1400)];
        let (restored_header, restored) = read_all(&write(&frames));
        assert_eq!(restored_header, header(2));
        assert_eq!(restored, frames);
    }

    /// Written field by field from the documented layout, independently of
    /// the writer, so it pins the format that external readers rely on.
    fn golden_bytes() -> Vec<u8> {
        let json = br#"{"n_bodies":1,"protein_molarity":0.005,
            "sweeps_per_frame":700,"equilibration_sweeps":5000,"equilibrated":true,
            "reference":{"positions":[[1.0,0.0,0.0]],"masses":[110.0],"charges":[1.0],
            "radii":[3.0],"kinds":["LYS"]},
            "settings":{"ph":7.0,"model":"calvados3","single_bead":true,"rmin":20.0,
            "rmax":60.0,"dr":0.5,"molarity":0.05,"temperature":298.15,"max_ndiv":2,
            "gradient_threshold":0.5,"cutoff":30.0,"grid_size":200,"homo_dimer":true}}"#;
        let mut bytes = b"DUELLO-TRAJ\0".to_vec();
        bytes.extend(1u32.to_le_bytes());
        bytes.extend((json.len() as u32).to_le_bytes());
        bytes.extend(json);
        bytes.extend(2u32.to_le_bytes()); // copy
        bytes.extend(1400u64.to_le_bytes()); // sweep
        bytes.extend(50.0f64.to_le_bytes()); // box
        for value in [1.0f64, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0] {
            bytes.extend(value.to_le_bytes()); // x y z qw qx qy qz
        }
        bytes
    }

    #[test]
    fn golden_bytes_decode() {
        let (restored_header, frames) = read_all(&golden_bytes());
        assert_eq!(restored_header, header(1));
        let frame = &frames[0];
        assert_eq!((frame.copy, frame.sweep, frame.box_length), (2, 1400, 50.0));
        assert_eq!(frame.poses[0].position, Vector3::new(1.0, 2.0, 3.0));
        // qz = 1: a half turn about z.
        let half_turn = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), std::f64::consts::PI);
        assert!(frame.poses[0].orientation.angle_to(&half_turn) < 1e-12);
    }

    /// A run stopped mid-write leaves a partial record; earlier frames survive.
    #[test]
    fn trailing_partial_record_is_ignored() {
        let mut bytes = write(&[frame(0, 700), frame(0, 1400)]);
        bytes.truncate(bytes.len() - 5);
        assert_eq!(read_all(&bytes).1, [frame(0, 700)]);
    }

    #[test]
    fn foreign_file_is_rejected() {
        assert!(TrajectoryReader::new(b"DUELLO-SCAN\0\x01\0\0\0".as_slice()).is_err());
    }

    /// The header is untrusted: a huge count must not overflow the record size.
    #[test]
    fn implausible_body_count_is_rejected() {
        let mut huge = header(1);
        huge.n_bodies = usize::MAX / 2;
        let json = serde_json::to_vec(&huge).unwrap();
        let mut file = b"DUELLO-TRAJ\0".to_vec();
        file.extend(1u32.to_le_bytes());
        file.extend((json.len() as u32).to_le_bytes());
        file.extend(json);
        assert!(TrajectoryReader::new(file.as_slice()).is_err());
    }

    /// Fails on the first record write, then would accept anything.
    struct FailOnce {
        bytes: Vec<u8>,
        failed: bool,
    }

    impl Write for FailOnce {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            let header_written = self.bytes.len() > 20;
            if header_written && !self.failed {
                self.failed = true;
                return Err(std::io::Error::other("disk full"));
            }
            self.bytes.extend(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// After a failed write a later record would land misaligned.
    #[test]
    fn writer_refuses_to_continue_after_a_failed_write() {
        let sink = FailOnce {
            bytes: Vec::new(),
            failed: false,
        };
        let mut writer = TrajectoryWriter::new(&header(2), sink).unwrap();
        assert!(writer.append(&frame(0, 700)).is_err());
        assert!(writer.append(&frame(0, 1400)).is_err());
    }

    /// Bead at r = (1, 0, 0) on a body at (10, 0, 0) turned 90° about z sits
    /// at (10, 1, 0).
    #[test]
    fn expanded_beads_follow_the_pose() {
        let pose = Pose {
            position: Vector3::new(10.0, 0.0, 0.0),
            orientation: UnitQuaternion::from_axis_angle(
                &Vector3::z_axis(),
                std::f64::consts::FRAC_PI_2,
            ),
        };
        let beads = expand(&header(1).reference, &pose);
        approx::assert_relative_eq!(beads[0], Vector3::new(10.0, 1.0, 0.0), epsilon = 1e-12);
    }

    /// The documented Python reader must agree with the Rust one.
    #[test]
    fn python_reader_decodes_the_golden_bytes() {
        let script = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/scripts/read_nbody_trajectory.py"
        );
        let path = std::env::temp_dir().join(format!("duello_golden_{}.traj", std::process::id()));
        std::fs::write(&path, golden_bytes()).unwrap();
        let output = std::process::Command::new("python3")
            .arg("-I")
            .arg(script)
            .arg(&path)
            .output();
        std::fs::remove_file(&path).ok();
        let Ok(output) = output else {
            eprintln!("python3 not found; skipping");
            return;
        };
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert_eq!(
            stdout.trim(),
            "frames=1 bodies=1 copy=2 sweep=1400 box=50.0 first=1.0,2.0,3.0,0.0,0.0,0.0,1.0"
        );
    }

    /// One body at (1, 2, 3) Å turned 90° about z, the other unturned at
    /// the origin; the reference bead sits at (1, 0, 0) Å.
    fn turned_frame(copy: usize, sweep: u64) -> Frame {
        let turn = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), std::f64::consts::FRAC_PI_2);
        Frame {
            copy,
            sweep,
            box_length: 50.0,
            poses: vec![
                Pose {
                    position: Vector3::new(1.0, 2.0, 3.0),
                    orientation: turn,
                },
                Pose {
                    position: Vector3::zeros(),
                    orientation: UnitQuaternion::identity(),
                },
            ],
        }
    }

    fn xtc_frames(xtc: &[u8]) -> Vec<molly::Frame> {
        molly::XTCReader::new(xtc).read_all_frames().unwrap().into()
    }

    #[test]
    fn xtc_holds_lab_frame_beads_in_nm() {
        let mut xtc = Vec::new();
        to_xtc(&write(&[turned_frame(0, 700)])[..], &mut xtc, None).unwrap();
        let frames = xtc_frames(&xtc);
        assert_eq!(frames.len(), 1);
        let p = &frames[0].positions;
        let expected = [0.1, 0.3, 0.3, 0.1, 0.0, 0.0];
        assert_eq!(p.len(), expected.len());
        p.iter()
            .zip(expected)
            .for_each(|(&got, want)| assert!((got - want).abs() < 1e-3, "{p:?}"));
        assert!((frames[0].boxvec[0] - 5.0).abs() < 1e-4);
        assert_eq!(frames[0].step, 700);
    }

    #[test]
    fn xtc_keeps_only_the_chosen_copy() {
        let frames = [turned_frame(0, 1), turned_frame(1, 1), turned_frame(0, 2)];
        let mut xtc = Vec::new();
        let summary = to_xtc(&write(&frames)[..], &mut xtc, Some(0)).unwrap();
        assert_eq!(summary.frames, 2);
        assert_eq!(xtc_frames(&xtc).len(), 2);
    }

    /// VMD and MDAnalysis need a structure with matching atom count.
    #[test]
    fn xtc_comes_with_a_matching_topology() {
        let mut xtc = Vec::new();
        let summary = to_xtc(&write(&[turned_frame(0, 1)])[..], &mut xtc, None).unwrap();
        assert_eq!(
            summary.topology_xyz,
            "2\nGenerated by Duello\nLYS 1.000 3.000 3.000\nLYS 1.000 0.000 0.000\n"
        );
    }

    #[test]
    fn xtc_of_no_frames_is_an_error() {
        let mut xtc = Vec::new();
        assert!(to_xtc(&write(&[turned_frame(1, 1)])[..], &mut xtc, Some(0)).is_err());
    }
}
