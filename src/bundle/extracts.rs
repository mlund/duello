//! Readable copies of results; derived on write, never read back.

use super::zipio::Entry;
use crate::archive::{xyz_with_comment, ScanArchive};
use crate::nbody::RunRecord;
use crate::poses::{self, TablePose};

/// As many as the web's Low Energy Poses window shows.
const POSES: usize = 10;
const XYZ: &str = "chemical/x-xyz";

pub fn twobody(scan: &ScanArchive) -> Vec<Entry<'static>> {
    let pmf = Entry::text(
        "twobody/pmf.csv",
        crate::report::pmf_csv(&scan.points),
        "text/csv",
        "Potential of mean force F(R), mean energy U(R) and its variance C(R)/k",
    );
    let bodies = (1..).zip(&scan.bodies).map(|(i, body)| {
        Entry::text(
            &format!("twobody/mol{i}_cg.xyz"),
            body.to_xyz(),
            XYZ,
            &format!("Molecule {i} beads (Å)"),
        )
    });
    std::iter::once(pmf)
        .chain(bodies)
        .chain(pose_entries(scan))
        .collect()
}

fn pose_entries(scan: &ScanArchive) -> Vec<Entry<'static>> {
    let poses = poses::lowest(scan, POSES);
    let mut csv = String::from("rank,U/kT,R/Å,ω/°\n");
    for (rank, pose) in (1..).zip(&poses) {
        csv += &format!("{rank},{}\n", pose_columns(pose).join(","));
    }
    let table = Entry::text(
        "twobody/poses/poses.csv",
        csv,
        "text/csv",
        "The lowest-energy states of the 6D table",
    );
    let files = (1..).zip(&poses).map(|(rank, pose)| {
        Entry::text(
            &format!("twobody/poses/pose{rank:02}.xyz"),
            pose_xyz(scan, pose),
            XYZ,
            &format!("Pose {rank} of poses.csv, both bodies placed (Å)"),
        )
    });
    std::iter::once(table).chain(files).collect()
}

fn pose_columns(pose: &TablePose) -> [String; 3] {
    [
        format!("{:.4}", pose.energy),
        format!("{:.2}", pose.r),
        format!("{:.1}", pose.omega.to_degrees()),
    ]
}

fn pose_xyz(scan: &ScanArchive, pose: &TablePose) -> String {
    let placed = pose.place(&scan.bodies);
    let atoms = scan
        .bodies
        .iter()
        .zip(&placed)
        .flat_map(|(body, beads)| body.kinds.iter().zip(beads.iter().copied()));
    let [u, r, omega] = pose_columns(pose);
    let comment = format!("U/kT = {u}, R/Å = {r}, ω/° = {omega}");
    xyz_with_comment(&comment, placed[0].len() + placed[1].len(), atoms)
}

pub fn manybody(run: &RunRecord) -> Vec<Entry<'static>> {
    vec![
        Entry::text(
            "manybody/scattering.csv",
            run.report_csv(),
            "text/csv",
            "Scattering I(q), form factor P(q) and structure factors, run statistics as # comments",
        ),
        Entry::text(
            "manybody/rdf.csv",
            run.rdf.to_csv(),
            "text/csv",
            "Mass-centre radial distribution function g(r)",
        ),
        Entry::text(
            "manybody/energy.csv",
            run.energy_csv(),
            "text/csv",
            "Energy per body U/N against sweeps",
        ),
    ]
}
