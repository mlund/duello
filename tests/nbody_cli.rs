//! `duello nbody` runs from a saved scan; `nbody-expand` turns its
//! trajectory into bead XTC.

mod common;

use common::{duello, example, run, workdir};
use duello::nbody::trajectory::TrajectoryReader;

#[test]
fn nbody_runs_from_a_scan_and_expands_to_xtc() {
    let dir = workdir("nbody_cli");
    let protein = example("calvados3/2CDS.cif");
    run(duello(&dir)
        .args(["scan", "-1"])
        .arg(&protein)
        .arg("-2")
        .arg(&protein)
        .args([
            "--rmin",
            "20",
            "--rmax",
            "40",
            "--dr",
            "2",
            "--max-ndiv",
            "1",
        ])
        .args([
            "--cutoff",
            "50",
            "--molarity",
            "0.05",
            "--backend",
            "reference",
        ])
        .args(["--savescan", "scan.duello"]));

    run(duello(&dir)
        .args(["nbody", "--scan", "scan.duello", "--bodies", "20"])
        .args(["--protein-molarity", "0.001", "--copies", "2"])
        .args(["--sweeps", "100", "--frame-interval", "10", "-o", "run.1"])
        .args(["--fixed-steps", "--translation", "2", "--rotation", "0.1"]));

    let csv = std::fs::read_to_string(dir.join("run.1.csv")).unwrap();
    assert!(csv.contains("# bodies: 20\n"), "{csv}");
    assert!(csv.contains("# translation step/Å: 2.0000\n"), "{csv}");
    assert!(csv.contains("# rotation step/rad: 0.1000\n"), "{csv}");
    let rows = csv.lines().skip_while(|l| l.starts_with('#')).skip(1);
    assert!(rows.count() > 10, "{csv}");

    let trajectory = std::fs::read(dir.join("run.1.traj")).unwrap();
    let reader = TrajectoryReader::new(&trajectory[..]).unwrap();
    let beads = 20 * reader.header().reference.positions.len();
    let frames: Vec<_> = reader.map(Result::unwrap).collect();
    // 100 production sweeps at 10 per frame, in each of two copies.
    assert_eq!(frames.len(), 20);

    // Writing over the input must fail and leave it intact.
    let clobber = duello(&dir)
        .args(["nbody-expand", "run.1.traj", "-o", "run.1.traj"])
        .output()
        .unwrap();
    assert!(!clobber.status.success());
    assert_eq!(std::fs::read(dir.join("run.1.traj")).unwrap(), trajectory);

    run(duello(&dir).args(["nbody-expand", "run.1.traj", "-o", "run.xtc", "--copy", "1"]));
    let xtc = std::fs::read(dir.join("run.xtc")).unwrap();
    let xtc_frames = molly::XTCReader::new(&xtc[..]).read_all_frames().unwrap();
    assert_eq!(xtc_frames.len(), 10);
    assert_eq!(xtc_frames[0].positions.len(), 3 * beads);
    let xyz = std::fs::read_to_string(dir.join("run.xyz")).unwrap();
    assert_eq!(xyz.lines().next(), Some(beads.to_string().as_str()));
}
