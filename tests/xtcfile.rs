//! End-to-end check that `scan --xtcfile` emits a readable trajectory whose
//! frames are coupled one-to-one with the energy CSV rows.

use std::io::Read;
use std::path::PathBuf;
use std::process::Command;

fn examples() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/cppm")
}

#[test]
fn scan_writes_coupled_trajectory_and_energies() {
    // Unique per run: a shared path makes concurrent test runs clobber each other.
    let dir = std::env::temp_dir().join(format!("duello_xtcfile_{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).unwrap();
    let xtc = dir.join("traj.xtc");
    let csv = dir.join("traj.energy.csv.gz");

    let mol = examples().join("cppm-p00.xyz");
    let out = Command::new(env!("CARGO_BIN_EXE_duello"))
        .current_dir(&dir)
        .args(["scan", "-1"])
        .arg(&mol)
        .arg("-2")
        .arg(&mol)
        .args(["--top"])
        .arg(examples().join("topology.yaml"))
        .args([
            "--rmin", "38", "--rmax", "42", "--dr", "2", "--max-ndiv", "0", "--cutoff", "2000",
            "--molarity", "0.05",
        ])
        .arg("--xtcfile")
        .arg(&xtc)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "scan failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let bytes = std::fs::read(&xtc).unwrap();
    let frames = molly::XTCReader::new(&bytes[..]).read_all_frames().unwrap();
    assert!(!frames.is_empty(), "no frames written");

    // Both molecules in every frame, and nothing left at the origin.
    assert_eq!(frames[0].natoms(), 2 * 67);
    assert!(frames[0].positions.iter().any(|&x| x.abs() > 1e-6));

    // The topology file XTC cannot carry, written by the same module.
    let topology = std::fs::read_to_string(dir.join("traj.xyz")).unwrap();
    assert_eq!(topology.lines().next().unwrap(), "134");

    let mut text = String::new();
    flate2::read::GzDecoder::new(std::fs::File::open(&csv).unwrap())
        .read_to_string(&mut text)
        .unwrap();
    // The header itself is asserted by the unit tests, against the shared const.
    let mut rows = text.lines();
    rows.next().unwrap();

    // One row per frame, in frame order: the coupling the viewer relies on.
    let rows: Vec<&str> = rows.collect();
    assert_eq!(rows.len(), frames.len());
    for (i, row) in rows.iter().enumerate() {
        let fields: Vec<&str> = row.split(',').collect();
        assert_eq!(fields[0].parse::<usize>().unwrap(), i);
        assert_eq!(frames[i].step as usize, i);
        let r: f64 = fields[1].parse().unwrap();
        assert!((38.0..=42.0).contains(&r), "r out of scanned range: {r}");
        assert!(fields[7].parse::<f64>().unwrap().is_finite());
    }
}
