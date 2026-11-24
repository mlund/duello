//! `duello modes` on a saved scan: the table, CG poses and all-atom models.

mod common;

use common::{duello, example, run, workdir};
use std::path::Path;

/// A coarse 2CDS homodimer scan, saved as a session.
fn scan(dir: &Path) {
    let cif = example("calvados3/2CDS.cif");
    run(duello(dir)
        .args(["scan", "-1"])
        .arg(&cif)
        .arg("-2")
        .arg(&cif)
        .args([
            "--rmin",
            "28",
            "--rmax",
            "32",
            "--dr",
            "2",
            "--max-ndiv",
            "0",
        ])
        .args(["--ionic-strength", "0.05", "--backend", "simd"])
        .args(["--savescan", "scan.duello.zip"]));
}

fn rows(csv: &Path) -> usize {
    std::fs::read_to_string(csv).unwrap().lines().count() - 1
}

#[test]
fn modes_are_written_as_cg_and_with_input_structures_all_atom() {
    let dir = workdir("modes_cli");
    scan(&dir);
    let cif = example("calvados3/2CDS.cif");

    run(duello(&dir)
        .args(["modes", "scan.duello.zip", "-1"])
        .arg(&cif)
        .arg("-2")
        .arg(&cif)
        .args(["-n", "3", "-o", "allatom"]));
    let n = rows(&dir.join("allatom/modes.csv"));
    assert!((1..=3).contains(&n), "{n} modes");
    assert!(dir.join("allatom/mode01_cg.xyz").exists());
    let pdb = std::fs::read_to_string(dir.join("allatom/modes_allatom.pdb")).unwrap();
    assert_eq!(pdb.lines().filter(|l| l.starts_with("MODEL")).count(), n);

    // Without structures (a CLI session keeps none), only the CG files.
    run(duello(&dir).args(["modes", "scan.duello.zip", "-o", "cg"]));
    assert!(rows(&dir.join("cg/modes.csv")) >= n); // default -n 10
    assert!(dir.join("cg/mode01_cg.xyz").exists());
    assert!(!dir.join("cg/modes_allatom.pdb").exists());

    // A smaller rerun into the same directory leaves no files of the first.
    run(duello(&dir).args(["modes", "scan.duello.zip", "-n", "1", "-o", "allatom"]));
    assert!(!dir.join("allatom/mode02_cg.xyz").exists());
    assert!(!dir.join("allatom/modes_allatom.pdb").exists());
}
