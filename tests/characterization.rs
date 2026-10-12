//! Pins CLI scan results so refactors of the setup path (loading, coarse-graining,
//! pair merging, pair matrix) can be checked for unchanged behaviour.
//!
//! Golden values come from `--backend reference`, which is deterministic.

mod common;

use common::{duello, example, run, workdir};
use std::path::Path;
use std::process::Command;

const REL_TOL: f64 = 1e-9;

/// Small and fast; resolution is irrelevant to a behaviour check.
const COARSE: [&str; 10] = [
    "--dr",
    "2",
    "--max-ndiv",
    "1",
    "--cutoff",
    "2000",
    "--molarity",
    "0.05",
    "--backend",
    "reference",
];

/// Coarse cppm pair scan; `mol2` may be relative to `dir`.
fn cppm_scan(dir: &Path, mol1: &str, mol2: &Path) -> Command {
    let mut cmd = duello(dir);
    cmd.args(["scan", "-1"])
        .arg(example(mol1))
        .arg("-2")
        .arg(mol2)
        .arg("--top")
        .arg(example("cppm/topology.yaml"))
        .args(["--rmin", "38", "--rmax", "44"])
        .args(COARSE);
    cmd
}

fn b2(dir: &Path) -> f64 {
    let json = std::fs::read_to_string(dir.join("pmf.json")).unwrap();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    v["B2"].as_f64().unwrap()
}

fn assert_rel_eq(actual: f64, expected: f64) {
    let rel = ((actual - expected) / expected).abs();
    assert!(
        rel < REL_TOL,
        "got {actual}, expected {expected} (rel {rel:e})"
    );
}

#[test]
fn scan_hetero_dimer_b2() {
    let dir = workdir("char_hetero");
    run(&mut cppm_scan(
        &dir,
        "cppm/cppm-p00.xyz",
        &example("cppm/cppm-p18.xyz"),
    ));
    assert_rel_eq(b2(&dir), 150934.92547782417);
}

#[test]
fn scan_homo_dimer_b2() {
    let dir = workdir("char_homo");
    run(&mut cppm_scan(
        &dir,
        "cppm/cppm-p18.xyz",
        &example("cppm/cppm-p18.xyz"),
    ));
    assert_rel_eq(b2(&dir), -240356.77888141517);
}

/// The same molecule under two paths takes the hetero-dimer branch (separate loads,
/// merged atom kinds, remapped ids; gh #39) yet must reproduce the homodimer exactly.
#[test]
fn hetero_branch_matches_homo_dimer() {
    let homo = workdir("char_branch_homo");
    run(&mut cppm_scan(
        &homo,
        "cppm/cppm-p18.xyz",
        &example("cppm/cppm-p18.xyz"),
    ));

    let hetero = workdir("char_branch_hetero");
    std::fs::copy(example("cppm/cppm-p18.xyz"), hetero.join("copy.xyz")).unwrap();
    run(&mut cppm_scan(
        &hetero,
        "cppm/cppm-p18.xyz",
        Path::new("copy.xyz"),
    ));

    let pmf = |d: &Path| std::fs::read_to_string(d.join("pmf.dat")).unwrap();
    assert_eq!(pmf(&homo), pmf(&hetero));
    assert_eq!(b2(&homo), b2(&hetero));
}

#[test]
fn atom_scan_b2() {
    let dir = workdir("char_atom");
    run(duello(&dir)
        .args(["atom-scan", "--atom", "PP", "-1"])
        .arg(example("cppm/cppm-p18.xyz"))
        .arg("--top")
        .arg(example("cppm/topology.yaml"))
        .args(["--rmin", "24", "--rmax", "30"])
        .args(&COARSE[..8])); // atom-scan has no --backend
    assert_rel_eq(b2(&dir), 3600.430156853588);
}

/// CIF input runs through cgkitten. Titration is stochastic, so only check that the
/// generated topology loads and the scan completes.
#[test]
fn cif_input_coarse_grains_and_scans() {
    let dir = workdir("char_cif");
    let cif = example("calvados3/2CDS.cif");
    run(duello(&dir)
        .args(["scan", "-1"])
        .arg(&cif)
        .arg("-2")
        .arg(&cif)
        .args(["--rmin", "40", "--rmax", "44"])
        .args(COARSE));
    assert!(b2(&dir).is_finite());
    let top = std::fs::read_to_string(dir.join("2CDS_topology.yaml")).unwrap();
    assert!(top.contains("- !Nonbonded"), "{top}");
}

/// `--check` loads everything but stops before scanning.
#[test]
fn check_validates_without_scanning() {
    let dir = workdir("char_check");
    run(cppm_scan(&dir, "cppm/cppm-p00.xyz", &example("cppm/cppm-p18.xyz")).arg("--check"));
    assert!(!dir.join("pmf.dat").exists());
}

/// An unknown test atom must fail `--check`, not just the real run.
#[test]
fn atom_scan_check_rejects_unknown_atom() {
    let dir = workdir("char_check_atom");
    let out = duello(&dir)
        .args(["atom-scan", "--check", "--atom", "XYZ", "-1"])
        .arg(example("cppm/cppm-p18.xyz"))
        .arg("--top")
        .arg(example("cppm/topology.yaml"))
        .args(["--rmin", "24", "--rmax", "30", "--dr", "2"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("Unknown atom type: XYZ"));
}

/// Same-named inputs from different directories would overwrite each other's XYZ.
#[test]
fn colliding_input_stems_are_rejected() {
    let dir = workdir("char_stems");
    for sub in ["a", "b"] {
        std::fs::create_dir_all(dir.join(sub)).unwrap();
        std::fs::copy(example("calvados3/2CDS.cif"), dir.join(sub).join("x.cif")).unwrap();
    }
    let out = duello(&dir)
        .args([
            "scan", "-1", "a/x.cif", "-2", "b/x.cif", "--rmin", "40", "--rmax", "44",
        ])
        .args(COARSE)
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("share the file name"));
    assert!(!dir.join("pmf.dat").exists());
}
