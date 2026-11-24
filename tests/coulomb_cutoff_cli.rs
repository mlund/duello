//! `--coulomb-cutoff` sets the pair's reach, which `--rmax` must cover.

mod common;

use common::{duello, example, run, workdir};
use duello::bundle::Bundle;
use std::process::Command;

/// Reach of this pair at a 15 Å cutoff is 84.9 Å.
fn scan(dir: &std::path::Path, rmax: &str) -> Command {
    let mol = example("cppm/cppm-p18.xyz");
    let mut cmd = duello(dir);
    cmd.args(["scan", "-1"])
        .arg(&mol)
        .arg("-2")
        .arg(&mol)
        .arg("--top")
        .arg(example("cppm/topology.yaml"))
        .args([
            "--rmin",
            "38",
            "--rmax",
            rmax,
            "--dr",
            "2",
            "--max-ndiv",
            "1",
        ])
        .args(["--coulomb-cutoff", "15", "--backend", "simd"]);
    cmd
}

#[test]
fn archive_records_the_coulomb_cutoff() {
    let dir = workdir("coulomb_cutoff_archive");
    run(scan(&dir, "88").args(["--savescan", "scan.duello.zip"]));
    let bytes = std::fs::read(dir.join("scan.duello.zip")).unwrap();
    let archive = Bundle::from_zip(&bytes).unwrap().scan.unwrap();
    assert_eq!(archive.settings.coulomb_cutoff, Some(15.0));
}

/// A dry run fails where the scan would.
#[test]
fn check_refuses_an_rmax_short_of_the_reach() {
    let out = scan(&workdir("coulomb_cutoff_check"), "60")
        .arg("--check")
        .output()
        .unwrap();
    assert!(!out.status.success(), "too short --rmax passed --check");
}
