//! A CLI scan saved with `--savescan` must load as a self-contained archive.

mod common;

use common::{duello, example, run, workdir};
use duello::archive::ScanArchive;

#[test]
fn cli_scan_archive_loads_with_its_results() {
    let dir = workdir("archive_cli");
    let mol = example("cppm/cppm-p18.xyz");
    run(duello(&dir)
        .args(["scan", "-1"])
        .arg(&mol)
        .arg("-2")
        .arg(&mol)
        .arg("--top")
        .arg(example("cppm/topology.yaml"))
        .args([
            "--rmin",
            "38",
            "--rmax",
            "44",
            "--dr",
            "2",
            "--max-ndiv",
            "1",
        ])
        .args([
            "--cutoff",
            "2000",
            "--molarity",
            "0.05",
            "--backend",
            "reference",
        ])
        .args(["--savescan", "scan.duello"]));

    let bytes = std::fs::read(dir.join("scan.duello")).unwrap();
    let archive = ScanArchive::from_bytes(&bytes).unwrap();

    // Same golden B2 as characterization.rs::scan_homo_dimer_b2.
    let expected = -240356.77888141517;
    assert!(((archive.virial.b2() - expected) / expected).abs() < 1e-9);
    assert!(archive.settings.homo_dimer);
    assert_eq!(archive.settings.molarity, 0.05);
    assert_eq!(archive.settings.model, "custom topology");
    assert_eq!(archive.settings.cutoff, 2000.0); // reference backend: full cutoff
    assert!(!archive.bodies[0].kinds.is_empty());
    let meta = archive.table.metadata.as_ref().unwrap();
    assert_eq!(meta.temperature, Some(298.15));
    // With --top there is no generated CG topology; the given one must be kept.
    assert!(
        archive.topology_yaml.contains("name: PP"),
        "{}",
        archive.topology_yaml
    );
}
