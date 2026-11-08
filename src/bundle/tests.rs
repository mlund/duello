use super::*;
use crate::archive::{tests::archive, FreeSaltEstimate, StructureFile};
use crate::nbody::{NBodyRun, RunSpec, Stage};
use crate::Vector3;
use icotable::lookup::Lookup6D;

/// Two loaded structures, both scanned, a free-salt estimate and the
/// many-body inputs.
fn session() -> Session {
    let file = |name: &str, contents: &[u8], chains: &[&str]| StructureFile {
        name: name.into(),
        contents: contents.to_vec(),
        chains: chains.iter().map(|c| c.to_string()).collect(),
    };
    let mut run = RunSpec::new(200, 0.002);
    run.record_trajectory = true;
    run.msa.debye_length = Some(12.0);
    let mut inputs = archive().settings;
    inputs.ph = 4.5;
    Session {
        inputs: Some(inputs),
        structures: vec![
            file("4lzt.pdb", b"ATOM 1", &["A", "B"]),
            file("1ubq.cif", b"data_", &[]),
        ],
        scanned: Some([1, 0]),
        estimate: Some(FreeSaltEstimate {
            reservoir: 0.02,
            protein_molarity: 0.002,
        }),
        run: Some(run),
    }
}

/// A short real run into production and its trajectory, so the record
/// holds what a saved run would.
fn run_record() -> (RunRecord, Vec<u8>) {
    use crate::nbody::run_tests::{archive, run_to, spec, UNIT};
    let mut run = NBodyRun::start(&archive(true, &["X"]), spec(), UNIT).unwrap();
    run_to(&mut run, |s| matches!(s, Stage::Production { .. }));
    run.step(10).unwrap();
    (run.results(), run.trajectory_bytes().unwrap().into_owned())
}

fn full() -> Bundle {
    let (run, trajectory) = run_record();
    Bundle {
        session: session(),
        scan: Some(steric_archive()),
        run: Some(run),
        trajectory: Some(trajectory),
    }
}

/// A reference unlike σ's hard sphere, which loading could rebuild without storing it.
fn steric_archive() -> ScanArchive {
    let mut scan = archive();
    scan.virial = scan.virial.clone().with_reference(1234.5);
    scan
}

fn round_trip(bundle: &Bundle) -> Bundle {
    Bundle::from_zip(&bundle.to_zip(&Extracts::default()).unwrap()).unwrap()
}

fn assert_same_scan(restored: &ScanArchive, original: &ScanArchive) {
    assert_eq!(restored.settings, original.settings);
    assert_eq!(restored.bodies, original.bodies);
    assert_eq!(restored.topology_yaml, original.topology_yaml);
    assert_eq!(restored.virial.b2(), original.virial.b2());
    assert_eq!(restored.virial.sigma(), original.virial.sigma());
    assert_eq!(restored.virial.reference(), original.virial.reference());
    let point =
        |p: &crate::report::PmfPoint| (p.r, p.free_energy, p.mean_energy, p.energy_variance);
    let points = |a: &ScanArchive| a.points.iter().map(point).collect::<Vec<_>>();
    assert_eq!(points(restored), points(original));
    let (a, b) = (Vector3::new(0.6, 0.0, 0.8), Vector3::new(0.0, -1.0, 0.0));
    assert_eq!(
        restored.table.lookup(6.2, 0.4, &a, &b, None),
        original.table.lookup(6.2, 0.4, &a, &b, None)
    );
    let metadata = |a: &ScanArchive| format!("{:?}", a.table.metadata);
    assert_eq!(metadata(restored), metadata(original));
}

#[test]
fn a_session_with_a_scan_and_a_run_round_trips() {
    let original = full();
    let restored = round_trip(&original);

    assert_eq!(restored.session, original.session);
    assert_same_scan(
        restored.scan.as_ref().unwrap(),
        original.scan.as_ref().unwrap(),
    );
    let run = original.run.as_ref().unwrap();
    assert!(original.trajectory.as_ref().is_some_and(|t| t.len() > 100));
    assert_eq!(restored.trajectory, original.trajectory);
    assert!(
        run.curve.frames > 0,
        "the fixture should have production frames"
    );
    assert_eq!(restored.run.as_ref(), Some(run));
}

fn entry_names(zip: &[u8]) -> Vec<String> {
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(zip)).unwrap();
    (0..archive.len())
        .map(|i| archive.by_index_raw(i).unwrap().name().unwrap().to_string())
        .collect()
}

#[test]
fn a_session_alone_round_trips_without_scan_or_run_entries() {
    let original = Bundle {
        session: session(),
        ..Default::default()
    };
    let zip = original.to_zip(&Extracts::default()).unwrap();
    let restored = Bundle::from_zip(&zip).unwrap();

    assert_eq!(restored.session, session());
    assert!(restored.scan.is_none() && restored.run.is_none());
    let names = entry_names(&zip);
    assert!(
        names.contains(&"structures/4lzt.pdb".to_string()),
        "{names:?}"
    );
    let stray = names
        .iter()
        .find(|n| n.starts_with("twobody/") || n.starts_with("manybody/"));
    assert_eq!(stray, None);
}

/// Copies `zip`, letting `edit` rename or rewrite `path`.
fn with_entry_edited(zip: &[u8], path: &str, edit: impl Fn(&mut String, &mut Vec<u8>)) -> Vec<u8> {
    use std::io::{Read, Write};
    let mut source = zip::ZipArchive::new(std::io::Cursor::new(zip)).unwrap();
    let mut out = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    let mut found = false;
    for i in 0..source.len() {
        let mut file = source.by_index(i).unwrap();
        let mut name = file.name().unwrap().to_string();
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).unwrap();
        if name == path {
            found = true;
            edit(&mut name, &mut bytes);
        }
        out.start_file(name, zip::write::SimpleFileOptions::default())
            .unwrap();
        out.write_all(&bytes).unwrap();
    }
    assert!(found, "{path} is not in the zip");
    out.finish().unwrap().into_inner()
}

fn with_entry_replaced(zip: &[u8], path: &str, contents: &[u8]) -> Vec<u8> {
    with_entry_edited(zip, path, |_, bytes| *bytes = contents.to_vec())
}

fn with_entry_renamed(zip: &[u8], path: &str, new: &str) -> Vec<u8> {
    with_entry_edited(zip, path, |name, _| *name = new.to_string())
}

#[test]
fn an_edited_pmf_extract_does_not_change_the_loaded_scan() {
    let bundle = Bundle {
        scan: Some(archive()),
        ..Default::default()
    };
    let zip = bundle.to_zip(&Extracts::default()).unwrap();
    let edited = with_entry_replaced(&zip, "twobody/pmf.csv", b"R,F\n1,999\n");
    let scan = Bundle::from_zip(&edited).unwrap().scan.unwrap();

    let points: Vec<_> = scan.points.iter().map(|p| (p.r, p.free_energy)).collect();
    assert_eq!(points, [(5.0, 1.5), (6.0, -0.25)]);
}

fn load_error(bytes: &[u8]) -> String {
    format!("{:#}", Bundle::from_zip(bytes).unwrap_err())
}

fn session_zip() -> Vec<u8> {
    let bundle = Bundle {
        session: session(),
        ..Default::default()
    };
    bundle.to_zip(&Extracts::default()).unwrap()
}

#[test]
fn an_old_format_file_is_refused_and_told_to_rescan() {
    let mut old = b"DUELLO-SCAN\0\x04\0\0\0".to_vec();
    old.extend([0; 64]);
    let error = load_error(&old);
    assert!(error.contains("re-run the scan"), "{error}");
}

#[test]
fn a_version_5_file_without_a_steric_reference_is_refused() {
    let manifest = r#"{"format":"duello-session","version":5,"parts":[]}"#;
    let error = load_error(&with_entry_replaced(
        &session_zip(),
        "manifest.json",
        manifest.as_bytes(),
    ));
    assert!(error.contains("re-run the scan"), "{error}");
}

#[test]
fn a_newer_version_is_refused() {
    let manifest = r#"{"format":"duello-session","version":99,"parts":[]}"#;
    let error = load_error(&with_entry_replaced(
        &session_zip(),
        "manifest.json",
        manifest.as_bytes(),
    ));
    assert!(error.contains("newer"), "{error}");
}

#[test]
fn a_missing_session_entry_is_named() {
    let renamed = with_entry_renamed(&session_zip(), "session.json", "other.json");
    let error = load_error(&renamed);
    assert!(error.contains("session.json"), "{error}");
}

/// Opened files are untrusted: restoring indexes structures by `scanned`,
/// per-bead code zips the body vectors, and the mesh label sits beside the table.
#[test]
fn inconsistent_sessions_and_scans_are_refused() {
    let save = |bundle: Bundle| bundle.to_zip(&Extracts::default()).unwrap();
    let beyond = Session {
        scanned: Some([2, 0]),
        ..session()
    };
    let error = load_error(&save(Bundle {
        session: beyond,
        ..Default::default()
    }));
    assert!(error.contains("scanned"), "{error}");

    let mut short = archive();
    short.bodies[1].kinds.pop();
    let mut mislabelled = archive();
    mislabelled.settings.subdivision = crate::Subdivision::Lattice;
    for (scan, why) in [(short, "bead"), (mislabelled, "mesh")] {
        let error = load_error(&save(Bundle {
            scan: Some(scan),
            ..Default::default()
        }));
        assert!(error.contains(why), "{error}");
    }
}

fn compression_of(zip: &[u8], path: &str) -> zip::CompressionMethod {
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(zip)).unwrap();
    let method = archive.by_name(path).unwrap().compression();
    method
}

#[test]
fn the_gzipped_table_is_stored_and_text_is_deflated() {
    let zip = full().to_zip(&Extracts::default()).unwrap();
    use zip::CompressionMethod::{Deflated, Stored};
    assert_eq!(compression_of(&zip, "twobody/table.bin.gz"), Stored);
    for text in [
        "session.json",
        "twobody/scan.json",
        "twobody/pmf.csv",
        "manybody/run.json",
    ] {
        assert_eq!(compression_of(&zip, text), Deflated, "{text}");
    }
}

fn entry_text(zip: &[u8], path: &str) -> String {
    let mut reader = super::zipio::Reader::new(zip).unwrap();
    reader.text(path).unwrap()
}

/// Bead positions of an XYZ file, in file order.
fn xyz_positions(xyz: &str) -> Vec<Vector3> {
    let count: usize = xyz.lines().next().unwrap().trim().parse().unwrap();
    let beads: Vec<Vector3> = xyz
        .lines()
        .skip(2)
        .map(|line| {
            let xyz: Vec<f64> = line
                .split_whitespace()
                .skip(1)
                .map(|v| v.parse().unwrap())
                .collect();
            Vector3::new(xyz[0], xyz[1], xyz[2])
        })
        .collect();
    assert_eq!(beads.len(), count);
    beads
}

#[test]
fn the_lowest_pose_is_written_first_with_both_bodies_r_apart() {
    use crate::poses::tests::{planted_archive, ISOTROPIC};
    let bundle = Bundle {
        scan: Some(planted_archive()),
        ..Default::default()
    };
    let zip = bundle.to_zip(&Extracts::default()).unwrap();

    let csv = entry_text(&zip, "twobody/poses/poses.csv");
    let first: Vec<&str> = csv.lines().nth(1).unwrap().split(',').collect();
    assert_eq!(first[0], "1");
    let kt = 2.4789570296023884; // kJ/mol at 298.15 K
    let energy: f64 = first[1].parse().unwrap();
    approx::assert_relative_eq!(energy, ISOTROPIC / kt, max_relative = 1e-4);
    let r: f64 = first[2].parse().unwrap();
    assert_eq!(r, 5.0); // the isotropic slab's R-bin

    // Fixture bodies: two beads each, of equal mass.
    let beads = xyz_positions(&entry_text(&zip, "twobody/poses/pose01.xyz"));
    assert_eq!(beads.len(), 4);
    let centre = |b: &[Vector3]| (b[0] + b[1]) / 2.0;
    approx::assert_relative_eq!(
        (centre(&beads[2..]) - centre(&beads[..2])).norm(),
        5.0,
        epsilon = 1e-3
    );
}

/// Two beads with round multipole moments, so the CSV values are literals.
fn two_bead_molecule() -> crate::molecule::Molecule {
    use crate::molecule::{BeadRow, Molecule};
    let row = |kind: &str, charge| BeadRow {
        chain: "A".into(),
        residue: 1,
        kind: kind.into(),
        charge,
        charge_variance: Some(0.25),
        mass: 110.0,
        sigma: Some(6.0),
        epsilon: None,
        lambda: Some(0.5),
    };
    Molecule {
        body: crate::archive::tests::body("LYS"),
        beads: vec![row("LYS", 1.0), row("GLU", -0.5)],
        multipole: cgkitten::Multipole {
            charge: 0.5,
            charge_sq: 1.25,
            dipole: 2.0,
            dipole_sq: 5.0,
            quadrupole: 3.0,
            quadrupole_sq: 10.0,
        },
    }
}

/// `quantity`'s value in a quantity,value,unit CSV.
fn quantity(csv: &str, quantity: &str) -> f64 {
    let row = csv
        .lines()
        .find(|line| line.starts_with(quantity))
        .unwrap_or_else(|| panic!("no {quantity} in\n{csv}"));
    row.split(',').nth(1).unwrap().parse().unwrap()
}

#[test]
fn onebody_molecules_get_beads_multipoles_and_a_curve_only_if_given() {
    use crate::molecule::{Curve, MoleculeSpec};
    let cif = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/calvados3/2CDS.cif"
    ))
    .unwrap();
    let mut curve = Curve::start(&cif, &MoleculeSpec::default(), &[], &[6.0, 7.0], 1).unwrap();
    curve.advance_ideal();
    let molecule = two_bead_molecule();
    let extracts = Extracts {
        molecules: vec![
            MoleculeExtract {
                name: "a.pdb",
                ph: 7.0,
                molecule: &molecule,
                curve: None,
            },
            MoleculeExtract {
                name: "b.pdb",
                ph: 7.0,
                molecule: &molecule,
                curve: Some(&curve),
            },
        ],
    };
    let zip = Bundle::default().to_zip(&extracts).unwrap();

    let multipoles = entry_text(&zip, "onebody/a/multipoles.csv");
    assert_eq!(quantity(&multipoles, "pH"), 7.0);
    assert_eq!(quantity(&multipoles, "net charge"), 0.5);
    assert_eq!(quantity(&multipoles, "capacitance"), 1.0);
    assert_eq!(quantity(&multipoles, "dipole"), 2.0);
    assert_eq!(quantity(&multipoles, "quadrupole"), 3.0);
    let beads = entry_text(&zip, "onebody/a/beads.csv");
    assert_eq!(beads.lines().count(), 1 + 2, "{beads}");
    assert!(beads.lines().nth(2).unwrap().contains("GLU"), "{beads}");

    let names = entry_names(&zip);
    assert!(!names.contains(&"onebody/a/titration.csv".to_string()));
    let titration = entry_text(&zip, "onebody/b/titration.csv");
    assert_eq!(titration.lines().count(), 1 + 2, "{titration}");
}

#[test]
fn the_ro_crate_describes_every_entry_in_the_zip() {
    let molecule = two_bead_molecule();
    let extracts = Extracts {
        molecules: vec![MoleculeExtract {
            name: "a.pdb",
            ph: 7.0,
            molecule: &molecule,
            curve: None,
        }],
    };
    let zip = full().to_zip(&extracts).unwrap();
    let crate_json: serde_json::Value =
        serde_json::from_str(&entry_text(&zip, "ro-crate-metadata.json")).unwrap();

    let graph = crate_json["@graph"].as_array().unwrap();
    let described = |name: &str| {
        graph.iter().any(|e| {
            e["@id"] == name && e["name"].is_string() && e["encodingFormat"].is_string()
                || e["@id"] == name && name == "ro-crate-metadata.json"
        })
    };
    let names = entry_names(&zip);
    assert!(names.contains(&"README.md".to_string()), "{names:?}");
    for name in &names {
        assert!(described(name), "{name} lacks an RO-Crate entity");
    }
    let files = graph.iter().filter(|e| e["@type"] == "File").count();
    assert_eq!(
        files + 1,
        names.len(),
        "one File per entry besides the descriptor"
    );
}

/// Salt-free runs have an unscreened sphere, which JSON can't write as a number.
#[test]
fn an_infinite_debye_length_round_trips() {
    let (mut run, _) = run_record();
    run.sphere.debye_length = f64::INFINITY;
    let bundle = Bundle {
        run: Some(run.clone()),
        ..Default::default()
    };
    assert_eq!(round_trip(&bundle).run, Some(run));
}

/// Opened files are untrusted; curve code indexes these vectors together.
#[test]
fn a_run_with_mismatched_curve_lengths_is_refused() {
    let bundle = Bundle {
        run: Some(run_record().0),
        ..Default::default()
    };
    let zip = bundle.to_zip(&Extracts::default()).unwrap();
    let short_g = with_entry_edited(&zip, "manybody/run.json", |_, bytes| {
        let mut json: serde_json::Value = serde_json::from_slice(bytes).unwrap();
        json["rdf"]["g"].as_array_mut().unwrap().pop();
        *bytes = serde_json::to_vec(&json).unwrap();
    });
    let error = load_error(&short_g);
    assert!(error.contains("g(r)"), "{error}");
}
