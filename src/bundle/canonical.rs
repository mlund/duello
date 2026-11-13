//! The entries the app reads back, and their JSON shapes.

use super::zipio::{Entry, Packing, Reader};
use crate::archive::{CgBody, FreeSaltEstimate, ScanArchive, ScanSettings, Session, StructureFile};
use crate::nbody::{RunRecord, RunSpec};
use crate::report::PmfPoint;
use crate::{Table6DAdaptive, VirialCoeff};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

pub const SESSION: &str = "session.json";
pub const SCAN: &str = "twobody/scan.json";
pub const TABLE: &str = "twobody/table.bin.gz";
pub const TOPOLOGY: &str = "twobody/topology.yaml";
pub const RUN: &str = "manybody/run.json";
pub const TRAJECTORY: &str = "manybody/trajectory.traj";

/// `Session` with each structure's contents moved to its own entry.
#[derive(Serialize, Deserialize)]
struct SessionJson {
    inputs: Option<ScanSettings>,
    structures: Vec<StructureRef>,
    scanned: Option<[usize; 2]>,
    estimate: Option<FreeSaltEstimate>,
    run: Option<RunSpec>,
}

#[derive(Serialize, Deserialize)]
struct StructureRef {
    /// As the user named it.
    name: String,
    /// Entry holding the file.
    path: String,
    /// Kept chains; empty keeps all.
    chains: Vec<String>,
}

/// The scan apart from its table and topology, which are files of their own.
#[derive(Serialize, Deserialize)]
struct ScanJson {
    settings: ScanSettings,
    bodies: [CgBody; 2],
    points: Vec<PmfPoint>,
    /// Å³
    b2: f64,
    /// Å
    sigma: f64,
    /// Steric B₂₂ʳᵉᶠ (Å³); needs the structures, so it is stored rather than rebuilt.
    b2_ref: f64,
}

pub fn session_entries(session: &Session) -> Result<Vec<Entry<'_>>> {
    let paths = structure_paths(&session.structures);
    let refs = session.structures.iter().zip(&paths);
    let json = SessionJson {
        inputs: session.inputs.clone(),
        structures: refs
            .clone()
            .map(|(s, path)| StructureRef {
                name: s.name.clone(),
                path: path.clone(),
                chains: s.chains.clone(),
            })
            .collect(),
        scanned: session.scanned,
        estimate: session.estimate,
        run: session.run.clone(),
    };
    let mut entries = vec![Entry::json(SESSION, &json, "Every tab's inputs")?];
    entries.extend(refs.map(|(s, path)| {
        Entry::new(
            path,
            s.contents.clone(),
            structure_format(path),
            "Input structure as loaded",
        )
    }));
    Ok(entries)
}

fn structure_format(path: &str) -> &'static str {
    if path.to_ascii_lowercase().ends_with(".cif") {
        "chemical/x-mmcif"
    } else {
        "chemical/x-pdb"
    }
}

fn structure_paths(structures: &[StructureFile]) -> Vec<String> {
    let names = structures.iter().map(|s| super::safe_name(&s.name));
    super::unique_names(names)
        .into_iter()
        .map(|name| format!("structures/{name}"))
        .collect()
}

pub fn read_session(zip: &mut Reader) -> Result<Session> {
    let json: SessionJson = zip.json(SESSION)?;
    let structures = json
        .structures
        .into_iter()
        .map(|r| {
            Ok(StructureFile {
                contents: zip.bytes(&r.path)?,
                name: r.name,
                chains: r.chains,
            })
        })
        .collect::<Result<_>>()?;
    let session = Session {
        inputs: json.inputs,
        structures,
        scanned: json.scanned,
        estimate: json.estimate,
        run: json.run,
    };
    session.check_consistency()?;
    Ok(session)
}

pub fn scan_entries(scan: &ScanArchive) -> Result<Vec<Entry<'static>>> {
    let json = ScanJson {
        settings: scan.settings.clone(),
        bodies: scan.bodies.clone(),
        points: scan.points.clone(),
        b2: scan.virial.b2(),
        sigma: scan.virial.sigma(),
        b2_ref: scan.virial.reference(),
    };
    Ok(vec![
        Entry::json(
            SCAN,
            &json,
            "Scan settings, coarse-grained bodies, PMF points and B₂",
        )?,
        Entry::new(
            TABLE,
            scan.table.to_bytes()?,
            "application/gzip",
            "6D energy table: gzipped bincode, read by icotable",
        )
        .packed(Packing::Stored),
        Entry::text(
            TOPOLOGY,
            scan.topology_yaml.clone(),
            "application/yaml",
            "Atom kinds and pair potentials of the scan",
        ),
    ])
}

pub fn read_scan(zip: &mut Reader) -> Result<ScanArchive> {
    let json: ScanJson = zip.json(SCAN)?;
    let scan = ScanArchive {
        table: Table6DAdaptive::from_bytes(&zip.bytes(TABLE)?)?,
        topology_yaml: zip.text(TOPOLOGY)?,
        settings: json.settings,
        bodies: json.bodies,
        points: json.points,
        virial: VirialCoeff::from_raw_parts(json.b2, json.sigma).with_reference(json.b2_ref),
    };
    check_scan(&scan)?;
    Ok(scan)
}

fn check_scan(scan: &ScanArchive) -> Result<()> {
    for body in &scan.bodies {
        body.check_consistency()?;
    }
    // The label is stored beside the table, so a writer could disagree.
    let labelled = scan.settings.subdivision;
    ensure!(
        scan.table
            .levels
            .iter()
            .all(|l| l.subdivision() == labelled),
        "settings say {labelled:?} but the table uses another angular mesh"
    );
    Ok(())
}

pub fn run_entries<'a>(run: &RunRecord, trajectory: Option<&'a [u8]>) -> Result<Vec<Entry<'a>>> {
    let mut entries = vec![Entry::json(
        RUN,
        run,
        "Many-body inputs and results: S(q), g(r), energy trace, move statistics",
    )?];
    if let Some(trajectory) = trajectory {
        entries.push(
            Entry::new(
                TRAJECTORY,
                trajectory,
                "application/octet-stream",
                "Many-body trajectory; `duello nbody-expand` turns it into XTC",
            )
            .packed(Packing::DeflatedFast),
        );
    }
    Ok(entries)
}

pub fn read_run(zip: &mut Reader) -> Result<RunRecord> {
    let run: RunRecord = zip.json(RUN)?;
    run.check_consistency()?;
    Ok(run)
}
