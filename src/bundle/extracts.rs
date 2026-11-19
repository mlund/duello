//! Readable copies of results; derived on write, never read back.

use super::zipio::Entry;
use crate::archive::{ScanArchive, Session};
use crate::modes::export::{session_atoms, ModeExport};
use crate::modes::ModeSpec;
use crate::nbody::RunRecord;

const XYZ: &str = "chemical/x-xyz";

pub fn twobody(scan: &ScanArchive, session: &Session) -> Vec<Entry<'static>> {
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
        .chain(mode_entries(scan, session))
        .collect()
}

/// Binding modes, with all-atom models if the session kept the scanned structures.
fn mode_entries(scan: &ScanArchive, session: &Session) -> Vec<Entry<'static>> {
    ModeExport::new(scan, ModeSpec::default(), session_atoms(session, scan))
        .files()
        .into_iter()
        .map(|f| {
            Entry::text(
                &format!("twobody/modes/{}", f.name),
                f.contents,
                f.media_type,
                &f.description,
            )
        })
        .collect()
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
            run.rdf_analysis().to_csv(),
            "text/csv",
            "Mass-centre g(r), its views along r and the Kirkwood–Buff G22 and S(0) as # comments",
        ),
        Entry::text(
            "manybody/energy.csv",
            run.energy_csv(),
            "text/csv",
            "Energy per body U/N against sweeps",
        ),
    ]
}
