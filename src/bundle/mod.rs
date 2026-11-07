//! The `.duello.zip` session file: every tab's inputs, the two-body scan and
//! the many-body run, as a zip any unzip tool opens.
//!
//! JSON, YAML and structure files are read back; CSV and XYZ extracts are
//! written for readers only, so editing one can't corrupt a session.

mod canonical;
mod extracts;
mod fair;
mod manifest;
mod onebody;
mod zipio;

use crate::archive::{ScanArchive, Session};
use crate::molecule::{Curve, Molecule};
use crate::nbody::RunRecord;
use anyhow::Result;
use manifest::{Manifest, Part};
use zipio::Reader;

/// One saved session; `scan` and `run` exist only if they did when saved.
#[derive(Clone, Debug, Default)]
pub struct Bundle {
    pub session: Session,
    pub scan: Option<ScanArchive>,
    pub run: Option<RunRecord>,
}

/// A session to save, borrowed from wherever its parts live.
#[derive(Clone, Copy)]
pub struct BundleRef<'a> {
    pub session: &'a Session,
    pub scan: Option<&'a ScanArchive>,
    pub run: Option<&'a RunRecord>,
}

/// Results written for readers but never read back, because they can be
/// recomputed from the session's inputs.
#[derive(Default)]
pub struct Extracts<'a> {
    pub molecules: Vec<MoleculeExtract<'a>>,
}

/// One Onebody molecule, saved under `onebody/<name>/`.
pub struct MoleculeExtract<'a> {
    pub name: &'a str,
    /// The pH it was titrated at.
    pub ph: f64,
    pub molecule: &'a Molecule,
    pub curve: Option<&'a Curve>,
}

impl Bundle {
    pub fn to_zip(&self, extracts: &Extracts) -> Result<Vec<u8>> {
        self.as_ref().to_zip(extracts)
    }

    pub fn as_ref(&self) -> BundleRef<'_> {
        BundleRef {
            session: &self.session,
            scan: self.scan.as_ref(),
            run: self.run.as_ref(),
        }
    }

    pub fn from_zip(bytes: &[u8]) -> Result<Self> {
        manifest::refuse_old_format(bytes)?;
        let mut zip = Reader::new(bytes)?;
        let manifest = Manifest::read(&mut zip)?;
        let has = |part| manifest.parts.contains(&part);
        Ok(Self {
            session: canonical::read_session(&mut zip)?,
            scan: has(Part::Twobody)
                .then(|| canonical::read_scan(&mut zip))
                .transpose()?,
            run: has(Part::Manybody)
                .then(|| canonical::read_run(&mut zip, has(Part::Trajectory)))
                .transpose()?,
        })
    }
}

impl BundleRef<'_> {
    pub fn to_zip(&self, extracts: &Extracts) -> Result<Vec<u8>> {
        let mut entries = canonical::session_entries(self.session)?;
        if let Some(scan) = self.scan {
            entries.extend(canonical::scan_entries(scan)?);
            entries.extend(extracts::twobody(scan));
        }
        if let Some(run) = self.run {
            entries.extend(canonical::run_entries(run)?);
            entries.extend(extracts::manybody(run));
        }
        entries.extend(onebody::entries(&extracts.molecules));
        let now = manifest::now();
        let manifest = Manifest::new(self.parts(), &now);
        entries.insert(0, manifest.entry()?);
        let entries = fair::with_metadata(entries, manifest.created())?;
        zipio::write_zip(&entries, now.zip())
    }

    /// The session itself is always present, so it isn't listed.
    fn parts(&self) -> Vec<Part> {
        let trajectory = self.run.is_some_and(|r| r.trajectory.is_some());
        [
            (Part::Twobody, self.scan.is_some()),
            (Part::Manybody, self.run.is_some()),
            (Part::Trajectory, trajectory),
        ]
        .into_iter()
        .filter_map(|(part, present)| present.then_some(part))
        .collect()
    }
}

/// A file or directory name safe in any unzip tool: no separators, no `..`.
fn safe_name(name: &str) -> String {
    let name: String = name
        .chars()
        .map(|c| match c {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '.' | '-' | '_' => c,
            _ => '_',
        })
        .collect();
    let name = name.trim_start_matches('.');
    if name.is_empty() {
        "unnamed".into()
    } else {
        name.into()
    }
}

/// `names` made unique by prefixing repeats with a counter, as users may
/// load two files of the same name.
fn unique_names(names: impl Iterator<Item = String>) -> Vec<String> {
    let mut unique: Vec<String> = Vec::new();
    for name in names {
        let free = (1..)
            .map(|n| match n {
                1 => name.clone(),
                n => format!("{n}-{name}"),
            })
            // Case-insensitive file systems would merge names differing by case.
            .find(|candidate| !unique.iter().any(|u| u.eq_ignore_ascii_case(candidate)))
            .expect("unbounded");
        unique.push(free);
    }
    unique
}

#[cfg(test)]
mod tests;
