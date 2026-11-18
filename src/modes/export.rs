//! Binding modes as files, shared by the CLI, session extracts and duello-web
//! so that each saves the same bytes.

use super::{binding_modes, ModeSpec, Modes};
use crate::allatom::{pdb, AllAtom};
use crate::archive::{xyz_with_comment, ScanArchive, Session};
use crate::poses::TablePose;
use anyhow::{Context, Result};

/// Binding modes ready to save, with all-atom models when the structures are known.
pub struct ModeExport<'a> {
    scan: &'a ScanArchive,
    pub modes: Modes,
    atoms: Option<[AllAtom; 2]>,
}

/// One file of a [`ModeExport`].
pub struct ModeFile {
    pub name: String,
    pub contents: String,
    pub media_type: &'static str,
    pub description: String,
}

impl<'a> ModeExport<'a> {
    pub fn new(scan: &'a ScanArchive, spec: ModeSpec, atoms: Option<[AllAtom; 2]>) -> Self {
        Self::with_modes(scan, binding_modes(scan, spec), atoms)
    }

    /// Reuses a ranking, as ranking walks the whole table. `modes` must come
    /// from `scan`.
    pub fn with_modes(scan: &'a ScanArchive, modes: Modes, atoms: Option<[AllAtom; 2]>) -> Self {
        Self { scan, modes, atoms }
    }

    /// One row per mode, strongest first.
    pub fn csv(&self) -> String {
        let header = "rank,share/%,share_Ka/%,b2/Å³,n_poses,U/kT,R/Å,ω/°,ix,iy,iz,jx,jy,jz";
        let rows = (1..).zip(&self.modes.modes).map(|(rank, mode)| {
            let [i, j] = mode.pose.directions();
            let directions = i.iter().chain(j.iter()).map(|x| format!("{x:.3}"));
            let fields = [
                rank.to_string(),
                format!("{:.2}", 100.0 * mode.share),
                mode.association_share
                    .map_or(String::new(), |s| format!("{:.2}", 100.0 * s)),
                format!("{:.4e}", mode.b2),
                mode.n_poses.to_string(),
            ];
            let fields = fields
                .into_iter()
                .chain(pose_columns(&mode.pose))
                .chain(directions);
            fields.collect::<Vec<_>>().join(",")
        });
        std::iter::once(header.to_string())
            .chain(rows)
            .map(|line| line + "\n")
            .collect()
    }

    /// Both bodies' beads (Å) of the mode at `index` in [`Modes::modes`].
    pub fn cg_xyz(&self, index: usize) -> String {
        let mode = &self.modes.modes[index];
        let placed = mode.pose.place(&self.scan.bodies);
        let beads = self.scan.bodies.iter().zip(&placed);
        let atoms = beads.flat_map(|(body, beads)| body.kinds.iter().zip(beads.iter().copied()));
        let [u, r, omega] = pose_columns(&mode.pose);
        let comment = format!(
            "Mode {}: {:.1}% of the B₂ attraction, U/kT = {u}, R/Å = {r}, ω/° = {omega}",
            index + 1,
            100.0 * mode.share
        );
        xyz_with_comment(&comment, placed[0].len() + placed[1].len(), atoms)
    }

    /// Every mode as one PDB model, if the input structures are known.
    pub fn allatom_pdb(&self) -> Option<String> {
        let [a, b] = self.atoms.as_ref()?;
        let a = a.place(&TablePose::a_pose());
        let models: Vec<_> = self
            .modes
            .modes
            .iter()
            .map(|mode| [a.clone(), b.place(&mode.pose.b_pose())])
            .collect();
        Some(pdb(&models))
    }

    /// The table, one CG file per mode and, if possible, the all-atom models.
    pub fn files(&self) -> Vec<ModeFile> {
        let cg = (0..self.modes.modes.len()).map(|i| self.cg_entry(i));
        std::iter::once(self.csv_file())
            .chain(cg)
            .chain(self.allatom_file())
            .collect()
    }

    pub fn csv_file(&self) -> ModeFile {
        ModeFile {
            name: TABLE.into(),
            contents: self.csv(),
            media_type: "text/csv",
            description: "Binding modes: poses grouped by similarity, ranked by B₂ contribution"
                .into(),
        }
    }

    /// Mode `index`'s beads; None past the last mode.
    pub fn cg_file(&self, index: usize) -> Option<ModeFile> {
        (index < self.modes.modes.len()).then(|| self.cg_entry(index))
    }

    fn cg_entry(&self, index: usize) -> ModeFile {
        ModeFile {
            name: format!("{CG_PREFIX}{:02}{CG_SUFFIX}", index + 1),
            contents: self.cg_xyz(index),
            media_type: "chemical/x-xyz",
            description: format!("Mode {} of modes.csv, both bodies' beads (Å)", index + 1),
        }
    }

    /// None without the input structures.
    pub fn allatom_file(&self) -> Option<ModeFile> {
        self.allatom_pdb().map(|contents| ModeFile {
            name: ALLATOM.into(),
            contents,
            media_type: "chemical/x-pdb",
            description: "All modes of modes.csv as models, all atoms, chains A and B".into(),
        })
    }
}

const TABLE: &str = "modes.csv";
const ALLATOM: &str = "modes_allatom.pdb";
const CG_PREFIX: &str = "mode";
const CG_SUFFIX: &str = "_cg.xyz";

/// Whether [`ModeExport::files`] could have written `name`, so that an earlier
/// export can be cleared without touching anything else.
pub fn is_mode_file(name: &str) -> bool {
    let numbered = name
        .strip_prefix(CG_PREFIX)
        .and_then(|n| n.strip_suffix(CG_SUFFIX))
        .is_some_and(|n| n.len() >= 2 && n.bytes().all(|b| b.is_ascii_digit()));
    numbered || name == TABLE || name == ALLATOM
}

fn pose_columns(pose: &TablePose) -> [String; 3] {
    [
        format!("{:.4}", pose.energy),
        format!("{:.2}", pose.r),
        format!("{:.1}", pose.omega.to_degrees()),
    ]
}

/// All-atom versions of the scanned bodies from `inputs`: (file contents, kept chains).
pub fn atom_pair(inputs: [(&[u8], &[String]); 2], scan: &ScanArchive) -> Result<[AllAtom; 2]> {
    let single_bead = scan.settings.single_bead;
    let atoms = |i: usize| {
        let (bytes, chains) = inputs[i];
        AllAtom::new(bytes, chains, single_bead, &scan.bodies[i])
            .with_context(|| format!("molecule {}", i + 1))
    };
    let a = atoms(0)?;
    // A homodimer's second input is usually the first again; parse it once.
    let same = inputs[0] == inputs[1] && scan.bodies[0].positions == scan.bodies[1].positions;
    let b = if same { a.clone() } else { atoms(1)? };
    Ok([a, b])
}

/// The scanned structures a session kept, as all-atom bodies. None if it kept
/// none, or if they no longer fit the scan (e.g. XYZ), which only loses the
/// all-atom models and so is logged rather than failed on.
pub fn session_atoms(session: &Session, scan: &ScanArchive) -> Option<[AllAtom; 2]> {
    let [Some(a), Some(b)] = session.scanned?.map(|i| session.structures.get(i)) else {
        return None;
    };
    let inputs = [a, b].map(|s| (s.contents.as_slice(), s.chains.as_slice()));
    atom_pair(inputs, scan)
        .inspect_err(|e| log::warn!("No all-atom binding modes: {e:#}"))
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modes::ModeSpec;
    use crate::poses::tests::planted_archive;

    /// A front end that already ranked the modes saves the same bytes without
    /// ranking again.
    #[test]
    fn an_existing_ranking_exports_the_same_files() {
        let scan = planted_archive();
        let fresh = ModeExport::new(&scan, ModeSpec::default(), None);
        assert!(!fresh.modes.modes.is_empty(), "the fixture should attract");
        let reused = ModeExport::with_modes(&scan, fresh.modes.clone(), None);
        assert_eq!(reused.csv(), fresh.csv());
        assert_eq!(reused.cg_xyz(0), fresh.cg_xyz(0));
    }

    /// Each file by what it is, so callers needn't know `files()`'s order.
    #[test]
    fn each_file_is_named_by_what_it_holds() {
        let scan = planted_archive();
        let export = ModeExport::new(&scan, ModeSpec::default(), None);
        assert_eq!(export.csv_file().name, "modes.csv");
        assert_eq!(
            export.cg_file(0).map(|f| f.name),
            Some("mode01_cg.xyz".into())
        );
        assert!(export.cg_file(export.modes.modes.len()).is_none());
        assert!(export.allatom_file().is_none(), "no structures given");
        let names: Vec<String> = export.files().into_iter().map(|f| f.name).collect();
        assert_eq!(names.first().map(String::as_str), Some("modes.csv"));
        assert_eq!(names.len(), 1 + export.modes.modes.len());
    }

    /// The K_a share sits beside the attraction share, empty without association.
    #[test]
    fn the_csv_has_a_ka_share_column() {
        let mut scan = planted_archive();
        let b2 = crate::modes::table_b2(&scan);
        scan.virial = scan.virial.clone().with_reference(b2 - 1.0);
        let csv = ModeExport::new(&scan, ModeSpec::default(), None).csv();
        let mut lines = csv.lines();
        let header: Vec<&str> = lines.next().unwrap().split(',').collect();
        let i = header
            .iter()
            .position(|h| *h == "share_Ka/%")
            .expect("column");
        assert_eq!(lines.next().unwrap().split(',').nth(i), Some(""));
    }
}
