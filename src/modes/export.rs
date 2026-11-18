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
        Self {
            scan,
            modes: binding_modes(scan, spec),
            atoms,
        }
    }

    /// One row per mode, strongest first.
    pub fn csv(&self) -> String {
        let header = "rank,share/%,b2/Å³,n_poses,U/kT,R/Å,ω/°,ix,iy,iz,jx,jy,jz";
        let rows = (1..).zip(&self.modes.modes).map(|(rank, mode)| {
            let [i, j] = mode.pose.directions();
            let directions = i.iter().chain(j.iter()).map(|x| format!("{x:.3}"));
            let fields = [
                rank.to_string(),
                format!("{:.2}", 100.0 * mode.share),
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
        let table = ModeFile {
            name: TABLE.into(),
            contents: self.csv(),
            media_type: "text/csv",
            description: "Binding modes: poses grouped by similarity, ranked by B₂ contribution"
                .into(),
        };
        let cg = (0..self.modes.modes.len()).map(|i| ModeFile {
            name: format!("{CG_PREFIX}{:02}{CG_SUFFIX}", i + 1),
            contents: self.cg_xyz(i),
            media_type: "chemical/x-xyz",
            description: format!("Mode {} of modes.csv, both bodies' beads (Å)", i + 1),
        });
        let allatom = self.allatom_pdb().map(|contents| ModeFile {
            name: ALLATOM.into(),
            contents,
            media_type: "chemical/x-pdb",
            description: "All modes of modes.csv as models, all atoms, chains A and B".into(),
        });
        std::iter::once(table).chain(cg).chain(allatom).collect()
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
