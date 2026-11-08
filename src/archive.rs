//! Self-contained two-body scan.
//!
//! Holds everything needed to view a scan or run a many-body simulation from
//! it without the original PDB: the 6D table, the coarse-grained bodies, and
//! the derived PMF and virial results.

use crate::icoscan::{table_metadata, ScanParams, ScanResult};
use crate::nbody::RunSpec;
use crate::report::{compute_pmf, PmfPoint};
use crate::structure::Structure;
use crate::{Subdivision, Table6DAdaptive, VirialCoeff};
use anyhow::{ensure, Result};
use faunus::topology::AtomKind;
use serde::{Deserialize, Serialize};

/// The scan settings shared by the web tool and the CLI, recorded with a scan.
///
/// Not exhaustive. CLI-only inputs that also shape the table are not recorded:
/// fixed dielectric, pKa overrides, hydrophobic scaling, chain selection,
/// locator, and spline grid type/shift/energy cap. A `--top` file is kept in
/// [`ScanArchive::topology_yaml`]; it replaces coarse-graining, so `model` is
/// then `"custom topology"` and `ph` has no meaning. `cutoff` is the
/// short-range spline cutoff actually used.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScanSettings {
    pub ph: f64,
    pub model: String,
    pub single_bead: bool,
    pub rmin: f64,
    pub rmax: f64,
    pub dr: f64,
    #[serde(alias = "molarity")]
    pub ionic_strength: f64,
    pub temperature: f64,
    pub max_ndiv: usize,
    pub gradient_threshold: f64,
    pub cutoff: f64,
    pub grid_size: usize,
    pub homo_dimer: bool,
    /// Angular mesh scheme of the table. Defaulted so trajectory headers
    /// (JSON) written before it existed still read.
    #[serde(default)]
    pub subdivision: Subdivision,
}

/// Coarse-grained rigid body in its own frame (mass center at origin).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CgBody {
    pub positions: Vec<[f64; 3]>,
    pub masses: Vec<f64>,
    pub charges: Vec<f64>,
    pub radii: Vec<f64>,
    /// Atom kind name per bead, e.g. residue names for single-bead models.
    pub kinds: Vec<String>,
}

/// A finished two-body scan; saved and loaded via [`crate::bundle`].
#[derive(Clone, Debug)]
pub struct ScanArchive {
    pub settings: ScanSettings,
    pub table: Table6DAdaptive<f32>,
    pub bodies: [CgBody; 2],
    pub points: Vec<PmfPoint>,
    pub virial: VirialCoeff,
    pub topology_yaml: String,
}

/// The web app's inputs, so opening a session restores the user's work;
/// inputs only, never results. Empty when the CLI saved it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Session {
    /// What the tabs showed when saved; None from the CLI, whose scan
    /// settings then stand in.
    pub inputs: Option<ScanSettings>,
    /// Input structures as loaded, so Onebody can be restored.
    pub structures: Vec<StructureFile>,
    /// Indices into `structures` of the scanned pair; None if made elsewhere.
    pub scanned: Option<[usize; 2]>,
    /// What a free-salt estimate assumed; None if the ionic strength was entered.
    pub estimate: Option<FreeSaltEstimate>,
    /// Many-body inputs last used with this scan.
    pub run: Option<RunSpec>,
}

/// A structure file as loaded, e.g. a PDB or mmCIF.
#[derive(Clone, Debug, PartialEq)]
pub struct StructureFile {
    pub name: String,
    pub contents: Vec<u8>,
}

/// Inputs to [`crate::solution::free_salt`]; the charge and volume come from
/// the scanned body.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FreeSaltEstimate {
    /// mol/L
    pub reservoir: f64,
    /// mol/L
    pub protein_molarity: f64,
}

impl ScanArchive {
    /// Packages a finished scan, attaching the table metadata (tail fit and
    /// temperature) that consumers need beyond `rmax`, unless already fitted.
    pub fn from_scan(
        settings: ScanSettings,
        params: &ScanParams,
        mut scan: ScanResult,
        bodies: [&Structure; 2],
        kinds: &[AtomKind],
        topology_yaml: String,
    ) -> Result<Self> {
        let pmf = compute_pmf(&scan.samples)?;
        if scan.table.metadata.is_none() {
            scan.table.metadata = Some(table_metadata(params, &scan));
        }
        let [a, b] = bodies;
        Ok(Self {
            settings,
            table: scan.table,
            bodies: [CgBody::new(a, kinds), CgBody::new(b, kinds)],
            points: pmf.points,
            virial: pmf.virial,
            topology_yaml,
        })
    }

    /// Molecular weights in g/mol.
    pub fn molar_masses(&self) -> (f64, f64) {
        (self.bodies[0].mass(), self.bodies[1].mass())
    }

    /// Net charges in elementary charges.
    pub fn net_charges(&self) -> (f64, f64) {
        (self.bodies[0].net_charge(), self.bodies[1].net_charge())
    }
}

/// Mass, charge and XYZ are derived rather than stored, so they always describe
/// the beads a simulation will use.
impl CgBody {
    pub(crate) fn new(structure: &Structure, kinds: &[AtomKind]) -> Self {
        Self {
            positions: structure.pos.iter().map(|p| [p.x, p.y, p.z]).collect(),
            masses: structure.masses.clone(),
            charges: structure.charges.clone(),
            radii: structure.radii.clone(),
            kinds: structure
                .atom_ids
                .iter()
                .map(|&id| kinds[id].name().to_string())
                .collect(),
        }
    }

    pub fn mass(&self) -> f64 {
        self.masses.iter().sum()
    }

    pub fn net_charge(&self) -> f64 {
        self.charges.iter().sum()
    }

    /// Same layout as [`Structure::to_xyz`].
    pub fn to_xyz(&self) -> String {
        xyz_string(
            self.positions.len(),
            self.kinds.iter().zip(self.positions.iter().copied()),
        )
    }
}

/// XYZ text (Å), shared by body and many-body trajectory topologies.
pub(crate) fn xyz_string<'a>(
    n_atoms: usize,
    atoms: impl Iterator<Item = (&'a String, [f64; 3])>,
) -> String {
    xyz_with_comment("Generated by Duello", n_atoms, atoms)
}

/// As [`xyz_string`], with `comment` on the second line.
pub(crate) fn xyz_with_comment<'a>(
    comment: &str,
    n_atoms: usize,
    atoms: impl Iterator<Item = (&'a String, [f64; 3])>,
) -> String {
    let mut xyz = format!("{n_atoms}\n{comment}\n");
    for (kind, [x, y, z]) in atoms {
        xyz += &format!("{kind} {x:.3} {y:.3} {z:.3}\n");
    }
    xyz
}

impl Session {
    /// Restoring indexes `structures` by `scanned`.
    pub(crate) fn check_consistency(&self) -> Result<()> {
        let n = self.structures.len();
        ensure!(
            self.scanned.is_none_or(|pair| pair.iter().all(|&i| i < n)),
            "session's scanned pair {:?} is beyond its {n} structures",
            self.scanned
        );
        Ok(())
    }
}

impl CgBody {
    /// Per-bead code zips these vectors, so a short one would silently drop beads.
    pub(crate) fn check_consistency(&self) -> Result<()> {
        let n = self.positions.len();
        let lengths = [
            self.masses.len(),
            self.charges.len(),
            self.radii.len(),
            self.kinds.len(),
        ];
        ensure!(
            lengths.iter().all(|&len| len == n),
            "body has inconsistent per-bead data ({n} positions vs {lengths:?})"
        );
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::AdaptiveBuilder;

    /// Energy varies with both directions, so a broken table shows up in lookups.
    fn table() -> Table6DAdaptive<f32> {
        let mut builder =
            AdaptiveBuilder::new(5.0, 7.0, 1.0, std::f64::consts::TAU / 4.0, 1, 1e-9, 1.0);
        for ri in 0..builder.n_r() {
            let verts = builder.vertex_directions(builder.current_level()).to_vec();
            let data: Vec<f64> = verts
                .iter()
                .flat_map(|a| verts.iter().map(move |b| a[0] - 2.0 * b[1]))
                .collect();
            for oi in 0..builder.n_omega() {
                builder.set_slab(ri, oi, &data);
            }
            builder.finish_r_slice(ri);
        }
        builder.build()
    }

    pub(crate) fn body(kind: &str) -> CgBody {
        CgBody {
            positions: vec![[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            masses: vec![110.0, 110.0],
            charges: vec![1.0, -0.5],
            radii: vec![3.0, 3.0],
            kinds: vec![kind.into(), "GLU".into()],
        }
    }

    pub(crate) fn archive() -> ScanArchive {
        let points = vec![
            PmfPoint {
                r: 5.0,
                free_energy: 1.5,
                mean_energy: 2.0,
                energy_variance: 0.5,
            },
            PmfPoint {
                r: 6.0,
                free_energy: -0.25,
                mean_energy: -0.5,
                energy_variance: 0.25,
            },
        ];
        ScanArchive {
            settings: ScanSettings {
                ph: 4.5,
                model: "calvados3".into(),
                single_bead: true,
                rmin: 5.0,
                rmax: 7.0,
                dr: 1.0,
                ionic_strength: 0.5,
                temperature: 298.15,
                max_ndiv: 1,
                gradient_threshold: 0.5,
                cutoff: 30.0,
                grid_size: 200,
                homo_dimer: false,
                subdivision: Default::default(),
            },
            table: table(),
            bodies: [body("LYS"), body("ARG")],
            virial: VirialCoeff::from_pmf([(5.0, 1.5), (6.0, -0.25)], None).unwrap(),
            points,
            topology_yaml: "atoms: []".into(),
        }
    }

    /// Derived from the beads, so they can't disagree with what a simulation uses.
    #[test]
    fn body_summaries_come_from_the_beads() {
        let body = body("LYS");
        assert_eq!(body.mass(), 220.0);
        assert_eq!(body.net_charge(), 0.5);
        assert_eq!(
            body.to_xyz(),
            "2\nGenerated by Duello\nLYS 1.000 0.000 0.000\nGLU -1.000 0.000 0.000\n"
        );
    }
}
