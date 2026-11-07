//! Self-contained two-body scan, exportable as bytes.
//!
//! Holds everything needed to view a scan or run a many-body simulation from
//! it without the original PDB: the 6D table, the coarse-grained bodies, and
//! the derived PMF and virial results. A web [`Session`] rides along.

use crate::icoscan::{table_metadata, ScanParams, ScanResult};
use crate::nbody::RunSpec;
use crate::report::{compute_pmf, PmfPoint};
use crate::structure::Structure;
use crate::{Subdivision, Table6DAdaptive, VirialCoeff};
use anyhow::{anyhow, ensure, Result};
use bincode::Options as _;
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

/// A finished two-body scan.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScanArchive {
    pub settings: ScanSettings,
    #[serde(with = "table_bytes")]
    pub table: Table6DAdaptive<f32>,
    pub bodies: [CgBody; 2],
    pub points: Vec<PmfPoint>,
    #[serde(with = "virial_parts")]
    pub virial: VirialCoeff,
    pub topology_yaml: String,
    pub session: Session,
}

/// The web app's inputs at export, so importing a scan restores the user's
/// work; inputs only, never results. Empty when the CLI made the scan.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Session {
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StructureFile {
    pub name: String,
    #[serde(with = "serde_bytes")]
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
            session: Session::default(),
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
    let mut xyz = format!("{n_atoms}\nGenerated by Duello\n");
    for (kind, [x, y, z]) in atoms {
        xyz += &format!("{kind} {x:.3} {y:.3} {z:.3}\n");
    }
    xyz
}

/// Identifies the file before any decoding, so a wrong file gives a clear error.
const MAGIC: &[u8; 12] = b"DUELLO-SCAN\0";
/// Bumped on any layout change: bincode isn't self-describing, so old readers
/// would otherwise misread new files silently.
const FORMAT_VERSION: u32 = 3;

impl ScanArchive {
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = MAGIC.to_vec();
        bytes.extend(FORMAT_VERSION.to_le_bytes());
        bincode::serialize_into(&mut bytes, self)?;
        Ok(bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let body = bytes
            .strip_prefix(MAGIC.as_slice())
            .ok_or_else(|| anyhow!("not a duello scan file"))?;
        let (version, body) = body
            .split_first_chunk::<4>()
            .ok_or_else(|| anyhow!("not a duello scan file"))?;
        let version = u32::from_le_bytes(*version);
        ensure!(
            version == FORMAT_VERSION,
            "scan file format version {version} is not supported (expected {FORMAT_VERSION})"
        );
        let archive: Self = bincode_options(body.len()).deserialize(body)?;
        for body in &archive.bodies {
            body.check_consistency()?;
        }
        archive.session.check_consistency()?;
        // The label is stored beside the table, so a writer could disagree.
        let labelled = archive.settings.subdivision;
        ensure!(
            archive
                .table
                .levels
                .iter()
                .all(|l| l.subdivision() == labelled),
            "settings say {labelled:?} but the table uses another angular mesh"
        );
        Ok(archive)
    }
}

impl Session {
    /// Restoring indexes `structures` by `scanned`.
    fn check_consistency(&self) -> Result<()> {
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
    fn check_consistency(&self) -> Result<()> {
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

/// Matches `bincode::serialize_into`. The limit stops a corrupt length prefix
/// from requesting more memory than the input could fill, and trailing bytes
/// mean a damaged or concatenated file, since `to_bytes` never writes any.
fn bincode_options(limit: usize) -> impl bincode::Options {
    use bincode::Options;
    bincode::options()
        .with_fixint_encoding()
        .reject_trailing_bytes()
        .with_limit(limit as u64)
}

/// Routes the table through icotable's byte format, which validates on load.
mod table_bytes {
    use crate::Table6DAdaptive;
    use serde::{de::Error as _, ser::Error as _, Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(table: &Table6DAdaptive<f32>, s: S) -> Result<S::Ok, S::Error> {
        let bytes = table.to_bytes().map_err(S::Error::custom)?;
        s.serialize_bytes(&bytes)
    }

    /// `ByteBuf` reads the bytes in one go; a plain `Vec<u8>` decodes them one
    /// at a time, which is slow for multi-MB tables, notably in WASM.
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Table6DAdaptive<f32>, D::Error> {
        let bytes = serde_bytes::ByteBuf::deserialize(d)?;
        Table6DAdaptive::from_bytes(&bytes).map_err(D::Error::custom)
    }
}

/// `VirialCoeff`'s own `Serialize` writes report JSON with optional fields,
/// which a non-self-describing format can't read back.
mod virial_parts {
    use crate::VirialCoeff;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(v: &VirialCoeff, s: S) -> Result<S::Ok, S::Error> {
        (v.b2(), v.sigma()).serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<VirialCoeff, D::Error> {
        let (b2, sigma) = <(f64, f64)>::deserialize(d)?;
        Ok(VirialCoeff::from_raw_parts(b2, sigma))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AdaptiveBuilder, Vector3};
    use icotable::lookup::Lookup6D;

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

    fn body(kind: &str) -> CgBody {
        CgBody {
            positions: vec![[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            masses: vec![110.0, 110.0],
            charges: vec![1.0, -0.5],
            radii: vec![3.0, 3.0],
            kinds: vec![kind.into(), "GLU".into()],
        }
    }

    fn archive() -> ScanArchive {
        let points = vec![
            PmfPoint {
                r: 5.0,
                free_energy: 1.5,
                mean_energy: 2.0,
            },
            PmfPoint {
                r: 6.0,
                free_energy: -0.25,
                mean_energy: -0.5,
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
            session: Session::default(),
        }
    }

    /// A web session: two loaded structures, both scanned, a free-salt
    /// estimate and the many-body inputs.
    fn session() -> Session {
        let file = |name: &str, contents: &[u8]| StructureFile {
            name: name.into(),
            contents: contents.to_vec(),
        };
        let mut run = crate::nbody::RunSpec::new(200, 0.002);
        run.record_trajectory = true;
        run.msa.debye_length = Some(12.0);
        Session {
            structures: vec![file("4lzt.pdb", b"ATOM 1"), file("1ubq.cif", b"data_")],
            scanned: Some([1, 0]),
            estimate: Some(FreeSaltEstimate {
                reservoir: 0.02,
                protein_molarity: 0.002,
            }),
            run: Some(run),
        }
    }

    #[test]
    fn round_trip_keeps_the_ionic_strength_and_the_session() {
        let mut original = archive();
        original.session = session();
        let restored = ScanArchive::from_bytes(&original.to_bytes().unwrap()).unwrap();
        assert_eq!(restored.settings.ionic_strength, 0.5);
        assert_eq!(restored.session, session());
    }

    #[test]
    fn a_scanned_pair_beyond_the_structures_is_refused() {
        let mut bad = archive();
        bad.session = Session {
            scanned: Some([2, 0]),
            ..session()
        };
        let error = ScanArchive::from_bytes(&bad.to_bytes().unwrap()).unwrap_err();
        assert!(error.to_string().contains("scanned"), "{error}");
    }

    #[test]
    fn version_2_files_are_refused() {
        let mut bytes = archive().to_bytes().unwrap();
        bytes[12] = 2;
        let error = ScanArchive::from_bytes(&bytes).unwrap_err().to_string();
        assert!(error.contains("version 2"), "{error}");
    }

    #[test]
    fn round_trip_keeps_everything_needed_without_the_pdb() {
        let original = archive();
        let restored = ScanArchive::from_bytes(&original.to_bytes().unwrap()).unwrap();

        assert_eq!(restored.settings, original.settings);
        assert_eq!(restored.bodies[1].kinds, ["ARG", "GLU"]);
        assert_eq!(restored.bodies[0].charges, [1.0, -0.5]);
        assert_eq!(restored.net_charges(), (0.5, 0.5));
        assert_eq!(restored.points[1].free_energy, -0.25);
        assert_eq!(restored.virial.b2(), original.virial.b2());
        assert_eq!(restored.topology_yaml, "atoms: []");

        let (a, b) = (Vector3::new(0.6, 0.0, 0.8), Vector3::new(0.0, -1.0, 0.0));
        assert_eq!(
            restored.table.lookup(6.2, 0.4, &a, &b, None),
            original.table.lookup(6.2, 0.4, &a, &b, None)
        );
    }

    #[test]
    fn a_scheme_label_that_disagrees_with_the_table_is_refused() {
        let mut mislabelled = archive();
        mislabelled.settings.subdivision = crate::Subdivision::Lattice;
        let bytes = mislabelled.to_bytes().unwrap();
        assert!(ScanArchive::from_bytes(&bytes).is_err());
    }

    #[test]
    fn header_is_magic_then_little_endian_version() {
        let bytes = archive().to_bytes().unwrap();
        assert_eq!(&bytes[..16], b"DUELLO-SCAN\0\x03\0\0\0");
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

    /// Imported files are untrusted; per-bead code zips these vectors.
    #[test]
    fn bodies_with_mismatched_bead_data_are_rejected() {
        let mut bad = archive();
        bad.bodies[1].kinds.pop();
        let error = ScanArchive::from_bytes(&bad.to_bytes().unwrap()).unwrap_err();
        assert!(error.to_string().contains("bead"), "{error}");
    }

    #[test]
    fn foreign_or_future_files_are_rejected_with_a_reason() {
        let bytes = archive().to_bytes().unwrap();
        let error = |bytes: &[u8]| ScanArchive::from_bytes(bytes).unwrap_err().to_string();

        let mut pdb = b"ATOM      1  CA  LYS A   1".to_vec();
        pdb.extend(&bytes[16..]);
        assert!(error(&pdb).contains("not a duello scan"), "{}", error(&pdb));
        assert!(error(b"DUELLO").contains("not a duello scan"));

        let mut future = bytes.clone();
        future[12] = 99;
        assert!(error(&future).contains("version 99"), "{}", error(&future));

        assert!(ScanArchive::from_bytes(&bytes[..bytes.len() - 10]).is_err());
    }
}
