// Copyright 2024 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.
// You may obtain a copy of the license at
//
//     http://www.apache.org/licenses/license-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the license is distributed on an "as is" basis,
// without warranties or conditions of any kind, either express or implied.
// See the license for the specific language governing permissions and
// limitations under the license.

use crate::structure::Structure;
use anyhow::{Context, Result};
use faunus::topology::{self, AtomKind, Topology};
use std::io::{BufReader, Cursor};
use std::path::{Path, PathBuf};

/// Coarse-graining options for PDB/CIF input.
#[derive(Clone, Debug)]
pub struct CgOptions {
    /// pH for charge calculation (default 7.0).
    pub ph: f64,
    /// Force field model name (default "calvados3").
    pub model: String,
    /// Use single-bead CG policy (default true).
    pub single_bead: bool,
    /// Hydrophobic scaling (default: no scaling).
    pub scale_hydrophobic: cgkitten::forcefield::HydrophobicScaling,
    /// Chain IDs to keep (empty = all chains).
    pub chains: Vec<String>,
}

impl Default for CgOptions {
    fn default() -> Self {
        Self {
            ph: 7.0,
            model: "calvados3".to_string(),
            single_bead: true,
            scale_hydrophobic: cgkitten::forcefield::HydrophobicScaling::NoScale,
            chains: Vec::new(),
        }
    }
}

/// Result of loading a molecule: the structure, finalized topology, and topology file path.
pub struct LoadedMolecule {
    pub structure: Structure,
    pub topology: Topology,
    pub topology_path: PathBuf,
}

/// Returns `true` if the file extension indicates a PDB or mmCIF file.
fn is_structure_file(path: &Path) -> bool {
    path.extension().is_some_and(|e| {
        let e = e.to_ascii_lowercase();
        e == "pdb" || e == "cif" || e == "mmcif"
    })
}

/// Load a molecule from either XYZ+topology or PDB/CIF (auto-detected by extension).
///
/// - **XYZ input** (`.xyz`): `top_path` must be provided; loads structure and topology directly.
/// - **PDB/CIF input** (`.pdb`, `.cif`, `.mmcif`): runs cgkitten coarse-graining on the fly,
///   writes generated `.xyz` and `_topology.yaml` files next to the input, then loads them.
///   The `cg_opts` control pH, model, and CG policy.
pub fn load_molecule(
    mol_path: &Path,
    top_path: Option<&Path>,
    cg_opts: &CgOptions,
) -> Result<LoadedMolecule> {
    if is_structure_file(mol_path) {
        load_from_structure(mol_path, cg_opts)
    } else {
        anyhow::ensure!(
            cg_opts.chains.is_empty(),
            "--chain is only supported for PDB/CIF input, not XYZ"
        );
        // XYZ path — topology file is required
        let top = top_path.context("--top is required when input is an XYZ file")?;
        load_from_xyz(mol_path, top)
    }
}

/// Load from pre-generated XYZ + topology YAML (existing path).
fn load_from_xyz(mol_path: &Path, top_path: &Path) -> Result<LoadedMolecule> {
    let mut topology = Topology::from_file_partial(top_path)?;
    topology.finalize_atoms()?;
    topology.finalize_molecules()?;
    topology::set_missing_epsilon(topology.atomkinds_mut(), 2.479);

    let structure = Structure::from_xyz(&mol_path.to_path_buf(), topology.atomkinds())?;
    Ok(LoadedMolecule {
        structure,
        topology,
        topology_path: top_path.to_path_buf(),
    })
}

/// Run cgkitten CG on a PDB/CIF file, write output files, then load them.
fn load_from_structure(mol_path: &Path, cg_opts: &CgOptions) -> Result<LoadedMolecule> {
    let stem = mol_path
        .file_stem()
        .context("input file has no stem")?
        .to_string_lossy();
    let xyz_path = PathBuf::from(format!("{stem}.xyz"));
    let top_path = PathBuf::from(format!("{stem}_topology.yaml"));

    let is_pdb = mol_path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("pdb"));

    let policy: &dyn cgkitten::CoarseGrain = if cg_opts.single_bead {
        &cgkitten::SingleBead
    } else {
        &cgkitten::MultiBead
    };

    let file = std::fs::File::open(mol_path)
        .with_context(|| format!("Could not open {}", mol_path.display()))?;
    let reader = BufReader::new(file);

    let charge_calc = cgkitten::ChargeCalc::new().ph(cg_opts.ph).mc(10000);

    cgkitten::coarse_grain_to_files(
        reader,
        is_pdb,
        &charge_calc,
        &cg_opts.model,
        policy,
        0.02,
        cg_opts.scale_hydrophobic,
        &cg_opts.chains,
        &xyz_path,
        &top_path,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;

    info!(
        "Coarse-grained {} → {} + {}",
        mol_path.display(),
        xyz_path.display(),
        top_path.display()
    );

    load_from_xyz(&xyz_path, &top_path)
}

/// Result of in-memory molecule loading (no file paths).
#[derive(Clone)]
pub struct LoadedMoleculeInMemory {
    pub structure: Structure,
    pub topology: Topology,
    /// The topology YAML string (for passing to NonbondedMatrix::from_str).
    pub topology_yaml: String,
}

/// Load a molecule from PDB bytes entirely in memory (no file I/O).
///
/// Runs cgkitten coarse-graining, returns structure + topology + YAML string.
pub fn load_molecule_from_pdb_bytes(
    pdb_data: &[u8],
    cg_opts: &CgOptions,
) -> Result<LoadedMoleculeInMemory> {
    let reader = BufReader::new(Cursor::new(pdb_data));
    let policy: &dyn cgkitten::CoarseGrain = if cg_opts.single_bead {
        &cgkitten::SingleBead
    } else {
        &cgkitten::MultiBead
    };

    let beads = cgkitten::coarse_grain_pdb_with(reader, policy);
    let beads = cgkitten::filter_chains(beads, &cg_opts.chains);

    let charge_calc = cgkitten::ChargeCalc::new().ph(cg_opts.ph).mc(10000);
    charge_calc.log_conditions();
    let result = charge_calc.run(&beads);
    let charged = result.apply(&beads);

    let topo = cgkitten::topology::Topology::new(&charged, 0.02);
    let names = topo.bead_names();

    let multi_bead = charged
        .iter()
        .any(|b| b.bead_type == cgkitten::BeadType::Virtual);
    let ff = cgkitten::forcefield::from_name(&cg_opts.model, cg_opts.scale_hydrophobic)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let xyz_str = cgkitten::format_xyz(&charged, names, "generated by duello");
    let topology_yaml = cgkitten::format_topology(&topo, ff.as_deref(), multi_bead);

    let mut topology = Topology::from_str_partial(&topology_yaml)?;
    topology.finalize_atoms()?;
    topology.finalize_molecules()?;
    topology::set_missing_epsilon(topology.atomkinds_mut(), 2.479);

    let structure = Structure::from_xyz_str(&xyz_str, topology.atomkinds())?;

    Ok(LoadedMoleculeInMemory {
        structure,
        topology,
        topology_yaml,
    })
}

/// Resolve molecule B's backend structure and the atom-kind namespace shared by both
/// bodies of a scan.
///
/// `molecule_b = None` is a homodimer: reuse A's structure and topology from a
/// single coarse-graining. Otherwise the two molecules may have been coarse-grained
/// into separate id spaces, so their topologies are combined and B's structure ids
/// are remapped into the shared table (gh #39).
pub fn prepare_pair(
    topology_a: &Topology,
    structure_a: &Structure,
    molecule_b: Option<(&Topology, &Structure)>,
) -> (Structure, Topology) {
    let Some((topology_b, structure_b)) = molecule_b else {
        return (structure_a.clone(), topology_a.clone());
    };
    let (merged, b_remap) = combine_topologies(topology_a, topology_b);
    let mut structure_b = structure_b.clone();
    structure_b.remap_atom_ids(&b_remap);
    (structure_b, merged)
}

/// Merge molecule B's atom kinds into A's, returning the combined topology and a
/// map from B's old atom-kind id to its id in the merged table.
///
/// Kinds with identical names and interaction parameters share an id, keeping the
/// `n_types^2` pair table small. Distinct names are preserved because topology
/// YAML may contain name-specific pair overrides.
fn combine_topologies(topology_a: &Topology, topology_b: &Topology) -> (Topology, Vec<usize>) {
    let mut merged = topology_a.clone();
    let mut b_remap = Vec::with_capacity(topology_b.atomkinds().len());

    for kind_b in topology_b.atomkinds() {
        let merged_id = merged
            .atomkinds()
            .iter()
            .position(|kind| same_interaction(kind, kind_b))
            .unwrap_or_else(|| {
                let id = merged.atomkinds().len();
                merged.add_atomkind(kind_b.clone());
                id
            });
        b_remap.push(merged_id);
    }

    // Re-index the combined table without `finalize_atoms`, which rejects duplicate
    // names that can legitimately arise from independent titration outcomes.
    for (id, kind) in merged.atomkinds_mut().iter_mut().enumerate() {
        kind.set_id(id);
    }

    (merged, b_remap)
}

/// True if two atom kinds are interchangeable for scan energy evaluation.
fn same_interaction(a: &AtomKind, b: &AtomKind) -> bool {
    let opt_bits = |x: Option<f64>| x.map(f64::to_bits);
    a.name() == b.name()
        && a.charge().to_bits() == b.charge().to_bits()
        && a.mass().to_bits() == b.mass().to_bits()
        && opt_bits(a.sigma()) == opt_bits(b.sigma())
        && opt_bits(a.epsilon()) == opt_bits(b.epsilon())
        && opt_bits(a.lambda()) == opt_bits(b.lambda())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finalized(yaml: &str) -> Topology {
        let mut top = Topology::from_str_partial(yaml).unwrap();
        top.finalize_atoms().unwrap();
        top
    }

    #[test]
    fn combine_topologies_dedups_shared_and_appends_distinct() {
        let top_a = finalized(
            "atoms:\n  - {name: GLY, mass: 57.0, charge: 0.0, sigma: 4.5, eps: 0.83}\n  \
             - {name: O1, mass: 0.0, charge: -0.9, sigma: 2.0, eps: 0.83}\n",
        );
        let top_b = finalized(
            "atoms:\n  - {name: GLY, mass: 57.0, charge: 0.0, sigma: 4.5, eps: 0.83}\n  \
             - {name: O1, mass: 0.0, charge: -0.4, sigma: 2.0, eps: 0.83}\n",
        );

        let (merged, b_remap) = combine_topologies(&top_a, &top_b);

        assert_eq!(merged.atomkinds().len(), 3);
        for (i, kind) in merged.atomkinds().iter().enumerate() {
            assert_eq!(kind.id(), i);
        }
        assert_eq!(b_remap, vec![0, 2]);
        assert_eq!(
            merged.atomkinds()[2].charge().to_bits(),
            (-0.4f64).to_bits()
        );
    }

    #[test]
    fn combine_identical_topologies_is_a_noop() {
        let top = finalized(
            "atoms:\n  - {name: GLY, mass: 57.0, charge: 0.0, sigma: 4.5, eps: 0.83}\n  \
             - {name: ASP, mass: 115.0, charge: -1.0, sigma: 5.6, eps: 0.83}\n",
        );
        let (merged, b_remap) = combine_topologies(&top, &top);
        assert_eq!(merged.atomkinds().len(), 2);
        assert_eq!(b_remap, vec![0, 1]);
    }

    #[test]
    fn combine_topologies_preserves_distinct_names_with_same_parameters() {
        let top_a =
            finalized("atoms:\n  - {name: A, mass: 10.0, charge: 0.0, sigma: 2.0, eps: 0.5}\n");
        let top_b =
            finalized("atoms:\n  - {name: B, mass: 10.0, charge: 0.0, sigma: 2.0, eps: 0.5}\n");

        let (merged, b_remap) = combine_topologies(&top_a, &top_b);

        assert_eq!(merged.atomkinds().len(), 2);
        assert_eq!(merged.atomkinds()[0].name(), "A");
        assert_eq!(merged.atomkinds()[1].name(), "B");
        assert_eq!(b_remap, vec![1]);
    }
}
