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
use faunus::topology::{self, Topology};
use std::io::BufReader;
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
}

impl Default for CgOptions {
    fn default() -> Self {
        Self {
            ph: 7.0,
            model: "calvados3".to_string(),
            single_bead: true,
            scale_hydrophobic: cgkitten::forcefield::HydrophobicScaling::NoScale,
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

    let charge_calc = cgkitten::ChargeCalc::new().ph(cg_opts.ph).mc(10000);

    let file = std::fs::File::open(mol_path)
        .with_context(|| format!("Could not open {}", mol_path.display()))?;
    let reader = BufReader::new(file);

    cgkitten::coarse_grain_to_files(
        reader,
        is_pdb,
        &charge_calc,
        &cg_opts.model,
        policy,
        0.02,
        cg_opts.scale_hydrophobic,
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
