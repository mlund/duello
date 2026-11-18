//! `duello modes`: binding modes of a saved scan, as in the web tool's
//! Binding Modes pane, written to a directory.

use anyhow::{Context, Result};
use duello::allatom::AllAtom;
use duello::archive::ScanArchive;
use duello::bundle::Bundle;
use duello::modes::export::{atom_pair, is_mode_file, session_atoms, ModeExport};
use duello::modes::ModeSpec;
use std::path::{Path, PathBuf};

#[derive(clap::Args)]
pub struct ModesArgs {
    /// Session (.duello.zip) with a two-body scan
    session: PathBuf,
    /// First structure (PDB/mmCIF) as scanned, for all-atom models; else the
    /// session's own, if it kept them
    #[arg(short = '1', long)]
    mol1: Option<PathBuf>,
    /// Second structure [default: the first]
    #[arg(short = '2', long, requires = "mol1")]
    mol2: Option<PathBuf>,
    /// Chain IDs kept in the scan. Repeat for multiple: --chain A --chain B
    #[arg(long)]
    chain: Vec<String>,
    /// Poses whose B beads lie within this RMSD (Å) are one mode
    #[arg(long, default_value = "4")]
    rmsd: f64,
    /// Number of modes
    #[arg(short = 'n', long, default_value = "10")]
    modes: usize,
    /// Output directory
    #[arg(short = 'o', long, default_value = "modes")]
    output: PathBuf,
}

pub fn run_modes(args: &ModesArgs) -> Result<()> {
    let bundle = Bundle::from_zip(&read(&args.session)?)?;
    let scan = bundle
        .scan
        .as_ref()
        .context("the session has no two-body scan")?;
    let spec = ModeSpec {
        rmsd: args.rmsd,
        max_modes: args.modes,
    };
    let atoms = all_atoms(args, &bundle, scan)?;
    let export = ModeExport::new(scan, spec, atoms);
    print_table(&export);
    write_files(&export, &args.output)
}

/// Given inputs win over the session's own and, unlike those, must fit the
/// scan; without either, CG only.
fn all_atoms(
    args: &ModesArgs,
    bundle: &Bundle,
    scan: &ScanArchive,
) -> Result<Option<[AllAtom; 2]>> {
    let Some(mol1) = &args.mol1 else {
        return Ok(session_atoms(&bundle.session, scan));
    };
    let a = read(mol1)?;
    let b = args.mol2.as_deref().map(read).transpose()?;
    let b = b.as_deref().unwrap_or(&a);
    atom_pair([(&a, &args.chain), (b, &args.chain)], scan).map(Some)
}

fn print_table(export: &ModeExport) {
    let modes = &export.modes;
    println!(
        "{:>4} {:>7} {:>8} {:>10} {:>9} {:>8} {:>6}",
        "mode", "share/%", "of Ka/%", "B₂/Å³", "poses", "U/kT", "R/Å"
    );
    for (rank, m) in (1..).zip(&modes.modes) {
        let ka = m
            .association_share
            .map_or("–".into(), |s| format!("{:.1}", 100.0 * s));
        println!(
            "{rank:>4} {:>7.1} {ka:>8} {:>10.3e} {:>9} {:>8.2} {:>6.1}",
            100.0 * m.share,
            m.b2,
            m.n_poses,
            m.pose.energy,
            m.pose.r
        );
    }
    let explained: f64 = modes.modes.iter().map(|m| m.share).sum();
    println!(
        "{} modes hold {:.0}% of the attraction, Σc = {:.3e} Å³",
        modes.modes.len(),
        100.0 * explained,
        modes.attractive_b2
    );
    if modes.association.is_some() {
        let ka: f64 = modes.modes.iter().filter_map(|m| m.association_share).sum();
        println!(
            "and {:.0}% of K_a's B₂ − B₂ʳᵉᶠ, which soft repulsion offsets, so it can pass 100%",
            100.0 * ka
        );
    }
}

fn write_files(export: &ModeExport, dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir).with_context(|| format!("Could not create {}", dir.display()))?;
    remove_earlier(dir)?;
    for file in export.files() {
        let path = dir.join(&file.name);
        std::fs::write(&path, file.contents)
            .with_context(|| format!("Could not write {}", path.display()))?;
    }
    log::info!("Wrote binding modes to {}", dir.display());
    Ok(())
}

/// Files of an earlier run would sit beside the new ones, numbered by a
/// table that no longer lists them. Only names this command writes go.
fn remove_earlier(dir: &Path) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        if is_mode_file(name) {
            std::fs::remove_file(&path)
                .with_context(|| format!("Could not remove {}", path.display()))?;
        }
    }
    Ok(())
}

fn read(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).with_context(|| format!("Could not open {}", path.display()))
}
