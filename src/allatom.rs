//! All-atom structures placed like the coarse-grained bodies a scan used.

use crate::archive::CgBody;
use crate::nbody::Pose;
use crate::system::{cg_policy, keep_chains};
use crate::Vector3;
use anyhow::{ensure, Result};
use cgkitten::Atom;
use std::collections::HashSet;
use std::fmt::Write;

/// One molecule's atoms in the frame of its scanned CG body.
#[derive(Clone, Debug)]
pub struct AllAtom {
    /// Shifted so that their beads coincide with the body's
    atoms: Vec<Atom>,
}

/// Scans centre each body by translation only, so one offset maps atoms onto it.
const FRAME_TOLERANCE: f64 = 1e-3; // Å

impl AllAtom {
    /// `input` (PDB or mmCIF) as coarse-grained into `body`, keeping `chains`
    /// (empty keeps all). Fails if `input` does not coarse-grain into `body`.
    pub fn new(input: &[u8], chains: &[String], single_bead: bool, body: &CgBody) -> Result<Self> {
        let beads = keep_chains(
            cgkitten::coarse_grain_bytes(input, cg_policy(single_bead)),
            chains,
        )?;
        let offset = frame_offset(&beads, body)?;
        let atoms = bead_atoms(input, &beads)
            .map(|atom| moved(atom, |x| x - offset))
            .collect();
        Ok(Self { atoms })
    }

    /// The atoms with the body rotated and shifted to `pose`.
    pub fn place(&self, pose: &Pose) -> Vec<Atom> {
        self.atoms
            .iter()
            .cloned()
            .map(|atom| moved(atom, |x| pose.apply(x)))
            .collect()
    }
}

/// What the scan subtracted from `beads` to centre them into `body`.
fn frame_offset(beads: &[cgkitten::Bead], body: &CgBody) -> Result<Vector3> {
    ensure!(
        beads.len() == body.positions.len(),
        "{} beads, but the scanned body has {}",
        beads.len(),
        body.positions.len()
    );
    let offsets: Vec<Vector3> = beads
        .iter()
        .zip(&body.positions)
        .map(|(b, &p)| Vector3::new(b.x, b.y, b.z) - Vector3::from(p))
        .collect();
    let offset = offsets.iter().sum::<Vector3>() / offsets.len() as f64;
    let worst = offsets
        .iter()
        .map(|o| (o - offset).norm())
        .fold(0.0, f64::max);
    ensure!(
        worst < FRAME_TOLERANCE,
        "not the structure that was scanned (off by {worst:.2} Å)"
    );
    Ok(offset)
}

/// Atoms of the residues that became `beads`, so chains not kept and ligands
/// the scan never saw stay out. Matched by number, not name, as beads rename
/// some residues (bridged cysteines become CYX).
fn bead_atoms<'a>(input: &[u8], beads: &'a [cgkitten::Bead]) -> impl Iterator<Item = Atom> + 'a {
    let scanned: HashSet<(&str, i32)> = beads
        .iter()
        .map(|b| (b.chain_id.as_str(), b.res_seq))
        .collect();
    cgkitten::atoms_bytes(input)
        .into_iter()
        .filter(move |a| scanned.contains(&(a.chain_id.as_str(), a.res_seq)))
}

fn moved(mut atom: Atom, to: impl Fn(Vector3) -> Vector3) -> Atom {
    let x = to(Vector3::new(atom.x, atom.y, atom.z));
    (atom.x, atom.y, atom.z) = (x.x, x.y, x.z);
    atom
}

/// Multi-model PDB, one model per complex, with the two molecules as chains A
/// and B so viewers tell them apart whatever their source chains.
pub fn pdb(models: &[[Vec<Atom>; 2]]) -> String {
    let mut out = String::new();
    for (i, model) in models.iter().enumerate() {
        writeln!(out, "MODEL     {:>4}", i + 1).unwrap();
        model_lines(&mut out, model);
        out.push_str("ENDMDL\n");
    }
    out.push_str("END\n");
    out
}

fn model_lines(out: &mut String, model: &[Vec<Atom>; 2]) {
    let mut serial = 0;
    for (chain, atoms) in ["A", "B"].iter().zip(model) {
        for atom in atoms {
            serial += 1;
            atom_line(out, serial, chain, atom);
        }
        // Without it, viewers bond the end of A to the start of B.
        out.push_str("TER\n");
    }
}

/// Fixed columns per the PDB format, which viewers read by position: fields
/// are cut to width and serials wrap within 1..=99999.
fn atom_line(out: &mut String, serial: usize, chain: &str, a: &Atom) {
    // Two-letter elements (Ca, Zn) start in column 13 so " CA " stays an α-carbon.
    let name = if a.name.len() >= 4 || a.element.len() == 2 {
        format!("{:<4}", fit(&a.name, 4))
    } else {
        format!(" {:<3}", fit(&a.name, 3))
    };
    writeln!(
        out,
        "{:<6}{:>5} {name} {:>3} {chain}{:>4}{:1}   {:8.3}{:8.3}{:8.3}  1.00  0.00          {:>2}",
        fit(&a.record_name, 6),
        (serial - 1) % 99_999 + 1,
        fit(&a.res_name, 3),
        a.res_seq.rem_euclid(10_000),
        fit(&a.i_code, 1),
        a.x,
        a.y,
        a.z,
        fit(&a.element, 2),
    )
    .unwrap();
}

/// At most `width` characters of `s`.
fn fit(s: &str, width: usize) -> &str {
    s.char_indices().nth(width).map_or(s, |(end, _)| &s[..end])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::archive::CgBody;
    use crate::nbody::{trajectory::expand, Pose};
    use crate::system::{CgOptions, PairSpec, PairSystem, Source};
    use crate::{UnitQuaternion, Vector3};
    use faunus::interatomic::coulomb::{Medium, Salt};
    use std::path::PathBuf;

    fn cif() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/calvados3/2CDS.cif")
    }

    /// The body a scan of `source` uses: coarse-grained, titrated and centred.
    fn scanned_body(source: Source, chains: &[String]) -> CgBody {
        let system = PairSystem::load(PairSpec {
            a: source,
            b: None,
            top: None,
            chains: [chains, &[]],
            cg: CgOptions::default(),
            medium: &Medium::salt_water(298.15, Salt::SodiumChloride, 0.1),
        })
        .unwrap();
        CgBody::new(&system.ref_a, system.topology.atomkinds())
    }

    fn pose() -> Pose {
        Pose {
            position: Vector3::new(30.0, -4.0, 12.0),
            orientation: UnitQuaternion::from_scaled_axis(Vector3::new(0.4, -1.1, 2.0)),
        }
    }

    fn identity() -> Pose {
        Pose {
            position: Vector3::zeros(),
            orientation: UnitQuaternion::identity(),
        }
    }

    fn bead_positions(pdb: &str, chain: &str) -> Vec<Vector3> {
        let beads = cgkitten::coarse_grain_bytes(pdb.as_bytes(), &cgkitten::SingleBead);
        cgkitten::filter_chains(beads, &[chain])
            .iter()
            .map(|b| Vector3::new(b.x, b.y, b.z))
            .collect()
    }

    /// 2CDS as scanned, and its atoms in the body's frame.
    fn lysozyme() -> (CgBody, AllAtom) {
        let body = scanned_body(Source::File(&cif()), &[]);
        let atoms = AllAtom::new(&std::fs::read(cif()).unwrap(), &[], true, &body).unwrap();
        (body, atoms)
    }

    #[test]
    fn placed_atoms_coarse_grain_to_the_placed_beads() {
        let (body, atoms) = lysozyme();
        let pdb = pdb(&[[atoms.place(&identity()), atoms.place(&pose())]]);

        for (chain, pose) in [("A", identity()), ("B", pose())] {
            let expected = expand(&body, &pose);
            let found = bead_positions(&pdb, chain);
            assert_eq!(found.len(), expected.len(), "chain {chain}");
            // PDB keeps three decimals.
            let worst = found.iter().zip(&expected).map(|(a, b)| (a - b).norm());
            assert!(worst.fold(0.0, f64::max) < 2e-3, "chain {chain}");
        }
    }

    /// 2CDS twice, far apart, as chains A and B of one file.
    fn two_chains() -> (String, usize) {
        let (_, atoms) = lysozyme();
        let file = pdb(&[[atoms.place(&identity()), atoms.place(&pose())]]);
        (file, atoms.atoms.len())
    }

    #[test]
    fn only_the_scanned_chains_are_placed() {
        let (file, n_atoms) = two_chains();
        let chain_b = ["B".to_string()];
        let body = scanned_body(Source::PdbBytes(file.as_bytes()), &chain_b);

        let atoms = AllAtom::new(file.as_bytes(), &chain_b, true, &body).unwrap();
        let placed = atoms.place(&identity());
        assert_eq!(placed.len(), n_atoms);
        assert!(placed.iter().all(|a| a.chain_id == "B"));
        assert!(AllAtom::new(file.as_bytes(), &[], true, &body).is_err());
    }

    /// Viewers read fixed columns (PDB v3.3, ATOM record), not whitespace.
    fn assert_columns(line: &str, atom: &Atom) {
        let field = |from: usize, to: usize| line[from - 1..to].trim();
        let number = |from, to| field(from, to).parse::<f64>().unwrap();
        assert_eq!(field(13, 16), atom.name, "{line}");
        assert_eq!(field(18, 20), atom.res_name, "{line}");
        assert_eq!(field(23, 26), atom.res_seq.to_string(), "{line}");
        assert_eq!(field(77, 78), atom.element, "{line}");
        for (from, x) in [(31, atom.x), (39, atom.y), (47, atom.z)] {
            assert!((number(from, from + 7) - x).abs() < 1e-3, "{line}");
        }
    }

    #[test]
    fn every_model_reads_back_with_its_atoms() {
        let (_, atoms) = lysozyme();
        let models = [
            [atoms.place(&identity()), atoms.place(&pose())],
            [atoms.place(&pose()), atoms.place(&identity())],
        ];
        let file = pdb(&models);

        let blocks: Vec<&str> = file
            .split("ENDMDL\n")
            .filter(|b| b.contains("ATOM"))
            .collect();
        assert_eq!(blocks.len(), models.len());
        for (block, [a, b]) in blocks.iter().zip(&models) {
            let written: Vec<&Atom> = a.iter().chain(b).collect();
            assert_eq!(cgkitten::atoms_bytes(block.as_bytes()).len(), written.len());
            let lines = block
                .lines()
                .filter(|l| l.starts_with("ATOM") || l.starts_with("HETATM"));
            lines
                .zip(written)
                .for_each(|(line, w)| assert_columns(line, w));
        }
    }

    #[test]
    fn atoms_without_a_bead_are_left_out() {
        let (file, n_atoms) = two_chains();
        // A ligand coarse-graining ignores, so the scan never saw it.
        let ligand =
            "HETATM99999  C1  HEM A 900       1.000   2.000   3.000  1.00  0.00           C\n";
        let file = file.replacen("ENDMDL", &format!("{ligand}ENDMDL"), 1);
        let chain_a = ["A".to_string()];
        let body = scanned_body(Source::PdbBytes(file.as_bytes()), &chain_a);

        let atoms = AllAtom::new(file.as_bytes(), &chain_a, true, &body).unwrap();
        assert_eq!(atoms.atoms.len(), n_atoms);
    }
}
