//! Onebody extracts: per-molecule beads, multipole moments and titration curve.

use super::zipio::Entry;
use super::MoleculeExtract;
use crate::molecule::{BeadRow, Curve, CurvePoint};
use cgkitten::Multipole;

pub fn entries(molecules: &[MoleculeExtract]) -> Vec<Entry<'static>> {
    let stems = molecules.iter().map(|m| stem(m.name));
    let dirs = super::unique_names(stems);
    molecules
        .iter()
        .zip(dirs)
        .flat_map(|(m, dir)| molecule_entries(m, &format!("onebody/{dir}")))
        .collect()
}

/// "4lzt.pdb" → "4lzt", so the directory reads as the molecule.
fn stem(name: &str) -> String {
    let name = super::safe_name(name);
    match name.rsplit_once('.') {
        Some((stem, _)) if !stem.is_empty() => stem.to_string(),
        _ => name,
    }
}

fn molecule_entries(m: &MoleculeExtract, dir: &str) -> Vec<Entry<'static>> {
    let mut entries = vec![
        Entry::text(
            &format!("{dir}/beads.csv"),
            beads_csv(&m.molecule.beads),
            "text/csv",
            "Coarse-grained beads: titrated charge, mass and force-field parameters",
        ),
        Entry::text(
            &format!("{dir}/multipoles.csv"),
            multipoles_csv(&m.molecule.multipole, m.ph),
            "text/csv",
            "Charge and multipole moments at the session's pH",
        ),
    ];
    if let Some(curve) = m.curve {
        entries.push(Entry::text(
            &format!("{dir}/titration.csv"),
            titration_csv(curve),
            "text/csv",
            "Titration curve: Monte Carlo (mc) and Henderson–Hasselbalch (ideal)",
        ));
    }
    entries
}

fn beads_csv(beads: &[BeadRow]) -> String {
    let show = |x: Option<f64>| x.map_or(String::new(), |v| v.to_string());
    let mut csv = String::from("chain,residue,kind,Z/e,var(Z)/e²,mass/(g/mol),σ/Å,ε/(kJ/mol),λ\n");
    for b in beads {
        csv += &format!(
            "{},{},{},{:.4},{},{},{},{},{}\n",
            b.chain,
            b.residue,
            b.kind,
            b.charge,
            show(b.charge_variance),
            b.mass,
            show(b.sigma),
            show(b.epsilon),
            show(b.lambda),
        );
    }
    csv
}

fn multipoles_csv(m: &Multipole, ph: f64) -> String {
    let rows = [
        ("pH", ph, ""),
        ("net charge ⟨Z⟩", m.charge, "e"),
        ("⟨Z²⟩", m.charge_sq, "e²"),
        ("capacitance ⟨Z²⟩−⟨Z⟩²", m.capacitance(), "e²"),
        ("dipole ⟨|μ|⟩", m.dipole, "e·Å"),
        ("⟨μ²⟩", m.dipole_sq, "(e·Å)²"),
        ("quadrupole ⟨‖Θ‖⟩", m.quadrupole, "e·Å²"),
        ("⟨‖Θ‖²⟩", m.quadrupole_sq, "(e·Å²)²"),
    ];
    let mut csv = String::from("quantity,value,unit\n");
    for (quantity, value, unit) in rows {
        csv += &format!("{quantity},{value},{unit}\n");
    }
    csv
}

fn titration_csv(curve: &Curve) -> String {
    let mut csv = String::from("method,pH,Z/e,C/e²,μ/(e·Å),Θ/(e·Å²)\n");
    let rows = |method, points: &[CurvePoint]| {
        points
            .iter()
            .map(|p| {
                format!(
                    "{method},{:.2},{:.4},{:.4},{:.4},{:.4}\n",
                    p.ph, p.charge, p.capacitance, p.dipole, p.quadrupole
                )
            })
            .collect::<String>()
    };
    csv += &rows("mc", curve.points());
    csv += &rows("ideal", curve.ideal());
    csv
}
