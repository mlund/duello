//! Salt left free to screen once net-charged proteins take up volume and
//! displace co-ions (Donnan). Its co-ion concentration is offered as the
//! ionic strength: counterions stay bound near the macroions and are left
//! out, which fits crowded CPPM repulsion better than the buffer value.

use anyhow::{ensure, Result};
use physical_constants::AVOGADRO_CONSTANT;

/// Typical for globular proteins, cm³/g; Kaieda et al. used measured values.
pub const PARTIAL_SPECIFIC_VOLUME: f64 = 0.73;

/// Å³ of a protein of `molar_mass` g/mol; one definition so the HSY sphere
/// and the free-salt estimate agree.
pub fn protein_volume(molar_mass: f64) -> f64 {
    molar_mass * PARTIAL_SPECIFIC_VOLUME * 1e24 / AVOGADRO_CONSTANT
}

/// Free salt beside proteins in Donnan equilibrium with a salt reservoir.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FreeSalt {
    /// Co-ion concentration in the free volume, mol/L.
    pub molarity: f64,
    /// Protein volume fraction η.
    pub volume_fraction: f64,
}

/// Molarities in mol/L, `protein_volume` in Å³, `charge` in e.
pub fn free_salt(
    reservoir: f64,
    charge: f64,
    protein_molarity: f64,
    protein_volume: f64,
) -> Result<FreeSalt> {
    ensure!(
        [reservoir, protein_molarity, protein_volume]
            .iter()
            .all(|&x| x >= 0.0),
        "salt, protein concentration and volume must be non-negative"
    );
    let volume_fraction = crate::number_density(protein_molarity) * protein_volume;
    ensure!(
        volume_fraction < 1.0,
        "proteins would fill {:.0}% of the volume",
        volume_fraction * 100.0
    );
    // Counterions per free volume; electroneutrality with the co-ions and
    // c₊c₋ = c_res² then give a quadratic in the co-ion concentration.
    let counterions = charge.abs() * protein_molarity / (1.0 - volume_fraction);
    let molarity = (-counterions + counterions.hypot(2.0 * reservoir)) / 2.0;
    Ok(FreeSalt {
        molarity,
        volume_fraction,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn without_protein_the_reservoir_is_free() {
        let salt = free_salt(0.02, 8.0, 0.0, 17_000.0).unwrap();
        assert_eq!(salt.molarity, 0.02);
        assert_eq!(salt.volume_fraction, 0.0);
    }

    /// ρ_c = 8 × 0.005 = 0.04 M; c = (−0.04 + √(0.04² + 4 × 0.02²))/2.
    #[test]
    fn counterions_displace_salt_whatever_their_sign() {
        for charge in [8.0, -8.0] {
            let salt = free_salt(0.02, charge, 0.005, 0.0).unwrap();
            approx::assert_relative_eq!(salt.molarity, 0.0082843, max_relative = 1e-5);
        }
    }

    /// 0.01 M × 6.0221e-4 Å⁻³/M × 16605.5 Å³ = 0.1.
    #[test]
    fn neutral_proteins_only_take_up_volume() {
        let salt = free_salt(0.02, 0.0, 0.01, 16_605.5).unwrap();
        approx::assert_relative_eq!(salt.volume_fraction, 0.1, max_relative = 1e-5);
        approx::assert_relative_eq!(salt.molarity, 0.02, max_relative = 1e-12);
    }

    #[test]
    fn impossible_solutions_are_refused() {
        assert!(free_salt(0.02, 0.0, 0.01, 166_055.0).is_err(), "η = 1");
        assert!(free_salt(-0.02, 0.0, 0.01, 1.0).is_err());
        assert!(free_salt(0.02, 0.0, -0.01, 1.0).is_err());
        assert!(free_salt(0.02, 0.0, 0.01, -1.0).is_err());
    }
}
