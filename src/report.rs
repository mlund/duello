use crate::Sample;
use coulomb::Vector3;
use itertools::Itertools;
use nu_ansi_term::Color::{Red, Yellow};
use num_traits::Inv;
use physical_constants::AVOGADRO_CONSTANT;
use rgb::RGB8;
use std::{
    f64::consts::PI,
    fs::File,
    io::Write,
    ops::{Add, Mul, Neg},
    path::PathBuf,
};
use textplots::{Chart, ColorPlot, Shape};

/// Helper struct for handling osmotic second virial coefficients, B2
#[derive(Debug, Clone)]
pub struct VirialCoeff {
    /// Osmotic second virial coefficient, B2 (Å³)
    b2: f64,
    /// Distance of closest approach, σ (Å)
    sigma: f64,
}

impl From<VirialCoeff> for f64 {
    fn from(v: VirialCoeff) -> f64 {
        v.b2
    }
}

impl VirialCoeff {
    // pub fn from_pmf_iterator(pomf: impl Iterator<Item = (f32, f32)>) -> Self {
    //     Self::from_pmf(&pomf.collect::<Vec<_>>())
    // }
    /// Calculate B2 from PMF data, pairs of (r, w(r)).
    /// w(r) should be in units of kT; r should be equidistant and in Å.
    /// Now calculate the osmotic second virial coefficient by integration:
    /// 𝐵₂ = -½ ∫ [ exp(-𝛽𝑤(𝑟) ) - 1 ] 4π𝑟² d𝑟
    pub fn from_pmf(pomf: &[(f32, f32)]) -> anyhow::Result<Self> {
        // use first two distances to calculate dr
        let mut iter = pomf.iter();
        let (r0, r1) = iter
            .by_ref()
            .map(|pair| pair.0)
            .take(2)
            .collect_tuple()
            .ok_or_else(|| anyhow::anyhow!("Error calculating PMF dr"))?;
        let dr = (r1 - r0) as f64;
        if dr <= 0.0 {
            anyhow::bail!("Negative dr in PMF");
        }
        let sigma = r0 as f64; // closest distance, "σ"
        let b2_hardsphere = 2.0 * PI / 3.0 * sigma.powi(3);
        // integrate
        let b2 = iter
            .map(|(r, w)| (*r as f64, *w as f64))
            .map(|(r, w)| w.neg().exp_m1() * r * r)
            .sum::<f64>()
            .mul(-2.0 * PI * dr)
            .add(b2_hardsphere);
        Ok(Self { b2, sigma })
    }
    /// Distance of closest approach, σ (Å)
    pub fn sigma(&self) -> f64 {
        self.sigma
    }
    /// Hard sphere contribution to second virial coefficient, B2hs (Å³)
    pub fn hardsphere(&self) -> f64 {
        2.0 * PI / 3.0 * self.sigma.powi(3)
    }
    /// Reduced second virial coefficient, B2 / B2hs
    pub fn reduced(&self) -> f64 {
        self.b2 / self.hardsphere()
    }
    /// Association constant, 𝐾𝑑⁻¹ = -2(𝐵₂ - 𝐵₂hs)
    /// See "Colloidal Domain" by Evans and Wennerström, 2nd Ed, p. 408
    pub fn association_const(&self) -> Option<f64> {
        const LITER_PER_CUBIC_ANGSTROM: f64 = 1e-27;
        let association_const =
            -2.0 * (self.b2 - self.hardsphere()) * LITER_PER_CUBIC_ANGSTROM * AVOGADRO_CONSTANT;
        if association_const.is_sign_positive() {
            Some(association_const)
        } else {
            None
        }
    }
    /// Dissociation constant, 𝐾𝑑
    /// See "Colloidal Domain" by Evans and Wennerström, 2nd Ed, p. 408
    pub fn dissociation_const(&self) -> Option<f64> {
        self.association_const().map(|k| k.inv())
    }
    /// Virial coefficient, B2 in mol⋅ml/g². Molar weights in g/mol.
    pub fn mol_ml_per_gram2(&self, mw1: f64, mw2: f64) -> f64 {
        const ML_PER_ANGSTROM3: f64 = 1e-24;
        self.b2 * ML_PER_ANGSTROM3 / (mw1 * mw2) * AVOGADRO_CONSTANT
    }
}

/// Write PMF and mean energy as a function of mass center separation to file
pub fn report_pmf(
    samples: &[(Vector3, Sample)],
    path: &PathBuf,
    masses: Option<(f64, f64)>,
) -> anyhow::Result<()> {
    // File with F(R) and U(R)
    let mut pmf_file = File::create(path).unwrap();
    let mut pmf_data = Vec::<(f32, f32)>::new();
    let mut mean_energy_data = Vec::<(f32, f32)>::new();
    writeln!(pmf_file, "# R/Å F/kT U/kT")?;
    samples.iter().for_each(|(r, sample)| {
        let mean_energy = sample.mean_energy() / sample.thermal_energy;
        let free_energy = sample.free_energy() / sample.thermal_energy;
        if mean_energy.is_finite() && free_energy.is_finite() {
            pmf_data.push((r.norm() as f32, free_energy as f32));
            mean_energy_data.push((r.norm() as f32, mean_energy as f32));
            writeln!(
                pmf_file,
                "{:.2} {:.2} {:.2}",
                r.norm(),
                free_energy,
                mean_energy
            )
            .or_else(|e| anyhow::bail!("Error writing to file: {}", e))
            .ok();
        }
    });

    let virial = VirialCoeff::from_pmf(&pmf_data)?;

    info!(
        "Second virial coefficient, 𝐵₂ = {:.2} Å³",
        f64::from(virial.clone())
    );
    if let Some((mw1, mw2)) = masses {
        info!(
            "                              = {:.2e} mol⋅ml/g²",
            virial.mol_ml_per_gram2(mw1, mw2)
        );
    }

    info!(
        "Reduced second virial coefficient, 𝐵₂ / 𝐵₂hs = {:.2} using σ = {:.2} Å",
        virial.reduced(),
        virial.sigma()
    );

    if let Some(kd) = virial.dissociation_const() {
        info!(
            "Dissociation constant, 𝐾𝑑 = {:.2e} mol/l using σ = {:.2} Å",
            kd,
            virial.sigma()
        );
    }

    info!(
        "Plot: {} and {} along mass center separation. In units of kT and angstroms.",
        Yellow.bold().paint("free energy"),
        Red.bold().paint("mean energy")
    );
    if log::max_level() >= log::Level::Info {
        const YELLOW: RGB8 = RGB8::new(255, 255, 0);
        const RED: RGB8 = RGB8::new(255, 0, 0);
        let rmin = mean_energy_data.first().unwrap().0;
        let rmax = mean_energy_data.last().unwrap().0;
        Chart::new(100, 50, rmin, rmax)
            .linecolorplot(&Shape::Lines(&mean_energy_data), RED)
            .linecolorplot(&Shape::Lines(&pmf_data), YELLOW)
            .nice();
    };
    Ok(())
}
