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

use crate::Vector3;
use crate::{Sample, VirialCoeff};

/// Computed PMF results for programmatic consumption.
pub struct PmfResult {
    /// Finite points only, in increasing R.
    pub points: Vec<PmfPoint>,
    pub virial: VirialCoeff,
}

/// One point of the PMF and mean energy curves, in kT.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct PmfPoint {
    pub r: f32,
    pub free_energy: f32,
    pub mean_energy: f32,
    /// ⟨δU²⟩ over orientations at R in kT², which equals the orientational heat capacity C(R)/k.
    pub energy_variance: f32,
}

impl PmfPoint {
    /// Depends only on its own R-bin, so it is final as soon as that bin is scanned.
    /// `None` where the angular average is not finite, e.g. full overlap.
    pub fn new(r: f64, sample: &Sample) -> Option<Self> {
        let mean_energy = sample.mean_energy() / sample.thermal_energy();
        let free_energy = sample.free_energy() / sample.thermal_energy();
        (mean_energy.is_finite() && free_energy.is_finite()).then_some(Self {
            r: r as f32,
            free_energy: free_energy as f32,
            mean_energy: mean_energy as f32,
            energy_variance: sample.heat_capacity() as f32,
        })
    }
}

/// PMF points as CSV, for spreadsheets.
pub fn pmf_csv(points: &[PmfPoint]) -> String {
    let mut csv = String::from("R/Å,F/kT,U/kT,C/k\n");
    for p in points {
        csv += &format!(
            "{:.2},{:.4},{:.4},{:.4}\n",
            p.r, p.free_energy, p.mean_energy, p.energy_variance
        );
    }
    csv
}

/// Compute PMF/virial from samples without writing files.
pub fn compute_pmf(samples: &[(Vector3, Sample)]) -> anyhow::Result<PmfResult> {
    let points: Vec<PmfPoint> = samples
        .iter()
        .filter_map(|(r, sample)| PmfPoint::new(r.norm(), sample))
        .collect();
    let virial = VirialCoeff::from_pmf(points.iter().map(|p| (p.r, p.free_energy)), None)?;
    Ok(PmfResult { points, virial })
}

/// Advice when the pair still meets at σ, since the hard-sphere fill below σ then biases B₂₂.
pub fn sigma_warning(pmf: &[PmfPoint], sigma: f64) -> Option<String> {
    let weight = crate::weight_at_sigma(pmf, sigma).filter(|&w| crate::open_at_sigma(w))?;
    Some(format!(
        "e^(−w/kT) = {weight:.2} at σ = {sigma:.1} Å: the pair still meets there, \
         so B₂₂ depends on --rmin; scan from a smaller --rmin, \
         or a smaller --dr if closer R overlap"
    ))
}

/// Write PMF and mean energy as a function of mass center separation to file
#[cfg(feature = "cli")]
pub fn report_pmf(
    samples: &[(Vector3, Sample)],
    path: &std::path::PathBuf,
    masses: Option<(f64, f64)>,
) -> anyhow::Result<()> {
    use anyhow::{bail, Context};
    use nu_ansi_term::Color::{Red, Yellow};
    use rgb::RGB8;
    use std::{fs::File, io::Write};
    use textplots::{Chart, ColorPlot, Shape};

    let result = compute_pmf(samples)?;

    // File with F(R) and U(R)
    let mut pmf_file = File::create(path).context("Cannot create pmf file")?;
    writeln!(pmf_file, "# R/Å F/kT U/kT C/R <exp(-u/kT)-1>")?;
    for (r, sample) in samples {
        let mean_energy = sample.mean_energy() / sample.thermal_energy();
        let free_energy = sample.free_energy() / sample.thermal_energy();
        let heat_capacity = sample.heat_capacity();
        writeln!(
            pmf_file,
            "{:.2} {:.4} {:.4e} {:.4e} {:.4e}",
            r.norm(),
            free_energy,
            mean_energy,
            heat_capacity,
            sample.mean_exp_energy_m1()
        )
        .or_else(|e| bail!("Error writing to file: {}", e))
        .ok();
    }

    // Write B2 etc. to a JSON file
    let mut json_file = File::create(path.with_extension("json")).expect("Cannot open JSON output");
    writeln!(json_file, "{}", serde_json::to_string(&result.virial)?)?;

    info!(
        "Second virial coefficient, 𝐵₂ = {:.2} Å³",
        f64::from(result.virial.clone())
    );
    if let Some((mw1, mw2)) = masses {
        info!(
            "                              = {:.2e} mol⋅ml/g²",
            result.virial.mol_ml_per_gram2(mw1, mw2)
        );
    }
    if let Some(warning) = sigma_warning(&result.points, result.virial.sigma()) {
        warn!("{warning}");
    }

    info!(
        "Reduced second virial coefficient, 𝐵₂ / 𝐵₂hs = {:.2} using σ = {:.2} Å",
        result.virial.reduced(),
        result.virial.sigma()
    );

    if let Some(kd) = result.virial.dissociation_const() {
        info!(
            "Dissociation constant, 𝐾𝑑 = {:.2e} mol/l using σ = {:.2} Å",
            kd,
            result.virial.sigma()
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
        let (pmf, mean_energy): (Vec<_>, Vec<_>) = result
            .points
            .iter()
            .map(|p| ((p.r, p.free_energy), (p.r, p.mean_energy)))
            .unzip();
        let rmin = mean_energy.first().unwrap().0;
        let rmax = mean_energy.last().unwrap().0;
        Chart::new(100, 50, rmin, rmax)
            .linecolorplot(&Shape::Lines(&mean_energy), RED)
            .linecolorplot(&Shape::Lines(&pmf), YELLOW)
            .nice();
    };
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn opening(weight: f64) -> [PmfPoint; 2] {
        let point = |r, w: f64| PmfPoint {
            r,
            free_energy: w as f32,
            mean_energy: 0.0,
            energy_variance: 0.0,
        };
        [point(26.0, -weight.ln()), point(26.5, 0.0)]
    }

    #[test]
    fn warns_only_when_pair_still_meets_at_sigma() {
        let warning = sigma_warning(&opening(0.12), 26.0).unwrap();
        assert!(
            warning.contains("0.12") && warning.contains("--rmin"),
            "{warning}"
        );
        assert_eq!(sigma_warning(&opening(0.001), 26.0), None);
    }

    /// Two levels, 0 and ε, with degeneracies 1 and 3: a Bernoulli variable,
    /// so ⟨δU²⟩ = p(1−p)ε² with p the weight of the ε level.
    #[test]
    fn energy_variance_of_a_two_level_system() {
        let temperature = 298.15;
        let kt = crate::thermal_energy(temperature);
        let epsilon = 2.0;
        let sample =
            Sample::new(0.0, temperature, 1.0) + Sample::new(epsilon * kt, temperature, 3.0);
        let point = PmfPoint::new(4.0, &sample).unwrap();

        let p = 3.0 * (-epsilon).exp() / (1.0 + 3.0 * (-epsilon).exp());
        assert_relative_eq!(
            point.energy_variance as f64,
            p * (1.0 - p) * epsilon * epsilon,
            max_relative = 1e-6
        );
    }
}
