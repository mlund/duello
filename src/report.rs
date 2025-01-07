use crate::Sample;
use coulomb::Vector3;
use nu_ansi_term::Color::{Red, Yellow};
use num_traits::Inv;
use physical_constants::AVOGADRO_CONSTANT;
use rgb::RGB8;
use std::{
    fs::File,
    io::Write,
    ops::{Add, Mul, Neg},
    path::PathBuf,
};
use textplots::{Chart, ColorPlot, Shape};

/// Write PMF and mean energy as a function of mass center separation to file
pub fn report_pmf(samples: &[(Vector3, Sample)], path: &PathBuf, masses: Option<(f64, f64)>) {
    // File with F(R) and U(R)
    let mut pmf_file = File::create(path).unwrap();
    let mut pmf_data = Vec::<(f32, f32)>::new();
    let mut mean_energy_data = Vec::<(f32, f32)>::new();
    writeln!(pmf_file, "# R/Å F/kT U/kT").unwrap();
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
            .unwrap();
        }
    });

    // Now calculate the osmotic second virial coefficient by integrating `pmf_data`, w(r)
    // 𝐵₂ = -½ ∫ [ exp(-𝛽𝑤(𝑟) ) - 1 ] 4π𝑟² d𝑟
    let (r0, r1) = (pmf_data[0].0, pmf_data[1].0);
    let dr = (r1 - r0) as f64;
    assert!(dr > 0.0);
    let closest_approach = r0 as f64; // "σ"
    use std::f64::consts::PI;
    let b2_hardsphere = 2.0 * PI / 3.0 * closest_approach.powi(3);
    let b2 = pmf_data
        .iter()
        .map(|(r, w)| (*r as f64, *w as f64))
        .map(|(r, w)| w.neg().exp_m1() * r * r)
        .sum::<f64>()
        .mul(-2.0 * PI * dr)
        .add(b2_hardsphere);
    info!("Second virial coefficient, 𝐵₂ = {:.2} Å³", b2);
    if let Some((mw1, mw2)) = masses {
        const ML_PER_ANGSTROM3: f64 = 1e-24;
        info!(
            "                              = {:.2e} mol⋅ml/g²",
            b2 * ML_PER_ANGSTROM3 / (mw1 * mw2) * AVOGADRO_CONSTANT
        );
    }

    info!(
        "Reduced second virial coefficient, 𝐵₂ / 𝐵₂hs = {:.2} using σ = {:.2} Å",
        b2 / b2_hardsphere,
        closest_approach
    );

    // See "Colloidal Domain" by Evans and Wennerström, 2nd Ed, p. 408
    // 𝐾𝑑⁻¹ = -2(𝐵₂ - 𝐵₂hs)
    const LITER_PER_CUBIC_ANGSTROM: f64 = 1e-27;
    let association_const =
        -2.0 * (b2 - b2_hardsphere) * LITER_PER_CUBIC_ANGSTROM * AVOGADRO_CONSTANT;
    if association_const.is_sign_positive() {
        info!(
            "Dissociation constant, 𝐾𝑑 = {:.2e} mol/l using σ = {:.2} Å",
            association_const.inv(),
            closest_approach
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
    }
}
