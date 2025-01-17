use crate::{Sample, VirialCoeff};
use coulomb::Vector3;
use nu_ansi_term::Color::{Red, Yellow};
use rgb::RGB8;
use std::{fs::File, io::Write, path::PathBuf};
use textplots::{Chart, ColorPlot, Shape};

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
                "{:.2} {:.4} {:.4e}",
                r.norm(),
                free_energy,
                mean_energy
            )
            .or_else(|e| anyhow::bail!("Error writing to file: {}", e))
            .ok();
        }
    });

    let virial = VirialCoeff::from_pmf(&pmf_data, None)?;

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
