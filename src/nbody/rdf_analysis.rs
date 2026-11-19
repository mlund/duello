//! Kirkwood–Buff analysis of the mass-centre g(r).
//!
//! A closed box biases the running integral 4π∫(g − 1)r²dr: bodies gathered
//! near one are missing far from it, so it drifts instead of levelling off.
//! G₂₂ therefore follows the published radial route (Simon et al. 2022,
//! 10.1063/5.0106162), implemented from the papers' equations:
//!
//! 1. Correct g for the fixed body count (Ganguly & van der Vegt 2013,
//!    10.1021/ct301017q, Eq. 5, p. 1352): "excess (depletion) of particle type
//!    j around particle type i at a local-scale is compensated by depletion
//!    (excess) of particle type j at long distances as the total number of
//!    particles of type j is fixed."
//! 2. Integrate it over spheres of diameter L with the weight
//!    4πr²(1 − 3x/2 + x³/2), x = r/L (Krüger et al. 2013, 10.1021/jz301992u,
//!    Table 1, p. 236), for which "G^V − G∞ ∼ S/V ∼ 1/R" (p. 236).
//! 3. Take G₂₂ as the slope of L·G^V(L) against L: "An equivalent method
//!    consists of plotting L × G_ab(L) as a function of L and extracting the
//!    linear slope at large L, which tends to G∞_ab" (Simon et al. 2022,
//!    pp. 130901-3–4). The fit spans the outer half of L and must be at least
//!    10 Å wide with R² ≥ 0.99, the thresholds of Peroutka et al. 2026
//!    (10.1021/acs.jcim.6c00344, p. 6825), who warn that passing them is no
//!    guarantee of accuracy. Here R² is taken about zero, not the mean, so a
//!    converged G₂₂ ≈ 0 (L·G^V level at F∞) still passes.
//!
//! Solvent is implicit (McMillan–Mayer), so G₂₂ is the solvent-averaged
//! integral at fixed solvent chemical potential; in the dilute limit it is
//! −2B₂₂ (Hill 1955, 10.1063/1.1742068, Eq. 89, p. 634).

use super::rdf::{ball, shell, DR};
use super::RdfCurve;
use std::f64::consts::PI;

/// Narrowest range of sphere diameters L, in Å, the extrapolation trusts.
const MIN_FIT_WIDTH: f64 = 10.0;
/// Least share of L·G^V(L) the fitted line explains for a converged G₂₂.
const MIN_R2: f64 = 0.99;
/// Deviations of L·G^V below this, in Å⁴, are physically nil: 1 Å⁴ at
/// L = 10 Å is G₂₂ = 0.1 Å³, against protein values of 10³–10⁶ Å³.
const NOISE_FLOOR: f64 = 1.0;

/// Kirkwood–Buff integral of the bodies, in Å³.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct G22 {
    /// Closed-box corrected and extrapolated to infinite volume.
    pub value: f64,
    /// Raw running integral to L/2, which a closed box biases.
    pub at_half_box: f64,
    pub fit: Fit,
}

/// Range of sphere diameters L, in Å, of the line through L·G^V(L) whose
/// slope is G₂₂.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fit {
    pub from: f64,
    pub to: f64,
    /// 1 − Σresidual²/Σ(L·G^V)², taken about zero rather than the mean:
    /// G₂₂ ≈ 0 leaves L·G^V level at F∞, which is still a line.
    pub r2: f64,
}

/// A quantity along the pair distance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Along {
    G,
    /// w(r) = −ln g in kT; bins with g = 0 are left out.
    PotentialOfMeanForce,
    /// (g − 1)r², whose area is G₂₂/4π.
    WeightedH,
    /// Other bodies within r of one body.
    Coordination,
    /// 4π∫₀^R (g − 1)r² dr of the uncorrected g.
    RunningG22,
    /// Closed-box corrected G^V(L) of a sphere of diameter L, with L as r.
    FiniteVolumeG22,
}

/// g(r) of N bodies in a cubic box, read as Kirkwood–Buff integrals. The
/// curve must have `PairCorrelation`'s bins: width `DR`, from r = 0.
pub struct RdfAnalysis {
    curve: RdfCurve,
    n: f64,
    volume: f64,
    raw: Running,
    finite_volume: Vec<f64>,
}

impl RdfAnalysis {
    pub fn new(curve: RdfCurve, n_bodies: usize, box_length: f64) -> Self {
        let (n, volume) = (n_bodies as f64, box_length.powi(3));
        let raw = running(&curve.r, curve.g.iter().map(|g| g - 1.0), shell);
        let finite_volume = finite_volume(&curve.r, &corrected(&curve, n, volume));
        Self {
            curve,
            n,
            volume,
            raw,
            finite_volume,
        }
    }

    /// `None` when L·G^V(L) is not a line over the outer half of L, that
    /// half is narrower than 10 Å, or the correction leaves no bodies.
    pub fn g22(&self) -> Option<G22> {
        let to = *self.curve.r.last()?;
        let points: Vec<(f64, f64)> = self
            .curve
            .r
            .iter()
            .zip(&self.finite_volume)
            .filter(|(l, _)| **l >= to / 2.0)
            .map(|(&l, g)| (l, l * g))
            .collect();
        let from = points.first()?.0;
        let (slope, r2) = line_fit(&points, NOISE_FLOOR)?;
        (to - from >= MIN_FIT_WIDTH && r2 >= MIN_R2).then_some(G22 {
            value: slope,
            at_half_box: self.raw.total,
            fit: Fit { from, to, r2 },
        })
    }

    /// S(q → 0) = 1 + ρG₂₂, by compressibility.
    pub fn s0(&self) -> Option<f64> {
        self.g22().map(|g22| self.s0_of(&g22))
    }

    fn s0_of(&self, g22: &G22) -> f64 {
        1.0 + self.n / self.volume * g22.value
    }

    /// (r, value) per bin, leaving out bins without a value.
    pub fn along(&self, view: Along) -> Vec<[f64; 2]> {
        self.curve
            .r
            .iter()
            .zip(self.values(view))
            .filter(|(_, v)| v.is_finite())
            .map(|(&r, v)| [r, v])
            .collect()
    }

    /// Every view per bin, under `# key = value` lines with G₂₂ and S(0);
    /// what has no value is left empty.
    pub fn to_csv(&self) -> String {
        self.summary_lines() + &self.table()
    }

    fn summary_lines(&self) -> String {
        let g22 = self.g22();
        let value = |f: fn(&Self, &G22) -> f64| g22.map(|g| f(self, &g));
        let lines = [
            ("G22/Å³", value(|_, g| g.value)),
            ("G22_half_box/Å³", value(|_, g| g.at_half_box)),
            ("fit_from/Å", value(|_, g| g.fit.from)),
            ("fit_to/Å", value(|_, g| g.fit.to)),
            ("fit_R2", value(|_, g| g.fit.r2)),
            ("S0", value(Self::s0_of)),
        ];
        lines
            .iter()
            .map(|(key, v)| {
                format!(
                    "# {key} = {}\n",
                    v.map(|v| v.to_string()).unwrap_or_default()
                )
            })
            .collect()
    }

    fn table(&self) -> String {
        let mut csv = String::from("r/Å,g,g_error,w/kT,h·r²/Å²,N,G22_raw/Å³,G22_V/Å³\n");
        let no_error = vec![f64::NAN; self.curve.r.len()];
        let mut columns = vec![
            self.curve.g.clone(),
            self.curve.error.clone().unwrap_or(no_error),
        ];
        columns.extend(
            [
                Along::PotentialOfMeanForce,
                Along::WeightedH,
                Along::Coordination,
                Along::RunningG22,
                Along::FiniteVolumeG22,
            ]
            .map(|view| self.values(view)),
        );
        for (k, r) in self.curve.r.iter().enumerate() {
            let fields: Vec<String> = columns.iter().map(|c| field(c[k])).collect();
            csv += &format!("{r:.2},{}\n", fields.join(","));
        }
        csv
    }

    /// One value per bin, NaN or infinite where it has none.
    fn values(&self, view: Along) -> Vec<f64> {
        let (g, r) = (&self.curve.g, &self.curve.r);
        match view {
            Along::G => g.clone(),
            Along::PotentialOfMeanForce => g.iter().map(|g| -g.ln()).collect(),
            Along::WeightedH => g.iter().zip(r).map(|(g, r)| (g - 1.0) * r * r).collect(),
            Along::Coordination => {
                let others = (self.n - 1.0) / self.volume;
                running(r, g.iter().map(|g| others * g), shell).at_centres
            }
            Along::RunningG22 => self.raw.at_centres.clone(),
            Along::FiniteVolumeG22 => self.finite_volume.clone(),
        }
    }
}

/// A running integral at each bin centre, and over all bins.
struct Running {
    at_centres: Vec<f64>,
    total: f64,
}

/// ∫₀^r of bin-constant `values` on bins centred at `r`, with `measure(a, b)`
/// the weight of the shell from a to b.
fn running(
    r: &[f64],
    values: impl Iterator<Item = f64>,
    measure: impl Fn(f64, f64) -> f64,
) -> Running {
    let mut total = 0.0;
    let at_centres = values
        .zip(r)
        .map(|(v, &centre)| {
            let a = centre - DR / 2.0;
            let at_centre = total + v * measure(a, centre);
            total += v * measure(a, a + DR);
            at_centre
        })
        .collect();
    Running { at_centres, total }
}

/// g corrected for the closed box (Ganguly & van der Vegt 2013, Eq. 5):
/// bodies gathered within r are missing from the rest of the box. NaN
/// where none are left there, as a bound dimer or noise can make it.
fn corrected(curve: &RdfCurve, n: f64, volume: f64) -> Vec<f64> {
    // GvdV normalise by ρ = N/V, not duello's (N − 1)/V.
    let g: Vec<f64> = curve.g.iter().map(|g| g * (n - 1.0) / n).collect();
    let excess = running(&curve.r, g.iter().map(|g| g - 1.0), shell).at_centres;
    g.iter()
        .zip(&excess)
        .zip(&curve.r)
        .map(|((g, excess), &r)| {
            let bulk = n * (1.0 - ball(r) / volume);
            let outside = bulk - n / volume * excess - 1.0;
            if outside > 0.0 {
                g * bulk / outside
            } else {
                f64::NAN
            }
        })
        .collect()
}

/// Krüger's finite-volume G^V(L) of `g`, for spheres of diameter L at each
/// bin centre. The weight 4πr²(1 − 3x/2 + x³/2) is a polynomial in r, so
/// three running sums give every L at once.
fn finite_volume(r: &[f64], g: &[f64]) -> Vec<f64> {
    let moment = |p: i32| {
        let h = g.iter().map(|g| g - 1.0);
        running(r, h, |a: f64, b: f64| b.powi(p) - a.powi(p)).at_centres
    };
    let (m3, m4, m6) = (moment(3), moment(4), moment(6));
    r.iter()
        .enumerate()
        .map(|(k, &l)| {
            4.0 * PI * (m3[k] / 3.0 - 3.0 * m4[k] / (8.0 * l) + m6[k] / (12.0 * l.powi(3)))
        })
        .collect()
}

/// Six decimals, or empty for NaN and infinities.
fn field(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.6}")
    } else {
        String::new()
    }
}

/// Least-squares slope of y on x, and the share of Σy² it explains (see
/// [`Fit::r2`]); a total spread below `floor` per point counts as none.
fn line_fit(points: &[(f64, f64)], floor: f64) -> Option<(f64, f64)> {
    let n = points.len() as f64;
    if n < 2.0 {
        return None;
    }
    let (mx, my) = points
        .iter()
        .fold((0.0, 0.0), |(x, y), p| (x + p.0 / n, y + p.1 / n));
    let (sxx, sxy) = points.iter().fold((0.0, 0.0), |(xx, xy), p| {
        let dx = p.0 - mx;
        (xx + dx * dx, xy + dx * (p.1 - my))
    });
    let slope = sxy / sxx;
    let (residual, total) = points.iter().fold((0.0, 0.0), |(res, tot), p| {
        let fitted = my + slope * (p.0 - mx);
        (res + (p.1 - fitted).powi(2), tot + p.1 * p.1)
    });
    let r2 = 1.0 - residual / total.max(n * floor * floor);
    (slope.is_finite() && r2.is_finite()).then_some((slope, r2))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `g(r)` on duello's 0.5 Å bins out to L/2.
    fn curve(box_length: f64, g: impl Fn(f64) -> f64) -> RdfCurve {
        let r: Vec<f64> = (0..(box_length / 2.0 / DR) as usize)
            .map(|k| (k as f64 + 0.5) * DR)
            .collect();
        RdfCurve {
            g: r.iter().map(|&r| g(r)).collect(),
            r,
            error: None,
        }
    }

    /// duello normalises pairs by N(N − 1), so an ideal gas has g = 1 at any
    /// N. Read with ρ = N/V, as GvdV does, that is the closed-box 1 − 1/N.
    #[test]
    fn ideal_gas_has_no_excess() {
        let analysis = RdfAnalysis::new(curve(100.0, |_| 1.0), 10, 100.0);
        let g22 = analysis.g22().expect("flat g converges");
        assert!(g22.value.abs() < 1e-6, "G22 = {}", g22.value);
        approx::assert_relative_eq!(analysis.s0().unwrap(), 1.0, epsilon = 1e-9);
    }

    /// Excluded sphere of the step g below, 4π·10³/3 Å³.
    const EXCLUDED: f64 = -4188.790205;

    fn step(r: f64) -> f64 {
        if r < 10.0 {
            0.0
        } else {
            1.0
        }
    }

    /// An open system's step g: the raw running integral reaches minus the
    /// excluded volume at σ and keeps it.
    #[test]
    fn raw_running_integral_of_a_step_is_minus_the_excluded_volume() {
        let analysis = RdfAnalysis::new(curve(100.0, step), 100, 100.0);
        let raw = analysis.along(Along::RunningG22);
        for [r, g] in raw.iter().filter(|p| p[0] > 10.0) {
            assert!((g / EXCLUDED - 1.0).abs() < 1e-9, "G22({r}) = {g}");
        }
    }

    /// In a closed box the N − 1 others fill V − v, so beyond σ the step
    /// reads g = V/(V − v) and the raw integral drifts to about half of −v
    /// by L/2. Correcting and extrapolating recovers −v; the 1 % allows the
    /// O(1/L²) term the line through L·G^V leaves out.
    #[test]
    fn closed_box_step_recovers_the_excluded_volume() {
        let volume = 200.0_f64.powi(3);
        let enriched = volume / (volume + EXCLUDED);
        let analysis = RdfAnalysis::new(curve(200.0, |r| step(r) * enriched), 100, 200.0);
        let g22 = analysis.g22().unwrap();
        approx::assert_relative_eq!(g22.value, EXCLUDED, max_relative = 0.01);
        assert!(g22.at_half_box / EXCLUDED < 0.6, "raw {}", g22.at_half_box);
    }

    /// The one other body of N = 2 is uniform in V = 10⁶ Å³, so a sphere of
    /// radius r holds 4πr³/3 · 10⁻⁶ of it.
    #[test]
    fn coordination_counts_the_other_bodies_within_r() {
        let analysis = RdfAnalysis::new(curve(100.0, |_| 1.0), 2, 100.0);
        for [r, n] in analysis.along(Along::Coordination) {
            approx::assert_relative_eq!(n, 4.0 / 3.0 * PI * r.powi(3) * 1e-6, max_relative = 1e-9);
        }
    }

    #[test]
    fn potential_of_mean_force_is_minus_ln_g_where_g_is_positive() {
        let g = |r| step(r) / std::f64::consts::E;
        let analysis = RdfAnalysis::new(curve(100.0, g), 100, 100.0);
        let w = analysis.along(Along::PotentialOfMeanForce);
        assert_eq!(w.first().unwrap()[0], 10.25, "g = 0 bins are skipped");
        assert!(w.iter().all(|[_, w]| (w - 1.0).abs() < 1e-12));
    }

    #[test]
    fn g_and_weighted_h_follow_g() {
        let analysis = RdfAnalysis::new(curve(100.0, step), 100, 100.0);
        let g = analysis.along(Along::G);
        assert_eq!((g[0], g[20]), ([0.25, 0.0], [10.25, 1.0]));
        let h = analysis.along(Along::WeightedH);
        assert_eq!((h[1], h[20]), ([0.75, -0.5625], [10.25, 0.0]));
    }

    /// Contact attraction that cancels the core: G₂₂ = 0 while L·G^V levels
    /// off at F∞ ≠ 0, a line R² about its mean would reject as flat. The
    /// bump on 5–10 Å holds the 4π·5³/3 Å³ the core lacks, so the box needs
    /// no correction.
    #[test]
    fn cancelling_contact_gives_a_converged_zero() {
        let bump = 1.0 + 125.0 / (1000.0 - 125.0);
        let g = |r| {
            if r < 5.0 {
                0.0
            } else if r < 10.0 {
                bump
            } else {
                1.0
            }
        };
        let g22 = RdfAnalysis::new(curve(200.0, g), 100, 200.0)
            .g22()
            .expect("converged");
        assert!(g22.value.abs() < 1.0, "G22 = {}", g22.value);
    }

    /// A dimer that never parts, counted 20 % high as noise might: past the
    /// peak the correction would leave fewer than no bodies outside and flip
    /// the sign of g, so there is no G₂₂.
    #[test]
    fn bound_dimer_has_no_g22() {
        let volume = 100.0_f64.powi(3);
        let g = |r| {
            if r < 10.0 {
                1.2 * volume / ball(10.0)
            } else {
                0.0
            }
        };
        let analysis = RdfAnalysis::new(curve(100.0, g), 2, 100.0);
        assert_eq!(analysis.g22(), None);
    }

    /// L/2 = 15 Å leaves 7.5 Å to fit over, too little to call converged.
    #[test]
    fn short_box_has_no_g22() {
        let analysis = RdfAnalysis::new(curve(30.0, |_| 1.0), 100, 30.0);
        assert_eq!(analysis.g22(), None);
        assert_eq!(analysis.s0(), None);
    }

    /// Structure that never decays keeps L·G^V(L) off any line.
    #[test]
    fn undamped_structure_has_no_g22() {
        let analysis = RdfAnalysis::new(curve(100.0, |r| 1.0 + 0.5 * r.sin()), 100, 100.0);
        assert_eq!(analysis.g22(), None);
    }

    /// Value of the `# key = value` line, `None` when the line is missing.
    fn summary<'a>(csv: &'a str, key: &str) -> Option<&'a str> {
        csv.lines()
            .find_map(|l| l.strip_prefix(&format!("# {key} =")))
            .map(str::trim)
    }

    #[test]
    fn csv_has_every_view_per_bin_under_the_summary() {
        let volume = 200.0_f64.powi(3);
        let g = |r| step(r) * volume / (volume + EXCLUDED);
        let analysis = RdfAnalysis::new(curve(200.0, g), 100, 200.0);
        let csv = analysis.to_csv();
        let g22 = analysis.g22().unwrap();
        let read = |key| summary(&csv, key).unwrap().parse::<f64>().unwrap();
        assert_eq!(read("G22/Å³"), g22.value);
        assert_eq!(read("G22_half_box/Å³"), g22.at_half_box);
        assert_eq!(read("fit_from/Å"), g22.fit.from);
        assert_eq!(read("fit_to/Å"), g22.fit.to);
        assert_eq!(read("fit_R2"), g22.fit.r2);
        assert_eq!(read("S0"), analysis.s0().unwrap());

        let mut table = csv.lines().skip_while(|l| l.starts_with('#'));
        assert_eq!(
            table.next(),
            Some("r/Å,g,g_error,w/kT,h·r²/Å²,N,G22_raw/Å³,G22_V/Å³")
        );
        let rows: Vec<&str> = table.collect();
        assert_eq!(rows.len(), 200, "0.5 Å bins to L/2 = 100 Å");
        assert!(rows[0].starts_with("0.25,0.000000,,,"), "{}", rows[0]);
        assert!(!csv.contains("NaN") && !csv.contains("inf"));
    }

    #[test]
    fn csv_leaves_an_unconverged_g22_empty() {
        let csv = RdfAnalysis::new(curve(30.0, |_| 1.0), 100, 30.0).to_csv();
        assert_eq!(summary(&csv, "G22/Å³"), Some(""));
        assert_eq!(summary(&csv, "S0"), Some(""));
        assert!(!csv.contains("NaN"));
    }
}
