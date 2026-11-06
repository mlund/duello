//! Analytic S(q) of a hard-sphere Yukawa fluid: a charged-sphere reference
//! for the simulated S_eff. Solved in the MPB-RMSA (Heinen et al., J. Chem.
//! Phys. 134, 044532, 2011), with the misprint-free equations of Kaieda et
//! al., J. Phys. Chem. B 118, 2014, SI eqs S5–S42.

use faunus::interatomic::coulomb::{Medium, Salt};
use nalgebra::Complex;
use physical_constants::AVOGADRO_CONSTANT;
use std::f64::consts::PI;

/// Typical for globular proteins, cm³/g; Kaieda et al. used measured values.
const PARTIAL_SPECIFIC_VOLUME: f64 = 0.73;

/// Charged hard spheres in 1:1 salt; lengths in Å.
#[derive(Clone, Debug, PartialEq)]
pub struct HsyFluid {
    pub diameter: f64,
    /// Spheres per Å³.
    pub number_density: f64,
    /// Net charge per sphere, e.
    pub charge: f64,
    /// Salt pairs per Å³.
    pub salt_density: f64,
    pub bjerrum_length: f64,
}

impl HsyFluid {
    /// Equivalent sphere of a protein's volume and net charge, in NaCl at
    /// `temperature` (K); molarities in mol/L.
    pub fn protein(
        molar_mass: f64,
        charge: f64,
        protein_molarity: f64,
        salt_molarity: f64,
        temperature: f64,
    ) -> Self {
        let per_cubic_angstrom = AVOGADRO_CONSTANT * 1e-27;
        let volume = molar_mass * PARTIAL_SPECIFIC_VOLUME * 1e24 / AVOGADRO_CONSTANT;
        Self {
            diameter: (6.0 * volume / PI).cbrt(),
            number_density: protein_molarity * per_cubic_angstrom,
            charge,
            salt_density: salt_molarity * per_cubic_angstrom,
            bjerrum_length: Medium::salt_water(temperature, Salt::SodiumChloride, salt_molarity)
                .bjerrum_length(),
        }
    }

    /// One line for reports.
    pub fn describe(&self) -> String {
        let phi = self.number_density * PI * self.diameter.powi(3) / 6.0;
        format!(
            "HSY sphere: σ = {:.1} Å, φ = {:.4}, Z = {:+.1}, salt {:.1} mM",
            self.diameter,
            phi,
            self.charge,
            self.salt_density / (AVOGADRO_CONSTANT * 1e-27) * 1e3
        )
    }

    /// One value per `q` (Å⁻¹); `None` where the rescaling finds no
    /// physical solution (dense and strongly repulsive).
    pub fn structure_factor(&self, q: &[f64]) -> Option<Vec<f64>> {
        let reduced = Reduced::of(self)?;
        let solution = reduced.solve();
        // Q → 0 is 0/0 in the G and H functions; this floor is far below any box q.
        let s: Vec<f64> = q
            .iter()
            .map(|&q| solution.s((q * reduced.sigma).max(1e-3)))
            .collect();
        s.iter().all(|s| s.is_finite()).then_some(s)
    }
}

/// Dimensionless model: diameter σ, volume fraction φ, coupling γ, screening k = κσ.
#[derive(Clone, Copy, Debug)]
struct Reduced {
    sigma: f64,
    phi: f64,
    gamma: f64,
    k: f64,
}

impl Reduced {
    /// MPB corrections (eqs S10, S37–S39): they account for counterions
    /// that penetrate the spheres, which the plain RMSA ignores.
    fn of(fluid: &HsyFluid) -> Option<Self> {
        let sigma = fluid.diameter;
        let phi = fluid.number_density * PI * sigma.powi(3) / 6.0;
        let lb = fluid.bjerrum_length;
        // S37's 1/(1−φ) and S38's √(1−φ) cancel, so k_mod is plain κσ and
        // γ_mod is S10 evaluated at it.
        let k_mod = (lb / sigma
            * (24.0 * phi * fluid.charge.abs() + 8.0 * PI * fluid.salt_density * sigma.powi(3)))
        .sqrt();
        let gamma_mod =
            lb / sigma * k_mod.exp() / (1.0 + k_mod / 2.0).powi(2) * fluid.charge.powi(2);
        let at = |s| Self::rescaled(sigma, phi, gamma_mod, k_mod, s);
        let contact = |s| at(s).solve().contact();
        if contact(1.0) >= 0.0 {
            return Some(at(1.0));
        }
        // RMSA: grow σ at fixed n until g(σ) = 0, as the true core no longer matters.
        let smallest_s = (phi / MAX_RESCALED_PHI).cbrt();
        if smallest_s.is_nan() || smallest_s >= 0.99 {
            return None;
        }
        newton(contact, 0.99, smallest_s).map(at)
    }

    /// Eq S41; s = 1 is eq S39.
    fn rescaled(sigma: f64, phi: f64, gamma_mod: f64, k_mod: f64, s: f64) -> Self {
        let phi = phi / s.powi(3);
        Self {
            sigma: sigma / s,
            phi,
            gamma: gamma_mod * s / (1.0 - phi).powi(2),
            k: k_mod / s - 2.0 * phi.cbrt() * (1.0 - phi).ln(),
        }
    }

    fn solve(&self) -> Solution {
        // Below this the Yukawa algebra is 0/0, yet indistinguishable from PY.
        if self.gamma * (-self.k).exp() < 1e-10 {
            Solution::hard_spheres(self.phi)
        } else {
            Solution::yukawa(self.phi, self.gamma, self.k)
        }
    }
}

/// Coefficients of the Wiener–Hopf factor F(Q) (eqs S17–S20); β = 0 is PY.
struct Solution {
    phi: f64,
    k: f64,
    /// γe^{−k}
    coupling: f64,
    a: f64,
    b: f64,
    beta: f64,
    d: f64,
    p: f64,
    t: f64,
}

/// φ- and k-only quantities shared by eqs S19–S34 and S40.
struct Auxiliary {
    a0: f64,
    b0: f64,
    delta: f64,
    mu: f64,
    p: f64,
    t: f64,
    tau: f64,
    rho: f64,
}

impl Auxiliary {
    fn new(phi: f64, k: f64) -> Self {
        let ek = (-k).exp();
        let mu = 1.0 - (1.0 + k) * ek;
        Self {
            a0: (1.0 + 2.0 * phi) / (1.0 - phi).powi(2),
            b0: -3.0 * phi / (2.0 * (1.0 - phi).powi(2)),
            delta: 12.0 * phi / (k * (1.0 - phi).powi(2)),
            mu,
            p: 12.0 * phi * (mu / (k * k) - 0.5),
            t: 12.0 * phi / k * (1.0 - k - ek),
            tau: 1.0 + 2.0 * phi - 6.0 * phi / k,
            rho: 1.5 * phi + (1.0 - 4.0 * phi) / k,
        }
    }
}

impl Solution {
    fn hard_spheres(phi: f64) -> Self {
        let aux = Auxiliary::new(phi, 1.0);
        Self {
            phi,
            k: 1.0,
            coupling: 0.0,
            a: aux.a0,
            b: aux.b0,
            beta: 0.0,
            d: 0.0,
            p: aux.p,
            t: aux.t,
        }
    }

    /// Eqs S19–S36.
    fn yukawa(phi: f64, gamma: f64, k: f64) -> Self {
        let aux = Auxiliary::new(phi, k);
        let ek = (-k).exp();
        let kk = gamma * ek;
        let beta = quartic_root(phi, k, kk);
        let d_big = k - aux.a0 * aux.p - aux.b0 * aux.t;
        let e_big = -6.0 * phi + aux.delta * (aux.tau * aux.p - aux.rho * aux.t);
        let f_big = -6.0 * phi * (1.0 - ek).powi(2)
            + aux.p * aux.delta * (aux.mu * aux.tau + 3.0 * phi * k * ek)
            - aux.t * aux.delta * (aux.mu * aux.rho - 0.5 * (1.0 - 4.0 * phi) * k * ek);
        let d = ((beta * d_big + kk) * ek + beta * beta * e_big) / (beta * beta * f_big);
        let omega = d * (1.0 + k) - (d - 1.0) * k.exp();
        let scale = aux.delta * beta;
        Self {
            phi,
            k,
            a: aux.a0 + scale * (3.0 * d * phi * k - omega * aux.tau),
            b: aux.b0 + scale * ((1.0 - 4.0 * phi) * k * d / 2.0 + omega * aux.rho),
            beta,
            d,
            p: aux.p,
            t: aux.t,
            coupling: kk,
        }
    }

    /// g(σ⁺), eq S40.
    fn contact(&self) -> f64 {
        let (k, d, beta) = (self.k, self.d, self.beta);
        self.a * (1.0 + beta * self.p) + self.b * (1.0 + beta * self.t) - self.coupling
            + beta * k * (d - 1.0)
            - 6.0 * self.phi * beta * beta * (2.0 * d * (k.cosh() - 1.0) - k.exp())
    }

    /// Eqs S5, S8, S17, S18.
    fn s(&self, q: f64) -> f64 {
        let (sin, cos) = q.sin_cos();
        let g_a = (q * cos - sin) / q.powi(3);
        let g_b = (cos - 1.0) / q.powi(2);
        let h_a = (q * sin + cos - 1.0 - q * q / 2.0) / q.powi(3);
        let h_b = (sin - q) / q.powi(2);
        let (k, d, exp_k) = (self.k, self.d, self.k.exp());
        let u = d * k * (cos + k * sin / q) - (d - 1.0) * k * exp_k;
        let v = d * k * (sin - k * (cos - 1.0) / q) + q * (d - (d - 1.0) * exp_k);
        let tail = self.beta / (k * k + q * q);
        let re = 1.0 - 12.0 * self.phi * (self.a * g_a + self.b * g_b + tail * u);
        let im = -12.0 * self.phi * (self.a * h_a + self.b * h_b + tail * v);
        1.0 / (re * re + im * im)
    }
}

/// Keeps φ* = φ/s³ well inside the fluid range while Newton searches.
const MAX_RESCALED_PHI: f64 = 0.5;

/// Root of `f` in [smallest, 1] by Newton–Raphson (eq S42).
fn newton(f: impl Fn(f64) -> f64, start: f64, smallest: f64) -> Option<f64> {
    const H: f64 = 1e-7;
    let mut s = start;
    for _ in 0..100 {
        let value = f(s);
        if value.abs() < 1e-6 {
            return Some(s);
        }
        let slope = (f(s + H) - f(s - H)) / (2.0 * H);
        let next = s - value / slope;
        if !next.is_finite() {
            return None;
        }
        s = next.clamp(smallest, 1.0);
    }
    None
}

/// The physical root β of eq S33, in closed form (eqs S34, S36).
fn quartic_root(phi: f64, k: f64, kk: f64) -> f64 {
    let ek = (-k).exp();
    let w = 18.0 * phi * phi / ((1.0 - phi).powi(2) * k * k) * (2.0 - k - (2.0 + k) * ek);
    let pre = 6.0 * phi / ((1.0 - phi) * k * k);
    let x = k * ek - pre * (2.0 - 2.0 * k - (2.0 - k * k) * ek) - w;
    let y = k - pre * (2.0 - k * k - 2.0 * (1.0 + k) * ek) - w;
    let xy = x * y;
    let m = (16.0 * kk).powi(2) * phi + 3.0 * kk * (3.0 * x * x + 3.0 * y * y - 2.0 * xy);
    let l = 2.0 * (48.0 * kk * phi * (x - y)).powi(2) - 2.0 * xy.powi(3)
        + 3.0
            * kk
            * phi
            * (2.0 * xy * xy
                + 9.0
                    * (3.0 * x.powi(4) - 4.0 * x.powi(3) * y - 4.0 * x * y.powi(3)
                        + 3.0 * y.powi(4)));
    // L < 0 is common (e.g. dilute), so R is complex while β stays real.
    let root_kl = Complex::new(kk * l / phi, 0.0).sqrt();
    let r = ((m - root_kl) * (2.0 * kk * phi * phi)).cbrt();
    let c = kk * (32.0 * kk * phi + xy) / (18.0 * phi * r);
    let gamma_big = (x / (12.0 * phi)).powi(2) + 2.0 * kk / (9.0 * phi);
    let lambda = r / (6.0 * phi).powi(2) + c;
    let b1 = lambda + gamma_big;
    let b2 = -lambda + 2.0 * gamma_big
        - (kk * (x - y) + x.powi(3) / (48.0 * phi)) / (b1.sqrt() * 18.0 * phi * phi);
    x / (24.0 * phi) + ((b2.sqrt() - b1.sqrt()) / 2.0).re
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const SIGMA: f64 = 30.0;

    fn density(phi: f64) -> f64 {
        6.0 * phi / (PI * SIGMA.powi(3))
    }

    fn uncharged(phi: f64) -> HsyFluid {
        HsyFluid {
            diameter: SIGMA,
            number_density: density(phi),
            charge: 0.0,
            salt_density: 1e-5,
            bjerrum_length: 7.0,
        }
    }

    fn s0(fluid: &HsyFluid) -> f64 {
        fluid.structure_factor(&[1e-4]).unwrap()[0]
    }

    #[test]
    fn uncharged_spheres_follow_percus_yevick() {
        // PY compressibility: (1−φ)⁴/(1+2φ)² = 0.8⁴/1.4² at φ = 0.2
        approx::assert_relative_eq!(s0(&uncharged(0.2)), 0.208_980, max_relative = 1e-4);
    }

    #[test]
    fn dilute_weak_charges_follow_linearised_msa() {
        // MSA to first order in n: c(r) = −βw(r) outside the core, so
        // S(0) − 1 = −n[4πσ³/3 + ∫4πr²βw dr], with
        // βw = λB Z² e^{−κ(r−σ)} / ((1+κσ/2)² r), integrated by hand.
        let (lb, z, kappa) = (7.0, 2.0, 0.1);
        let fluid = HsyFluid {
            charge: z,
            salt_density: kappa * kappa / (8.0 * PI * lb),
            bjerrum_length: lb,
            ..uncharged(1e-4)
        };
        let core = 4.0 * PI * SIGMA.powi(3) / 3.0;
        let yukawa = 4.0 * PI * lb * z * z / (1.0 + kappa * SIGMA / 2.0).powi(2)
            * (SIGMA / kappa + 1.0 / kappa.powi(2));
        let expected = -fluid.number_density * (core + yukawa);
        approx::assert_relative_eq!(s0(&fluid) - 1.0, expected, max_relative = 0.01);
    }

    /// g(r) = 1 + ∫ q (S − 1) sin(qr) dq / (2π² n r), by midpoint rule.
    fn pair_correlation(fluid: &HsyFluid, r: f64) -> f64 {
        let dq = 1e-3;
        let q: Vec<f64> = (0..20_000).map(|i| (i as f64 + 0.5) * dq).collect();
        let s = fluid.structure_factor(&q).unwrap();
        let integral: f64 = q
            .iter()
            .zip(&s)
            .map(|(q, s)| q * (s - 1.0) * (q * r).sin() * dq)
            .sum();
        1.0 + integral / (2.0 * PI * PI * fluid.number_density * r)
    }

    #[test]
    fn strong_repulsion_keeps_pair_correlation_non_negative() {
        // βw(σ) ≈ 12 kT: the plain MSA gives g(σ) < 0 here.
        let fluid = HsyFluid {
            charge: 10.0,
            salt_density: 0.0,
            ..uncharged(0.01)
        };
        approx::assert_abs_diff_eq!(pair_correlation(&fluid, SIGMA / 2.0), 0.0, epsilon = 0.02);
        let lowest = (0..40)
            .map(|i| pair_correlation(&fluid, SIGMA * (1.0 + i as f64 / 20.0)))
            .fold(f64::INFINITY, f64::min);
        assert!(lowest > -0.02, "g(r) dips to {lowest}");
    }

    #[test]
    fn a_protein_maps_to_a_sphere_of_its_volume() {
        // Lysozyme: V = 14300 g/mol × 0.73 cm³/g / N_A = 17334 Å³; σ = (6V/π)^⅓
        let fluid = HsyFluid::protein(14_300.0, 8.0, 0.002, 0.1, 298.15);
        approx::assert_relative_eq!(fluid.diameter, 32.11, max_relative = 1e-3);
        approx::assert_relative_eq!(fluid.number_density, 1.2044e-6, max_relative = 1e-3);
        approx::assert_relative_eq!(fluid.salt_density, 6.0221e-5, max_relative = 1e-3);
        approx::assert_relative_eq!(fluid.bjerrum_length, 7.15, max_relative = 0.01);
        assert_eq!(fluid.charge, 8.0);
    }

    #[test]
    fn a_trace_of_charge_is_still_percus_yevick() {
        let fluid = HsyFluid {
            charge: 1e-13,
            salt_density: 0.0,
            ..uncharged(0.2)
        };
        approx::assert_relative_eq!(s0(&fluid), 0.208_980, max_relative = 1e-4);
    }

    #[test]
    fn an_undefined_fluid_has_no_solution_rather_than_a_panic() {
        let fluid = HsyFluid {
            number_density: f64::NAN,
            charge: 5.0,
            ..uncharged(0.1)
        };
        assert_eq!(fluid.structure_factor(&[0.1]), None);
    }
}
