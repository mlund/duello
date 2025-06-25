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

use physical_constants::MOLAR_GAS_CONSTANT;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Neg};

/// Structure to store energy samples and used mainly as a helper when integrating results
#[derive(Debug, Default, Clone)]
pub struct Sample {
    /// Number of samples
    n: u64,
    /// Thermal energy, RT in kJ/mol
    thermal_energy: f64,
    /// Boltzmann weighted energy, U * g x exp(-U/kT)
    mean_energy: f64,
    /// Boltzmann weighted squared energy, U^2 * g x exp(-U/kT)
    mean_squared_energy: f64,
    /// Boltzmann factored energy, g x exp(-U/kT)
    exp_energy: f64,
    /// Boltzmann factored energy minus one, g x exp(-U/kT) - 1
    exp_energy_m1: f64,
    /// Degeneracy of the state, g
    degeneracy: f64,
}

impl Sample {
    /// New from energy in kJ/mol and temperature in K
    pub fn new(energy: f64, temperature: f64, degeneracy: f64) -> Self {
        const KJ_PER_J: f64 = 1e-3;
        let thermal_energy = MOLAR_GAS_CONSTANT * temperature * KJ_PER_J; // kJ/mol
        let exp_energy = degeneracy * (-energy / thermal_energy).exp();
        let exp_energy_m1 = degeneracy * (-energy / thermal_energy).exp_m1();
        Self {
            n: 1,
            thermal_energy,
            mean_energy: energy * exp_energy,
            mean_squared_energy: energy.powi(2) * exp_energy,
            exp_energy,
            exp_energy_m1,
            degeneracy,
        }
    }
    /// Thermal energy (kJ/mol)
    pub fn thermal_energy(&self) -> f64 {
        self.thermal_energy
    }
    /// Mean energy (kJ/mol)
    pub fn mean_energy(&self) -> f64 {
        self.mean_energy / self.exp_energy
    }
    /// Mean squared energy (kJ/mol)^2
    pub fn mean_squared_energy(&self) -> f64 {
        self.mean_squared_energy / self.exp_energy
    }
    /// Heat capacity C/R
    pub fn heat_capacity(&self) -> f64 {
        (self.mean_squared_energy() - self.mean_energy().powi(2)) / self.thermal_energy.powi(2)
    }
    /// Free energy (kJ / mol)
    pub fn free_energy(&self) -> f64 {
        (self.exp_energy / self.degeneracy as f64).ln().neg() * self.thermal_energy
    }
    /// Mean <exp(-U/kT)-1>
    pub fn mean_exp_energy_m1(&self) -> f64 {
        self.exp_energy_m1 / self.degeneracy as f64
    }
    /// Number of samples
    pub fn n(&self) -> u64 {
        self.n
    }
}

impl std::fmt::Display for Sample {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "n: {}, ⟨U⟩: {:.3} kJ/mol, C/𝑘𝐵: {:.3}, -𝑘𝑇⟨exp(-𝛽U)⟩: {:.3} kJ/mol, ⟨exp(-𝛽U)-1⟩: {:.3}, g = {:.3}",
            self.n,
            self.mean_energy(),
            self.heat_capacity(),
            self.free_energy(),
            self.mean_exp_energy_m1(),
            self.degeneracy,
        )
    }
}

impl Sum for Sample {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Sample::default(), |sum, s| sum + s)
    }
}

impl Add for Sample {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            n: self.n + other.n,
            thermal_energy: f64::max(self.thermal_energy, other.thermal_energy),
            mean_energy: self.mean_energy + other.mean_energy,
            mean_squared_energy: self.mean_squared_energy + other.mean_squared_energy,
            exp_energy: self.exp_energy + other.exp_energy,
            exp_energy_m1: self.exp_energy_m1 + other.exp_energy_m1,
            degeneracy: self.degeneracy + other.degeneracy,
        }
    }
}

impl AddAssign for Sample {
    fn add_assign(&mut self, other: Self) {
        self.n += other.n;
        self.mean_energy += other.mean_energy;
        self.mean_squared_energy += other.mean_squared_energy;
        self.exp_energy += other.exp_energy;
        self.exp_energy_m1 += other.exp_energy_m1;
        self.thermal_energy = f64::max(self.thermal_energy, other.thermal_energy);
        self.degeneracy += other.degeneracy;
    }
}
