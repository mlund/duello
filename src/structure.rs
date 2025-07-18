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
use anyhow::{anyhow, Context, Result};
use faunus::topology::{AtomKind, FindByName};
use itertools::Itertools;
use nalgebra::Matrix3;
use std::{
    fmt::{self, Display, Formatter},
    path::PathBuf,
};

/// Ad hoc molecular structure containing atoms with positions, masses, charges, and radii
#[derive(Debug, Clone)]
pub struct Structure {
    /// Particle positions
    pub pos: Vec<Vector3>,
    /// Particle masses
    pub masses: Vec<f64>,
    /// Particle charges
    pub charges: Vec<f64>,
    /// Particle radii
    pub radii: Vec<f64>,
    /// Atom kind ids
    pub atom_ids: Vec<usize>,
}

impl Structure {
    /// Constructs a new structure from an XYZ file, centering the structure at the origin
    pub fn from_xyz(path: &PathBuf, atomkinds: &[AtomKind]) -> Result<Self> {
        let nxyz: Vec<(String, Vector3)> = std::fs::read_to_string(path)
            .context(format!("Could not read XYZ file {}", path.display()))?
            .lines()
            .skip(2) // skip header
            .map(from_xyz_line)
            .try_collect()?;

        let atom_ids = nxyz
            .iter()
            .map(|(name, _)| {
                atomkinds
                    .find_name(name)
                    .map(|kind| kind.id())
                    .context(format!("Unknown atom name in structure file: {name:?}"))
            })
            .try_collect()?;

        let masses = nxyz
            .iter()
            .map(|(name, _)| {
                atomkinds
                    .find_name(name)
                    .map(|kind| kind.mass())
                    .context(format!("Unknown atom name in XYZ file: {name}"))
            })
            .try_collect()?;

        let charges = nxyz
            .iter()
            .map(|(name, _)| {
                atomkinds
                    .find_name(name)
                    .map(|i| i.charge())
                    .context(format!("Unknown atom name in XYZ file: {name}"))
            })
            .try_collect()?;

        let radii = nxyz
            .iter()
            .map(|(name, _)| {
                atomkinds
                    .find_name(name)
                    .map(|i| i.sigma().unwrap_or(0.0) * 0.5)
                    .context(format!("Unknown atom name in XYZ file: {name}"))
            })
            .try_collect()?;
        let mut structure = Self {
            pos: nxyz.iter().map(|(_, pos)| *pos).collect(),
            masses,
            charges,
            radii,
            atom_ids,
        };
        let center = structure.mass_center();
        structure.translate(&-center); // translate to 0,0,0
        debug!("Read {}: {}", path.display(), structure);
        Ok(structure)
    }

    /// Write to XYZ file
    pub fn to_xyz(&self, stream: &mut impl std::io::Write, atomkinds: &[AtomKind]) -> Result<()> {
        writeln!(stream, "{}", self.pos.len())?; // number of atoms
        writeln!(stream, "Generated by Duello")?; // comment line
        for (i, pos) in self.pos.iter().enumerate() {
            let atom_id = self.atom_ids[i];
            let name = atomkinds[atom_id].name();
            writeln!(stream, "{} {:.3} {:.3} {:.3}", name, pos.x, pos.y, pos.z)?;
        }
        Ok(())
    }

    /// Constructs a new structure from a Faunus AAM file
    pub fn from_aam(path: &PathBuf, atomkinds: &[AtomKind]) -> Result<Self> {
        let aam: Vec<AminoAcidModelRecord> = std::fs::read_to_string(path)?
            .lines()
            .skip(1) // skip header
            .map(AminoAcidModelRecord::from_line)
            .try_collect()?;

        let atom_ids = aam
            .iter()
            .map(|i| {
                atomkinds
                    .find_name(&i.name)
                    .map(|kind| kind.id())
                    .context(format!("Unknown atom name in structure file: {:?}", i.name))
            })
            .try_collect()?;

        let mut structure = Self {
            pos: aam.iter().map(|i| i.pos).collect(),
            masses: aam.iter().map(|i| i.mass).collect(),
            charges: aam.iter().map(|i| i.charge).collect(),
            radii: aam.iter().map(|i| i.radius).collect(),
            atom_ids,
        };
        let center = structure.mass_center();
        structure.translate(&-center); // translate to 0,0,0
        Ok(structure)
    }

    /// Returns the center of mass of the structure
    pub fn mass_center(&self) -> Vector3 {
        self.pos
            .iter()
            .zip(&self.masses)
            .map(|(pos, mass)| pos.scale(*mass))
            .fold(Vector3::zeros(), |sum, i| sum + i)
            / self.total_mass()
    }
    /// Translates the coordinates by a displacement vector
    pub fn translate(&mut self, displacement: &Vector3) {
        self.transform(|pos| pos + displacement);
    }

    /// Transform the coordinates using a function
    pub fn transform(&mut self, func: impl Fn(Vector3) -> Vector3) {
        self.pos.iter_mut().for_each(|pos| *pos = func(*pos));
    }

    /// Net charge of the structure
    pub fn net_charge(&self) -> f64 {
        self.charges.iter().sum()
    }

    /// Molecular dipole moment of the structure with respect to the mass center
    ///
    /// The dipole moment is calculated as
    /// 𝒎 = ∑ 𝒓ᵢ𝑞ᵢ where 𝒓ᵢ = 𝒑ᵢ - 𝑪 and 𝑪 is the mass center.
    /// Note that if the molecule is not neutral, the dipole moment depends on the
    /// choice of origin, here the mass center.
    pub fn dipole_moment(&self) -> Vector3 {
        let center = self.mass_center();
        self.pos
            .iter()
            .zip(&self.charges)
            .map(|(pos, charge)| (pos - center).scale(*charge))
            .fold(Vector3::zeros(), |sum, i| sum + i)
    }

    /// Total mass of the structure
    pub fn total_mass(&self) -> f64 {
        self.masses.iter().sum()
    }

    /// Calculates the inertia tensor of the structure
    ///
    /// The inertia tensor is computed from positions, 𝒑ᵢ,…𝒑ₙ, with
    /// respect to a reference point, 𝒑ᵣ, here the center of mass.
    ///
    /// 𝐈 = ∑ mᵢ(|𝒓ᵢ|²𝑰₃ - 𝒓ᵢ𝒓ᵢᵀ) where 𝒓ᵢ = 𝒑ᵢ - 𝒑ᵣ.
    ///
    pub fn inertia_tensor(&self) -> nalgebra::Matrix3<f64> {
        let center = self.mass_center();
        inertia_tensor(
            self.pos.iter().cloned(),
            self.masses.iter().cloned(),
            Some(center),
        )
    }
}

impl std::ops::Add for Structure {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        let mut s = self.clone();
        s.pos.extend(other.pos);
        s.masses.extend(other.masses);
        s.charges.extend(other.charges);
        s.radii.extend(other.radii);
        s.atom_ids.extend(other.atom_ids);
        s
    }
}

/// Calculates the mass center of a set of point masses.
///
/// The mass center is computed from positions, 𝒑₁,…,𝒑ₙ, as 𝑪 = ∑ mᵢ𝒑ᵢ / ∑ mᵢ.
///
pub fn mass_center(
    positions: impl IntoIterator<Item = Vector3>,
    masses: impl IntoIterator<Item = f64>,
) -> Vector3 {
    let mut total_mass: f64 = 0.0;
    let mut c = Vector3::zeros();
    for (r, m) in positions.into_iter().zip(masses) {
        total_mass += m;
        c += m * r;
    }
    assert!(total_mass > 0.0, "Total mass must be positive");
    c / total_mass
}

/// Calculates the moment of inertia tensor of a set of point masses.
///
/// The inertia tensor is computed from positions, 𝒑₁,…,𝒑ₙ, with
///
/// 𝐈 = ∑ mᵢ(|𝒓ᵢ|²𝑰₃ - 𝒓ᵢ𝒓ᵢᵀ) where 𝑰₃ is the 3×3 identity matrix
/// and 𝒓ᵢ = 𝒑ᵢ - 𝑪.
/// The center, 𝑪, is optional and should normally be set to the
/// mass center. 𝑪 defaults to (0,0,0).
///
/// # Examples:
/// ~~~
/// use nalgebra::Vector3;
/// use duello::structure::{inertia_tensor, mass_center};
///
/// let masses: Vec<f64> = vec![1.0, 1.0, 2.0];
/// let pos = [
///    Vector3::new(0.0, 0.0, 0.0),
///    Vector3::new(1.0, 0.0, 0.0),
///    Vector3::new(0.0, 1.0, 0.0),
/// ];
/// let center = mass_center(pos, masses.iter().cloned());
/// let inertia = inertia_tensor(pos, masses, Some(center));
/// let principal_moments = inertia.symmetric_eigenvalues();
///
/// approx::assert_relative_eq!(principal_moments.x, 1.3903882032022075);
/// approx::assert_relative_eq!(principal_moments.y, 0.35961179679779254);
/// approx::assert_relative_eq!(principal_moments.z, 1.75);
/// ~~~
///
/// # Further Reading:
///
/// - <https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor>
///
pub fn inertia_tensor(
    positions: impl IntoIterator<Item = Vector3>,
    masses: impl IntoIterator<Item = f64>,
    center: Option<Vector3>,
) -> Matrix3<f64> {
    positions
        .into_iter()
        .map(|r| r - center.unwrap_or(Vector3::zeros()))
        .zip(masses)
        .map(|(r, m)| m * (r.norm_squared() * Matrix3::<f64>::identity() - r * r.transpose()))
        .sum()
}

/// Principal moments of inertia from the inertia tensor
pub fn principal_moments_of_inertia(inertia: &Matrix3<f64>) -> Vector3 {
    inertia.symmetric_eigenvalues()
}

/// Calculates the gyration tensor of a set of positions.
///
/// The gyration tensor is computed from positions, 𝒑₁,…,𝒑ₙ, with
/// respect to the geometric center, 𝑪:
///
/// 𝐆 = ∑ 𝒓ᵢ𝒓ᵢᵀ where 𝒓ᵢ = 𝒑ᵢ - 𝑪.
///
/// # Further Reading
///
/// - <https://en.wikipedia.org/wiki/Gyration_tensor>
///
pub fn gyration_tensor(positions: impl IntoIterator<Item = Vector3> + Clone) -> Matrix3<f64> {
    let c: Vector3 = positions.clone().into_iter().sum();
    positions
        .into_iter()
        .map(|p| p - c)
        .map(|r| r * r.transpose())
        .sum()
}

/// Display number of atoms, mass center etc.
impl Display for Structure {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "𝑁={}, ∑𝑞ᵢ={:.2}𝑒, ∑𝑚ᵢ={:.2}",
            self.pos.len(),
            self.net_charge(),
            self.masses.iter().sum::<f64>()
        )
    }
}

/// Parse a single line from an XYZ file
fn from_xyz_line(line: &str) -> Result<(String, Vector3)> {
    let [name, x, y, z] = line
        .split_whitespace()
        .collect_array()
        .ok_or_else(|| anyhow!("Invalid XYZ record: {}", line))?;
    Ok((
        name.to_string(),
        Vector3::new(x.parse()?, y.parse()?, z.parse()?),
    ))
}

/// Write single ATOM record in PQR file stream
pub fn pqr_write_atom(
    stream: &mut impl std::io::Write,
    atom_id: usize,
    pos: &Vector3,
    charge: f64,
    radius: f64,
) -> Result<()> {
    writeln!(
        stream,
        "ATOM  {:5} {:4.4} {:4.3}{:5}    {:8.3} {:8.3} {:8.3} {:.3} {:.3}",
        atom_id, "A", "AAA", 1, pos.x, pos.y, pos.z, charge, radius
    )?;
    Ok(())
}

/// Ancient AAM file format from Faunus
#[derive(Debug, Default)]
pub struct AminoAcidModelRecord {
    pub name: String,
    pub pos: Vector3,
    pub charge: f64,
    pub mass: f64,
    pub radius: f64,
}

impl AminoAcidModelRecord {
    pub fn from_line(line: &str) -> Result<Self> {
        let [name, _, x, y, z, charge, mass, radius] = line
            .split_whitespace()
            .collect_array()
            .ok_or_else(|| anyhow!("Invalid AAM record: {}", line))?;

        Ok(Self {
            name: name.to_string(),
            pos: Vector3::new(x.parse()?, y.parse()?, z.parse()?),
            charge: charge.parse()?,
            mass: mass.parse()?,
            radius: radius.parse()?,
        })
    }
}
