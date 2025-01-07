use crate::Vector3;
use chemfiles::Frame;
use faunus::topology::AtomKind;
use itertools::Itertools;
use nalgebra::Matrix3;
use std::{
    fmt::{self, Display, Formatter},
    path::PathBuf,
};

/// Ancient AAM file format from Faunus
#[derive(Debug, Default)]
pub struct AminoAcidModelRecord {
    pub name: String,
    pub pos: Vector3<f64>,
    pub charge: f64,
    pub mass: f64,
    pub radius: f64,
}

impl AminoAcidModelRecord {
    /// Create from space-separated text record (name, _, x, y, z, charge, mass, radius)
    pub fn from_line(text: &str) -> Self {
        let mut parts = text.split_whitespace();
        assert!(parts.clone().count() == 8);
        let name = parts.next().unwrap().to_string();
        parts.next(); // skip the second field
        let pos = Vector3::new(
            parts.next().unwrap().parse().unwrap(),
            parts.next().unwrap().parse().unwrap(),
            parts.next().unwrap().parse().unwrap(),
        );
        let (charge, mass, radius) = parts.map(|i| i.parse().unwrap()).next_tuple().unwrap();
        Self {
            name,
            pos,
            charge,
            mass,
            radius,
        }
    }
}

/// Ad hoc molecular structure containing atoms with positions, masses, charges, and radii
#[derive(Debug, Clone)]
pub struct Structure {
    /// Particle positions
    pub pos: Vec<Vector3<f64>>,
    /// Particle masses
    pub masses: Vec<f64>,
    /// Particle charges
    pub charges: Vec<f64>,
    /// Particle radii
    pub radii: Vec<f64>,
    /// Atom kind ids
    pub atom_ids: Vec<usize>,
}

/// Parse a single line from an XYZ file
fn from_xyz_line(line: &str) -> anyhow::Result<(String, Vector3<f64>)> {
    let mut parts = line.split_whitespace();
    if parts.clone().count() != 4 {
        anyhow::bail!("Expected 4 columns in XYZ file: name x y z");
    }
    let name = parts.next().unwrap().to_string();
    let x = parts.next().unwrap().parse::<f64>()?;
    let y = parts.next().unwrap().parse::<f64>()?;
    let z = parts.next().unwrap().parse::<f64>()?;
    Ok((name, Vector3::new(x, y, z)))
}

impl Structure {
    /// Constructs a new structure from an XYZ file, centering the structure at the origin
    pub fn from_xyz(path: &PathBuf, atomkinds: &[AtomKind]) -> Self {
        let nxyz: Vec<(String, Vector3<f64>)> = std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .skip(2) // skip header
            .map(from_xyz_line)
            .map(|r| r.unwrap())
            .collect();

        let atom_ids = nxyz
            .iter()
            .map(|(name, _)| {
                atomkinds
                    .iter()
                    .position(|j| j.name() == name)
                    .unwrap_or_else(|| panic!("Unknown atom name in XYZ file: {}", name))
            })
            .collect();

        let masses = nxyz
            .iter()
            .map(|(name, _)| atomkinds.iter().find(|i| i.name() == name).unwrap().mass())
            .collect::<Vec<f64>>();

        let charges = nxyz
            .iter()
            .map(|(name, _)| {
                atomkinds
                    .iter()
                    .find(|i| i.name() == name)
                    .unwrap()
                    .charge()
            })
            .collect::<Vec<f64>>();

        let radii = nxyz
            .iter()
            .map(|(name, _)| {
                atomkinds
                    .iter()
                    .find(|i| i.name() == name)
                    .unwrap()
                    .sigma()
                    .unwrap_or(0.0)
            })
            .collect();
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
        structure
    }
    /// Constructs a new structure from a Faunus AAM file
    pub fn from_aam(path: &PathBuf, atomkinds: &[AtomKind]) -> Self {
        let aam: Vec<AminoAcidModelRecord> = std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .skip(1) // skip header
            .map(AminoAcidModelRecord::from_line)
            .collect();

        let atom_ids = aam
            .iter()
            .map(|i| {
                atomkinds
                    .iter()
                    .position(|j| j.name() == i.name)
                    .unwrap_or_else(|| panic!("Unknown atom name in AAM file: {}", i.name))
            })
            .collect();

        let mut structure = Self {
            pos: aam.iter().map(|i| i.pos).collect(),
            masses: aam.iter().map(|i| i.mass).collect(),
            charges: aam.iter().map(|i| i.charge).collect(),
            radii: aam.iter().map(|i| i.radius).collect(),
            atom_ids,
        };
        let center = structure.mass_center();
        structure.translate(&-center); // translate to 0,0,0
        structure
    }

    /// Constructs a new structure from a chemfiles trajectory file
    pub fn from_chemfiles(path: &PathBuf, atomkinds: &[AtomKind]) -> Self {
        let mut traj = chemfiles::Trajectory::open(path, 'r').unwrap();
        let mut frame = Frame::new();
        traj.read(&mut frame).unwrap();
        let masses = frame
            .iter_atoms()
            .map(|atom| atom.mass())
            .collect::<Vec<f64>>();
        let positions = frame
            .positions()
            .iter()
            .map(to_vector3)
            .collect::<Vec<Vector3<f64>>>();
        let atom_ids = frame
            .iter_atoms()
            .map(|atom| {
                atomkinds
                    .iter()
                    .position(|kind| kind.name() == atom.name())
                    .unwrap_or_else(|| panic!("Unknown atom name in structure file: {:?}", atom))
            })
            .collect::<Vec<usize>>();
        let mut structure = Self {
            pos: positions,
            masses,
            charges: vec![0.0; frame.size()],
            radii: vec![0.0; frame.size()],
            atom_ids,
        };
        let center = structure.mass_center();
        structure.translate(&-center);
        structure
    }

    /// Returns the center of mass of the structure
    pub fn mass_center(&self) -> Vector3<f64> {
        let total_mass: f64 = self.masses.iter().sum();
        self.pos
            .iter()
            .zip(&self.masses)
            .map(|(pos, mass)| pos.scale(*mass))
            .fold(Vector3::<f64>::zeros(), |sum, i| sum + i)
            / total_mass
    }
    /// Translates the coordinates by a displacement vector
    pub fn translate(&mut self, displacement: &Vector3<f64>) {
        self.transform(|pos| pos + displacement);
    }

    /// Transform the coordinates using a function
    pub fn transform(&mut self, f: impl Fn(Vector3<f64>) -> Vector3<f64>) {
        self.pos.iter_mut().for_each(|pos| *pos = f(*pos));
    }

    /// Net charge of the structure
    pub fn net_charge(&self) -> f64 {
        self.charges.iter().sum()
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

/// Calculates the mass center of a set of point masses.
///
/// The mass center is computed from positions, 𝒑₁,…,𝒑ₙ, as 𝑪 = ∑ mᵢ𝒑ᵢ / ∑ mᵢ.
///
pub fn mass_center(
    positions: impl IntoIterator<Item = Vector3<f64>>,
    masses: impl IntoIterator<Item = f64>,
) -> Vector3<f64> {
    let mut total_mass: f64 = 0.0;
    let mut c = Vector3::<f64>::zeros();
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
    positions: impl IntoIterator<Item = Vector3<f64>>,
    masses: impl IntoIterator<Item = f64>,
    center: Option<Vector3<f64>>,
) -> Matrix3<f64> {
    positions
        .into_iter()
        .map(|r| r - center.unwrap_or(Vector3::<f64>::zeros()))
        .zip(masses)
        .map(|(r, m)| m * (r.norm_squared() * Matrix3::<f64>::identity() - r * r.transpose()))
        .sum()
}

/// Principal moments of inertia from the inertia tensor
pub fn principal_moments_of_inertia(inertia: &Matrix3<f64>) -> Vector3<f64> {
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
pub fn gyration_tensor(positions: impl IntoIterator<Item = Vector3<f64>> + Clone) -> Matrix3<f64> {
    let c: Vector3<f64> = positions.clone().into_iter().sum();
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

/// Converts a slice of f64 to a nalgebra Vector3
fn to_vector3(pos: &[f64; 3]) -> Vector3<f64> {
    Vector3::<f64>::new(pos[0], pos[1], pos[2])
}
