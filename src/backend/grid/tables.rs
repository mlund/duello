//! CPU-side tables the grid kernels index: map slots, charged beads, reach.

use crate::backend::gpu::pack_bead;
use crate::energy::{CoulombParams, SplinedPotentials};
use crate::structure::Structure;

/// One map per atom type present in B.
pub(super) struct SlotTable {
    /// B atom type of each slot.
    pub types: Vec<u32>,
    /// Slot of each B bead.
    pub of_bead: Vec<u32>,
}

impl SlotTable {
    pub fn new(b: &Structure) -> Self {
        let types = b.unique_atom_types();
        let of_bead = b
            .atom_ids
            .iter()
            .map(|t| types.binary_search(t).unwrap() as u32)
            .collect();
        Self {
            types: types.into_iter().map(|t| t as u32).collect(),
            of_bead,
        }
    }
}

/// Distance beyond which every A–B type pair's short-range energy vanishes.
pub(super) fn effective_range(splines: &SplinedPotentials, a: &Structure, b: &Structure) -> f64 {
    itertools::iproduct!(a.unique_atom_types(), b.unique_atom_types())
        .map(|(ta, tb)| splines.vanishes_beyond(ta, tb))
        .fold(0.0, f64::max)
}

/// Beads of `s` with a nonzero prefactor towards some type in `other`, as
/// (x, y, z, type); the rest never enter the Yukawa sum.
pub(super) fn charged_beads(
    s: &Structure,
    other: &Structure,
    coulomb: &CoulombParams,
    n: usize,
) -> Vec<[f32; 4]> {
    let other_types = other.unique_atom_types();
    let charged = |t: usize| {
        other_types
            .iter()
            .any(|&o| coulomb.prefactors[t * n + o] != 0.0)
    };
    s.pos
        .iter()
        .zip(&s.atom_ids)
        .filter(|&(_, &t)| charged(t))
        .map(|(p, &t)| pack_bead(p, t as f32))
        .collect()
}
