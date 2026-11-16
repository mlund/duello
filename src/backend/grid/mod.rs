//! Grid backend: A's short-range field precomputed on a Cartesian map.
//!
//! Each pose then costs one trilinear lookup per B bead plus an exact Yukawa
//! sum over charged pairs, instead of a full N_A·N_B pair sum.
//!
//! This is AutoDock's affinity-grid scheme. Goodsell & Olson (1990, p. 196,
//! doi:10.1002/prot.340080302): "The protein is embedded in a three-dimensional
//! grid and a probe atom is sequentially placed at each grid point. [...] An
//! affinity grid is calculated for each type of atom in a typical substrate",
//! and the energy of each substrate atom "is found by trilinear interpolation
//! of the affinity values of the eight grid points surrounding the atom
//! position", so that the cost is "proportional only to the number of atoms in
//! the substrate, not to any function of the number of protein atoms". Morris
//! et al. (2009, p. 2785, doi:10.1002/jcc.21256) describe AutoDock4's grid the
//! same way. Here A is the protein, B the substrate, and B's atom types pick
//! the maps. Unlike AutoDock, electrostatics are not gridded (Goodsell & Olson
//! also interpolate an electrostatic potential grid): an exact Yukawa sum keeps
//! low ionic strength from growing the box.
//!
//! Provenance: written from these published descriptions. No AutoDock or
//! AutoGrid source code (MPL) was copied or adapted.

mod domain;
mod maps;
#[cfg(test)]
mod scan_bench;
mod tables;

pub use domain::GridDomain;

use super::gpu::{pack_positions, PoseKernel};
use super::gpu_context::{Binding, GpuContext, Limits};
use super::{EnergyBackend, PoseParams};
use crate::energy::{CoulombParams, SplinedPotentials};
use crate::structure::Structure;
use bytemuck::{Pod, Zeroable};
use maps::{build_maps, Maps};
use tables::{charged_beads, effective_range, SlotTable};

/// Wall height when none is given. A pose touching the wall counts at least
/// this much, so it only needs to be many kT; results match from ~30 kJ/mol up.
pub const DEFAULT_ENERGY_CAP: f64 = 1000.0;

/// Map resolution and wall height.
#[derive(Clone, Copy, Debug)]
pub struct GridSpec {
    /// Node spacing (Å).
    pub spacing: f64,
    /// Voxel energies are capped here (kJ/mol) so walls don't wreck interpolation.
    pub energy_cap: f64,
}

impl Default for GridSpec {
    fn default() -> Self {
        Self {
            spacing: 0.5,
            energy_cap: DEFAULT_ENERGY_CAP,
        }
    }
}

/// `Batch` in grid_energy.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BatchUniform {
    n_poses: u32,
    n_atoms_b: u32,
    sr_range: f32,
    kappa: f32,
    n_charged_a: u32,
    n_charged_b: u32,
    _pad: [u32; 2],
}

const ENERGY_BINDINGS: [Binding; 10] = [
    Binding::Read,      // maps
    Binding::Read,      // positions B
    Binding::Read,      // slot of each B bead
    Binding::Read,      // poses
    Binding::ReadWrite, // energies
    Binding::Uniform,   // grid
    Binding::Uniform,   // batch
    Binding::Read,      // charged A
    Binding::Read,      // charged B
    Binding::Read,      // Coulomb prefactors
];

/// Short-range energy from per-B-type maps of A's field; electrostatics exact.
pub struct GridBackend {
    gpu: GpuContext,
    kernel: PoseKernel,
    maps: Maps,
    /// B positions and the map slot of each B bead.
    beads_b: [wgpu::Buffer; 2],
    /// Charged A, charged B, Coulomb prefactors.
    electrostatic: [wgpu::Buffer; 3],
    /// Pose-independent part of the batch uniform.
    batch: BatchUniform,
    ref_a: Structure,
    ref_b: Structure,
}

// SAFETY: wasm32 is single-threaded; see `GpuContext`.
#[cfg(target_arch = "wasm32")]
unsafe impl Send for GridBackend {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for GridBackend {}

impl GridBackend {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(
        ref_a: Structure,
        ref_b: Structure,
        splines: &SplinedPotentials,
        coulomb: &CoulombParams,
        spec: GridSpec,
    ) -> anyhow::Result<Self> {
        pollster::block_on(Self::new_async(ref_a, ref_b, splines, coulomb, spec))
    }

    pub async fn new_async(
        ref_a: Structure,
        ref_b: Structure,
        splines: &SplinedPotentials,
        coulomb: &CoulombParams,
        spec: GridSpec,
    ) -> anyhow::Result<Self> {
        validate(&ref_a, &ref_b, spec)?;
        let gpu = GpuContext::new_async(Limits::Adapter).await?;
        let slots = SlotTable::new(&ref_b);
        let reach = effective_range(splines, &ref_a, &ref_b);
        let domain = GridDomain::covering(&ref_a.pos, reach, spec.spacing);
        let maps = build_maps(&gpu, &ref_a, splines, &slots, &domain, spec, reach).await?;
        log::info!(
            "Grid backend: {:?} nodes × {} slots, reach {reach:.1} Å",
            domain.dims(),
            slots.types.len()
        );
        let beads_b = [
            gpu.storage("Positions B", bytemuck::cast_slice(&pack_positions(&ref_b))),
            gpu.storage("Slots B", bytemuck::cast_slice(&slots.of_bead)),
        ];
        let (electrostatic, [n_charged_a, n_charged_b]) =
            electrostatics(&gpu, &ref_a, &ref_b, coulomb, splines.n_types());
        let batch = BatchUniform {
            n_poses: 0, // set per batch
            n_atoms_b: ref_b.pos.len() as u32,
            sr_range: (ref_a.radius() + reach + ref_b.radius()) as f32,
            kappa: coulomb.kappa,
            n_charged_a,
            n_charged_b,
            _pad: [0; 2],
        };
        let kernel = PoseKernel::new(&gpu, "Grid energy", &energy_source(), &ENERGY_BINDINGS);
        Ok(Self {
            gpu,
            kernel,
            maps,
            beads_b,
            electrostatic,
            batch,
            ref_a,
            ref_b,
        })
    }

    fn bind(&self, poses: &wgpu::Buffer, energies: &wgpu::Buffer, n_poses: u32) -> wgpu::BindGroup {
        let batch = BatchUniform {
            n_poses,
            ..self.batch
        };
        let batch_uniform = self.gpu.uniform("Batch", bytemuck::bytes_of(&batch));
        let [pos_b, slot_b] = &self.beads_b;
        let [charged_a, charged_b, prefactors] = &self.electrostatic;
        let buffers = [
            &self.maps.values,
            pos_b,
            slot_b,
            poses,
            energies,
            &self.maps.grid_uniform,
            &batch_uniform,
            charged_a,
            charged_b,
            prefactors,
        ];
        self.gpu.bind_group(&self.kernel.layout, &buffers)
    }

    /// As [`EnergyBackend::compute_energies`], for WASM.
    pub async fn compute_energies_async(&self, poses: &[PoseParams]) -> Vec<f64> {
        let bind = |p: &_, e: &_, n| self.bind(p, e, n);
        self.kernel.energies_async(&self.gpu, poses, bind).await
    }
}

/// Errors here rather than panics deeper in, since spacing is user input.
fn validate(ref_a: &Structure, ref_b: &Structure, spec: GridSpec) -> anyhow::Result<()> {
    anyhow::ensure!(
        spec.spacing > 0.0,
        "grid spacing must be positive, got {}",
        spec.spacing
    );
    anyhow::ensure!(
        !ref_a.pos.is_empty() && !ref_b.pos.is_empty(),
        "grid backend needs beads in both molecules"
    );
    Ok(())
}

/// Charged-only bead lists and Coulomb prefactors, plus the list lengths.
fn electrostatics(
    gpu: &GpuContext,
    ref_a: &Structure,
    ref_b: &Structure,
    coulomb: &CoulombParams,
    n_types: usize,
) -> ([wgpu::Buffer; 3], [u32; 2]) {
    let charged_a = charged_beads(ref_a, ref_b, coulomb, n_types);
    let charged_b = charged_beads(ref_b, ref_a, coulomb, n_types);
    let counts = [charged_a.len() as u32, charged_b.len() as u32];
    let buffers = [
        gpu.storage("Charged A", bytemuck::cast_slice(&charged_a)),
        gpu.storage("Charged B", bytemuck::cast_slice(&charged_b)),
        gpu.storage(
            "Coulomb prefactors",
            bytemuck::cast_slice(&coulomb.prefactors),
        ),
    ];
    (buffers, counts)
}

fn energy_source() -> String {
    format!(
        "{}\n{}\n{}",
        include_str!("../../shaders/common.wgsl"),
        include_str!("../../shaders/grid_common.wgsl"),
        include_str!("../../shaders/grid_energy.wgsl")
    )
}

impl EnergyBackend for GridBackend {
    fn compute_energy(&self, pose: &PoseParams) -> f64 {
        self.compute_energies(std::slice::from_ref(pose))[0]
    }

    fn compute_energies(&self, poses: &[PoseParams]) -> Vec<f64> {
        let bind = |p: &_, e: &_, n| self.bind(p, e, n);
        self.kernel.energies(&self.gpu, poses, bind)
    }

    fn prefers_batch(&self) -> bool {
        true
    }

    fn ref_a(&self) -> &Structure {
        &self.ref_a
    }

    fn ref_b(&self) -> &Structure {
        &self.ref_b
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::test_support::{bead, kt, lysozyme, no_gpu, pose_list};
    use crate::Vector3;

    const SPEC: GridSpec = GridSpec {
        spacing: 0.5,
        energy_cap: 1e4,
    };

    /// B at (r, 0, 0): `vertex_i = x̂` maps the separation axis onto x.
    fn along_x(r: f64) -> PoseParams {
        PoseParams {
            r,
            omega: 0.0,
            vertex_i: Vector3::x(),
            vertex_j: Vector3::z(),
        }
    }

    #[test]
    fn bead_on_a_node_matches_exact_pair() {
        if no_gpu() {
            return;
        }
        let sys = lysozyme();
        let (neutral, _) = sys.types_by_charge();
        let (ta, tb) = (neutral[0], neutral[1]);
        let grid = GridBackend::new(bead(ta), bead(tb), &sys.splines, &sys.coulomb, SPEC).unwrap();
        // A node at 0.5 Å spacing, outside the AH wall where splines are poorest.
        let pose = along_x(7.0);
        let u_grid = grid.compute_energy(&pose);
        let exact = sys.into_exact(bead(ta), bead(tb));
        let u_exact = exact.compute_energy(&pose);
        assert!(
            u_exact.abs() > 0.05,
            "pick a pair that interacts at 7 Å: {u_exact}"
        );
        approx::assert_relative_eq!(u_grid, u_exact, max_relative = 1e-4);
    }

    /// Largest |u''| over [lo, hi] along x, by central differences on `u`.
    fn max_curvature(u: impl Fn(f64) -> f64, lo: f64, hi: f64) -> f64 {
        const D: f64 = 0.01;
        (0..=10)
            .map(|k| lo + (hi - lo) * f64::from(k) / 10.0)
            .map(|x| ((u(x + D) - 2.0 * u(x) + u(x - D)) / (D * D)).abs())
            .fold(0.0, f64::max)
    }

    /// Mid-cell is the worst case for trilinear; error ≤ h²/8 · max|u''| there.
    #[test]
    fn bead_between_nodes_within_trilinear_bound() {
        if no_gpu() {
            return;
        }
        let sys = lysozyme();
        let (neutral, _) = sys.types_by_charge();
        let (ta, tb) = (neutral[0], neutral[1]);
        let grid = GridBackend::new(bead(ta), bead(tb), &sys.splines, &sys.coulomb, SPEC).unwrap();
        let exact = sys.into_exact(bead(ta), bead(tb));
        let u = |r: f64| exact.compute_energy(&along_x(r));
        let h = SPEC.spacing;
        for k in 0..10 {
            let lo = 7.0 + h * f64::from(k);
            let mid = lo + h / 2.0;
            let bound = h * h / 8.0 * max_curvature(u, lo, lo + h) + 1e-5;
            let err = (grid.compute_energy(&along_x(mid)) - u(mid)).abs();
            assert!(err <= bound, "r={mid}: |ΔU| {err:.2e} > bound {bound:.2e}");
        }
    }

    /// Beyond the short-range reach there is nothing to interpolate.
    #[test]
    fn bead_out_of_reach_gives_zero() {
        if no_gpu() {
            return;
        }
        let sys = lysozyme();
        let (neutral, _) = sys.types_by_charge();
        let (ta, tb) = (neutral[0], neutral[1]);
        let grid = GridBackend::new(bead(ta), bead(tb), &sys.splines, &sys.coulomb, SPEC).unwrap();
        for r in [25.0, 60.0, 500.0] {
            for side in [1.0, -1.0] {
                let pose = PoseParams {
                    vertex_i: side * Vector3::x(),
                    ..along_x(r)
                };
                assert_eq!(grid.compute_energy(&pose), 0.0, "r={r}, side {side}");
            }
        }
    }

    /// Electrostatics are summed exactly, so they can't depend on the map.
    #[test]
    fn charged_pair_is_exact_yukawa_at_any_spacing() {
        if no_gpu() {
            return;
        }
        let sys = lysozyme();
        let (_, charged) = sys.types_by_charge();
        let (ta, tb) = (charged[0], *charged.last().unwrap());
        let pose = along_x(25.0); // beyond short-range reach
        let grids: Vec<f64> = [0.5, 2.0]
            .into_iter()
            .map(|spacing| {
                let spec = GridSpec { spacing, ..SPEC };
                let grid = GridBackend::new(bead(ta), bead(tb), &sys.splines, &sys.coulomb, spec);
                grid.unwrap().compute_energy(&pose)
            })
            .collect();
        let u_exact = sys.into_exact(bead(ta), bead(tb)).compute_energy(&pose);
        assert!(u_exact.abs() > 0.01, "pick charged types: {u_exact}");
        for u_grid in grids {
            approx::assert_relative_eq!(u_grid, u_exact, max_relative = 1e-5);
        }
    }

    /// Capped voxels make the wall flat, so interpolating across it can't
    /// overshoot or go soft.
    #[test]
    fn overlap_sits_on_the_energy_cap() {
        if no_gpu() {
            return;
        }
        let sys = lysozyme();
        let (neutral, _) = sys.types_by_charge();
        let (ta, tb) = (neutral[0], neutral[1]);
        let spec = GridSpec {
            energy_cap: 100.0,
            ..SPEC
        };
        let grid = GridBackend::new(bead(ta), bead(tb), &sys.splines, &sys.coulomb, spec).unwrap();
        for r in [1.0, 1.25] {
            let u = grid.compute_energy(&along_x(r));
            approx::assert_relative_eq!(u, spec.energy_cap, max_relative = 1e-6);
        }
    }

    /// A bead in the wall makes the pose an overlap, however much the other
    /// beads attract; otherwise a low cap lets attraction win.
    #[test]
    fn wall_contact_outweighs_attraction() {
        if no_gpu() {
            return;
        }
        let sys = lysozyme();
        let (neutral, _) = sys.types_by_charge();
        let t = neutral[0];
        // B beads at ±3 Å on the axis the pose turns onto the separation, so at
        // R = 4 one sits 1 Å from A and the other in the well at 7 Å.
        let mut dimer = bead(t);
        dimer.pos = vec![Vector3::new(0.0, 0.0, -3.0), Vector3::new(0.0, 0.0, 3.0)];
        dimer.atom_ids = vec![t, t];
        let spec = GridSpec {
            energy_cap: 0.1, // below the far bead's attraction
            ..SPEC
        };
        let grid = GridBackend::new(bead(t), dimer, &sys.splines, &sys.coulomb, spec).unwrap();
        let u = grid.compute_energy(&along_x(4.0));
        assert!(u >= spec.energy_cap, "overlap read as {u} kJ/mol");
    }

    /// The full multi-type, multi-bead path on the benchmark system: poses that
    /// carry Boltzmann weight stay accurate and repulsive ones stay repulsive.
    #[test]
    fn lysozyme_pair_matches_exact_where_it_matters() {
        if no_gpu() {
            return;
        }
        let sys = lysozyme();
        let (a, b) = (sys.ref_a.clone(), sys.ref_b.clone());
        let grid =
            GridBackend::new(a.clone(), b.clone(), &sys.splines, &sys.coulomb, SPEC).unwrap();
        let exact = sys.into_exact(a, b);
        let poses = pose_list();
        let beta = 1.0 / kt();
        let mut n_weighted = 0;
        for (pose, u_grid) in poses.iter().zip(grid.compute_energies(&poses)) {
            let (bu_grid, bu_exact) = (beta * u_grid, beta * exact.compute_energy(pose));
            let ok = if bu_exact < 10.0 {
                n_weighted += 1;
                (bu_grid - bu_exact).abs() < 0.1
            } else {
                bu_exact <= 50.0 || bu_grid > 20.0
            };
            assert!(
                ok,
                "R={} ω={}: βU grid {bu_grid:.3}, exact {bu_exact:.3}",
                pose.r, pose.omega
            );
        }
        assert!(n_weighted >= 14, "only {n_weighted} weighted poses");
    }
}
