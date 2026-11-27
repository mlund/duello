//! Grid backend: A's short-range field precomputed on a Cartesian map.
//!
//! Each pose then costs one trilinear lookup per B bead, an exact sum over
//! the A beads close to it, and an exact Yukawa sum over charged pairs,
//! instead of a full N_A·N_B pair sum.
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
//! Trilinear lookups smear the narrow, curved contact wells and the walls.
//! So each pair energy U is split by a smooth switch S(r) (see `switch`):
//! S·U, which holds every wall and well, is summed exactly over the A beads
//! near each B bead (see `near`); only the smooth remainder (1 − S)·U is
//! gridded. This is the kernel split of multilevel summation at one level.
//! Tameling et al. (2014, J. Chem. Phys. 140, 024105, p. 024105-3,
//! doi:10.1063/1.4857735) split r⁻⁶ "into a smooth term and a remainder
//! term. The smooth part can be calculated using MLS, while the
//! asymptotically smooth remainder term decays much faster than the
//! original"; here the smooth part goes on the map and the remainder is
//! summed exactly.
//!
//! The far part's tail (e.g. −r⁻⁶) is curved one way, so plain trilinear
//! lookups would be biased one way too, and each B bead sums ~100 such
//! pairs. Nodes therefore hold f − h²/12·∇²f: linear interpolation errs by
//! h²/12·f″ on average over a cell, so these values cancel that leading error
//! where f is smooth. At a cutoff's kink the finite differences are right
//! only on average over node positions, so the map keeps the stencil's full
//! support: one step beyond the potentials' range, and inside r₁.
//!
//! Provenance: written from these published descriptions. No AutoDock or
//! AutoGrid source code (MPL) was copied or adapted.

mod domain;
mod maps;
mod near;
#[cfg(test)]
mod scan_bench;
mod switch;
mod tables;

pub use domain::GridDomain;

use super::gpu::{PoseKernel, SplineTables};
use super::gpu_context::{Binding, GpuContext, Limits};
use super::{EnergyBackend, PoseParams};
use crate::archive::ScanBackend;
use crate::energy::{CoulombParams, SplinedPotentials};
use crate::structure::Structure;
use bytemuck::{Pod, Zeroable};
use maps::{build_maps, Field, Maps};
use near::{CellsUniform, NearCells};
use switch::Switch;
use tables::{bead_list, effective_range, SlotTable};

/// Map spacing (Å) when none is given: B₂ within ≈0.2 % of the pair kernel on
/// chymotrypsinogen (CALVADOS 3 and Kim–Hummer), 1.2 % at 2 Å.
pub const DEFAULT_GRID_SPACING: f64 = 1.0;

/// Overlap cap when none is given. A pose with a bead reaching it counts at
/// least this much, so it only needs to be many kT; results match from ~30
/// kJ/mol up.
pub const DEFAULT_ENERGY_CAP: f64 = 1000.0;

/// Map resolution and overlap cap; recorded in scan sessions as is.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GridSpec {
    /// Node spacing (Å).
    pub spacing: f64,
    /// Each B bead's near energy is capped here (kJ/mol), so overlaps stay finite.
    pub energy_cap: f64,
}

impl Default for GridSpec {
    fn default() -> Self {
        Self {
            spacing: DEFAULT_GRID_SPACING,
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
    b_offset: u32,
    coulomb_cutoff_sq: f32,
    coulomb_shift: f32,
    _pad: [u32; 3],
    cells: CellsUniform,
}

impl BatchUniform {
    /// The parts that don't depend on the bead lists; counts are zero.
    fn new(field: &Field, ref_b: &Structure, coulomb: &CoulombParams) -> Self {
        // Lookups interpolate up to a cell diagonal past the last nonzero node.
        let map_range = field.support + 3f64.sqrt() * field.domain.spacing();
        let sr_range = field.ref_a.radius() + map_range + ref_b.radius();
        Self {
            n_atoms_b: ref_b.pos.len() as u32,
            sr_range: sr_range as f32,
            kappa: coulomb.kappa,
            coulomb_cutoff_sq: coulomb.cutoff_sq(),
            coulomb_shift: coulomb.shift,
            ..Self::zeroed()
        }
    }
}

const ENERGY_BINDINGS: [Binding; 10] = [
    Binding::Read,      // maps
    Binding::Read,      // A, charged A, B, charged B beads
    Binding::Read,      // cell starts
    Binding::Read,      // poses
    Binding::ReadWrite, // energies
    Binding::Uniform,   // grid
    Binding::Uniform,   // batch
    Binding::Read,      // Coulomb prefactors
    Binding::Read,      // spline coefficients
    Binding::Read,      // spline params
];

/// Short-range energy from per-B-type maps of A's far field plus exact near
/// pairs; electrostatics exact.
pub struct GridBackend {
    gpu: GpuContext,
    kernel: PoseKernel,
    maps: Maps,
    tables: PoseTables,
    /// Pose-independent part of the batch uniform.
    batch: BatchUniform,
    spec: GridSpec,
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

    /// As [`Self::try_new_async`], blocking.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn try_new(
        ref_a: Structure,
        ref_b: Structure,
        splines: &SplinedPotentials,
        coulomb: &CoulombParams,
        spec: GridSpec,
    ) -> Option<Self> {
        pollster::block_on(Self::try_new_async(ref_a, ref_b, splines, coulomb, spec))
    }

    /// The grid, or `None` with a warning when it can't be set up (e.g. maps
    /// too large for the GPU), so callers can fall back to the pair kernel:
    /// a slower scan beats none where the backend wasn't the user's choice.
    pub async fn try_new_async(
        ref_a: Structure,
        ref_b: Structure,
        splines: &SplinedPotentials,
        coulomb: &CoulombParams,
        spec: GridSpec,
    ) -> Option<Self> {
        Self::new_async(ref_a, ref_b, splines, coulomb, spec)
            .await
            .inspect_err(|err| {
                log::warn!("Grid backend unavailable ({err:#}); using the GPU pair kernel")
            })
            .ok()
    }

    pub async fn new_async(
        ref_a: Structure,
        ref_b: Structure,
        splines: &SplinedPotentials,
        coulomb: &CoulombParams,
        spec: GridSpec,
    ) -> anyhow::Result<Self> {
        validate(&ref_a, &ref_b, splines, spec)?;
        let gpu = GpuContext::new_async(Limits::Adapter).await?;
        let slots = SlotTable::new(&ref_b);
        let reach = effective_range(splines, &ref_a, &ref_b);
        // The prefilter's stencil reaches one step past where the potentials
        // vanish (e.g. a cutoff's kink), and its correction there counts too.
        let support = reach + maps::LAPLACIAN_STEP;
        let field = Field {
            ref_a: &ref_a,
            n_types: splines.n_types(),
            slots: &slots,
            domain: GridDomain::covering(&ref_a.pos, support, spec.spacing),
            spec,
            support,
            switch: Switch::new(splines, &ref_a, &ref_b),
        };
        let spline_tables = SplineTables::new(splines)?;
        let spline_buffers = [
            gpu.storage("Spline Coefficients", &spline_tables.coeffs),
            gpu.storage("Spline Params", &spline_tables.params),
        ];
        let maps = build_maps(&gpu, &field, &spline_tables, &spline_buffers).await?;
        log::info!(
            "Grid backend: {:?} nodes × {} slots, reach {reach:.1} Å, near pairs within {:.1} Å",
            field.domain.dims(),
            slots.types.len(),
            field.switch.r2
        );
        let (tables, batch) = PoseTables::new(&gpu, &field, &ref_b, coulomb, spline_buffers);
        let source = energy_source(spline_tables.wgsl);
        let kernel = PoseKernel::new(&gpu, "Grid energy", &source, &ENERGY_BINDINGS);
        Ok(Self {
            gpu,
            kernel,
            maps,
            tables,
            batch,
            spec,
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
        let t = &self.tables;
        let [coeffs, params] = &t.splines;
        let buffers = [
            &self.maps.values,
            &t.beads,
            &t.cell_start,
            poses,
            energies,
            &self.maps.grid_uniform,
            &batch_uniform,
            &t.prefactors,
            coeffs,
            params,
        ];
        self.gpu.bind_group(&self.kernel.layout, &buffers)
    }
}

/// The pose kernel's buffers besides the maps; see grid_energy.wgsl.
struct PoseTables {
    beads: wgpu::Buffer,
    cell_start: wgpu::Buffer,
    prefactors: wgpu::Buffer,
    /// Spline coefficients and params, shared with the map build.
    splines: [wgpu::Buffer; 2],
}

impl PoseTables {
    /// The tables, and the batch uniform that indexes them, less its pose count.
    fn new(
        gpu: &GpuContext,
        field: &Field,
        ref_b: &Structure,
        coulomb: &CoulombParams,
        splines: [wgpu::Buffer; 2],
    ) -> (Self, BatchUniform) {
        let (n, ref_a, slots) = (field.n_types, field.ref_a, field.slots);
        let cells = NearCells::new(&ref_a.pos, field.switch.r2);
        let order_a = cells.order().iter().map(|&i| i as usize);
        let (mut beads, n_charged_a) = bead_list(ref_a, ref_b, coulomb, n, order_a, |k| {
            ref_a.atom_ids[k] as u32
        });
        let code_b = |k: usize| slots.of_bead[k] << 16 | ref_b.atom_ids[k] as u32;
        let b_offset = beads.len() as u32;
        let (b, n_charged_b) = bead_list(ref_b, ref_a, coulomb, n, 0..ref_b.pos.len(), code_b);
        beads.extend(b);
        let tables = Self {
            beads: gpu.storage("Beads", bytemuck::cast_slice(&beads)),
            cell_start: gpu.storage("Cell starts", bytemuck::cast_slice(cells.starts())),
            prefactors: gpu.storage(
                "Coulomb prefactors",
                bytemuck::cast_slice(&coulomb.prefactors),
            ),
            splines,
        };
        let batch = BatchUniform {
            n_charged_a,
            n_charged_b,
            b_offset,
            cells: cells.uniform(),
            ..BatchUniform::new(field, ref_b, coulomb)
        };
        (tables, batch)
    }
}

/// Errors here rather than panics deeper in, since spacing is user input.
fn validate(
    ref_a: &Structure,
    ref_b: &Structure,
    splines: &SplinedPotentials,
    spec: GridSpec,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        spec.spacing > 0.0 && spec.spacing.is_finite(),
        "grid spacing must be positive, got {}",
        spec.spacing
    );
    // Kept within f16, the maps' storage, since stored values are clamped to it.
    anyhow::ensure!(
        spec.energy_cap > 0.0 && spec.energy_cap <= maps::F16_MAX,
        "grid energy cap must be in (0, {}] kJ/mol, got {}",
        maps::F16_MAX,
        spec.energy_cap
    );
    anyhow::ensure!(
        !ref_a.pos.is_empty() && !ref_b.pos.is_empty(),
        "grid backend needs beads in both molecules"
    );
    anyhow::ensure!(
        splines.n_types() <= tables::MAX_TYPES,
        "grid backend supports up to {} atom types, got {}",
        tables::MAX_TYPES,
        splines.n_types()
    );
    Ok(())
}

fn energy_source(spline_wgsl: &str) -> String {
    format!(
        "{spline_wgsl}\n{}{}\n{}\n{}\n{}\n{}\n{}",
        maps::layout_consts(),
        include_str!("../../shaders/common.wgsl"),
        crate::backend::gpu::spline_eval_wgsl(),
        include_str!("../../shaders/grid_common.wgsl"),
        switch::WGSL,
        near::WGSL,
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

    fn kind(&self) -> ScanBackend {
        ScanBackend::Grid(self.spec)
    }

    fn ref_a(&self) -> &Structure {
        &self.ref_a
    }

    fn ref_b(&self) -> &Structure {
        &self.ref_b
    }
}

impl crate::backend::AsyncEnergyBackend for GridBackend {
    async fn compute_energies_async(&self, poses: &[PoseParams]) -> Vec<f64> {
        let bind = |p: &_, e: &_, n| self.bind(p, e, n);
        self.kernel.energies_async(&self.gpu, poses, bind).await
    }
}

#[cfg(test)]
mod tests {
    use super::switch::Switch;
    use super::*;
    use crate::backend::test_support::{
        along_x, bead, contact_poses, kt, lysozyme, lysozyme_with_range, no_gpu, pose_list,
        random_direction,
    };
    use crate::backend::GpuBackend;
    use crate::Vector3;

    const SPEC: GridSpec = GridSpec {
        spacing: 0.5,
        energy_cap: 1e4,
    };

    /// Yukawa is summed outside the maps, so a cutoff applies as in the
    /// reference: shifted inside, zero beyond.
    #[test]
    fn charged_pair_follows_the_coulomb_cutoff() {
        if no_gpu() {
            return;
        }
        let sys = lysozyme().with_coulomb_cutoff(30.0);
        let (_, charged) = sys.types_by_charge();
        let (ta, tb) = (charged[0], *charged.last().unwrap());
        let grid = GridBackend::new(bead(ta), bead(tb), &sys.splines, &sys.coulomb, SPEC).unwrap();
        let exact = sys.into_exact(bead(ta), bead(tb));
        for r in [25.0, 29.0, 35.0] {
            let (u_grid, u_exact) = (
                grid.compute_energy(&along_x(r)),
                exact.compute_energy(&along_x(r)),
            );
            approx::assert_abs_diff_eq!(u_grid, u_exact, epsilon = 1e-5 * u_exact.abs() + 1e-6);
        }
    }

    /// A neutral pair at the origin and the pair kernel on the same splines,
    /// so only the grid's own error shows, even up the wall.
    fn neutral_pair(spec: GridSpec) -> (GridBackend, GpuBackend, Switch) {
        let sys = lysozyme();
        let (neutral, _) = sys.types_by_charge();
        let (a, b) = (bead(neutral[0]), bead(neutral[1]));
        let switch = Switch::new(&sys.splines, &a, &b);
        let grid = GridBackend::new(a.clone(), b.clone(), &sys.splines, &sys.coulomb, spec);
        let pairs = GpuBackend::new(a, b, &sys.splines, &sys.coulomb).unwrap();
        (grid.unwrap(), pairs, switch)
    }

    /// Nodes at 0.5 Å from the wall, through the well and switch, to the far
    /// tail: near pairs are exact, and a node holds the far part prefiltered
    /// as f − h²/12·∇²f, with ∇²f = f″ + 2f′/r for a single pair.
    #[test]
    fn bead_on_a_node_matches_exact_pair() {
        if no_gpu() {
            return;
        }
        let (grid, exact, switch) = neutral_pair(SPEC);
        let u = |r: f64| exact.compute_energy(&along_x(r));
        let far = |r: f64| (1.0 - switch.near_weight(r)) * u(r);
        // The build's step, so both apply one operator.
        let laplacian = |r: f64| radial_laplacian(far, r, maps::LAPLACIAN_STEP);
        let h = SPEC.spacing;
        let far_tail = (2.0 * switch.r2).ceil() / 2.0 + 1.0;
        let mid_switch = (switch.r1 + switch.r2).round() / 2.0;
        for r in [4.0, 5.0, 6.0, 7.0, mid_switch, far_tail] {
            let u_grid = grid.compute_energy(&along_x(r));
            let want = u(r) - h * h / 12.0 * laplacian(r);
            // f16 maps resolve ~5e-4 relative; the floor covers f32 sums.
            let tol = 1e-3 * want.abs() + 1e-5;
            assert!(
                (u_grid - want).abs() < tol,
                "r={r}: grid {u_grid}, want {want}"
            );
        }
    }

    /// Largest |u''| over [lo, hi] along x, by central differences on `u`.
    fn max_curvature(u: impl Fn(f64) -> f64, lo: f64, hi: f64) -> f64 {
        const D: f64 = 0.01;
        max_over(
            |x| ((u(x + D) - 2.0 * u(x) + u(x - D)) / (D * D)).abs(),
            lo,
            hi,
        )
    }

    /// Largest of `g` at 11 points spanning [lo, hi].
    fn max_over(g: impl Fn(f64) -> f64, lo: f64, hi: f64) -> f64 {
        (0..=10)
            .map(|k| g(lo + (hi - lo) * f64::from(k) / 10.0))
            .fold(f64::MIN, f64::max)
    }

    /// ∇²f = f″ + 2f′/r of a radial `f` at `r`, by central differences of step `d`.
    fn radial_laplacian(f: impl Fn(f64) -> f64, r: f64, d: f64) -> f64 {
        let (lo, mid, hi) = (f(r - d), f(r), f(r + d));
        (hi - 2.0 * mid + lo) / (d * d) + (hi - lo) / (d * r)
    }

    /// Spec values come from users (CLI, web UI); bad ones must error, not
    /// build nonsense maps. Checked before touching the GPU.
    #[test]
    fn rejects_unusable_specs() {
        let sys = lysozyme();
        let t = sys.types_by_charge().0[0];
        for (spacing, energy_cap) in [
            (0.0, 1e3),
            (f64::NAN, 1e3),
            (0.5, 0.0),
            (0.5, f64::INFINITY),
            (0.5, 1e5),
        ] {
            let spec = GridSpec {
                spacing,
                energy_cap,
            };
            let result = GridBackend::new(bead(t), bead(t), &sys.splines, &sys.coulomb, spec);
            assert!(result.is_err(), "accepted {spec:?}");
        }
    }

    /// Mid-cell is the worst case for trilinear: error ≤ h²/8·max|f″| of the
    /// far part f alone, from the wall out past the switch, since near pairs
    /// are summed exactly; plus the prefilter's shift, h²/12·max|∇²f|.
    #[test]
    fn bead_between_nodes_within_trilinear_bound() {
        if no_gpu() {
            return;
        }
        let (grid, exact, switch) = neutral_pair(SPEC);
        let u = |r: f64| exact.compute_energy(&along_x(r));
        let far = |r: f64| (1.0 - switch.near_weight(r)) * u(r);
        let h = SPEC.spacing;
        for k in 0..24 {
            let lo = 4.0 + h * f64::from(k);
            let mid = lo + h / 2.0;
            let shift = max_over(|r| radial_laplacian(far, r, 0.01).abs(), lo, lo + h);
            let bound = h * h / 8.0 * max_curvature(far, lo, lo + h)
                + h * h / 12.0 * shift
                + 1e-5
                + 1e-3 * u(mid).abs();
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

    /// Near pairs are capped, so an overlap reads as the cap, not as a
    /// spline extrapolated far up the wall.
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
        // Just outside the table start: inside it, a pose is OVERLAP_ENERGY instead.
        let start = sys.splines.get(ta, tb).r_min();
        for r in [start + 0.05, start + 0.3] {
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

    /// Overlapping several A beads at once is an overlap too, even when each
    /// pair alone stays under the cap and many other beads attract.
    #[test]
    fn joint_wall_contact_outweighs_attraction() {
        if no_gpu() {
            return;
        }
        let sys = lysozyme();
        let t = sys.types_by_charge().0[0];
        let (r_m, _) = sys.splines.minimum(t, t).unwrap();
        let pair = GpuBackend::new(bead(t), bead(t), &sys.splines, &sys.coulomb).unwrap();
        let u_pair = pair.compute_energy(&along_x(r_m - 1.0));
        assert!(u_pair > 0.0, "r_m − 1 Å should be on the wall: {u_pair}");
        let mut trimer = bead(t);
        trimer.pos = vec![Vector3::zeros(); 3];
        trimer.atom_ids = vec![t; 3];
        // At R = r_m − ½ the pose turns +z towards A: the first B bead sits
        // 1 Å up the wall of all three A beads, ten more in the well behind.
        let mut b = bead(t);
        b.pos = std::iter::once(0.5)
            .chain([-0.5; 10])
            .map(|z| Vector3::new(0.0, 0.0, z))
            .collect();
        b.atom_ids = vec![t; 11];
        let spec = GridSpec {
            energy_cap: 2.0 * u_pair,
            ..SPEC
        };
        let grid = GridBackend::new(trimer, b, &sys.splines, &sys.coulomb, spec).unwrap();
        let u = grid.compute_energy(&along_x(r_m - 0.5));
        assert!(u >= spec.energy_cap, "joint overlap read as {u} kJ/mol");
    }

    /// The cap applies to a bead's whole near energy, not pair by pair: a
    /// bead up one A bead's wall but held in the wells of many others is
    /// attractive overall, whatever the cap.
    #[test]
    fn wall_pair_offset_by_attraction_is_not_an_overlap() {
        if no_gpu() {
            return;
        }
        let sys = lysozyme();
        let t = sys.types_by_charge().0[0];
        let (r_m, u_m) = sys.splines.minimum(t, t).unwrap();
        let pair = GpuBackend::new(bead(t), bead(t), &sys.splines, &sys.coulomb).unwrap();
        let r_wall = r_m - 1.0;
        let u_wall = pair.compute_energy(&along_x(r_wall));
        // B sits at x = r_wall: one A bead at the origin, 40 at r_m beyond B.
        let mut a = bead(t);
        a.pos = std::iter::once(Vector3::zeros())
            .chain(std::iter::repeat_n(
                Vector3::new(r_wall + r_m, 0.0, 0.0),
                40,
            ))
            .collect();
        a.atom_ids = vec![t; 41];
        let pose = along_x(r_wall);
        let exact = GpuBackend::new(a.clone(), bead(t), &sys.splines, &sys.coulomb).unwrap();
        let u_exact = exact.compute_energy(&pose);
        assert!(
            u_wall > 0.0 && u_exact < 0.0,
            "wall {u_wall}, net {u_exact}, U_m {u_m}"
        );
        let spec = GridSpec {
            energy_cap: 0.5 * u_wall, // below the wall pair, far above the net
            ..SPEC
        };
        let grid = GridBackend::new(a, bead(t), &sys.splines, &sys.coulomb, spec).unwrap();
        let u = grid.compute_energy(&pose);
        assert!((u - u_exact).abs() < 1e-3, "grid {u}, exact {u_exact}");
    }

    /// A short pair range puts a kink in the far part. The prefilter's
    /// finite differences straddle it, which is right on average over node
    /// positions only if the map keeps the stencil's full support. Checked
    /// over the far part out past the cutoff, with A beads off the lattice
    /// as in any protein.
    #[test]
    fn prefilter_stays_unbiased_across_a_cutoff() {
        if no_gpu() {
            return;
        }
        let sys = lysozyme_with_range(12.0);
        let (neutral, _) = sys.types_by_charge();
        let spec = GridSpec {
            spacing: 1.0,
            ..SPEC
        };
        let offsets = [[0.37, 0.61, 0.13], [0.81, 0.22, 0.55], [0.5, 0.5, 0.5]];
        let (mut bias, mut scale) = (0.0, 0.0);
        for offset in offsets {
            let mut a = bead(neutral[0]);
            a.pos = vec![Vector3::from(offset)];
            let b = bead(neutral[1]);
            let grid = GridBackend::new(a.clone(), b.clone(), &sys.splines, &sys.coulomb, spec);
            let pairs = GpuBackend::new(a, b, &sys.splines, &sys.coulomb).unwrap();
            let poses = shell_poses(9.0, 13.0, 40_000);
            let got = grid.unwrap().compute_energies(&poses);
            let want = pairs.compute_energies(&poses);
            bias += got.iter().zip(&want).map(|(g, w)| g - w).sum::<f64>();
            scale += want.iter().map(|w| w.abs()).sum::<f64>();
        }
        let relative = bias / scale;
        // Without the stencil's support kept, −0.5 %.
        assert!(
            relative.abs() < 3e-3,
            "far part biased by {:.2} %",
            100.0 * relative
        );
    }

    /// Poses uniform in the volume of the shell [lo, hi] around A; seeded.
    fn shell_poses(lo: f64, hi: f64, n: usize) -> Vec<PoseParams> {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_pcg::Pcg64::seed_from_u64(3);
        let (lo3, hi3) = (lo.powi(3), hi.powi(3));
        (0..n)
            .map(|_| PoseParams {
                r: rng.gen_range(lo3..hi3).cbrt(),
                omega: 0.0,
                vertex_i: random_direction(&mut rng),
                vertex_j: Vector3::z(),
            })
            .collect()
    }

    /// Pose by pose against the pair kernel on the same splines: near pairs
    /// are exact, so only the far map's trilinear error remains: ≤ 5e-3 kT
    /// at 0.5 Å on lysozyme, falling as h². Random contacts sweep B beads through every
    /// cell neighbourhood the GPU's cell list returns.
    #[test]
    fn lysozyme_poses_match_pair_kernel() {
        if no_gpu() {
            return;
        }
        let sys = lysozyme();
        let (a, b) = (sys.ref_a.clone(), sys.ref_b.clone());
        // No bead reaches this cap, so overlaps aren't clamped either.
        let spec = GridSpec {
            energy_cap: maps::F16_MAX,
            ..SPEC
        };
        let grid = GridBackend::new(a.clone(), b.clone(), &sys.splines, &sys.coulomb, spec);
        let pairs = GpuBackend::new(a, b, &sys.splines, &sys.coulomb).unwrap();
        let poses = [pose_list(), contact_poses(200, 28.0)].concat();
        let got = grid.unwrap().compute_energies(&poses);
        let want = pairs.compute_energies(&poses);
        for ((pose, g), w) in poses.iter().zip(got).zip(want) {
            let err = (g - w).abs() / kt();
            // Overlaps sum large f32 terms, so their error scales with |βU|.
            let tol = 1e-2 + 1e-4 * (w / kt()).abs();
            assert!(
                err < tol,
                "R={} ω={}: βΔU {err:.2e}, βU {:.1}",
                pose.r,
                pose.omega,
                w / kt()
            );
        }
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
