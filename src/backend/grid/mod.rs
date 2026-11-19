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
//! S·U, which holds every wall and well, is summed exactly over nearby
//! pairs; only the smooth remainder (1 − S)·U is gridded.
//!
//! Provenance: written from these published descriptions. No AutoDock or
//! AutoGrid source code (MPL) was copied or adapted.

mod domain;
mod maps;
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
use switch::Switch;
use tables::{bead_list, effective_range, SlotTable};

/// Map spacing (Å) when none is given: B₂ within 0.01 % of the pair kernel on
/// lysozyme, at about the time of 0.5 Å.
pub const DEFAULT_GRID_SPACING: f64 = 1.0;

/// Wall height when none is given. A pose touching the wall counts at least
/// this much, so it only needs to be many kT; results match from ~30 kJ/mol up.
pub const DEFAULT_ENERGY_CAP: f64 = 1000.0;

/// Map resolution and wall height; recorded in scan sessions as is.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GridSpec {
    /// Node spacing (Å).
    pub spacing: f64,
    /// Voxel energies are capped here (kJ/mol) so walls don't wreck interpolation.
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
    _pad: [u32; 2],
}

const ENERGY_BINDINGS: [Binding; 10] = [
    Binding::Read,      // maps
    Binding::Read,      // A beads, then charged A
    Binding::Read,      // B beads, then charged B
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
        let field = Field {
            ref_a: &ref_a,
            n_types: splines.n_types(),
            slots: &slots,
            domain: GridDomain::covering(&ref_a.pos, reach, spec.spacing),
            spec,
            reach,
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
        let (tables, [n_charged_a, n_charged_b]) =
            PoseTables::new(&gpu, &field, &ref_b, coulomb, spline_buffers);
        let batch = BatchUniform {
            n_poses: 0, // set per batch
            n_atoms_b: ref_b.pos.len() as u32,
            sr_range: (ref_a.radius() + reach + ref_b.radius()) as f32,
            kappa: coulomb.kappa,
            n_charged_a,
            n_charged_b,
            _pad: [0; 2],
        };
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
            &t.beads_a,
            &t.beads_b,
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
    beads_a: wgpu::Buffer,
    beads_b: wgpu::Buffer,
    prefactors: wgpu::Buffer,
    /// Spline coefficients and params, shared with the map build.
    splines: [wgpu::Buffer; 2],
}

impl PoseTables {
    /// The tables, and the charged A and B counts.
    fn new(
        gpu: &GpuContext,
        field: &Field,
        ref_b: &Structure,
        coulomb: &CoulombParams,
        splines: [wgpu::Buffer; 2],
    ) -> (Self, [u32; 2]) {
        let (n, ref_a, slots) = (field.n_types, field.ref_a, field.slots);
        let (a, n_charged_a) = bead_list(ref_a, ref_b, coulomb, n, |k| ref_a.atom_ids[k] as u32);
        let code_b = |k: usize| slots.of_bead[k] << 16 | ref_b.atom_ids[k] as u32;
        let (b, n_charged_b) = bead_list(ref_b, ref_a, coulomb, n, code_b);
        let tables = Self {
            beads_a: gpu.storage("Beads A", bytemuck::cast_slice(&a)),
            beads_b: gpu.storage("Beads B", bytemuck::cast_slice(&b)),
            prefactors: gpu.storage(
                "Coulomb prefactors",
                bytemuck::cast_slice(&coulomb.prefactors),
            ),
            splines,
        };
        (tables, [n_charged_a, n_charged_b])
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
        "{spline_wgsl}\n{}{}\n{}\n{}\n{}\n{}",
        maps::layout_consts(),
        include_str!("../../shaders/common.wgsl"),
        include_str!("../../shaders/spline_eval.wgsl"),
        include_str!("../../shaders/grid_common.wgsl"),
        switch::WGSL,
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
    use crate::backend::test_support::{bead, kt, lysozyme, no_gpu, pose_list};
    use crate::backend::GpuBackend;
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
    /// tail: near pairs are exact, far ones read straight off a node.
    #[test]
    fn bead_on_a_node_matches_exact_pair() {
        if no_gpu() {
            return;
        }
        let (grid, exact, switch) = neutral_pair(SPEC);
        let far_tail = (2.0 * switch.r2).ceil() / 2.0 + 1.0;
        for r in [
            4.0,
            5.0,
            6.0,
            7.0,
            (switch.r1 + switch.r2).round() / 2.0,
            far_tail,
        ] {
            let pose = along_x(r);
            let (u_grid, u_exact) = (grid.compute_energy(&pose), exact.compute_energy(&pose));
            // f16 maps resolve ~5e-4 relative; the floor covers f32 near sums.
            let tol = 1e-3 * u_exact.abs() + 1e-5;
            assert!(
                (u_grid - u_exact).abs() < tol,
                "r={r}: grid {u_grid}, exact {u_exact}"
            );
        }
    }

    /// Largest |u''| over [lo, hi] along x, by central differences on `u`.
    fn max_curvature(u: impl Fn(f64) -> f64, lo: f64, hi: f64) -> f64 {
        const D: f64 = 0.01;
        (0..=10)
            .map(|k| lo + (hi - lo) * f64::from(k) / 10.0)
            .map(|x| ((u(x + D) - 2.0 * u(x) + u(x - D)) / (D * D)).abs())
            .fold(0.0, f64::max)
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

    /// Mid-cell is the worst case for trilinear: error ≤ h²/8 · max|u″| of the
    /// far part alone, from the wall out past the switch, since near pairs
    /// are summed exactly.
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
            let bound = h * h / 8.0 * max_curvature(far, lo, lo + h) + 1e-5 + 1e-3 * u(mid).abs();
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
