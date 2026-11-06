//! Step-8 benchmark: native sampler, pair-energy costs, GPU lookup throughput
//! and scattering-frame cost on a real table. Not a test; prints numbers.
//!
//! `DUELLO_BENCH_ARCHIVE=lysozyme.duello cargo test --release -p duello \
//!   --lib bench -- --ignored --nocapture`

use super::{box_length, Device, Frame, PairEnergy, Pose, Sampler, SamplerSpec, Scattering};
use crate::archive::ScanArchive;
use crate::{UnitQuaternion, Vector3};
use rand::{Rng, SeedableRng};
use std::sync::Arc;
use std::time::Instant;

fn archive() -> Option<ScanArchive> {
    let path = std::env::var("DUELLO_BENCH_ARCHIVE").ok()?;
    Some(ScanArchive::from_bytes(&std::fs::read(path).ok()?).unwrap())
}

fn random_pose(rng: &mut impl Rng, center: Vector3, spread: f64) -> Pose {
    let offset = Vector3::from_fn(|_, _| spread * rng.gen_range(-1.0..1.0));
    let axis = Vector3::from_fn(|_, _| rng.gen_range(-3.0..3.0));
    Pose {
        position: center + offset,
        orientation: UnitQuaternion::from_scaled_axis(axis),
    }
}

/// ns per PairEnergy call for pairs inside the table and for distant pairs,
/// which only pay the minimum image and tail.
fn pair_costs(energy: &PairEnergy, rmax: f64) {
    let mut rng = rand_pcg::Pcg64::seed_from_u64(1);
    let l = 4.0 * rmax;
    for (label, distance) in [("in table", 0.8 * rmax), ("distant", 1.6 * rmax)] {
        let pairs: Vec<(Pose, Pose)> = (0..20_000)
            .map(|_| {
                let a = random_pose(&mut rng, Vector3::zeros(), 0.0);
                let dir = Vector3::from_fn(|_, _| rng.gen_range(-1.0..1.0)).normalize();
                (a, random_pose(&mut rng, dir * distance, 0.0))
            })
            .collect();
        let start = Instant::now();
        let mut sink = 0.0;
        for (a, b) in &pairs {
            if let super::PairValue::Finite(u) = energy.energy(a, b, l) {
                sink += u;
            }
        }
        let ns = start.elapsed().as_nanos() as f64 / pairs.len() as f64;
        println!("pair energy, {label}: {ns:.0} ns  (sink {sink:.1e})");
    }
}

fn sampler_speed(energy: &Arc<PairEnergy>, n: usize, molarity: f64, label: &str) {
    let spec = SamplerSpec {
        n_bodies: n,
        box_length: box_length(n, molarity),
        translation: 3.0,
        rotation: 0.2,
    };
    let Ok(mut sampler) = Sampler::new(energy.clone(), spec.clone(), 7) else {
        println!("sampler N={n} {label}: could not insert");
        return;
    };
    sampler.advance(2);
    let sweeps = (40_000 / n).max(2) as u64;
    let start = Instant::now();
    sampler.advance(sweeps);
    let per_sweep = start.elapsed().as_secs_f64() / sweeps as f64;
    let stats = sampler.take_move_stats();
    let acceptance = stats.translate.acceptance().unwrap_or(f64::NAN);
    println!(
        "native sampler N={n:>4} {label} (L={:.0} Å): {:.2} ms/sweep, {:.0} sweeps/s, translate acceptance {acceptance:.2}",
        spec.box_length,
        per_sweep * 1e3,
        1.0 / per_sweep
    );
}

#[test]
#[ignore = "benchmark: needs DUELLO_BENCH_ARCHIVE; run in release with --nocapture"]
fn step8_benchmark() {
    let Some(archive) = archive() else {
        eprintln!("set DUELLO_BENCH_ARCHIVE to a scan archive");
        return;
    };
    let rmax = archive.table.rmax;
    println!(
        "table: R {}–{} Å, {} R × {} ω bins, {:.1} MB of energies",
        archive.table.rmin,
        rmax,
        archive.table.n_r,
        archive.table.n_omega,
        archive.table.data.len() as f64 * 4e-6
    );
    let energy = Arc::new(PairEnergy::new(archive.table.clone()).unwrap());
    pair_costs(&energy, rmax);
    for n in [100, 200, 500, 1000] {
        sampler_speed(&energy, n, 0.005, "at 5 mM");
        sampler_speed(&energy, n, 0.028, "at 28 mM (φ≈0.3)");
    }
    gpu::lookup_throughput(&archive);
    scattering_cost(&archive, 200, 0.005);
}

/// One I(q) frame for N lysozymes, to compare with a sweep.
fn scattering_cost(archive: &ScanArchive, n: usize, molarity: f64) {
    let (reference, yaml) = scatterable(archive);
    let yaml = yaml.as_str();
    let l = box_length(n, molarity);
    let mut rng = rand_pcg::Pcg64::seed_from_u64(3);
    let poses = (0..n)
        .map(|_| random_pose(&mut rng, Vector3::from_element(l / 2.0), l / 2.0))
        .collect();
    let frame = Frame {
        copy: 0,
        sweep: 0,
        box_length: l,
        poses,
    };
    for device in [Device::Cpu, Device::Gpu] {
        let made = pollster::block_on(Scattering::new(&reference, n, l, 0.5, 16, yaml, device));
        let Ok(mut scattering) = made else {
            println!(
                "scattering {device:?}: unavailable ({})",
                made.err().unwrap()
            );
            continue;
        };
        pollster::block_on(scattering.observe(&frame)).unwrap();
        let start = Instant::now();
        let repeats = 5;
        for _ in 0..repeats {
            pollster::block_on(scattering.observe(&frame)).unwrap();
        }
        let ms = start.elapsed().as_secs_f64() * 1e3 / repeats as f64;
        let points = scattering.curve().q.len();
        println!(
            "scattering frame {device:?}: N={n}, {} beads/body, {points} q-points: {ms:.1} ms",
            reference.positions.len()
        );
    }
}

mod gpu {
    use crate::archive::ScanArchive;
    use icotable::gpu::{wgsl, GpuTable};
    use std::time::Instant;
    use wgpu::util::DeviceExt;

    /// Each thread evaluates `iterations` random in-table pairs exactly as
    /// PairEnergy does: inverse_orient and lookup, for both orders.
    const KERNEL: &str = "
        @group(0) @binding(0) var<storage, read> words: array<u32>;
        @group(0) @binding(1) var<storage, read> floats: array<f32>;
        @group(0) @binding(2) var<storage, read_write> out: array<f32>;
        @group(0) @binding(3) var<uniform> params: vec4<f32>; // iterations, beta, rmin, rmax
        fn icotable_word(i: u32) -> u32 { return words[i]; }
        fn icotable_float(i: u32) -> f32 { return floats[i]; }
        fn next(state: ptr<function, u32>) -> f32 {
            *state = *state * 747796405u + 2891336453u;
            var w = ((*state >> ((*state >> 28u) + 4u)) ^ *state) * 277803737u;
            w = (w >> 22u) ^ w;
            return f32(w) / 4294967296.0;
        }
        fn unit(state: ptr<function, u32>) -> vec3<f32> {
            let z = 2.0 * next(state) - 1.0;
            let phi = 6.2831853 * next(state);
            let s = sqrt(1.0 - z * z);
            return vec3<f32>(s * cos(phi), s * sin(phi), z);
        }
        fn quaternion(state: ptr<function, u32>) -> vec4<f32> {
            return normalize(vec4<f32>(next(state), next(state), next(state), next(state)) - 0.5);
        }
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            if id.x >= arrayLength(&out) { return; }
            var state = id.x * 9781u + 1u;
            var sum = 0.0;
            for (var it = 0u; it < u32(params.x); it++) {
                let r = mix(params.z, params.w, next(&state));
                let sep = unit(&state) * r;
                let qa = quaternion(&state);
                let qb = quaternion(&state);
                let f = icotable_inverse_orient(sep, qa, qb);
                let b = icotable_inverse_orient(-sep, qb, qa);
                let u1 = icotable_lookup(f.r, f.omega, f.dir_a, f.dir_b, params.y);
                let u2 = icotable_lookup(b.r, b.omega, b.dir_a, b.dir_b, params.y);
                if u1.repulsive == 0u && u2.repulsive == 0u { sum += 0.5 * (u1.energy + u2.energy); }
            }
            out[id.x] = sum;
        }
    ";

    pub fn lookup_throughput(archive: &ScanArchive) {
        let Some((device, queue)) = device() else {
            println!("GPU: no adapter");
            return;
        };
        let table = GpuTable::new(&archive.table).unwrap();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bench"),
            source: wgpu::ShaderSource::Wgsl(format!("{KERNEL}\n{}", wgsl()).into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bench"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let beta = 1.0 / crate::thermal_energy(298.15);
        for (threads, iterations) in [(1, 2000), (64, 2000), (1024, 2000), (4096, 1000)] {
            let run = || {
                dispatch(
                    &device, &queue, &pipeline, &table, threads, iterations, beta, archive,
                )
            };
            run(); // warm-up: shader compilation and buffer setup
            let seconds = run();
            let pairs = threads as f64 * iterations as f64;
            println!(
                "GPU {threads:>4} threads: {:.2e} pair energies/s ({:.0} ns per pair per thread)",
                pairs / seconds,
                seconds * 1e9 / iterations as f64
            );
        }
    }

    fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
        println!("GPU adapter: {}", adapter.get_info().name);
        let limits = adapter.limits();
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: limits,
            ..Default::default()
        }))
        .ok()
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipeline: &wgpu::ComputePipeline,
        table: &GpuTable,
        threads: u32,
        iterations: u32,
        beta: f64,
        archive: &ScanArchive,
    ) -> f64 {
        let storage = |contents: &[u8], extra: wgpu::BufferUsages| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents,
                usage: wgpu::BufferUsages::STORAGE | extra,
            })
        };
        let words = storage(
            bytemuck::cast_slice(&table.words),
            wgpu::BufferUsages::empty(),
        );
        let floats = storage(
            bytemuck::cast_slice(&table.floats),
            wgpu::BufferUsages::empty(),
        );
        let out = storage(
            &vec![0u8; threads as usize * 4],
            wgpu::BufferUsages::COPY_SRC,
        );
        let params = [
            iterations as f32,
            beta as f32,
            archive.table.rmin as f32,
            archive.table.rmax as f32,
        ];
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let read = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: threads as u64 * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: words.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: floats.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });
        let start = Instant::now();
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(threads.div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out, 0, &read, 0, threads as u64 * 4);
        queue.submit([encoder.finish()]);
        read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        start.elapsed().as_secs_f64()
    }
}

/// Whole run on the real table: timings and a sanity look at S_eff.
#[test]
#[ignore = "benchmark: needs DUELLO_BENCH_ARCHIVE; run in release with --nocapture"]
fn lysozyme_run() {
    let Some(archive) = archive() else {
        eprintln!("set DUELLO_BENCH_ARCHIVE to a scan archive");
        return;
    };
    let n = 200;
    let spec = super::RunSpec::new(n, 0.005);
    let mut run = super::NBodyRun::start(&archive, spec, super::RESIDUE_FORM_FACTORS).unwrap();
    let start = Instant::now();
    let mut production_at = None;
    while start.elapsed().as_secs() < 60 {
        let p = run.step(50).unwrap();
        if production_at.is_none() && matches!(p.stage, super::Stage::Production { .. }) {
            production_at = Some((start.elapsed().as_secs_f64(), p));
        }
    }
    let p = run.step(0).unwrap();
    println!("equilibrated: {production_at:?}");
    println!("after 60 s: {p:?}");
    let curve = run.curve();
    let errors = curve.s_eff_error.clone().unwrap_or_default();
    for k in (0..curve.q.len()).step_by(4) {
        println!(
            "q={:.3} S_eff={:.3} ± {:.3} P={:.0}",
            curve.q[k],
            curve.s_eff[k],
            errors.get(k).unwrap_or(&f64::NAN),
            curve.form_factor[k]
        );
    }
    let stats = run.stats();
    println!(
        "acceptance translate {:.2} rotate {:.2}; <U>/N = {:.2} kT",
        stats.moves.translate.acceptance().unwrap_or(f64::NAN),
        stats.moves.rotate.acceptance().unwrap_or(f64::NAN),
        stats.energy.mean / n as f64
    );
}

/// Miller (13 directions) against full-lattice q-grids: ms per frame and
/// how smooth S(q) gets from the same MC frames. Roughness is the rms
/// second difference of S across neighbouring q-points above 0.1 Å⁻¹, where
/// the true S(q) is nearly flat, so it is mostly sampling noise.
#[test]
#[ignore = "benchmark: needs DUELLO_BENCH_ARCHIVE; run in release with --nocapture"]
fn q_grid_benchmark() {
    let Some(archive) = archive() else {
        eprintln!("set DUELLO_BENCH_ARCHIVE to a scan archive");
        return;
    };
    let (n, q_max) = (200, 0.5);
    let l = box_length(n, 0.005);
    let frames = mc_frames(&archive, n, l, 60);
    let (reference, bead_yaml) = scatterable(&archive);
    // Tolerance: a q_max exactly on a shell keeps it.
    let p_max = (q_max / (std::f64::consts::TAU / l * 3f64.sqrt()) + 1e-9).floor() as usize;
    println!(
        "N={n}, L={l:.0} Å, q_max {q_max}, {} frames 10 sweeps apart",
        frames.len()
    );
    let lattice = |cap| pripps::QGrid::lattice([l; 3], q_max, cap).unwrap();
    for (label, grid) in [
        ("COM miller", pripps::QGrid::miller(p_max)),
        ("COM lattice cap 500", lattice(500)),
        ("COM lattice all", lattice(usize::MAX)),
    ] {
        let kinds = ["COM"];
        let yaml = "COM: { formfactor: !Constant 1.0 }\n";
        grid_row(
            label,
            &frames,
            l,
            &grid,
            |grid, device| new_scatterer(yaml, &kinds, n, grid, device),
            |frame| frame.poses.iter().map(|p| p.position.into()).collect(),
        );
    }
    let yaml = bead_yaml.as_str();
    for (label, grid) in [
        ("beads miller", pripps::QGrid::miller(p_max)),
        ("beads lattice cap 8", lattice(8)),
        ("beads lattice cap 16", lattice(16)),
        ("beads lattice cap 32", lattice(32)),
    ] {
        grid_row(
            label,
            &frames,
            l,
            &grid,
            |grid, device| new_scatterer(yaml, &reference.kinds, n, grid, device),
            |frame| bead_positions(&reference, frame),
        );
    }
}

fn new_scatterer(
    yaml: &str,
    kinds: &[impl AsRef<str>],
    n: usize,
    grid: pripps::QGrid,
    device: Device,
) -> Option<pripps::BoxScatterer> {
    match device {
        Device::Cpu => pripps::BoxScatterer::new_cpu(yaml, kinds, n, grid).ok(),
        Device::Gpu => pollster::block_on(pripps::BoxScatterer::new_gpu(yaml, kinds, n, grid)).ok(),
    }
}

/// One table row per device: vectors, ms/frame, roughness of total/self.
fn grid_row(
    label: &str,
    frames: &[Frame],
    l: f64,
    grid: &pripps::QGrid,
    make: impl Fn(pripps::QGrid, Device) -> Option<pripps::BoxScatterer>,
    positions: impl Fn(&Frame) -> Vec<[f64; 3]>,
) {
    for device in [Device::Cpu, Device::Gpu] {
        let Some(scatterer) = make(grid.clone(), device) else {
            println!("{label:<22} {device:?}: unavailable");
            continue;
        };
        let evaluate = |frame: &Frame| {
            pollster::block_on(scatterer.evaluate(&positions(frame), [l; 3])).unwrap()
        };
        let first = evaluate(&frames[0]);
        let start = Instant::now();
        let mut total = vec![0.0; first.q.len()];
        let mut self_term = vec![0.0; first.q.len()];
        for frame in frames {
            let result = evaluate(frame);
            super::scattering::add_into(&mut total, &result.total);
            super::scattering::add_into(&mut self_term, &result.self_term);
        }
        let ms = start.elapsed().as_secs_f64() * 1e3 / frames.len() as f64;
        let s: Vec<f64> = total.iter().zip(&self_term).map(|(t, p)| t / p).collect();
        let vectors = match grid.vectors_per_bin().iter().sum::<usize>() {
            0 => 13 * first.q.len() / 3, // Miller: 13 directions over 3 families per p
            sum => sum,
        };
        println!(
            "{label:<22} {device:?}: {:>3} q-points, {vectors:>6} vectors, {ms:>7.2} ms/frame, roughness {:.4}",
            first.q.len(),
            roughness(&first.q, &s)
        );
    }
}

/// Rms deviation of each point from the line through its neighbours, so
/// unevenly spaced grids (Miller: 1, √2, √3, 2, … × 2π/L) aren't penalised.
fn roughness(q: &[f64], s: &[f64]) -> f64 {
    let deviations: Vec<f64> = (1..s.len().saturating_sub(1))
        .filter(|&k| q[k] > 0.1)
        .map(|k| {
            let t = (q[k] - q[k - 1]) / (q[k + 1] - q[k - 1]);
            s[k] - (s[k - 1] + t * (s[k + 1] - s[k - 1]))
        })
        .collect();
    if deviations.is_empty() {
        return f64::NAN;
    }
    (deviations.iter().map(|d| d * d).sum::<f64>() / deviations.len() as f64).sqrt()
}

/// Equilibrated-ish frames from one copy: 2000 sweeps, then every 10.
fn mc_frames(archive: &ScanArchive, n: usize, l: f64, count: usize) -> Vec<Frame> {
    let energy = Arc::new(PairEnergy::new(archive.table.clone()).unwrap());
    let spec = SamplerSpec {
        n_bodies: n,
        box_length: l,
        translation: 3.0,
        rotation: 0.2,
    };
    let mut sampler = Sampler::new(energy, spec, 11).unwrap();
    sampler.advance(2000);
    (0..count)
        .map(|_| {
            sampler.advance(10);
            sampler.frame()
        })
        .collect()
}

/// The archive's first body and form factors for all its kinds, as runs
/// use them (kinds the residue file lacks scatter nothing).
fn scatterable(archive: &ScanArchive) -> (crate::archive::CgBody, String) {
    let reference = archive.bodies[0].clone();
    let yaml = super::run::with_missing_kinds(super::RESIDUE_FORM_FACTORS, &reference.kinds);
    (reference, yaml)
}

fn bead_positions(reference: &crate::archive::CgBody, frame: &Frame) -> Vec<[f64; 3]> {
    frame
        .poses
        .iter()
        .flat_map(|pose| {
            reference.positions.iter().map(move |&r| {
                let p = pose.position + pose.orientation * Vector3::from(r);
                [p.x, p.y, p.z]
            })
        })
        .collect()
}
