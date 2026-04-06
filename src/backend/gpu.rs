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

//! GPU backend using wgpu compute shaders.

use super::{EnergyBackend, PoseParams};
use crate::energy::{CoulombParams, SplinedPotentials};
use crate::structure::Structure;
use bytemuck::{Pod, Zeroable};
use faunus::interatomic::gpu::{GpuGridType, GpuSplineData, InverseRsq, PowerLaw2};
use faunus::interatomic::twobody::GridType;
use std::sync::{Arc, Mutex};
use wgpu::util::DeviceExt;

/// GPU-compatible pose parameters.
/// Note: WGSL struct alignment requires vec4 to be 16-byte aligned,
/// so we add padding after omega.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuPoseParams {
    pub(crate) r: f32,
    pub(crate) omega: f32,
    pub(crate) _pad: [f32; 2],     // Padding to align vertex_i to 16 bytes
    pub(crate) vertex_i: [f32; 4], // vec4<f32>
    pub(crate) vertex_j: [f32; 4], // vec4<f32>
}

impl From<&PoseParams> for GpuPoseParams {
    fn from(p: &PoseParams) -> Self {
        Self {
            r: p.r as f32,
            omega: p.omega as f32,
            _pad: [0.0, 0.0],
            vertex_i: [
                p.vertex_i.x as f32,
                p.vertex_i.y as f32,
                p.vertex_i.z as f32,
                0.0,
            ],
            vertex_j: [
                p.vertex_j.x as f32,
                p.vertex_j.y as f32,
                p.vertex_j.z as f32,
                0.0,
            ],
        }
    }
}

/// GPU-compatible uniform parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuUniforms {
    n_atoms_a: u32,
    n_atoms_b: u32,
    n_atom_types: u32,
    n_poses: u32,
    kappa: f32,
    _pad: [u32; 3],
}

/// GPU backend for energy calculations using wgpu compute shaders.
pub struct GpuBackend {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    // Constant buffers (uploaded once)
    spline_coeffs_buffer: wgpu::Buffer,
    spline_params_buffer: wgpu::Buffer,
    ref_pos_a_buffer: wgpu::Buffer,
    ref_pos_b_buffer: wgpu::Buffer,
    atom_ids_buffer: wgpu::Buffer,
    coulomb_prefactors_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    // Reference structures
    ref_a: Structure,
    ref_b: Structure,
    // Dimensions
    n_atoms_a: u32,
    n_atoms_b: u32,
    n_atom_types: u32,
    // Coulomb screening parameter
    kappa: f32,
    // Maximum batch size
    max_batch_size: usize,
    // Mutex for serializing GPU submissions (wgpu doesn't handle concurrent submissions well)
    submit_lock: Mutex<()>,
}

// SAFETY: On wasm32, execution is single-threaded so Send + Sync are trivially safe.
// wgpu types don't implement Send on wasm32 because they wrap JS handles, but since
// there's no multi-threading, this is sound.
#[cfg(target_arch = "wasm32")]
unsafe impl Send for GpuBackend {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for GpuBackend {}

impl GpuBackend {
    /// Check if a GPU is available for compute (not available on WASM — use async path).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn is_available() -> bool {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());

        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .is_ok()
    }

    /// Create a new GPU backend, auto-detecting grid type from the splined matrix (not available on WASM).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(
        ref_a: Structure,
        ref_b: Structure,
        splined_matrix: &SplinedPotentials,
        coulomb: &CoulombParams,
    ) -> anyhow::Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;

        log::info!("Using GPU adapter: {:?}", adapter.get_info().name);

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Duello GPU"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::default(),
            },
        ))?;

        Self::setup(Arc::new(device), Arc::new(queue), ref_a, ref_b, splined_matrix, coulomb)
    }

    /// Create a new GPU backend asynchronously (required for WASM).
    pub async fn new_async(
        ref_a: Structure,
        ref_b: Structure,
        splined_matrix: &SplinedPotentials,
        coulomb: &CoulombParams,
    ) -> anyhow::Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;

        log::info!("Using GPU adapter: {:?}", adapter.get_info().name);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Duello GPU"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                    trace: wgpu::Trace::Off,
                    experimental_features: wgpu::ExperimentalFeatures::default(),
                },
            )
            .await?;

        Self::setup(Arc::new(device), Arc::new(queue), ref_a, ref_b, splined_matrix, coulomb)
    }

    /// Shared setup: create pipeline and buffers from an already-obtained device+queue.
    fn setup(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        ref_a: Structure,
        ref_b: Structure,
        splined_matrix: &SplinedPotentials,
        coulomb: &CoulombParams,
    ) -> anyhow::Result<Self> {
        let grid_type = splined_matrix
            .iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("Empty spline matrix"))?
            .grid_type();
        match grid_type {
            GridType::PowerLaw2 => {
                Self::setup_typed::<PowerLaw2>(device, queue, ref_a, ref_b, splined_matrix, coulomb)
            }
            GridType::InverseRsq => {
                Self::setup_typed::<InverseRsq>(device, queue, ref_a, ref_b, splined_matrix, coulomb)
            }
            other => {
                anyhow::bail!("GPU backend requires PowerLaw2 or InverseRsq grid, got {other:?}")
            }
        }
    }

    /// Create pipeline and buffers for a specific grid type.
    fn setup_typed<G: GpuGridType>(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        ref_a: Structure,
        ref_b: Structure,
        splined_matrix: &SplinedPotentials,
        coulomb: &CoulombParams,
    ) -> anyhow::Result<Self> {
        let n_atom_types = splined_matrix.n_types() as u32;
        let spline_data = GpuSplineData::<G>::from_potentials(splined_matrix.iter());

        // Prepend grid-type-specific WGSL (SplineParams, SplineCoeffs, spline_index_eps)
        let shader_source = format!(
            "{}\n{}",
            G::SPLINE_WGSL,
            include_str!("../shaders/energy.wgsl")
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Energy Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Energy Bind Group Layout"),
            entries: &[
                storage_entry(0, true),  // Spline coefficients
                storage_entry(1, true),  // Spline params
                storage_entry(2, true),  // Reference positions A
                storage_entry(3, true),  // Reference positions B
                storage_entry(4, true),  // Atom IDs
                storage_entry(5, true),  // Poses (input)
                storage_entry(6, false), // Energies (output)
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_entry(8, true), // Coulomb prefactors
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Energy Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Energy Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let spline_coeffs_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Spline Coefficients"),
            contents: spline_data.coefficients_as_bytes(),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let spline_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Spline Params"),
            contents: spline_data.params_as_bytes(),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Pack reference positions as vec4<f32> (padded)
        let pos_a: Vec<[f32; 4]> = ref_a
            .pos
            .iter()
            .map(|p| [p.x as f32, p.y as f32, p.z as f32, 0.0])
            .collect();
        let pos_b: Vec<[f32; 4]> = ref_b
            .pos
            .iter()
            .map(|p| [p.x as f32, p.y as f32, p.z as f32, 0.0])
            .collect();

        let ref_pos_a_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Reference Positions A"),
            contents: bytemuck::cast_slice(&pos_a),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let ref_pos_b_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Reference Positions B"),
            contents: bytemuck::cast_slice(&pos_b),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Pack atom IDs
        let atom_ids: Vec<u32> = ref_a
            .atom_ids
            .iter()
            .chain(ref_b.atom_ids.iter())
            .map(|&id| id as u32)
            .collect();

        let atom_ids_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Atom IDs"),
            contents: bytemuck::cast_slice(&atom_ids),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let coulomb_prefactors_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Coulomb Prefactors"),
                contents: bytemuck::cast_slice(&coulomb.prefactors),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let n_atoms_a = ref_a.pos.len() as u32;
        let n_atoms_b = ref_b.pos.len() as u32;

        let uniforms = GpuUniforms {
            n_atoms_a,
            n_atoms_b,
            n_atom_types,
            n_poses: 0,
            kappa: coulomb.kappa,
            _pad: [0; 3],
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        log::info!(
            "GPU backend: {} coefficients, {} atom types, {} + {} atoms",
            spline_data.coefficients.len(),
            n_atom_types,
            n_atoms_a,
            n_atoms_b
        );

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            spline_coeffs_buffer,
            spline_params_buffer,
            ref_pos_a_buffer,
            ref_pos_b_buffer,
            atom_ids_buffer,
            coulomb_prefactors_buffer,
            uniform_buffer,
            ref_a,
            ref_b,
            n_atoms_a,
            n_atoms_b,
            n_atom_types,
            kappa: coulomb.kappa,
            max_batch_size: 100_000,
            submit_lock: Mutex::new(()),
        })
    }

    /// Encode and submit a GPU compute pass, returning the staging buffer.
    fn encode_and_submit(&self, poses: &[GpuPoseParams]) -> (wgpu::Buffer, usize) {
        let n_poses = poses.len();

        let pose_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Poses"),
                contents: bytemuck::cast_slice(poses),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Energies Output"),
            size: (n_poses * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging"),
            size: (n_poses * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Update uniforms
        let uniforms = GpuUniforms {
            n_atoms_a: self.n_atoms_a,
            n_atoms_b: self.n_atoms_b,
            n_atom_types: self.n_atom_types,
            n_poses: n_poses as u32,
            kappa: self.kappa,
            _pad: [0; 3],
        };
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Energy Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.spline_coeffs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.spline_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.ref_pos_a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.ref_pos_b_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.atom_ids_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: pose_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self.coulomb_prefactors_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Compute Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Energy Compute Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (n_poses as u32).div_ceil(64);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            (n_poses * std::mem::size_of::<f32>()) as u64,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        (staging_buffer, n_poses)
    }

    /// Read back energies from a mapped staging buffer.
    fn read_staging(staging_buffer: &wgpu::Buffer) -> Vec<f32> {
        let data = staging_buffer.slice(..).get_mapped_range();
        let energies: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();
        energies
    }

    /// Compute energies for a batch of poses on the GPU (synchronous).
    fn compute_batch(&self, poses: &[GpuPoseParams]) -> Vec<f32> {
        if poses.is_empty() {
            return Vec::new();
        }
        let _lock = self.submit_lock.lock().unwrap();
        let (staging_buffer, _) = self.encode_and_submit(poses);

        // Synchronous readback
        let slice = staging_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();

        Self::read_staging(&staging_buffer)
    }

    /// Compute energies for a batch of poses on the GPU (async, for WASM).
    pub async fn compute_batch_async(&self, poses: &[GpuPoseParams]) -> Vec<f32> {
        if poses.is_empty() {
            return Vec::new();
        }
        let (staging_buffer, _) = self.encode_and_submit(poses);

        let slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        // On native, we need to poll; on WASM, the browser event loop handles it
        #[cfg(not(target_arch = "wasm32"))]
        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();

        receiver.await.unwrap().unwrap();
        Self::read_staging(&staging_buffer)
    }

    /// Async version of compute_energies (processes in max_batch_size chunks).
    pub async fn compute_energies_async(&self, poses: &[PoseParams]) -> Vec<f64> {
        let gpu_poses: Vec<GpuPoseParams> = poses.iter().map(GpuPoseParams::from).collect();
        let mut all_energies = Vec::with_capacity(poses.len());
        for chunk in gpu_poses.chunks(self.max_batch_size) {
            let batch_energies = self.compute_batch_async(chunk).await;
            all_energies.extend(batch_energies.into_iter().map(|e| e as f64));
        }
        all_energies
    }
}

/// Helper to create a read-only or read-write storage bind group layout entry.
fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl EnergyBackend for GpuBackend {
    fn compute_energy(&self, pose: &PoseParams) -> f64 {
        let gpu_pose = GpuPoseParams::from(pose);
        let energies = self.compute_batch(&[gpu_pose]);
        energies.first().copied().unwrap_or(0.0) as f64
    }

    fn compute_energies(&self, poses: &[PoseParams]) -> Vec<f64> {
        let gpu_poses: Vec<GpuPoseParams> = poses.iter().map(GpuPoseParams::from).collect();
        let mut all_energies = Vec::with_capacity(poses.len());
        for chunk in gpu_poses.chunks(self.max_batch_size) {
            let batch_energies = self.compute_batch(chunk);
            all_energies.extend(batch_energies.into_iter().map(|e| e as f64));
        }
        all_energies
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
