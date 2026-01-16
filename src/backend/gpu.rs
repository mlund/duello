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
use crate::structure::Structure;
use bytemuck::{Pod, Zeroable};
use faunus::energy::NonbondedMatrixSplined;
use std::sync::{Arc, Mutex};
use wgpu::util::DeviceExt;

/// GPU-compatible spline parameters for a single pair type.
/// Uses PowerLaw grid: r(x) = r_min + (r_max - r_min) * x^power, where x ∈ [0,1]
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuSplineParams {
    r_min: f32,
    r_max: f32,
    power: f32,        // Power-law exponent (typically 2.0)
    n_coeffs: u32,
    coeff_offset: u32, // Offset into the coefficient buffer
    _pad: [u32; 3],    // Padding to 32 bytes for alignment
}

/// GPU-compatible pose parameters.
/// Note: WGSL struct alignment requires vec4 to be 16-byte aligned,
/// so we add padding after omega.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuPoseParams {
    pub r: f32,
    pub omega: f32,
    pub _pad: [f32; 2],    // Padding to align vertex_i to 16 bytes
    pub vertex_i: [f32; 4], // vec4<f32>
    pub vertex_j: [f32; 4], // vec4<f32>
}

impl From<&PoseParams> for GpuPoseParams {
    fn from(p: &PoseParams) -> Self {
        Self {
            r: p.r as f32,
            omega: p.omega as f32,
            _pad: [0.0, 0.0],
            vertex_i: [p.vertex_i.x as f32, p.vertex_i.y as f32, p.vertex_i.z as f32, 0.0],
            vertex_j: [p.vertex_j.x as f32, p.vertex_j.y as f32, p.vertex_j.z as f32, 0.0],
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
}

/// Extracted spline data ready for GPU upload.
struct GpuSplineData {
    /// Flattened spline coefficients: [pair0_coeff0, pair0_coeff1, ..., pair1_coeff0, ...]
    /// Each coefficient is vec4<f32> = [a0, a1, a2, a3]
    coefficients: Vec<[f32; 4]>,
    /// Parameters for each pair type (n_types × n_types)
    params: Vec<GpuSplineParams>,
    /// Number of atom types
    n_types: usize,
}

impl GpuSplineData {
    /// Extract spline data from NonbondedMatrixSplined.
    fn from_splined_matrix(matrix: &NonbondedMatrixSplined) -> Self {
        use interatomic::twobody::GridType;

        let potentials = matrix.get_potentials();
        let shape = potentials.shape();
        let n_types = shape[0];

        let mut coefficients = Vec::new();
        let mut params = Vec::with_capacity(n_types * n_types);

        for i in 0..n_types {
            for j in 0..n_types {
                let spline = &potentials[(i, j)];
                let stats = spline.stats();
                let coeffs = spline.coefficients();

                let coeff_offset = coefficients.len() as u32;

                // Extract energy coefficients (u[0..4]) for each interval
                for c in coeffs {
                    coefficients.push([c.u[0] as f32, c.u[1] as f32, c.u[2] as f32, c.u[3] as f32]);
                }

                // Extract power-law exponent from grid type
                let power = match stats.grid_type {
                    GridType::PowerLaw(p) => p as f32,
                    _ => panic!("GPU backend requires PowerLaw grid type"),
                };

                params.push(GpuSplineParams {
                    r_min: stats.r_min as f32,
                    r_max: stats.r_max as f32,
                    power,
                    n_coeffs: coeffs.len() as u32,
                    coeff_offset,
                    _pad: [0; 3],
                });
            }
        }

        Self {
            coefficients,
            params,
            n_types,
        }
    }
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
    uniform_buffer: wgpu::Buffer,
    // Reference structures
    ref_a: Structure,
    ref_b: Structure,
    // Dimensions
    n_atoms_a: u32,
    n_atoms_b: u32,
    n_atom_types: u32,
    // Maximum batch size
    max_batch_size: usize,
    // Mutex for serializing GPU submissions (wgpu doesn't handle concurrent submissions well)
    submit_lock: Mutex<()>,
}

impl GpuBackend {
    /// Check if a GPU is available for compute.
    pub fn is_available() -> bool {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .is_some()
    }

    /// Create a new GPU backend.
    ///
    /// # Arguments
    /// * `ref_a` - Reference structure for molecule A
    /// * `ref_b` - Reference structure for molecule B
    /// * `splined_matrix` - Splined pair potentials matrix
    pub fn new(
        ref_a: Structure,
        ref_b: Structure,
        splined_matrix: &NonbondedMatrixSplined,
    ) -> anyhow::Result<Self> {
        // Initialize wgpu
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| anyhow::anyhow!("Failed to find a suitable GPU adapter"))?;

        log::info!("Using GPU adapter: {:?}", adapter.get_info().name);

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Duello GPU"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // Extract spline data
        let spline_data = GpuSplineData::from_splined_matrix(splined_matrix);
        let n_atom_types = spline_data.n_types as u32;

        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Energy Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/energy.wgsl").into()),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Energy Bind Group Layout"),
            entries: &[
                // Spline coefficients
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Spline params
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Reference positions A
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Reference positions B
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Atom IDs
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Poses (input)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Energies (output)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Uniforms
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
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Energy Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Energy Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create constant buffers
        let spline_coeffs_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Spline Coefficients"),
            contents: bytemuck::cast_slice(&spline_data.coefficients),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let spline_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Spline Params"),
            contents: bytemuck::cast_slice(&spline_data.params),
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

        let n_atoms_a = ref_a.pos.len() as u32;
        let n_atoms_b = ref_b.pos.len() as u32;

        // Create uniform buffer (will be updated per batch)
        let uniforms = GpuUniforms {
            n_atoms_a,
            n_atoms_b,
            n_atom_types,
            n_poses: 0,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        log::info!(
            "GPU backend initialized: {} spline coefficients, {} atom types, {} + {} atoms",
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
            uniform_buffer,
            ref_a,
            ref_b,
            n_atoms_a,
            n_atoms_b,
            n_atom_types,
            max_batch_size: 100_000,
            submit_lock: Mutex::new(()),
        })
    }

    /// Compute energies for a batch of poses on the GPU.
    fn compute_batch(&self, poses: &[GpuPoseParams]) -> Vec<f32> {
        let n_poses = poses.len();
        if n_poses == 0 {
            return Vec::new();
        }

        // Acquire lock to serialize GPU submissions
        let _lock = self.submit_lock.lock().unwrap();

        // Create pose buffer
        let pose_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Poses"),
                contents: bytemuck::cast_slice(poses),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Create output buffer
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Energies Output"),
            size: (n_poses * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create staging buffer for readback
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
        };
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        // Create bind group
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
            ],
        });

        // Encode and submit
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
            // Dispatch workgroups: ceil(n_poses / 64)
            let workgroups = (n_poses as u32 + 63) / 64;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Copy output to staging buffer
        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            (n_poses * std::mem::size_of::<f32>()) as u64,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back results
        let slice = staging_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        let data = slice.get_mapped_range();
        let energies: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        energies
    }
}

impl EnergyBackend for GpuBackend {
    fn compute_energy(&self, pose: &PoseParams) -> f64 {
        let gpu_pose = GpuPoseParams::from(pose);
        let energies = self.compute_batch(&[gpu_pose]);
        energies.first().copied().unwrap_or(0.0) as f64
    }

    fn compute_energies(&self, poses: &[PoseParams]) -> Vec<f64> {
        // Convert to GPU format
        let gpu_poses: Vec<GpuPoseParams> = poses.iter().map(GpuPoseParams::from).collect();

        // Process in batches
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
