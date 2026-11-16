//! Device, buffers and dispatch shared by the wgpu backends.
//!
//! Kernels bind their buffers at consecutive binding indices, which lets one
//! helper build both the layout and the bind group from a buffer list.

use std::sync::Mutex;
use wgpu::util::DeviceExt;

/// Which device limits a backend needs.
#[derive(Clone, Copy)]
pub(crate) enum Limits {
    /// WebGPU defaults (128 MiB storage bindings).
    Default,
    /// Everything the adapter offers; grid maps can exceed the defaults.
    Adapter,
}

/// One binding in a kernel's layout.
#[derive(Clone, Copy)]
pub(crate) enum Binding {
    Read,
    ReadWrite,
    Uniform,
}

pub(crate) struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// wgpu doesn't handle concurrent submit + poll from rayon threads well.
    submit_lock: Mutex<()>,
}

// SAFETY: wasm32 is single-threaded; wgpu handles there wrap JS objects that
// aren't Send, but nothing can race on them.
#[cfg(target_arch = "wasm32")]
unsafe impl Send for GpuContext {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for GpuContext {}

/// Headless compute: no surface, prefer the discrete GPU.
fn adapter_options() -> wgpu::RequestAdapterOptions<'static, 'static> {
    wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }
}

fn instance() -> wgpu::Instance {
    wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env())
}

impl GpuContext {
    /// Whether a compute adapter exists (WASM must use the async path).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn is_available() -> bool {
        pollster::block_on(instance().request_adapter(&adapter_options())).is_ok()
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(limits: Limits) -> anyhow::Result<Self> {
        pollster::block_on(Self::new_async(limits))
    }

    pub async fn new_async(limits: Limits) -> anyhow::Result<Self> {
        let adapter = instance().request_adapter(&adapter_options()).await?;
        log::info!("Using GPU adapter: {:?}", adapter.get_info().name);
        let required_limits = match limits {
            Limits::Default => wgpu::Limits::default(),
            Limits::Adapter => adapter.limits(),
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Duello GPU"),
                required_features: wgpu::Features::empty(),
                required_limits,
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::default(),
            })
            .await?;
        Ok(Self {
            device,
            queue,
            submit_lock: Mutex::new(()),
        })
    }

    /// Read-only storage buffer initialised from `contents`. Empty input gets
    /// one zeroed vec4, since wgpu rejects zero-sized buffers; kernels are
    /// told the true count, so the pad is never read.
    pub fn storage(&self, label: &str, contents: &[u8]) -> wgpu::Buffer {
        let contents = if contents.is_empty() {
            &[0; 16][..]
        } else {
            contents
        };
        self.init_buffer(label, contents, wgpu::BufferUsages::STORAGE)
    }

    pub fn uniform(&self, label: &str, contents: &[u8]) -> wgpu::Buffer {
        self.init_buffer(label, contents, wgpu::BufferUsages::UNIFORM)
    }

    /// Writable storage buffer of `n` f32 values that can be read back.
    pub fn output(&self, label: &str, n: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: f32_bytes(n),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    fn init_buffer(&self, label: &str, contents: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage,
            })
    }

    /// Compile `source` into a pipeline whose bindings are `bindings` in order.
    pub fn pipeline(
        &self,
        label: &str,
        source: &str,
        bindings: &[Binding],
    ) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
        let entries: Vec<_> = (0u32..)
            .zip(bindings)
            .map(|(i, b)| layout_entry(i, *b))
            .collect();
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries: &entries,
            });
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        (pipeline, layout)
    }

    /// Bind `buffers` at bindings 0, 1, 2, … of `layout`.
    pub fn bind_group(
        &self,
        layout: &wgpu::BindGroupLayout,
        buffers: &[&wgpu::Buffer],
    ) -> wgpu::BindGroup {
        let entries: Vec<_> = (0u32..)
            .zip(buffers)
            .map(|(binding, b)| wgpu::BindGroupEntry {
                binding,
                resource: b.as_entire_binding(),
            })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &entries,
        })
    }

    /// Queue `pipeline` without waiting; later dispatches run after it.
    pub fn dispatch(
        &self,
        pipeline: &wgpu::ComputePipeline,
        group: &wgpu::BindGroup,
        workgroups: [u32; 2],
    ) {
        let _lock = self.submit_lock.lock().unwrap();
        self.submit(pipeline, group, workgroups, None);
    }

    /// Run `pipeline` and read back the f32 buffer `output`.
    pub fn dispatch_read(
        &self,
        pipeline: &wgpu::ComputePipeline,
        group: &wgpu::BindGroup,
        workgroups: [u32; 2],
        output: &wgpu::Buffer,
    ) -> Vec<f32> {
        let _lock = self.submit_lock.lock().unwrap();
        let staging = self
            .submit(pipeline, group, workgroups, Some(output))
            .unwrap();
        let (sender, receiver) = std::sync::mpsc::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        self.wait();
        receiver
            .recv()
            .unwrap()
            .expect("GPU staging buffer map failed");
        read_staging(&staging)
    }

    /// As [`Self::dispatch_read`], for WASM where the browser drives mapping.
    pub async fn dispatch_read_async(
        &self,
        pipeline: &wgpu::ComputePipeline,
        group: &wgpu::BindGroup,
        workgroups: [u32; 2],
        output: &wgpu::Buffer,
    ) -> Vec<f32> {
        let staging = self
            .submit(pipeline, group, workgroups, Some(output))
            .unwrap();
        let (sender, receiver) = futures_channel::oneshot::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        #[cfg(not(target_arch = "wasm32"))]
        self.wait();
        receiver
            .await
            .unwrap()
            .expect("GPU staging buffer map failed");
        read_staging(&staging)
    }

    /// Encode one compute pass, plus a copy of `output` into a staging buffer.
    fn submit(
        &self,
        pipeline: &wgpu::ComputePipeline,
        group: &wgpu::BindGroup,
        [x, y]: [u32; 2],
        output: Option<&wgpu::Buffer>,
    ) -> Option<wgpu::Buffer> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(x, y, 1);
        }
        let staging = output.map(|out| {
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging"),
                size: out.size(),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            encoder.copy_buffer_to_buffer(out, 0, &staging, 0, out.size());
            staging
        });
        self.queue.submit(std::iter::once(encoder.finish()));
        staging
    }

    fn wait(&self) {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();
    }
}

pub(crate) const fn f32_bytes(n: usize) -> u64 {
    (n * std::mem::size_of::<f32>()) as u64
}

fn read_staging(staging: &wgpu::Buffer) -> Vec<f32> {
    // Callers have already checked the map succeeded, so a failed view is a bug.
    let data = staging
        .slice(..)
        .get_mapped_range()
        .expect("staging buffer is mapped");
    let values = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    values
}

fn layout_entry(binding: u32, kind: Binding) -> wgpu::BindGroupLayoutEntry {
    let ty = match kind {
        Binding::Read => wgpu::BufferBindingType::Storage { read_only: true },
        Binding::ReadWrite => wgpu::BufferBindingType::Storage { read_only: false },
        Binding::Uniform => wgpu::BufferBindingType::Uniform,
    };
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
