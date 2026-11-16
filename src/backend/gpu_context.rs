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
        self.submit_encoded(|encoder| encode_pass(encoder, pipeline, group, workgroups));
    }

    /// Run `pipeline` on each batch and return all f32 outputs, in order.
    ///
    /// A rolling window keeps up to [`IN_FLIGHT`] batches queued: before each
    /// new submission the oldest is waited for and read. The GPU never drains
    /// between batches, yet live buffers stay bounded however many poses the
    /// (lazy) iterator yields.
    pub fn dispatch_read(
        &self,
        pipeline: &wgpu::ComputePipeline,
        batches: impl IntoIterator<Item = Batch>,
    ) -> Vec<f32> {
        let _lock = self.submit_lock.lock().unwrap();
        let mut window = std::collections::VecDeque::with_capacity(IN_FLIGHT);
        let mut out = Vec::new();
        for batch in batches {
            if window.len() == IN_FLIGHT {
                out.extend(self.finish(window.pop_front().unwrap()));
            }
            window.push_back(self.launch(pipeline, &batch));
        }
        window.into_iter().for_each(|f| out.extend(self.finish(f)));
        out
    }

    /// As [`Self::dispatch_read`], for WASM where the browser drives mapping.
    pub async fn dispatch_read_async(
        &self,
        pipeline: &wgpu::ComputePipeline,
        batches: impl IntoIterator<Item = Batch>,
    ) -> Vec<f32> {
        let mut window = std::collections::VecDeque::with_capacity(IN_FLIGHT);
        let mut out = Vec::new();
        for batch in batches {
            if window.len() == IN_FLIGHT {
                out.extend(self.finish_async(window.pop_front().unwrap()).await);
            }
            window.push_back(self.launch(pipeline, &batch));
        }
        for f in window {
            out.extend(self.finish_async(f).await);
        }
        out
    }

    /// Submit one batch with a copy of its output into a staging buffer, and
    /// request that buffer's mapping.
    fn launch(&self, pipeline: &wgpu::ComputePipeline, batch: &Batch) -> InFlight {
        let size = batch.output.size();
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let submission = self.submit_encoded(|encoder| {
            encode_pass(encoder, pipeline, &batch.group, batch.workgroups);
            encoder.copy_buffer_to_buffer(&batch.output, 0, &staging, 0, size);
        });
        let (sender, mapped) = futures_channel::oneshot::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        InFlight {
            staging,
            mapped,
            submission,
        }
    }

    /// Wait for one batch only, so later ones keep the GPU busy, and read it.
    fn finish(&self, mut f: InFlight) -> Vec<f32> {
        self.wait_for(f.submission.clone());
        let mapped = f.mapped.try_recv().ok().flatten();
        mapped.expect(MAP_PENDING).expect(MAP_FAILED);
        read_staging(&f.staging)
    }

    async fn finish_async(&self, f: InFlight) -> Vec<f32> {
        #[cfg(not(target_arch = "wasm32"))]
        self.wait_for(f.submission.clone());
        f.mapped.await.expect(MAP_PENDING).expect(MAP_FAILED);
        read_staging(&f.staging)
    }

    fn submit_encoded(
        &self,
        encode: impl FnOnce(&mut wgpu::CommandEncoder),
    ) -> wgpu::SubmissionIndex {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encode(&mut encoder);
        self.queue.submit(std::iter::once(encoder.finish()))
    }

    fn wait_for(&self, submission: wgpu::SubmissionIndex) {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .unwrap();
    }
}

/// Batches queued at once by [`GpuContext::dispatch_read`]: enough to keep
/// the GPU busy across readbacks, few enough to bound live buffers and keep
/// each submission short for driver watchdogs.
const IN_FLIGHT: usize = 4;

const MAP_PENDING: &str = "GPU staging buffer not mapped after its submission completed";
const MAP_FAILED: &str = "GPU staging buffer map failed";

/// A submitted batch awaiting readback.
struct InFlight {
    staging: wgpu::Buffer,
    mapped: futures_channel::oneshot::Receiver<Result<(), wgpu::BufferAsyncError>>,
    submission: wgpu::SubmissionIndex,
}

/// One dispatch of a pipeline and the f32 buffer it writes.
pub(crate) struct Batch {
    pub group: wgpu::BindGroup,
    pub workgroups: [u32; 2],
    pub output: wgpu::Buffer,
}

fn encode_pass(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    group: &wgpu::BindGroup,
    [x, y]: [u32; 2],
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, group, &[]);
    pass.dispatch_workgroups(x, y, 1);
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
