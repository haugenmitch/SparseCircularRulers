use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SearchParams {
    pub length: u32,
    pub num_segments: u32,
    /// Number of (slot, prefix) entries in this pass (one GPU thread each).
    pub num_work: u32,
    pub _padding0: u32,
    /// 32-bit words per bitset.
    pub words: u32,
    pub max_results: u32,
    /// Search nodes each thread may visit before it saves its state and stops for this pass.
    pub node_budget: u32,
    pub _padding: u32,
}

/// Marks the work entry of a slot that resumes its saved state instead of starting a prefix.
pub const RESUME: u32 = u32::MAX;

const WORKGROUP_SIZE: u32 = 64;
const MAX_MARKS: u64 = 32;
/// u32s of saved search state per slot: depth, then 32 marks, then 32 next positions, then
/// the depth of the slot's partial ruler.
pub const STATE_STRIDE: usize = 2 + 2 * MAX_MARKS as usize;
/// Longest a pass may take before the GPU is considered lost.
const PASS_TIMEOUT: Duration = Duration::from_secs(60);

/// What one pass returned.
pub struct Pass {
    /// Marks of the complete rulers found (`num_segments` per ruler).
    pub rulers: Vec<u32>,
    /// Slots whose partial ruler was fully searched.
    pub finished: Vec<u32>,
    pub nodes: u64,
}

pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline_layout: wgpu::PipelineLayout,
    /// One pipeline per bitset size (`SearchParams::words`, and the words that hold distances
    /// up to half the length), built on first use.
    pipelines: Mutex<HashMap<(u32, u32), wgpu::ComputePipeline>>,
    bind_group: wgpu::BindGroup,
    params_buffer: wgpu::Buffer,
    work_buffer: wgpu::Buffer,
    prefix_buffer: wgpu::Buffer,
    results_buffer: wgpu::Buffer,
    counter_buffer: wgpu::Buffer,
    finished_buffer: wgpu::Buffer,
    staging_counter: wgpu::Buffer,
    staging_finished: wgpu::Buffer,
    staging_results: wgpu::Buffer,
    state_buffer: wgpu::Buffer,
    staging_state: wgpu::Buffer,
    /// Partial rulers searched at once; each keeps its search state in a slot between passes.
    pub max_slots: u32,
    /// Most rulers one pass can return.
    pub max_results: u32,
}

impl GpuContext {
    pub async fn new(max_slots: u32, max_results: u32) -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Sparse Ruler Compute Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .ok()?;

        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage = |read_only| wgpu::BufferBindingType::Storage { read_only };
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sparse Ruler Bind Group Layout"),
            entries: &[
                entry(0, wgpu::BufferBindingType::Uniform),
                entry(1, storage(true)),
                entry(2, storage(true)),
                entry(3, storage(false)),
                entry(4, storage(false)),
                entry(5, storage(false)),
                entry(6, storage(false)),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sparse Ruler Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let buffer = |label, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        use wgpu::BufferUsages as U;
        let slots = max_slots as u64;
        let results_size = max_results as u64 * MAX_MARKS * 4;
        let params_buffer = buffer(
            "Params",
            std::mem::size_of::<SearchParams>() as u64,
            U::UNIFORM | U::COPY_DST,
        );
        let work_buffer = buffer("Work", slots * 2 * 4, U::STORAGE | U::COPY_DST);
        let prefix_buffer = buffer("Prefixes", slots * MAX_MARKS * 4, U::STORAGE | U::COPY_DST);
        let results_buffer = buffer("Results", results_size, U::STORAGE | U::COPY_SRC);
        let counter_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Counters"),
            contents: bytemuck::cast_slice(&[0u32; 4]),
            usage: U::STORAGE | U::COPY_SRC | U::COPY_DST,
        });
        let state_bytes = slots * STATE_STRIDE as u64 * 4;
        let state_buffer = buffer("Search State", state_bytes, U::STORAGE | U::COPY_SRC);
        let staging_state = buffer("Staging State", state_bytes, U::MAP_READ | U::COPY_DST);
        let finished_buffer = buffer("Finished Slots", slots * 4, U::STORAGE | U::COPY_SRC);
        let staging_counter = buffer("Staging Counters", 16, U::MAP_READ | U::COPY_DST);
        let staging_finished = buffer("Staging Finished", slots * 4, U::MAP_READ | U::COPY_DST);
        let staging_results = buffer("Staging Results", results_size, U::MAP_READ | U::COPY_DST);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sparse Ruler Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                &params_buffer,
                &work_buffer,
                &prefix_buffer,
                &results_buffer,
                &counter_buffer,
                &state_buffer,
                &finished_buffer,
            ]
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>(),
        });

        Some(Self {
            device,
            queue,
            pipeline_layout,
            pipelines: Mutex::new(HashMap::new()),
            bind_group,
            params_buffer,
            work_buffer,
            prefix_buffer,
            results_buffer,
            counter_buffer,
            finished_buffer,
            staging_counter,
            staging_finished,
            staging_results,
            state_buffer,
            staging_state,
            max_slots,
            max_results,
        })
    }

    fn pipeline(&self, words: u32, half_words: u32) -> wgpu::ComputePipeline {
        let mut pipelines = self.pipelines.lock().unwrap();
        pipelines
            .entry((words, half_words))
            .or_insert_with(|| {
                let source = include_str!("shader.wgsl")
                    .replace("__WORDS__", &words.to_string())
                    .replace("__HWORDS__", &half_words.to_string());
                let shader = self
                    .device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("Sparse Ruler Shader"),
                        source: wgpu::ShaderSource::Wgsl(source.into()),
                    });
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Sparse Ruler Compute Pipeline"),
                        layout: Some(&self.pipeline_layout),
                        module: &shader,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        cache: None,
                    })
            })
            .clone()
    }

    /// Maps the first `bytes` of each staging buffer and reads them, failing if the GPU does
    /// not finish within `PASS_TIMEOUT` (for example after the driver reset a hung GPU).
    fn read(&self, buffers: &[(&wgpu::Buffer, u64)]) -> Result<Vec<Vec<u32>>, String> {
        let (tx, rx) = std::sync::mpsc::channel();
        for (i, (buffer, bytes)) in buffers.iter().enumerate() {
            let tx = tx.clone();
            buffer
                .slice(..*bytes)
                .map_async(wgpu::MapMode::Read, move |r| tx.send((i, r)).unwrap());
        }
        let start = Instant::now();
        let mut mapped = 0;
        while mapped < buffers.len() {
            self.device.poll(wgpu::Maintain::Poll);
            match rx.recv_timeout(Duration::from_millis(1)) {
                Ok((_, Ok(()))) => mapped += 1,
                Ok((_, Err(e))) => return Err(format!("failed to read GPU results: {e}")),
                Err(_) if start.elapsed() > PASS_TIMEOUT => {
                    return Err(format!(
                        "the GPU did not finish a pass within {PASS_TIMEOUT:?}; it may have been reset"
                    ));
                }
                Err(_) => {}
            }
        }
        Ok(buffers
            .iter()
            .map(|(buffer, bytes)| {
                let slice = buffer.slice(..*bytes);
                let data = bytemuck::cast_slice::<u8, u32>(&slice.get_mapped_range()).to_vec();
                buffer.unmap();
                data
            })
            .collect())
    }

    /// Reads the saved search state of `slots` (`STATE_STRIDE` u32s each, in order).
    pub fn read_states(&self, slots: &[u32]) -> Result<Vec<u32>, String> {
        let stride = STATE_STRIDE as u64 * 4;
        let mut encoder = self.device.create_command_encoder(&Default::default());
        for (i, &slot) in slots.iter().enumerate() {
            encoder.copy_buffer_to_buffer(
                &self.state_buffer,
                slot as u64 * stride,
                &self.staging_state,
                i as u64 * stride,
                stride,
            );
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(self
            .read(&[(&self.staging_state, slots.len() as u64 * stride)])?
            .remove(0))
    }

    /// Runs one pass. `work` holds a (slot, prefix offset) pair per thread: the offset points
    /// into `prefixes`, at a partial ruler's depth followed by its marks, for a partial ruler
    /// starting in that slot, or is `RESUME` to continue the slot's saved search.
    pub fn run_pass(
        &self,
        params: &SearchParams,
        work: &[u32],
        prefixes: &[u32],
    ) -> Result<Pass, String> {
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[*params]));
        self.queue
            .write_buffer(&self.work_buffer, 0, bytemuck::cast_slice(work));
        if !prefixes.is_empty() {
            self.queue
                .write_buffer(&self.prefix_buffer, 0, bytemuck::cast_slice(prefixes));
        }
        self.queue
            .write_buffer(&self.counter_buffer, 0, bytemuck::cast_slice(&[0u32; 4]));

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline(params.words, params.length / 2 / 32 + 1));
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(params.num_work.div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.counter_buffer, 0, &self.staging_counter, 0, 16);
        self.queue.submit(Some(encoder.finish()));

        let counters = self.read(&[(&self.staging_counter, 16)])?.remove(0);
        let (found, finished) = (counters[0] as u64, counters[3] as u64);
        let nodes = counters[1] as u64 | (counters[2] as u64) << 32;

        // Copy back only the rulers and finished slots that were written.
        let result_bytes = found * params.num_segments as u64 * 4;
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let mut reads = Vec::new();
        if finished > 0 {
            encoder.copy_buffer_to_buffer(
                &self.finished_buffer,
                0,
                &self.staging_finished,
                0,
                finished * 4,
            );
            reads.push((&self.staging_finished, finished * 4));
        }
        if found > 0 {
            encoder.copy_buffer_to_buffer(
                &self.results_buffer,
                0,
                &self.staging_results,
                0,
                result_bytes,
            );
            reads.push((&self.staging_results, result_bytes));
        }
        if reads.is_empty() {
            return Ok(Pass {
                rulers: Vec::new(),
                finished: Vec::new(),
                nodes,
            });
        }
        self.queue.submit(Some(encoder.finish()));
        let mut data = self.read(&reads)?.into_iter();
        let finished = if finished > 0 {
            data.next().unwrap()
        } else {
            Vec::new()
        };
        let rulers = if found > 0 {
            data.next().unwrap()
        } else {
            Vec::new()
        };
        Ok(Pass {
            rulers,
            finished,
            nodes,
        })
    }
}
