use bytemuck::{Pod, Zeroable};
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SearchParams {
    pub length: u32,
    pub num_segments: u32,
    pub batch_size: u32, // Number of threads
    pub start_rank_low: u32,
    pub start_rank_high: u32,
    pub steps_per_thread: u32,
    pub _padding: [u32; 2],
}

pub struct GpuConfig {
    pub steps_per_thread: u32,
    pub threads_per_batch: u32,
}

impl GpuConfig {
    pub fn for_length(length: u32) -> Self {
        if length < 70 {
            // Small lengths: Low complexity, reduce launch overhead
            Self {
                steps_per_thread: 1024,
                threads_per_batch: 131072,
            }
        } else if length < 110 {
            // Medium lengths: Divergence starts, lower steps for better scheduling
            Self {
                steps_per_thread: 128,
                threads_per_batch: 1048576,
            }
        } else {
            // Large lengths: Maximize throughput with very large batches
            Self {
                steps_per_thread: 128,
                threads_per_batch: 2097152,
            }
        }
    }
}

pub struct GpuBuffers {
    pub params_buffer: wgpu::Buffer,
    pub results_buffer: wgpu::Buffer,
    pub counter_buffer: wgpu::Buffer,
    pub staging_results: wgpu::Buffer,
    pub staging_counter: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub binomial_buffer: wgpu::Buffer,
    pub buffer_pool: Mutex<VecDeque<GpuBuffers>>,
    pub max_results: u32,
}

pub struct GpuSearchTask {
    pub r_receiver: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    pub c_receiver: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    pub bufs: GpuBuffers,
}

impl GpuContext {
    pub async fn new(max_results: u32) -> Option<Self> {
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

        // Generate Binomial Table for n=0..255, k=0..32
        // Stored as 256 rows of 33 u64 values
        let mut table = Vec::new();
        for n in 0..=255 {
            for k in 0..=32 {
                table.push(binomial_u64(n as u64, k as u64));
            }
        }

        let binomial_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Binomial Buffer"),
            contents: bytemuck::cast_slice(&table),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let shader_src = include_str!("shader.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sparse Ruler Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sparse Ruler Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sparse Ruler Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Sparse Ruler Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Some(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            pipeline,
            bind_group_layout,
            binomial_buffer,
            buffer_pool: Mutex::new(VecDeque::new()),
            max_results,
        })
    }

    fn get_buffers(&self) -> GpuBuffers {
        if let Some(bufs) = self.buffer_pool.lock().unwrap().pop_front() {
            return bufs;
        }

        let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Params Buffer"),
            size: std::mem::size_of::<SearchParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let results_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Results Buffer"),
            size: (self.max_results as u64) * 8, // u64 ranks
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let counter_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Counter Buffer"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_results = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Results"),
            size: (self.max_results as u64) * 8,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_counter = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Counter"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sparse Ruler Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.binomial_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: results_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: counter_buffer.as_entire_binding(),
                },
            ],
        });

        GpuBuffers {
            params_buffer,
            results_buffer,
            counter_buffer,
            staging_results,
            staging_counter,
            bind_group,
        }
    }

    fn return_buffers(&self, bufs: GpuBuffers) {
        self.buffer_pool.lock().unwrap().push_back(bufs);
    }

    pub fn submit_search(&self, params: &SearchParams) -> GpuSearchTask {
        let bufs = self.get_buffers();

        self.queue
            .write_buffer(&bufs.params_buffer, 0, bytemuck::cast_slice(&[*params]));
        self.queue
            .write_buffer(&bufs.counter_buffer, 0, bytemuck::cast_slice(&[0u32]));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Compute Encoder"),
            });

        let workgroup_size: u32 = std::env::var("GPU_WORKGROUP_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(256);

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bufs.bind_group, &[]);
            let workgroup_count = params.batch_size.div_ceil(workgroup_size);
            if workgroup_count <= 65535 {
                compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
            } else {
                let x_groups = 65535;
                let y_groups = workgroup_count.div_ceil(65535);
                compute_pass.dispatch_workgroups(x_groups, y_groups, 1);
            }
        }

        encoder.copy_buffer_to_buffer(
            &bufs.results_buffer,
            0,
            &bufs.staging_results,
            0,
            (self.max_results as u64) * 8,
        );
        encoder.copy_buffer_to_buffer(&bufs.counter_buffer, 0, &bufs.staging_counter, 0, 4);

        self.queue.submit(Some(encoder.finish()));

        let results_slice = bufs.staging_results.slice(..);
        let counter_slice = bufs.staging_counter.slice(..);

        let (r_sender, r_receiver) = std::sync::mpsc::channel();
        let (c_sender, c_receiver) = std::sync::mpsc::channel();

        results_slice.map_async(wgpu::MapMode::Read, move |v| r_sender.send(v).unwrap());
        counter_slice.map_async(wgpu::MapMode::Read, move |v| c_sender.send(v).unwrap());

        GpuSearchTask {
            r_receiver,
            c_receiver,
            bufs,
        }
    }

    pub fn wait_for_search(&self, task: GpuSearchTask) -> Vec<u64> {
        self.device.poll(wgpu::Maintain::Wait);

        let results =
            if let (Ok(Ok(())), Ok(Ok(()))) = (task.c_receiver.recv(), task.r_receiver.recv()) {
                let counter_slice = task.bufs.staging_counter.slice(..);
                let counter_data = counter_slice.get_mapped_range();
                let count = bytemuck::cast_slice::<u8, u32>(&counter_data)[0] as usize;
                drop(counter_data);
                task.bufs.staging_counter.unmap();

                let results_slice = task
                    .bufs
                    .staging_results
                    .slice(0..(self.max_results as u64 * 8));
                let results_data = results_slice.get_mapped_range();
                let mut ranks = bytemuck::cast_slice::<u8, u64>(&results_data)
                    [..count.min(self.max_results as usize)]
                    .to_vec();
                drop(results_data);
                task.bufs.staging_results.unmap();

                ranks.sort_unstable();
                ranks.dedup();
                ranks
            } else {
                Vec::new()
            };

        self.return_buffers(task.bufs);
        results
    }
}

fn binomial_u64(n: u64, k: u64) -> u64 {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k);
    let mut res = 1.0f64;
    for i in 1..=k {
        res = res * (n - i + 1) as f64 / i as f64;
    }
    res.round() as u64
}
