//! Vulkan compute backend via `ash`.
//!
//! Compute-only subset of Vulkan:
//! - Instance → physical device → logical device → compute queue
//! - Storage buffer allocation via `gpu-allocator`
//! - Descriptor set / pipeline / command buffer management
//! - Single-shot dispatch with fence synchronization
//!
//! No render passes, no swapchains, no framebuffers.

use std::ffi::CStr;
use std::sync::{Arc, Mutex};

use ash::vk;
use gpu_allocator::MemoryLocation;

use crate::backend::{Backend, BackendBufferOps};
use crate::error::{backend_err, BackendOp, GpuError, Result};

/// Default fence timeout: 5 seconds (in nanoseconds).
const FENCE_TIMEOUT_NS: u64 = 5_000_000_000;

// ── Shared state (Arc'd between backend and buffers) ──

/// Persistent command buffer + fence + descriptor pool for single-shot dispatches.
/// Protected by a mutex so concurrent dispatch/upload/download are serialized.
struct SubmissionContext {
    fence: vk::Fence,
    cmd: vk::CommandBuffer,
    desc_pool: vk::DescriptorPool,
}

struct SharedState {
    device: ash::Device,
    /// Serializes queue submissions (both one-shot and batch paths).
    queue: Mutex<vk::Queue>,
    /// Serializes command buffer alloc/free from the shared pool.
    cmd_pool: Mutex<vk::CommandPool>,
    /// Serializes one-shot dispatch: record → submit → fence-wait.
    submit_ctx: Mutex<SubmissionContext>,
    /// Vulkan pipeline cache — speeds up `create_compute_pipelines` by
    /// reusing driver-compiled ISA across calls and across process restarts
    /// (persisted to disk on drop, loaded on init).
    pipeline_cache: vk::PipelineCache,
    /// Path to the on-disk cache file, if any.
    cache_path: Option<std::path::PathBuf>,
    allocator: std::mem::ManuallyDrop<Mutex<gpu_allocator::vulkan::Allocator>>,
    // Must outlive device — dropped after ManuallyDrop'd allocator + destroy_device.
    instance: ash::Instance,
    _entry: ash::Entry,
}

impl Drop for SharedState {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();

            // Persist pipeline cache to disk (best-effort).
            if let Some(path) = &self.cache_path {
                if let Ok(data) = self.device.get_pipeline_cache_data(self.pipeline_cache) {
                    if let Some(dir) = path.parent() {
                        let _ = std::fs::create_dir_all(dir);
                    }
                    let _ = std::fs::write(path, data);
                }
            }
            self.device
                .destroy_pipeline_cache(self.pipeline_cache, None);

            // &mut self guarantees exclusive access — no lock contention.
            let ctx = self.submit_ctx.get_mut().unwrap();
            self.device.destroy_descriptor_pool(ctx.desc_pool, None);
            self.device.destroy_fence(ctx.fence, None);
            // cmd is freed implicitly when cmd_pool is destroyed
            let cmd_pool = *self.cmd_pool.get_mut().unwrap();
            self.device.destroy_command_pool(cmd_pool, None);

            std::mem::ManuallyDrop::drop(&mut self.allocator);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

impl SharedState {
    /// Record, submit, and fence-wait using an already-locked [`SubmissionContext`].
    ///
    /// Callers that need the descriptor pool from `ctx` (e.g. `dispatch_pipeline`)
    /// lock `submit_ctx` themselves and call this directly.
    fn submit_with_ctx(
        &self,
        ctx: &SubmissionContext,
        record: impl FnOnce(vk::CommandBuffer),
    ) -> Result<()> {
        unsafe {
            self.device
                .reset_command_buffer(ctx.cmd, vk::CommandBufferResetFlags::empty())
                .map_err(|e| backend_err(BackendOp::ResetCommandBuffer, e))?;

            self.device
                .begin_command_buffer(
                    ctx.cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|e| backend_err(BackendOp::BeginCommandBuffer, e))?;

            record(ctx.cmd);

            self.device
                .end_command_buffer(ctx.cmd)
                .map_err(|e| backend_err(BackendOp::EndCommandBuffer, e))?;

            self.device
                .reset_fences(&[ctx.fence])
                .map_err(|e| backend_err(BackendOp::ResetFence, e))?;

            // Lock ordering: submit_ctx (held by caller) → queue.
            let queue = self
                .queue
                .lock()
                .map_err(|_| backend_err(BackendOp::MutexPoisoned, "queue"))?;

            self.device
                .queue_submit(
                    *queue,
                    &[vk::SubmitInfo::default().command_buffers(&[ctx.cmd])],
                    ctx.fence,
                )
                .map_err(|e| backend_err(BackendOp::QueueSubmit, e))?;

            drop(queue); // release queue lock before blocking on fence

            let wait = self
                .device
                .wait_for_fences(&[ctx.fence], true, FENCE_TIMEOUT_NS);
            match wait {
                Ok(()) => {}
                Err(vk::Result::TIMEOUT) => {
                    return Err(GpuError::ReadbackTimeout {
                        ms: FENCE_TIMEOUT_NS / 1_000_000,
                    })
                }
                Err(e) => return Err(backend_err(BackendOp::WaitFence, e)),
            }
        }

        Ok(())
    }

    /// Record a one-shot command buffer, submit it, and fence-wait.
    ///
    /// Acquires the submission lock, then delegates to [`submit_with_ctx`].
    fn one_shot_submit(&self, record: impl FnOnce(vk::CommandBuffer)) -> Result<()> {
        let ctx = self
            .submit_ctx
            .lock()
            .map_err(|_| backend_err(BackendOp::MutexPoisoned, "submission"))?;
        self.submit_with_ctx(&ctx, record)
    }
}

// ── Public types ──

/// Vulkan compute backend state.
pub struct VulkanBackend {
    state: Arc<SharedState>,
    device_name: String,
    device_memory: u64,
    subgroup_size: u32,
}

/// A buffer allocated on the Vulkan device.
pub struct VulkanBuffer {
    buffer: vk::Buffer,
    allocation: Option<gpu_allocator::vulkan::Allocation>,
    size: u64,
    state: Arc<SharedState>,
}

impl Drop for VulkanBuffer {
    fn drop(&mut self) {
        if let Some(alloc) = self.allocation.take() {
            if let Ok(mut a) = self.state.allocator.lock() {
                let _ = a.free(alloc);
            }
        }
        unsafe {
            self.state.device.destroy_buffer(self.buffer, None);
        }
    }
}

/// Inner Vulkan pipeline objects behind an `Arc` so that batches can
/// retain a reference and prevent destruction while recorded commands
/// are still pending.
pub(crate) struct VulkanKernelInner {
    pub(crate) shader_module: vk::ShaderModule,
    pub(crate) descriptor_set_layout: vk::DescriptorSetLayout,
    pub(crate) pipeline_layout: vk::PipelineLayout,
    pub(crate) pipeline: vk::Pipeline,
    state: Arc<SharedState>,
}

impl Drop for VulkanKernelInner {
    fn drop(&mut self) {
        // No device_wait_idle — callers guarantee no in-flight references:
        // • One-shot dispatches fence-wait before returning.
        // • Batches hold an Arc<VulkanKernelInner> until submitted + waited.
        unsafe {
            self.state.device.destroy_pipeline(self.pipeline, None);
            self.state
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.state
                .device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.state
                .device
                .destroy_shader_module(self.shader_module, None);
        }
    }
}

/// A compiled Vulkan compute pipeline, reusable across dispatches.
pub struct VulkanKernel {
    pub(crate) inner: Arc<VulkanKernelInner>,
}

// ── Backend trait impl ──

impl Backend for VulkanBackend {
    type Buffer = VulkanBuffer;
    type Pipeline = VulkanKernel;

    fn create() -> Result<Self> {
        unsafe { Self::init() }
    }

    fn upload(&self, data: &[u8]) -> Result<Self::Buffer> {
        let size = data.len() as u64;
        if size == 0 {
            return self.alloc(4); // Vulkan needs non-zero size
        }

        // Device-local storage buffer
        let (storage_buf, storage_alloc) = self.create_buffer(
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::GpuOnly,
            "storage",
        )?;

        // Host-visible staging buffer
        let (staging_buf, staging_alloc) = self.create_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            "staging_upload",
        )?;

        // Copy data into staging
        Self::write_mapped(&staging_alloc, data)?;

        // Transfer staging → storage
        self.copy_buffer(staging_buf, storage_buf, size)?;

        // Free staging
        self.free_buffer(staging_buf, staging_alloc)?;

        Ok(VulkanBuffer {
            buffer: storage_buf,
            allocation: Some(storage_alloc),
            size,
            state: Arc::clone(&self.state),
        })
    }

    fn alloc(&self, size: u64) -> Result<Self::Buffer> {
        let actual = size.max(4); // Vulkan requires non-zero

        let (buffer, allocation) = self.create_buffer(
            actual,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::GpuOnly,
            "storage",
        )?;

        Ok(VulkanBuffer {
            buffer,
            allocation: Some(allocation),
            size,
            state: Arc::clone(&self.state),
        })
    }

    #[allow(clippy::too_many_lines)]
    fn dispatch(
        &self,
        spirv: &[u32],
        entry_point: &str,
        buffers: &[&Self::Buffer],
        workgroups: [u32; 3],
        push_constants: Option<&[u8]>,
    ) -> Result<()> {
        let device = &self.state.device;

        unsafe {
            // Shader module
            let shader_module = device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::default().code(spirv),
                    None,
                )
                .map_err(|e| backend_err(BackendOp::CreateShaderModule, e))?;

            // Descriptor set layout: N storage buffers
            let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..buffers.len())
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();

            let desc_layout = device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .map_err(|e| backend_err(BackendOp::CreateDescriptorSetLayout, e))?;

            // Pipeline layout (+ optional push constants)
            let pc_ranges;
            let mut layout_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&desc_layout));

            if let Some(pc) = push_constants {
                pc_ranges = [vk::PushConstantRange {
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                    offset: 0,
                    size: pc.len() as u32,
                }];
                layout_info = layout_info.push_constant_ranges(&pc_ranges);
            }

            let pipeline_layout = device
                .create_pipeline_layout(&layout_info, None)
                .map_err(|e| backend_err(BackendOp::CreatePipelineLayout, e))?;

            // Compute pipeline
            let entry_name = std::ffi::CString::new(entry_point)
                .map_err(|e| backend_err(BackendOp::CreatePipeline, e))?;

            let pipeline_info = vk::ComputePipelineCreateInfo::default()
                .layout(pipeline_layout)
                .stage(
                    vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk::ShaderStageFlags::COMPUTE)
                        .module(shader_module)
                        .name(&entry_name),
                );

            let pipeline = device
                .create_compute_pipelines(
                    self.state.pipeline_cache,
                    &[pipeline_info],
                    None,
                )
                .map_err(|(_, e)| backend_err(BackendOp::CreatePipeline, e))?[0];

            // Descriptor pool + set
            let desc_pool = device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(&[vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: buffers.len().max(1) as u32,
                        }]),
                    None,
                )
                .map_err(|e| backend_err(BackendOp::CreateDescriptorPool, e))?;

            let desc_set = device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(desc_pool)
                        .set_layouts(std::slice::from_ref(&desc_layout)),
                )
                .map_err(|e| backend_err(BackendOp::AllocDescriptorSet, e))?[0];

            // Write buffer bindings
            let buf_infos: Vec<vk::DescriptorBufferInfo> = buffers
                .iter()
                .map(|b| vk::DescriptorBufferInfo {
                    buffer: b.buffer,
                    offset: 0,
                    range: vk::WHOLE_SIZE,
                })
                .collect();

            let writes: Vec<vk::WriteDescriptorSet> = buf_infos
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(desc_set)
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(info))
                })
                .collect();

            device.update_descriptor_sets(&writes, &[]);

            // Record + submit
            self.state.one_shot_submit(|cmd| {
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);

                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline_layout,
                    0,
                    &[desc_set],
                    &[],
                );

                if let Some(pc) = push_constants {
                    device.cmd_push_constants(
                        cmd,
                        pipeline_layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        pc,
                    );
                }

                device.cmd_dispatch(cmd, workgroups[0], workgroups[1], workgroups[2]);
            })?;

            // Cleanup transient objects
            device.destroy_pipeline(pipeline, None);
            device.destroy_pipeline_layout(pipeline_layout, None);
            device.destroy_descriptor_pool(desc_pool, None);
            device.destroy_descriptor_set_layout(desc_layout, None);
            device.destroy_shader_module(shader_module, None);
        }

        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn create_pipeline(
        &self,
        spirv: &[u32],
        entry_point: &str,
        binding_count: usize,
        push_constant_size: u32,
    ) -> Result<VulkanKernel> {
        let device = &self.state.device;

        unsafe {
            // Shader module
            let shader_module = device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::default().code(spirv),
                    None,
                )
                .map_err(|e| backend_err(BackendOp::CreateShaderModule, e))?;

            // Descriptor set layout: N storage buffers
            let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..binding_count)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();

            let descriptor_set_layout = device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .map_err(|e| {
                    device.destroy_shader_module(shader_module, None);
                    backend_err(BackendOp::CreateDescriptorSetLayout, e)
                })?;

            // Pipeline layout (with optional push constant range)
            let pc_ranges = if push_constant_size > 0 {
                vec![vk::PushConstantRange {
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                    offset: 0,
                    size: push_constant_size,
                }]
            } else {
                vec![]
            };

            let mut layout_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&descriptor_set_layout));

            if !pc_ranges.is_empty() {
                layout_info = layout_info.push_constant_ranges(&pc_ranges);
            }

            let pipeline_layout = device
                .create_pipeline_layout(&layout_info, None)
                .map_err(|e| {
                    device.destroy_descriptor_set_layout(descriptor_set_layout, None);
                    device.destroy_shader_module(shader_module, None);
                    backend_err(BackendOp::CreatePipelineLayout, e)
                })?;

            // Compute pipeline
            let entry_name = std::ffi::CString::new(entry_point).map_err(|e| {
                device.destroy_pipeline_layout(pipeline_layout, None);
                device.destroy_descriptor_set_layout(descriptor_set_layout, None);
                device.destroy_shader_module(shader_module, None);
                backend_err(BackendOp::CreatePipeline, e)
            })?;

            let pipeline_info = vk::ComputePipelineCreateInfo::default()
                .layout(pipeline_layout)
                .stage(
                    vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk::ShaderStageFlags::COMPUTE)
                        .module(shader_module)
                        .name(&entry_name),
                );

            let pipeline = device
                .create_compute_pipelines(
                    self.state.pipeline_cache,
                    &[pipeline_info],
                    None,
                )
                .map_err(|(_, e)| {
                    device.destroy_pipeline_layout(pipeline_layout, None);
                    device.destroy_descriptor_set_layout(descriptor_set_layout, None);
                    device.destroy_shader_module(shader_module, None);
                    backend_err(BackendOp::CreatePipeline, e)
                })?[0];

            Ok(VulkanKernel {
                inner: Arc::new(VulkanKernelInner {
                    shader_module,
                    descriptor_set_layout,
                    pipeline_layout,
                    pipeline,
                    state: Arc::clone(&self.state),
                }),
            })
        }
    }

    fn dispatch_pipeline(
        &self,
        kernel: &VulkanKernel,
        buffers: &[&Self::Buffer],
        workgroups: [u32; 3],
        push_constants: Option<&[u8]>,
    ) -> Result<()> {
        let device = &self.state.device;

        // Lock submit_ctx for the descriptor pool AND the command buffer.
        let ctx = self
            .state
            .submit_ctx
            .lock()
            .map_err(|_| backend_err(BackendOp::MutexPoisoned, "submission"))?;

        unsafe {
            // Reset persistent pool (safe: previous fence was waited on)
            device
                .reset_descriptor_pool(
                    ctx.desc_pool,
                    vk::DescriptorPoolResetFlags::empty(),
                )
                .map_err(|e| backend_err(BackendOp::ResetDescriptorPool, e))?;

            let desc_set = device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(ctx.desc_pool)
                        .set_layouts(std::slice::from_ref(
                            &kernel.inner.descriptor_set_layout,
                        )),
                )
                .map_err(|e| backend_err(BackendOp::AllocDescriptorSet, e))?[0];

            // Write buffer bindings
            let buf_infos: Vec<vk::DescriptorBufferInfo> = buffers
                .iter()
                .map(|b| vk::DescriptorBufferInfo {
                    buffer: b.buffer,
                    offset: 0,
                    range: vk::WHOLE_SIZE,
                })
                .collect();

            let writes: Vec<vk::WriteDescriptorSet> = buf_infos
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(desc_set)
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(info))
                })
                .collect();

            device.update_descriptor_sets(&writes, &[]);

            // Record + submit (ctx already locked, use submit_with_ctx)
            let pl = kernel.inner.pipeline_layout;
            let pipe = kernel.inner.pipeline;
            self.state.submit_with_ctx(&ctx, |cmd| {
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe);

                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    pl,
                    0,
                    &[desc_set],
                    &[],
                );

                if let Some(pc) = push_constants {
                    device.cmd_push_constants(
                        cmd,
                        pl,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        pc,
                    );
                }

                device.cmd_dispatch(cmd, workgroups[0], workgroups[1], workgroups[2]);
            })?;
        }

        Ok(())
    }

    fn device_name(&self) -> &str {
        &self.device_name
    }

    fn device_memory(&self) -> u64 {
        self.device_memory
    }

    fn subgroup_size(&self) -> u32 {
        self.subgroup_size
    }
}

// ── Pipeline cache helpers ──

/// Returns `~/.cache/scry-gpu/<vendor_id>-<device_id>.bin`, or `None` if
/// the home directory cannot be determined.
fn cache_path(props: &vk::PhysicalDeviceProperties) -> Option<std::path::PathBuf> {
    let home = std::env::var_os("HOME")?;
    let mut p = std::path::PathBuf::from(home);
    p.push(".cache");
    p.push("scry-gpu");
    p.push(format!("{:04x}-{:04x}.bin", props.vendor_id, props.device_id));
    Some(p)
}

/// Try to load an existing pipeline cache blob, or create an empty one.
unsafe fn create_pipeline_cache(
    device: &ash::Device,
    path: &Option<std::path::PathBuf>,
) -> vk::PipelineCache {
    let data = path.as_ref().and_then(|p| std::fs::read(p).ok());
    let info = match &data {
        Some(blob) => vk::PipelineCacheCreateInfo::default().initial_data(blob),
        None => vk::PipelineCacheCreateInfo::default(),
    };
    // If loading fails (e.g. stale blob), fall back to empty cache.
    device
        .create_pipeline_cache(&info, None)
        .or_else(|_| device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None))
        .unwrap_or(vk::PipelineCache::null())
}

// ── Initialization ──

impl VulkanBackend {
    unsafe fn init() -> Result<Self> {
        let entry = ash::Entry::linked();

        // Instance — compute-only, no surface extensions
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"scry-gpu")
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(c"scry-gpu")
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_2);

        let instance = entry
            .create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app_info),
                None,
            )
            .map_err(|e| GpuError::BackendUnavailable(format!("vk instance: {e}")))?;

        // Physical device — prefer discrete, fall back to any
        let phys_devs = instance
            .enumerate_physical_devices()
            .map_err(|e| GpuError::BackendUnavailable(format!("enumerate: {e}")))?;

        if phys_devs.is_empty() {
            return Err(GpuError::NoDevice);
        }

        let pick = |ty| {
            phys_devs.iter().find(|&&pd| {
                instance
                    .get_physical_device_properties(pd)
                    .device_type
                    == ty
            })
        };

        let &physical_device = pick(vk::PhysicalDeviceType::DISCRETE_GPU)
            .or_else(|| pick(vk::PhysicalDeviceType::INTEGRATED_GPU))
            .unwrap_or(&phys_devs[0]);

        let props = instance.get_physical_device_properties(physical_device);
        let device_name = CStr::from_ptr(props.device_name.as_ptr())
            .to_string_lossy()
            .into_owned();

        // Query subgroup properties (core in Vulkan 1.1+)
        let mut subgroup_props = vk::PhysicalDeviceSubgroupProperties::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut subgroup_props);
        instance.get_physical_device_properties2(physical_device, &mut props2);
        let subgroup_size = subgroup_props.subgroup_size;

        let mem_props = instance.get_physical_device_memory_properties(physical_device);
        let device_memory: u64 = mem_props.memory_heaps
            [..mem_props.memory_heap_count as usize]
            .iter()
            .filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|h| h.size)
            .sum();

        // Compute queue family
        let queue_families =
            instance.get_physical_device_queue_family_properties(physical_device);

        let qf_index = queue_families
            .iter()
            .position(|qf| qf.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .ok_or_else(|| GpuError::BackendUnavailable("no compute queue".into()))?
            as u32;

        // Logical device + queue
        let queue_priorities = [1.0f32];
        let device = instance
            .create_device(
                physical_device,
                &vk::DeviceCreateInfo::default().queue_create_infos(&[
                    vk::DeviceQueueCreateInfo::default()
                        .queue_family_index(qf_index)
                        .queue_priorities(&queue_priorities),
                ]),
                None,
            )
            .map_err(|e| GpuError::BackendUnavailable(format!("create device: {e}")))?;

        let queue = device.get_device_queue(qf_index, 0);

        // Pipeline cache — load from disk or create empty.
        let cp = cache_path(&props);
        let pipeline_cache = create_pipeline_cache(&device, &cp);

        // Memory allocator
        let allocator = gpu_allocator::vulkan::Allocator::new(
            &gpu_allocator::vulkan::AllocatorCreateDesc {
                instance: instance.clone(),
                device: device.clone(),
                physical_device,
                debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
                buffer_device_address: false,
                allocation_sizes: gpu_allocator::AllocationSizes::default(),
            },
        )
        .map_err(|e| backend_err(BackendOp::CreateAllocator, e))?;

        // Command pool
        let cmd_pool = device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(qf_index)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
            .map_err(|e| backend_err(BackendOp::CreateCommandPool, e))?;

        // Persistent dispatch resources — reused across calls
        let fence = device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .map_err(|e| backend_err(BackendOp::CreateFence, e))?;

        let cmd = device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .map_err(|e| backend_err(BackendOp::AllocCommandBuffer, e))?[0];

        let desc_pool = device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&[vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::STORAGE_BUFFER,
                        descriptor_count: 16,
                    }]),
                None,
            )
            .map_err(|e| backend_err(BackendOp::CreateDescriptorPool, e))?;

        let state = Arc::new(SharedState {
            device,
            queue: Mutex::new(queue),
            cmd_pool: Mutex::new(cmd_pool),
            submit_ctx: Mutex::new(SubmissionContext { fence, cmd, desc_pool }),
            pipeline_cache,
            cache_path: cp,
            allocator: std::mem::ManuallyDrop::new(Mutex::new(allocator)),
            instance,
            _entry: entry,
        });

        Ok(Self {
            state,
            device_name,
            device_memory,
            subgroup_size,
        })
    }
}

// ── Batch dispatch ──

/// A command buffer being recorded with multiple dispatches.
///
/// Created via [`VulkanBackend::begin_batch`]. Records dispatches into a
/// single command buffer, then submits them all at once with a single fence.
///
/// Owns its own fence and command buffer — fully independent of the
/// shared one-shot submission path, so batches and single dispatches
/// can coexist safely across threads.
pub struct VulkanBatch {
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    desc_pool: vk::DescriptorPool,
    overflow_pools: Vec<vk::DescriptorPool>,
    sets_allocated: u32,
    pool_capacity: u32,
    /// Keeps kernel pipelines alive while the batch's command buffer
    /// references them. Dropped after the batch is submitted + waited.
    retained_kernels: Vec<Arc<VulkanKernelInner>>,
    state: Arc<SharedState>,
}

const BATCH_POOL_SETS: u32 = 64;
const BATCH_POOL_DESCRIPTORS: u32 = BATCH_POOL_SETS * 16;

impl VulkanBatch {
    fn current_pool(&mut self) -> Result<vk::DescriptorPool> {
        if self.sets_allocated < self.pool_capacity {
            return Ok(self.desc_pool);
        }
        // Current pool exhausted — retire it and allocate a new one.
        let new_pool = unsafe {
            self.state
                .device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(BATCH_POOL_SETS)
                        .pool_sizes(&[vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: BATCH_POOL_DESCRIPTORS,
                        }]),
                    None,
                )
                .map_err(|e| backend_err(BackendOp::CreateDescriptorPool, e))?
        };
        let old = std::mem::replace(&mut self.desc_pool, new_pool);
        self.overflow_pools.push(old);
        self.sets_allocated = 0;
        Ok(self.desc_pool)
    }

    pub fn record_dispatch(
        &mut self,
        kernel: &VulkanKernel,
        buffers: &[&VulkanBuffer],
        workgroups: [u32; 3],
        push_constants: Option<&[u8]>,
    ) -> Result<()> {
        // Retain kernel so its pipeline stays alive until this batch is done.
        self.retained_kernels.push(Arc::clone(&kernel.inner));

        let pool = self.current_pool()?;
        let device = &self.state.device;

        unsafe {
            let desc_set = device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(pool)
                        .set_layouts(std::slice::from_ref(
                            &kernel.inner.descriptor_set_layout,
                        )),
                )
                .map_err(|e| backend_err(BackendOp::AllocDescriptorSet, e))?[0];

            self.sets_allocated += 1;

            let buf_infos: Vec<vk::DescriptorBufferInfo> = buffers
                .iter()
                .map(|b| vk::DescriptorBufferInfo {
                    buffer: b.buffer,
                    offset: 0,
                    range: vk::WHOLE_SIZE,
                })
                .collect();

            let writes: Vec<vk::WriteDescriptorSet> = buf_infos
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(desc_set)
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(info))
                })
                .collect();

            device.update_descriptor_sets(&writes, &[]);

            device.cmd_bind_pipeline(
                self.cmd,
                vk::PipelineBindPoint::COMPUTE,
                kernel.inner.pipeline,
            );

            device.cmd_bind_descriptor_sets(
                self.cmd,
                vk::PipelineBindPoint::COMPUTE,
                kernel.inner.pipeline_layout,
                0,
                &[desc_set],
                &[],
            );

            if let Some(pc) = push_constants {
                device.cmd_push_constants(
                    self.cmd,
                    kernel.inner.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    pc,
                );
            }

            device.cmd_dispatch(self.cmd, workgroups[0], workgroups[1], workgroups[2]);
        }

        Ok(())
    }

    pub fn record_barrier(&mut self) {
        let device = &self.state.device;
        unsafe {
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);

            device.cmd_pipeline_barrier(
                self.cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
    }

    pub fn submit(self) -> Result<()> {
        let device = &self.state.device;

        unsafe {
            device
                .end_command_buffer(self.cmd)
                .map_err(|e| backend_err(BackendOp::EndCommandBuffer, e))?;

            // Own fence — independent of the shared one-shot path.
            device
                .reset_fences(&[self.fence])
                .map_err(|e| backend_err(BackendOp::ResetFence, e))?;

            let queue = self
                .state
                .queue
                .lock()
                .map_err(|_| backend_err(BackendOp::MutexPoisoned, "queue"))?;

            device
                .queue_submit(
                    *queue,
                    &[vk::SubmitInfo::default().command_buffers(&[self.cmd])],
                    self.fence,
                )
                .map_err(|e| backend_err(BackendOp::QueueSubmit, e))?;

            drop(queue);

            let wait = device.wait_for_fences(&[self.fence], true, FENCE_TIMEOUT_NS);
            match wait {
                Ok(()) => {}
                Err(vk::Result::TIMEOUT) => {
                    return Err(GpuError::ReadbackTimeout {
                        ms: FENCE_TIMEOUT_NS / 1_000_000,
                    })
                }
                Err(e) => return Err(backend_err(BackendOp::WaitFence, e)),
            }
        }

        Ok(())
    }
}

impl Drop for VulkanBatch {
    fn drop(&mut self) {
        unsafe {
            let device = &self.state.device;
            if let Ok(cmd_pool) = self.state.cmd_pool.lock() {
                device.free_command_buffers(*cmd_pool, &[self.cmd]);
            }
            device.destroy_fence(self.fence, None);
            device.destroy_descriptor_pool(self.desc_pool, None);
            for pool in &self.overflow_pools {
                device.destroy_descriptor_pool(*pool, None);
            }
        }
    }
}

impl VulkanBackend {
    pub fn begin_batch(&self) -> Result<VulkanBatch> {
        let device = &self.state.device;

        // Lock cmd_pool for the allocation only.
        let cmd = {
            let cmd_pool = self
                .state
                .cmd_pool
                .lock()
                .map_err(|_| backend_err(BackendOp::MutexPoisoned, "cmd pool"))?;
            unsafe {
                device
                    .allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_pool(*cmd_pool)
                            .level(vk::CommandBufferLevel::PRIMARY)
                            .command_buffer_count(1),
                    )
                    .map_err(|e| backend_err(BackendOp::AllocCommandBuffer, e))?[0]
            }
        };

        // Own fence — independent of the shared one-shot path.
        let fence = unsafe {
            device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(|e| backend_err(BackendOp::CreateFence, e))?
        };

        unsafe {
            device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|e| backend_err(BackendOp::BeginCommandBuffer, e))?;
        }

        let desc_pool = unsafe {
            device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(BATCH_POOL_SETS)
                        .pool_sizes(&[vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: BATCH_POOL_DESCRIPTORS,
                        }]),
                    None,
                )
                .map_err(|e| backend_err(BackendOp::CreateDescriptorPool, e))?
        };

        Ok(VulkanBatch {
            cmd,
            fence,
            desc_pool,
            overflow_pools: Vec::new(),
            sets_allocated: 0,
            pool_capacity: BATCH_POOL_SETS,
            retained_kernels: Vec::new(),
            state: Arc::clone(&self.state),
        })
    }
}

// ── Buffer helpers ──

impl VulkanBackend {
    fn create_buffer(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
        name: &str,
    ) -> Result<(vk::Buffer, gpu_allocator::vulkan::Allocation)> {
        let device = &self.state.device;

        let buffer = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        }
        .map_err(|e| backend_err(BackendOp::CreateBuffer, e))?;

        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let allocation = self
            .state
            .allocator
            .lock()
            .map_err(|_| {
                unsafe { device.destroy_buffer(buffer, None) };
                backend_err(BackendOp::MutexPoisoned, "allocator")
            })?
            .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                name,
                requirements,
                location,
                linear: true,
                allocation_scheme:
                    gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(|_| {
                // Clean up the buffer on allocation failure
                unsafe { device.destroy_buffer(buffer, None) };
                GpuError::AllocationFailed {
                    requested: size,
                    device_max: self.device_memory,
                }
            })?;

        unsafe { device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()) }
            .map_err(|e| backend_err(BackendOp::BindMemory, e))?;

        Ok((buffer, allocation))
    }

    fn write_mapped(
        alloc: &gpu_allocator::vulkan::Allocation,
        data: &[u8],
    ) -> Result<()> {
        let ptr = alloc
            .mapped_ptr()
            .ok_or_else(|| backend_err(BackendOp::MapMemory, "staging buffer not mappable"))?;

        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                ptr.as_ptr().cast::<u8>(),
                data.len(),
            );
        }

        Ok(())
    }

    fn copy_buffer(&self, src: vk::Buffer, dst: vk::Buffer, size: u64) -> Result<()> {
        let device = &self.state.device;
        self.state.one_shot_submit(|cmd| unsafe {
            device.cmd_copy_buffer(
                cmd,
                src,
                dst,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size,
                }],
            );
        })
    }

    fn free_buffer(
        &self,
        buffer: vk::Buffer,
        alloc: gpu_allocator::vulkan::Allocation,
    ) -> Result<()> {
        self.state
            .allocator
            .lock()
            .map_err(|_| backend_err(BackendOp::MutexPoisoned, "allocator"))?
            .free(alloc)
            .map_err(|e| backend_err(BackendOp::FreeMemory, e))?;
        unsafe { self.state.device.destroy_buffer(buffer, None) };
        Ok(())
    }
}

// ── Buffer readback ──

impl BackendBufferOps for VulkanBuffer {
    fn read_back(&self) -> Result<Vec<u8>> {
        if self.size == 0 {
            return Ok(Vec::new());
        }

        let device = &self.state.device;

        // Host-visible staging buffer for readback
        let staging_buf = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(self.size)
                    .usage(vk::BufferUsageFlags::TRANSFER_DST)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        }
        .map_err(|e| backend_err(BackendOp::CreateBuffer, e))?;

        let requirements = unsafe { device.get_buffer_memory_requirements(staging_buf) };

        let staging_alloc = self
            .state
            .allocator
            .lock()
            .map_err(|_| {
                unsafe { device.destroy_buffer(staging_buf, None) };
                backend_err(BackendOp::MutexPoisoned, "allocator")
            })?
            .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                name: "staging_readback",
                requirements,
                location: MemoryLocation::GpuToCpu,
                linear: true,
                allocation_scheme:
                    gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(|e| backend_err(BackendOp::CreateBuffer, e))?;

        unsafe {
            device.bind_buffer_memory(
                staging_buf,
                staging_alloc.memory(),
                staging_alloc.offset(),
            )
        }
        .map_err(|e| backend_err(BackendOp::BindMemory, e))?;

        // Copy device → staging
        let src = self.buffer;
        let size = self.size;
        self.state.one_shot_submit(|cmd| unsafe {
            device.cmd_copy_buffer(
                cmd,
                src,
                staging_buf,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size,
                }],
            );
        })?;

        // Read mapped memory
        let ptr = staging_alloc
            .mapped_ptr()
            .ok_or_else(|| backend_err(BackendOp::MapMemory, "readback not mappable"))?;

        let mut data = vec![0u8; self.size as usize];
        unsafe {
            std::ptr::copy_nonoverlapping(
                ptr.as_ptr().cast::<u8>(),
                data.as_mut_ptr(),
                self.size as usize,
            );
        }

        // Cleanup staging
        if let Ok(mut a) = self.state.allocator.lock() {
            let _ = a.free(staging_alloc);
        }
        unsafe { device.destroy_buffer(staging_buf, None) };

        Ok(data)
    }

    fn byte_size(&self) -> u64 {
        self.size
    }
}
