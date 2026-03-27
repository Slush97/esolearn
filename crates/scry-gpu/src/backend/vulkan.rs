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
use crate::error::{GpuError, Result};

// ── Shared state (Arc'd between backend and buffers) ──

struct SharedState {
    device: ash::Device,
    queue: vk::Queue,
    cmd_pool: vk::CommandPool,
    allocator: std::mem::ManuallyDrop<Mutex<gpu_allocator::vulkan::Allocator>>,
    // Must outlive device — dropped after ManuallyDrop'd allocator + destroy_device.
    instance: ash::Instance,
    _entry: ash::Entry,
}

impl Drop for SharedState {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_command_pool(self.cmd_pool, None);
            std::mem::ManuallyDrop::drop(&mut self.allocator);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

impl SharedState {
    /// Record a one-shot command buffer, submit it, and fence-wait.
    fn one_shot_submit(&self, record: impl FnOnce(vk::CommandBuffer)) -> Result<()> {
        unsafe {
            let cmd = self
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.cmd_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .map_err(|e| GpuError::Backend(format!("allocate cmd buf: {e}")))?[0];

            self.device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|e| GpuError::Backend(format!("begin cmd buf: {e}")))?;

            record(cmd);

            self.device
                .end_command_buffer(cmd)
                .map_err(|e| GpuError::Backend(format!("end cmd buf: {e}")))?;

            let fence = self
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(|e| GpuError::Backend(format!("create fence: {e}")))?;

            self.device
                .queue_submit(
                    self.queue,
                    &[vk::SubmitInfo::default().command_buffers(&[cmd])],
                    fence,
                )
                .map_err(|e| GpuError::Backend(format!("queue submit: {e}")))?;

            self.device
                .wait_for_fences(&[fence], true, u64::MAX)
                .map_err(|e| GpuError::Backend(format!("wait fence: {e}")))?;

            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(self.cmd_pool, &[cmd]);
        }

        Ok(())
    }
}

// ── Public types ──

/// Vulkan compute backend state.
pub struct VulkanBackend {
    state: Arc<SharedState>,
    device_name: String,
    device_memory: u64,
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
            let _ = self.state.allocator.lock().unwrap().free(alloc);
        }
        unsafe {
            self.state.device.destroy_buffer(self.buffer, None);
        }
    }
}

/// A compiled Vulkan compute pipeline, reusable across dispatches.
pub struct VulkanKernel {
    shader_module: vk::ShaderModule,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    binding_count: usize,
    state: Arc<SharedState>,
}

impl Drop for VulkanKernel {
    fn drop(&mut self) {
        unsafe {
            let _ = self.state.device.device_wait_idle();
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
                .map_err(|e| GpuError::Backend(format!("shader module: {e}")))?;

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
                .map_err(|e| GpuError::Backend(format!("desc set layout: {e}")))?;

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
                .map_err(|e| GpuError::Backend(format!("pipeline layout: {e}")))?;

            // Compute pipeline
            let entry_name = std::ffi::CString::new(entry_point)
                .map_err(|e| GpuError::Backend(format!("entry point name: {e}")))?;

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
                    vk::PipelineCache::null(),
                    &[pipeline_info],
                    None,
                )
                .map_err(|(_, e)| GpuError::Backend(format!("compute pipeline: {e}")))?[0];

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
                .map_err(|e| GpuError::Backend(format!("desc pool: {e}")))?;

            let desc_set = device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(desc_pool)
                        .set_layouts(std::slice::from_ref(&desc_layout)),
                )
                .map_err(|e| GpuError::Backend(format!("alloc desc set: {e}")))?[0];

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
                .map_err(|e| GpuError::Backend(format!("shader module: {e}")))?;

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
                    GpuError::Backend(format!("desc set layout: {e}"))
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
                    GpuError::Backend(format!("pipeline layout: {e}"))
                })?;

            // Compute pipeline
            let entry_name = std::ffi::CString::new(entry_point).map_err(|e| {
                device.destroy_pipeline_layout(pipeline_layout, None);
                device.destroy_descriptor_set_layout(descriptor_set_layout, None);
                device.destroy_shader_module(shader_module, None);
                GpuError::Backend(format!("entry point name: {e}"))
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
                    vk::PipelineCache::null(),
                    &[pipeline_info],
                    None,
                )
                .map_err(|(_, e)| {
                    device.destroy_pipeline_layout(pipeline_layout, None);
                    device.destroy_descriptor_set_layout(descriptor_set_layout, None);
                    device.destroy_shader_module(shader_module, None);
                    GpuError::Backend(format!("compute pipeline: {e}"))
                })?[0];

            Ok(VulkanKernel {
                shader_module,
                descriptor_set_layout,
                pipeline_layout,
                pipeline,
                binding_count,
                state: Arc::clone(&self.state),
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

        unsafe {
            // Ephemeral descriptor pool + set
            let desc_pool = device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(&[vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: kernel.binding_count.max(1) as u32,
                        }]),
                    None,
                )
                .map_err(|e| GpuError::Backend(format!("desc pool: {e}")))?;

            let desc_set = device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(desc_pool)
                        .set_layouts(std::slice::from_ref(
                            &kernel.descriptor_set_layout,
                        )),
                )
                .map_err(|e| {
                    device.destroy_descriptor_pool(desc_pool, None);
                    GpuError::Backend(format!("alloc desc set: {e}"))
                })?[0];

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
            let pl = kernel.pipeline_layout;
            let pipe = kernel.pipeline;
            self.state.one_shot_submit(|cmd| {
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

            // Cleanup ephemeral descriptor pool (destroys its sets too)
            device.destroy_descriptor_pool(desc_pool, None);
        }

        Ok(())
    }

    fn device_name(&self) -> &str {
        &self.device_name
    }

    fn device_memory(&self) -> u64 {
        self.device_memory
    }
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
        .map_err(|e| GpuError::Backend(format!("allocator: {e}")))?;

        // Command pool
        let cmd_pool = device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(qf_index)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
            .map_err(|e| GpuError::Backend(format!("cmd pool: {e}")))?;

        let state = Arc::new(SharedState {
            device,
            queue,
            cmd_pool,
            allocator: std::mem::ManuallyDrop::new(Mutex::new(allocator)),
            instance,
            _entry: entry,
        });

        Ok(Self {
            state,
            device_name,
            device_memory,
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
        .map_err(|e| GpuError::Backend(format!("create buffer: {e}")))?;

        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let allocation = self
            .state
            .allocator
            .lock()
            .unwrap()
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
            .map_err(|e| GpuError::Backend(format!("bind memory: {e}")))?;

        Ok((buffer, allocation))
    }

    fn write_mapped(
        alloc: &gpu_allocator::vulkan::Allocation,
        data: &[u8],
    ) -> Result<()> {
        let ptr = alloc
            .mapped_ptr()
            .ok_or_else(|| GpuError::Backend("staging buffer not mappable".into()))?;

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
            .unwrap()
            .free(alloc)
            .map_err(|e| GpuError::Backend(format!("free: {e}")))?;
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
        .map_err(|e| GpuError::Backend(format!("readback buffer: {e}")))?;

        let requirements = unsafe { device.get_buffer_memory_requirements(staging_buf) };

        let staging_alloc = self
            .state
            .allocator
            .lock()
            .unwrap()
            .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                name: "staging_readback",
                requirements,
                location: MemoryLocation::GpuToCpu,
                linear: true,
                allocation_scheme:
                    gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(|e| GpuError::Backend(format!("readback alloc: {e}")))?;

        unsafe {
            device.bind_buffer_memory(
                staging_buf,
                staging_alloc.memory(),
                staging_alloc.offset(),
            )
        }
        .map_err(|e| GpuError::Backend(format!("bind readback: {e}")))?;

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
            .ok_or_else(|| GpuError::Backend("readback not mappable".into()))?;

        let mut data = vec![0u8; self.size as usize];
        unsafe {
            std::ptr::copy_nonoverlapping(
                ptr.as_ptr().cast::<u8>(),
                data.as_mut_ptr(),
                self.size as usize,
            );
        }

        // Cleanup staging
        let _ = self.state.allocator.lock().unwrap().free(staging_alloc);
        unsafe { device.destroy_buffer(staging_buf, None) };

        Ok(data)
    }

    fn byte_size(&self) -> u64 {
        self.size
    }
}
