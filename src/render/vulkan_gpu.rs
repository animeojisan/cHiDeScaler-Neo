//! Experimental Vulkan GPU-selection foundation for the future GLSL backend.
//!
//! v613 keeps the production OpenGL path authoritative while validating the
//! first real Neo user-shader pass on the explicitly selected Vulkan GPU. Auto
//! remains the untouched legacy OpenGL path and never initializes Vulkan. The
//! production display/GLSL chain is still unchanged; the new one-pass runner is
//! shadow-only until its compatibility and pixel behaviour are proven.

use super::mpv::{PassOffset, UserShader};
use anyhow::{Context, Result, anyhow};
use ash::{Entry, vk};
use std::ffi::{CStr, CString};
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

/// Stable routing contract for the future production GLSL backend. `Auto` is
/// represented by `None` and must never initialize Vulkan. An explicit GPU
/// selection carries the exact DXGI LUID into the Vulkan path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlslBackendRoute {
    LegacyOpenGl,
    VulkanSelected { luid: u64 },
}

pub fn glsl_backend_route(selected_luid: Option<u64>) -> GlslBackendRoute {
    match selected_luid {
        Some(luid) => GlslBackendRoute::VulkanSelected { luid },
        None => GlslBackendRoute::LegacyOpenGl,
    }
}

// v614 production one-pass route. Zero means Auto/no explicit GPU. This is
// only routing state; reading it never initializes Vulkan.
static PRODUCTION_SELECTED_LUID: AtomicU64 = AtomicU64::new(0);

pub fn set_production_selected_luid(selected_luid: Option<u64>) {
    let next = selected_luid.unwrap_or(0);
    let previous = PRODUCTION_SELECTED_LUID.swap(next, Ordering::AcqRel);
    // Only a real GPU-route change invalidates every Vulkan object. Session
    // restarts on the same adapter keep stateless multipass pipelines warm;
    // engine startup separately drops stateful STORAGE runtimes.
    if previous != next {
        super::vulkan_onepass::reset_runtime();
        super::vulkan_multipass::reset_runtime();
    }
}

pub fn production_selected_luid() -> Option<u64> {
    match PRODUCTION_SELECTED_LUID.load(Ordering::Acquire) {
        0 => None,
        luid => Some(luid),
    }
}

#[derive(Clone, Debug)]
pub struct VulkanGpuProbeResult {
    pub luid: u64,
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub api_version: u32,
    pub queue_family_index: u32,
    pub queue_flags: vk::QueueFlags,
}

impl VulkanGpuProbeResult {
    pub fn api_version_string(&self) -> String {
        format!(
            "{}.{}.{}",
            vk::api_version_major(self.api_version),
            vk::api_version_minor(self.api_version),
            vk::api_version_patch(self.api_version)
        )
    }
}

pub fn probe_requested() -> bool {
    std::env::var("NEO_VULKAN_GPU_PROBE")
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn luid_from_vk(bytes: &[u8; vk::LUID_SIZE]) -> u64 {
    u64::from_le_bytes(*bytes)
}

fn device_name(properties: &vk::PhysicalDeviceProperties) -> String {
    unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
        .to_string_lossy()
        .trim()
        .to_string()
}

/// Locate the Vulkan physical device whose Windows LUID exactly matches
/// `requested_luid`, create one logical device and a graphics/compute queue,
/// then destroy them immediately. This is a capability probe only.
///
/// `Entry::load()` is used instead of link-time Vulkan loading so machines
/// without a Vulkan loader continue to start normally and simply fall back to
/// the untouched OpenGL backend.
pub fn probe_selected_luid(requested_luid: u64) -> Result<VulkanGpuProbeResult> {
    let entry = unsafe { Entry::load() }.context("Vulkan loader could not be loaded")?;
    let app_name = CString::new("cHiDeScaler-Neo Vulkan GPU Probe")?;
    let engine_name = CString::new("cHiDeScaler-Neo")?;
    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(1)
        .engine_name(&engine_name)
        .engine_version(1)
        // VkPhysicalDeviceIDProperties/deviceLUID is core in Vulkan 1.1.
        .api_version(vk::API_VERSION_1_1);
    let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    let instance =
        unsafe { entry.create_instance(&create_info, None) }.context("vkCreateInstance failed")?;

    let result = (|| -> Result<VulkanGpuProbeResult> {
        let physical_devices = unsafe { instance.enumerate_physical_devices() }
            .context("vkEnumeratePhysicalDevices failed")?;
        if physical_devices.is_empty() {
            return Err(anyhow!("Vulkan reported no physical devices"));
        }

        let mut seen = Vec::new();
        for physical_device in physical_devices {
            let mut id = vk::PhysicalDeviceIDProperties::default();
            let mut properties2 = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
            unsafe {
                instance.get_physical_device_properties2(physical_device, &mut properties2);
            }
            let properties = properties2.properties;
            let name = device_name(&properties);
            let luid = if id.device_luid_valid == vk::TRUE {
                Some(luid_from_vk(&id.device_luid))
            } else {
                None
            };
            seen.push(format!(
                "{}:{}",
                name,
                luid.map(|value| format!("{value:016x}"))
                    .unwrap_or_else(|| "no-luid".to_string())
            ));
            if luid != Some(requested_luid) {
                continue;
            }

            let queue_families =
                unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
            let queue_family_index = queue_families
                .iter()
                .enumerate()
                .find(|(_, family)| {
                    family.queue_count > 0
                        && family
                            .queue_flags
                            .intersects(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
                })
                .map(|(index, _)| index as u32)
                .ok_or_else(|| anyhow!("matched GPU has no graphics/compute queue"))?;
            let queue_flags = queue_families[queue_family_index as usize].queue_flags;
            let priorities = [1.0f32];
            let queue_info = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index)
                .queue_priorities(&priorities)];
            let device_info = vk::DeviceCreateInfo::default().queue_create_infos(&queue_info);
            let device = unsafe { instance.create_device(physical_device, &device_info, None) }
                .context("vkCreateDevice failed for the LUID-matched GPU")?;

            // Fetching the queue verifies that the newly created logical device
            // is usable. No commands are submitted in v607.
            let _queue = unsafe { device.get_device_queue(queue_family_index, 0) };
            unsafe {
                device.destroy_device(None);
            }

            return Ok(VulkanGpuProbeResult {
                luid: requested_luid,
                name,
                vendor_id: properties.vendor_id,
                device_id: properties.device_id,
                api_version: properties.api_version,
                queue_family_index,
                queue_flags,
            });
        }

        Err(anyhow!(
            "no Vulkan physical device matched DXGI LUID {requested_luid:016x}; enumerated=[{}]",
            seen.join(", ")
        ))
    })();

    unsafe {
        instance.destroy_instance(None);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn luid_conversion_matches_dxgi_byte_order() {
        let bytes = [0x40, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00];
        assert_eq!(luid_from_vk(&bytes), 0x0000_0000_0001_0040);
    }

    #[test]
    fn probe_is_off_by_default_for_arbitrary_value() {
        // Do not mutate the process environment here; simply keep the actual
        // probe behind an explicit runtime opt-in in production code.
        let _ = probe_requested();
    }
}

// ---------------------------------------------------------------------------
// v609 diagnostic GLSL -> SPIR-V -> Vulkan execution probe.
// ---------------------------------------------------------------------------

/// Result of the v609 shader execution probe. This is intentionally a tiny
/// synthetic compute workload; no Neo frame or OpenGL texture is involved.
#[derive(Clone, Debug)]
pub struct VulkanGlslProbeResult {
    pub gpu: VulkanGpuProbeResult,
    pub spirv_words: usize,
    pub verified_values: usize,
    pub checksum: u64,
}

pub fn glsl_probe_requested() -> bool {
    std::env::var("NEO_VULKAN_GLSL_PROBE")
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

/// Append one diagnostic probe result to an opt-in result file selected by the
/// `NEO_VULKAN_PROBE_RESULT` environment variable. This is deliberately
/// independent of Neo's normal "Save log" setting. Failure to write the helper
/// file is non-fatal and never changes the production OpenGL path.
pub fn record_probe_result(line: &str) {
    let Ok(path) = std::env::var("NEO_VULKAN_PROBE_RESULT") else {
        return;
    };
    let path = path.trim();
    if path.is_empty() {
        return;
    }
    match OpenOptions::new().create(true).append(true).open(path) {
        Ok(mut file) => {
            if let Err(error) = writeln!(file, "{line}") {
                log::warn!(
                    "vulkan-probe-result-file: result=write-failed path={} error={error}",
                    path
                );
            }
        }
        Err(error) => log::warn!(
            "vulkan-probe-result-file: result=open-failed path={} error={error}",
            path
        ),
    }
}

const GLSL_PROBE_VALUE_COUNT: usize = 64;

const GLSL_PROBE_SOURCE: &str = r#"#version 450
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
layout(std430, set = 0, binding = 0) buffer ProbeData {
    uint values[64];
} data_buf;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < 64u) {
        data_buf.values[i] = data_buf.values[i] * 2u + 1u;
    }
}
"#;

fn compile_probe_glsl_to_spirv() -> Result<Vec<u32>> {
    let mut frontend = naga::front::glsl::Frontend::default();
    let options = naga::front::glsl::Options::from(naga::ShaderStage::Compute);
    let module = frontend
        .parse(&options, GLSL_PROBE_SOURCE)
        .map_err(|error| anyhow!("Naga GLSL parse failed: {error:?}"))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|error| anyhow!("Naga GLSL validation failed: {error:?}"))?;
    let spv_options = naga::back::spv::Options {
        lang_version: (1, 3),
        ..Default::default()
    };
    naga::back::spv::write_vec(&module, &info, &spv_options, None)
        .map_err(|error| anyhow!("Naga SPIR-V generation failed: {error:?}"))
}

fn find_host_visible_coherent_memory_type(
    properties: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
) -> Option<u32> {
    let required = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    (0..properties.memory_type_count).find(|index| {
        let bit = 1u32 << *index;
        type_bits & bit != 0
            && properties.memory_types[*index as usize]
                .property_flags
                .contains(required)
    })
}

struct VulkanGlslResources {
    device: ash::Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: bool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    pipeline_layout: vk::PipelineLayout,
    shader_module: vk::ShaderModule,
    pipeline: vk::Pipeline,
    command_pool: vk::CommandPool,
    fence: vk::Fence,
}

impl VulkanGlslResources {
    fn new(device: ash::Device) -> Self {
        Self {
            device,
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            mapped: false,
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            shader_module: vk::ShaderModule::null(),
            pipeline: vk::Pipeline::null(),
            command_pool: vk::CommandPool::null(),
            fence: vk::Fence::null(),
        }
    }
}

impl Drop for VulkanGlslResources {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            if self.fence != vk::Fence::null() {
                self.device.destroy_fence(self.fence, None);
            }
            if self.command_pool != vk::CommandPool::null() {
                self.device.destroy_command_pool(self.command_pool, None);
            }
            if self.pipeline != vk::Pipeline::null() {
                self.device.destroy_pipeline(self.pipeline, None);
            }
            if self.shader_module != vk::ShaderModule::null() {
                self.device.destroy_shader_module(self.shader_module, None);
            }
            if self.pipeline_layout != vk::PipelineLayout::null() {
                self.device
                    .destroy_pipeline_layout(self.pipeline_layout, None);
            }
            if self.descriptor_pool != vk::DescriptorPool::null() {
                self.device
                    .destroy_descriptor_pool(self.descriptor_pool, None);
            }
            if self.descriptor_set_layout != vk::DescriptorSetLayout::null() {
                self.device
                    .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            }
            if self.mapped && self.memory != vk::DeviceMemory::null() {
                self.device.unmap_memory(self.memory);
            }
            if self.buffer != vk::Buffer::null() {
                self.device.destroy_buffer(self.buffer, None);
            }
            if self.memory != vk::DeviceMemory::null() {
                self.device.free_memory(self.memory, None);
            }
            self.device.destroy_device(None);
        }
    }
}

/// Compile a small GLSL 450 compute shader with Naga, execute it on the
/// Vulkan physical device whose Windows LUID exactly matches `requested_luid`,
/// and verify the GPU-written buffer. This proves the complete
/// GLSL -> SPIR-V -> Vulkan execution path without touching Neo's stable
/// OpenGL/GLSL frame pipeline.
pub fn probe_glsl_compute_selected_luid(requested_luid: u64) -> Result<VulkanGlslProbeResult> {
    let spirv = compile_probe_glsl_to_spirv()?;
    if spirv.is_empty() {
        return Err(anyhow!("Naga returned an empty SPIR-V module"));
    }

    let entry = unsafe { Entry::load() }.context("Vulkan loader could not be loaded")?;
    let app_name = CString::new("cHiDeScaler-Neo Vulkan GLSL Probe")?;
    let engine_name = CString::new("cHiDeScaler-Neo")?;
    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(1)
        .engine_name(&engine_name)
        .engine_version(1)
        .api_version(vk::API_VERSION_1_1);
    let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    let instance = unsafe { entry.create_instance(&create_info, None) }
        .context("vkCreateInstance failed for GLSL probe")?;

    let result = (|| -> Result<VulkanGlslProbeResult> {
        let physical_devices = unsafe { instance.enumerate_physical_devices() }
            .context("vkEnumeratePhysicalDevices failed for GLSL probe")?;
        if physical_devices.is_empty() {
            return Err(anyhow!("Vulkan reported no physical devices"));
        }

        let mut seen = Vec::new();
        let mut selected = None;
        for physical_device in physical_devices {
            let mut id = vk::PhysicalDeviceIDProperties::default();
            let mut properties2 = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
            unsafe {
                instance.get_physical_device_properties2(physical_device, &mut properties2);
            }
            let properties = properties2.properties;
            let name = device_name(&properties);
            let luid = if id.device_luid_valid == vk::TRUE {
                Some(luid_from_vk(&id.device_luid))
            } else {
                None
            };
            seen.push(format!(
                "{}:{}",
                name,
                luid.map(|value| format!("{value:016x}"))
                    .unwrap_or_else(|| "no-luid".to_string())
            ));
            if luid != Some(requested_luid) {
                continue;
            }

            let queue_families =
                unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
            let queue_family_index = queue_families
                .iter()
                .enumerate()
                .find(|(_, family)| {
                    family.queue_count > 0 && family.queue_flags.contains(vk::QueueFlags::COMPUTE)
                })
                .map(|(index, _)| index as u32)
                .ok_or_else(|| anyhow!("matched GPU has no compute queue"))?;
            let queue_flags = queue_families[queue_family_index as usize].queue_flags;
            selected = Some((
                physical_device,
                properties,
                name,
                queue_family_index,
                queue_flags,
            ));
            break;
        }

        let (physical_device, properties, name, queue_family_index, queue_flags) = selected
            .ok_or_else(|| anyhow!(
                "no Vulkan physical device matched DXGI LUID {requested_luid:016x}; enumerated=[{}]",
                seen.join(", ")
            ))?;

        let priorities = [1.0f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities)];
        let device_info = vk::DeviceCreateInfo::default().queue_create_infos(&queue_info);
        let device = unsafe { instance.create_device(physical_device, &device_info, None) }
            .context("vkCreateDevice failed for GLSL probe")?;
        let mut resources = VulkanGlslResources::new(device);
        let queue = unsafe { resources.device.get_device_queue(queue_family_index, 0) };

        let buffer_size = (GLSL_PROBE_VALUE_COUNT * std::mem::size_of::<u32>()) as vk::DeviceSize;
        let buffer_info = vk::BufferCreateInfo::default()
            .size(buffer_size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        resources.buffer = unsafe { resources.device.create_buffer(&buffer_info, None) }
            .context("vkCreateBuffer failed for GLSL probe")?;
        let requirements = unsafe {
            resources
                .device
                .get_buffer_memory_requirements(resources.buffer)
        };
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let memory_type_index = find_host_visible_coherent_memory_type(
            &memory_properties,
            requirements.memory_type_bits,
        )
        .ok_or_else(|| {
            anyhow!("matched GPU has no HOST_VISIBLE|HOST_COHERENT memory type for probe buffer")
        })?;
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index);
        resources.memory = unsafe { resources.device.allocate_memory(&alloc_info, None) }
            .context("vkAllocateMemory failed for GLSL probe")?;
        unsafe {
            resources
                .device
                .bind_buffer_memory(resources.buffer, resources.memory, 0)
                .context("vkBindBufferMemory failed for GLSL probe")?;
        }

        let mapped = unsafe {
            resources.device.map_memory(
                resources.memory,
                0,
                buffer_size,
                vk::MemoryMapFlags::empty(),
            )
        }
        .context("vkMapMemory failed for GLSL probe")?;
        resources.mapped = true;
        let input: Vec<u32> = (0..GLSL_PROBE_VALUE_COUNT as u32).collect();
        unsafe {
            std::ptr::copy_nonoverlapping(
                input.as_ptr(),
                mapped.cast::<u32>(),
                GLSL_PROBE_VALUE_COUNT,
            );
        }

        let layout_bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];
        let descriptor_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&layout_bindings);
        resources.descriptor_set_layout = unsafe {
            resources
                .device
                .create_descriptor_set_layout(&descriptor_layout_info, None)
        }
        .context("vkCreateDescriptorSetLayout failed for GLSL probe")?;

        let set_layouts = [resources.descriptor_set_layout];
        let pipeline_layout_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        resources.pipeline_layout = unsafe {
            resources
                .device
                .create_pipeline_layout(&pipeline_layout_info, None)
        }
        .context("vkCreatePipelineLayout failed for GLSL probe")?;

        let shader_info = vk::ShaderModuleCreateInfo::default().code(&spirv);
        resources.shader_module =
            unsafe { resources.device.create_shader_module(&shader_info, None) }
                .context("vkCreateShaderModule failed for Naga SPIR-V")?;
        let entry_name = CString::new("main")?;
        let shader_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(resources.shader_module)
            .name(&entry_name);
        let compute_info = [vk::ComputePipelineCreateInfo::default()
            .stage(shader_stage)
            .layout(resources.pipeline_layout)];
        let pipelines = unsafe {
            resources.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &compute_info,
                None,
            )
        }
        .map_err(|(partial, error)| {
            unsafe {
                for pipeline in partial {
                    resources.device.destroy_pipeline(pipeline, None);
                }
            }
            anyhow!("vkCreateComputePipelines failed for GLSL probe: {error:?}")
        })?;
        resources.pipeline = *pipelines
            .first()
            .ok_or_else(|| anyhow!("Vulkan returned no compute pipeline"))?;

        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
        }];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        resources.descriptor_pool =
            unsafe { resources.device.create_descriptor_pool(&pool_info, None) }
                .context("vkCreateDescriptorPool failed for GLSL probe")?;
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(resources.descriptor_pool)
            .set_layouts(&set_layouts);
        let descriptor_sets = unsafe { resources.device.allocate_descriptor_sets(&allocate_info) }
            .context("vkAllocateDescriptorSets failed for GLSL probe")?;
        let descriptor_set = descriptor_sets[0];
        let buffer_infos = [vk::DescriptorBufferInfo::default()
            .buffer(resources.buffer)
            .offset(0)
            .range(buffer_size)];
        let writes = [vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buffer_infos)];
        unsafe {
            resources.device.update_descriptor_sets(&writes, &[]);
        }

        let command_pool_info =
            vk::CommandPoolCreateInfo::default().queue_family_index(queue_family_index);
        resources.command_pool = unsafe {
            resources
                .device
                .create_command_pool(&command_pool_info, None)
        }
        .context("vkCreateCommandPool failed for GLSL probe")?;
        let command_allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(resources.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffers = unsafe {
            resources
                .device
                .allocate_command_buffers(&command_allocate_info)
        }
        .context("vkAllocateCommandBuffers failed for GLSL probe")?;
        let command_buffer = command_buffers[0];
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            resources
                .device
                .begin_command_buffer(command_buffer, &begin_info)
                .context("vkBeginCommandBuffer failed for GLSL probe")?;
            resources.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                resources.pipeline,
            );
            resources.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                resources.pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
            resources.device.cmd_dispatch(command_buffer, 1, 1, 1);
            let barriers = [vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(resources.buffer)
                .offset(0)
                .size(buffer_size)];
            resources.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[],
                &barriers,
                &[],
            );
            resources
                .device
                .end_command_buffer(command_buffer)
                .context("vkEndCommandBuffer failed for GLSL probe")?;
        }

        resources.fence = unsafe {
            resources
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        }
        .context("vkCreateFence failed for GLSL probe")?;
        let submitted_buffers = [command_buffer];
        let submits = [vk::SubmitInfo::default().command_buffers(&submitted_buffers)];
        unsafe {
            resources
                .device
                .queue_submit(queue, &submits, resources.fence)
                .context("vkQueueSubmit failed for GLSL probe")?;
            resources
                .device
                .wait_for_fences(&[resources.fence], true, 5_000_000_000)
                .context("Vulkan GLSL probe timed out or fence wait failed")?;
        }

        let output =
            unsafe { std::slice::from_raw_parts(mapped.cast::<u32>(), GLSL_PROBE_VALUE_COUNT) };
        let mut verified = 0usize;
        let mut checksum = 0u64;
        for (index, &value) in output.iter().enumerate() {
            let expected = index as u32 * 2 + 1;
            if value == expected {
                verified += 1;
            }
            checksum += value as u64;
        }
        if verified != GLSL_PROBE_VALUE_COUNT {
            let mismatches = output
                .iter()
                .enumerate()
                .filter_map(|(index, &value)| {
                    let expected = index as u32 * 2 + 1;
                    (value != expected).then_some(format!("{index}:{value}!={expected}"))
                })
                .take(8)
                .collect::<Vec<_>>()
                .join(",");
            return Err(anyhow!(
                "GLSL compute verification failed: verified={verified}/{GLSL_PROBE_VALUE_COUNT} mismatches=[{mismatches}]"
            ));
        }

        Ok(VulkanGlslProbeResult {
            gpu: VulkanGpuProbeResult {
                luid: requested_luid,
                name,
                vendor_id: properties.vendor_id,
                device_id: properties.device_id,
                api_version: properties.api_version,
                queue_family_index,
                queue_flags,
            },
            spirv_words: spirv.len(),
            verified_values: verified,
            checksum,
        })
    })();

    unsafe {
        instance.destroy_instance(None);
    }
    result
}

// ---------------------------------------------------------------------------
// v610 diagnostic synthetic RGBA image -> Vulkan GLSL -> readback probe.
// ---------------------------------------------------------------------------

const IMAGE_PROBE_WIDTH: u32 = 8;
const IMAGE_PROBE_HEIGHT: u32 = 8;
const IMAGE_PROBE_PIXEL_COUNT: usize = (IMAGE_PROBE_WIDTH as usize) * (IMAGE_PROBE_HEIGHT as usize);
const IMAGE_PROBE_BYTE_COUNT: usize = IMAGE_PROBE_PIXEL_COUNT * 4;

fn image_probe_source(width: u32, height: u32) -> String {
    format!(
        r#"#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(rgba8, set = 0, binding = 0) readonly uniform image2D src_img;
layout(rgba8, set = 0, binding = 1) writeonly uniform image2D dst_img;

void main() {{
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x < {width} && p.y < {height}) {{
        vec4 c = imageLoad(src_img, p);
        imageStore(dst_img, p, c.bgra);
    }}
}}
"#
    )
}

#[derive(Clone, Debug)]
pub struct VulkanImageProbeResult {
    pub gpu: VulkanGpuProbeResult,
    pub spirv_words: usize,
    pub verified_pixels: usize,
    pub checksum: u64,
    pub expected_checksum: u64,
}

pub fn image_probe_requested() -> bool {
    std::env::var("NEO_VULKAN_IMAGE_PROBE")
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn compile_image_probe_glsl_to_spirv(width: u32, height: u32) -> Result<Vec<u32>> {
    let source = image_probe_source(width, height);
    let mut frontend = naga::front::glsl::Frontend::default();
    let options = naga::front::glsl::Options::from(naga::ShaderStage::Compute);
    let module = frontend
        .parse(&options, &source)
        .map_err(|error| anyhow!("Naga image-probe GLSL parse failed: {error:?}"))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|error| anyhow!("Naga image-probe GLSL validation failed: {error:?}"))?;
    let spv_options = naga::back::spv::Options {
        lang_version: (1, 3),
        ..Default::default()
    };
    naga::back::spv::write_vec(&module, &info, &spv_options, None)
        .map_err(|error| anyhow!("Naga image-probe SPIR-V generation failed: {error:?}"))
}

fn find_memory_type_with_flags(
    properties: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    required: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..properties.memory_type_count).find(|index| {
        let bit = 1u32 << *index;
        type_bits & bit != 0
            && properties.memory_types[*index as usize]
                .property_flags
                .contains(required)
    })
}

fn image_probe_input() -> Vec<u8> {
    let mut bytes = vec![0u8; IMAGE_PROBE_BYTE_COUNT];
    for index in 0..IMAGE_PROBE_PIXEL_COUNT {
        let base = index * 4;
        bytes[base] = ((index * 3 + 1) & 0xff) as u8;
        bytes[base + 1] = ((index * 5 + 7) & 0xff) as u8;
        bytes[base + 2] = ((index * 11 + 13) & 0xff) as u8;
        bytes[base + 3] = 255;
    }
    bytes
}

fn image_probe_expected(input: &[u8]) -> Vec<u8> {
    let mut output = vec![0u8; input.len()];
    for (src, dst) in input.chunks_exact(4).zip(output.chunks_exact_mut(4)) {
        dst[0] = src[2];
        dst[1] = src[1];
        dst[2] = src[0];
        dst[3] = src[3];
    }
    output
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3u64);
    }
    hash
}

struct VulkanImageProbeResources {
    device: ash::Device,
    upload_buffer: vk::Buffer,
    upload_memory: vk::DeviceMemory,
    upload_mapped: bool,
    readback_buffer: vk::Buffer,
    readback_memory: vk::DeviceMemory,
    readback_mapped: bool,
    src_image: vk::Image,
    src_memory: vk::DeviceMemory,
    src_view: vk::ImageView,
    dst_image: vk::Image,
    dst_memory: vk::DeviceMemory,
    dst_view: vk::ImageView,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    pipeline_layout: vk::PipelineLayout,
    shader_module: vk::ShaderModule,
    pipeline: vk::Pipeline,
    command_pool: vk::CommandPool,
    fence: vk::Fence,
}

impl VulkanImageProbeResources {
    fn new(device: ash::Device) -> Self {
        Self {
            device,
            upload_buffer: vk::Buffer::null(),
            upload_memory: vk::DeviceMemory::null(),
            upload_mapped: false,
            readback_buffer: vk::Buffer::null(),
            readback_memory: vk::DeviceMemory::null(),
            readback_mapped: false,
            src_image: vk::Image::null(),
            src_memory: vk::DeviceMemory::null(),
            src_view: vk::ImageView::null(),
            dst_image: vk::Image::null(),
            dst_memory: vk::DeviceMemory::null(),
            dst_view: vk::ImageView::null(),
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            shader_module: vk::ShaderModule::null(),
            pipeline: vk::Pipeline::null(),
            command_pool: vk::CommandPool::null(),
            fence: vk::Fence::null(),
        }
    }
}

impl Drop for VulkanImageProbeResources {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            if self.fence != vk::Fence::null() {
                self.device.destroy_fence(self.fence, None);
            }
            if self.command_pool != vk::CommandPool::null() {
                self.device.destroy_command_pool(self.command_pool, None);
            }
            if self.pipeline != vk::Pipeline::null() {
                self.device.destroy_pipeline(self.pipeline, None);
            }
            if self.shader_module != vk::ShaderModule::null() {
                self.device.destroy_shader_module(self.shader_module, None);
            }
            if self.pipeline_layout != vk::PipelineLayout::null() {
                self.device
                    .destroy_pipeline_layout(self.pipeline_layout, None);
            }
            if self.descriptor_pool != vk::DescriptorPool::null() {
                self.device
                    .destroy_descriptor_pool(self.descriptor_pool, None);
            }
            if self.descriptor_set_layout != vk::DescriptorSetLayout::null() {
                self.device
                    .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            }
            if self.src_view != vk::ImageView::null() {
                self.device.destroy_image_view(self.src_view, None);
            }
            if self.dst_view != vk::ImageView::null() {
                self.device.destroy_image_view(self.dst_view, None);
            }
            if self.src_image != vk::Image::null() {
                self.device.destroy_image(self.src_image, None);
            }
            if self.dst_image != vk::Image::null() {
                self.device.destroy_image(self.dst_image, None);
            }
            if self.src_memory != vk::DeviceMemory::null() {
                self.device.free_memory(self.src_memory, None);
            }
            if self.dst_memory != vk::DeviceMemory::null() {
                self.device.free_memory(self.dst_memory, None);
            }
            if self.upload_mapped && self.upload_memory != vk::DeviceMemory::null() {
                self.device.unmap_memory(self.upload_memory);
            }
            if self.readback_mapped && self.readback_memory != vk::DeviceMemory::null() {
                self.device.unmap_memory(self.readback_memory);
            }
            if self.upload_buffer != vk::Buffer::null() {
                self.device.destroy_buffer(self.upload_buffer, None);
            }
            if self.readback_buffer != vk::Buffer::null() {
                self.device.destroy_buffer(self.readback_buffer, None);
            }
            if self.upload_memory != vk::DeviceMemory::null() {
                self.device.free_memory(self.upload_memory, None);
            }
            if self.readback_memory != vk::DeviceMemory::null() {
                self.device.free_memory(self.readback_memory, None);
            }
            self.device.destroy_device(None);
        }
    }
}

/// Process a tiny synthetic 8x8 RGBA8 image through a real Vulkan storage-image
/// compute pass on the exact DXGI-LUID-selected GPU, copy the output back to a
/// host-visible buffer, and verify every pixel. This remains diagnostic-only:
/// no capture frame, OpenGL texture, user shader, compositor, or presentation
/// resource is touched.
fn probe_glsl_image_input_selected_luid(
    requested_luid: u64,
    input: &[u8],
    width: u32,
    height: u32,
) -> Result<VulkanImageProbeResult> {
    let spirv = compile_image_probe_glsl_to_spirv(width, height)?;
    let expected = image_probe_expected(input);
    run_glsl_image_input_selected_luid(
        requested_luid,
        input,
        width,
        height,
        spirv,
        expected,
        0,
        "cHiDeScaler-Neo Vulkan Image Probe",
    )
}

fn run_glsl_image_input_selected_luid(
    requested_luid: u64,
    input: &[u8],
    width: u32,
    height: u32,
    spirv: Vec<u32>,
    expected: Vec<u8>,
    verification_tolerance: u8,
    app_label: &str,
) -> Result<VulkanImageProbeResult> {
    if width == 0 || height == 0 {
        return Err(anyhow!("image probe dimensions must be non-zero"));
    }
    let pixel_count = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| anyhow!("image probe dimensions overflow: {width}x{height}"))?;
    let byte_count = pixel_count
        .checked_mul(4)
        .ok_or_else(|| anyhow!("image probe byte count overflow: {width}x{height}"))?;
    if input.len() != byte_count {
        return Err(anyhow!(
            "image probe requires exactly {byte_count} RGBA8 bytes for {width}x{height}, got {}",
            input.len()
        ));
    }
    if spirv.is_empty() {
        return Err(anyhow!(
            "Naga returned an empty SPIR-V module for image runner"
        ));
    }
    if expected.len() != byte_count {
        return Err(anyhow!(
            "image runner expected buffer requires exactly {byte_count} bytes, got {}",
            expected.len()
        ));
    }
    let expected_checksum = fnv1a64(&expected);

    let entry = unsafe { Entry::load() }.context("Vulkan loader could not be loaded")?;
    let app_name = CString::new(app_label)?;
    let engine_name = CString::new("cHiDeScaler-Neo")?;
    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(1)
        .engine_name(&engine_name)
        .engine_version(1)
        .api_version(vk::API_VERSION_1_1);
    let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    let instance = unsafe { entry.create_instance(&create_info, None) }
        .context("vkCreateInstance failed for image probe")?;

    let result = (|| -> Result<VulkanImageProbeResult> {
        let physical_devices = unsafe { instance.enumerate_physical_devices() }
            .context("vkEnumeratePhysicalDevices failed for image probe")?;
        if physical_devices.is_empty() {
            return Err(anyhow!("Vulkan reported no physical devices"));
        }

        let mut seen = Vec::new();
        let mut selected = None;
        for physical_device in physical_devices {
            let mut id = vk::PhysicalDeviceIDProperties::default();
            let mut properties2 = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
            unsafe {
                instance.get_physical_device_properties2(physical_device, &mut properties2);
            }
            let properties = properties2.properties;
            let name = device_name(&properties);
            let luid = if id.device_luid_valid == vk::TRUE {
                Some(luid_from_vk(&id.device_luid))
            } else {
                None
            };
            seen.push(format!(
                "{}:{}",
                name,
                luid.map(|value| format!("{value:016x}"))
                    .unwrap_or_else(|| "no-luid".to_string())
            ));
            if luid != Some(requested_luid) {
                continue;
            }

            let queue_families =
                unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
            let queue_family_index = queue_families
                .iter()
                .enumerate()
                .find(|(_, family)| {
                    family.queue_count > 0 && family.queue_flags.contains(vk::QueueFlags::COMPUTE)
                })
                .map(|(index, _)| index as u32)
                .ok_or_else(|| anyhow!("matched GPU has no compute queue for image probe"))?;
            let queue_flags = queue_families[queue_family_index as usize].queue_flags;
            selected = Some((
                physical_device,
                properties,
                name,
                queue_family_index,
                queue_flags,
            ));
            break;
        }

        let (physical_device, properties, name, queue_family_index, queue_flags) = selected
            .ok_or_else(|| anyhow!(
                "no Vulkan physical device matched DXGI LUID {requested_luid:016x}; enumerated=[{}]",
                seen.join(", ")
            ))?;

        let priorities = [1.0f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities)];
        let device_info = vk::DeviceCreateInfo::default().queue_create_infos(&queue_info);
        let device = unsafe { instance.create_device(physical_device, &device_info, None) }
            .context("vkCreateDevice failed for image probe")?;
        let mut resources = VulkanImageProbeResources::new(device);
        let queue = unsafe { resources.device.get_device_queue(queue_family_index, 0) };
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let byte_size = byte_count as vk::DeviceSize;

        // Host-visible upload buffer.
        let upload_info = vk::BufferCreateInfo::default()
            .size(byte_size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        resources.upload_buffer = unsafe { resources.device.create_buffer(&upload_info, None) }
            .context("vkCreateBuffer failed for image-probe upload")?;
        let upload_req = unsafe {
            resources
                .device
                .get_buffer_memory_requirements(resources.upload_buffer)
        };
        let host_required =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let upload_type = find_memory_type_with_flags(
            &memory_properties,
            upload_req.memory_type_bits,
            host_required,
        )
        .ok_or_else(|| anyhow!("no HOST_VISIBLE|HOST_COHERENT memory for image-probe upload"))?;
        let upload_alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(upload_req.size)
            .memory_type_index(upload_type);
        resources.upload_memory = unsafe { resources.device.allocate_memory(&upload_alloc, None) }
            .context("vkAllocateMemory failed for image-probe upload")?;
        unsafe {
            resources
                .device
                .bind_buffer_memory(resources.upload_buffer, resources.upload_memory, 0)
                .context("vkBindBufferMemory failed for image-probe upload")?;
        }
        let upload_mapped = unsafe {
            resources.device.map_memory(
                resources.upload_memory,
                0,
                byte_size,
                vk::MemoryMapFlags::empty(),
            )
        }
        .context("vkMapMemory failed for image-probe upload")?;
        resources.upload_mapped = true;
        unsafe {
            std::ptr::copy_nonoverlapping(input.as_ptr(), upload_mapped.cast::<u8>(), input.len());
        }

        // Host-visible readback buffer.
        let readback_info = vk::BufferCreateInfo::default()
            .size(byte_size)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        resources.readback_buffer = unsafe { resources.device.create_buffer(&readback_info, None) }
            .context("vkCreateBuffer failed for image-probe readback")?;
        let readback_req = unsafe {
            resources
                .device
                .get_buffer_memory_requirements(resources.readback_buffer)
        };
        let readback_type = find_memory_type_with_flags(
            &memory_properties,
            readback_req.memory_type_bits,
            host_required,
        )
        .ok_or_else(|| anyhow!("no HOST_VISIBLE|HOST_COHERENT memory for image-probe readback"))?;
        let readback_alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(readback_req.size)
            .memory_type_index(readback_type);
        resources.readback_memory =
            unsafe { resources.device.allocate_memory(&readback_alloc, None) }
                .context("vkAllocateMemory failed for image-probe readback")?;
        unsafe {
            resources
                .device
                .bind_buffer_memory(resources.readback_buffer, resources.readback_memory, 0)
                .context("vkBindBufferMemory failed for image-probe readback")?;
        }
        let readback_mapped = unsafe {
            resources.device.map_memory(
                resources.readback_memory,
                0,
                byte_size,
                vk::MemoryMapFlags::empty(),
            )
        }
        .context("vkMapMemory failed for image-probe readback")?;
        resources.readback_mapped = true;
        unsafe {
            std::ptr::write_bytes(readback_mapped.cast::<u8>(), 0, byte_count);
        }

        let extent = vk::Extent3D {
            width,
            height,
            depth: 1,
        };
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        resources.src_image = unsafe { resources.device.create_image(&image_info, None) }
            .context("vkCreateImage failed for image-probe source")?;
        let src_req = unsafe {
            resources
                .device
                .get_image_memory_requirements(resources.src_image)
        };
        let src_type = find_memory_type_with_flags(
            &memory_properties,
            src_req.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .or_else(|| {
            find_memory_type_with_flags(
                &memory_properties,
                src_req.memory_type_bits,
                vk::MemoryPropertyFlags::empty(),
            )
        })
        .ok_or_else(|| anyhow!("no compatible memory type for image-probe source image"))?;
        let src_alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(src_req.size)
            .memory_type_index(src_type);
        resources.src_memory = unsafe { resources.device.allocate_memory(&src_alloc, None) }
            .context("vkAllocateMemory failed for image-probe source image")?;
        unsafe {
            resources
                .device
                .bind_image_memory(resources.src_image, resources.src_memory, 0)
                .context("vkBindImageMemory failed for image-probe source")?;
        }

        resources.dst_image = unsafe { resources.device.create_image(&image_info, None) }
            .context("vkCreateImage failed for image-probe destination")?;
        let dst_req = unsafe {
            resources
                .device
                .get_image_memory_requirements(resources.dst_image)
        };
        let dst_type = find_memory_type_with_flags(
            &memory_properties,
            dst_req.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .or_else(|| {
            find_memory_type_with_flags(
                &memory_properties,
                dst_req.memory_type_bits,
                vk::MemoryPropertyFlags::empty(),
            )
        })
        .ok_or_else(|| anyhow!("no compatible memory type for image-probe destination image"))?;
        let dst_alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(dst_req.size)
            .memory_type_index(dst_type);
        resources.dst_memory = unsafe { resources.device.allocate_memory(&dst_alloc, None) }
            .context("vkAllocateMemory failed for image-probe destination image")?;
        unsafe {
            resources
                .device
                .bind_image_memory(resources.dst_image, resources.dst_memory, 0)
                .context("vkBindImageMemory failed for image-probe destination")?;
        }

        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);
        let src_view_info = vk::ImageViewCreateInfo::default()
            .image(resources.src_image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .subresource_range(subresource_range);
        resources.src_view = unsafe { resources.device.create_image_view(&src_view_info, None) }
            .context("vkCreateImageView failed for image-probe source")?;
        let dst_view_info = vk::ImageViewCreateInfo::default()
            .image(resources.dst_image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .subresource_range(subresource_range);
        resources.dst_view = unsafe { resources.device.create_image_view(&dst_view_info, None) }
            .context("vkCreateImageView failed for image-probe destination")?;

        let layout_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let descriptor_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&layout_bindings);
        resources.descriptor_set_layout = unsafe {
            resources
                .device
                .create_descriptor_set_layout(&descriptor_layout_info, None)
        }
        .context("vkCreateDescriptorSetLayout failed for image probe")?;

        let set_layouts = [resources.descriptor_set_layout];
        let pipeline_layout_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        resources.pipeline_layout = unsafe {
            resources
                .device
                .create_pipeline_layout(&pipeline_layout_info, None)
        }
        .context("vkCreatePipelineLayout failed for image probe")?;

        let shader_info = vk::ShaderModuleCreateInfo::default().code(&spirv);
        resources.shader_module =
            unsafe { resources.device.create_shader_module(&shader_info, None) }
                .context("vkCreateShaderModule failed for image-probe SPIR-V")?;
        let entry_name = CString::new("main")?;
        let shader_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(resources.shader_module)
            .name(&entry_name);
        let compute_info = [vk::ComputePipelineCreateInfo::default()
            .stage(shader_stage)
            .layout(resources.pipeline_layout)];
        let pipelines = unsafe {
            resources.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &compute_info,
                None,
            )
        }
        .map_err(|(partial, error)| {
            unsafe {
                for pipeline in partial {
                    resources.device.destroy_pipeline(pipeline, None);
                }
            }
            anyhow!("vkCreateComputePipelines failed for image probe: {error:?}")
        })?;
        resources.pipeline = *pipelines
            .first()
            .ok_or_else(|| anyhow!("Vulkan returned no image-probe pipeline"))?;

        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_IMAGE,
            descriptor_count: 2,
        }];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        resources.descriptor_pool =
            unsafe { resources.device.create_descriptor_pool(&pool_info, None) }
                .context("vkCreateDescriptorPool failed for image probe")?;
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(resources.descriptor_pool)
            .set_layouts(&set_layouts);
        let descriptor_sets = unsafe { resources.device.allocate_descriptor_sets(&allocate_info) }
            .context("vkAllocateDescriptorSets failed for image probe")?;
        let descriptor_set = descriptor_sets[0];
        let src_image_info = [vk::DescriptorImageInfo::default()
            .image_view(resources.src_view)
            .image_layout(vk::ImageLayout::GENERAL)];
        let dst_image_info = [vk::DescriptorImageInfo::default()
            .image_view(resources.dst_view)
            .image_layout(vk::ImageLayout::GENERAL)];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&src_image_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&dst_image_info),
        ];
        unsafe {
            resources.device.update_descriptor_sets(&writes, &[]);
        }

        let command_pool_info =
            vk::CommandPoolCreateInfo::default().queue_family_index(queue_family_index);
        resources.command_pool = unsafe {
            resources
                .device
                .create_command_pool(&command_pool_info, None)
        }
        .context("vkCreateCommandPool failed for image probe")?;
        let command_allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(resources.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffers = unsafe {
            resources
                .device
                .allocate_command_buffers(&command_allocate_info)
        }
        .context("vkAllocateCommandBuffers failed for image probe")?;
        let command_buffer = command_buffers[0];
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        let layers = vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(0)
            .base_array_layer(0)
            .layer_count(1);
        let copy_region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(layers)
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(extent);

        unsafe {
            resources
                .device
                .begin_command_buffer(command_buffer, &begin_info)
                .context("vkBeginCommandBuffer failed for image probe")?;

            let upload_barrier = [vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::HOST_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(resources.upload_buffer)
                .offset(0)
                .size(byte_size)];
            resources.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &upload_barrier,
                &[],
            );

            let src_to_transfer = [vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(resources.src_image)
                .subresource_range(subresource_range)];
            resources.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &src_to_transfer,
            );
            resources.device.cmd_copy_buffer_to_image(
                command_buffer,
                resources.upload_buffer,
                resources.src_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy_region],
            );

            let src_to_general = [vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(resources.src_image)
                .subresource_range(subresource_range)];
            let dst_to_general = [vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(resources.dst_image)
                .subresource_range(subresource_range)];
            resources.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &src_to_general,
            );
            resources.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &dst_to_general,
            );

            resources.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                resources.pipeline,
            );
            resources.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                resources.pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
            resources
                .device
                .cmd_dispatch(command_buffer, width.div_ceil(8), height.div_ceil(8), 1);

            let dst_to_transfer = [vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(resources.dst_image)
                .subresource_range(subresource_range)];
            resources.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &dst_to_transfer,
            );
            resources.device.cmd_copy_image_to_buffer(
                command_buffer,
                resources.dst_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                resources.readback_buffer,
                &[copy_region],
            );
            let readback_barrier = [vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(resources.readback_buffer)
                .offset(0)
                .size(byte_size)];
            resources.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[],
                &readback_barrier,
                &[],
            );
            resources
                .device
                .end_command_buffer(command_buffer)
                .context("vkEndCommandBuffer failed for image probe")?;
        }

        resources.fence = unsafe {
            resources
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        }
        .context("vkCreateFence failed for image probe")?;
        let submitted_buffers = [command_buffer];
        let submits = [vk::SubmitInfo::default().command_buffers(&submitted_buffers)];
        unsafe {
            resources
                .device
                .queue_submit(queue, &submits, resources.fence)
                .context("vkQueueSubmit failed for image probe")?;
            resources
                .device
                .wait_for_fences(&[resources.fence], true, 5_000_000_000)
                .context("Vulkan image probe timed out or fence wait failed")?;
        }

        let output =
            unsafe { std::slice::from_raw_parts(readback_mapped.cast::<u8>(), byte_count) };
        let mut verified_pixels = 0usize;
        let mut mismatch_samples = Vec::new();
        for index in 0..pixel_count {
            let base = index * 4;
            let within_tolerance = (0..4).all(|channel| {
                output[base + channel].abs_diff(expected[base + channel]) <= verification_tolerance
            });
            if within_tolerance {
                verified_pixels += 1;
            } else if mismatch_samples.len() < 8 {
                mismatch_samples.push(format!(
                    "{}:{:?}!={:?}",
                    index,
                    &output[base..base + 4],
                    &expected[base..base + 4]
                ));
            }
        }
        let checksum = fnv1a64(output);
        if verified_pixels != pixel_count
            || (verification_tolerance == 0 && checksum != expected_checksum)
        {
            return Err(anyhow!(
                "RGBA image verification failed: verified={verified_pixels}/{pixel_count} tolerance={verification_tolerance} checksum={checksum:016x} expected_checksum={expected_checksum:016x} mismatches=[{}]",
                mismatch_samples.join(",")
            ));
        }

        Ok(VulkanImageProbeResult {
            gpu: VulkanGpuProbeResult {
                luid: requested_luid,
                name,
                vendor_id: properties.vendor_id,
                device_id: properties.device_id,
                api_version: properties.api_version,
                queue_family_index,
                queue_flags,
            },
            spirv_words: spirv.len(),
            verified_pixels,
            checksum,
            expected_checksum,
        })
    })();

    unsafe {
        instance.destroy_instance(None);
    }
    result
}

/// Keep the v610 synthetic probe API unchanged so the already-proven 8x8 test
/// remains available as a regression check.
pub fn probe_glsl_image_selected_luid(requested_luid: u64) -> Result<VulkanImageProbeResult> {
    let input = image_probe_input();
    probe_glsl_image_input_selected_luid(
        requested_luid,
        &input,
        IMAGE_PROBE_WIDTH,
        IMAGE_PROBE_HEIGHT,
    )
}

/// Opt-in v611 real-capture shadow probe. The render thread supplies one 8x8
/// RGBA8 sample copied from an actual accepted WGC frame. The same proven
/// storage-image shader swaps R/B on the exact selected Vulkan GPU and verifies
/// every output pixel. This function owns no OpenGL/WGC/ONNX resource.
#[derive(Clone, Debug)]
pub struct VulkanCaptureSampleProbeResult {
    pub image: VulkanImageProbeResult,
    pub input_checksum: u64,
}

pub fn capture_probe_requested() -> bool {
    std::env::var("NEO_VULKAN_CAPTURE_PROBE")
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

pub fn probe_capture_sample_selected_luid(
    requested_luid: u64,
    sample_rgba8: &[u8],
) -> Result<VulkanCaptureSampleProbeResult> {
    if sample_rgba8.len() != IMAGE_PROBE_BYTE_COUNT {
        return Err(anyhow!(
            "capture sample requires exactly {} bytes (8x8 RGBA8), got {}",
            IMAGE_PROBE_BYTE_COUNT,
            sample_rgba8.len()
        ));
    }
    let input_checksum = fnv1a64(sample_rgba8);
    let image = probe_glsl_image_input_selected_luid(
        requested_luid,
        sample_rgba8,
        IMAGE_PROBE_WIDTH,
        IMAGE_PROBE_HEIGHT,
    )?;
    Ok(VulkanCaptureSampleProbeResult {
        image,
        input_checksum,
    })
}

// ---------------------------------------------------------------------------
// v612 real-frame 64x64 tile shadow probe.
// ---------------------------------------------------------------------------

pub const FRAME_TILE_PROBE_WIDTH: u32 = 64;
pub const FRAME_TILE_PROBE_HEIGHT: u32 = 64;
pub const FRAME_TILE_PROBE_PIXEL_COUNT: usize =
    (FRAME_TILE_PROBE_WIDTH as usize) * (FRAME_TILE_PROBE_HEIGHT as usize);
pub const FRAME_TILE_PROBE_BYTE_COUNT: usize = FRAME_TILE_PROBE_PIXEL_COUNT * 4;

#[derive(Clone, Debug)]
pub struct VulkanFrameTileProbeResult {
    pub image: VulkanImageProbeResult,
    pub input_checksum: u64,
}

pub fn frame_probe_requested() -> bool {
    std::env::var("NEO_VULKAN_FRAME_PROBE")
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

/// Process one real-capture 64x64 RGBA8 tile on the exact selected Vulkan GPU.
/// The CPU expected transform is byte-exact and is used only as a diagnostic
/// reference. The returned pixels never feed the production OpenGL path.
pub fn probe_capture_tile_selected_luid(
    requested_luid: u64,
    tile_rgba8: &[u8],
) -> Result<VulkanFrameTileProbeResult> {
    if tile_rgba8.len() != FRAME_TILE_PROBE_BYTE_COUNT {
        return Err(anyhow!(
            "frame tile requires exactly {} bytes (64x64 RGBA8), got {}",
            FRAME_TILE_PROBE_BYTE_COUNT,
            tile_rgba8.len()
        ));
    }
    let input_checksum = fnv1a64(tile_rgba8);
    let image = probe_glsl_image_input_selected_luid(
        requested_luid,
        tile_rgba8,
        FRAME_TILE_PROBE_WIDTH,
        FRAME_TILE_PROBE_HEIGHT,
    )?;
    Ok(VulkanFrameTileProbeResult {
        image,
        input_checksum,
    })
}

// ---------------------------------------------------------------------------
// v613 real Neo user-GLSL one-pass shadow runner.
// ---------------------------------------------------------------------------

pub const USER_GLSL_TILE_WIDTH: u32 = 64;
pub const USER_GLSL_TILE_HEIGHT: u32 = 64;
pub const USER_GLSL_TILE_BYTE_COUNT: usize =
    (USER_GLSL_TILE_WIDTH as usize) * (USER_GLSL_TILE_HEIGHT as usize) * 4;

#[derive(Clone, Debug)]
pub struct VulkanUserGlslProbeResult {
    pub image: VulkanImageProbeResult,
    pub shader_name: String,
    pub pass_desc: String,
    pub input_checksum: u64,
    pub tolerance: u8,
}

pub fn user_glsl_probe_requested() -> bool {
    std::env::var("NEO_VULKAN_USER_GLSL_PROBE")
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn v613_one_pass_eligible(shader: &UserShader) -> Result<&super::mpv::Pass> {
    if shader.passes.len() != 1 {
        return Err(anyhow!(
            "v613 one-pass runner requires exactly one pass, shader has {}",
            shader.passes.len()
        ));
    }
    if !shader.params.is_empty() {
        return Err(anyhow!(
            "v613 one-pass runner does not support //!PARAM yet"
        ));
    }
    if !shader.textures.is_empty() {
        return Err(anyhow!(
            "v613 one-pass runner does not support //!TEXTURE yet"
        ));
    }
    if shader.uses_chroma || shader.is_compute {
        return Err(anyhow!(
            "v613 one-pass runner supports ordinary RGB fragment-style hooks only"
        ));
    }
    let pass = &shader.passes[0];
    if pass.compute.is_some()
        || pass.save.is_some()
        || pass.width.is_some()
        || pass.height.is_some()
        || pass.components != 4
        || pass.offset != PassOffset::None
    {
        return Err(anyhow!(
            "v613 one-pass runner requires no COMPUTE/SAVE/resize/OFFSET and 4 output components"
        ));
    }
    if pass.binds.len() != 1 || !pass.binds[0].eq_ignore_ascii_case("HOOKED") {
        return Err(anyhow!(
            "v613 one-pass runner requires exactly one //!BIND HOOKED"
        ));
    }
    let code = pass.code();
    for unsupported in [
        "HOOKED_raw",
        "texelFetch",
        "textureGrad",
        "textureLod",
        "dFdx",
        "dFdy",
        "fwidth",
        "gl_FragCoord",
    ] {
        if code.contains(unsupported) {
            return Err(anyhow!(
                "v613 one-pass runner does not support `{unsupported}` in compute mode"
            ));
        }
    }
    Ok(pass)
}

fn v613_user_glsl_compute_source(pass: &super::mpv::Pass, width: u32, height: u32) -> String {
    let body = pass.code();
    format!(
        r#"#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(rgba8, set = 0, binding = 0) readonly uniform image2D src_img;
layout(rgba8, set = 0, binding = 1) writeonly uniform image2D dst_img;

const vec2 HOOKED_size = vec2({width}.0, {height}.0);
const vec2 HOOKED_pt = vec2(1.0 / {width}.0, 1.0 / {height}.0);
#define HOOKED_pos ((vec2(gl_GlobalInvocationID.xy) + vec2(0.5)) * HOOKED_pt)
#define HOOKED_tex(pos) imageLoad(src_img, clamp(ivec2((pos) * HOOKED_size), ivec2(0), ivec2({width_minus}, {height_minus})))
#define HOOKED_texOff(off) imageLoad(src_img, clamp(ivec2(gl_GlobalInvocationID.xy) + ivec2(off), ivec2(0), ivec2({width_minus}, {height_minus})))
#define HOOKED_raw src_img

{body}

void main() {{
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x < {width} && p.y < {height}) {{
        imageStore(dst_img, p, clamp(hook(), 0.0, 1.0));
    }}
}}
"#,
        width_minus = width.saturating_sub(1),
        height_minus = height.saturating_sub(1),
    )
}

fn compile_v613_user_glsl_to_spirv(
    pass: &super::mpv::Pass,
    width: u32,
    height: u32,
) -> Result<Vec<u32>> {
    let source = v613_user_glsl_compute_source(pass, width, height);
    let mut frontend = naga::front::glsl::Frontend::default();
    let options = naga::front::glsl::Options::from(naga::ShaderStage::Compute);
    let module = frontend
        .parse(&options, &source)
        .map_err(|error| anyhow!("Naga user-GLSL parse failed: {error:?}"))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|error| anyhow!("Naga user-GLSL validation failed: {error:?}"))?;
    let spv_options = naga::back::spv::Options {
        lang_version: (1, 3),
        ..Default::default()
    };
    naga::back::spv::write_vec(&module, &info, &spv_options, None)
        .map_err(|error| anyhow!("Naga user-GLSL SPIR-V generation failed: {error:?}"))
}

/// CPU reference for the bundled deint_swa one-pass shader. The conversion to
/// RGBA8 may differ by one code value between implementations, so the v613
/// validation deliberately allows +/-1 per channel while still checking every
/// one of the 4096 output pixels.
fn v613_deint_swa_expected(input: &[u8], width: u32, height: u32) -> Vec<u8> {
    let mut out = vec![0u8; input.len()];
    let w = width as usize;
    let h = height as usize;
    for y in 0..h {
        let ya = y.saturating_sub(1);
        let yb = (y + 1).min(h - 1);
        for x in 0..w {
            for c in 0..4 {
                let cur = input[(y * w + x) * 4 + c] as f32;
                let above = input[(ya * w + x) * 4 + c] as f32;
                let below = input[(yb * w + x) * 4 + c] as f32;
                let value = cur * 0.5 + above * 0.25 + below * 0.25;
                out[(y * w + x) * 4 + c] = value.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

pub fn probe_real_user_glsl_selected_luid(
    requested_luid: u64,
    tile_rgba8: &[u8],
    shader_path: &std::path::Path,
) -> Result<VulkanUserGlslProbeResult> {
    if tile_rgba8.len() != USER_GLSL_TILE_BYTE_COUNT {
        return Err(anyhow!(
            "user GLSL tile requires exactly {} bytes (64x64 RGBA8), got {}",
            USER_GLSL_TILE_BYTE_COUNT,
            tile_rgba8.len()
        ));
    }
    let path_string = shader_path.to_string_lossy().into_owned();
    let shader = UserShader::load(&path_string)
        .with_context(|| format!("load real user GLSL {}", shader_path.display()))?;
    if !shader.name().eq_ignore_ascii_case("deint_swa.glsl") {
        return Err(anyhow!(
            "v613 CPU verification is defined only for deint_swa.glsl, got {}",
            shader.name()
        ));
    }
    let pass = v613_one_pass_eligible(&shader)?;
    let spirv = compile_v613_user_glsl_to_spirv(pass, USER_GLSL_TILE_WIDTH, USER_GLSL_TILE_HEIGHT)?;
    let expected = v613_deint_swa_expected(tile_rgba8, USER_GLSL_TILE_WIDTH, USER_GLSL_TILE_HEIGHT);
    let tolerance = 1u8;
    let image = run_glsl_image_input_selected_luid(
        requested_luid,
        tile_rgba8,
        USER_GLSL_TILE_WIDTH,
        USER_GLSL_TILE_HEIGHT,
        spirv,
        expected,
        tolerance,
        "cHiDeScaler-Neo Vulkan User GLSL Probe",
    )?;
    Ok(VulkanUserGlslProbeResult {
        image,
        shader_name: shader.name(),
        pass_desc: pass.desc.clone(),
        input_checksum: fnv1a64(tile_rgba8),
        tolerance,
    })
}

#[cfg(test)]
mod v609_tests {
    use super::*;

    #[test]
    fn auto_route_never_selects_vulkan() {
        assert_eq!(glsl_backend_route(None), GlslBackendRoute::LegacyOpenGl);
        assert_eq!(
            glsl_backend_route(Some(0x1234)),
            GlslBackendRoute::VulkanSelected { luid: 0x1234 }
        );
    }

    #[test]
    fn glsl_probe_source_compiles_to_spirv() {
        let words = compile_probe_glsl_to_spirv().expect("GLSL probe source should compile");
        assert!(!words.is_empty());
        assert_eq!(words[0], 0x0723_0203);
    }

    #[test]
    fn glsl_probe_expected_checksum_is_stable() {
        let checksum: u64 = (0..GLSL_PROBE_VALUE_COUNT)
            .map(|index| (index as u64) * 2 + 1)
            .sum();
        assert_eq!(checksum, 4096);
    }

    #[test]
    fn v613_real_deint_swa_is_one_pass_eligible_and_compiles() {
        let source = include_str!("../../shaders/Deinterlace/deint_swa.glsl");
        let shader = UserShader::parse("deint_swa.glsl", source);
        let pass = v613_one_pass_eligible(&shader).expect("deint_swa should fit v613 subset");
        let words = compile_v613_user_glsl_to_spirv(pass, 64, 64)
            .expect("deint_swa Vulkan wrapper should compile");
        assert!(!words.is_empty());
        assert_eq!(words[0], 0x0723_0203);
    }

    #[test]
    fn v613_deint_cpu_reference_keeps_uniform_pixels_stable() {
        let input = vec![64u8; USER_GLSL_TILE_BYTE_COUNT];
        let expected = v613_deint_swa_expected(&input, 64, 64);
        assert_eq!(expected, input);
    }

    #[test]
    fn image_probe_source_tracks_dynamic_dimensions() {
        let source = image_probe_source(64, 32);
        assert!(source.contains("p.x < 64"));
        assert!(source.contains("p.y < 32"));
    }

    #[test]
    fn image_probe_expected_is_size_independent() {
        let input = [1u8, 2, 3, 4, 10, 20, 30, 40];
        assert_eq!(
            image_probe_expected(&input),
            vec![3, 2, 1, 4, 30, 20, 10, 40]
        );
    }
}
