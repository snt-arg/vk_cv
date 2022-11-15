use std::error::Error;
use std::sync::Arc;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::{
    FreeListAllocator, GenericMemoryAllocator, StandardMemoryAllocator,
};
use vulkano::{Version, VulkanLibrary};

// type MemoryAllocator
pub struct Memory {
    pub allocator: GenericMemoryAllocator<Arc<FreeListAllocator>>,
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
}

pub struct VkContext {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub memory: Memory,
}

pub fn init() -> Result<VkContext, Box<dyn Error>> {
    // Note RPI4 claims VK1.1 'compliance'
    let ci = InstanceCreateInfo {
        max_api_version: Some(Version::V1_1),
        ..Default::default()
    };
    let instance = Instance::new(VulkanLibrary::new().unwrap(), ci)?;

    // extensions
    let device_extensions = DeviceExtensions {
        // Note: RPI4 doesn't support storage buffers
        // khr_storage_buffer_storage_class: true,
        ..DeviceExtensions::empty()
    };

    // queue devices
    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            // The Vulkan specs guarantee that a compliant implementation must provide at least one queue
            // that supports compute operations.
            p.queue_family_properties()
                .iter()
                .position(|q| q.queue_flags.compute)
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
        })
        .unwrap();

    println!(
        ">> Selected Device: '{}' (type: '{:?}')",
        physical_device.properties().device_name,
        physical_device.properties().device_type
    );

    // init device
    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )?;

    let queue = queues.next().unwrap();
    let allocator = StandardMemoryAllocator::new_default(device.clone());
    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        Default::default(),
    ));
    let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(device.clone()));

    let memory = Memory {
        allocator,
        command_buffer_allocator,
        descriptor_set_allocator,
    };

    Ok(VkContext {
        device,
        queue,
        memory,
    })
}
