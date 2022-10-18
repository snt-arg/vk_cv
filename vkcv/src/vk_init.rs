use std::sync::Arc;
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::{Version, VulkanLibrary};

pub fn init() -> (Arc<Device>, Arc<Queue>) {
    // Note RPI4 claims VK1.1 'compliance'
    let ci = InstanceCreateInfo {
        max_api_version: Some(Version::V1_1),
        ..Default::default()
    };
    let instance = Instance::new(VulkanLibrary::new().unwrap(), ci).unwrap();

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
    )
    .unwrap();

    (device, queues.next().unwrap())
}
