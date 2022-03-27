use std::sync::Arc;
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::Version;

pub fn init() -> (Arc<Device>, Arc<Queue>) {
    // Note RPI4 claims VK1.1 'compliance'
    let ci = InstanceCreateInfo {
        max_api_version: Some(Version::V1_1),
        ..Default::default()
    };
    let instance = Instance::new(ci).unwrap();

    // extensions
    let device_extensions = DeviceExtensions {
        // Note: RPI4 doesn't support storage buffers
        // khr_storage_buffer_storage_class: true,
        ..DeviceExtensions::none()
    };

    // queue devices
    let (physical_device, _queue_family) = PhysicalDevice::enumerate(&instance)
        .filter(|&p| p.supported_extensions().is_superset_of(&device_extensions))
        .filter_map(|p| {
            // The Vulkan specs guarantee that a compliant implementation must provide at least one queue
            // that supports compute operations.
            p.queue_families()
                .find(|&q| q.supports_compute())
                .map(|q| (p, q))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
        })
        .unwrap();

    println!(
        ">> Selected Device: '{}' (type: '{:?}')",
        physical_device.properties().device_name,
        physical_device.properties().device_type
    );

    let (_gfx_index, queue_family_graphics) = physical_device
        .queue_families()
        .enumerate()
        .find(|&(_i, q)| q.supports_compute())
        .unwrap();

    // init device
    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: physical_device
                .required_extensions()
                .union(&device_extensions),
            queue_create_infos: vec![QueueCreateInfo {
                queues: vec![0.5],
                ..QueueCreateInfo::family(queue_family_graphics)
            }],
            ..Default::default()
        },
    )
    .unwrap();

    (device, queues.next().unwrap())
}
