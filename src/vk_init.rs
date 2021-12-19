use std::sync::Arc;
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{Device, DeviceExtensions, Features, QueuesIter};
use vulkano::instance::{Instance, InstanceExtensions};
use vulkano::Version;

pub fn init() -> (Arc<Device>, QueuesIter) {
    // Note RPI4 claims VK1.1 compliance
    let instance = Instance::new(None, Version::V1_1, &InstanceExtensions::none(), None).unwrap();

    // extensions
    let device_extensions = DeviceExtensions {
        // Note: RPI4 doesn't support storage buffers
        // khr_storage_buffer_storage_class: true,
        ..DeviceExtensions::none()
    };

    // queue devices
    let (physical_device, queue_family) = PhysicalDevice::enumerate(&instance)
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

    // init device
    Device::new(
        physical_device,
        &Features::none(),
        &physical_device
            .required_extensions()
            .union(&device_extensions),
        [(queue_family, 0.5)].iter().cloned(),
    )
    .unwrap()
}
