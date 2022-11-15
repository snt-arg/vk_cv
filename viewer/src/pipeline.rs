use std::sync::Arc;

use vkcv::{
    draw::{draw_centroid, OwnedImage},
    endpoints::{image_download::ImageDownload, image_upload::ImageUpload},
    processing_elements::{
        color_filter::ColorFilter,
        hsvconv::Hsvconv,
        input::Input,
        morphology::{Morphology, Operation},
        output::Output,
        pooling::{self, Pooling},
        tracker::{self, Canvas, PoolingStrategy, Tracker},
    },
    realsense::Realsense,
    utils::cv_pipeline_sequential,
    utils::ImageInfo,
    vk_init::{self, VkContext},
    vulkano::command_buffer::PrimaryAutoCommandBuffer,
};

use vkcv::vulkano::sync::{self, GpuFuture};

pub struct Pipeline {
    upload: ImageUpload,
    download: ImageDownload,
    ctx: VkContext,
    cam: Realsense,
    cb: Arc<PrimaryAutoCommandBuffer>,
}

pub fn init() -> Pipeline {
    let res = [640, 480];
    let fps = 60;
    let mut camera = Realsense::open(&res, fps, &res, fps).unwrap();

    // grab a couple of frames
    for _ in 0..5 {
        camera.fetch_image(false);
    }

    let img_info = camera.fetch_image(false).0.image_info();

    // init device
    let ctx = vk_init::init().unwrap();

    // create a color tracking pipeline
    let pe_input = Input::new(img_info);
    let pe_hsv = Hsvconv::new();
    let pe_hsv_filter = ColorFilter::new([0.20, 0.4, 0.239], [0.429, 1.0, 1.0]);
    let pe_erode = Morphology::new(Operation::Erode);
    let pe_dilate = Morphology::new(Operation::Dilate);
    let pe_pooling = Pooling::new(pooling::Operation::Max); // 2x2
    let pe_tracker = Tracker::new(PoolingStrategy::Pooling4, Canvas::Pad);
    let pe_out = Output::new();

    let (pipeline_cb, input_io, output_io) = cv_pipeline_sequential(
        &ctx,
        &pe_input,
        &[
            &pe_hsv,
            &pe_hsv_filter,
            &pe_erode,
            &pe_dilate,
            &pe_pooling,
            &pe_tracker,
        ],
        &pe_out,
    );

    let upload = ImageUpload::from_io(input_io).unwrap();
    let download = ImageDownload::from_io(output_io).unwrap();

    Pipeline {
        upload,
        download,
        ctx,
        cam: camera,
        cb: pipeline_cb,
    }
}

pub struct PipelineResult {
    pub image: OwnedImage,
    pub target_pos: Option<[f32; 3]>,
    pub area: u32,
    pub dt: std::time::Duration,
}

pub fn fetch_and_process(p: &mut Pipeline) -> PipelineResult {
    // grab depth and color image from the realsense
    let (color_image, depth_image) = p.cam.fetch_image(true);
    let t0 = std::time::Instant::now();

    // upload image to GPU
    p.upload.copy_input_data(color_image.data_slice());

    // process on GPU
    let future = sync::now(p.ctx.device.clone())
        .then_execute(p.ctx.queue.clone(), p.cb.clone())
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    // wait till finished
    future.wait(None).unwrap(); // spin-lock?
    let dt = std::time::Instant::now().duration_since(t0);

    // get processed depth image
    let depth_image = depth_image.get();

    // print results
    let tf_image = p.download.transfer();
    let (c, area) = tracker::centroid(&tf_image);
    let area_px = (area * color_image.area() as f32) as u32;

    // get the depth only if our object is bigger than 225px² (15x15)
    let mut owned_image = OwnedImage {
        buffer: color_image.data_slice().to_vec(),
        info: ImageInfo {
            width: color_image.width(),
            height: color_image.height(),
            format: vkcv::vulkano::format::Format::R8G8B8A8_UINT,
        },
    };

    let mut point = None;

    if area_px > 16 {
        let pixel_coords = [
            c[0] * color_image.width() as f32,
            c[1] * color_image.height() as f32,
        ];
        let depth = p
            .cam
            .depth_at_pixel(&pixel_coords, &color_image, &depth_image);

        // de-project to obtain a 3D point in camera coordinates
        if let Some(depth) = depth {
            point = Some(p.cam.deproject_pixel(&pixel_coords, depth, &color_image));
        }

        // paint the centroid
        draw_centroid(&mut owned_image, &pixel_coords, 2.0);
    }

    PipelineResult {
        image: owned_image,
        target_pos: point,
        area: area_px,
        dt,
    }
}
