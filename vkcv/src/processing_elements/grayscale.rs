use vulkano::{
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    format::Format,
    image::{view::ImageView, ImageAccess},
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
};

use crate::{
    utils::{self, ImageInfo},
    vk_init::VkContext,
};

use super::{AutoCommandBufferBuilder, Io, IoFragment, ProcessingElement};

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/grayscale.comp.glsl",
    }
}

pub struct Grayscale {}

impl Grayscale {
    pub fn new() -> Self {
        Self {}
    }
}

impl ProcessingElement for Grayscale {
    fn build(
        &self,
        ctx: &VkContext,
        builder: &mut AutoCommandBufferBuilder,
        input: &IoFragment,
    ) -> IoFragment {
        let pipeline = {
            let shader = cs::load(ctx.device.clone()).unwrap();
            ComputePipeline::new(
                ctx.device.clone(),
                shader.entry_point("main").unwrap(),
                &cs::SpecializationConstants {},
                None,
                |_| {},
            )
            .unwrap()
        };

        // input image
        let input_img = input.output_image().unwrap();

        // output image
        let output_img =
            utils::create_storage_image(ctx, &ImageInfo::from_image(&input_img, Format::R8_UNORM));

        // setup layout
        let layout = pipeline.layout().set_layouts().get(0).unwrap();
        let input_img_view = ImageView::new_default(input_img.clone()).unwrap();
        let output_img_view = ImageView::new_default(output_img.clone()).unwrap();

        let set = PersistentDescriptorSet::new(
            &ctx.memory.descriptor_set_allocator,
            layout.clone(),
            [
                WriteDescriptorSet::image_view(0, input_img_view),
                WriteDescriptorSet::image_view(1, output_img_view),
            ],
        )
        .unwrap();

        // build command buffer
        builder
            .bind_pipeline_compute(pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline.layout().clone(),
                0,
                set,
            )
            .dispatch(utils::workgroups(
                &input_img.dimensions().width_height(),
                &[16, 16],
            ))
            .unwrap();

        IoFragment {
            input: Io::Image(input_img),
            output: Io::Image(output_img.clone()),
            label: utils::basic_label("Grayscale", &output_img),
        }
    }
}
