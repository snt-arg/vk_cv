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
        path: "src/shaders/color_filter.comp.glsl",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}

pub struct ColorFilter {
    rgb_min: [f32; 3],
    rgb_max: [f32; 3],
}

impl ColorFilter {
    pub fn new(rgb_min: [f32; 3], rgb_max: [f32; 3]) -> Self {
        Self { rgb_min, rgb_max }
    }
}

impl ProcessingElement for ColorFilter {
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
        let push_constants = cs::ty::PushConstants {
            rgb_min: self.rgb_min,
            rgb_max: self.rgb_max,
            _dummy0: [0, 0, 0, 0],
        };

        builder
            .bind_pipeline_compute(pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline.layout().clone(),
                0,
                set,
            )
            .push_constants(pipeline.layout().clone(), 0, push_constants)
            .dispatch(utils::workgroups(
                &input_img.dimensions().width_height(),
                &[16, 16],
            ))
            .unwrap();

        IoFragment {
            input: Io::Image(input_img),
            output: Io::Image(output_img.clone()),
            label: utils::basic_label("ColorFilter", &output_img),
        }
    }
}
