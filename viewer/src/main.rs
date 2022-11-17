use eframe::egui;
use egui::{Color32, ColorImage};
use egui_extras::RetainedImage;
use vkcv::utils::image_to_rgba8;

mod pipeline;

fn main() {
    let options = eframe::NativeOptions {
        initial_window_size: Some(egui::vec2(500.0, 900.0)),
        ..Default::default()
    };

    eframe::run_native(
        "vkcv viewer",
        options,
        Box::new(|_cc| Box::new(MyApp::default())),
    );
}

struct MyApp {
    image: RetainedImage,
    pipeline: pipeline::Pipeline,
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            image: RetainedImage::from_color_image(
                "",
                ColorImage::new([32, 32], Color32::from_rgb(0, 0, 0)),
            ),
            pipeline: pipeline::init(),
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("pipeline output");
            self.image.show(ui);

            let res = self.pipeline.fetch_and_process();

            self.image = RetainedImage::from_color_image(
                "",
                ColorImage::from_rgba_unmultiplied(
                    [
                        res.image.info.width as usize,
                        res.image.info.height as usize,
                    ],
                    &res.image.buffer,
                ),
            );

            match res.target_pos {
                Some(point) => ui.heading(format!(
                    "X: {:.2}, Y: {:.2}, Z: {:.2}, Area: {}, dt= {:.2}ms",
                    point[0],
                    point[1],
                    point[2],
                    res.area,
                    res.dt.as_secs_f64() / 1e3
                )),
                None => ui.heading("Nothing detected"),
            };

            egui::ScrollArea::new([false, true]).show(ui, |ui| {
                for dl in &self.pipeline.download {
                    let input_image = dl.transferred_image();
                    let (info, rgba) =
                        image_to_rgba8(input_image.info(), input_image.buffer_content());
                    let input_image = RetainedImage::from_color_image(
                        "",
                        ColorImage::from_rgba_unmultiplied(
                            [info.width as usize, info.height as usize],
                            &rgba,
                        ),
                    );
                    input_image.show(ui);
                }
            });

            ctx.request_repaint();
        });
    }
}
