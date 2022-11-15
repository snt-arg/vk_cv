use eframe::egui;
use egui::{Color32, ColorImage};
use egui_extras::RetainedImage;
use pipeline::Pipeline;

mod pipeline;

fn main() {
    let options = eframe::NativeOptions {
        initial_window_size: Some(egui::vec2(500.0, 900.0)),
        ..Default::default()
    };

    eframe::run_native(
        "Show an image with eframe/egui",
        options,
        Box::new(|_cc| Box::new(MyApp::default())),
    );
}

struct MyApp {
    image: RetainedImage,
    tint: egui::Color32,
    pipeline: pipeline::Pipeline,
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            image: RetainedImage::from_color_image(
                "",
                ColorImage::new([100, 100], Color32::from_rgb(0, 0, 0)),
            ),
            tint: egui::Color32::from_rgb(255, 0, 255),
            pipeline: pipeline::init(),
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("pipeline output");
            self.image.show(ui);

            let res = pipeline::fetch_and_process(&mut self.pipeline);
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

            ctx.request_repaint();
        });
    }
}
