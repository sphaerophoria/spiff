use libdiff::{DiffAlgo, DiffAlgoAction, DiffAlgoDebugInfo, MyersTrace};
use eframe::{egui::{self, Galley, Vec2, TextStyle, Painter, Color32, Pos2, Ui}, epaint::Stroke};
use std::{sync::Arc, rc::Rc, pin::Pin};

struct Args {
    a: Vec<i32>,
    b: Vec<i32>,
    top: Option<usize>,
    bottom: Option<usize>,
    left: Option<usize>,
    right: Option<usize>,
    forgetful: bool,
    backwards: bool,
}

impl Args {
    fn parse<It: Iterator<Item=String>>(mut it: It) -> Args {
        let mut it = it.peekable();
        let program_name = it.next().unwrap_or_else(|| "diff_viz".to_string());

        let mut top = None;
        let mut bottom = None;
        let mut left = None;
        let mut right = None;
        let mut forgetful = false;
        let mut backwards = false;
        let mut a = Vec::new();
        let mut b = Vec::new();

        let parse_i32_arg_list = |output: &mut Vec<i32>, it: &mut std::iter::Peekable<It>| {
            while let Some(v) = it.peek().and_then(|x| x.parse::<i32>().ok()) {
                it.next();
                output.push(v)
            }
        };
        while let Some(arg) = it.next() {
            match arg.as_ref() {
                "--a" => {
                    parse_i32_arg_list(&mut a, &mut it);
                }
                "--b" => {
                    parse_i32_arg_list(&mut b, &mut it);
                }
                "--top" => top = it.next(),
                "--left" => left = it.next(),
                "--right" => right = it.next(),
                "--bottom" => bottom = it.next(),
                "--forgetful" => forgetful = true,
                "--backwards" => backwards = true,
                "--help" => {
                    Self::help(&program_name);
                }
                e => {
                    eprintln!("Unexpected argument: {e}");
                    Self::help(&program_name);
                }
            }
        }

        let res = (|| -> Result<Args, &'static str> {
            macro_rules! parse_arg {
                ($x:expr) => {{
                    let res = $x.map(|x| x.parse().map_err(|_| concat!(stringify!($x), " is not a valid usize")));
                    res.transpose()?
                }}
            }
            let top = parse_arg!(top);
            let left = parse_arg!(left);
            let right = parse_arg!(right);
            let bottom = parse_arg!(bottom);

            if backwards && forgetful {
                return Err("forgetful cannot be set in backwards mode");
            }

            Ok(Args {
                top, left, bottom, right, forgetful, backwards, a, b,
            })
        })();

        match res {
            Ok(v) => v,
            Err(e) => {
                eprintln!("{e}");
                Self::help(&program_name);
            }
        }
    }

    fn help(program_name: &str) -> ! {
        eprintln!("Usage: {program_name} [ARGS]\n\
                Args:\n\
                --top: Search start in b\n\
                --left: Search start in a\n\
                --bottom: Search end in b\n\
                --right: Search end in a\n\
                --forgetful: Use forgetful version of algo"
                );
        std::process::exit(1);
    }
}

fn main() {
    let args = Args::parse(std::env::args());
    let native_options = eframe::NativeOptions::default();
    eframe::run_native("My egui App", native_options, Box::new(|cc| Box::new(MyEguiApp::new(cc, args))));
}

fn lay_out_elems(elems: &[i32], ui: &Ui) -> Vec<Arc<Galley>> {
   let mut laid_out = Vec::new();
   let font_id = TextStyle::Monospace.resolve(ui.style());
   ui.fonts(|f| {
       for item in elems {
           laid_out.push(f.layout_no_wrap(
               item.to_string(), font_id.clone(), ui.style().visuals.text_color()));

       }
   });
   laid_out
}

fn draw_elem_at_pos(elem: &Arc<Galley>, pos: Pos2, painter: &Painter) {
    painter.galley(pos, Arc::clone(&elem), Color32::WHITE);
}

struct Grid {
    x_start: f32,
    x_spacing: f32,
    y_start: f32,
    y_spacing: f32,
}

impl Grid {
    fn x_spacing(&self) -> f32 {
        self.x_spacing
    }

    fn y_spacing(&self) -> f32 {
        self.y_spacing
    }

    fn x_idx_to_pos(&self, idx: usize) -> f32 {
        self.x_start + idx as f32 * self.x_spacing
    }

    fn y_idx_to_pos(&self, idx: usize) -> f32 {
        self.y_start + idx as f32 * self.y_spacing
    }
}

struct MyEguiApp {
    a: Vec<i32>,
    b: Vec<i32>,
    algo: DiffAlgo,
    finished: bool,
}

impl MyEguiApp {
    fn new(cc: &eframe::CreationContext<'_>, args: Args) -> Self {
        // Customize egui here with cc.egui_ctx.set_fonts and cc.egui_ctx.set_visuals.
        // Restore app state using cc.storage (requires the "persistence" feature).
        // Use the cc.gl (a glow::Context) to create graphics shaders and buffers that you can use
        // for e.g. egui::PaintCallback.
        let a = args.a;
        let b = args.b;
        let top = args.top.unwrap_or(0) as i64;
        let left = args.left.unwrap_or(0) as i64;
        let right = args.right.unwrap_or(a.len()) as i64;
        let bottom = args.bottom.unwrap_or(b.len()) as i64;
        let algo = if args.backwards {
            DiffAlgo::new_backwards(a.as_ref(), b.as_ref(), top, left, bottom, right)
        } else if args.forgetful {
            DiffAlgo::new_forgetful(a.as_ref(), b.as_ref(), top, left, bottom, right)
        } else {
            DiffAlgo::new(a.as_ref(), b.as_ref(), top, left, bottom, right, 100 * 1024 * 1024).unwrap()
        };
        MyEguiApp {
            a, b, algo, finished: false,
        }
    }
}

impl eframe::App for MyEguiApp {
   fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
       egui::CentralPanel::default().show(ctx, |ui| {

           if ui.button("next").clicked() && !self.finished {
               if self.algo.step(self.a.as_ref(), self.b.as_ref()) == DiffAlgoAction::Finish {
                   self.finished = true;
               }
           }

           let mut a_laid_out = lay_out_elems(self.a.as_ref(), ui);
           let mut b_laid_out = lay_out_elems(self.b.as_ref(), ui);

           let available_size = ui.available_size();
           let mut rect = ui.allocate_space(available_size).1;
           const MARGIN: f32 = 30.0;
           rect.min.x += MARGIN;
           rect.min.y += MARGIN;
           rect.max.x -= MARGIN;
           rect.max.y -= MARGIN;

           // Without the first element, spacing is width / num_elems
           // Spacing == width / (num_elems - 1
           //ui.allocate_space(available_size);
           let painter = ui.painter();
           let x_spacing = rect.width()  / a_laid_out.len() as f32;
           let y_spacing = rect.height()  / b_laid_out.len() as f32;

           let grid = Grid {
               x_spacing,
               y_spacing,
               x_start: rect.left(),
               y_start: rect.top(),
           };

           for (idx, elem) in a_laid_out.iter().enumerate() {
               draw_elem_at_pos(elem, egui::pos2(grid.x_idx_to_pos(idx + 1), 10.0), painter);
           }

           for (idx, elem) in b_laid_out.iter().enumerate() {
               draw_elem_at_pos(elem, egui::pos2(10.0, grid.y_idx_to_pos(idx + 1)), painter);
           }

           let color = ui.style().visuals.text_color();
           let stroke = Stroke::new(4.0, color);
           for y_idx in 0..=self.b.len() {
               let x_start = grid.x_idx_to_pos(0);
               let x_end = grid.x_idx_to_pos(self.a.len());
               let y = grid.y_idx_to_pos(y_idx);
               painter.line_segment([[x_start, y].into(), [x_end, y].into()], stroke);
           }

           for x_idx in 0..=self.a.len() {
               let y_start = grid.y_idx_to_pos(0);
               let y_end = grid.y_idx_to_pos(self.b.len());
               let x = grid.x_idx_to_pos(x_idx);
               painter.line_segment([[x, y_start].into(), [x, y_end].into()], stroke);
           }

           let debug_info = self.algo.debug_info(&self.a, &self.b);
           for y_idx in 0..=self.b.len() {
               for x_idx in 0..=self.a.len() {
                   let last_step = debug_info.steps.last().and_then(|v| v.last());
                   let color = if let Some(last_step) = last_step {
                       if y_idx == last_step.1 as usize && x_idx == last_step.0 as usize {
                           Color32::GREEN
                       } else {
                           color
                       }
                   } else {
                       color
                   };

                   let x_pos = grid.x_idx_to_pos(x_idx);
                   let y_pos = grid.y_idx_to_pos(y_idx);

                   if y_idx < self.b.len() && x_idx < self.a.len() && self.b.get(y_idx) == self.a.get(x_idx) {
                       let x_end = grid.x_idx_to_pos(x_idx + 1);
                       let y_end = grid.y_idx_to_pos(y_idx + 1);
                       painter.line_segment([[x_pos, y_pos].into(), [x_end, y_end].into()], stroke);
                   }

                   painter.circle_filled(egui::pos2(x_pos, y_pos), 10.0, color);
               }
           }

           for k_steps in debug_info.steps {
               for step in k_steps.windows(2) {
                   if step[0].0 < 0 || step[0].1 < 0 || step[1].0 < 0 || step[1].1 < 0 {
                       continue;
                   }

                   let line_start_x = grid.x_idx_to_pos(step[0].0 as usize);
                   let line_start_y = grid.y_idx_to_pos(step[0].1 as usize);
                   let line_end_x = grid.x_idx_to_pos(step[1].0 as usize);
                   let line_end_y = grid.y_idx_to_pos(step[1].1 as usize);
                   let mut move_stroke = stroke;
                   move_stroke.color = Color32::GREEN;
                   painter.line_segment([[line_start_x, line_start_y].into(), [line_end_x, line_end_y].into()], move_stroke);

               }
           }

           let left_px = grid.x_idx_to_pos(debug_info.left as usize) - grid.x_spacing() / 2.0;
           let right_px = grid.x_idx_to_pos(debug_info.right as usize) + grid.x_spacing() / 2.0;
           let top_px = grid.y_idx_to_pos(debug_info.top as usize) - grid.y_spacing() / 2.0;
           let bottom_px = grid.y_idx_to_pos(debug_info.bottom as usize) + grid.y_spacing() / 2.0;

           let rect = egui::Rect {
               min: egui::pos2(left_px, top_px),
               max: egui::pos2(right_px, bottom_px),
           };
           let mut rect_stroke = stroke;
           rect_stroke.color = Color32::from_rgba_unmultiplied(0, 255, 255, 255);

           painter.rect_stroke(rect, 3.0, rect_stroke)

       });
   }
}
