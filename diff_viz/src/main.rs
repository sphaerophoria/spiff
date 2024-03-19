use eframe::{
    egui::{self, Color32, Galley, Painter, Pos2, Rect, TextStyle, Ui},
    epaint::Stroke,
};
use libdiff::{DiffAlgo, DiffAlgoDebugInfo, DiffAlgoState};
use std::sync::Arc;

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
    fn parse<It: Iterator<Item = String>>(it: It) -> Args {
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
                    let res = $x.map(|x| {
                        x.parse()
                            .map_err(|_| concat!(stringify!($x), " is not a valid usize"))
                    });
                    res.transpose()?
                }};
            }
            let top = parse_arg!(top);
            let left = parse_arg!(left);
            let right = parse_arg!(right);
            let bottom = parse_arg!(bottom);

            if backwards && forgetful {
                return Err("forgetful cannot be set in backwards mode");
            }

            Ok(Args {
                top,
                left,
                bottom,
                right,
                forgetful,
                backwards,
                a,
                b,
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
        eprintln!(
            "Usage: {program_name} [ARGS]\n\
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
    eframe::run_native(
        "Diff Viz",
        native_options,
        Box::new(|cc| Box::new(DiffViz::new(cc, args))),
    )
    .unwrap();
}

fn lay_out_elems(elems: &[i32], ui: &Ui) -> Vec<Arc<Galley>> {
    let mut laid_out = Vec::new();
    let font_id = TextStyle::Monospace.resolve(ui.style());
    ui.fonts(|f| {
        for item in elems {
            laid_out.push(f.layout_no_wrap(
                item.to_string(),
                font_id.clone(),
                ui.style().visuals.text_color(),
            ));
        }
    });
    laid_out
}

fn draw_elem_at_pos(elem: &Arc<Galley>, pos: Pos2, painter: &Painter) {
    painter.galley(pos, Arc::clone(elem), Color32::WHITE);
}

fn construct_diff_algo(args: &Args) -> DiffAlgo {
    let top = args.top.unwrap_or(0) as i64;
    let left = args.left.unwrap_or(0) as i64;
    let right = args.right.unwrap_or(args.a.len()) as i64;
    let bottom = args.bottom.unwrap_or(args.b.len()) as i64;

    if args.backwards {
        DiffAlgo::new_backwards(args.a.as_ref(), args.b.as_ref(), top, left, bottom, right)
    } else if args.forgetful {
        DiffAlgo::new_forgetful(args.a.as_ref(), args.b.as_ref(), top, left, bottom, right)
    } else {
        DiffAlgo::new(
            args.a.as_ref(),
            args.b.as_ref(),
            top,
            left,
            bottom,
            right,
            100 * 1024 * 1024,
        )
        .unwrap()
    }
}

fn add_margin_to_rect(mut rect: Rect) -> Rect {
    const MARGIN: f32 = 30.0;
    rect.min.x += MARGIN;
    rect.min.y += MARGIN;
    rect.max.x -= MARGIN;
    rect.max.y -= MARGIN;
    rect
}

fn draw_headers(a: &[i32], b: &[i32], grid: &Grid, ui: &Ui) {
    let a_laid_out = lay_out_elems(a, ui);
    let b_laid_out = lay_out_elems(b, ui);
    let painter = ui.painter();
    for (idx, elem) in a_laid_out.iter().enumerate() {
        draw_elem_at_pos(elem, egui::pos2(grid.x_idx_to_pos(idx + 1), 10.0), painter);
    }

    for (idx, elem) in b_laid_out.iter().enumerate() {
        draw_elem_at_pos(elem, egui::pos2(10.0, grid.y_idx_to_pos(idx + 1)), painter);
    }
}

fn draw_grid_lines(width: usize, height: usize, grid: &Grid, painter: &Painter, stroke: Stroke) {
    for y_idx in 0..=height {
        let x_start = grid.x_idx_to_pos(0);
        let x_end = grid.x_idx_to_pos(width);
        let y = grid.y_idx_to_pos(y_idx);
        painter.line_segment([[x_start, y].into(), [x_end, y].into()], stroke);
    }

    for x_idx in 0..=height {
        let y_start = grid.y_idx_to_pos(0);
        let y_end = grid.y_idx_to_pos(width);
        let x = grid.x_idx_to_pos(x_idx);
        painter.line_segment([[x, y_start].into(), [x, y_end].into()], stroke);
    }
}

fn draw_diagonals(grid: &Grid, a: &[i32], b: &[i32], painter: &Painter, stroke: Stroke) {
    for y_idx in 0..=b.len() {
        for x_idx in 0..=a.len() {
            let x_pos = grid.x_idx_to_pos(x_idx);
            let y_pos = grid.y_idx_to_pos(y_idx);

            if y_idx < b.len() && x_idx < a.len() && b.get(y_idx) == a.get(x_idx) {
                let x_end = grid.x_idx_to_pos(x_idx + 1);
                let y_end = grid.y_idx_to_pos(y_idx + 1);
                painter.line_segment([[x_pos, y_pos].into(), [x_end, y_end].into()], stroke);
            }
        }
    }
}

const CIRCLE_RADIUS: f32 = 10.0;

fn draw_dots(grid: &Grid, width: usize, height: usize, painter: &Painter, color: Color32) {
    for y_idx in 0..=height {
        for x_idx in 0..=width {
            let x_pos = grid.x_idx_to_pos(x_idx);
            let y_pos = grid.y_idx_to_pos(y_idx);

            painter.circle_filled(egui::pos2(x_pos, y_pos), CIRCLE_RADIUS, color);
        }
    }
}

fn draw_active_dot(grid: &Grid, debug_info: &DiffAlgoDebugInfo, painter: &Painter, color: Color32) {
    let last_step = debug_info.steps.last().and_then(|v| v.last());
    if let Some(last_step) = last_step {
        if last_step.0 < 0 || last_step.1 < 0 {
            return;
        }
        let x_pos = grid.x_idx_to_pos(last_step.0 as usize);
        let y_pos = grid.y_idx_to_pos(last_step.1 as usize);
        painter.circle_filled(egui::pos2(x_pos, y_pos), CIRCLE_RADIUS, color);
    }
}

fn draw_active_paths(
    grid: &Grid,
    debug_info: &DiffAlgoDebugInfo,
    painter: &Painter,
    stroke: Stroke,
) {
    for k_steps in &debug_info.steps {
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
            painter.line_segment(
                [
                    [line_start_x, line_start_y].into(),
                    [line_end_x, line_end_y].into(),
                ],
                move_stroke,
            );
        }
    }
}

fn draw_bounding_rect(
    grid: &Grid,
    debug_info: &DiffAlgoDebugInfo,
    painter: &Painter,
    stroke: Stroke,
) {
    let left_px = grid.x_idx_to_pos(debug_info.left as usize) - grid.x_spacing() / 2.0;
    let right_px = grid.x_idx_to_pos(debug_info.right as usize) + grid.x_spacing() / 2.0;
    let top_px = grid.y_idx_to_pos(debug_info.top as usize) - grid.y_spacing() / 2.0;
    let bottom_px = grid.y_idx_to_pos(debug_info.bottom as usize) + grid.y_spacing() / 2.0;

    let rect = egui::Rect {
        min: egui::pos2(left_px, top_px),
        max: egui::pos2(right_px, bottom_px),
    };

    painter.rect_stroke(rect, 3.0, stroke);
}

struct Grid {
    x_start: f32,
    x_spacing: f32,
    y_start: f32,
    y_spacing: f32,
}

impl Grid {
    fn from_rect_and_elems(rect: &Rect, a_len: usize, b_len: usize) -> Grid {
        let x_spacing = rect.width() / a_len as f32;
        let y_spacing = rect.height() / b_len as f32;

        Grid {
            x_spacing,
            y_spacing,
            x_start: rect.left(),
            y_start: rect.top(),
        }
    }

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

struct DiffViz {
    a: Vec<i32>,
    b: Vec<i32>,
    algo: DiffAlgo,
    finished: bool,
}

impl DiffViz {
    fn new(_cc: &eframe::CreationContext<'_>, args: Args) -> Self {
        let algo = construct_diff_algo(&args);
        DiffViz {
            a: args.a,
            b: args.b,
            algo,
            finished: false,
        }
    }
}

impl eframe::App for DiffViz {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            if ui.button("next").clicked()
                && !self.finished
                && self.algo.step(self.a.as_ref(), self.b.as_ref()) == DiffAlgoState::Finished
            {
                self.finished = true;
            }

            let available_size = ui.available_size();
            let rect = add_margin_to_rect(ui.allocate_space(available_size).1);

            let painter = ui.painter();
            let grid = Grid::from_rect_and_elems(&rect, self.a.len(), self.b.len());

            draw_headers(&self.a, &self.b, &grid, ui);

            let color = ui.style().visuals.text_color();
            let stroke = Stroke::new(4.0, color);
            draw_grid_lines(self.a.len(), self.b.len(), &grid, painter, stroke);
            draw_diagonals(&grid, &self.a, &self.b, painter, stroke);

            let debug_info = self.algo.debug_info();
            draw_dots(&grid, self.a.len(), self.b.len(), painter, color);
            draw_active_dot(&grid, &debug_info, painter, Color32::GREEN);

            let mut move_stroke = stroke;
            move_stroke.color = Color32::GREEN;
            draw_active_paths(&grid, &debug_info, painter, move_stroke);

            let mut rect_stroke = stroke;
            rect_stroke.color = Color32::from_rgba_unmultiplied(0, 255, 255, 255);
            draw_bounding_rect(&grid, &debug_info, painter, rect_stroke);
        });
    }
}
