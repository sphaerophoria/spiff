use anyhow::{Context, Result};
use eframe::{
    egui::{
        self, text::LayoutJob, Color32, Context as EContext, Label, RichText, ScrollArea, Slider,
        TextEdit, TextFormat, Ui, Visuals,
    },
    epaint::FontId,
};
use libdiff::{DiffAction, MatchedDiff};
use memmap2::Mmap;

use std::fmt::Write as FmtWrite;

fn main() -> Result<()> {
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "Spiff",
        native_options,
        Box::new(|cc| Box::new(Spiff::new(cc))),
    );
    Ok(())
}

struct Spiff {
    content_a: Mmap,
    content_b: Mmap,
    options: DiffOptions,
    diff_view: Result<DiffView>,
}

impl Spiff {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        // Customize egui here with cc.egui_ctx.set_fonts and cc.egui_ctx.set_visuals.
        // Restore app state using cc.storage (requires the "persistence" feature).
        // Use the cc.gl (a glow::Context) to create graphics shaders and buffers that you can use
        // for e.g. egui::PaintCallback.
        let path_1 = std::env::args().nth(1).unwrap();
        let path_2 = std::env::args().nth(2).unwrap();
        let content_a = spiff::open_file(path_1).unwrap();
        let content_b = spiff::open_file(path_2).unwrap();
        let options = DiffOptions::default();
        let diff_view = DiffView::new(&content_a, &content_b, &options);

        Spiff {
            content_a,
            content_b,
            options,
            diff_view,
        }
    }
}

impl eframe::App for Spiff {
    fn update(&mut self, ctx: &EContext, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let new_options = self.options.show(ui);
            if new_options {
                self.diff_view = DiffView::new(&self.content_a, &self.content_b, &self.options);
            }

            match self.diff_view.as_mut() {
                Ok(diff_view) => {
                    diff_view.show(ui);
                }
                Err(e) => {
                    ui.centered_and_justified(|ui| {
                        ui.label(format!("Failed to render diff: {:?}", e));
                    });
                }
            }
        });
    }
}

struct DiffOptions {
    context_lines: usize,
    track_moves: bool,
    consider_whitespace: bool,
    line_numbers: bool,
}

impl Default for DiffOptions {
    fn default() -> DiffOptions {
        DiffOptions {
            context_lines: 5,
            consider_whitespace: true,
            track_moves: true,
            line_numbers: true,
        }
    }
}

impl DiffOptions {
    fn show(&mut self, ui: &mut Ui) -> bool {
        ui.horizontal(|ui| {
            let mut changed = ui
                .add(Slider::new(&mut self.context_lines, 0..=100).text("Context lines"))
                .changed();

            changed |= ui
                .checkbox(&mut self.consider_whitespace, "Whitespace")
                .changed();

            changed |= ui.checkbox(&mut self.track_moves, "Track Moves").changed();

            changed |= ui
                .checkbox(&mut self.line_numbers, "Line Numbers")
                .changed();

            changed
        })
        .inner
    }
}

struct DiffView {
    processed_diff: String,
    line_numbers: String,
    coloring: Vec<(std::ops::Range<usize>, Color32)>,
}

impl DiffView {
    fn new(a: &[u8], b: &[u8], options: &DiffOptions) -> Result<DiffView> {
        let lines_a = spiff::buf_to_lines(a).context("Failed to split file A by line")?;
        let lines_b = spiff::buf_to_lines(b).context("Failed to split file B by line")?;
        let trimmed_lines_a = lines_a.iter().map(|x| x.trim()).collect::<Vec<_>>();
        let trimmed_lines_b = lines_b.iter().map(|x| x.trim()).collect::<Vec<_>>();

        let diff = if options.consider_whitespace {
            libdiff::diff(&lines_a, &lines_b)
        } else {
            libdiff::diff(&trimmed_lines_a, &trimmed_lines_b)
        };

        let (diff, matches) = if options.track_moves {
            let MatchedDiff { diff, matches } =
                libdiff::match_insertions_removals(diff, &lines_a, &lines_b);
            (diff, matches)
        } else {
            (diff, [].into())
        };

        let (processed_diff, line_numbers, coloring) = DiffProcessor {
            lines_a: &lines_a,
            lines_b: &lines_b,
            options,
            diff: &diff,
            matches: &matches,
            processed_diff: String::new(),
            line_numbers: String::new(),
            coloring: Vec::new(),
        }
        .process();

        Ok(DiffView {
            processed_diff,
            line_numbers,
            coloring,
        })
    }

    fn show(&self, ui: &mut Ui) {
        const FONT_SIZE: f32 = 14.0;

        let mut layouter = |ui: &Ui, s: &str, _wrap_width: f32| {
            // FIXME: memoize

            let mut job = LayoutJob::default();
            for (range, color) in &self.coloring {
                job.append(
                    &s[range.clone()],
                    0.0,
                    TextFormat {
                        color: *color,
                        font_id: FontId::monospace(FONT_SIZE),
                        ..Default::default()
                    },
                );
            }

            ui.fonts().layout_job(job)
        };

        ScrollArea::both()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.add(Label::new(
                        RichText::new(&self.line_numbers)
                            .monospace()
                            .size(FONT_SIZE)
                            .color(ui.visuals().weak_text_color()),
                    ));

                    // HACK: Abusing the code editor to get free features like multi-line selection
                    // and copy paste (which IIRC is the best way to get what I want)
                    TextEdit::multiline(&mut self.processed_diff.as_str())
                        .code_editor()
                        .desired_width(f32::INFINITY)
                        .layouter(&mut layouter)
                        .show(ui);
                });
            });
    }
}

struct DiffProcessor<'a> {
    options: &'a DiffOptions,
    lines_a: &'a [&'a str],
    lines_b: &'a [&'a str],
    diff: &'a [libdiff::DiffAction],
    matches: &'a std::collections::HashMap<usize, usize>,
    processed_diff: String,
    line_numbers: String,
    coloring: Vec<(std::ops::Range<usize>, Color32)>,
}

impl DiffProcessor<'_> {
    fn process(mut self) -> (String, String, Vec<(std::ops::Range<usize>, Color32)>) {
        for (idx, action) in self.diff.iter().enumerate() {
            use DiffAction::*;
            match action {
                Traverse(traversal) => {
                    self.process_traversal(traversal);
                }
                Insert(insertion) => {
                    self.process_insertion(insertion, idx);
                }
                Remove(removal) => {
                    self.process_removal(removal, idx);
                }
            }
        }

        (self.processed_diff, self.line_numbers, self.coloring)
    }

    fn process_traversal(&mut self, traversal: &libdiff::Traversal) {
        let visuals = Visuals::default();

        let b_idx_offset = traversal.b_idx as i64 - traversal.a_idx as i64;
        let start_length = self.processed_diff.len();

        if traversal.length > self.options.context_lines * 2 {
            if !self.processed_diff.is_empty() {
                for (idx, line) in self
                    .lines_a
                    .iter()
                    .enumerate()
                    .skip(traversal.a_idx)
                    .take(self.options.context_lines)
                {
                    self.process_line_numbers(Some(idx), Some((idx as i64 + b_idx_offset) as usize));
                    writeln!(self.processed_diff, " {}", line).expect("Failed to write line");
                }
            }

            writeln!(self.processed_diff, " ...").expect("Failed to write line");
            self.process_line_numbers(None, None);

            for (idx, line) in self
                .lines_a
                .iter()
                .enumerate()
                .skip(traversal.a_idx + traversal.length - self.options.context_lines)
                .take(self.options.context_lines)
            {
                self.process_line_numbers(Some(idx), Some((idx as i64 + b_idx_offset) as usize));
                writeln!(self.processed_diff, " {}", line).expect("Failed to write line");
            }
        } else {
            for (idx, line) in self
                .lines_a
                .iter()
                .enumerate()
                .skip(traversal.a_idx)
                .take(traversal.length)
            {
                self.process_line_numbers(Some(idx), Some((idx as i64 + b_idx_offset) as usize));
                writeln!(self.processed_diff, " {}", line).expect("Failed to write line");
            }
        }
        self.coloring
            .push((start_length..self.processed_diff.len(), visuals.text_color()));
    }

    fn process_insertion(&mut self, insertion: &libdiff::Insertion, idx: usize) {
        let color = if self.matches.contains_key(&idx) {
            Color32::LIGHT_BLUE
        } else {
            Color32::LIGHT_GREEN
        };

        for (idx, line) in self
            .lines_b
            .iter()
            .enumerate()
            .skip(insertion.b_idx)
            .take(insertion.length)
        {
            self.process_line_numbers(None, Some(idx));
            let start_length = self.processed_diff.len();
            writeln!(self.processed_diff, "+{}", line).expect("Failed to write line");
            self.coloring
                .push((start_length..self.processed_diff.len(), color));
        }
    }

    fn process_removal(&mut self, removal: &libdiff::Removal, idx: usize) {
        let start_length = self.processed_diff.len();

        let color = if self.matches.contains_key(&idx) {
            Color32::KHAKI
        } else {
            Color32::LIGHT_RED
        };

        for (a_idx, line) in self
            .lines_a
            .iter()
            .enumerate()
            .skip(removal.a_idx)
            .take(removal.length)
        {
            self.process_line_numbers(Some(a_idx), None);
            writeln!(self.processed_diff, "-{}", line).expect("Failed to write line");
        }

        self.coloring
            .push((start_length..self.processed_diff.len(), color));
    }

    fn process_line_numbers(&mut self, line_a: Option<usize>, line_b: Option<usize>) {
        if !self.options.line_numbers {
            return;
        }

        let num_to_string = |line: Option<usize>| {
            line.map(|x| format!("{:4}", x))
                .unwrap_or_else(|| "    ".to_string())
        };

        let line_a = num_to_string(line_a);
        let line_b = num_to_string(line_b);

        writeln!(self.line_numbers, "{}|{}", line_a, line_b).expect("Failed to write line numbers");
    }
}
