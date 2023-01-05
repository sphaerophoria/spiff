use anyhow::{anyhow, Context, Result};
use eframe::{
    egui::{
        self, text::LayoutJob, Align, CollapsingHeader, Color32, ComboBox, Context as EContext,
        DragValue, Label, Layout, RichText, ScrollArea, TextEdit, TextFormat, Ui, Visuals,
    },
    epaint::FontId,
};

use spiff::{
    Contents, DiffCollectionProcessor, DiffOptions, ProcessedDiffData, SegmentPurpose, ViewMode,
};

use std::{
    path::{Path, PathBuf},
    sync::mpsc::{self, Receiver, Sender},
    thread,
    time::{Duration, Instant},
};

fn main() -> Result<()> {
    let root_a = Path::new(&std::env::args().nth(1).context("Path 1 not provided")?).to_path_buf();
    let root_b = Path::new(&std::env::args().nth(2).context("Path 2 not provided")?).to_path_buf();

    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "Spiff",
        native_options,
        Box::new(|cc| Box::new(Spiff::new(cc, root_a, root_b))),
    );
    Ok(())
}

enum ThreadRequest {
    ProcessDiff { options: DiffOptions },
}

enum ThreadResponse {
    DiffProcessed {
        options: DiffOptions,
        time: Duration,
        diffs: Vec<ProcessedDiffData>,
    },
}

fn processor_thread<P: AsRef<Path>>(
    ctx: EContext,
    root_a: P,
    root_b: P,
    rx: Receiver<ThreadRequest>,
    tx: Sender<Result<ThreadResponse>>,
) {
    let root_a = root_a.as_ref();
    let root_b = root_b.as_ref();

    let fail_forever = |error: anyhow::Error| {
        while let Ok(request) = rx.recv() {
            match request {
                ThreadRequest::ProcessDiff { .. } => {
                    // Stringize the error here so that we can send it over and over
                    if tx.send(Err(anyhow!("{:?}", error))).is_err() {
                        break;
                    }
                }
            }
        }
    };

    let Contents {
        content_a,
        content_b,
        labels,
    } = match spiff::contents_from_roots(root_a, root_b) {
        Ok(v) => v,
        Err(e) => {
            fail_forever(e);
            return;
        }
    };

    let mut request_processor = match DiffCollectionProcessor::new(&content_a, &content_b, &labels)
    {
        Ok(v) => v,
        Err(e) => {
            fail_forever(e);
            return;
        }
    };

    while let Ok(mut request) = rx.recv() {
        while let Ok(latest_request) = rx.try_recv() {
            request = latest_request;
        }

        match request {
            ThreadRequest::ProcessDiff { options } => {
                let start = Instant::now();
                let diffs = request_processor.process_new_options(&options);
                let end = Instant::now();
                if tx
                    .send(Ok(ThreadResponse::DiffProcessed {
                        diffs,
                        time: end - start,
                        options,
                    }))
                    .is_err()
                {
                    break;
                }

                ctx.request_repaint();
            }
        }
    }
}

enum DiffStatus {
    Processing,
    Success { num_diffs: usize, time: Duration },
    Failure,
}

impl DiffStatus {
    fn show(&self, ui: &mut Ui) {
        let layout = Layout::right_to_left(Align::Center);
        ui.with_layout(layout, |ui| {
            let text = match self {
                DiffStatus::Processing => "Processing...".to_string(),
                DiffStatus::Success { num_diffs, time } => {
                    format!("Processed {} diffs in {}s", num_diffs, time.as_secs_f32())
                }
                DiffStatus::Failure => "Failed to process diff".to_string(),
            };

            ui.label(text);
        });
    }
}

struct Spiff {
    options: DiffOptions,
    status: DiffStatus,
    diff_view: Result<DiffView>,
    thread_tx: Sender<ThreadRequest>,
    thread_rx: Receiver<Result<ThreadResponse>>,
}

impl Spiff {
    fn new(cc: &eframe::CreationContext<'_>, root_a: PathBuf, root_b: PathBuf) -> Self {
        // Customize egui here with cc.egui_ctx.set_fonts and cc.egui_ctx.set_visuals.
        // Restore app state using cc.storage (requires the "persistence" feature).
        // Use the cc.gl (a glow::Context) to create graphics shaders and buffers that you can use
        // for e.g. egui::PaintCallback.

        let (request_tx, request_rx) = mpsc::channel();
        let (response_tx, response_rx) = mpsc::channel();

        let ctx = cc.egui_ctx.clone();
        thread::spawn(move || processor_thread(ctx, root_a, root_b, request_rx, response_tx));

        let options = DiffOptions::default();
        request_tx
            .send(ThreadRequest::ProcessDiff {
                options: options.clone(),
            })
            .unwrap();

        Spiff {
            status: DiffStatus::Processing,
            diff_view: Err(anyhow!("Waiting for initial evaluation")),
            options,
            thread_tx: request_tx,
            thread_rx: response_rx,
        }
    }
}

impl eframe::App for Spiff {
    fn update(&mut self, ctx: &EContext, _frame: &mut eframe::Frame) {
        let mut response = None;
        while let Ok(last_response) = self.thread_rx.try_recv() {
            response = Some(last_response);
        }

        if let Some(response) = response {
            match response {
                Ok(ThreadResponse::DiffProcessed {
                    options,
                    time,
                    diffs,
                }) => {
                    if options == self.options {
                        self.status = DiffStatus::Success {
                            time,
                            num_diffs: diffs.len(),
                        };
                    }
                    self.diff_view = Ok(DiffView::new(diffs));
                }
                Err(e) => {
                    self.status = DiffStatus::Failure;
                    self.diff_view = Err(e);
                }
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            let new_options = show_options(&mut self.options, ui);
            if new_options {
                self.thread_tx
                    .send(ThreadRequest::ProcessDiff {
                        options: self.options.clone(),
                    })
                    .unwrap();

                self.status = DiffStatus::Processing;
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
        egui::TopBottomPanel::bottom("Status").show(ctx, |ui| {
            self.status.show(ui);
        });
    }
}

fn show_options(options: &mut DiffOptions, ui: &mut Ui) -> bool {
    ui.horizontal(|ui| {
        let mut changed = ui.add(DragValue::new(&mut options.context_lines)).changed();

        ui.label("Context lines");

        changed |= ui
            .checkbox(&mut options.consider_whitespace, "Whitespace")
            .changed();

        changed |= ui
            .checkbox(&mut options.track_moves, "Track Moves")
            .changed();

        changed |= ui
            .checkbox(&mut options.line_numbers, "Line Numbers")
            .changed();

        ComboBox::from_label("Diff mode")
            .selected_text(options.view_mode.to_string())
            .show_ui(ui, |ui| {
                let mut add_value = |v| {
                    changed |= ui
                        .selectable_value(&mut options.view_mode, v, v.to_string())
                        .changed();
                };
                add_value(ViewMode::AOnly);
                add_value(ViewMode::BOnly);
                add_value(ViewMode::Unified);
            });

        changed
    })
    .inner
}

fn purpose_to_format(purpose: &SegmentPurpose, visuals: &Visuals, font_size: f32) -> TextFormat {
    use SegmentPurpose::*;
    let mut format = TextFormat {
        font_id: FontId::monospace(font_size),
        ..Default::default()
    };

    match purpose {
        Context => {
            format.color = visuals.text_color();
        }
        Addition => {
            format.color = Color32::LIGHT_GREEN;
        }
        Removal => {
            format.color = Color32::LIGHT_RED;
        }
        MoveFrom => {
            format.color = Color32::KHAKI;
        }
        MoveTo => {
            format.color = Color32::LIGHT_BLUE;
        }
    }

    format
}

struct DiffView {
    diffs: Vec<ProcessedDiffData>,
}

impl DiffView {
    fn new(diffs: Vec<ProcessedDiffData>) -> DiffView {
        DiffView { diffs }
    }

    fn show(&self, ui: &mut Ui) {
        const FONT_SIZE: f32 = 14.0;

        ScrollArea::both()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for (idx, diff) in self.diffs.iter().enumerate() {
                    let mut layouter = |ui: &Ui, s: &str, _wrap_width: f32| {
                        // FIXME: memoize

                        let mut job = LayoutJob::default();
                        for (range, purpose) in &diff.coloring {
                            job.append(
                                &s[range.clone()],
                                0.0,
                                purpose_to_format(purpose, ui.visuals(), FONT_SIZE),
                            );
                        }

                        ui.fonts().layout_job(job)
                    };

                    let mut header = CollapsingHeader::new(&diff.label);
                    if idx == 0 {
                        header = header.default_open(true);
                    }

                    header.show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.add(Label::new(
                                RichText::new(&diff.line_numbers)
                                    .monospace()
                                    .size(FONT_SIZE)
                                    .color(ui.visuals().weak_text_color()),
                            ));

                            // HACK: Abusing the code editor to get free features like multi-line selection
                            // and copy paste (which IIRC is the best way to get what I want)
                            TextEdit::multiline(&mut diff.processed_diff.as_str())
                                .code_editor()
                                .desired_width(f32::INFINITY)
                                .layouter(&mut layouter)
                                .show(ui);
                        });
                    });
                }
            });
    }
}
