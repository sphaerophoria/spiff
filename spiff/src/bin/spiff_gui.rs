use anyhow::{anyhow, Context, Result};
use eframe::egui::{self, Align, Context as EContext, Layout, ScrollArea, Ui};

use spiff::{
    widget::{
        self as spiff_widget, search_bar_wrapped, DiffView, HeaderAction, SearchBar,
        SearchBarAction,
    },
    Contents, DiffCollectionProcessor, DiffOptions, ProcessedDiffCollection,
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
    )
    .expect("failed to create gui");
    Ok(())
}

enum ThreadRequest {
    ProcessDiff { options: DiffOptions },
    SetSearchQuery(String),
}

enum ThreadResponse {
    DiffProcessed {
        reason: String,
        options: DiffOptions,
        time: Duration,
        processed_diffs: ProcessedDiffCollection,
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
        while rx.recv().is_ok() {
            // Stringize the error here so that we can send it over and over
            if tx.send(Err(anyhow!("{:?}", error))).is_err() {
                break;
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
                request_processor.process_new_options(&options);
                let processed_diffs = request_processor.generate();
                let end = Instant::now();
                if tx
                    .send(Ok(ThreadResponse::DiffProcessed {
                        reason: "options changed".to_string(),
                        processed_diffs,
                        time: end - start,
                        options,
                    }))
                    .is_err()
                {
                    break;
                }

                ctx.request_repaint();
            }
            ThreadRequest::SetSearchQuery(query) => {
                let start = Instant::now();
                request_processor.set_search_query(query);
                let processed_diffs = request_processor.generate();
                let end = Instant::now();

                if tx
                    .send(Ok(ThreadResponse::DiffProcessed {
                        reason: "search string changed".to_string(),
                        processed_diffs,
                        time: end - start,
                        options: request_processor.options().clone(),
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
    Success {
        reason: String,
        num_diffs: usize,
        time: Duration,
    },
    Failure,
}

impl DiffStatus {
    fn show(&self, ui: &mut Ui) {
        let layout = Layout::right_to_left(Align::Center);
        ui.with_layout(layout, |ui| {
            let text = match self {
                DiffStatus::Processing => "Processing...".to_string(),
                DiffStatus::Success {
                    reason,
                    num_diffs,
                    time,
                } => {
                    format!(
                        "Processed {} diffs from {} in {}s",
                        num_diffs,
                        reason,
                        time.as_secs_f32()
                    )
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
    search_bar: SearchBar,
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
            search_bar: SearchBar::default(),
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
                    reason,
                    options,
                    time,
                    processed_diffs,
                }) => {
                    if options == self.options {
                        self.status = DiffStatus::Success {
                            reason,
                            time,
                            num_diffs: processed_diffs.processed_diffs.len(),
                        };
                    }

                    match &mut self.diff_view {
                        Ok(v) => {
                            v.update_data(processed_diffs.processed_diffs);
                        }
                        Err(_) => {
                            self.diff_view = Ok(DiffView::new(processed_diffs.processed_diffs));
                        }
                    }

                    self.search_bar.update_data(processed_diffs.search_results);
                }
                Err(e) => {
                    self.status = DiffStatus::Failure;
                    self.diff_view = Err(e);
                }
            }
        }

        egui::TopBottomPanel::bottom("Status").show(ctx, |ui| {
            self.status.show(ui);
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let header_action = spiff_widget::show_header(&mut self.options, ui);
            if let HeaderAction::RequestProcess = &header_action {
                self.thread_tx
                    .send(ThreadRequest::ProcessDiff {
                        options: self.options.clone(),
                    })
                    .unwrap();

                self.status = DiffStatus::Processing;
            }

            let force_collapse_state = match header_action {
                HeaderAction::ExpandAll => Some(true),
                HeaderAction::CollapseAll => Some(false),
                _ => None,
            };

            let response = search_bar_wrapped(&mut self.search_bar, ui, |ui, jump_idx| {
                ScrollArea::both()
                    .auto_shrink([false, false])
                    .show(ui, |ui| match self.diff_view.as_mut() {
                        Ok(diff_view) => diff_view.show(ui, jump_idx, force_collapse_state),
                        Err(e) => {
                            ui.centered_and_justified(|ui| {
                                ui.label(format!("Failed to render diff: {:?}", e));
                            });
                        }
                    });
            });

            if let SearchBarAction::UpdateSearch(query) = response.action {
                self.thread_tx
                    .send(ThreadRequest::SetSearchQuery(query))
                    .unwrap();
            }
        });
    }
}
