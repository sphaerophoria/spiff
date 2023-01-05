use anyhow::{anyhow, Context, Result};
use eframe::{
    egui::{
        self,
        text::{CCursor, LayoutJob},
        Align, CollapsingHeader, Color32, ComboBox, Context as EContext, DragValue, Label, Layout,
        RichText, ScrollArea, TextEdit, TextFormat, Ui, Visuals,
    },
    epaint::FontId,
};

use spiff::{
    Contents, DiffCollectionProcessor, DiffOptions, DisplayInfo, ProcessedDiffCollection,
    ProcessedDiffData, ProcessedDiffIdx, SegmentPurpose, ViewMode,
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
                            v.update_data(processed_diffs);
                        }
                        Err(_) => {
                            self.diff_view = Ok(DiffView::new(processed_diffs));
                        }
                    }
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
                    if let DiffViewAction::UpdateSearch(query) = diff_view.show(ui) {
                        self.thread_tx
                            .send(ThreadRequest::SetSearchQuery(query))
                            .unwrap();
                    }
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

fn info_to_format(
    info: &DisplayInfo,
    visuals: &Visuals,
    font_size: f32,
    select_color: Color32,
) -> TextFormat {
    use SegmentPurpose::*;
    let mut format = TextFormat {
        font_id: FontId::monospace(font_size),
        ..Default::default()
    };

    match info.purpose {
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

    if info.matches_search {
        format.background = select_color;
    }

    format
}

enum SearchBarAction {
    Jump,
    UpdateSearch(String),
    None,
}

struct SearchBar {
    visible: bool,
    search_query: String,
    search_results: Vec<ProcessedDiffIdx>,
    current_search_idx: usize,
}

impl Default for SearchBar {
    fn default() -> Self {
        SearchBar {
            visible: false,
            search_query: String::new(),
            search_results: Vec::new(),
            current_search_idx: usize::MAX,
        }
    }
}

impl SearchBar {
    fn new() -> SearchBar {
        Default::default()
    }

    fn update_data(&mut self, data: Vec<ProcessedDiffIdx>) {
        self.search_results = data;
        self.current_search_idx = 0;
    }

    fn increment_search_idx(&mut self) {
        self.current_search_idx = self.current_search_idx.wrapping_add(1);
        if self.current_search_idx >= self.search_results.len() {
            self.current_search_idx = usize::MAX;
        }
    }

    fn decrement_search_idx(&mut self) {
        self.current_search_idx = self.current_search_idx.wrapping_sub(1);

        if self.current_search_idx >= self.search_results.len() {
            self.current_search_idx = self.search_results.len() - 1;
        }
    }

    fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    fn diff_idx(&self) -> usize {
        if self.current_search_idx < self.search_results.len() {
            self.search_results[self.current_search_idx].diff_idx
        } else {
            usize::MAX
        }
    }

    fn string_idx(&self) -> usize {
        if self.current_search_idx < self.search_results.len() {
            self.search_results[self.current_search_idx].string_idx
        } else {
            usize::MAX
        }
    }

    fn show(&mut self, search_bar_request_input: bool, ui: &mut Ui) -> SearchBarAction {
        if !self.visible {
            return SearchBarAction::None;
        }

        let mut action = SearchBarAction::None;

        let search_response = ui.text_edit_singleline(&mut self.search_query);

        if search_response.lost_focus() && ui.input().key_pressed(egui::Key::Enter) {
            search_response.request_focus();
            if ui.input().modifiers.shift {
                self.decrement_search_idx();
            } else {
                self.increment_search_idx();
            }
            action = SearchBarAction::Jump;
        }

        if search_response.changed() {
            action = SearchBarAction::UpdateSearch(self.search_query.clone());
        }

        if search_bar_request_input {
            search_response.request_focus();
        }

        if ui.input().key_down(egui::Key::Escape) {
            self.visible = false;
            self.search_query.clear();
            action = SearchBarAction::UpdateSearch(self.search_query.clone());
        }

        action
    }
}

enum DiffViewAction {
    UpdateSearch(String),
    None,
}

struct DiffView {
    processed_diffs: Vec<ProcessedDiffData>,
    search_bar: SearchBar,
}

impl DiffView {
    fn new(diffs: ProcessedDiffCollection) -> DiffView {
        let mut view = DiffView {
            processed_diffs: Vec::new(),
            search_bar: SearchBar::new(),
        };

        view.update_data(diffs);

        view
    }

    fn update_data(&mut self, diffs: ProcessedDiffCollection) {
        self.processed_diffs = diffs.processed_diffs;
        self.search_bar.update_data(diffs.search_results);
    }

    fn show(&mut self, ui: &mut Ui) -> DiffViewAction {
        // ScrollArea will expand to fill all remaining space, layout bottom to top so that we can
        // add a search bar at the bottom. Invert the layout again to get a normal layout for the
        // Scroll area

        let outer_layout = *ui.layout();
        let mut action = DiffViewAction::None;
        ui.with_layout(Layout::bottom_up(Align::LEFT), |ui| {
            let mut jump_to_search = false;

            let search_bar_request_input =
                if ui.input().key_down(egui::Key::F) && ui.input().modifiers.ctrl {
                    self.search_bar.set_visible(true);
                    true
                } else {
                    false
                };

            match self.search_bar.show(search_bar_request_input, ui) {
                SearchBarAction::Jump => {
                    jump_to_search = true;
                }
                SearchBarAction::UpdateSearch(query) => {
                    action = DiffViewAction::UpdateSearch(query);
                }
                SearchBarAction::None => (),
            }

            ui.with_layout(outer_layout, |ui| {
                ScrollArea::both()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        for idx in 0..self.processed_diffs.len() {
                            self.show_diff(idx, jump_to_search, ui);
                        }
                    });
            });
        });

        action
    }

    fn show_diff(&mut self, diff_idx: usize, jump_to_search: bool, ui: &mut Ui) {
        const FONT_SIZE: f32 = 14.0;

        let diff = &self.processed_diffs[diff_idx];

        let mut layouter = |ui: &Ui, s: &str, _wrap_width: f32| {
            // FIXME: memoize
            let mut job = LayoutJob::default();
            for (range, info) in &diff.display_info {
                let select_color = if self.search_bar.diff_idx() == diff_idx
                    && self.search_bar.string_idx() == range.start
                {
                    Color32::from_rgb(0, 0xcb, 0xff)
                } else {
                    ui.visuals().selection.bg_fill
                };

                job.append(
                    &s[range.clone()],
                    0.0,
                    info_to_format(info, ui.visuals(), FONT_SIZE, select_color),
                );
            }

            ui.fonts().layout_job(job)
        };

        let mut header = CollapsingHeader::new(&diff.label);

        if jump_to_search && diff_idx == self.search_bar.diff_idx() {
            header = header.open(Some(true));
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
                let response = TextEdit::multiline(&mut diff.processed_diff.as_str())
                    .code_editor()
                    .desired_width(f32::INFINITY)
                    .layouter(&mut layouter)
                    .show(ui);

                if jump_to_search {
                    let string_idx = self.search_bar.string_idx();

                    let cursor = response.galley.from_ccursor(CCursor::new(string_idx));
                    let mut pos = response.galley.pos_from_cursor(&cursor);
                    pos.min.x += response.text_draw_pos.x;
                    pos.min.y += response.text_draw_pos.y;
                    pos.max = pos.min;
                    ui.scroll_to_rect(pos, Some(Align::Center));
                }
            });
        });
    }
}
