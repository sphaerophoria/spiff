use eframe::{
    egui::{
        self,
        text::{CCursor, LayoutJob},
        Align, CollapsingHeader, Color32, ComboBox, DragValue, Label, Layout, RichText, TextEdit,
        TextFormat, TextStyle, Ui, Visuals,
    },
    epaint::FontId,
};

use crate::{DiffOptions, ProcessedDiffData, ProcessedDiffIdx, SegmentPurpose, ViewMode};

pub enum HeaderAction {
    RequestProcess,
    ExpandAll,
    CollapseAll,
    None,
}

pub fn show_header(options: &mut DiffOptions, ui: &mut Ui) -> HeaderAction {
    ui.horizontal(|ui| {
        let mut action = HeaderAction::None;

        if ui.button("Expand All").clicked() {
            action = HeaderAction::ExpandAll;
        }

        if ui.button("Collapse All").clicked() {
            action = HeaderAction::CollapseAll;
        }

        if ui.add(DragValue::new(&mut options.context_lines)).changed() {
            action = HeaderAction::RequestProcess;
        }

        ui.label("Context lines");

        if ui
            .checkbox(&mut options.consider_whitespace, "Whitespace")
            .changed()
        {
            action = HeaderAction::RequestProcess;
        }

        if ui
            .checkbox(&mut options.track_moves, "Track Moves")
            .changed()
        {
            action = HeaderAction::RequestProcess;
        }

        if ui
            .checkbox(&mut options.line_numbers, "Line Numbers")
            .changed()
        {
            action = HeaderAction::RequestProcess;
        }

        ComboBox::from_label("Diff mode")
            .selected_text(options.view_mode.to_string())
            .show_ui(ui, |ui| {
                let mut add_value = |v| {
                    if ui
                        .selectable_value(&mut options.view_mode, v, v.to_string())
                        .changed()
                    {
                        action = HeaderAction::RequestProcess;
                    }
                };

                add_value(ViewMode::AOnly);
                add_value(ViewMode::BOnly);
                add_value(ViewMode::Unified);
            });

        if ui
            .add(egui::DragValue::new(&mut options.max_memory_mb))
            .changed()
        {
            action = HeaderAction::RequestProcess;
        }
        ui.label("Max mem MB");

        action
    })
    .inner
}

pub struct DiffView {
    processed_diffs: Vec<ProcessedDiffData>,
    highlight_idx: Option<(usize, usize)>,
}

impl DiffView {
    pub fn new(diffs: Vec<ProcessedDiffData>) -> DiffView {
        let mut view = DiffView {
            processed_diffs: Vec::new(),
            highlight_idx: None,
        };

        view.update_data(diffs);

        view
    }

    pub fn update_data(&mut self, diffs: Vec<ProcessedDiffData>) {
        self.processed_diffs = diffs;
    }

    pub fn show(
        &mut self,
        ui: &mut Ui,
        jump_idx: Option<(usize, usize)>,
        force_collapse_state: Option<bool>,
    ) {
        if let Some(idx) = &jump_idx {
            self.highlight_idx = Some(*idx);
        }

        for idx in 0..self.processed_diffs.len() {
            self.show_diff(idx, force_collapse_state, jump_idx, ui);
        }
    }

    fn show_diff(
        &mut self,
        diff_idx: usize,
        force_collapse_state: Option<bool>,
        jump_idx: Option<(usize, usize)>,
        ui: &mut Ui,
    ) {
        const FONT_SIZE: f32 = 14.0;

        let diff = &self.processed_diffs[diff_idx];

        let mut color_generator = TextColorGenerator {
            last_movefrom: 0,
            last_moveto: 0,
            visuals: ui.visuals().clone(),
        };

        let mut layouter = |ui: &Ui, s: &str, _wrap_width: f32| {
            // FIXME: memoize
            let mut job = LayoutJob::default();
            for (range, info) in &diff.display_info {
                let select_color = if self.highlight_idx.map(|x| x.0) == Some(diff_idx)
                    && self.highlight_idx.map(|x| x.1) == Some(range.start)
                {
                    Color32::from_rgb(0, 0xcb, 0xff)
                } else {
                    ui.visuals().selection.bg_fill
                };

                job.append(
                    &s[range.clone()],
                    0.0,
                    info_to_format(info, &mut color_generator, FONT_SIZE, select_color),
                );
            }

            ui.fonts(|f| f.layout_job(job))
        };

        let mut label_job = LayoutJob::default();
        let label_fontid = TextStyle::Monospace.resolve(ui.style());
        let mut label_append = |s: &str, color| {
            label_job.append(s, 0.0, TextFormat::simple(label_fontid.clone(), color));
        };

        match diff.change_overview {
            crate::ChangeOverview::Known {
                num_inserted_lines,
                num_removed_lines,
                num_moved_insertions,
                num_moved_removals,
            } => {
                label_append(&diff.label, ui.visuals().text_color());
                label_append("\n", ui.visuals().text_color());
                label_append(
                    &format!("{} ", num_inserted_lines + num_removed_lines),
                    ui.visuals().text_color(),
                );

                label_append(
                    std::str::from_utf8(&vec![
                        b'+';
                        ((num_inserted_lines - num_moved_insertions) + 1) / 2
                    ])
                    .unwrap(),
                    Color32::LIGHT_GREEN,
                );
                label_append(
                    std::str::from_utf8(&vec![b'+'; num_moved_insertions / 2]).unwrap(),
                    Color32::LIGHT_BLUE,
                );

                label_append(
                    std::str::from_utf8(&vec![
                        b'-';
                        (num_removed_lines - num_moved_removals + 1) / 2
                    ])
                    .unwrap(),
                    Color32::LIGHT_RED,
                );

                label_append(
                    std::str::from_utf8(&vec![b'-'; num_moved_removals / 2]).unwrap(),
                    Color32::KHAKI,
                );
            }
            crate::ChangeOverview::Unknown => {
                label_append(&format!("{}\n?????", &diff.label), Color32::LIGHT_RED);
            }
        }

        let mut header = CollapsingHeader::new(label_job)
            .id_source(&diff.label)
            .open(force_collapse_state);

        if jump_idx.map(|x| x.0) == Some(diff_idx) {
            header = header.open(Some(true));
        }

        header.show(ui, |ui| {
            ui.horizontal(|ui| {
                let label_response = ui.add(Label::new(
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

                let rect_min = response.response.rect.min;
                let galley = std::sync::Arc::clone(&response.galley);
                let pointer_pos = ui.input(|i| i.pointer.hover_pos());
                if let Some(pointer_pos) = pointer_pos {
                    let cursor_pos = galley
                        .cursor_from_pos(egui::vec2(0.0, pointer_pos.y - rect_min.y))
                        .ccursor
                        .index;

                    let mut hover_segment = None;
                    for segment in &diff.display_info {
                        if cursor_pos >= segment.0.start && cursor_pos < segment.0.end {
                            hover_segment = Some(&segment.1)
                        }
                    }

                    match hover_segment {
                        Some(SegmentPurpose::MoveTo(loc)) => {
                            label_response.on_hover_ui_at_pointer(|ui| {
                                ui.label(format!(
                                    "Moved from {}:{}",
                                    self.processed_diffs[loc.diff_idx].label, loc.line_idx
                                ));
                            });
                        }
                        Some(SegmentPurpose::MoveFrom(loc)) => {
                            label_response.on_hover_ui_at_pointer(|ui| {
                                ui.label(format!(
                                    "Moved to {}:{}",
                                    self.processed_diffs[loc.diff_idx].label, loc.line_idx
                                ));
                            });
                        }
                        _ => (),
                    }
                }

                if jump_idx.map(|x| x.0) == Some(diff_idx) {
                    let string_idx = jump_idx.as_ref().unwrap().1;

                    let cursor = response.galley.from_ccursor(CCursor::new(string_idx));
                    let mut pos = response.galley.pos_from_cursor(&cursor);
                    pos.min.x += response.galley_pos.x;
                    pos.min.y += response.galley_pos.y;
                    pos.max = pos.min;
                    ui.scroll_to_rect(pos, Some(Align::Center));
                }
            });
        });
    }
}

struct TextColorGenerator {
    last_movefrom: u8,
    last_moveto: u8,
    visuals: Visuals,
}

impl TextColorGenerator {
    fn fg_color_from(&mut self, purpose: &SegmentPurpose) -> Color32 {
        use SegmentPurpose::*;

        match purpose {
            Context | TrailingWhitespace | MatchSearch => self.visuals.text_color(),
            Addition => Color32::LIGHT_GREEN,
            Removal => Color32::LIGHT_RED,
            MoveFrom(_) => {
                self.last_movefrom = (self.last_movefrom + 1) % 2;
                [Color32::KHAKI, Color32::from_rgb(0xdd, 0x98, 0x63)][self.last_movefrom as usize]
            }
            MoveTo(_) => {
                self.last_moveto = (self.last_moveto + 1) % 2;
                [Color32::LIGHT_BLUE, Color32::from_rgb(0x9d, 0xa1, 0xea)]
                    [self.last_moveto as usize]
            }
            Failed => Color32::LIGHT_RED,
        }
    }

    fn bg_color_from(&mut self, purpose: &SegmentPurpose, select_color: Color32) -> Color32 {
        use SegmentPurpose::*;
        match purpose {
            TrailingWhitespace => Color32::LIGHT_RED,
            MatchSearch => select_color,
            _ => Color32::TRANSPARENT,
        }
    }
}

fn info_to_format(
    purpose: &SegmentPurpose,
    color_generator: &mut TextColorGenerator,
    font_size: f32,
    select_color: Color32,
) -> TextFormat {
    let mut format = TextFormat {
        font_id: FontId::monospace(font_size),
        ..Default::default()
    };

    format.color = color_generator.fg_color_from(purpose);
    format.background = color_generator.bg_color_from(purpose, select_color);

    format
}

#[derive(Eq, PartialEq)]
pub enum SearchBarAction {
    Jump,
    UpdateSearch(String),
    None,
}

pub struct SearchBar {
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
    pub fn new() -> SearchBar {
        Default::default()
    }

    pub fn update_data(&mut self, data: Vec<ProcessedDiffIdx>) {
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

    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    pub fn diff_idx(&self) -> usize {
        if self.current_search_idx < self.search_results.len() {
            self.search_results[self.current_search_idx].diff_idx
        } else {
            usize::MAX
        }
    }

    pub fn string_idx(&self) -> usize {
        if self.current_search_idx < self.search_results.len() {
            self.search_results[self.current_search_idx].string_idx
        } else {
            usize::MAX
        }
    }

    pub fn show(&mut self, search_bar_request_input: bool, ui: &mut Ui) -> SearchBarAction {
        if !self.visible {
            return SearchBarAction::None;
        }

        let mut action = SearchBarAction::None;

        let search_response = ui.text_edit_singleline(&mut self.search_query);

        if search_response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
            search_response.request_focus();
            if ui.input(|i| i.modifiers.shift) {
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

        if ui.input(|i| i.key_down(egui::Key::Escape)) {
            self.visible = false;
            self.search_query.clear();
            action = SearchBarAction::UpdateSearch(self.search_query.clone());
        }

        action
    }
}

pub struct SearchBarWrappedOutput<T> {
    pub inner: T,
    pub action: SearchBarAction,
}

pub fn search_bar_wrapped<T, F: FnOnce(&mut Ui, Option<(usize, usize)>) -> T>(
    search_bar: &mut SearchBar,
    ui: &mut Ui,
    layout_fn: F,
) -> SearchBarWrappedOutput<T> {
    // Some widgets (e.g. ScrollArea) will expand to fill all remaining space, layout bottom to
    // top so that we can add a search bar at the bottom. Invert the layout again to get a
    // normal layout for the children

    let outer_layout = *ui.layout();

    ui.with_layout(Layout::bottom_up(Align::LEFT), |ui| {
        let search_bar_request_input = if ui.input(|i| i.key_down(egui::Key::F) && i.modifiers.ctrl)
        {
            search_bar.set_visible(true);
            true
        } else {
            false
        };

        let action = search_bar.show(search_bar_request_input, ui);
        let jump_idx = match &action {
            SearchBarAction::Jump => Some((search_bar.diff_idx(), search_bar.string_idx())),
            _ => None,
        };

        let inner = ui
            .with_layout(outer_layout, |ui| layout_fn(ui, jump_idx))
            .inner;

        SearchBarWrappedOutput { inner, action }
    })
    .inner
}
