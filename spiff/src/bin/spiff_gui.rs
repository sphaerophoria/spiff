use anyhow::{anyhow, Context, Result};
use eframe::{
    egui::{
        self, text::LayoutJob, CollapsingHeader, Color32, Context as EContext, Label, RichText,
        ScrollArea, Slider, TextEdit, TextFormat, Ui, Visuals,
    },
    epaint::FontId,
};
use libdiff::DiffAction;
use memmap2::Mmap;

use std::{
    collections::BTreeSet,
    fmt::Write as FmtWrite,
    fs,
    path::{Path, PathBuf},
};

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
    content_a: Vec<Option<Mmap>>,
    content_b: Vec<Option<Mmap>>,
    labels: Vec<String>,
    options: DiffOptions,
    diff_view: Result<DiffView>,
}

impl Spiff {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        // Customize egui here with cc.egui_ctx.set_fonts and cc.egui_ctx.set_visuals.
        // Restore app state using cc.storage (requires the "persistence" feature).
        // Use the cc.gl (a glow::Context) to create graphics shaders and buffers that you can use
        // for e.g. egui::PaintCallback.
        let root_a = Path::new(&std::env::args().nth(1).unwrap()).to_path_buf();
        let root_b = Path::new(&std::env::args().nth(2).unwrap()).to_path_buf();

        let Contents {
            content_a,
            content_b,
            labels,
        } = contents_from_roots(root_a, root_b).unwrap();

        let options = DiffOptions::default();
        let diff_view = DiffView::new(&content_a, &content_b, &labels, &options);

        Spiff {
            content_a,
            content_b,
            labels,
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
                self.diff_view = DiffView::new(
                    &self.content_a,
                    &self.content_b,
                    &self.labels,
                    &self.options,
                );
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

struct DiffViewFileData {
    label: String,
    processed_diff: String,
    line_numbers: String,
    coloring: Vec<(std::ops::Range<usize>, Color32)>,
}

struct DiffView {
    diffs: Vec<DiffViewFileData>,
}

impl DiffView {
    // Nasty generics to allow Mmap -> &[u8] coercion
    fn new<C>(
        a_bufs: &[Option<C>],
        b_bufs: &[Option<C>],
        labels: &[String],
        options: &DiffOptions,
    ) -> Result<DiffView>
    where
        C: AsRef<[u8]>,
    {
        assert_eq!(a_bufs.len(), labels.len());
        assert_eq!(a_bufs.len(), b_bufs.len());

        let mut diff_sequences = Vec::new();
        let lines_a = bufs_to_lines(a_bufs)?;
        let lines_b = bufs_to_lines(b_bufs)?;

        let trimmed_lines_a = trim_lines(&lines_a);
        let trimmed_lines_b = trim_lines(&lines_b);

        for i in 0..a_bufs.len() {
            let diff = if options.consider_whitespace {
                libdiff::diff(&lines_a[i], &lines_b[i])
            } else {
                libdiff::diff(&trimmed_lines_a[i], &trimmed_lines_b[i])
            };

            diff_sequences.push(diff);
        }

        let (diff_sequences, matches) = if options.track_moves {
            let matched_diffs = if options.consider_whitespace {
                libdiff::match_insertions_removals(diff_sequences, &lines_a, &lines_b)
            } else {
                libdiff::match_insertions_removals(
                    diff_sequences,
                    &trimmed_lines_a,
                    &trimmed_lines_b,
                )
            };
            (matched_diffs.diffs, matched_diffs.matches)
        } else {
            (diff_sequences, [].into())
        };

        let mut diffs = Vec::new();

        for i in 0..a_bufs.len() {
            let a = a_bufs[i].as_ref().map_or([].as_ref(), |x| x.as_ref());
            let b = b_bufs[i].as_ref().map_or([].as_ref(), |x| x.as_ref());
            let diff = &diff_sequences[i];
            let label = &labels[i];

            let lines_a = spiff::buf_to_lines(a).context("Failed to split file A by line")?;
            let lines_b = spiff::buf_to_lines(b).context("Failed to split file B by line")?;
            let (processed_diff, line_numbers, coloring) = DiffProcessor {
                lines_a: &lines_a,
                lines_b: &lines_b,
                options,
                diff,
                diff_idx: i,
                matches: &matches,
                processed_diff: String::new(),
                line_numbers: String::new(),
                coloring: Vec::new(),
            }
            .process();

            diffs.push(DiffViewFileData {
                label: label.to_string(),
                processed_diff,
                line_numbers,
                coloring,
            });
        }

        Ok(DiffView { diffs })
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
                        for (range, color) in &diff.coloring {
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

struct DiffProcessor<'a> {
    options: &'a DiffOptions,
    lines_a: &'a [&'a str],
    lines_b: &'a [&'a str],
    diff: &'a [libdiff::DiffAction],
    diff_idx: usize,
    matches: &'a std::collections::HashMap<(usize, usize), (usize, usize)>,
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
                    self.process_line_numbers(
                        Some(idx),
                        Some((idx as i64 + b_idx_offset) as usize),
                    );
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
        self.coloring.push((
            start_length..self.processed_diff.len(),
            visuals.text_color(),
        ));
    }

    fn process_insertion(&mut self, insertion: &libdiff::Insertion, idx: usize) {
        let color = if self.matches.contains_key(&(self.diff_idx, idx)) {
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

        let color = if self.matches.contains_key(&(self.diff_idx, idx)) {
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

fn get_all_files<P: AsRef<Path>>(root: P, path: P) -> Result<Vec<PathBuf>> {
    let path = path.as_ref();
    let root = root.as_ref();

    if !path.exists() {
        return Err(anyhow!("Path {} does not exist", path.display()));
    }

    if !path.is_dir() {
        return Ok(vec![path
            .strip_prefix(root)
            .with_context(|| format!("Failed to strip {} from {}", root.display(), path.display()))?
            .to_path_buf()]);
    }

    let mut paths = Vec::new();
    for entry in
        fs::read_dir(path).with_context(|| format!("Failed to read dir {}", path.display()))?
    {
        let entry = entry.with_context(|| format!("Failed to read entry in {}", path.display()))?;
        let visited_path = entry.path();
        paths.extend(get_all_files(root, &visited_path)?);
    }
    Ok(paths)
}

struct Contents {
    content_a: Vec<Option<Mmap>>,
    content_b: Vec<Option<Mmap>>,
    labels: Vec<String>,
}

fn contents_from_roots<P: AsRef<Path>>(root_a: P, root_b: P) -> Result<Contents> {
    let root_a = root_a.as_ref();
    let root_b = root_b.as_ref();

    let paths_1 = get_all_files(&root_a, &root_a).context("Failed to get paths from a")?;
    let paths_2 = get_all_files(&root_b, &root_b).context("Failed to get paths from b")?;

    let (content_a, content_b, labels) = if !root_a.is_dir() && !root_b.is_dir() {
        let a = spiff::open_file(root_a)
            .with_context(|| format!("Failed to open {}", root_a.display()))?;
        let b = spiff::open_file(root_b)
            .with_context(|| format!("Failed to open {}", root_b.display()))?;

        let label = format!("{} -> {}", root_a.display(), root_b.display());

        (vec![Some(a)], vec![Some(b)], vec![label])
    } else if paths_1.len() == 1 && paths_2.len() == 1 {
        let full_path_a = root_a.join(&paths_1[0]);
        let full_path_b = root_b.join(&paths_2[0]);
        let a = spiff::open_file(&full_path_a)
            .with_context(|| format!("Failed to open {}", full_path_a.display()))?;
        let b = spiff::open_file(&full_path_b)
            .with_context(|| format!("Failed to open {}", full_path_b.display()))?;

        let label = format!("{} -> {}", root_a.display(), root_b.display());

        (vec![Some(a)], vec![Some(b)], vec![label])
    } else {
        // BTreeSet gives us free dedup and sorting
        let mut all_paths = paths_1.into_iter().collect::<BTreeSet<_>>();
        all_paths.extend(paths_2.into_iter());
        let paths = all_paths.into_iter().collect::<Vec<_>>();

        let (content_a, content_b): (Vec<_>, Vec<_>) = paths
            .iter()
            .map(|p| {
                let a = spiff::open_file(root_a.join(p)).map(Some).unwrap_or(None);
                let b = spiff::open_file(root_b.join(p)).map(Some).unwrap_or(None);

                (a, b)
            })
            .unzip();

        let labels = paths
            .into_iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect();

        (content_a, content_b, labels)
    };

    Ok(Contents {
        content_a,
        content_b,
        labels,
    })
}

fn unwrap_or_empty_buf<C>(option: &Option<C>) -> &[u8]
where
    C: AsRef<[u8]>,
{
    option.as_ref().map_or([].as_ref(), AsRef::as_ref)
}

fn bufs_to_lines<C>(bufs: &[Option<C>]) -> Result<Vec<Vec<&str>>>
where
    C: AsRef<[u8]>,
{
    bufs.iter()
        .map(|x| spiff::buf_to_lines(unwrap_or_empty_buf(x)))
        .collect::<Result<Vec<_>>>()
}

fn trim_lines<'a>(lines: &[Vec<&'a str>]) -> Vec<Vec<&'a str>> {
    lines
        .iter()
        .map(|x| x.iter().map(|x| x.trim()).collect::<Vec<_>>())
        .collect::<Vec<_>>()
}
