use anyhow::{anyhow, Context, Result};
use memmap2::Mmap;
use std::{
    collections::{BTreeSet, HashMap},
    fmt::{self, Write as FmtWrite},
    fs,
    fs::File,
    path::{Path, PathBuf},
};

use libdiff::DiffAction;
#[derive(Copy, Eq, PartialEq, Clone)]
pub enum ViewMode {
    AOnly,
    BOnly,
    Unified,
}

impl fmt::Display for ViewMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use ViewMode::*;

        let s = match self {
            AOnly => "A Only",
            BOnly => "B Only",
            Unified => "Unified",
        };

        f.write_str(s)
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct DiffOptions {
    pub context_lines: usize,
    pub track_moves: bool,
    pub consider_whitespace: bool,
    pub line_numbers: bool,
    pub view_mode: ViewMode,
}

impl Default for DiffOptions {
    fn default() -> DiffOptions {
        DiffOptions {
            context_lines: 5,
            consider_whitespace: true,
            track_moves: true,
            line_numbers: true,
            view_mode: ViewMode::Unified,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum SegmentPurpose {
    Context,
    Addition,
    Removal,
    MoveFrom,
    MoveTo,
}

pub struct DiffViewFileData {
    pub label: String,
    pub processed_diff: String,
    pub line_numbers: String,
    pub coloring: Vec<(std::ops::Range<usize>, SegmentPurpose)>,
}

pub struct DiffCollectionProcessor<'a> {
    options: Option<DiffOptions>,
    labels: &'a [String],
    lines_a: Vec<Vec<&'a str>>,
    lines_b: Vec<Vec<&'a str>>,
    trimmed_lines_a: Vec<Vec<&'a str>>,
    trimmed_lines_b: Vec<Vec<&'a str>>,
    diffs: Vec<Vec<DiffAction>>,
    matches: HashMap<(usize, usize), (usize, usize)>,
}

impl<'a> DiffCollectionProcessor<'a> {
    pub fn new<C>(
        content_a: &'a [Option<C>],
        content_b: &'a [Option<C>],
        labels: &'a [String],
    ) -> Result<DiffCollectionProcessor<'a>>
    where
        C: AsRef<[u8]> + 'a,
    {
        let lines_a = bufs_to_lines(content_a)?;
        let lines_b = bufs_to_lines(content_b)?;

        let trimmed_lines_a = trim_lines(&lines_a);
        let trimmed_lines_b = trim_lines(&lines_b);

        Ok(DiffCollectionProcessor {
            options: None,
            labels,
            lines_a,
            lines_b,
            trimmed_lines_a,
            trimmed_lines_b,
            diffs: Vec::new(),
            matches: HashMap::new(),
        })
    }

    fn recompute_diffs(&mut self, options: &DiffOptions) {
        self.diffs.clear();

        for i in 0..self.lines_a.len() {
            let begin = std::time::Instant::now();
            let diff = if options.consider_whitespace {
                libdiff::diff(&self.lines_a[i], &self.lines_b[i])
            } else {
                libdiff::diff(&self.trimmed_lines_a[i], &self.trimmed_lines_b[i])
            };
            let duration = (std::time::Instant::now() - begin).as_millis();
            if duration > 300 {
                println!("Diffing {} took {}", self.labels[i], duration);
            }
            self.diffs.push(diff);
        }

        (self.diffs, self.matches) = if options.track_moves {
            let mut diffs = Vec::new();
            std::mem::swap(&mut diffs, &mut self.diffs);

            let matched_diffs = if options.consider_whitespace {
                libdiff::match_insertions_removals(diffs, &self.lines_a, &self.lines_b)
            } else {
                libdiff::match_insertions_removals(
                    diffs,
                    &self.trimmed_lines_a,
                    &self.trimmed_lines_b,
                )
            };
            (matched_diffs.diffs, matched_diffs.matches)
        } else {
            let mut diffs = Vec::new();
            std::mem::swap(&mut diffs, &mut self.diffs);
            (diffs, [].into())
        };
    }

    pub fn process_new_options(&mut self, options: &DiffOptions) -> Vec<DiffViewFileData> {
        if self.options.is_none() {
            self.recompute_diffs(options);
            self.options = Some(options.clone());
        }

        let current_options = self.options.as_ref().unwrap();

        if options.consider_whitespace != current_options.consider_whitespace
            || options.track_moves != current_options.track_moves
        {
            self.recompute_diffs(options);
            self.options = Some(options.clone());
        }

        let mut diffs = Vec::new();

        for i in 0..self.lines_a.len() {
            let data = SingleDiffProcessor {
                lines_a: &self.lines_a[i],
                lines_b: &self.lines_b[i],
                label: &self.labels[i],
                options,
                diff: &self.diffs[i],
                diff_idx: i,
                matches: &self.matches,
                processed_diff: String::new(),
                line_numbers: String::new(),
                coloring: Vec::new(),
            }
            .process();

            diffs.push(data);
        }

        diffs
    }
}

/// Takes diff/matches/files/options and generates a DiffViewFileData for a single file pair
struct SingleDiffProcessor<'a> {
    options: &'a DiffOptions,
    lines_a: &'a [&'a str],
    lines_b: &'a [&'a str],
    label: &'a str,
    diff: &'a [libdiff::DiffAction],
    diff_idx: usize,
    matches: &'a std::collections::HashMap<(usize, usize), (usize, usize)>,
    processed_diff: String,
    line_numbers: String,
    coloring: Vec<(std::ops::Range<usize>, SegmentPurpose)>,
}

impl SingleDiffProcessor<'_> {
    fn process(mut self) -> DiffViewFileData {
        for (idx, action) in self.diff.iter().enumerate() {
            use DiffAction::*;
            match action {
                Traverse(traversal) => {
                    self.process_traversal(traversal, idx != 0, idx != self.diff.len() - 1);
                }
                Insert(insertion) => {
                    self.process_insertion(insertion, idx);
                }
                Remove(removal) => {
                    self.process_removal(removal, idx);
                }
            }
        }

        DiffViewFileData {
            processed_diff: self.processed_diff,
            line_numbers: self.line_numbers,
            label: self.label.to_string(),
            coloring: self.coloring,
        }
    }

    fn process_traversal(
        &mut self,
        traversal: &libdiff::Traversal,
        show_start: bool,
        show_end: bool,
    ) {
        let a_idx_offset = traversal.a_idx as i64 - traversal.b_idx as i64;
        let start_length = self.processed_diff.len();

        if traversal.length > self.options.context_lines * 2 {
            if show_start {
                for (idx, line) in self
                    .lines_b
                    .iter()
                    .enumerate()
                    .skip(traversal.b_idx)
                    .take(self.options.context_lines)
                {
                    self.process_line_numbers(
                        Some((idx as i64 + a_idx_offset) as usize),
                        Some(idx),
                    );
                    writeln!(self.processed_diff, " {}", line).expect("Failed to write line");
                }
            }

            writeln!(self.processed_diff, " ...").expect("Failed to write line");
            self.process_line_numbers(None, None);

            if show_end {
                for (idx, line) in self
                    .lines_b
                    .iter()
                    .enumerate()
                    .skip(traversal.b_idx + traversal.length - self.options.context_lines)
                    .take(self.options.context_lines)
                {
                    self.process_line_numbers(
                        Some((idx as i64 + a_idx_offset) as usize),
                        Some(idx),
                    );
                    writeln!(self.processed_diff, " {}", line).expect("Failed to write line");
                }
            }
        } else {
            for (idx, line) in self
                .lines_b
                .iter()
                .enumerate()
                .skip(traversal.b_idx)
                .take(traversal.length)
            {
                self.process_line_numbers(Some((idx as i64 + a_idx_offset) as usize), Some(idx));
                writeln!(self.processed_diff, " {}", line).expect("Failed to write line");
            }
        }
        self.coloring.push((
            start_length..self.processed_diff.len(),
            SegmentPurpose::Context,
        ));
    }

    fn process_insertion(&mut self, insertion: &libdiff::Insertion, idx: usize) {
        if self.options.view_mode == ViewMode::AOnly {
            return;
        }

        let purpose = if self.matches.contains_key(&(self.diff_idx, idx)) {
            SegmentPurpose::MoveTo
        } else {
            SegmentPurpose::Addition
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
                .push((start_length..self.processed_diff.len(), purpose));
        }
    }

    fn process_removal(&mut self, removal: &libdiff::Removal, idx: usize) {
        if self.options.view_mode == ViewMode::BOnly {
            return;
        }

        let start_length = self.processed_diff.len();

        let purpose = if self.matches.contains_key(&(self.diff_idx, idx)) {
            SegmentPurpose::MoveFrom
        } else {
            SegmentPurpose::Removal
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
            .push((start_length..self.processed_diff.len(), purpose));
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

fn unwrap_or_empty_buf<C>(option: &Option<C>) -> &[u8]
where
    C: AsRef<[u8]>,
{
    option.as_ref().map_or([].as_ref(), AsRef::as_ref)
}

fn open_file<P: AsRef<Path>>(path: P) -> Result<Mmap> {
    let file = File::open(path).context("Failed to open file")?;
    let mmap = unsafe { Mmap::map(&file) }.context("Failed to map file")?;
    Ok(mmap)
}

fn buf_to_lines(buf: &[u8]) -> Result<Vec<&str>> {
    let s = std::str::from_utf8(buf).context("Failed to parse string")?;

    Ok(s.lines().collect())
}

fn bufs_to_lines<C>(bufs: &[Option<C>]) -> Result<Vec<Vec<&str>>>
where
    C: AsRef<[u8]>,
{
    bufs.iter()
        .map(|x| buf_to_lines(unwrap_or_empty_buf(x)))
        .collect::<Result<Vec<_>>>()
}

fn trim_lines<'a>(lines: &[Vec<&'a str>]) -> Vec<Vec<&'a str>> {
    lines
        .iter()
        .map(|x| x.iter().map(|x| x.trim()).collect::<Vec<_>>())
        .collect::<Vec<_>>()
}

pub struct Contents {
    pub content_a: Vec<Option<Mmap>>,
    pub content_b: Vec<Option<Mmap>>,
    pub labels: Vec<String>,
}

pub fn contents_from_roots<P: AsRef<Path>>(root_a: P, root_b: P) -> Result<Contents> {
    let root_a = root_a.as_ref();
    let root_b = root_b.as_ref();

    let paths_1 = get_all_files(&root_a, &root_a).context("Failed to get paths from a")?;
    let paths_2 = get_all_files(&root_b, &root_b).context("Failed to get paths from b")?;

    let (content_a, content_b, labels) = if !root_a.is_dir() && !root_b.is_dir() {
        let a =
            open_file(root_a).with_context(|| format!("Failed to open {}", root_a.display()))?;
        let b =
            open_file(root_b).with_context(|| format!("Failed to open {}", root_b.display()))?;

        let label = format!("{} -> {}", root_a.display(), root_b.display());

        (vec![Some(a)], vec![Some(b)], vec![label])
    } else if paths_1.len() == 1 && paths_2.len() == 1 {
        let full_path_a = root_a.join(&paths_1[0]);
        let full_path_b = root_b.join(&paths_2[0]);
        let a = open_file(&full_path_a)
            .with_context(|| format!("Failed to open {}", full_path_a.display()))?;
        let b = open_file(&full_path_b)
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
                let a = open_file(root_a.join(p)).map(Some).unwrap_or(None);
                let b = open_file(root_b.join(p)).map(Some).unwrap_or(None);

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
