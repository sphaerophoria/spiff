use anyhow::{anyhow, Context, Result};
use memmap2::Mmap;
use sha2::{Digest, Sha256};
use std::{
    borrow::Cow,
    collections::{BTreeSet, HashMap},
    fmt::{self, Write as FmtWrite},
    fs,
    fs::File,
    path::{Path, PathBuf},
};

use libdiff::{DiffAction, OverMemoryLimitError};

pub mod widget;

#[derive(Hash, Copy, Eq, PartialEq, Clone)]
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

#[derive(Hash, Clone, Eq, PartialEq)]
pub struct DiffOptions {
    pub context_lines: usize,
    pub track_moves: bool,
    pub consider_whitespace: bool,
    pub line_numbers: bool,
    pub view_mode: ViewMode,
    pub max_memory_mb: usize,
}

impl Default for DiffOptions {
    fn default() -> DiffOptions {
        DiffOptions {
            context_lines: 5,
            consider_whitespace: true,
            track_moves: true,
            line_numbers: true,
            view_mode: ViewMode::Unified,
            max_memory_mb: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SegmentPurpose {
    Context,
    Addition,
    Removal,
    MoveFrom(DiffLineIdx),
    MoveTo(DiffLineIdx),
    MatchSearch,
    TrailingWhitespace,
    Failed,
}

#[derive(Debug)]
enum DiffGenerationError {
    GenerateDiffs(OverMemoryLimitError),
    GenerateMatches(OverMemoryLimitError),
}

impl fmt::Display for DiffGenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use DiffGenerationError::*;
        match self {
            GenerateDiffs(e) => write!(f, "failed to generate diffs: {e}"),
            GenerateMatches(e) => write!(f, "failed to generate matches: {e}"),
        }
    }
}
#[derive(Debug, Hash, Clone, PartialEq, Eq)]
pub struct DiffLineIdx {
    pub diff_idx: usize,
    pub line_idx: usize,
}

#[derive(Debug, Hash, Clone)]
pub struct ProcessedDiffIdx {
    pub diff_idx: usize,
    pub string_idx: usize,
}

pub enum ChangeOverview {
    Known {
        num_inserted_lines: usize,
        num_removed_lines: usize,
        num_moved_insertions: usize,
        num_moved_removals: usize,
    },
    Unknown,
}

pub struct ProcessedDiffCollection {
    pub processed_diffs: Vec<ProcessedDiffData>,
    pub search_results: Vec<ProcessedDiffIdx>,
}

pub struct ProcessedDiffData {
    pub label: String,
    pub processed_diff: String,
    pub line_numbers: String,
    pub display_info: Vec<(std::ops::Range<usize>, SegmentPurpose)>,
    pub change_overview: ChangeOverview,
}

pub struct DiffCollectionProcessor<'a> {
    options: DiffOptions,
    labels: &'a [String],
    lines_a: Vec<Vec<Cow<'a, str>>>,
    lines_b: Vec<Vec<Cow<'a, str>>>,
    trimmed_lines_a: Vec<Vec<Cow<'a, str>>>,
    trimmed_lines_b: Vec<Vec<Cow<'a, str>>>,
    diffs: Vec<Result<Vec<DiffAction>, DiffGenerationError>>,
    matches: HashMap<(usize, usize), (usize, usize)>,
    search_query: String,
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

        let mut processor = DiffCollectionProcessor {
            options: Default::default(),
            labels,
            lines_a,
            lines_b,
            trimmed_lines_a,
            trimmed_lines_b,
            diffs: Vec::new(),
            matches: HashMap::new(),
            search_query: String::new(),
        };

        processor.recompute_diffs();

        Ok(processor)
    }

    pub fn options(&self) -> &DiffOptions {
        &self.options
    }

    fn recompute_diffs(&mut self) {
        const MB_TO_B: usize = 1024 * 1024;
        let max_mem_bytes = self.options.max_memory_mb * MB_TO_B;
        self.diffs.clear();

        for i in 0..self.lines_a.len() {
            let diff = if self.options.consider_whitespace {
                libdiff::diff(&self.lines_a[i], &self.lines_b[i], max_mem_bytes)
            } else {
                libdiff::diff(
                    &self.trimmed_lines_a[i],
                    &self.trimmed_lines_b[i],
                    max_mem_bytes,
                )
            };
            self.diffs
                .push(diff.map_err(DiffGenerationError::GenerateDiffs));
        }

        let all_diffs_generated = self.diffs.iter().all(|v| v.is_ok());
        (self.diffs, self.matches) = if self.options.track_moves && all_diffs_generated {
            let num_diffs = self.diffs.len();
            let mut diffs = Vec::new();
            std::mem::swap(&mut diffs, &mut self.diffs);

            let diffs: Vec<Vec<DiffAction>> = diffs.into_iter().map(|v| v.unwrap()).collect();
            let matched_diffs = if self.options.consider_whitespace {
                libdiff::match_insertions_removals(
                    diffs,
                    &self.lines_a,
                    &self.lines_b,
                    max_mem_bytes,
                )
            } else {
                libdiff::match_insertions_removals(
                    diffs,
                    &self.trimmed_lines_a,
                    &self.trimmed_lines_b,
                    max_mem_bytes,
                )
            };

            match matched_diffs {
                Ok(v) => {
                    let diffs = v.diffs.into_iter().map(Ok).collect();
                    (diffs, v.matches)
                }
                Err(e) => (
                    (0..num_diffs)
                        .map(|_| Err(DiffGenerationError::GenerateMatches(e)))
                        .collect(),
                    HashMap::new(),
                ),
            }
        } else {
            let mut diffs = Vec::new();
            std::mem::swap(&mut diffs, &mut self.diffs);
            (diffs, [].into())
        };
    }

    pub fn process_new_options(&mut self, options: &DiffOptions) {
        if self.options != *options {
            self.options = options.clone();
            self.recompute_diffs();
        }
    }

    pub fn set_search_query(&mut self, query: String) {
        self.search_query = query;
    }

    pub fn generate(&self) -> ProcessedDiffCollection {
        let mut diffs = Vec::new();
        let mut search_results = Vec::new();

        for i in 0..self.lines_a.len() {
            let (data, single_search_results) = {
                match &self.diffs[i] {
                    Ok(_diffs) => SingleDiffProcessor::new(SingleDiffProcessorInputs {
                        options: &self.options,
                        lines_a: &self.lines_a[i],
                        lines_b: &self.lines_b[i],
                        label: &self.labels[i],
                        diffs: &self.diffs,
                        diff_idx: i,
                        matches: &self.matches,
                        search_query: &self.search_query,
                    })
                    .process(),
                    Err(e) => {
                        let processed_string = e.to_string();
                        let display_info =
                            vec![(0..processed_string.len(), SegmentPurpose::Failed)];
                        let change_overview = ChangeOverview::Unknown;

                        let data = ProcessedDiffData {
                            label: self.labels[i].clone(),
                            processed_diff: processed_string,
                            line_numbers: "".to_string(),
                            display_info,
                            change_overview,
                        };

                        (data, Vec::new())
                    }
                }
            };

            diffs.push(data);
            search_results.extend(single_search_results.into_iter().map(|string_idx| {
                ProcessedDiffIdx {
                    diff_idx: i,
                    string_idx,
                }
            }));
        }

        ProcessedDiffCollection {
            processed_diffs: diffs,
            search_results,
        }
    }
}

struct SingleDiffProcessorInputs<'a> {
    options: &'a DiffOptions,
    lines_a: &'a [Cow<'a, str>],
    lines_b: &'a [Cow<'a, str>],
    label: &'a str,
    diffs: &'a [Result<Vec<libdiff::DiffAction>, DiffGenerationError>],
    diff_idx: usize,
    matches: &'a std::collections::HashMap<(usize, usize), (usize, usize)>,
    search_query: &'a str,
}

/// Takes diff/matches/files/options and generates a DiffViewFileData for a single file pair
struct SingleDiffProcessor<'a> {
    options: &'a DiffOptions,
    lines_a: &'a [Cow<'a, str>],
    lines_b: &'a [Cow<'a, str>],
    label: &'a str,
    diffs: &'a [Result<Vec<libdiff::DiffAction>, DiffGenerationError>],
    diff_idx: usize,
    matches: &'a std::collections::HashMap<(usize, usize), (usize, usize)>,
    search_query: &'a str,
    search_results: Vec<usize>,
    processed_diff: String,
    line_numbers: String,
    display_info: Vec<(std::ops::Range<usize>, SegmentPurpose)>,
    segment_start: usize,
    num_inserted_lines: usize,
    num_removed_lines: usize,
    num_moved_insertions: usize,
    // It may seem like this should == insertions, however we track moves across files, so we may
    // have lines that are move-removed in one file and move-inserted in another
    num_moved_removals: usize,
}

impl<'a> SingleDiffProcessor<'a> {
    fn new(inputs: SingleDiffProcessorInputs<'a>) -> SingleDiffProcessor<'a> {
        SingleDiffProcessor {
            lines_a: inputs.lines_a,
            lines_b: inputs.lines_b,
            label: inputs.label,
            options: inputs.options,
            diffs: inputs.diffs,
            diff_idx: inputs.diff_idx,
            matches: inputs.matches,
            search_query: inputs.search_query,
            search_results: Vec::new(),
            processed_diff: String::new(),
            line_numbers: String::new(),
            display_info: Vec::new(),
            segment_start: 0,
            num_inserted_lines: 0,
            num_removed_lines: 0,
            num_moved_insertions: 0,
            num_moved_removals: 0,
        }
    }

    fn process(mut self) -> (ProcessedDiffData, Vec<usize>) {
        for (idx, action) in self.diffs[self.diff_idx].iter().flatten().enumerate() {
            use DiffAction::*;
            match action {
                Traverse(traversal) => {
                    self.process_traversal(
                        traversal,
                        idx != 0,
                        idx != self.diffs[self.diff_idx]
                            .as_ref()
                            .map(|x| x.len())
                            .unwrap_or(1)
                            - 1,
                    );
                }
                Insert(insertion) => {
                    self.process_insertion(insertion, idx);
                }
                Remove(removal) => {
                    self.process_removal(removal, idx);
                }
            }
        }

        let change_overview = ChangeOverview::Known {
            num_removed_lines: self.num_removed_lines,
            num_inserted_lines: self.num_inserted_lines,
            num_moved_insertions: self.num_moved_insertions,
            num_moved_removals: self.num_moved_removals,
        };
        let data = ProcessedDiffData {
            processed_diff: self.processed_diff,
            line_numbers: self.line_numbers,
            label: self.label.to_string(),
            display_info: self.display_info,
            change_overview,
        };

        (data, self.search_results)
    }

    fn process_traversal(
        &mut self,
        traversal: &libdiff::Traversal,
        show_start: bool,
        show_end: bool,
    ) {
        let a_idx_offset = traversal.a_idx as i64 - traversal.b_idx as i64;

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

        self.push_coloring_with_search_highlights(SegmentPurpose::Context);
    }

    fn push_coloring_with_search_highlights(&mut self, purpose: SegmentPurpose) {
        if self.segment_start == self.processed_diff.len() {
            return;
        }

        let altered_string = &self.processed_diff[self.segment_start..];

        let mut pos = self.segment_start;

        if !self.search_query.is_empty() {
            let match_indices = altered_string
                .match_indices(self.search_query)
                .map(|(idx, _)| idx + self.segment_start)
                .collect::<Vec<_>>();

            for result_idx in match_indices {
                self.search_results.push(result_idx);

                self.display_info.push((pos..result_idx, purpose.clone()));
                self.display_info.push((
                    result_idx..result_idx + self.search_query.len(),
                    SegmentPurpose::MatchSearch,
                ));

                pos = result_idx + self.search_query.len();
            }
        }

        self.display_info
            .push((pos..self.processed_diff.len(), purpose));

        self.segment_start = self.processed_diff.len();
    }

    fn process_insertion(&mut self, insertion: &libdiff::Insertion, idx: usize) {
        if self.options.view_mode == ViewMode::AOnly {
            return;
        }

        let purpose = if let Some((diff_idx, chunk_idx)) = self.matches.get(&(self.diff_idx, idx)) {
            let line_idx = match &self.diffs[*diff_idx]
                .as_ref()
                .expect("Should not resolve match with invalid diff")[*chunk_idx]
            {
                DiffAction::Remove(removal) => removal.a_idx,
                _ => panic!("Invalid match"),
            };
            self.num_moved_insertions += insertion.length;
            SegmentPurpose::MoveTo(DiffLineIdx {
                diff_idx: *diff_idx,
                line_idx,
            })
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

            let right_trimmed = line.trim_end();
            write!(self.processed_diff, "+{}", right_trimmed).expect("Failed to write line");
            self.end_line(right_trimmed, line, &purpose);
        }

        self.push_coloring_with_search_highlights(purpose);
        self.num_inserted_lines += insertion.length;
    }

    fn process_removal(&mut self, removal: &libdiff::Removal, idx: usize) {
        if self.options.view_mode == ViewMode::BOnly {
            return;
        }

        let purpose = if let Some((diff_idx, chunk_idx)) = self.matches.get(&(self.diff_idx, idx)) {
            let line_idx = match &self.diffs[*diff_idx]
                .as_ref()
                .expect("Should not resolve match with invalid diff")[*chunk_idx]
            {
                DiffAction::Insert(insertion) => insertion.b_idx,
                _ => panic!("Invalid match"),
            };
            self.num_moved_removals += removal.length;
            SegmentPurpose::MoveFrom(DiffLineIdx {
                diff_idx: *diff_idx,
                line_idx,
            })
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

            let right_trimmed = line.trim_end();
            write!(self.processed_diff, "-{}", right_trimmed).expect("Failed to write line");
            self.end_line(right_trimmed, line, &purpose);
        }

        self.push_coloring_with_search_highlights(purpose);
        self.num_removed_lines += removal.length;
    }

    fn end_line(&mut self, right_trimmed: &str, line: &str, purpose: &SegmentPurpose) {
        if right_trimmed.len() < line.len() {
            self.push_coloring_with_search_highlights(purpose.clone());

            writeln!(self.processed_diff, "{}", &line[right_trimmed.len()..])
                .expect("Failed to write line");
            self.display_info.push((
                self.segment_start..self.processed_diff.len(),
                SegmentPurpose::TrailingWhitespace,
            ));

            self.segment_start = self.processed_diff.len();
        } else {
            writeln!(self.processed_diff).expect("Failed to write line");
        }
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

fn buf_to_lines(buf: &[u8]) -> Result<Vec<Cow<str>>> {
    let s = match std::str::from_utf8(buf) {
        Ok(s) => s,
        Err(_) => {
            let hash = Sha256::digest(buf);
            let hash_str = hex::encode(hash);
            return Ok(vec![Cow::Owned(format!(
                "Binary file with sha256 digest: {}",
                hash_str
            ))]);
        }
    };

    Ok(s.lines().map(Cow::Borrowed).collect())
}

fn bufs_to_lines<C>(bufs: &[Option<C>]) -> Result<Vec<Vec<Cow<str>>>>
where
    C: AsRef<[u8]>,
{
    bufs.iter()
        .map(|x| buf_to_lines(unwrap_or_empty_buf(x)))
        .collect::<Result<Vec<_>>>()
}

fn trim_lines<'a>(lines: &[Vec<Cow<'a, str>>]) -> Vec<Vec<Cow<'a, str>>> {
    lines
        .iter()
        .map(|x| {
            x.iter()
                .map(|x| match x {
                    Cow::Owned(x) => Cow::Owned(x.trim().to_string()),
                    Cow::Borrowed(x) => Cow::Borrowed(x.trim()),
                })
                .collect::<Vec<_>>()
        })
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
        all_paths.extend(paths_2);
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

#[cfg(test)]
mod test {
    use super::*;

    #[derive(Default)]
    struct SingleDiffProcessorInputsBuilder {
        options: DiffOptions,
        label: String,
        diffs: Vec<Result<Vec<DiffAction>, DiffGenerationError>>,
        matches: std::collections::HashMap<(usize, usize), (usize, usize)>,
    }

    impl SingleDiffProcessorInputsBuilder {
        fn inputs<'a>(
            &'a mut self,
            lines_a: &'a [Cow<'a, str>],
            lines_b: &'a [Cow<'a, str>],
        ) -> SingleDiffProcessorInputs<'a> {
            let diffs = vec![libdiff::diff(&lines_a, &lines_b, usize::MAX).unwrap()];
            let libdiff::MatchedDiffs { diffs, matches } =
                libdiff::match_insertions_removals(diffs, &[lines_a], &[lines_b], usize::MAX)
                    .unwrap();

            self.diffs = diffs.into_iter().map(Ok).collect();
            self.matches = matches;

            SingleDiffProcessorInputs {
                options: &self.options,
                lines_a: &lines_a,
                lines_b: &lines_b,
                label: &self.label,
                diffs: &self.diffs,
                diff_idx: 0,
                matches: &self.matches,
                search_query: "",
            }
        }
    }

    #[test]
    fn test_trailing_whitespace_simple() -> Result<()> {
        let b1 = buf_to_lines("test  ".as_bytes())?;
        let b2 = buf_to_lines("test".as_bytes())?;
        let mut builder = SingleDiffProcessorInputsBuilder::default();

        let (process_output, _) = SingleDiffProcessor::new(builder.inputs(&b1, &b2)).process();

        assert_eq!(
            process_output.processed_diff,
            "-test  \n\
             +test\n"
        );

        assert_eq!(
            process_output.display_info,
            [
                (0..5, SegmentPurpose::Removal),
                (5..8, SegmentPurpose::TrailingWhitespace),
                (8..14, SegmentPurpose::Addition),
            ]
        );

        let (process_output, _) = SingleDiffProcessor::new(builder.inputs(&b2, &b1)).process();

        assert_eq!(
            process_output.processed_diff,
            "-test\n\
             +test  \n"
        );

        assert_eq!(
            process_output.display_info,
            [
                (0..6, SegmentPurpose::Removal),
                (6..11, SegmentPurpose::Addition),
                (11..14, SegmentPurpose::TrailingWhitespace),
            ]
        );

        Ok(())
    }

    #[test]
    fn test_insertion_removal_count() -> Result<()> {
        let b1 = buf_to_lines(
            "test\n\
             test2\n\
             test3\n\
             test4\n\
             test5"
                .as_bytes(),
        )?;
        let b2 = buf_to_lines(
            "test2\n\
             test\n\
             inserted\n\
             inserted\n\
             inserted\n\
             test4\n\
             inserted\n\
             inserted\n\
             test3\n"
                .as_bytes(),
        )?;

        let mut builder = SingleDiffProcessorInputsBuilder::default();

        let (process_output, _) = SingleDiffProcessor::new(builder.inputs(&b1, &b2)).process();

        let ChangeOverview::Known {
            num_inserted_lines,
            num_moved_insertions,
            num_removed_lines,
            num_moved_removals,
        } = process_output.change_overview
        else {
            panic!("change overview should be known")
        };

        assert_eq!(num_inserted_lines, 7);
        assert_eq!(num_moved_insertions, 2);
        assert_eq!(num_removed_lines, 3);
        assert_eq!(num_moved_removals, 2);

        Ok(())
    }
}
