use std::{
    cmp::{Eq, PartialEq},
    collections::{BinaryHeap, HashMap, HashSet},
    fmt,
};

use rayon::prelude::*;

/// An insertion into a
#[derive(Eq, PartialEq, Debug, Clone, Ord, PartialOrd, Hash)]
pub struct Insertion {
    /// Where in b the content to insert is
    pub b_idx: usize,
    /// How many elements to insert
    pub length: usize,
}

/// A removal from a
#[derive(Eq, PartialEq, Debug, Clone, Ord, PartialOrd, Hash)]
pub struct Removal {
    /// Where in a to remove from
    pub a_idx: usize,
    /// How many elements to remove
    pub length: usize,
}

/// A traversal. No changes made to a
#[derive(Eq, PartialEq, Debug, Clone, Ord, PartialOrd, Hash)]
pub struct Traversal {
    /// Where the content is in a
    pub a_idx: usize,
    /// Where the content is in b
    pub b_idx: usize,
    /// How many elements
    pub length: usize,
}

/// Representation of a single action in a diff. A sequence of actions applied to a should end with
/// b
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DiffAction {
    Traverse(Traversal),
    Insert(Insertion),
    Remove(Removal),
}

/// Helper struct to build a DiffAction sequence
///
/// Sequence is expected to be built in reverse order as we back-track through our path generated
/// by the myers diff algorithm
#[derive(Clone, Default, Debug, Eq, PartialEq)]
struct DiffBuilder {
    seq: Vec<DiffAction>,
}

impl DiffBuilder {
    /// Add a traversed item to the diff
    ///
    /// If the last element is already a traversal its length will be amended
    fn push_traversed_item(&mut self, a_idx: usize, b_idx: usize, length: usize) {
        if let Some(DiffAction::Traverse(traversal)) = self.seq.last_mut() {
            if traversal.a_idx == a_idx + 1 && traversal.b_idx == b_idx + 1 {
                traversal.a_idx -= 1;
                traversal.b_idx -= 1;
                traversal.length += 1;
                return;
            }
        }

        self.seq.push(DiffAction::Traverse(Traversal {
            a_idx,
            b_idx,
            length,
        }));
    }

    /// Add an inserted item to the diff
    ///
    /// If the last element is already an insertion its length will be amended
    fn push_insertion(&mut self, b_idx: usize) {
        if let Some(DiffAction::Insert(insertion)) = self.seq.last_mut() {
            if insertion.b_idx == b_idx + 1 {
                insertion.b_idx -= 1;
                insertion.length += 1;
                return;
            }
        }

        self.seq
            .push(DiffAction::Insert(Insertion { b_idx, length: 1 }));
    }

    /// Add a removed item to the diff
    ///
    /// If the last element is already a removal its length will be amended
    fn push_removal(&mut self, a_idx: usize) {
        if let Some(DiffAction::Remove(removal)) = self.seq.last_mut() {
            if removal.a_idx == a_idx + 1 {
                removal.a_idx -= 1;
                removal.length += 1;
                return;
            }
        }

        self.seq
            .push(DiffAction::Remove(Removal { a_idx, length: 1 }));
    }
}

#[derive(Debug, Clone, Copy)]
pub struct OverMemoryLimitError {
    required: usize,
    maximum: usize,
}

impl fmt::Display for OverMemoryLimitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "algorithm required {} bytes, which is over the maximum of {} bytes",
            self.required, self.maximum
        )
    }
}

enum MoveSourceDirection {
    Root,
    Left,
    Up,
}

struct MoveSource {
    d: i64,
    k: i64,
    direction: MoveSourceDirection,
}

/// How did we get to a given d, k position? Assumes that trace has been populated for 0..d
fn get_move_source<T: MyersTrace>(trace: &mut T, d: i64, k: i64) -> MoveSource {
    let direction = if d == 0 {
        MoveSourceDirection::Root
    } else if k == -d {
        MoveSourceDirection::Up
    } else if k == d {
        MoveSourceDirection::Left
    } else {
        let left = *trace.get_mut(d - 1, k - 1);
        let right = *trace.get_mut(d - 1, k + 1);
        if left < right {
            MoveSourceDirection::Up
        } else {
            MoveSourceDirection::Left
        }
    };

    let prev_k = match direction {
        MoveSourceDirection::Root => 0,
        MoveSourceDirection::Up => k + 1,
        MoveSourceDirection::Left => k - 1,
    };

    MoveSource {
        d: d - 1,
        k: prev_k,
        direction,
    }
}

/// Extract the x value associated with a given d, k from the trace. This handles the edge case
/// where we are at the trace root and the previous d value is invalid
fn resolve_x_for_d_k<T: MyersTrace>(trace: &mut T, d: i64, k: i64) -> i64 {
    if d < 0 {
        0
    } else {
        *trace.get_mut(d, k)
    }
}

/// Calculate the x position for a given d, k without advancing through the next diagonal. Assumes
/// that previous d/k positions have already been walked
fn calc_x_without_diagonal<T: MyersTrace>(trace: &mut T, d: i64, k: i64) -> i64 {
    let move_source = get_move_source(trace, d, k);
    let prev_x = resolve_x_for_d_k(trace, move_source.d, move_source.k);
    match move_source.direction {
        MoveSourceDirection::Root => 0,
        MoveSourceDirection::Up => prev_x,
        MoveSourceDirection::Left => prev_x + 1,
    }
}

pub trait MyersTrace {
    fn get_mut(&mut self, d: i64, k: i64) -> &mut i64;
}

/// Myers trace with information required to walk all the way back to the root
#[derive(Debug)]
pub struct FullMyersTrace {
    data: Box<[i64]>,
}

impl FullMyersTrace {
    fn new(
        a_len: usize,
        b_len: usize,
        max_mem_bytes: usize,
    ) -> Result<FullMyersTrace, OverMemoryLimitError> {
        let max_d = a_len + b_len;

        // k is iterated from [-d, d], on every other
        // e.g.
        //
        //             k
        //   ...-3-2-1 0 1 2 3...
        //   . . . . . o . . . . .
        //   . . . . o . o . . . .
        // d . . . o . o . o . . .
        //   . . o . o . o . o . .
        //   . o . o . o . o . o .
        //   o . o . o . o . o . o
        //
        // * For each value of d, we need 1 + d slots
        // * d is iterated from 0 to max_d
        // * sum of integers is n(n + 1) / 2
        // so for 0..=max_d we need 1..=(max_d + 1) slots which is
        let num_slots = (max_d + 1) * (max_d + 2) / 2;

        let required_mem_bytes = num_slots * std::mem::size_of::<i64>();
        if required_mem_bytes > max_mem_bytes {
            return Err(OverMemoryLimitError {
                required: required_mem_bytes,
                maximum: max_mem_bytes,
            });
        }
        Ok(FullMyersTrace {
            data: vec![0; num_slots].into(),
        })
    }
}

impl MyersTrace for FullMyersTrace {
    fn get_mut(&mut self, d: i64, k: i64) -> &mut i64 {
        // See new() for data layout
        // Indexing is the same logic as generation. Generate the pyramid for d - 1, then move over
        // by k slots
        let k_start = (d) * (d + 1) / 2;
        // k goes from -d to d, so for this row we need to map [-d, d] -> [0, 2d]
        assert!(k >= -d && k <= d);
        let unsigned_k = k + d;
        let idx = (k_start + unsigned_k / 2) as usize;
        &mut self.data[idx]
    }
}

/// Myers trace that can only remember enough data to continue the walk. Backtracking will not
/// work.
#[derive(Debug)]
pub struct ForgetfulMyersTrace {
    data: Box<[i64]>,
    max_distance: usize,
}

impl ForgetfulMyersTrace {
    fn new(a_len: usize, b_len: usize) -> ForgetfulMyersTrace {
        let max_distance = a_len + b_len;
        let num_slots = 2 * max_distance + 1;

        ForgetfulMyersTrace {
            data: vec![0; num_slots].into(),
            max_distance,
        }
    }
}

impl MyersTrace for ForgetfulMyersTrace {
    fn get_mut(&mut self, _d: i64, k: i64) -> &mut i64 {
        // k is in -d..=d
        // data is in 0..=2d +1
        let idx = k + self.max_distance as i64;
        debug_assert!(idx < self.data.len() as i64 && idx >= 0);
        &mut self.data[idx as usize]
    }
}

/// Wrapper around other MyersTrace implementations
#[derive(Debug)]
enum AlgoTrace {
    Full(FullMyersTrace),
    Forgetful(ForgetfulMyersTrace),
}

impl MyersTrace for AlgoTrace {
    fn get_mut(&mut self, d: i64, k: i64) -> &mut i64 {
        match self {
            AlgoTrace::Full(x) => x.get_mut(d, k),
            AlgoTrace::Forgetful(x) => x.get_mut(d, k),
        }
    }
}

/// Snapshot of the DiffAlgo state. Useful for debugging/visualization
#[derive(Debug)]
pub struct DiffAlgoDebugInfo {
    // steps[k][line_segment]
    pub steps: Vec<Vec<(i64, i64)>>,
    // Limits
    pub top: i64,
    pub left: i64,
    pub bottom: i64,
    pub right: i64,
}

#[derive(PartialEq)]
pub enum DiffAlgoState {
    Finished,
    InProgress,
}

#[derive(Debug, Copy, Clone)]
enum AlgoWalkDirection {
    Forwards,
    Backwards,
}

#[derive(Debug)]
pub struct DiffAlgo {
    max_distance: i64,
    direction: AlgoWalkDirection,
    trace: AlgoTrace,
    d: i64,
    k: i64,
    // Algo limits
    top: i64,
    left: i64,
    bottom: i64,
    right: i64,
}

impl DiffAlgo {
    pub fn new<U>(
        a: &[U],
        b: &[U],
        top: i64,
        left: i64,
        bottom: i64,
        right: i64,
        max_memory_bytes: usize,
    ) -> Result<DiffAlgo, OverMemoryLimitError> {
        let trace = FullMyersTrace::new(a.len(), b.len(), max_memory_bytes)?;
        Ok(Self::new_with_trace(
            a,
            b,
            AlgoWalkDirection::Forwards,
            top,
            left,
            bottom,
            right,
            AlgoTrace::Full(trace),
        ))
    }

    pub fn new_forgetful<U>(
        a: &[U],
        b: &[U],
        top: i64,
        left: i64,
        bottom: i64,
        right: i64,
    ) -> DiffAlgo {
        let trace = ForgetfulMyersTrace::new(a.len(), b.len());
        Self::new_with_trace(
            a,
            b,
            AlgoWalkDirection::Forwards,
            top,
            left,
            bottom,
            right,
            AlgoTrace::Forgetful(trace),
        )
    }

    pub fn new_backwards<U>(
        a: &[U],
        b: &[U],
        top: i64,
        left: i64,
        bottom: i64,
        right: i64,
    ) -> DiffAlgo {
        // FIXME: Backwards walk does not need full trace, however it is useful for debugging
        let trace = FullMyersTrace::new(a.len(), b.len(), 100 * 1024 * 1024).unwrap();
        Self::new_with_trace(
            a,
            b,
            AlgoWalkDirection::Backwards,
            top,
            left,
            bottom,
            right,
            AlgoTrace::Full(trace),
        )
    }

    fn new_with_trace<U>(
        a: &[U],
        b: &[U],
        direction: AlgoWalkDirection,
        top: i64,
        left: i64,
        bottom: i64,
        right: i64,
        trace: AlgoTrace,
    ) -> DiffAlgo {
        assert!(bottom >= top);
        assert!(right >= left);
        assert!((0..=a.len() as i64).contains(&left));
        assert!((0..=a.len() as i64).contains(&right));
        assert!((0..=b.len() as i64).contains(&top));
        assert!((0..=b.len() as i64).contains(&bottom));
        let max_distance = (bottom - top) + (right - left);

        assert!(top >= 0);
        assert!(left >= 0);
        DiffAlgo {
            max_distance,
            direction,
            trace,
            d: 0,
            k: -2,
            top,
            left,
            bottom,
            right,
        }
    }

    pub fn step<U: PartialEq>(&mut self, a: &[U], b: &[U]) -> DiffAlgoState {
        let a = &a[self.left as usize..self.right as usize];
        let b = &b[self.top as usize..self.bottom as usize];

        // A backwards walk is the same as a forwards walk flipped over x=y
        // If we just invert everything, all other code continues working for free
        let (left, right, top, bottom, a, b) = match self.direction {
            AlgoWalkDirection::Backwards => (self.top, self.bottom, self.left, self.right, b, a),
            AlgoWalkDirection::Forwards => (self.left, self.right, self.top, self.bottom, a, b),
        };

        // The actual myers diff algorithm
        //
        // http://www.xmailserver.org/diff2.pdf
        // https://blog.jcoglan.com/2017/02/12/the-myers-diff-algorithm-part-1/
        let mut x;
        let mut y;
        loop {
            // Each level of the algorithm only works on odd or even values of k, see original
            // paper
            self.k += 2;

            // Translation of outer loop. If we hit the limit of k d has to be increased by 1, and
            // k has to be reset
            if self.k > self.d {
                self.d += 1;
                self.k = -self.d;
            }

            x = calc_x_without_diagonal(&mut self.trace, self.d, self.k);
            y = x - self.k;

            if x <= right - left && y <= bottom - top {
                break;
            }

            // If we have passed the above condition, we are outside the region of interest.
            // However we still need to mark this location in the trace as the next stage of the
            // algorithm will try to read from it. No need to advance diagonals though, and no need
            // to pause the algorithm here
            *self.trace.get_mut(self.d, self.k) = x;
        }
        assert!(y >= 0);

        while x < a.len() as i64
            && y < b.len() as i64
            && extract_from_source(x, a, self.direction)
                == extract_from_source(y, b, self.direction)
        {
            x += 1;
            y += 1;
        }

        *self.trace.get_mut(self.d, self.k) = x;

        if x >= a.len() as i64 && y >= b.len() as i64 {
            return DiffAlgoState::Finished;
        }

        DiffAlgoState::InProgress
    }

    pub fn debug_info(&mut self) -> DiffAlgoDebugInfo {
        if self.k < -self.d || self.k > self.d {
            return DiffAlgoDebugInfo {
                steps: Vec::new(),
                top: self.top,
                left: self.left,
                right: self.right,
                bottom: self.bottom,
            };
        }
        let mut steps = Vec::new();

        for k in (-self.d..=self.d).step_by(2) {
            if k > self.k {
                break;
            }

            let x = *self.trace.get_mut(self.d, k);
            let y = x - k;
            let local_to_global_xy = |(mut x, mut y)| match self.direction {
                AlgoWalkDirection::Forwards => (x + self.left, y + self.top),
                AlgoWalkDirection::Backwards => {
                    x = self.bottom - x;
                    y = self.right - y;
                    (y, x)
                }
            };

            let mut steps_for_k: Vec<(i64, i64)> = match &mut self.trace {
                AlgoTrace::Full(trace) => {
                    let backwards_iter = MyersBackwardsIterator::new(self.d, 0, x, y, trace);
                    backwards_iter.map(local_to_global_xy).collect()
                }
                AlgoTrace::Forgetful(trace) => {
                    let backwards_iter = MyersBackwardsIterator::new(self.d, self.d, x, y, trace);
                    backwards_iter.map(local_to_global_xy).collect()
                }
            };

            steps_for_k.reverse();
            steps.push(steps_for_k);
        }

        DiffAlgoDebugInfo {
            steps,
            top: self.top,
            left: self.left,
            right: self.right,
            bottom: self.bottom,
        }
    }
}

fn extract_from_source<U>(pos: i64, source: &[U], direction: AlgoWalkDirection) -> &U {
    match direction {
        AlgoWalkDirection::Forwards => &source[pos as usize],
        AlgoWalkDirection::Backwards => &source[source.len() - pos as usize - 1],
    }
}

/// Find a sequence of actions that applied to a results in b
pub fn diff<U>(
    a: &[U],
    b: &[U],
    max_memory_bytes: usize,
) -> Result<Vec<DiffAction>, OverMemoryLimitError>
where
    U: Eq + PartialEq,
{
    // Myers diff algorithm, no splitting into smaller segments
    //
    // http://www.xmailserver.org/diff2.pdf
    // https://blog.jcoglan.com/2017/02/12/the-myers-diff-algorithm-part-1/
    let mut algo = DiffAlgo::new(a, b, 0, 0, b.len() as i64, a.len() as i64, max_memory_bytes)?;
    if algo.max_distance == 0 {
        return Ok(vec![DiffAction::Traverse(Traversal {
            a_idx: 0,
            b_idx: 0,
            length: 0,
        })]);
    }

    while algo.step(a, b) != DiffAlgoState::Finished {}

    let shortest_edit_distance = algo.d;
    let trace = algo.trace;
    let AlgoTrace::Full(mut trace) = trace else {
        panic!("Unexpected forgetful trace");
    };

    let mut builder = DiffBuilder::default();

    let x = a.len() as i64;
    let y = b.len() as i64;
    let mut it = MyersBackwardsIterator::new(shortest_edit_distance, 0, x, y, &mut trace);

    // FIXME: invalid unwrap?
    let last = it.next().unwrap();
    let mut prev_x = last.0;
    let mut prev_y = last.1;
    for (x, y) in it {
        assert!(x >= 0 && y >= 0);
        if prev_y == y {
            builder.push_removal(x as usize);
        } else if prev_x == x {
            builder.push_insertion(y as usize);
        } else {
            let length = prev_x - x;
            builder.push_traversed_item(x as usize, y as usize, length as usize);
        }
        prev_x = x;
        prev_y = y;
    }

    builder.seq.reverse();
    Ok(builder.seq)
}

#[derive(Debug)]
struct MyersBackwardsIterator<'a, T> {
    d_limit: i64,
    d: i64,
    k: i64,
    cached_xy: Option<(i64, i64)>,
    trace: &'a mut T,
}

impl<T: MyersTrace> MyersBackwardsIterator<'_, T> {
    fn new(d: i64, d_limit: i64, x: i64, y: i64, trace: &mut T) -> MyersBackwardsIterator<'_, T> {
        let k = x - y;
        MyersBackwardsIterator {
            d_limit,
            d,
            k,
            cached_xy: None,
            trace,
        }
    }
}

impl<T: MyersTrace> Iterator for MyersBackwardsIterator<'_, T> {
    type Item = (i64, i64);

    fn next(&mut self) -> Option<Self::Item> {
        // Note: Due to the greedy nature of the algorithm. We may get a movement that looks like
        // the following
        // o - o
        //      \
        //       \
        //        o
        //
        // This needs to be mapped to both a traversal and a deletion, which means we need to
        // return two line segments. From the perspective of this iterator, this means we need to
        // return twice for this move. For each d, k if we see a diagnoal, we cache the second move
        if let Some(ret) = self.cached_xy.take() {
            return Some(ret);
        }

        if self.d < self.d_limit {
            return None;
        }

        let move_source = get_move_source(self.trace, self.d, self.k);

        let x = *self.trace.get_mut(self.d, self.k);
        let y = x - self.k;

        let prev_x = resolve_x_for_d_k(self.trace, move_source.d, move_source.k);
        let prev_y = prev_x - move_source.k;

        let x_diff = x - prev_x;
        let y_diff = y - prev_y;

        // If x and y have both changed by more than 1 element, we have a diagonal intermediate
        // move to return
        let traversal_diff = x_diff.min(y_diff);
        if traversal_diff != 0 {
            self.cached_xy = Some((x - traversal_diff, y - traversal_diff));
        }

        self.k = move_source.k;
        self.d -= 1;
        Some((x, y))
    }
}

/// Splits a sequence of diff actions into the insertions and removals that makes it up. Discards
/// traversals
#[allow(clippy::type_complexity)]
fn insertions_removals_from_actions(
    diffs: &[Vec<DiffAction>],
) -> (Vec<(usize, Insertion)>, Vec<(usize, Removal)>) {
    let mut insertions = Vec::new();
    let mut removals = Vec::new();
    for (idx, diff) in diffs.iter().enumerate() {
        for action in diff {
            match action {
                DiffAction::Traverse(_) => {}
                DiffAction::Insert(insertion) => {
                    insertions.push((idx, insertion.clone()));
                }
                DiffAction::Remove(removal) => {
                    removals.push((idx, removal.clone()));
                }
            }
        }
    }

    (insertions, removals)
}

/// Calculates how similar the given insertion and removal are. Score is how many lines match
fn calculate_match_score<U>(insertion: &Insertion, removal: &Removal, a: &[U], b: &[U]) -> f32
where
    U: Eq + PartialEq,
{
    let removal_content = &a[removal.a_idx..removal.a_idx + removal.length];
    let total = (insertion.b_idx..insertion.b_idx + insertion.length).fold(0.0, |acc, b_idx| {
        let insertion_item = &b[b_idx];
        if removal_content.contains(insertion_item) {
            acc + 1.0
        } else {
            acc
        }
    });

    total / (removal.length + insertion.length) as f32
}

/// Splits input insertion and removal to get segments with either a 100% match score or a 0% match
/// score
#[allow(clippy::type_complexity)]
fn split_insertion_removal_pair<U, C>(
    insertion: &(usize, Insertion),
    removal: &(usize, Removal),
    a: &[C],
    b: &[C],
    max_memory_bytes: usize,
) -> Result<(Vec<(usize, Insertion)>, Vec<(usize, Removal)>), OverMemoryLimitError>
where
    U: Eq + PartialEq,
    C: AsRef<[U]>,
{
    let insertion_content =
        &b[insertion.0].as_ref()[insertion.1.b_idx..insertion.1.b_idx + insertion.1.length];
    let removal_content =
        &a[removal.0].as_ref()[removal.1.a_idx..removal.1.a_idx + removal.1.length];

    // If we diff the diff, we should find traversal segments which indicate 100% overlap. Each
    // insertion will correspond to one of the split insertions, each removal will correspond to
    // one of the split removals
    let insertion_removal_diff = diff(removal_content, insertion_content, max_memory_bytes)?;

    let mut output_insertions = Vec::new();
    let mut output_removals = Vec::new();

    for action in insertion_removal_diff {
        match action {
            DiffAction::Traverse(traversal) => {
                output_insertions.push((
                    insertion.0,
                    Insertion {
                        b_idx: insertion.1.b_idx + traversal.b_idx,
                        length: traversal.length,
                    },
                ));
                output_removals.push((
                    removal.0,
                    Removal {
                        a_idx: removal.1.a_idx + traversal.a_idx,
                        length: traversal.length,
                    },
                ));
            }
            DiffAction::Insert(diff_insertion) => {
                output_insertions.push((
                    insertion.0,
                    Insertion {
                        b_idx: insertion.1.b_idx + diff_insertion.b_idx,
                        length: diff_insertion.length,
                    },
                ));
            }
            DiffAction::Remove(diff_removal) => {
                output_removals.push((
                    removal.0,
                    Removal {
                        a_idx: removal.1.a_idx + diff_removal.a_idx,
                        length: diff_removal.length,
                    },
                ));
            }
        }
    }

    Ok((output_insertions, output_removals))
}

fn replace_element_with_sequence<U: Eq + PartialEq>(elem: &U, seq: Vec<U>, vec: &mut Vec<U>) {
    let position = vec.iter().position(|removal| removal == elem).unwrap();
    vec.splice(position..position + 1, seq);
}

#[derive(PartialEq, Debug)]
struct MatchCandidate {
    score: f32,
    insertion: (usize, Insertion),
    removal: (usize, Removal),
}

impl Eq for MatchCandidate {}

impl PartialOrd for MatchCandidate {
    fn partial_cmp(&self, rhs: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(rhs))
    }
}

impl Ord for MatchCandidate {
    fn cmp(&self, rhs: &Self) -> std::cmp::Ordering {
        // Ignore insertion and removal in ordering. We only need to sort by the score
        self.score.partial_cmp(&rhs.score).unwrap()
    }
}

/// Calculate scores between all insertions and removals
///
/// Return all candidates where any lines match
fn calculate_match_scores<U, C>(
    insertions: &[(usize, Insertion)],
    removals: &[(usize, Removal)],
    a: &[C],
    b: &[C],
) -> BinaryHeap<MatchCandidate>
where
    U: PartialEq + Eq,
    C: AsRef<[U]> + Sync,
{
    let mut combinations = Vec::with_capacity(insertions.len() * removals.len());
    for insertion_idx in 0..insertions.len() {
        for removal_idx in 0..removals.len() {
            combinations.push((insertion_idx, removal_idx));
        }
    }

    combinations
        .into_par_iter()
        .filter_map(|(insertion_idx, removal_idx)| {
            let insertion = &insertions[insertion_idx];
            let removal = &removals[removal_idx];

            let score = calculate_match_score(
                &insertion.1,
                &removal.1,
                a[removal.0].as_ref(),
                b[insertion.0].as_ref(),
            );

            if score > 0.0 {
                Some(MatchCandidate {
                    insertion: insertion.clone(),
                    removal: removal.clone(),
                    score,
                })
            } else {
                None
            }
        })
        .collect()
}

pub struct MatchedDiffs {
    /// Sequence of diff sequences
    /// Inner vec is a sequence of actions that has to be applied to get from a to b
    /// Outer vec is a collection of different diffs for different files
    pub diffs: Vec<Vec<DiffAction>>,
    /// Indexes in diff that correspond to other indexes in diff
    /// Items are (diff_idx, seq_idx)
    pub matches: HashMap<(usize, usize), (usize, usize)>,
}

/// Find moves within the provided diff. Split long segments into smaller segments that can be
/// matched
pub fn match_insertions_removals<'a, U, C>(
    mut d: Vec<Vec<DiffAction>>,
    a: &[C],
    b: &[C],
    max_memory_bytes: usize,
) -> Result<MatchedDiffs, OverMemoryLimitError>
where
    U: Eq + PartialEq + 'a,
    C: AsRef<[U]> + Sync,
{
    let (mut insertions, mut removals) = insertions_removals_from_actions(&d);

    let mut match_candidates = calculate_match_scores(&insertions, &removals, a, b);

    let mut insertion_matches = HashMap::new();
    let mut removal_matches = HashSet::new();

    let mut ignored_removals = HashSet::new();
    let mut ignored_insertions = HashSet::new();

    while let Some(match_candidate) = match_candidates.pop() {
        if insertion_matches.contains_key(&match_candidate.insertion) {
            continue;
        }

        if removal_matches.contains(&match_candidate.removal) {
            continue;
        }

        if ignored_removals.contains(&match_candidate.removal) {
            continue;
        }

        if ignored_insertions.contains(&match_candidate.insertion) {
            continue;
        }

        let (mut split_insertions, mut split_removals) = split_insertion_removal_pair(
            &match_candidate.insertion,
            &match_candidate.removal,
            a,
            b,
            max_memory_bytes,
        )?;

        if split_insertions.len() == 1 && split_removals.len() == 1 {
            insertion_matches.insert(split_insertions.pop().unwrap(), split_removals[0].clone());
            removal_matches.insert(split_removals.pop().unwrap());
            continue;
        }

        if split_removals.len() > 1 {
            replace_element_with_sequence(
                &match_candidate.removal,
                split_removals.clone(),
                &mut removals,
            );

            replace_element_with_sequence(
                &DiffAction::Remove(match_candidate.removal.1.clone()),
                split_removals
                    .iter()
                    .cloned()
                    .map(|x| DiffAction::Remove(x.1))
                    .collect(),
                &mut d[match_candidate.removal.0],
            );

            // Re-compute scores of new removals with all insertions
            let new_candidates = calculate_match_scores(&insertions, &split_removals, a, b);
            match_candidates.extend(new_candidates);

            // Remove dangling candidates
            ignored_removals.insert(match_candidate.removal);
        }

        if split_insertions.len() > 1 {
            replace_element_with_sequence(
                &match_candidate.insertion,
                split_insertions.clone(),
                &mut insertions,
            );

            replace_element_with_sequence(
                &DiffAction::Insert(match_candidate.insertion.1.clone()),
                split_insertions
                    .iter()
                    .cloned()
                    .map(|x| DiffAction::Insert(x.1))
                    .collect(),
                &mut d[match_candidate.insertion.0],
            );

            // Re-compute scores of new insertions with all insertions
            let new_candidates = calculate_match_scores(&split_insertions, &removals, a, b);
            match_candidates.extend(new_candidates);

            // Remove dangling candidates
            ignored_insertions.insert(match_candidate.insertion);
        }
    }

    let mut matches = HashMap::new();
    for (insertion, removal) in insertion_matches {
        let insertion_pos = d[insertion.0]
            .iter()
            .position(|i| *i == DiffAction::Insert(insertion.1.clone()));
        let removal_pos = d[removal.0]
            .iter()
            .position(|i| *i == DiffAction::Remove(removal.1.clone()));

        matches.insert(
            (insertion.0, insertion_pos.unwrap()),
            (removal.0, removal_pos.unwrap()),
        );
        matches.insert(
            (removal.0, removal_pos.unwrap()),
            (insertion.0, insertion_pos.unwrap()),
        );
    }

    Ok(MatchedDiffs { diffs: d, matches })
}

#[cfg(test)]
mod test {
    use super::*;

    fn diff_unfailable<U>(a: &[U], b: &[U]) -> Vec<DiffAction>
    where
        U: Eq + PartialEq,
    {
        super::diff(a, b, usize::MAX).expect("failed to execute diff")
    }

    fn match_insertions_removals_unfailable<'a, U, C>(
        d: Vec<Vec<DiffAction>>,
        a: &[C],
        b: &[C],
    ) -> MatchedDiffs
    where
        U: Eq + PartialEq + 'a,
        C: AsRef<[U]> + Sync,
    {
        super::match_insertions_removals(d, a, b, usize::MAX)
            .expect("failed to match insertions and removals")
    }

    #[test]
    fn same() {
        let a = [1, 2, 3, 4];
        let b = [1, 2, 3, 4];
        let d = diff_unfailable(&a, &b);
        assert_eq!(
            d,
            vec![DiffAction::Traverse(Traversal {
                a_idx: 0,
                b_idx: 0,
                length: 4,
            })]
        );

        let MatchedDiffs { mut diffs, matches } =
            match_insertions_removals_unfailable(vec![d.clone()], &[&a], &[&b]);
        let d2 = diffs.pop().unwrap();

        assert_eq!(d, d2);
        assert_eq!(matches.len(), 0);
    }

    #[test]
    fn swap_elements() {
        let a = [1, 2, 3, 4];
        let b = [1, 2, 4, 3];
        let d = diff_unfailable(&a, &b);
        assert_eq!(
            d,
            vec![
                DiffAction::Traverse(Traversal {
                    a_idx: 0,
                    b_idx: 0,
                    length: 2,
                }),
                DiffAction::Remove(Removal {
                    a_idx: 2,
                    length: 1,
                }),
                DiffAction::Traverse(Traversal {
                    a_idx: 3,
                    b_idx: 2,
                    length: 1,
                }),
                DiffAction::Insert(Insertion {
                    b_idx: 3,
                    length: 1
                }),
            ]
        );

        let MatchedDiffs { mut diffs, matches } =
            match_insertions_removals_unfailable(vec![d.clone()], &[&a], &[&b]);
        let d2 = diffs.pop().unwrap();

        assert_eq!(d, d2);
        assert_eq!(matches.len(), 2);
        assert_eq!(matches.get(&(0, 1)), Some(&(0, 3)));
        assert_eq!(matches.get(&(0, 3)), Some(&(0, 1)));
    }

    #[test]
    fn replace_element() {
        let a = [1, 2, 3, 5];
        let b = [1, 2, 4, 5];
        let d = diff_unfailable(&a, &b);
        assert_eq!(
            d,
            vec![
                DiffAction::Traverse(Traversal {
                    a_idx: 0,
                    b_idx: 0,
                    length: 2
                }),
                DiffAction::Remove(Removal {
                    a_idx: 2,
                    length: 1,
                }),
                DiffAction::Insert(Insertion {
                    b_idx: 2,
                    length: 1,
                }),
                DiffAction::Traverse(Traversal {
                    a_idx: 3,
                    b_idx: 3,
                    length: 1
                })
            ]
        );

        let MatchedDiffs { mut diffs, matches } =
            match_insertions_removals_unfailable(vec![d.clone()], &[&a], &[&b]);
        let d2 = diffs.pop().unwrap();

        assert_eq!(d, d2);
        assert_eq!(matches.len(), 0);
    }

    #[test]
    fn replace_grow() {
        let a = [1, 2, 3, 5];
        let b = [1, 2, 4, 4, 4, 4, 5];
        let d = diff_unfailable(&a, &b);
        assert_eq!(
            d,
            vec![
                DiffAction::Traverse(Traversal {
                    a_idx: 0,
                    b_idx: 0,
                    length: 2,
                }),
                DiffAction::Remove(Removal {
                    a_idx: 2,
                    length: 1,
                }),
                DiffAction::Insert(Insertion {
                    b_idx: 2,
                    length: 4,
                }),
                DiffAction::Traverse(Traversal {
                    a_idx: 3,
                    b_idx: 6,
                    length: 1,
                })
            ]
        );

        let MatchedDiffs { mut diffs, matches } =
            match_insertions_removals_unfailable(vec![d.clone()], &[a.as_slice()], &[b.as_slice()]);
        let d2 = diffs.pop().unwrap();

        assert_eq!(d, d2);
        assert_eq!(matches.len(), 0);
    }

    #[test]
    fn remove_all() {
        let a = [1, 2, 3, 4];
        let b = [];
        let d = diff_unfailable(&a, &b);
        assert_eq!(
            d,
            vec![DiffAction::Remove(Removal {
                a_idx: 0,
                length: 4
            })]
        );

        let MatchedDiffs { mut diffs, matches } =
            match_insertions_removals_unfailable(vec![d.clone()], &[a.as_slice()], &[b.as_slice()]);
        let d2 = diffs.pop().unwrap();

        assert_eq!(d, d2);
        assert_eq!(matches.len(), 0);
    }

    #[test]
    fn prepend() {
        let a = [1, 2, 3, 4];
        let b = [0, 0, 1, 2, 3, 4];
        let d = diff_unfailable(&a, &b);
        assert_eq!(
            d,
            vec![
                DiffAction::Insert(Insertion {
                    b_idx: 0,
                    length: 2,
                }),
                DiffAction::Traverse(Traversal {
                    a_idx: 0,
                    b_idx: 2,
                    length: 4
                })
            ]
        );

        let MatchedDiffs { mut diffs, matches } =
            match_insertions_removals_unfailable(vec![d.clone()], &[a.as_slice()], &[b.as_slice()]);
        let d2 = diffs.pop().unwrap();

        assert_eq!(d, d2);
        assert_eq!(matches.len(), 0);
    }

    #[test]
    fn nothing_in_common() {
        let a = [1, 2, 3, 4];
        let b = [5, 6, 7, 8];
        let d = diff_unfailable(&a, &b);
        assert_eq!(
            d,
            vec![
                DiffAction::Remove(Removal {
                    a_idx: 0,
                    length: 4,
                }),
                DiffAction::Insert(Insertion {
                    b_idx: 0,
                    length: 4,
                })
            ]
        );

        let MatchedDiffs { mut diffs, matches } =
            match_insertions_removals_unfailable(vec![d.clone()], &[&a], &[&b]);
        let d2 = diffs.pop().unwrap();

        assert_eq!(d, d2);
        assert_eq!(matches.len(), 0);
    }

    #[test]
    fn block_move() {
        let a = [1, 2, 3, 4, 5, 6, 7, 8];
        let b = [1, 2, 3, 6, 7, 8, 4, 5];
        let d = diff_unfailable(&a, &b);
        assert_eq!(
            d,
            vec![
                DiffAction::Traverse(Traversal {
                    a_idx: 0,
                    b_idx: 0,
                    length: 3
                }),
                DiffAction::Remove(Removal {
                    a_idx: 3,
                    length: 2,
                }),
                DiffAction::Traverse(Traversal {
                    a_idx: 5,
                    b_idx: 3,
                    length: 3
                }),
                DiffAction::Insert(Insertion {
                    b_idx: 6,
                    length: 2,
                })
            ]
        );

        let MatchedDiffs { mut diffs, matches } =
            match_insertions_removals_unfailable(vec![d.clone()], &[&a], &[&b]);
        let d2 = diffs.pop().unwrap();

        assert_eq!(d, d2);
        assert_eq!(matches.len(), 2);
        assert_eq!(matches.get(&(0, 1)), Some(&(0, 3)));
        assert_eq!(matches.get(&(0, 3)), Some(&(0, 1)));
    }

    #[test]
    fn block_move_change_order() {
        let a = [1, 2, 3, 4, 5, 6, 7, 8];
        let b = [1, 2, 3, 6, 7, 8, 5, 4];
        let d = diff_unfailable(&a, &b);
        assert_eq!(
            d,
            vec![
                DiffAction::Traverse(Traversal {
                    a_idx: 0,
                    b_idx: 0,
                    length: 3
                }),
                DiffAction::Remove(Removal {
                    a_idx: 3,
                    length: 2,
                }),
                DiffAction::Traverse(Traversal {
                    a_idx: 5,
                    b_idx: 3,
                    length: 3
                }),
                DiffAction::Insert(Insertion {
                    b_idx: 6,
                    length: 2,
                })
            ]
        );

        let MatchedDiffs { mut diffs, matches } =
            match_insertions_removals_unfailable(vec![d.clone()], &[&a], &[&b]);
        let d2 = diffs.pop().unwrap();

        assert_eq!(
            d2,
            vec![
                DiffAction::Traverse(Traversal {
                    a_idx: 0,
                    b_idx: 0,
                    length: 3
                }),
                DiffAction::Remove(Removal {
                    a_idx: 3,
                    length: 1,
                }),
                DiffAction::Remove(Removal {
                    a_idx: 4,
                    length: 1,
                }),
                DiffAction::Traverse(Traversal {
                    a_idx: 5,
                    b_idx: 3,
                    length: 3
                }),
                DiffAction::Insert(Insertion {
                    b_idx: 6,
                    length: 1,
                }),
                DiffAction::Insert(Insertion {
                    b_idx: 7,
                    length: 1,
                })
            ]
        );

        assert_eq!(matches.len(), 4);
        assert_eq!(matches.get(&(0, 1)), Some(&(0, 5)));
        assert_eq!(matches.get(&(0, 5)), Some(&(0, 1)));
        assert_eq!(matches.get(&(0, 2)), Some(&(0, 4)));
        assert_eq!(matches.get(&(0, 4)), Some(&(0, 2)));
    }

    #[test]
    fn remove_3_move_2_1() {
        let a = [1, 2, 3, 4, 5, 6, 7, 8];
        let b = [1, 4, 2, 3, 7, 8, 5, 6];
        let d = diff_unfailable(&a, &b);
        assert_eq!(
            d,
            vec![
                DiffAction::Traverse(Traversal {
                    a_idx: 0,
                    b_idx: 0,
                    length: 1
                }),
                DiffAction::Insert(Insertion {
                    b_idx: 1,
                    length: 1,
                }),
                DiffAction::Traverse(Traversal {
                    a_idx: 1,
                    b_idx: 2,
                    length: 2
                }),
                DiffAction::Remove(Removal {
                    a_idx: 3,
                    length: 3
                }),
                DiffAction::Traverse(Traversal {
                    a_idx: 6,
                    b_idx: 4,
                    length: 2
                }),
                DiffAction::Insert(Insertion {
                    b_idx: 6,
                    length: 2,
                })
            ]
        );

        let MatchedDiffs { mut diffs, matches } =
            match_insertions_removals_unfailable(vec![d.clone()], &[&a], &[&b]);
        let d2 = diffs.pop().unwrap();

        assert_eq!(
            d2,
            vec![
                DiffAction::Traverse(Traversal {
                    a_idx: 0,
                    b_idx: 0,
                    length: 1
                }),
                DiffAction::Insert(Insertion {
                    b_idx: 1,
                    length: 1,
                }),
                DiffAction::Traverse(Traversal {
                    a_idx: 1,
                    b_idx: 2,
                    length: 2
                }),
                DiffAction::Remove(Removal {
                    a_idx: 3,
                    length: 1
                }),
                DiffAction::Remove(Removal {
                    a_idx: 4,
                    length: 2
                }),
                DiffAction::Traverse(Traversal {
                    a_idx: 6,
                    b_idx: 4,
                    length: 2
                }),
                DiffAction::Insert(Insertion {
                    b_idx: 6,
                    length: 2,
                })
            ]
        );

        assert_eq!(matches.len(), 4);
        assert_eq!(matches.get(&(0, 1)), Some(&(0, 3)));
        assert_eq!(matches.get(&(0, 3)), Some(&(0, 1)));
        assert_eq!(matches.get(&(0, 4)), Some(&(0, 6)));
        assert_eq!(matches.get(&(0, 6)), Some(&(0, 4)));
    }

    #[test]
    fn remove_3_move_1_1() {
        let a = [1, 2, 3, 4, 5, 6, 7, 8];
        let b = [4, 5, 6, 2, 1, 7, 8];
        let d = diff_unfailable(&a, &b);
        assert_eq!(
            d,
            vec![
                DiffAction::Remove(Removal {
                    a_idx: 0,
                    length: 3
                }),
                DiffAction::Traverse(Traversal {
                    a_idx: 3,
                    b_idx: 0,
                    length: 3
                }),
                DiffAction::Insert(Insertion {
                    b_idx: 3,
                    length: 2
                }),
                DiffAction::Traverse(Traversal {
                    a_idx: 6,
                    b_idx: 5,
                    length: 2
                }),
            ]
        );

        let MatchedDiffs { mut diffs, matches } =
            match_insertions_removals_unfailable(vec![d.clone()], &[a.as_slice()], &[b.as_slice()]);
        let d2 = diffs.pop().unwrap();

        assert_eq!(
            d2,
            vec![
                DiffAction::Remove(Removal {
                    a_idx: 0,
                    length: 1
                }),
                DiffAction::Remove(Removal {
                    a_idx: 1,
                    length: 1
                }),
                DiffAction::Remove(Removal {
                    a_idx: 2,
                    length: 1
                }),
                DiffAction::Traverse(Traversal {
                    a_idx: 3,
                    b_idx: 0,
                    length: 3
                }),
                DiffAction::Insert(Insertion {
                    b_idx: 3,
                    length: 1
                }),
                DiffAction::Insert(Insertion {
                    b_idx: 4,
                    length: 1
                }),
                DiffAction::Traverse(Traversal {
                    a_idx: 6,
                    b_idx: 5,
                    length: 2
                }),
            ]
        );

        assert_eq!(matches.len(), 4);
        assert_eq!(matches.get(&(0, 0)), Some(&(0, 5)));
        assert_eq!(matches.get(&(0, 5)), Some(&(0, 0)));
        assert_eq!(matches.get(&(0, 1)), Some(&(0, 4)));
        assert_eq!(matches.get(&(0, 4)), Some(&(0, 1)));
    }

    #[test]
    fn remove_2_move_1_insert_1_move_1() {
        let a = [1, 2, 3, 4, 5, 6, 7, 8];
        let b = [1, 2, 5, 6, 3, 1, 4, 7, 8];
        let d = diff_unfailable(&a, &b);
        assert_eq!(
            d,
            vec![
                DiffAction::Traverse(Traversal {
                    a_idx: 0,
                    b_idx: 0,
                    length: 2
                }),
                DiffAction::Remove(Removal {
                    a_idx: 2,
                    length: 2
                }),
                DiffAction::Traverse(Traversal {
                    a_idx: 4,
                    b_idx: 2,
                    length: 2
                }),
                DiffAction::Insert(Insertion {
                    b_idx: 4,
                    length: 3
                }),
                DiffAction::Traverse(Traversal {
                    a_idx: 6,
                    b_idx: 7,
                    length: 2
                }),
            ]
        );

        let MatchedDiffs { mut diffs, matches } =
            match_insertions_removals_unfailable(vec![d.clone()], &[a.as_slice()], &[b.as_slice()]);
        let d2 = diffs.pop().unwrap();

        assert_eq!(
            d2,
            vec![
                DiffAction::Traverse(Traversal {
                    a_idx: 0,
                    b_idx: 0,
                    length: 2
                }),
                DiffAction::Remove(Removal {
                    a_idx: 2,
                    length: 1
                }),
                DiffAction::Remove(Removal {
                    a_idx: 3,
                    length: 1
                }),
                DiffAction::Traverse(Traversal {
                    a_idx: 4,
                    b_idx: 2,
                    length: 2
                }),
                DiffAction::Insert(Insertion {
                    b_idx: 4,
                    length: 1
                }),
                DiffAction::Insert(Insertion {
                    b_idx: 5,
                    length: 1
                }),
                DiffAction::Insert(Insertion {
                    b_idx: 6,
                    length: 1
                }),
                DiffAction::Traverse(Traversal {
                    a_idx: 6,
                    b_idx: 7,
                    length: 2
                }),
            ]
        );

        assert_eq!(matches.len(), 4);
        assert_eq!(matches.get(&(0, 1)), Some(&(0, 4)));
        assert_eq!(matches.get(&(0, 4)), Some(&(0, 1)));
        assert_eq!(matches.get(&(0, 2)), Some(&(0, 6)));
        assert_eq!(matches.get(&(0, 6)), Some(&(0, 2)));
    }

    #[test]
    fn cross_file_move() {
        let a1 = [1, 2, 3, 4, 5, 6, 7, 8];
        let b1 = [1, 2, 3, 4, 7, 8];
        let a2 = [1, 2, 3];
        let b2 = [1, 2, 5, 6, 3];
        let diffs = [diff_unfailable(&a1, &b1), diff_unfailable(&a2, &b2)];
        assert_eq!(
            diffs,
            [
                vec![
                    DiffAction::Traverse(Traversal {
                        a_idx: 0,
                        b_idx: 0,
                        length: 4
                    }),
                    DiffAction::Remove(Removal {
                        a_idx: 4,
                        length: 2
                    }),
                    DiffAction::Traverse(Traversal {
                        a_idx: 6,
                        b_idx: 4,
                        length: 2
                    }),
                ],
                vec![
                    DiffAction::Traverse(Traversal {
                        a_idx: 0,
                        b_idx: 0,
                        length: 2
                    }),
                    DiffAction::Insert(Insertion {
                        b_idx: 2,
                        length: 2
                    }),
                    DiffAction::Traverse(Traversal {
                        a_idx: 2,
                        b_idx: 4,
                        length: 1
                    }),
                ]
            ]
        );

        let MatchedDiffs { diffs, matches } = match_insertions_removals_unfailable(
            diffs.to_vec(),
            &[a1.as_slice(), &a2],
            &[&b1, &b2],
        );

        assert_eq!(
            diffs,
            [
                vec![
                    DiffAction::Traverse(Traversal {
                        a_idx: 0,
                        b_idx: 0,
                        length: 4
                    }),
                    DiffAction::Remove(Removal {
                        a_idx: 4,
                        length: 2
                    }),
                    DiffAction::Traverse(Traversal {
                        a_idx: 6,
                        b_idx: 4,
                        length: 2
                    }),
                ],
                vec![
                    DiffAction::Traverse(Traversal {
                        a_idx: 0,
                        b_idx: 0,
                        length: 2
                    }),
                    DiffAction::Insert(Insertion {
                        b_idx: 2,
                        length: 2
                    }),
                    DiffAction::Traverse(Traversal {
                        a_idx: 2,
                        b_idx: 4,
                        length: 1
                    }),
                ]
            ]
        );

        assert_eq!(matches.len(), 2);
        assert_eq!(matches.get(&(0, 1)), Some(&(1, 1)));
        assert_eq!(matches.get(&(1, 1)), Some(&(0, 1)));
    }

    #[test]
    fn test_myers_trace() {
        let mut trace = FullMyersTrace::new(2, 3, usize::MAX).expect("failed to generate trace");

        assert_eq!(trace.data.len(), 6 + 5 + 4 + 3 + 2 + 1);

        let mut val = 1;
        for d in 0..=5 {
            for k in (-d..d).step_by(2) {
                *trace.get_mut(d, k) = val;
                val += 1;
            }
        }

        let mut val = 1;
        for d in 0..=5 {
            for k in (-d..d).step_by(2) {
                assert_eq!(*trace.get_mut(d, k), val);
                val += 1;
            }
        }
    }
}
