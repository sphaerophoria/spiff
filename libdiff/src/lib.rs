use std::{
    cmp::{Eq, PartialEq},
    collections::VecDeque,
    collections::{BinaryHeap, HashMap, HashSet},
};

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
#[derive(Clone, Default, Debug, Eq, PartialEq)]
struct DiffBuilder {
    seq: Vec<DiffAction>,
}

impl DiffBuilder {
    /// Add a traversed item to the diff
    ///
    /// If the last element is already a traversal its length will be amended
    fn push_traversed_item(&mut self, a_idx: usize, b_idx: usize) {
        if let Some(DiffAction::Traverse(traversal)) = self.seq.last_mut() {
            if traversal.a_idx == a_idx - traversal.length
                && traversal.b_idx == b_idx - traversal.length
            {
                traversal.length += 1;
                return;
            }
        }

        self.seq.push(DiffAction::Traverse(Traversal {
            a_idx,
            b_idx,
            length: 1,
        }));
    }

    /// Add an inserted item to the diff
    ///
    /// If the last element is already an insertion its length will be amended
    fn push_insertion(&mut self, b_idx: usize) {
        if let Some(DiffAction::Insert(insertion)) = self.seq.last_mut() {
            if insertion.b_idx == b_idx - insertion.length {
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
            if removal.a_idx == a_idx - removal.length {
                removal.length += 1;
                return;
            }
        }

        self.seq
            .push(DiffAction::Remove(Removal { a_idx, length: 1 }));
    }
}

/// Find a sequence of actions that applied to a results in b
pub fn diff<U>(a: &[U], b: &[U]) -> Vec<DiffAction>
where
    U: Eq + PartialEq,
{
    let mut visited = HashSet::new();
    let mut tails = VecDeque::new();
    tails.push_back((0, 0, DiffBuilder::default()));

    // Myers diff algorithm
    // BFS where traversals are free
    while let Some((a_idx, b_idx, diff)) = tails.pop_front() {
        // If we've already been here it means we have a shorter path to this node
        if visited.contains(&(a_idx, b_idx)) {
            continue;
        }
        visited.insert((a_idx, b_idx));

        // Reached the end of the graph, we're done
        if a_idx == a.len() && b_idx == b.len() {
            return diff.seq;
        }

        // Traversals are free, so we traverse them greedily
        {
            let mut new_diff = diff.clone();
            let mut a_idx = a_idx;
            let mut b_idx = b_idx;
            while a_idx < a.len() && b_idx < b.len() && a[a_idx] == b[b_idx] {
                new_diff.push_traversed_item(a_idx, b_idx);
                visited.insert((a_idx, b_idx));
                a_idx += 1;
                b_idx += 1;
            }
            tails.push_back((a_idx, b_idx, new_diff));
        }

        // And add on the removal/insertions
        if a.len() > a_idx {
            let mut new_diff = diff.clone();
            new_diff.push_removal(a_idx);
            tails.push_back((a_idx + 1, b_idx, new_diff));
        }

        if b.len() > b_idx {
            let mut new_diff = diff.clone();
            new_diff.push_insertion(b_idx);
            tails.push_back((a_idx, b_idx + 1, new_diff));
        }
    }

    unreachable!();
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
fn calculate_match_score<U>(insertion: &Insertion, removal: &Removal, a: &[U], b: &[U]) -> usize
where
    U: Eq + PartialEq,
{
    let removal_content = &a[removal.a_idx..removal.a_idx + removal.length];
    (insertion.b_idx..insertion.b_idx + insertion.length).fold(0, |acc, b_idx| {
        let insertion_item = &b[b_idx];
        if removal_content.contains(insertion_item) {
            acc + 1
        } else {
            acc
        }
    })
}

/// Splits input insertion and removal to get segments with either a 100% match score or a 0% match
/// score
#[allow(clippy::type_complexity)]
fn split_insertion_removal_pair<U, C>(
    insertion: &(usize, Insertion),
    removal: &(usize, Removal),
    a: &[C],
    b: &[C],
) -> (Vec<(usize, Insertion)>, Vec<(usize, Removal)>)
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
    let insertion_removal_diff = diff(removal_content, insertion_content);

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

    (output_insertions, output_removals)
}

fn replace_element_with_sequence<U: Eq + PartialEq>(elem: &U, seq: Vec<U>, vec: &mut Vec<U>) {
    let position = vec.iter().position(|removal| removal == elem).unwrap();
    vec.splice(position..position + 1, seq);
}

#[derive(Eq, PartialEq, PartialOrd, Ord, Debug)]
struct MatchCandidate {
    // Order of elements important for ordering
    score: usize,
    insertion: (usize, Insertion),
    removal: (usize, Removal),
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
    C: AsRef<[U]>,
{
    let mut match_candidates = BinaryHeap::new();

    for insertion in insertions {
        for removal in removals {
            let score = calculate_match_score(
                &insertion.1,
                &removal.1,
                a[removal.0].as_ref(),
                b[insertion.0].as_ref(),
            );

            if score > 0 {
                match_candidates.push(MatchCandidate {
                    insertion: insertion.clone(),
                    removal: removal.clone(),
                    score,
                });
            }
        }
    }

    match_candidates
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
) -> MatchedDiffs
where
    U: Eq + PartialEq + 'a,
    C: AsRef<[U]>,
{
    let (mut insertions, mut removals) = insertions_removals_from_actions(&d);

    let mut match_candidates = calculate_match_scores(&insertions, &removals, a, b);

    let mut insertion_matches = HashMap::new();
    let mut removal_matches = HashSet::new();

    while let Some(match_candidate) = match_candidates.pop() {
        if insertion_matches.contains_key(&match_candidate.insertion) {
            continue;
        }

        if removal_matches.contains(&match_candidate.removal) {
            continue;
        }

        let (mut split_insertions, mut split_removals) = split_insertion_removal_pair(
            &match_candidate.insertion,
            &match_candidate.removal,
            a,
            b,
        );

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
            for removal in split_removals {
                for insertion in &insertions {
                    let score = calculate_match_score(
                        &insertion.1,
                        &removal.1,
                        a[removal.0].as_ref(),
                        b[insertion.0].as_ref(),
                    );

                    if score > 0 {
                        match_candidates.push(MatchCandidate {
                            insertion: insertion.clone(),
                            removal: removal.clone(),
                            score,
                        });
                    }
                }
            }

            // Remove dangling candidates
            match_candidates = match_candidates
                .into_iter()
                .filter(|candidate| candidate.removal != match_candidate.removal)
                .collect();
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
            for insertion in split_insertions {
                for removal in &removals {
                    let score = calculate_match_score(
                        &insertion.1,
                        &removal.1,
                        a[removal.0].as_ref(),
                        b[insertion.0].as_ref(),
                    );

                    if score > 0 {
                        match_candidates.push(MatchCandidate {
                            insertion: insertion.clone(),
                            removal: removal.clone(),
                            score,
                        });
                    }
                }
            }

            // Remove dangling candidates
            match_candidates = match_candidates
                .into_iter()
                .filter(|candidate| candidate.insertion != match_candidate.insertion)
                .collect();
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

    MatchedDiffs { diffs: d, matches }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn same() {
        let a = [1, 2, 3, 4];
        let b = [1, 2, 3, 4];
        let d = diff(&a, &b);
        assert_eq!(
            d,
            vec![DiffAction::Traverse(Traversal {
                a_idx: 0,
                b_idx: 0,
                length: 4,
            })]
        );

        let MatchedDiffs { mut diffs, matches } =
            match_insertions_removals(vec![d.clone()], &[&a], &[&b]);
        let d2 = diffs.pop().unwrap();

        assert_eq!(d, d2);
        assert_eq!(matches.len(), 0);
    }

    #[test]
    fn swap_elements() {
        let a = [1, 2, 3, 4];
        let b = [1, 2, 4, 3];
        let d = diff(&a, &b);
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
            match_insertions_removals(vec![d.clone()], &[&a], &[&b]);
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
        let d = diff(&a, &b);
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
            match_insertions_removals(vec![d.clone()], &[&a], &[&b]);
        let d2 = diffs.pop().unwrap();

        assert_eq!(d, d2);
        assert_eq!(matches.len(), 0);
    }

    #[test]
    fn replace_grow() {
        let a = [1, 2, 3, 5];
        let b = [1, 2, 4, 4, 4, 4, 5];
        let d = diff(&a, &b);
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
            match_insertions_removals(vec![d.clone()], &[a.as_slice()], &[b.as_slice()]);
        let d2 = diffs.pop().unwrap();

        assert_eq!(d, d2);
        assert_eq!(matches.len(), 0);
    }

    #[test]
    fn remove_all() {
        let a = [1, 2, 3, 4];
        let b = [];
        let d = diff(&a, &b);
        assert_eq!(
            d,
            vec![DiffAction::Remove(Removal {
                a_idx: 0,
                length: 4
            })]
        );

        let MatchedDiffs { mut diffs, matches } =
            match_insertions_removals(vec![d.clone()], &[a.as_slice()], &[b.as_slice()]);
        let d2 = diffs.pop().unwrap();

        assert_eq!(d, d2);
        assert_eq!(matches.len(), 0);
    }

    #[test]
    fn prepend() {
        let a = [1, 2, 3, 4];
        let b = [0, 0, 1, 2, 3, 4];
        let d = diff(&a, &b);
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
            match_insertions_removals(vec![d.clone()], &[a.as_slice()], &[b.as_slice()]);
        let d2 = diffs.pop().unwrap();

        assert_eq!(d, d2);
        assert_eq!(matches.len(), 0);
    }

    #[test]
    fn nothing_in_common() {
        let a = [1, 2, 3, 4];
        let b = [5, 6, 7, 8];
        let d = diff(&a, &b);
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
            match_insertions_removals(vec![d.clone()], &[&a], &[&b]);
        let d2 = diffs.pop().unwrap();

        assert_eq!(d, d2);
        assert_eq!(matches.len(), 0);
    }

    #[test]
    fn block_move() {
        let a = [1, 2, 3, 4, 5, 6, 7, 8];
        let b = [1, 2, 3, 6, 7, 8, 4, 5];
        let d = diff(&a, &b);
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
            match_insertions_removals(vec![d.clone()], &[&a], &[&b]);
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
        let d = diff(&a, &b);
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
            match_insertions_removals(vec![d.clone()], &[&a], &[&b]);
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
        let d = diff(&a, &b);
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
            match_insertions_removals(vec![d.clone()], &[&a], &[&b]);
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
        let d = diff(&a, &b);
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
            match_insertions_removals(vec![d.clone()], &[a.as_slice()], &[b.as_slice()]);
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
        let d = diff(&a, &b);
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
            match_insertions_removals(vec![d.clone()], &[a.as_slice()], &[b.as_slice()]);
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
        let diffs = [diff(&a1, &b1), diff(&a2, &b2)];
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

        let MatchedDiffs { diffs, matches } =
            match_insertions_removals(diffs.to_vec(), &[a1.as_slice(), &a2], &[&b1, &b2]);

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
}
