use std::{
    cmp::{Eq, PartialEq},
    collections::HashSet,
    collections::VecDeque,
};

// T may be line, or character
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DiffAction {
    Traverse(usize),
    Insert {
        num_items: usize,
        starting_index: usize,
    },
    Remove(usize),
}

#[derive(Clone, Default, Debug, Eq, PartialEq)]
pub struct Diff {
    seq: Vec<DiffAction>,
}

impl Diff {
    pub fn actions(&self) -> &[DiffAction] {
        &self.seq
    }

    fn push_traversed_item(&mut self) {
        match self.seq.last_mut() {
            Some(DiffAction::Traverse(n)) => *n += 1,
            _ => {
                self.seq.push(DiffAction::Traverse(1));
            }
        }
    }

    fn push_insertion(&mut self, b_idx: usize) {
        match self.seq.last_mut() {
            Some(DiffAction::Insert { num_items, .. }) => *num_items += 1,
            _ => {
                self.seq.push(DiffAction::Insert {
                    num_items: 1,
                    starting_index: b_idx,
                });
            }
        }
    }

    fn push_removal(&mut self) {
        match self.seq.last_mut() {
            Some(DiffAction::Remove(n)) => *n += 1,
            _ => {
                self.seq.push(DiffAction::Remove(1));
            }
        }
    }
}

pub fn diff<U>(a: &[U], b: &[U]) -> Diff
where
    U: Eq + PartialEq,
{
    let mut visited = HashSet::new();
    let mut tails = VecDeque::new();
    tails.push_back((0, 0, Diff::default()));

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
            return diff;
        }

        // Traversals are free, so greedy search em
        {
            let mut new_diff = diff.clone();
            let mut a_idx = a_idx;
            let mut b_idx = b_idx;
            while a_idx < a.len() && b_idx < b.len() && a[a_idx] == b[b_idx] {
                new_diff.push_traversed_item();
                visited.insert((a_idx, b_idx));
                a_idx += 1;
                b_idx += 1;
            }
            tails.push_back((a_idx, b_idx, new_diff));
        }

        // And add on the removal/insertions
        if a.len() > a_idx {
            let mut new_diff = diff.clone();
            new_diff.push_removal();
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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn same() {
        let d = diff(&[1, 2, 3, 4], &[1, 2, 3, 4]);
        assert_eq!(
            d,
            Diff {
                seq: vec![DiffAction::Traverse(4)]
            }
        );
    }

    #[test]
    fn swap() {
        let d = diff(&[1, 2, 3, 4], &[1, 2, 4, 3]);
        assert_eq!(
            d,
            Diff {
                seq: vec![
                    DiffAction::Traverse(2),
                    DiffAction::Remove(1),
                    DiffAction::Traverse(1),
                    DiffAction::Insert {
                        num_items: 1,
                        starting_index: 3
                    },
                ]
            }
        );
    }

    #[test]
    fn change() {
        let d = diff(&[1, 2, 3, 5], &[1, 2, 4, 5]);
        assert_eq!(
            d,
            Diff {
                seq: vec![
                    DiffAction::Traverse(2),
                    DiffAction::Remove(1),
                    DiffAction::Insert {
                        num_items: 1,
                        starting_index: 2
                    },
                    DiffAction::Traverse(1)
                ]
            }
        );
    }

    #[test]
    fn swap_grow() {
        let d = diff(&[1, 2, 3, 5], &[1, 2, 4, 4, 4, 4, 5]);
        assert_eq!(
            d,
            Diff {
                seq: vec![
                    DiffAction::Traverse(2),
                    DiffAction::Remove(1),
                    DiffAction::Insert {
                        num_items: 4,
                        starting_index: 2
                    },
                    DiffAction::Traverse(1)
                ]
            }
        );
    }

    #[test]
    fn remove_all() {
        let d = diff(&[1, 2, 3, 4], &[]);
        assert_eq!(
            d,
            Diff {
                seq: vec![DiffAction::Remove(4),]
            }
        );
    }

    #[test]
    fn prepend() {
        let d = diff(&[1, 2, 3, 4], &[0, 0, 1, 2, 3, 4]);
        assert_eq!(
            d,
            Diff {
                seq: vec![
                    DiffAction::Insert {
                        starting_index: 0,
                        num_items: 2,
                    },
                    DiffAction::Traverse(4)
                ]
            }
        );
    }

    #[test]
    fn nothing_in_common() {
        let d = diff(&[1, 2, 3, 4], &[5, 6, 7, 8]);
        assert_eq!(
            d,
            Diff {
                seq: vec![
                    DiffAction::Remove(4),
                    DiffAction::Insert {
                        starting_index: 0,
                        num_items: 4,
                    }
                ]
            }
        );
    }
}
