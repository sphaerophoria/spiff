use anyhow::{Context, Result};

use libdiff::MatchedDiffs;

fn main() -> Result<()> {
    let path1 = std::env::args()
        .nth(1)
        .context("Path for file 1 not provided")?;
    let path2 = std::env::args()
        .nth(2)
        .context("Path for file 2 not provided")?;

    let file1 = spiff::open_file(path1).unwrap();
    let file2 = spiff::open_file(path2).unwrap();

    let lines1 = spiff::buf_to_lines(&file1).unwrap();
    let lines2 = spiff::buf_to_lines(&file2).unwrap();

    let diff = libdiff::diff(&lines1, &lines2);
    let MatchedDiffs { mut diffs, matches } =
        libdiff::match_insertions_removals([diff].to_vec(), &[&lines1], &[&lines2]);
    let diff = diffs.pop().unwrap();
    let matches = matches
        .into_iter()
        .map(|((idx, x), (idx2, y))| {
            assert_eq!(idx, 0);
            assert_eq!(idx, idx2);
            (x, y)
        })
        .collect::<std::collections::HashMap<usize, usize>>();
    for (action_idx, action) in diff.into_iter().enumerate() {
        match action {
            libdiff::DiffAction::Traverse(traversal) => {
                for line in lines1.iter().skip(traversal.a_idx).take(traversal.length) {
                    print!(" ");
                    println!("{}", line);
                }
            }
            libdiff::DiffAction::Remove(removal) => {
                let colour = if matches.contains_key(&action_idx) {
                    ansi_term::Colour::Yellow
                } else {
                    ansi_term::Colour::Red
                };
                print!("{}", colour.prefix());
                for line in lines1.iter().skip(removal.a_idx).take(removal.length) {
                    print!("-");
                    println!("{}", line);
                }
                print!("{}", colour.suffix());
            }
            libdiff::DiffAction::Insert(insertion) => {
                let colour = if matches.contains_key(&action_idx) {
                    ansi_term::Colour::Blue
                } else {
                    ansi_term::Colour::Green
                };
                print!("{}", colour.prefix());

                for line in lines2.iter().skip(insertion.b_idx).take(insertion.length) {
                    print!("+");
                    println!("{}", line);
                }

                print!("{}", colour.suffix());
            }
        }
    }

    Ok(())
}
