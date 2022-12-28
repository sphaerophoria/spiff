use anyhow::{Context, Result};
use memmap2::Mmap;

use libdiff::MatchedDiff;
use std::{fs::File, io::Write};

fn main() -> Result<()> {
    let path1 = std::env::args()
        .nth(1)
        .context("Path for file 1 not provided")?;
    let path2 = std::env::args()
        .nth(2)
        .context("Path for file 2 not provided")?;

    let file1 = File::open(path1).context("Failed to open file 1")?;
    let file2 = File::open(path2).context("Failed to open file 2")?;

    let map1 = unsafe { Mmap::map(&file1) }.context("Failed to map file 1")?;
    let map2 = unsafe { Mmap::map(&file2) }.context("Failed to map file 2")?;

    let lines1: Vec<&[u8]> = map1.split(|x| *x == b'\n').collect();
    let lines2: Vec<&[u8]> = map2.split(|x| *x == b'\n').collect();

    let diff = libdiff::diff(&lines1, &lines2);
    let MatchedDiff { diff, matches } = libdiff::match_insertions_removals(diff, &lines1, &lines2);
    for (action_idx, action) in diff.into_iter().enumerate() {
        match action {
            libdiff::DiffAction::Traverse(traversal) => {
                for line in lines1.iter().skip(traversal.a_idx).take(traversal.length) {
                    print!(" ");
                    std::io::stdout()
                        .write_all(line)
                        .context("Failed to write to stdout")?;
                    println!();
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
                    std::io::stdout()
                        .write_all(line)
                        .context("Failed to write to stdout")?;
                    println!();
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
                    std::io::stdout()
                        .write_all(line)
                        .context("Failed to write to stdout")?;
                    println!();
                }

                print!("{}", colour.suffix());
            }
        }
    }

    Ok(())
}
