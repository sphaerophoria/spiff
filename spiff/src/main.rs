use anyhow::{Context, Result};
use memmap2::Mmap;

use std::{
    fs::File,
    io::Write,
};

fn main() -> Result<()> {
    let path1 = std::env::args().nth(1).context("Path for file 1 not provided")?;
    let path2 = std::env::args().nth(2).context("Path for file 2 not provided")?;

    let file1 = File::open(path1).context("Failed to open file 1")?;
    let file2 = File::open(path2).context("Failed to open file 2")?;

    let map1 = unsafe { Mmap::map(&file1) }.context("Failed to map file 1")?;
    let map2 = unsafe { Mmap::map(&file2) }.context("Failed to map file 2")?;

    let lines1: Vec<&[u8]> = map1.split(|x| *x == b'\n').collect();
    let lines2: Vec<&[u8]> = map2.split(|x| *x == b'\n').collect();

    let diff = libdiff::diff(&lines1, &lines2);
    let mut idx = 0;
    for action in diff.actions() {
        match action {
            libdiff::DiffAction::Traverse(n) => {
                for i in idx..idx+n {
                    print!(" ");
                    std::io::stdout().write_all(lines1[i]).context("Failed to write to stdout")?;
                    print!("\n");
                }
                idx = idx + n;
            }
            libdiff::DiffAction::Remove(n) => {
                print!("{}", ansi_term::Color::Red.prefix());
                for i in idx..idx+n {
                    print!("-");
                    std::io::stdout().write_all(lines1[i]).context("Failed to write to stdout")?;
                    print!("\n");
                }
                print!("{}", ansi_term::Color::Red.suffix());
                idx = idx + n;
            }
            libdiff::DiffAction::Insert{ starting_index, num_items} => {
                print!("{}", ansi_term::Color::Green.prefix());
                for i in *starting_index..*starting_index+*num_items {
                    print!("+");
                    std::io::stdout().write_all(lines2[i]).context("Failed to write to stdout")?;
                    print!("\n");
                }
                print!("{}", ansi_term::Color::Green.suffix());
            }
        }

    }

    Ok(())
}
