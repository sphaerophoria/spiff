use anyhow::{Context, Result};
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

pub fn open_file<P: AsRef<Path>>(path: P) -> Result<Mmap> {
    let file = File::open(path).context("Failed to open file")?;
    let mmap = unsafe { Mmap::map(&file) }.context("Failed to map file")?;
    Ok(mmap)
}

pub fn buf_to_lines(buf: &[u8]) -> Result<Vec<&str>> {
    let s = std::str::from_utf8(buf).context("Failed to parse string")?;

    Ok(s.lines().collect())
}
