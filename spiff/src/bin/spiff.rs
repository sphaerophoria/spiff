use anyhow::{Context, Result};
use spiff::{DiffCollectionProcessor, SegmentPurpose};

fn main() -> Result<()> {
    let path1 = std::env::args()
        .nth(1)
        .context("Path for file 1 not provided")?;
    let path2 = std::env::args()
        .nth(2)
        .context("Path for file 2 not provided")?;

    let spiff::Contents {
        content_a,
        content_b,
        labels,
    } = spiff::contents_from_roots(path1, path2)?;

    let request_processor = DiffCollectionProcessor::new(&content_a, &content_b, &labels)?;

    let processed_diff_collection = request_processor.generate();

    for diff in processed_diff_collection.processed_diffs {
        println!("{}", diff.label);

        // Track "columns" manually since we have no vertical layout
        let mut line_number_lines = diff.line_numbers.lines();

        if let Some(first_line) = line_number_lines.next() {
            print!("{} ", first_line);
        }

        let mut segment_iter = diff.display_info.into_iter().peekable();
        while let Some((range, purpose)) = segment_iter.next() {
            let is_last_segment = segment_iter.peek().is_none();
            let segment_text = &diff.processed_diff[range];
            let segment_ends_in_newline =
                segment_text.ends_with("\r\n") || segment_text.ends_with('\n');

            let style = match purpose {
                SegmentPurpose::Context => None,
                SegmentPurpose::Addition => Some(ansi_term::Colour::Green.normal()),
                SegmentPurpose::Removal => Some(ansi_term::Colour::Red.normal()),
                SegmentPurpose::MoveFrom(_) => Some(ansi_term::Colour::Yellow.normal()),
                SegmentPurpose::MoveTo(_) => Some(ansi_term::Colour::Blue.normal()),
                SegmentPurpose::MatchSearch => {
                    Some(ansi_term::Style::new().on(ansi_term::Colour::Blue))
                }
                SegmentPurpose::TrailingWhitespace => {
                    Some(ansi_term::Style::new().on(ansi_term::Colour::Red))
                }
                SegmentPurpose::Failed => {
                    Some(ansi_term::Colour::Red.normal())
                }
            };

            let mut lines = segment_text.lines().peekable();

            while let Some(line) = lines.next() {
                if let Some(style) = style {
                    print!("{}", style.paint(line));
                } else {
                    print!("{}", line);
                }

                if lines.peek().is_some() || segment_ends_in_newline {
                    println!();
                    if !is_last_segment {
                        print!("{}", line_number_lines.next().unwrap());
                    }
                }
            }
        }
        println!();
    }

    Ok(())
}
