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

        for (range, info) in diff.display_info {
            let color = match info.purpose {
                SegmentPurpose::Context => None,
                SegmentPurpose::Addition => Some(ansi_term::Colour::Green),
                SegmentPurpose::Removal => Some(ansi_term::Colour::Red),
                SegmentPurpose::MoveFrom(_) => Some(ansi_term::Colour::Yellow),
                SegmentPurpose::MoveTo(_) => Some(ansi_term::Colour::Blue),
            };

            let start_color = || {
                if let Some(color) = color.as_ref() {
                    print!("{}", color.prefix());
                }
            };

            let end_color = || {
                if let Some(color) = color.as_ref() {
                    print!("{}", color.suffix());
                }
            };

            for line in diff.processed_diff[range].lines() {
                end_color();
                print!("{} ", line_number_lines.next().unwrap());

                start_color();
                println!("{}", line);
            }

            end_color();
        }
    }

    Ok(())
}
