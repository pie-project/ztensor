//! Utility functions for the zTensor CLI

use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};

/// Create a standard progress bar style
pub fn create_progress_style() -> Result<ProgressStyle> {
    ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
        .map_err(|e| anyhow::anyhow!("Failed to create progress style: {}", e))
        .map(|s| s.progress_chars("#>-"))
}

/// Create a new progress bar with standard styling
pub fn create_progress_bar(len: u64) -> Result<ProgressBar> {
    let pb = ProgressBar::new(len);
    pb.set_style(create_progress_style()?);
    Ok(pb)
}
