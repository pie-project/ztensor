//! Command handlers for the zTensor CLI

use anyhow::{Context, Result, bail};
use comfy_table::{Cell, ContentArrangement, Row, Table, presets::UTF8_FULL};
use hf_hub::api::sync::ApiBuilder;
use hf_hub::Repo;
use std::collections::{BTreeMap, HashSet};
use std::path::Path;
use ztensor::{ChecksumAlgorithm, Encoding, ZTensorReader, ZTensorWriter};

use crate::extractors::{SafeTensorExtractor, TensorExtractor};
use crate::utils::create_progress_bar;

// ============================================================================
// Table Printing
// ============================================================================

/// Print detailed metadata for a single tensor
pub fn print_tensor_metadata(name: &str, tensor: &ztensor::Tensor) {
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic);
    table.set_header(vec!["Field", "Value"]);
    table.add_row(Row::from(vec![Cell::new("Name"), Cell::new(name)]));
    table.add_row(Row::from(vec![
        Cell::new("DType"),
        Cell::new(format!("{:?}", tensor.dtype)),
    ]));
    table.add_row(Row::from(vec![
        Cell::new("Format"),
        Cell::new(&tensor.format),
    ]));
    
    let shape_str = if tensor.shape.is_empty() {
        "(scalar)".to_string()
    } else {
        format!(
            "[{}]",
            tensor.shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    table.add_row(Row::from(vec![Cell::new("Shape"), Cell::new(shape_str)]));
    
    let mut components_str = String::new();
    for (key, comp) in &tensor.components {
        components_str.push_str(&format!("{}: [offset={}, len={}", key, comp.offset, comp.length));
        if let Some(enc) = &comp.encoding {
             components_str.push_str(&format!(", enc={:?}", enc));
        }
         if let Some(dig) = &comp.digest {
             components_str.push_str(&format!(", digest={}", dig));
        }
        components_str.push_str("]\n");
    }
    table.add_row(Row::from(vec![
        Cell::new("Components"),
        Cell::new(components_str.trim()),
    ]));
    
    println!("{}", table);
}

/// Print a table listing all tensors
pub fn print_tensors_table(tensors: &BTreeMap<String, ztensor::Tensor>) {
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic);
    table.set_header(vec![
        "#",
        "Name",
        "Shape",
        "DType",
        "Format",
        "Components",
    ]);
    for (i, (name, tensor)) in tensors.iter().enumerate() {
        let shape_str = if tensor.shape.is_empty() {
            "(scalar)".to_string()
        } else {
            format!(
                "[{}]",
                tensor.shape
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        let dtype_str = format!("{:?}", tensor.dtype);
        let format_str = &tensor.format;
        let components_count = tensor.components.len().to_string();
        
        table.add_row(Row::from(vec![
            Cell::new(i),
            Cell::new(name),
            Cell::new(shape_str),
            Cell::new(dtype_str),
            Cell::new(format_str),
            Cell::new(components_count),
        ]));
    }
    println!("{}", table);
}

// ============================================================================
// Conversion Driver
// ============================================================================

/// Run a conversion using the given extractor
pub fn run_conversion(
    extractor: impl TensorExtractor,
    inputs: &[String],
    output: &str,
    compress: bool,
    preserve: bool,
) -> Result<()> {
    let mut writer = ZTensorWriter::create(output)
        .with_context(|| format!("Failed to create output file '{}'", output))?;
    let mut seen_names = HashSet::new();
    let encoding = if compress { Encoding::Zstd } else { Encoding::Raw };
    
    let pb = create_progress_bar(inputs.len() as u64)?;
    
    for input in inputs {
        pb.set_message(format!("Processing {}", input));
        let tensors = extractor.extract_tensors(input)?;
        
        for t in tensors {
            if seen_names.contains(&t.name) {
                bail!("Duplicate tensor name '{}' found in file '{}'. Aborting.", t.name, input);
            }
            writer.add_tensor(
                &t.name,
                t.shape,
                t.dtype,
                encoding.clone(),
                t.data,
                ChecksumAlgorithm::None,
            )?;
            seen_names.insert(t.name);
        }
        
        if !preserve {
            std::fs::remove_file(input)
                .with_context(|| format!("Failed to remove original file '{}'", input))?;
        }
        pb.inc(1);
    }
    
    pb.finish_with_message("Done");
    writer.finalize()?;
    Ok(())
}

// ============================================================================
// Compress / Decompress / Merge
// ============================================================================

/// Compress a zTensor file (raw -> zstd)
pub fn compress_ztensor(input: &str, output: &str) -> Result<()> {
    let mut reader = ZTensorReader::open(input)
        .with_context(|| format!("Failed to open input file '{}'", input))?;
    let mut writer = ZTensorWriter::create(output)
        .with_context(|| format!("Failed to create output file '{}'", output))?;
    let tensors = reader.list_tensors().clone();
    
    let pb = create_progress_bar(tensors.len() as u64)?;
    
    for (name, tensor) in tensors {
        if tensor.format != "dense" {
            eprintln!("Skipping non-dense tensor '{}' during compression", name);
            pb.inc(1);
            continue;
        }
        let data = reader.read_tensor(&name)?;
        writer.add_tensor(
            &name,
            tensor.shape,
            tensor.dtype,
            Encoding::Zstd,
            data,
            ChecksumAlgorithm::None,
        )?;
        pb.inc(1);
    }
    
    pb.finish_with_message("Done");
    writer.finalize()?;
    Ok(())
}

/// Decompress a zTensor file (zstd -> raw)
pub fn decompress_ztensor(input: &str, output: &str) -> Result<()> {
    let mut reader = ZTensorReader::open(input)
        .with_context(|| format!("Failed to open input file '{}'", input))?;
    let mut writer = ZTensorWriter::create(output)
        .with_context(|| format!("Failed to create output file '{}'", output))?;
    let tensors = reader.list_tensors().clone();
    
    let pb = create_progress_bar(tensors.len() as u64)?;
    
    for (name, tensor) in tensors {
        if tensor.format != "dense" {
            eprintln!("Skipping non-dense tensor '{}' during decompression", name);
            pb.inc(1);
            continue;
        }
        let data = reader.read_tensor(&name)?;
        writer.add_tensor(
            &name,
            tensor.shape,
            tensor.dtype,
            Encoding::Raw,
            data,
            ChecksumAlgorithm::None,
        )?;
        pb.inc(1);
    }
    
    pb.finish_with_message("Done");
    writer.finalize()?;
    Ok(())
}

/// Merge multiple zTensor files into one
pub fn merge_ztensor_files(
    inputs: &[String],
    output: &str,
    preserve_original: bool,
) -> Result<()> {
    let mut writer = ZTensorWriter::create(output)
        .with_context(|| format!("Failed to create output file '{}'", output))?;
    let mut seen_names = HashSet::new();
    
    let pb = create_progress_bar(inputs.len() as u64)?;
    
    for input in inputs {
        pb.set_message(format!("Processing {}", input));
        let mut reader = ZTensorReader::open(input)
            .with_context(|| format!("Failed to open input file '{}'", input))?;
        let tensors = reader.list_tensors().clone();
        
        for (name, tensor) in tensors {
            if seen_names.contains(&name) {
                bail!("Duplicate tensor name '{}' found in file '{}'. Aborting merge.", name, input);
            }
            if tensor.format != "dense" {
                eprintln!("Skipping non-dense tensor '{}' during merge", name);
                continue;
            }
            let data = reader.read_tensor(&name)?;
            writer.add_tensor(
                &name,
                tensor.shape,
                tensor.dtype,
                Encoding::Raw,
                data,
                ChecksumAlgorithm::None,
            )?;
            seen_names.insert(name);
        }
        
        if !preserve_original {
            std::fs::remove_file(input)
                .with_context(|| format!("Failed to remove original file '{}'", input))?;
        }
        pb.inc(1);
    }
    
    pb.finish_with_message("Done");
    writer.finalize()?;
    Ok(())
}

// ============================================================================
// HuggingFace Download
// ============================================================================

/// Download safetensors from HuggingFace and convert to zTensor
pub fn download_hf(
    repo_id: &str,
    output_dir: &str,
    token: Option<&str>,
    revision: &str,
    compress: bool,
) -> Result<()> {
    // Create output directory if it doesn't exist
    let output_path = Path::new(output_dir);
    if !output_path.exists() {
        std::fs::create_dir_all(output_path)
            .with_context(|| format!("Failed to create output directory '{}'", output_dir))?;
    }

    // Build HuggingFace API client
    let mut builder = ApiBuilder::new().with_progress(true);
    if let Some(t) = token {
        builder = builder.with_token(Some(t.to_string()));
    }
    let api = builder.build()
        .map_err(|e| anyhow::anyhow!("Failed to build HuggingFace API client: {}", e))?;

    // Create repo handle with revision
    let repo = Repo::with_revision(repo_id.to_string(), hf_hub::RepoType::Model, revision.to_string());
    let api_repo = api.repo(repo);

    // Get repository info to list files
    println!("Fetching repository info for '{}'...", repo_id);
    let repo_info = api_repo.info()
        .map_err(|e| anyhow::anyhow!("Failed to get repository info for '{}': {}", repo_id, e))?;

    // Filter for *.safetensors files
    let safetensor_files: Vec<_> = repo_info
        .siblings
        .iter()
        .filter(|s| s.rfilename.ends_with(".safetensors"))
        .collect();

    if safetensor_files.is_empty() {
        bail!("No .safetensors files found in repository '{}'", repo_id);
    }

    println!("Found {} safetensors file(s) to download and convert:", safetensor_files.len());
    for f in &safetensor_files {
        println!("  - {}", f.rfilename);
    }
    println!();

    let encoding = if compress { Encoding::Zstd } else { Encoding::Raw };
    let extractor = SafeTensorExtractor;
    
    let file_count = safetensor_files.len();
    let pb = create_progress_bar(file_count as u64)?;

    for sibling in &safetensor_files {
        let filename = &sibling.rfilename;
        pb.set_message(format!("Downloading {}", filename));

        // Download file (returns path in HF cache)
        let cached_path = api_repo.get(filename)
            .map_err(|e| anyhow::anyhow!("Failed to download '{}': {}", filename, e))?;

        pb.set_message(format!("Converting {}", filename));

        // Determine output filename: replace .safetensors with .zt
        let basename = Path::new(filename)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(filename);
        let output_name = basename.replace(".safetensors", ".zt");
        let output_file = output_path.join(&output_name);

        if output_file.exists() {
            eprintln!("Warning: Output file '{}' already exists, skipping...", output_file.display());
            pb.inc(1);
            continue;
        }

        // Extract tensors from the cached safetensors file
        let tensors = extractor.extract_tensors(cached_path.to_str().unwrap_or_default())
            .with_context(|| format!("Failed to extract tensors from '{}'", filename))?;

        // Write to zTensor format
        let mut writer = ZTensorWriter::create(&output_file)
            .with_context(|| format!("Failed to create output file '{}'", output_file.display()))?;

        for t in tensors {
            writer.add_tensor(
                &t.name,
                t.shape,
                t.dtype,
                encoding.clone(),
                t.data,
                ChecksumAlgorithm::None,
            )?;
        }

        writer.finalize()?;
        println!("  Created: {}", output_file.display());
        pb.inc(1);
    }

    pb.finish_with_message("Done");
    println!("\nSuccessfully downloaded and converted {} file(s) to {}", 
             file_count, output_dir);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extractors::ExtractedTensor;
    use tempfile::tempdir;

    fn create_test_ztensor(path: &std::path::Path, tensors: Vec<(&str, Vec<u64>, ztensor::DType, Vec<u8>)>) {
        let mut writer = ZTensorWriter::create(path).unwrap();
        for (name, shape, dtype, data) in tensors {
            writer.add_tensor(name, shape, dtype, Encoding::Raw, data, ChecksumAlgorithm::None).unwrap();
        }
        writer.finalize().unwrap();
    }

    #[test]
    fn test_compress_decompress_roundtrip() {
        let dir = tempdir().unwrap();
        let input_path = dir.path().join("input.zt");
        let compressed_path = dir.path().join("compressed.zt");
        let decompressed_path = dir.path().join("decompressed.zt");
        
        let data: Vec<u8> = (0..6).flat_map(|i| (i as f32).to_le_bytes()).collect();
        create_test_ztensor(&input_path, vec![
            ("test_tensor", vec![2, 3], ztensor::DType::Float32, data.clone()),
        ]);
        
        compress_ztensor(input_path.to_str().unwrap(), compressed_path.to_str().unwrap()).unwrap();
        assert!(compressed_path.exists());
        
        decompress_ztensor(compressed_path.to_str().unwrap(), decompressed_path.to_str().unwrap()).unwrap();
        
        let mut reader = ZTensorReader::open(decompressed_path.to_str().unwrap()).unwrap();
        let read_data = reader.read_tensor("test_tensor").unwrap();
        assert_eq!(read_data, data);
    }

    #[test]
    fn test_merge_ztensor_files() {
        let dir = tempdir().unwrap();
        let file1_path = dir.path().join("file1.zt");
        let file2_path = dir.path().join("file2.zt");
        let merged_path = dir.path().join("merged.zt");
        
        create_test_ztensor(&file1_path, vec![
            ("tensor_a", vec![4], ztensor::DType::Uint8, vec![1, 2, 3, 4]),
        ]);
        create_test_ztensor(&file2_path, vec![
            ("tensor_b", vec![4], ztensor::DType::Uint8, vec![5, 6, 7, 8]),
        ]);
        
        let inputs = vec![
            file1_path.to_str().unwrap().to_string(),
            file2_path.to_str().unwrap().to_string(),
        ];
        merge_ztensor_files(&inputs, merged_path.to_str().unwrap(), true).unwrap();
        
        let reader = ZTensorReader::open(merged_path.to_str().unwrap()).unwrap();
        assert!(reader.list_tensors().contains_key("tensor_a"));
        assert!(reader.list_tensors().contains_key("tensor_b"));
    }

    #[test]
    fn test_merge_duplicate_names_fails() {
        let dir = tempdir().unwrap();
        let file1_path = dir.path().join("file1.zt");
        let file2_path = dir.path().join("file2.zt");
        let merged_path = dir.path().join("merged.zt");
        
        create_test_ztensor(&file1_path, vec![
            ("same_name", vec![2], ztensor::DType::Uint8, vec![1, 2]),
        ]);
        create_test_ztensor(&file2_path, vec![
            ("same_name", vec![2], ztensor::DType::Uint8, vec![3, 4]),
        ]);
        
        let inputs = vec![
            file1_path.to_str().unwrap().to_string(),
            file2_path.to_str().unwrap().to_string(),
        ];
        
        let result = merge_ztensor_files(&inputs, merged_path.to_str().unwrap(), true);
        assert!(result.is_err());
    }

    #[test]
    fn test_compress_nonexistent_file() {
        let result = compress_ztensor("/nonexistent/path.zt", "/tmp/out.zt");
        assert!(result.is_err());
    }

    struct MockExtractor {
        tensors: Vec<ExtractedTensor>,
    }

    impl TensorExtractor for MockExtractor {
        fn extract_tensors(&self, _path: &str) -> Result<Vec<ExtractedTensor>> {
            Ok(self.tensors.clone())
        }
    }

    #[test]
    fn test_run_conversion_basic() {
        let dir = tempdir().unwrap();
        let output_path = dir.path().join("output.zt");
        
        let extractor = MockExtractor {
            tensors: vec![
                ExtractedTensor {
                    name: "test".to_string(),
                    shape: vec![2, 3],
                    dtype: ztensor::DType::Float32,
                    data: vec![0u8; 24],
                },
            ],
        };
        
        let inputs = vec!["dummy.input".to_string()];
        run_conversion(extractor, &inputs, output_path.to_str().unwrap(), false, true).unwrap();
        
        assert!(output_path.exists());
    }

    #[test]
    fn test_run_conversion_duplicate_names_fails() {
        let dir = tempdir().unwrap();
        let output_path = dir.path().join("dup.zt");
        
        let extractor = MockExtractor {
            tensors: vec![
                ExtractedTensor {
                    name: "dup_tensor".to_string(),
                    shape: vec![1],
                    dtype: ztensor::DType::Uint8,
                    data: vec![1],
                },
                ExtractedTensor {
                    name: "dup_tensor".to_string(),
                    shape: vec![1],
                    dtype: ztensor::DType::Uint8,
                    data: vec![2],
                },
            ],
        };
        
        let inputs = vec!["dummy.input".to_string()];
        let result = run_conversion(extractor, &inputs, output_path.to_str().unwrap(), false, true);
        assert!(result.is_err());
    }
}
