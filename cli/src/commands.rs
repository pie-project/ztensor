//! Command handlers for the zTensor CLI

use anyhow::{Context, Result, bail};
use comfy_table::{Cell, ContentArrangement, Row, Table, presets::UTF8_FULL};
use hf_hub::api::sync::ApiBuilder;
use hf_hub::Repo;
use std::collections::{BTreeMap, HashSet};
use std::path::Path;
use ztensor::{ChecksumAlgorithm, ZTensorReader, ZTensorWriter};
use ztensor::writer::Compression;

use crate::extractors::{SafeTensorExtractor, TensorExtractor};
use crate::utils::create_progress_bar;

/// Helper to determine compression settings
fn get_compression(compress: bool, level: Option<i32>) -> Compression {
    if let Some(l) = level {
        Compression::Zstd(l)
    } else if compress {
        Compression::Zstd(3) // Default level
    } else {
        Compression::Raw
    }
}

// ============================================================================
// Table Printing
// ============================================================================

/// Print detailed metadata for a single tensor
pub fn print_tensor_metadata(name: &str, obj: &ztensor::Object) {
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic);
    table.set_header(vec!["Field", "Value"]);
    table.add_row(Row::from(vec![Cell::new("Name"), Cell::new(name)]));
    
    // Get dtype from data component if available
    let dtype_str = obj.components.get("data")
        .map(|c| format!("{:?}", c.dtype))
        .unwrap_or_else(|| "N/A".to_string());
    table.add_row(Row::from(vec![
        Cell::new("DType"),
        Cell::new(&dtype_str),
    ]));
    table.add_row(Row::from(vec![
        Cell::new("Format"),
        Cell::new(&obj.format),
    ]));
    
    let shape_str = if obj.shape.is_empty() {
        "(scalar)".to_string()
    } else {
        format!(
            "[{}]",
            obj.shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    table.add_row(Row::from(vec![Cell::new("Shape"), Cell::new(shape_str)]));
    
    let mut components_str = String::new();
    for (key, comp) in &obj.components {
        components_str.push_str(&format!("{}: [offset={}, len={}", key, comp.offset, comp.length));
        if comp.encoding != ztensor::Encoding::Raw {
             components_str.push_str(&format!(", enc={:?}", comp.encoding));
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
pub fn print_tensors_table(objects: &BTreeMap<String, ztensor::Object>) {
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
    for (i, (name, obj)) in objects.iter().enumerate() {
        let shape_str = if obj.shape.is_empty() {
            "(scalar)".to_string()
        } else {
            format!(
                "[{}]",
                obj.shape
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        let dtype_str = obj.components.get("data")
            .map(|c| format!("{:?}", c.dtype))
            .unwrap_or_else(|| "N/A".to_string());
        let format_str = &obj.format;
        let components_count = obj.components.len().to_string();
        
        table.add_row(Row::from(vec![
            Cell::new(i),
            Cell::new(name),
            Cell::new(shape_str),
            Cell::new(&dtype_str),
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
    level: Option<i32>,
    preserve: bool,
) -> Result<()> {
    let mut writer = ZTensorWriter::create(output)
        .with_context(|| format!("Failed to create output file '{}'", output))?;
    let mut seen_names = HashSet::new();
    let compression = get_compression(compress, level);
    
    let pb = create_progress_bar(inputs.len() as u64)?;
    
    for input in inputs {
        pb.set_message(format!("Processing {}", input));
        let tensors = extractor.extract_tensors(input)?;
        
        for t in tensors {
            if seen_names.contains(&t.name) {
                bail!("Duplicate tensor name '{}' found in file '{}'. Aborting.", t.name, input);
            }
            writer.add_object_bytes(
                &t.name,
                t.shape,
                t.dtype,
                compression,
                &t.data,
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
pub fn compress_ztensor(input: &str, output: &str, level: Option<i32>) -> Result<()> {
    let mut reader = ZTensorReader::open(input)
        .with_context(|| format!("Failed to open input file '{}'", input))?;
    let mut writer = ZTensorWriter::create(output)
        .with_context(|| format!("Failed to create output file '{}'", output))?;
    let objects = reader.list_objects().clone();
    
    let compression = if let Some(l) = level {
        Compression::Zstd(l)
    } else {
        Compression::Zstd(3)
    };
    
    let pb = create_progress_bar(objects.len() as u64)?;
    
    for (name, obj) in objects {
        if obj.format != "dense" {
            eprintln!("Skipping non-dense tensor '{}' during compression", name);
            pb.inc(1);
            continue;
        }
        let data_comp = obj.components.get("data")
            .ok_or_else(|| anyhow::anyhow!("Missing data component for tensor '{}'", name))?;
        let data = reader.read_object(&name, true)?;
        writer.add_object_bytes(
            &name,
            obj.shape,
            data_comp.dtype,
            compression,
            &data,
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
    let objects = reader.list_objects().clone();
    
    let pb = create_progress_bar(objects.len() as u64)?;
    
    for (name, obj) in objects {
        if obj.format != "dense" {
            eprintln!("Skipping non-dense tensor '{}' during decompression", name);
            pb.inc(1);
            continue;
        }
        let data_comp = obj.components.get("data")
            .ok_or_else(|| anyhow::anyhow!("Missing data component for tensor '{}'", name))?;
        let data = reader.read_object(&name, true)?;
        writer.add_object_bytes(
            &name,
            obj.shape,
            data_comp.dtype,
            Compression::Raw,
            &data,
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
        let objects = reader.list_objects().clone();
        
        for (name, obj) in objects {
            if seen_names.contains(&name) {
                bail!("Duplicate tensor name '{}' found in file '{}'. Aborting merge.", name, input);
            }
            if obj.format != "dense" {
                eprintln!("Skipping non-dense tensor '{}' during merge", name);
                continue;
            }
            let data_comp = obj.components.get("data")
                .ok_or_else(|| anyhow::anyhow!("Missing data component for tensor '{}'", name))?;
            let data = reader.read_object(&name, true)?;
            writer.add_object_bytes(
                &name,
                obj.shape,
                data_comp.dtype,
                Compression::Raw,
                &data,
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

/// Migrate a v0.1.0 ztensor file to v1.1.0 format
pub fn migrate_ztensor(input: &str, output: &str, compress: bool, level: Option<i32>) -> Result<()> {
    use ztensor::compat::LegacyReader;
    use std::fs::File;

    let file = File::open(input)
        .with_context(|| format!("Failed to open input file '{}'", input))?;
    
    // Check if it's actually a legacy file
    if !ztensor::compat::is_legacy_file(input)? {
        bail!("Input file '{}' is not a valid zTensor v0.1.0 legacy file.", input);
    }

    let mut reader = LegacyReader::new(file)
        .with_context(|| format!("Failed to create legacy reader for '{}'", input))?;
    
    let mut writer = ZTensorWriter::create(output)
        .with_context(|| format!("Failed to create output file '{}'", output))?;
    
    let objects = reader.list_objects().clone();
    let pb = create_progress_bar(objects.len() as u64)?;
    
    let compression = get_compression(compress, level);

    for (name, obj) in objects {
        pb.set_message(format!("Migrating {}", name));
        
        // Legacy files only supported dense tensors
        let data_comp = obj.components.get("data")
            .ok_or_else(|| anyhow::anyhow!("Missing data component for tensor '{}'", name))?;
            
        let data = reader.read_object(&name, true)?;
        
        writer.add_object_bytes(
            &name,
            obj.shape,
            data_comp.dtype,
            compression,
            &data,
            ChecksumAlgorithm::None,
        )?;
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
    level: Option<i32>,
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

    let compression = get_compression(compress, level);
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
            writer.add_object_bytes(
                &t.name,
                t.shape,
                t.dtype,
                compression,
                &t.data,
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
            writer.add_object_bytes(name, shape, dtype, Compression::Raw, &data, ChecksumAlgorithm::None).unwrap();
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
            ("test_tensor", vec![2, 3], ztensor::DType::F32, data.clone()),
        ]);
        
        compress_ztensor(input_path.to_str().unwrap(), compressed_path.to_str().unwrap(), Some(1)).unwrap();
        assert!(compressed_path.exists());
        
        decompress_ztensor(compressed_path.to_str().unwrap(), decompressed_path.to_str().unwrap()).unwrap();
        
        let mut reader = ZTensorReader::open(decompressed_path.to_str().unwrap()).unwrap();
        let read_data = reader.read_object("test_tensor", true).unwrap();
        assert_eq!(read_data, data);
    }

    #[test]
    fn test_merge_ztensor_files() {
        let dir = tempdir().unwrap();
        let file1_path = dir.path().join("file1.zt");
        let file2_path = dir.path().join("file2.zt");
        let merged_path = dir.path().join("merged.zt");
        
        create_test_ztensor(&file1_path, vec![
            ("tensor_a", vec![4], ztensor::DType::U8, vec![1, 2, 3, 4]),
        ]);
        create_test_ztensor(&file2_path, vec![
            ("tensor_b", vec![4], ztensor::DType::U8, vec![5, 6, 7, 8]),
        ]);
        
        let inputs = vec![
            file1_path.to_str().unwrap().to_string(),
            file2_path.to_str().unwrap().to_string(),
        ];
        merge_ztensor_files(&inputs, merged_path.to_str().unwrap(), true).unwrap();
        
        let reader = ZTensorReader::open(merged_path.to_str().unwrap()).unwrap();
        assert!(reader.list_objects().contains_key("tensor_a"));
        assert!(reader.list_objects().contains_key("tensor_b"));
    }

    #[test]
    fn test_merge_duplicate_names_fails() {
        let dir = tempdir().unwrap();
        let file1_path = dir.path().join("file1.zt");
        let file2_path = dir.path().join("file2.zt");
        let merged_path = dir.path().join("merged.zt");
        
        create_test_ztensor(&file1_path, vec![
            ("same_name", vec![2], ztensor::DType::U8, vec![1, 2]),
        ]);
        create_test_ztensor(&file2_path, vec![
            ("same_name", vec![2], ztensor::DType::U8, vec![3, 4]),
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
        let result = compress_ztensor("/nonexistent/path.zt", "/tmp/out.zt", None);
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
                    dtype: ztensor::DType::F32,
                    data: vec![0u8; 24],
                },
            ],
        };
        
        let inputs = vec!["dummy.input".to_string()];
        run_conversion(extractor, &inputs, output_path.to_str().unwrap(), false, None, true).unwrap();
        
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
                    dtype: ztensor::DType::U8,
                    data: vec![1],
                },
                ExtractedTensor {
                    name: "dup_tensor".to_string(),
                    shape: vec![1],
                    dtype: ztensor::DType::U8,
                    data: vec![2],
                },
            ],
        };
        
        let inputs = vec!["dummy.input".to_string()];
        let result = run_conversion(extractor, &inputs, output_path.to_str().unwrap(), false, None, true);
        assert!(result.is_err());
    }
}
