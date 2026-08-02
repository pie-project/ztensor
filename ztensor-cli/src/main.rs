//! `zt` — inspect, verify, convert, and diff tensor files.
//!
//! Reads every format the compat crate knows (safetensors, gguf, npz, torch
//! .pt, hdf5, onnx) and writes exactly one: canonical `.zt`.

use std::path::Path;
use std::process::ExitCode;

use xxhash_rust::xxh3::xxh3_64;
use ztensor::{Error, Verified, Writer};
use ztensor_compat::{detect, open};

const USAGE: &str = "\
zt — tensor file tool (zTensor v2)

USAGE:
    zt ls <file>                  list tensors, shapes, and layouts
    zt verify <file> [--deep]     validate; check digests (--deep: shard digests too)
    zt convert <in> <out.zt>      convert any supported format to canonical .zt
               [--align <bytes>]  non-canonical placement (power of two >= 4096)
    zt diff <a> <b>               compare two tensor files by content

Reads: .zt, .safetensors, .gguf, .npz, .pt, .h5, .onnx — writes: .zt only.";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("ls") if args.len() == 2 => ls(Path::new(&args[1])),
        Some("verify") if args.len() == 2 || (args.len() == 3 && args[2] == "--deep") => {
            verify(Path::new(&args[1]), args.len() == 3)
        }
        Some("convert") if args.len() == 3 || (args.len() == 5 && args[3] == "--align") => {
            let align = if args.len() == 5 {
                match args[4].parse::<u64>() {
                    Ok(a) => Some(a),
                    Err(_) => return usage(),
                }
            } else {
                None
            };
            convert(Path::new(&args[1]), Path::new(&args[2]), align)
        }
        Some("diff") if args.len() == 3 => diff(Path::new(&args[1]), Path::new(&args[2])),
        _ => return usage(),
    };
    match result {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn usage() -> ExitCode {
    eprintln!("{USAGE}");
    ExitCode::from(2)
}

fn human(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = bytes as f64;
    let mut unit = 0;
    while v >= 1024.0 && unit < UNITS.len() - 1 {
        v /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{v:.1} {}", UNITS[unit])
    }
}

fn shape_str(shape: &[u64]) -> String {
    let dims: Vec<String> = shape.iter().map(u64::to_string).collect();
    format!("[{}]", dims.join(","))
}

// ---- ls ---------------------------------------------------------------

fn ls(path: &Path) -> Result<ExitCode, Error> {
    let format = detect(path)?;

    // A `.zt` root is listed from its own manifest, with nothing resolved, so
    // `ls` still answers when the shard files are not here — listing is a
    // question about this file.
    if format == "zt" {
        let Some(manifest) = ztensor::manifest_of(path)? else {
            println!("{}: zt data shard (no manifest)", path.display());
            return Ok(ExitCode::SUCCESS);
        };
        println!(
            "{}: zt, {} tensor(s)",
            path.display(),
            manifest.objects.len()
        );
        for (name, obj) in &manifest.objects {
            let total: u64 = obj.parts.values().map(|p| p.decoded_size()).sum();
            let part = obj.parts.values().next();
            let dtype = part.map(|p| p.dtype.as_str()).unwrap_or("?");
            let logical = part
                .and_then(|p| p.logical.as_deref())
                .map(|t| format!(" ({t})"))
                .unwrap_or_default();
            println!(
                "  {name}  {} {dtype}{logical} {}  {}",
                obj.layout,
                shape_str(&obj.shape),
                human(total),
            );
        }
        if manifest.attributes.is_some() {
            println!("  + file attributes");
        }
        if !manifest.shards.is_empty() {
            println!(
                "\nshards ({} expected alongside the root):",
                manifest.shards.len()
            );
            let resolver = ztensor::PositionalResolver::for_root(path);
            for (name, shard) in &manifest.shards {
                let expected = ztensor::ShardResolver::resolve(&resolver, name, shard)
                    .map(|p| p.display().to_string())
                    .unwrap_or_default();
                println!(
                    "  {name}: {expected}  {}  {}",
                    human(shard.size),
                    shard.digest
                );
            }
        }
        return Ok(ExitCode::SUCCESS);
    }

    let src = open(path)?;
    println!("{}: {format}, {} tensor(s)", path.display(), src.len());
    for tensor in src.tensors() {
        let total: u64 = tensor
            .parts()
            .filter_map(|p| tensor.part(p).ok())
            .map(|p| p.nbytes())
            .sum();
        let first = tensor.parts().next().and_then(|p| tensor.part(p).ok());
        let dtype = first.map(|p| p.dtype().as_str()).unwrap_or("?");
        let logical = first
            .and_then(|p| p.logical())
            .map(|t| format!(" ({t})"))
            .unwrap_or_default();
        println!(
            "  {}  {} {dtype}{logical} {}  {}",
            tensor.name(),
            tensor.layout(),
            shape_str(tensor.shape()),
            human(total),
        );
    }
    if src.attributes().is_some() {
        println!("  + file attributes");
    }
    Ok(ExitCode::SUCCESS)
}

// ---- verify -----------------------------------------------------------

fn verify(path: &Path, deep: bool) -> Result<ExitCode, Error> {
    let format = detect(path)?;
    let src = open(path)?;
    if format != "zt" {
        // Opening ran the projection's structural validation.
        println!(
            "{}: {format} opened cleanly, {} tensor(s); no digests to check \
             (convert to .zt for verifiable files)",
            path.display(),
            src.len()
        );
        return Ok(ExitCode::SUCCESS);
    }

    let (mut verified, mut undigested) = (0u64, 0u64);
    for tensor in src.tensors() {
        for name in tensor.parts() {
            match tensor.part(name)?.verify()? {
                Verified::Digest => verified += 1,
                Verified::NoDigest => undigested += 1,
            }
        }
    }
    if deep {
        src.verify_shards()?;
    }
    let shard_count = src.manifest().map(|m| m.shards.len()).unwrap_or(0);
    println!(
        "{}: ok — {verified} part(s) digest-verified, {undigested} without digests{}{}",
        path.display(),
        if shard_count == 0 {
            String::new()
        } else {
            format!(", {shard_count} shard(s) resolved")
        },
        if deep { ", shard digests verified" } else { "" },
    );
    Ok(ExitCode::SUCCESS)
}

// ---- convert ----------------------------------------------------------

fn convert(input: &Path, output: &Path, align: Option<u64>) -> Result<ExitCode, Error> {
    let src = open(input)?;
    let mut writer = match align {
        None => Writer::create(output)?,
        Some(a) => Writer::options().canonical(false).align(a).create(output)?,
    };
    writer.ingest(&src)?;
    let out_size = writer.finish()?;
    println!(
        "{} -> {}: {} tensor(s), {} written{}",
        input.display(),
        output.display(),
        src.len(),
        human(out_size),
        if align.is_none() { " (canonical)" } else { "" },
    );
    Ok(ExitCode::SUCCESS)
}

// ---- diff -------------------------------------------------------------

fn diff(a_path: &Path, b_path: &Path) -> Result<ExitCode, Error> {
    let a = open(a_path)?;
    let b = open(b_path)?;

    let mut added = 0u64;
    let mut removed = 0u64;
    let mut changed = 0u64;

    for name in b.names() {
        if !a.contains(name) {
            println!("+ {name}");
            added += 1;
        }
    }
    for ta in a.tensors() {
        let Some(tb) = b.get(ta.name()) else {
            println!("- {}", ta.name());
            removed += 1;
            continue;
        };
        let mut reasons = Vec::new();
        if ta.shape() != tb.shape() {
            reasons.push(format!(
                "shape {} -> {}",
                shape_str(ta.shape()),
                shape_str(tb.shape())
            ));
        }
        if ta.layout() != tb.layout() {
            reasons.push(format!("layout {} -> {}", ta.layout(), tb.layout()));
        }
        if reasons.is_empty() {
            for part in ta.parts() {
                let Ok(pb) = tb.part(part) else {
                    reasons.push(format!("part {part:?} removed"));
                    continue;
                };
                if xxh3_64(&ta.part(part)?.bytes()?) != xxh3_64(&pb.bytes()?) {
                    reasons.push(format!("content of {part:?}"));
                }
            }
        }
        if !reasons.is_empty() {
            println!("~ {}: {}", ta.name(), reasons.join(", "));
            changed += 1;
        }
    }

    if added + removed + changed == 0 {
        println!("identical: {} tensor(s)", a.len());
        Ok(ExitCode::SUCCESS)
    } else {
        println!("{changed} changed, {added} added, {removed} removed");
        Ok(ExitCode::FAILURE)
    }
}
