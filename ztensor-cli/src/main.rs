//! `zt` — inspect, verify, convert, and diff tensor files.
//!
//! Reads every format the compat crate knows (safetensors, gguf, npz,
//! torch .pt, hdf5, onnx) and writes exactly one: canonical `.zt`.

use std::path::Path;
use std::process::ExitCode;

use xxhash_rust::xxh3::xxh3_64;
use ztensor::{Error, Reader, Source, Writer};
use ztensor_compat::{detect, open_any};

const USAGE: &str = "\
zt — tensor file tool (zTensor v2)

USAGE:
    zt ls <file>                  list objects, shapes, and layouts
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

    // .zt roots are opened alone (no shard resolution) so `ls` also works
    // when the shard files are not present.
    if format == "zt" {
        let reader = Reader::open(path)?;
        if reader.is_data_shard() {
            println!("{}: zt data shard (no manifest)", path.display());
            return Ok(ExitCode::SUCCESS);
        }
        print_objects(path, format, reader.manifest());
        let shards = &reader.manifest().shards;
        if !shards.is_empty() {
            println!("\nshards ({} expected alongside the root):", shards.len());
            let resolver = ztensor::PositionalResolver::for_root(path);
            for (&idx, shard) in shards {
                let expected = ztensor::ShardResolver::resolve(&resolver, idx, shard)
                    .map(|p| p.display().to_string())
                    .unwrap_or_default();
                println!(
                    "  {idx}: {}  {}  {}",
                    expected,
                    human(shard.size),
                    shard.digest
                );
            }
        }
        return Ok(ExitCode::SUCCESS);
    }

    let src = open_any(path)?;
    print_objects(path, format, src.manifest());
    Ok(ExitCode::SUCCESS)
}

fn print_objects(path: &Path, format: &str, manifest: &ztensor::Manifest) {
    println!(
        "{}: {format}, {} object(s)",
        path.display(),
        manifest.objects.len()
    );
    for (name, obj) in &manifest.objects {
        let total: u64 = obj.parts.values().map(|p| p.decoded_size()).sum();
        let part = obj.parts.values().next();
        let dtype = part.map(|p| p.dtype.as_str()).unwrap_or("?");
        let ltype = part
            .and_then(|p| p.ltype.as_deref())
            .map(|t| format!(" ({t})"))
            .unwrap_or_default();
        println!(
            "  {name}  {} {dtype}{ltype} {}  {}",
            obj.layout.as_str(),
            shape_str(&obj.shape),
            human(total),
        );
    }
    if manifest.attributes.is_some() {
        println!("  + file attributes");
    }
}

// ---- verify -----------------------------------------------------------

fn verify(path: &Path, deep: bool) -> Result<ExitCode, Error> {
    let format = detect(path)?;
    if format != "zt" {
        // Opening runs each projection's structural validation.
        let src = open_any(path)?;
        println!(
            "{}: {format} opened cleanly, {} object(s); no digests to check \
             (convert to .zt for verifiable files)",
            path.display(),
            src.manifest().objects.len()
        );
        return Ok(ExitCode::SUCCESS);
    }

    let model = ztensor::Model::open(path)?;
    let manifest = model.manifest().clone();
    let (mut verified, mut undigested) = (0u64, 0u64);
    for (name, obj) in &manifest.objects {
        for part in obj.parts.keys() {
            if model.verify(name, part)? {
                verified += 1;
            } else {
                undigested += 1;
            }
        }
    }
    if deep {
        model.verify_shards()?;
    }
    println!(
        "{}: ok — {verified} part(s) digest-verified, {undigested} without digests{}{}",
        path.display(),
        if manifest.shards.is_empty() {
            String::new()
        } else {
            format!(", {} shard(s) resolved", manifest.shards.len())
        },
        if deep { ", shard digests verified" } else { "" },
    );
    Ok(ExitCode::SUCCESS)
}

// ---- convert ----------------------------------------------------------

fn convert(input: &Path, output: &Path, align: Option<u64>) -> Result<ExitCode, Error> {
    let src = open_any(input)?;
    let mut writer = match align {
        None => Writer::create(output)?,
        Some(a) => Writer::create_with_alignment(output, a)?,
    };
    writer.ingest(src.as_ref())?;
    let out_size = writer.finish()?;
    let objects = src.manifest().objects.len();
    println!(
        "{} -> {}: {objects} object(s), {} written{}",
        input.display(),
        output.display(),
        human(out_size),
        if align.is_none() { " (canonical)" } else { "" },
    );
    Ok(ExitCode::SUCCESS)
}

// ---- diff -------------------------------------------------------------

fn diff(a_path: &Path, b_path: &Path) -> Result<ExitCode, Error> {
    let a = open_any(a_path)?;
    let b = open_any(b_path)?;
    let (ma, mb) = (a.manifest().clone(), b.manifest().clone());

    let mut added = 0u64;
    let mut removed = 0u64;
    let mut changed = 0u64;

    for name in mb.objects.keys() {
        if !ma.objects.contains_key(name) {
            println!("+ {name}");
            added += 1;
        }
    }
    for (name, oa) in &ma.objects {
        let Some(ob) = mb.objects.get(name) else {
            println!("- {name}");
            removed += 1;
            continue;
        };
        let mut reasons = Vec::new();
        if oa.shape != ob.shape {
            reasons.push(format!(
                "shape {} -> {}",
                shape_str(&oa.shape),
                shape_str(&ob.shape)
            ));
        }
        if oa.layout != ob.layout {
            reasons.push(format!(
                "layout {} -> {}",
                oa.layout.as_str(),
                ob.layout.as_str()
            ));
        }
        if reasons.is_empty() {
            for part in oa.parts.keys() {
                if !ob.parts.contains_key(part) {
                    reasons.push(format!("part {part:?} removed"));
                    continue;
                }
                let ha = xxh3_64(&a.read(name, part)?);
                let hb = xxh3_64(&b.read(name, part)?);
                if ha != hb {
                    reasons.push(format!("content of {part:?}"));
                }
            }
        }
        if !reasons.is_empty() {
            println!("~ {name}: {}", reasons.join(", "));
            changed += 1;
        }
    }

    if added + removed + changed == 0 {
        println!("identical: {} object(s)", ma.objects.len());
        Ok(ExitCode::SUCCESS)
    } else {
        println!("{changed} changed, {added} added, {removed} removed");
        Ok(ExitCode::FAILURE)
    }
}
