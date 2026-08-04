//! Benchmark harness for the zTensor website numbers.
//!
//! Measures what a weight loader actually does, on a synthetic model whose
//! shape mirrors a transformer checkpoint (many medium tensors, a few
//! large ones). Every number is wall-clock over the whole operation,
//! median of N runs, with the page cache dropped or warmed explicitly:
//! whichever the scenario is about.
//!
//! Usage: `cargo run --release -p benchmark -- [--size-mb N] [--runs N]`
//!
//! Output is a Markdown table on stdout, for pasting into the website.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ztensor::{DType, Writer};

/// A synthetic checkpoint: `n` tensors totalling roughly `total_mb`.
struct Model {
    tensors: Vec<(String, Vec<u64>, Vec<u8>)>,
}

impl Model {
    /// `tensor_mb` sets the size of a per-layer weight. Alignment padding
    /// costs ~32 KiB per tensor, so this knob is what decides the `.zt`
    /// size overhead: 1 MiB tensors are a deliberate worst case, 32 MiB is
    /// typical for a transformer's projection matrices.
    fn synth(total_mb: usize, tensor_mb: usize) -> Self {
        // A transformer-ish mix: 4 big embedding/output matrices, then
        // per-layer weights, then a tail of small biases and norms.
        let total = total_mb * (1 << 20);
        let mut tensors = Vec::new();
        let mut written = 0usize;
        let mut layer = 0usize;

        // Every tensor gets distinct bytes. Identical tensors would be
        // deduplicated into one blob by canonical form. That is a real feature,
        // but it would make a size comparison meaningless here.
        let fill = |seed: u64, len: usize| -> Vec<u8> {
            let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
            (0..len)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    (x >> 24) as u8
                })
                .collect()
        };

        let big = total / 4;
        for i in 0..2u64 {
            let elems = (big / 2) / 2; // bf16
            let bytes = fill(i, elems * 2);
            tensors.push((format!("embed.{i}.weight"), vec![elems as u64], bytes));
            written += elems * 2;
        }
        while written < total {
            let remaining = total - written;
            let elems = (tensor_mb * (1 << 20) / 2).min(remaining / 2).max(1);
            let bytes = fill(0x1000 + layer as u64, elems * 2);
            written += bytes.len();
            tensors.push((
                format!("blk.{layer:04}.attn.weight"),
                vec![elems as u64],
                bytes,
            ));
            let norm = fill(0x2000 + layer as u64, 512);
            tensors.push((format!("blk.{layer:04}.attn_norm.weight"), vec![256], norm));
            written += 512;
            layer += 1;
        }
        tensors.sort_by(|a, b| a.0.cmp(&b.0));
        Model { tensors }
    }

    fn bytes(&self) -> u64 {
        self.tensors.iter().map(|(_, _, b)| b.len() as u64).sum()
    }

    fn write_zt(&self, path: &Path, align: Option<u64>) -> std::io::Result<u64> {
        let mut w = match align {
            None => Writer::create(path),
            Some(a) => Writer::options().canonical(false).align(a).create(path),
        }
        .expect("create");
        for (name, shape, data) in &self.tensors {
            w.add(name, shape.to_vec(), DType::BF16, data).expect("add");
        }
        Ok(w.finish().expect("finish"))
    }

    /// The same tensors as a safetensors file (header + packed data).
    fn write_safetensors(&self, path: &Path) -> std::io::Result<u64> {
        let mut entries = Vec::new();
        let mut cursor = 0usize;
        for (name, shape, data) in &self.tensors {
            let dims: Vec<String> = shape.iter().map(u64::to_string).collect();
            entries.push(format!(
                "\"{name}\":{{\"dtype\":\"BF16\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
                dims.join(","),
                cursor,
                cursor + data.len()
            ));
            cursor += data.len();
        }
        let header = format!("{{{}}}", entries.join(","));
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(header.as_bytes());
        for (_, _, data) in &self.tensors {
            out.extend_from_slice(data);
        }
        fs::write(path, &out)?;
        Ok(out.len() as u64)
    }
}

/// Sums every byte, to fault and touch the whole mapping:
/// anything that skips bytes measures address arithmetic, not I/O.
fn checksum(bytes: &[u8]) -> u64 {
    bytes
        .chunks(8)
        .map(|c| c.iter().map(|&b| b as u64).sum::<u64>())
        .sum()
}

fn median(mut xs: Vec<Duration>) -> Duration {
    xs.sort();
    xs[xs.len() / 2]
}

fn gbps(bytes: u64, d: Duration) -> f64 {
    bytes as f64 / d.as_secs_f64() / 1e9
}

/// Drops this file's page cache so the next read really touches storage.
fn drop_cache(path: &Path) {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::io::AsRawFd;
        if let Ok(f) = fs::File::open(path) {
            // SAFETY: POSIX_FADV_DONTNEED on a read-only fd only drops
            // clean page-cache pages for this file.
            unsafe {
                libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
            }
        }
    }
    let _ = path;
}

fn bench(name: &str, runs: usize, bytes: u64, mut f: impl FnMut()) -> (String, f64, Duration) {
    let mut times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t = Instant::now();
        f();
        times.push(t.elapsed());
    }
    let d = median(times);
    (name.to_string(), gbps(bytes, d), d)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |flag: &str, default: usize| -> usize {
        args.windows(2)
            .find(|w| w[0] == flag)
            .and_then(|w| w[1].parse().ok())
            .unwrap_or(default)
    };
    let size_mb = arg("--size-mb", 512);
    let tensor_mb = arg("--tensor-mb", 1);
    let runs = arg("--runs", 5);

    // Files go on real storage, not /tmp: on many systems /tmp is tmpfs,
    // where a "cold cache" read never touches a device and the number
    // would be a fiction. Override with ZTENSOR_BENCH_DIR.
    let dir = std::env::var_os("ZTENSOR_BENCH_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target/bench"));
    fs::create_dir_all(&dir).expect("bench dir");
    let zt: PathBuf = dir.join("model.zt");
    let zt4k: PathBuf = dir.join("model-4k.zt");
    let st: PathBuf = dir.join("model.safetensors");

    eprintln!("building a ~{size_mb} MiB synthetic checkpoint ({tensor_mb} MiB tensors)...");
    let model = Model::synth(size_mb, tensor_mb);
    let payload = model.bytes();
    eprintln!(
        "  {} tensors, {:.1} MiB of tensor data",
        model.tensors.len(),
        payload as f64 / (1 << 20) as f64
    );

    let mut rows: Vec<(String, f64, Duration)> = Vec::new();

    // ---- write ---------------------------------------------------------
    rows.push(bench(
        "write .zt (canonical, 64 KiB)",
        runs,
        payload,
        || {
            model.write_zt(&zt, None).unwrap();
        },
    ));
    rows.push(bench("write .zt (4 KiB floor)", runs, payload, || {
        model.write_zt(&zt4k, Some(4096)).unwrap();
    }));
    rows.push(bench("write .safetensors", runs, payload, || {
        model.write_safetensors(&st).unwrap();
    }));

    let zt_size = fs::metadata(&zt).unwrap().len();
    let st_size = fs::metadata(&st).unwrap().len();

    // ---- open (metadata only) ------------------------------------------
    // Not a throughput number: this is the latency of learning what is in
    // the file, which is what a loader pays before it can plan anything.
    let open_zt = {
        let mut times = Vec::new();
        for _ in 0..runs {
            let t = Instant::now();
            let r = ztensor::Source::open(&zt).expect("open");
            std::hint::black_box(r.len());
            times.push(t.elapsed());
        }
        median(times)
    };
    let open_st = {
        let mut times = Vec::new();
        for _ in 0..runs {
            let t = Instant::now();
            let s = ztensor_compat::open(&st).expect("open");
            std::hint::black_box(s.len());
            times.push(t.elapsed());
        }
        median(times)
    };

    // ---- warm zero-copy view -------------------------------------------
    // Touch every byte so the mmap actually faults; a view that is never
    // read measures nothing.
    let reader = ztensor::Source::open(&zt).expect("open");
    let names: Vec<String> = reader.names().map(str::to_string).collect();
    rows.push(bench(
        "read .zt zero-copy, full traversal (warm)",
        runs,
        payload,
        || {
            let mut sum = 0u64;
            for n in &names {
                sum += checksum(
                    reader
                        .tensor(n)
                        .expect("tensor")
                        .data()
                        .unwrap()
                        .map()
                        .expect("view"),
                );
            }
            std::hint::black_box(sum);
        },
    ));

    let stf = ztensor_compat::open(&st).expect("open");
    let st_names: Vec<String> = stf.names().map(str::to_string).collect();
    rows.push(bench(
        "read .safetensors zero-copy, full traversal (warm)",
        runs,
        payload,
        || {
            let mut sum = 0u64;
            for n in &st_names {
                sum += checksum(
                    stf.tensor(n)
                        .expect("tensor")
                        .data()
                        .unwrap()
                        .map()
                        .expect("view"),
                );
            }
            std::hint::black_box(sum);
        },
    ));

    // ---- copying read ---------------------------------------------------
    rows.push(bench(
        "copy .zt into owned buffers (warm, memcpy; not comparable to the \
         traversals above, which sum every byte)",
        runs,
        payload,
        || {
            let mut total = 0usize;
            for n in &names {
                // `into_owned` is the point of this row: `bytes()` hands back a
                // borrow when the file is mapped, and timing that would be
                // timing nothing.
                let owned = reader
                    .tensor(n)
                    .expect("tensor")
                    .data()
                    .unwrap()
                    .bytes()
                    .expect("read")
                    .into_owned();
                total += owned.len();
                std::hint::black_box(&owned);
            }
            std::hint::black_box(total);
        },
    ));

    // ---- cold read ------------------------------------------------------
    // Cold reads use a private copy: an existing mapping of the same file
    // pins its pages, so dropping the cache would be a no-op otherwise.
    let cold_path = dir.join("cold.zt");
    let cold_zt = {
        let mut times = Vec::new();
        for _ in 0..runs {
            fs::copy(&zt, &cold_path).expect("copy");
            drop_cache(&cold_path);
            let t = Instant::now();
            let r = ztensor::Source::open(&cold_path).expect("open");
            let mut sum = 0u64;
            for n in &names {
                sum += checksum(
                    r.tensor(n)
                        .expect("tensor")
                        .data()
                        .unwrap()
                        .map()
                        .expect("view"),
                );
            }
            std::hint::black_box(sum);
            drop(r);
            times.push(t.elapsed());
        }
        median(times)
    };
    // A cold read that is not slower than the warm one means the cache
    // drop did not take (tmpfs, an overlay, or a pinned mapping), so report
    // nothing rather than a fabricated bandwidth.
    let warm = rows
        .iter()
        .find(|(n, _, _)| n.starts_with("read .zt zero-copy, full traversal (warm)"))
        .map(|(_, _, d)| *d)
        .unwrap_or_default();
    let cold_trustworthy = cold_zt > warm * 2;
    if cold_trustworthy {
        rows.push((
            "read .zt zero-copy, full traversal (cold cache)".into(),
            gbps(payload, cold_zt),
            cold_zt,
        ));
    } else {
        eprintln!(
            "note: cold-cache read ({:.1} ms) is not meaningfully slower than warm \
             ({:.1} ms). The page-cache drop had no effect on this filesystem, so \
             the cold number is omitted",
            cold_zt.as_secs_f64() * 1e3,
            warm.as_secs_f64() * 1e3
        );
    }

    // ---- verification ---------------------------------------------------
    rows.push(bench("verify every digest (xxh3)", runs, payload, || {
        for n in &names {
            reader.tensor(n).expect("tensor").verify().expect("verify");
        }
    }));

    // ---- conversion ------------------------------------------------------
    let conv = dir.join("converted.zt");
    rows.push(bench(
        "convert .safetensors -> canonical .zt",
        runs,
        payload,
        || {
            let src = ztensor_compat::open(&st).expect("open");
            let mut w = Writer::create(&conv).expect("create");
            w.ingest(&src).expect("ingest");
            w.finish().expect("finish");
        },
    ));

    // ---- report ----------------------------------------------------------
    println!("## Throughput\n");
    println!("| Operation | Throughput | Median time |");
    println!("| --- | --- | --- |");
    for (name, rate, d) in &rows {
        println!(
            "| {name} | {rate:.2} GB/s | {:.1} ms |",
            d.as_secs_f64() * 1e3
        );
    }

    println!("\n## Open latency (metadata only)\n");
    println!("| Format | Time to enumerate {} tensors |", names.len());
    println!("| --- | --- |");
    println!("| `.zt` | {:.2} ms |", open_zt.as_secs_f64() * 1e3);
    println!("| `.safetensors` | {:.2} ms |", open_st.as_secs_f64() * 1e3);

    println!("\n## File size\n");
    println!("| File | Size | Overhead vs payload |");
    println!("| --- | --- | --- |");
    let pct = |n: u64| (n as f64 - payload as f64) / payload as f64 * 100.0;
    println!(
        "| `.zt` canonical (64 KiB) | {:.1} MiB | {:+.2}% |",
        zt_size as f64 / (1 << 20) as f64,
        pct(zt_size)
    );
    let zt4k_size = fs::metadata(&zt4k).unwrap().len();
    println!(
        "| `.zt` 4 KiB floor | {:.1} MiB | {:+.2}% |",
        zt4k_size as f64 / (1 << 20) as f64,
        pct(zt4k_size)
    );
    println!(
        "| `.safetensors` | {:.1} MiB | {:+.2}% |",
        st_size as f64 / (1 << 20) as f64,
        pct(st_size)
    );

    println!(
        "\n_{} tensors ({tensor_mb} MiB each for the layer weights), {:.0} MiB of \
         tensor data; median of {runs} runs._",
        names.len(),
        payload as f64 / (1 << 20) as f64
    );

    let _ = fs::remove_dir_all(&dir);
}
