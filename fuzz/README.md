# Fuzzing

Three targets, each aimed at a different door into the library:

| Target | What it feeds | What must never happen |
| --- | --- | --- |
| `fuzz_validate` | arbitrary bytes through the §8/§3.6 container validator | a panic |
| `fuzz_cbor` | arbitrary bytes through the CBOR codec, then re-encode and re-decode whatever it accepted | a panic, or a value that will not round-trip |
| `fuzz_compat` | arbitrary bytes through every foreign-format projection, mapped and indexed, then every read of everything it found | a panic |

```bash
cargo +nightly fuzz run fuzz_compat -- -max_total_time=900 -rss_limit_mb=4096 -malloc_limit_mb=1024
cargo +nightly fuzz run fuzz_validate fuzz/corpus/fuzz_validate conformance/corpus/valid \
    -- -max_total_time=300 -rss_limit_mb=4096 -malloc_limit_mb=1024
```

CI runs each for a minute or two, which catches a regression that panics
immediately. It is not a campaign; before a release, run the numbers above.

## The two memory limits are not the same question

`-malloc_limit_mb` is the one about this code. It fires on a *single*
allocation, which is the bug worth catching in a binary decoder: reading a
length from the input and reserving that much before checking the bytes are
there. (They are checked — see `a_declared_length_is_checked_before_it_is_believed`
in `ztensor/src/cbor.rs` — and this limit is what keeps that true.)

`-rss_limit_mb` watches the whole process, which over a long run includes
libFuzzer's own bookkeeping as much as the target's. Measured on the 2.0
release: a 240-second `fuzz_cbor` run climbs to 2 GB of RSS and trips the
default limit of 2048, while the same work — the target's exact
decode/encode/decode over the whole corpus, 486,000 executions — holds a flat
3.5 MB in an ordinary binary. The RSS growth is the harness, not the codec, so
the limit is raised rather than chased.

If an `oom-*` artifact appears, check which kind it is before believing it:
run the artifact on its own (`cargo fuzz run <target> <artifact>`). A real
finding reproduces in milliseconds; a harness artifact executes cleanly,
because the input it names is only the one that happened to be running when
the process crossed the line.

## Release campaign, 2.0.0

| Target | Executions | Result |
| --- | ---: | --- |
| `fuzz_compat` | 5,191,331 in 901 s | clean |
| `fuzz_validate` | 9,456,195 in 301 s | clean |
| `fuzz_cbor` | ~215,000 in 240 s | clean; tripped the default RSS limit, investigated above |

`fuzz_compat` was seeded with a minimal safetensors, GGUF and `.npz` file so
the run started inside each parser rather than spending its budget finding the
magic bytes.

## Corpus

`fuzz/corpus/<target>/` is what libFuzzer writes to and reads from.
`conformance/corpus/valid` is passed after it as read-only seeds for
`fuzz_validate` — those are the files the conformance suite says must open.

Inputs named `regression-*` are past findings, kept as seeds so a fix cannot
quietly come undone. Each also has a unit test; the corpus entry is the belt to
that test's braces.
