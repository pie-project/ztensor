# Encoding profile `zt.zstd-seekable/1`

**Status:** Registry profile · **Core spec:** zTensor v2 §5.3

Compression that keeps a part's decoded stream addressable, so a reader can
serve a range without inflating everything before it.

## Stream

The encoded bytes are a stream in the **zstd seekable format**: a sequence
of independent zstd frames followed by a standard skippable-frame seek
table. Any conforming zstd implementation can decode it.

## Rules

- Every frame MUST carry a content checksum.
- Frame content size MUST be ≤ 16 MiB.
- All frames except the last MUST have equal content size, which makes
  chunk → frame lookup a division rather than a search.
- The concatenated decoded output MUST be exactly the part's
  `decoded_length`; any mismatch is an error, never a short read or a
  zero fill.

## Reading

A reader MAY decode individual frames to serve a range of the decoded
stream, and MUST NOT report a part encoded this way as zero-copy: the
bytes on disk are not the bytes the part denotes.

## Why the constraints

A single zstd stream would make the part decodable only from its start,
which turns a tensor-parallel slice into a full inflate. The equal-frame
rule is what makes the seek table an index rather than a list to scan, and
the checksum is what keeps a corrupt frame from decoding into plausible
garbage.
