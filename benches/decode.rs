//! Criterion benchmarks for heic-decoder-rust end-to-end decoding.
//!
//! Measures the actual pipeline a caller experiences:
//! - `container_parse`: parse ISOBMFF without decoding pixels
//! - `decode_file`: full decode from bytes → planar YUV
//!
//! The inputs are not bundled. Point `HEIF_CONFORMANCE_DIR` at the Nokia
//! conformance files (<https://github.com/nokiatech/heif_conformance>);
//! the benchmarks use C001 and C013 and skip files that are missing.
//!
//! Run with: `HEIF_CONFORMANCE_DIR=/path/to/conformance_files cargo bench --bench decode`
//! Run specific group: `cargo bench --bench decode -- container_parse`

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use std::fs;

const FILES: [&str; 2] = ["C001.heic", "C013.heic"];

fn fixture(name: &str) -> Option<Vec<u8>> {
    let dir = std::env::var_os("HEIF_CONFORMANCE_DIR")?;
    fs::read(std::path::Path::new(&dir).join(name)).ok()
}

/// ISOBMFF container parsing: walk boxes, find primary item, read properties.
/// Does not decode any pixels.
fn container_parse(c: &mut Criterion) {
    let mut group = c.benchmark_group("container_parse");
    group.sample_size(10);

    for name in FILES {
        if let Some(bytes) = fixture(name) {
            group.bench_function(name, |b| {
                b.iter(|| heic_decoder_rust::isobmff::parse(black_box(&bytes)))
            });
        }
    }
    group.finish();
}

/// Full end-to-end decode: bytes in, planar YUV out.
/// This is what a user actually waits for.
fn decode_file(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode_file");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(30)); // Longer measurement for slow operations

    for name in FILES {
        if let Some(bytes) = fixture(name) {
            // Measure throughput in pixels decoded
            if let Ok(img) = heic_decoder_rust::decode(&bytes) {
                group.throughput(Throughput::Elements(
                    (img.width as u64) * (img.height as u64),
                ));
            }
            group.bench_function(name, |b| {
                b.iter(|| heic_decoder_rust::decode(black_box(&bytes)))
            });
        }
    }
    group.finish();
}

criterion_group!(benches, container_parse, decode_file);
criterion_main!(benches);
