# heic-decoder-rust

A pure-Rust HEIF/HEIC still-image decoder. Opens the photos your iPhone takes — no C libraries, no `unsafe`, no system codecs.

- **62 of 63** files in the Nokia HEIF conformance suite decode
- **Bit-exact** against HM, the official HEVC reference decoder, on every image compared
- **Pure Rust** — `#![forbid(unsafe_code)]`, only two dependencies (`thiserror`, `rayon`)
- **Grid tiles decoded in parallel** across all CPU cores
- **MIT OR Apache-2.0**

```rust
let data = std::fs::read("photo.heic")?;
let image = heic_decoder_rust::decode(&data)?;
println!("{}×{}, {}-bit", image.width, image.height, image.bit_depth_luma);
// image.y / image.cb / image.cr hold the planar samples

// ICC profile, EXIF and XMP as raw bytes; also available without
// decoding via heic_decoder_rust::read_metadata(&data)
let icc = image.metadata.icc_profile; // e.g. Display P3 on iPhone photos
let exif = image.metadata.exif;       // TIFF-format, ready for an EXIF parser
```

Or from the command line:

```sh
cargo run --release --example decode_file -- photo.heic out.yuv
```

## Features

- HEVC Main and Main10 still images (8-bit and 10-bit)
- Grid-tiled images, as produced by iPhones and dedicated cameras
- Independent intra slices covering complete CTU rows, with loop filters disabled
- 4:2:0, 4:2:2 and 4:4:4 chroma; 4:2:2 checked at 8, 10 and 12 bits
- Rotation, mirroring and clean-aperture cropping (`irot`, `imir`, `clap`)
- Derived identity (`iden`) images
- First frame of HEIF image sequences
- P-pictures predicted from a single IDR reference
- ICC profile, EXIF and XMP extraction
- Safe on untrusted input: malformed files return an error instead of crashing, and pictures above 268 megapixels (`MAX_IMAGE_PIXELS`) are refused before anything is allocated

Output is planar YCbCr at the coded bit depth. Rotation preserves native chroma samples: rotated 4:2:2 can have full-width, half-height chroma, so use `cb_width` and `cb_height` for output plane dimensions. The `decode_file` example writes samples above 8 bits as little-endian `u16`. Color conversion is left to the caller, since the right matrix (BT.601 / BT.709 / …) depends on the file.

## Not supported

- AVC (H.264) and layered/scalable HEVC — the one conformance file that fails (`multilayer004`) uses these
- HEVC-internal tiles, dependent slices, and multiple slices with loop filtering or boundaries inside a CTU row (HEIF grids are fine)
- B-pictures, weighted prediction, video playback
- Monochrome
- PCM and lossless (`cu_transquant_bypass`) coding
- Advanced Range Extensions tools such as RDPCM, persistent Rice adaptation and cross-component prediction
- Auxiliary images such as depth maps and HDR gain maps

## Comparison with other Rust decoders

| | heic-decoder-rust | [heic-rs](https://github.com/tbraun96/heic-rs) 0.1.1 | [imazen/heic](https://github.com/imazen/heic) |
|---|---|---|---|
| Nokia conformance files decoded | **62 / 63** | 44 / 63 | 61 / 63 |
| Raw YCbCr output on 788 coded images¹ | **bit-exact** | **bit-exact** | ±1–2 errors in 114 images |
| iPhone photo, 12 MP grid | 392 ms | **291 ms** | 1271 ms |
| HEIC sample photo, 12 MP grid | 345 ms | **202 ms** | 771 ms |
| 44 conformance files all three decode | 2.65 s | **1.90 s** | 2.47 s |

¹ Every grid tile and single image from the conformance suite and 15 real photos, each decoded by all three from the identical bitstream and compared sample by sample before colour conversion. heic-decoder-rust and heic-rs agree on all 784 images both decode. On the 114 images where imazen/heic differs (1,055 samples, mostly luma), [HM](https://vcgit.hhi.fraunhofer.de/jvet/HM), the official HEVC reference decoder, matches heic-decoder-rust and heic-rs exactly.

Timings: median of repeated runs, release builds, Apple M4 (5 cores). heic-rs and imazen/heic were asked for RGB output; heic-decoder-rust returns YCbCr and leaves colour conversion to the caller, so it does slightly less work per image.

## Testing

```sh
cargo test                     # unit tests
cargo bench                    # decode benchmarks

# bit-exact checks against HM output (needs the Nokia conformance files)
HEIF_CONFORMANCE_DIR=/path/to/conformance_files cargo test -- --ignored
```

The conformance files are available at [nokiatech/heif_conformance](https://github.com/nokiatech/heif_conformance).

## Provenance

The decoder was written from the ITU-T H.265 and ISO/IEC 23008-12 specifications. The spec's constant tables (CABAC state transitions, context initialization values, transform matrices, scan orders, deblocking thresholds) were cross-checked against [libde265](https://github.com/strukturag/libde265).

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.

Contributions are welcome. Please make sure `cargo test` and `cargo clippy` pass.
