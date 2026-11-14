# heic-decoder

A pure-Rust HEIF/HEIC still-image decoder. Opens the photos your iPhone takes — no C libraries, no `unsafe`, no system codecs.

- **62 of 63** files in the Nokia HEIF conformance suite decode
- **Planar samples are bit-exact** against HM, the official HEVC reference decoder, on every image compared
- **Pure Rust** — `#![forbid(unsafe_code)]`, only two dependencies (`thiserror`, `rayon`)
- **Grid tiles decoded in parallel** across all CPU cores
- **MIT OR Apache-2.0**

```toml
[dependencies]
heic-decoder = "0.1"
```

```rust
let data = std::fs::read("photo.heic")?;
// Most callers want RGB: this decodes straight to it, without holding
// the planar picture (53 MB instead of 150 MB for a 12 MP iPhone photo).
let rgb = heic_decoder::decode_rgb8(&data)?; // rgb.data holds u8 RGB triples
// A smaller copy in bounded memory (35 MB for that photo, ~50 MB for 200 MP):
let small = heic_decoder::decode_rgb8_with(&data, &heic_decoder::DecodeRgbOptions { max_edge: Some(1568), ..Default::default() })?;

// The planar samples, at their coded bit depth:
let image = heic_decoder::decode(&data)?;
println!("{}×{}, {}-bit", image.width, image.height, image.bit_depth_luma);
// image.y / image.cb / image.cr hold the planar samples

// Converted from those, at higher depth or with alpha:
let rgb = image.to_rgb16()?; // rgb.data holds u16 RGB triples, 0..=65535
// image.to_rgb8()?.data holds u8 RGB triples instead.
// Preserve transparency when needed:
let rgba = image.to_rgba16()?; // rgba.data holds u16 RGBA quadruples

// Size, depth and alpha without decoding pixels, e.g. to enforce limits:
let info = heic_decoder::probe(&data)?;
// Pixels as coded, without the container's crop/rotation/mirror:
let coded = heic_decoder::decode_with(&data, &heic_decoder::DecodeOptions { apply_transforms: false })?;

// ICC profile, EXIF and XMP as raw bytes; also available without
// decoding via heic_decoder::read_metadata(&data)
let icc = image.metadata.icc_profile; // e.g. Display P3 on iPhone photos
let exif = image.metadata.exif;       // TIFF-format, ready for an EXIF parser
```

Or from the command line:

```sh
cargo run --release --example decode_file -- photo.heic out.yuv
# Optional RGB output (PPM; use 8 for RGB8 or 16 for RGB16):
cargo run --release --example decode_rgb -- photo.heic out.ppm 16
```

## Features

- HEVC Main and Main10 still images (8-bit and 10-bit)
- Grid-tiled images, as produced by iPhones and dedicated cameras
- Independent intra slices covering complete CTU rows, with loop filters disabled
- Monochrome and 4:2:0, 4:2:2 and 4:4:4 chroma; 4:2:2 checked at 8, 10 and 12 bits
- SPS conformance-window cropping, rotation, mirroring and clean-aperture cropping (`irot`, `imir`, `clap`)
- Derived identity (`iden`) images
- First frame of HEIF image sequences
- P-pictures predicted from a single IDR reference
- ICC profile, EXIF and XMP extraction
- Auxiliary alpha images, including monochrome and alpha grids
- Basic lossless CU transquant bypass, including bypass of loop filters
- Optional RGB8/RGB16 and RGBA8/RGBA16 conversion with file matrix/range information and chroma upsampling
- Safe on untrusted input: malformed files return an error instead of crashing, and pictures above 268 megapixels (`MAX_IMAGE_PIXELS`) are refused before anything is allocated

`decode` returns monochrome or planar YCbCr at the coded bit depth, with an optional native alpha plane. It does no RGB conversion. Rotation preserves native chroma samples: rotated 4:2:2 can have full-width, half-height chroma, so use `cb_width` and `cb_height` for output plane dimensions. The `decode_file` example writes samples above 8 bits as little-endian `u16`.

## Decoding straight to RGB

`decode_rgb8(&data)` returns the same pixels as `decode(&data)?.to_rgb8()`, bit for bit, without assembling the planar picture. A grid is decoded one row of tiles at a time. Each source row is converted as soon as the chroma rows it reads have been decoded, and is then released. The container's crop, rotation and mirroring are resolved into index maps with the chroma geometry `decode` would produce, so every pixel goes through the planar converter's arithmetic. Peak memory, one core:

| photo | `decode` + `to_rgb8` | `decode_rgb8` | `decode_rgb8_with`, `max_edge: Some(1568)` |
|---|---|---|---|
| iPhone 12 MP grid | 151 MB | 53 MB (36 MB of it the output) | 35 MB |
| 48 MP grid | 582 MB | 193 MB | 66 MB |
| 200 MP grid | 3.4 GB | 608 MB | 49 MB |

`decode_rgb8_with(&data, &DecodeRgbOptions { .. })` adds options:

- `max_edge`: scale the longest edge down to at most this size, by averaging the full-size pixels that fall in each output pixel. Only the output is held.
- `alpha`: `AlphaMode::Ignore` (the default, as `to_rgb8` does; the alpha image is then not decoded at all) or `AlphaMode::Over([r, g, b])`, which composites over a background using alpha as `to_rgba8` scales it.
- `rgb`: the same `RgbOptions` as `to_rgb8_with_options`.
- `max_picture_pixels`, `max_total_pixels`: limits on every coded picture, at its uncropped SPS size, and on a grid's canvas and the sum of its tiles. They are checked before anything is allocated, because a failed allocation aborts the process.

The time is within a few percent of `decode` + `to_rgb8`. A file that doesn't stream (a predicted image, a grid with uneven or odd-sized tiles) goes through `decode` instead, under the same limits and with the same result. `tests/rgb_decode.rs` checks the equality on every bundled fixture, and on every Nokia conformance file with `--ignored`.

## Optional RGB conversion

Call `image.to_rgb8()` or `image.to_rgb16()` after decoding. RGB16 preserves high-bit-depth samples without an intermediate 8-bit conversion. The returned `RgbImage` contains tightly packed, row-major RGB triples.

The converter uses container `colr/nclx` information first, then SPS VUI, then an explicit BT.709 limited-range fallback for unlabelled images. It supports BT.601, BT.709, BT.2020 non-constant luminance, FCC, SMPTE 240M and identity/GBR matrices. Other matrix codes return an unsupported error during RGB conversion; planar decoding remains available. Primaries and transfer codes are exposed in `image.color_info`; ICC remains in `image.metadata.icc_profile`.

Bilinear chroma reconstruction is the default. `image.to_rgb8_with_options(&RgbOptions { color: Some(color_info), upsampling: ChromaUpsampling::Nearest })` can override matrix/range or select nearest-neighbour reconstruction; RGB16 has the same options. Chroma sample positions from SPS VUI are tracked through cropping, mirroring and rotation. Samples outside the returned native planes are edge-extended.

RGB output retains the source primaries and transfer function. Apply the source ICC profile or appropriate color management when displaying it; this conversion does not assume sRGB, linearise pixels or tone-map HDR. The PPM example does not embed ICC metadata.

The conversion computes in floats. Every RGB8 value is the correctly rounded result, and every RGB16 value is within 0.51 of exact. heic-rs uses integer arithmetic instead; [Accuracy compared with heic-rs](#accuracy-compared-with-heic-rs) shows what that costs.

## Transparency

`image.alpha` contains full-range native opacity samples and its own bit depth.
Zero is transparent; `(1 << bit_depth) - 1` is opaque. Colour and alpha each
receive their own conformance crop, grid assembly and container transforms.
Their final dimensions must match; alpha resizing is not implemented.

`image.to_rgba8()` and `image.to_rgba16()` return tightly packed RGBA with
opacity scaled to 0..=255 or 0..=65535. The `*_with_options` variants accept
`RgbOptions`. Images without alpha get an opaque channel. RGB conversion
omits alpha. A `prem` reference is preserved as `rgba.premultiplied`; conversion
does not multiply or unpremultiply colour. Monochrome images convert to equal
R/G/B channels using the signalled luminance range.

Transparent HEICs written by macOS (8- and 10-bit, single images and grids)
decode with alpha identical to Apple's own decoder, and every colour and alpha
tile matches HM.

## Not supported

- AVC (H.264) and layered/scalable HEVC — the one conformance file that fails (`multilayer004`) uses these
- HEVC-internal tiles, dependent slices, and multiple slices with loop filtering or boundaries inside a CTU row (HEIF grids are fine)
- B-pictures, weighted prediction, video playback
- PCM
- Advanced Range Extensions tools such as RDPCM, persistent Rice adaptation, transform-skip rotation, disabled intra smoothing and cross-component prediction; streams that enable them return an `Unsupported` error
- Auxiliary depth maps and HDR gain maps; alpha attached separately to individual colour-grid tiles; resizing alpha to a different display resolution

## Comparison with heic-rs

| | heic-decoder 0.1.1 | [heic-rs](https://github.com/tbraun96/heic-rs) 0.1.4 |
|---|---|---|
| Nokia conformance files decoded | **62 / 63** | 43 / 63 |
| Decoded YCbCr¹ | **matches HM** | matches on 54 Nokia images; differs on 4:2:2 |
| RGB output | **correctly rounded** | up to 2.6 levels off; colours wrong when only the HEVC stream labels them ([details](#accuracy-compared-with-heic-rs)) |
| 12 MP grid photos (3), all cores | 75.5 ms | **53.5 ms** |
| 12 MP grid photos (3), one thread | 319 ms | **264 ms** |
| 43 Nokia files both decode, all cores | 41.2 ms | **41.1 ms** |
| 43 Nokia files both decode, one thread | 49.0 ms | **48.0 ms** |

¹ Sample by sample before colour conversion, from the identical bitstream. heic-rs's samples match heic-decoder's on every primary image in the Nokia suite it decodes. On this crate's 4:2:2 fixtures they differ, while heic-decoder's match HM, the [official HEVC reference decoder](https://vcgit.hhi.fraunhofer.de/jvet/HM).

Timings measure the whole decode to RGB through the same wrapper for both decoders: PhotoCraft's HEIF crate, which probes, decodes and converts to RGB: 8-bit for 8-bit sources and 16-bit for deeper ones, with an alpha channel when the image has one. Each figure is the mean time per file, using each file's best of 3 interleaved rounds, where every round is the median of 5 decodes. The machine is an Apple M4 (4 performance and 6 efficiency cores, macOS 26.7) running release builds with thin LTO. The 12 MP files are 4032×3024 grid HEICs written by macOS `sips`. For one of them, heic-rs's faster result has the wrong colours (see below).

## Accuracy compared with heic-rs

heic-rs is faster, and much of that speed is paid for in accuracy. Both decoders converted the same decoded samples, and every channel was compared against an exact float64 conversion that uses the same chroma siting. The test set was 70 files: Nokia conformance files, 12 MP photos and this crate's fixtures, with heic-rs at version 0.1.4.

| Output | Decoder | correctly rounded | 1 level off | 2 levels off | 3+ off | worst error |
|---|---|---|---|---|---|---|
| RGB8 | **heic-decoder** | **99.9999 %**¹ | 0.0001 % | 0 | 0 | **0.50** |
| RGB8 | heic-rs | 88.27 % | 11.29 % | 0.44 % | 0.005 % | 2.6 |
| RGB16 | **heic-decoder** | **99.85 %** | 0.15 % | 0 | 0 | **0.51** |
| RGB16 | heic-rs | 72.44 % | 0.10 % | 0.03 % | 27.43 % | 543 |

¹ The remaining values are exact rounding ties.

- **Rounding.**
  - heic-rs rounds the interpolated chroma to whole input steps, twice, before applying a fixed-point matrix.
  - As a result, one RGB8 value in nine is wrong, some by two levels.
  - Its RGB16 output is wrong in more than a quarter of all values, by up to 543. That is about two 8-bit levels, so asking heic-rs for 16 bits gains little over 8.
- **Colour information.**
  - heic-rs reads the matrix and range only from a `colr/nclx` box. It ignores the colour information in the HEVC stream's VUI.
  - A file that signals its colour only there, for example one with an ICC `colr` box instead of nclx, is converted as BT.709 limited range whatever it actually is.
  - One example is a 12 MP HEIC written by macOS `sips`, whose stream declares BT.601 full range. heic-rs shifts its colours by up to 31 levels across 82 % of the image.
  - A 10-bit test file from pillow-heif is off by up to 35 levels.
- **4:2:2.**
  - 4:2:2 is the format Sony and Canon cameras write.
  - On this crate's 4:2:2 fixtures, the chroma samples heic-rs 0.1.4 decodes differ from HM's: 378 samples in a single 64×64 picture.
  - In RGB, that becomes errors of up to 48 levels.
- **Streams heic-rs cannot open.** Eight of this crate's fixtures fail in heic-rs with an error: 10- and 12-bit, 4:4:4, lossless and multi-slice streams.

heic-decoder matches HM on all of these files, and its RGB output is the correctly rounded value.

## Testing

```sh
cargo test                     # unit and bundled-fixture tests

# need the Nokia conformance files:
export HEIF_CONFORMANCE_DIR=/path/to/conformance_files
cargo test -- --ignored        # bit-exact checks against HM output
cargo bench                    # decode benchmarks
```

The `internals` cargo feature exposes the container parser and per-image HEVC decoding for these comparisons and for `examples/reference_image.rs`. It is unstable and not covered by semver.

The conformance files are available at [nokiatech/heif_conformance](https://github.com/nokiatech/heif_conformance).

The bundled 4:2:2 regression fixtures run with `cargo test` and need no external decoder. A downloaded Sony a6700 photo also matches HM in every coded tile and in the assembled, rotated 25.6 MP image. The photo ([`FSC00119.HIF`](https://github.com/immich-app/immich/issues/16018)) is not bundled. Additional 4:2:2 streams encoded with HM from natural photos (8/10/12-bit, QP 2–45, CTU 16–64, transform skip, scaling lists, WPP, QP deltas, lossless) also match HM bit for bit.

Bundled [conformance-window fixtures](tests/fixtures/crop/README.md) check cropped 4:2:0, 4:2:2 and 4:4:4 output against HM, including cropped tiles followed by grid assembly, rotation and clean-aperture cropping.

Bundled [RGB fixtures](tests/fixtures/rgb/README.md) check RGB8/RGB16 against an independent float64 reference built from HM output, including VUI color information and a conflicting nclx override. Unit tests cover range, matrices, mixed channel depths, upsampling, transformed chroma positions and malformed public inputs.

Bundled [alpha/lossless fixtures](tests/fixtures/alpha/README.md) verify monochrome, basic transquant bypass and native alpha against HM at 8, 10 and 12 bits, including an alpha grid with rotation, mirroring and a clean aperture.

## Provenance

The decoder was written from the ITU-T H.265 and ISO/IEC 23008-12 specifications. The spec's constant tables (CABAC state transitions, context initialization values, transform matrices, scan orders, deblocking thresholds) were cross-checked against [libde265](https://github.com/strukturag/libde265).

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.

Contributions are welcome. Please make sure `cargo test` and `cargo clippy` pass.
