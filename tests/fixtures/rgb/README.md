# Optional RGB conversion oracles

Original synthetic 10-bit 4:2:0 pixels, encoded by HM 18.0. Both HEIF fixtures
share the same bitstream: SPS VUI declares BT.2020 non-constant luminance,
full range, PQ transfer, BT.2020 primaries and centred chroma (location 1).
The SPS crops a 64×64 coded image to 58×54.

- `vui2020.heic` has no container colour override and uses the VUI description.
- `nclx601.heic` supplies a conflicting primary-item `colr/nclx`: BT.601 matrix,
  limited range, primaries 1, transfer 13. It verifies that nclx overrides VUI.

The RGB references come from HM's cropped planar output, processed by an
independent scalar Python float64 implementation of the matrix/range equations
with bilinear chroma reconstruction. RGB16 files store little-endian u16 values;
RGB8 files store bytes. `manifest.json` records SHA-256 values. Rust's float32
converter is checked sample by sample with at most one output value of rounding
difference, for RGB8 and RGB16. These tests do not linearise PQ or perform an ICC
conversion: output retains the source transfer and primaries.

The original synthetic inputs, HEIF files and reference data are covered by the
repository's MIT OR Apache-2.0 licence. No third-party fixture is bundled.

Reproduce:

```sh
python3 scripts/generate_rgb_fixtures.py /path/to/HM /tmp/rgb-fixtures
cargo test --release --test rgb
```

HM is a validation tool, not a dependency of the converter.
