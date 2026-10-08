# SPS conformance-window regression fixtures

Original synthetic pixels, encoded by HM 18.0 with one intra picture,
`Profile=main-RExt`, QP 27 and `ConformanceWindowMode=3`. These fixtures and
inputs are covered by this repository's MIT OR Apache-2.0 licence.

All coded pictures are 64×64; the crop offsets below are in luma samples.
Deblocking and SAO run on the complete picture before cropping.

| Fixture | Sampling/depth | Left/right/top/bottom | Output |
|---|---|---|---|
| crop420 | 4:2:0, 8-bit | 2/4/4/6 | 58×54 |
| crop422 | 4:2:2, 10-bit | 2/4/3/5 | 58×56 |
| crop444 | 4:4:4, 12-bit | 1/3/3/5 | 60×56 |
| grid422 | Two crop422 tiles | Grid trim, rotation, offset clean aperture | 40×84 |

Expected planar samples come from HM's independently decoded, cropped output.
The grid reference combines two HM tile outputs, trims to 112×56, rotates all
planes anticlockwise, then crops at luma (10,17) to 40×84. Its chroma origin is
(10,8) on the rotated native sampling grid, producing 40×42 Cb and Cr planes.
`manifest.json` records FNV-1a hashes over sample values and SHA-256 hashes over
raw planar bytes (little-endian u16 above 8 bits).

Reproduce without copying any third-party fixture:

```sh
python3 scripts/generate_crop_fixtures.py /path/to/HM /tmp/crop-fixtures
cargo test --release --test conformance_crop
```

The fixtures are bundled; the Rust tests need no HM installation.
