# Original monochrome, lossless and alpha fixtures

All image patterns, containers and reference assembly are original to this
repository and covered by its MIT OR Apache-2.0 licence. No third-party images
or decoder implementation are bundled here. The files total about 100 KiB.

Generate with:

```sh
python3 scripts/generate_alpha_fixtures.py /path/to/HM /tmp/alpha-fixtures
```

HM 18 encodes and independently decodes the six source pictures. FNV-1a hashes
in `manifest.json` cover each decoded sample (not the serialized bytes), in
Y/Cb/Cr order. Lossless cases also match their original input bytes exactly.

- `mono8` and `mono12`: monochrome, PPS transquant bypass enabled, encoder
  chooses CU modes; SAO/deblocking enabled.
- `mono10`: 10-bit monochrome, forced lossless CU bypass.
- `lossless420`, `lossless422`, `lossless444`: forced lossless CU bypass,
  10-, 12- and 8-bit respectively.
- `rgba10`, `rgba8alpha`: 10-bit colour paired with 10- or 8-bit native alpha.
- `rgba-grid`: separate two-tile colour/alpha grids, each rotated CCW, mirrored
  horizontally and clean-aperture cropped to 56×100. The alpha hash comes from
  independently transforming and assembling HM's source alpha samples.
- `rgba-prem`: same pixels as `rgba10`, with a `prem` reference to test flag
  propagation. The source colour is deliberately unchanged; this checks the
  declaration and API behaviour, not physically premultiplied rendering.

RDPCM and other advanced range-extension tools are disabled. Lossless bypass
must suppress transform-skip syntax, sign hiding, scaling/transforms and loop
filter changes to protected CUs, while preserving normal neighbour filtering.
