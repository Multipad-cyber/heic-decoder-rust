# 4:2:2 regression fixtures

These eight small HEIF images contain original, deterministic synthetic patterns,
generated with HM 18.0 using `cfg/encoder_intra_main.cfg`, `Profile=main-RExt`,
`InputChromaFormat=422`, and one intra picture. No third-party photos are bundled.
The synthetic inputs and their encoded fixtures are covered by the repository's
MIT OR Apache-2.0 license.

| Fixture | Purpose |
|---|---|
| `pattern8` | 8-bit prediction, transform splits, deblocking and SAO |
| `pattern10` | 10-bit, partial CTUs, WPP, QP deltas and default scaling lists |
| `smooth10` | High QP and different Cb/Cr QP offsets, including the QP-51 clamp |
| `flat10` | Zero-residual areas and large unsplit transforms |
| `skip10` | QP 0 with transform skipping enabled and loop filters disabled |
| `pattern12` | 12-bit SAO with the default zero offset scales |
| `scaled12` | Explicit PPS SAO offset scales: luma 2, chroma 1 |
| `slices10` | Three independent CTU-row slices, QP deltas, disabled loop filters |

`manifest.json` records geometry, encoder options, and hashes of independently
decoded HM output. FNV-1a hashes operate on sample values, one value per step;
SHA-256 hashes operate on raw Y/Cb/Cr bytes, little-endian u16 above 8 bits.
The HM decoder's output was also compared with the HM encoder's reconstruction.
The Rust tests check geometry, sample ranges, and every plane through the
reference hash. They require no HM installation.

Regenerate into a temporary directory, then review changed hashes and fixtures:

```sh
python3 scripts/generate_422_fixtures.py /path/to/hm-reference /tmp/422-fixtures
cargo test --test chroma422
```

HM is an external validation tool and is not linked into the decoder.
