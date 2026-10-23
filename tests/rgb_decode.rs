//! `decode_rgb8_with` must give exactly what the planar path gives: the
//! bundled fixtures (crops, rotations, mirrors, alpha, 4:2:2, 10/12-bit,
//! monochrome) here, and every Nokia conformance file with
//! `HEIF_CONFORMANCE_DIR=/path/to/conformance_files cargo test -- --ignored`.
use heic_decoder::{AlphaMode, ChromaUpsampling, DecodeRgbOptions, RgbImage, RgbOptions};
use std::{env, fs, path::Path};

fn planar(data: &[u8], options: &DecodeRgbOptions) -> heic_decoder::Result<RgbImage<u8>> {
    let image = heic_decoder::decode(data)?;
    match options.alpha {
        AlphaMode::Over(bg) if image.alpha.is_some() => {
            let rgba = image.to_rgba8_with_options(&options.rgb)?;
            let blend = |c: u8, a: u32, b: u8| ((u32::from(c) * a + u32::from(b) * (255 - a) + 127) / 255) as u8;
            let data = rgba
                .data
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|&[r, g, b, a]| [blend(r, a.into(), bg[0]), blend(g, a.into(), bg[1]), blend(b, a.into(), bg[2])])
                .collect();
            Ok(RgbImage { width: rgba.width, height: rgba.height, data })
        }
        _ => image.to_rgb8_with_options(&options.rgb),
    }
}

/// The box average `max_edge` asks for, computed from the full-size pixels.
fn averaged(full: &RgbImage<u8>, max_edge: usize) -> RgbImage<u8> {
    let (w, h) = (full.width, full.height);
    let longest = w.max(h);
    let scale = |side: usize| if longest > max_edge { ((side * max_edge + longest / 2) / longest).max(1) } else { side };
    let (ow, oh) = (scale(w), scale(h));
    let (mut sums, mut counts) = (vec![0u64; ow * oh * 3], vec![0u64; ow * oh]);
    for y in 0..h {
        for x in 0..w {
            let bin = (y * oh / h) * ow + x * ow / w;
            counts[bin] += 1;
            for c in 0..3 {
                sums[bin * 3 + c] += u64::from(full.data[(y * w + x) * 3 + c]);
            }
        }
    }
    let data = sums.iter().enumerate().map(|(i, &s)| ((s + counts[i / 3] / 2) / counts[i / 3]) as u8).collect();
    RgbImage { width: ow, height: oh, data }
}

/// Checks one file; returns how many images were compared.
fn check(path: &Path) -> usize {
    let data = fs::read(path).unwrap();
    let cases = [
        DecodeRgbOptions::default(),
        DecodeRgbOptions { rgb: RgbOptions { upsampling: ChromaUpsampling::Nearest, ..Default::default() }, ..Default::default() },
        DecodeRgbOptions { alpha: AlphaMode::Over([255, 128, 0]), ..Default::default() },
    ];
    let mut compared = 0;
    for options in &cases {
        let Ok(want) = planar(&data, options) else { continue };
        let got = heic_decoder::decode_rgb8_with(&data, options).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!((got.width, got.height), (want.width, want.height), "{}", path.display());
        assert!(got.data == want.data, "{}: {options:?}", path.display());
        compared += 1;
    }
    if let Ok(full) = planar(&data, &DecodeRgbOptions::default()) {
        for max_edge in [1, 7, 64] {
            let got = heic_decoder::decode_rgb8_with(&data, &DecodeRgbOptions { max_edge: Some(max_edge), ..Default::default() }).unwrap();
            assert!(got == averaged(&full, max_edge), "{}: max_edge {max_edge}", path.display());
        }
    }
    compared
}

fn heif_files(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            files.extend(heif_files(&path));
        } else if path.extension().is_some_and(|e| ["heic", "heif", "hif"].contains(&e.to_ascii_lowercase().to_str().unwrap_or(""))) {
            files.push(path);
        }
    }
    files.sort();
    files
}

#[test]
fn bundled_fixtures_match_the_planar_path() {
    let files = heif_files(Path::new("tests/fixtures"));
    assert!(!files.is_empty());
    let compared: usize = files.iter().map(|p| check(p)).sum();
    assert!(compared >= files.len(), "only {compared} images compared");
}

#[test]
#[ignore = "requires HEIF_CONFORMANCE_DIR pointing to Nokia conformance_files"]
fn conformance_files_match_the_planar_path() {
    let root = env::var_os("HEIF_CONFORMANCE_DIR").expect("set HEIF_CONFORMANCE_DIR");
    let compared: usize = heif_files(Path::new(&root)).iter().map(|p| check(p)).sum();
    assert!(compared > 100, "only {compared} images compared");
}
