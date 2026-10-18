//! `probe` must predict what decoding returns, with and without transforms.
use heic_decoder::{DecodeOptions, decode_with, probe};

#[test]
fn probe_matches_decoded_geometry_depth_and_alpha() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut checked = 0;
    for group in ["422", "alpha", "crop", "rgb"] {
        for entry in std::fs::read_dir(dir.join(group)).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_none_or(|e| e != "heic") {
                continue;
            }
            let bytes = std::fs::read(&path).unwrap();
            let info = probe(&bytes).unwrap();
            let shown = decode_with(&bytes, &DecodeOptions::default()).unwrap();
            let coded = decode_with(&bytes, &DecodeOptions { apply_transforms: false }).unwrap();
            let name = path.display();
            assert_eq!((info.width, info.height), (shown.width, shown.height), "{name}");
            assert_eq!((info.coded_width, info.coded_height), (coded.width, coded.height), "{name}");
            assert_eq!(
                (info.bit_depth_luma, info.bit_depth_chroma, info.chroma_format_idc),
                (shown.bit_depth_luma, shown.bit_depth_chroma, shown.chroma_format_idc),
                "{name}"
            );
            assert_eq!(info.alpha_bit_depth, shown.alpha.as_ref().map(|a| a.bit_depth), "{name}");
            for image in [&shown, &coded] {
                if let Some(alpha) = &image.alpha {
                    assert_eq!(alpha.samples.len(), image.width * image.height, "{name}");
                }
            }
            checked += 1;
        }
    }
    assert!(checked >= 20, "only {checked} fixtures");
}

#[test]
fn transforms_can_be_skipped() {
    // grid422: two tiles, rotated a quarter turn and clean-aperture cropped.
    let bytes = include_bytes!("fixtures/crop/grid422.heic");
    let info = probe(bytes).unwrap();
    assert_ne!((info.width, info.height), (info.coded_width, info.coded_height));
    let coded = decode_with(bytes, &DecodeOptions { apply_transforms: false }).unwrap();
    assert_eq!(coded.cb_height, coded.height, "4:2:2 chroma stays full height when not rotated");
}
