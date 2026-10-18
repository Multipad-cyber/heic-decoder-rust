//! Independent float64 RGB oracles from HM planar output, with colour
//! signalled in SPS VUI and with a conflicting container nclx override.
use heic_decoder::{ColorInfo, decode, decode_one_image, isobmff::parse};

#[test]
fn rgb8_and_rgb16_match_independent_reference_with_vui_and_nclx() {
    for (name, input, expected8, expected16, color) in [
        (
            "vui2020",
            &include_bytes!("fixtures/rgb/vui2020.heic")[..],
            &include_bytes!("fixtures/rgb/vui2020-rgb8.bin")[..],
            &include_bytes!("fixtures/rgb/vui2020-rgb16.bin")[..],
            ColorInfo { color_primaries: 9, transfer_characteristics: 16, matrix_coefficients: 9, full_range: true },
        ),
        (
            "nclx601",
            &include_bytes!("fixtures/rgb/nclx601.heic")[..],
            &include_bytes!("fixtures/rgb/nclx601-rgb8.bin")[..],
            &include_bytes!("fixtures/rgb/nclx601-rgb16.bin")[..],
            ColorInfo { color_primaries: 1, transfer_characteristics: 13, matrix_coefficients: 6, full_range: false },
        ),
    ] {
        let parsed = parse(input).unwrap();
        let (sps, _) = decode_one_image(&parsed.images[0]).unwrap();
        assert_eq!(sps.color_info.unwrap().matrix_coefficients, 9);
        assert_eq!(sps.chroma_location, Some(1));
        let image = decode(input).unwrap();
        assert_eq!(image.color_info, Some(color), "{name}: nclx must override VUI");
        assert_eq!((image.width, image.height), (58, 54));
        assert_eq!((image.chroma_sampling.origin_x, image.chroma_sampling.origin_y), (0.5, 0.5));
        let rgb8 = image.to_rgb8().unwrap();
        assert_eq!((rgb8.width, rgb8.height), (58, 54));
        assert_eq!(rgb8.data.len(), expected8.len());
        let max8 = rgb8.data.iter().zip(expected8).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
        assert!(max8 <= 1, "{name}: f32 RGB8 vs independent f64 reference, max={max8}");
        let reference16: Vec<_> = expected16.as_chunks::<2>().0.iter().map(|b| u16::from_le_bytes(*b)).collect();
        let rgb16 = image.to_rgb16().unwrap();
        assert_eq!(rgb16.data.len(), reference16.len());
        let max16 = rgb16.data.iter().zip(&reference16).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
        assert!(max16 <= 1, "{name}: f32 RGB16 vs independent f64 reference, max={max16}");
        assert!(rgb16.data.iter().filter(|&&v| v % 257 != 0).count() > rgb16.data.len() / 2, "{name}: higher-depth colour must survive");
    }
}
