//! HM-reference checks for SPS output crops, including cropped grid tiles.
use heic_decoder_rust::{decode, decode_one_image, isobmff::parse};

fn sample_hash(image: &heic_decoder_rust::DecodedImage) -> u64 {
    image.y.iter().chain(&image.cb).chain(&image.cr).fold(0xcbf29ce484222325, |hash, &sample| {
        (hash ^ sample as u64).wrapping_mul(0x100000001b3)
    })
}

#[test]
fn conformance_window_crops_filtered_planes_at_native_sampling() {
    for (name, bytes, format, depth, window, width, height, cw, ch, hash) in [
        ("420", &include_bytes!("fixtures/crop/crop420.heic")[..], 1, 8, [2,4,4,6], 58, 54, 29, 27, 0xa123b37cf99d368cu64),
        ("422", &include_bytes!("fixtures/crop/crop422.heic")[..], 2, 10, [2,4,3,5], 58, 56, 29, 56, 0xe20ce72f2c03163du64),
        ("444", &include_bytes!("fixtures/crop/crop444.heic")[..], 3, 12, [1,3,3,5], 60, 56, 60, 56, 0xbb1a073405ca218cu64),
    ] {
        let parsed = parse(bytes).unwrap();
        let (sps, pic) = decode_one_image(&parsed.images[0]).unwrap();
        // Reference prediction and the low-level API retain coded dimensions.
        assert_eq!((sps.pic_width_in_luma_samples, sps.pic_height_in_luma_samples), (64, 64), "{name}");
        assert_eq!((pic.y.width, pic.y.height), (64, 64), "{name}");
        assert_eq!(sps.conformance_window, window, "{name}");
        let image = decode(bytes).unwrap();
        assert_eq!((image.width, image.height, image.cb_width, image.cb_height), (width, height, cw, ch), "{name}");
        assert_eq!((image.chroma_format_idc, image.bit_depth_luma, image.bit_depth_chroma), (format, depth, depth), "{name}");
        assert_eq!((image.y.len(), image.cb.len(), image.cr.len()), (width*height, cw*ch, cw*ch), "{name}");
        assert_eq!(sample_hash(&image), hash, "{name}: all cropped samples must match HM");
    }
}

#[test]
fn cropped_tiles_are_assembled_before_rotation_and_clean_aperture() {
    let bytes = include_bytes!("fixtures/crop/grid422.heic");
    let parsed = parse(bytes).unwrap();
    let grid = parsed.grid.as_ref().unwrap();
    assert_eq!((grid.rows, grid.columns, grid.output_width, grid.output_height), (1,2,112,56));
    assert_eq!(parsed.images.len(), 2);
    assert_eq!(parsed.transforms.len(), 2);
    let image = decode(bytes).unwrap();
    assert_eq!((image.width, image.height, image.cb_width, image.cb_height), (40,84,40,42));
    assert_eq!(image.chroma_format_idc, 2);
    // Independent assembly of the HM-cropped tile planes, clipping the grid,
    // rotating CCW and selecting the aperture at luma (10,17).
    assert_eq!(sample_hash(&image), 0xf6e2008f2a517de3);
}
