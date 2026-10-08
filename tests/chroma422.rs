//! Independently encoded/decoded HM fixtures; see fixtures/422/README.md.
use heic_decoder_rust::{decode, decode_one_image, isobmff::{parse, split_length_prefixed_nalus}};

#[test]
fn chroma_422_matches_hm_including_loop_filters() {
    for (name, data, width, height, depth, expected_hash) in [
        ("pattern8", include_bytes!("fixtures/422/pattern8.heic").as_slice(), 64, 64, 8, 0x77e549612970dc84u64),
        ("pattern10", include_bytes!("fixtures/422/pattern10.heic").as_slice(), 96, 80, 10, 0xa99b07a692b1bbe2u64),
        ("smooth10", include_bytes!("fixtures/422/smooth10.heic").as_slice(), 128, 128, 10, 0x8fdac98cc8c1c947u64),
        ("flat10", include_bytes!("fixtures/422/flat10.heic").as_slice(), 64, 64, 10, 0x9664b27478157325u64),
        ("skip10", include_bytes!("fixtures/422/skip10.heic").as_slice(), 64, 64, 10, 0x3977b95084ec1e37u64),
        ("pattern12", include_bytes!("fixtures/422/pattern12.heic").as_slice(), 64, 64, 12, 0x77dc7f4e9bf5bf5au64),
        ("slices10", include_bytes!("fixtures/422/slices10.heic").as_slice(), 96, 96, 10, 0xc3120db030308147u64),
        ("scaled12", include_bytes!("fixtures/422/scaled12.heic").as_slice(), 64, 64, 12, 0x87d3f64ae3ee4aaau64),
    ] {
        let img = decode(data).unwrap_or_else(|err| panic!("{name}: {err}"));
        assert_eq!((img.width, img.height, img.cb_width, img.cb_height), (width, height, width / 2, height), "{name}");
        assert_eq!((img.chroma_format_idc, img.bit_depth_luma, img.bit_depth_chroma), (2, depth, depth));
        assert_eq!((img.y.len(), img.cb.len(), img.cr.len()), (width * height, width * height / 2, width * height / 2));
        let hash = img.y.iter().chain(&img.cb).chain(&img.cr).fold(0xcbf29ce484222325u64, |hash, &sample| {
            assert!((0..1 << depth).contains(&sample), "{name}: sample out of range");
            (hash ^ sample as u64).wrapping_mul(0x100000001b3)
        });
        assert_eq!(hash, expected_hash, "{name}: all Y/Cb/Cr samples must match HM");
    }
}

#[test]
fn independent_slices_require_complete_ordered_picture() {
    let parsed = parse(include_bytes!("fixtures/422/slices10.heic")).unwrap();
    let mut image = parsed.images[0].clone();
    let size = image.hevc_config.length_size_minus_one as usize + 1;
    let nalus = split_length_prefixed_nalus(&image.data, size).unwrap();
    let slice_indices: Vec<_> = nalus.iter().enumerate().filter(|(_, n)| (n[0] >> 1) & 63 < 32).map(|(i, _)| i).collect();
    assert_eq!(slice_indices.len(), 3);
    let original = image.data.clone();
    // Keep valid syntax, but remove the final slice: no silent partial image.
    let last = nalus[slice_indices[2]].len() + size;
    image.data.truncate(original.len() - last);
    assert!(decode_one_image(&image).is_err());
    // Repeat a slice address: overlapping reconstruction is invalid.
    image.data = original;
    let duplicate = &nalus[slice_indices[1]];
    image.data.extend_from_slice(&(duplicate.len() as u32).to_be_bytes());
    image.data.extend_from_slice(duplicate);
    assert!(decode_one_image(&image).is_err());
}
