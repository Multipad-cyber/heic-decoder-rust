//! Original HM oracles; see fixtures/alpha/README.md for provenance.
use heic_decoder_rust::{decode, isobmff::parse};

fn hash(samples: impl Iterator<Item = i32>) -> u64 {
    samples.fold(0xcbf29ce484222325, |hash, sample| (hash ^ sample as u64).wrapping_mul(0x100000001b3))
}

#[test]
fn monochrome_and_transquant_bypass_match_every_hm_sample() {
    for (name, data, chroma, depth, expected) in [
        ("mono8", &include_bytes!("fixtures/alpha/mono8.heic")[..], 0, 8, 0x82af5f05e57d3a52),
        ("mono10", &include_bytes!("fixtures/alpha/mono10.heic")[..], 0, 10, 0x65fa9e52bf25f24d),
        ("mono12", &include_bytes!("fixtures/alpha/mono12.heic")[..], 0, 12, 0x9613d8de1b04050a),
        ("lossless420", &include_bytes!("fixtures/alpha/lossless420.heic")[..], 1, 10, 0x5f5536b1a84f36e1),
        ("lossless422", &include_bytes!("fixtures/alpha/lossless422.heic")[..], 2, 12, 0xfed7154b652a6d65),
        ("lossless444", &include_bytes!("fixtures/alpha/lossless444.heic")[..], 3, 8, 0xab99e0505da9a472),
    ] {
        let image = decode(data).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!((image.width, image.height, image.chroma_format_idc, image.bit_depth_luma), (64, 64, chroma, depth));
        assert_eq!(hash(image.y.iter().chain(&image.cb).chain(&image.cr).copied()), expected, "{name}: all samples must match HM");
        if chroma == 0 {
            assert_eq!((image.cb_width, image.cb_height, image.cb.len(), image.cr.len()), (0, 0, 0, 0));
            assert!(image.to_rgb16().unwrap().data.as_chunks::<3>().0.iter().all(|pixel| pixel[0] == pixel[1] && pixel[1] == pixel[2]));
        }
        assert!(image.alpha.is_none());
    }
}

#[test]
fn alpha_grids_transforms_mixed_depth_and_premultiplication_are_preserved() {
    for (name, data, width, height, depth, expected, prem) in [
        ("rgba10", &include_bytes!("fixtures/alpha/rgba10.heic")[..], 64, 64, 10, 0x65fa9e52bf25f24d, false),
        ("rgba8alpha", &include_bytes!("fixtures/alpha/rgba8alpha.heic")[..], 64, 64, 8, 0x82af5f05e57d3a52, false),
        ("rgba-grid", &include_bytes!("fixtures/alpha/rgba-grid.heic")[..], 56, 100, 10, 0x92de4cd991cbf529, false),
        ("rgba-prem", &include_bytes!("fixtures/alpha/rgba-prem.heic")[..], 64, 64, 10, 0x65fa9e52bf25f24d, true),
    ] {
        assert!(parse(data).unwrap().alpha.is_some());
        let image = decode(data).unwrap();
        assert_eq!((image.width, image.height), (width, height), "{name}");
        let alpha = image.alpha.as_ref().unwrap();
        assert_eq!((alpha.bit_depth, alpha.premultiplied), (depth, prem));
        assert_eq!(alpha.samples.len(), width * height);
        assert_eq!(hash(alpha.samples.iter().copied()), expected, "{name}: native alpha samples must match HM");
        let rgba8 = image.to_rgba8().unwrap();
        let rgba16 = image.to_rgba16().unwrap();
        let rgb = image.to_rgb16().unwrap();
        assert_eq!((rgba8.data.len(), rgba16.data.len()), (width * height * 4, width * height * 4));
        assert_eq!((rgba8.premultiplied, rgba16.premultiplied), (prem, prem));
        let max = (1u64 << depth) - 1;
        for (((pixel8, pixel16), rgb), &sample) in rgba8.data.as_chunks::<4>().0.iter().zip(rgba16.data.as_chunks::<4>().0.iter()).zip(rgb.data.as_chunks::<3>().0.iter()).zip(&alpha.samples) {
            assert_eq!(&pixel16[..3], rgb, "RGBA must preserve RGB conversion");
            assert_eq!(pixel8[3], ((sample as u64 * 255 + max / 2) / max) as u8);
            assert_eq!(pixel16[3], ((sample as u64 * 65535 + max / 2) / max) as u16);
        }
        if depth == 10 { assert!(rgba16.data.as_chunks::<4>().0.iter().any(|p| p[3] % 257 != 0)); }
    }
}

#[test]
fn unrelated_depth_images_are_ignored_and_malformed_alpha_is_rejected() {
    let original = include_bytes!("fixtures/alpha/rgba10.heic");
    let mut depth = original.to_vec();
    let urn = b"urn:mpeg:hevc:2015:auxid:1";
    let at = depth.windows(urn.len()).position(|w| w == urn).unwrap();
    depth[at + urn.len() - 1] = b'2';
    assert!(decode(&depth).unwrap().alpha.is_none());
    // The NUL terminator is required; do not silently treat malformed alpha
    // metadata as an opaque image.
    let mut truncated = original.to_vec();
    truncated[at + urn.len()] = b'x';
    assert!(parse(&truncated).is_err());
    // Remove only the alpha grid's clap association. Its transformed size
    // becomes 64x128 while colour remains 56x100; fail instead of misaligning.
    let mut grid = include_bytes!("fixtures/alpha/rgba-grid.heic").to_vec();
    let association = [5, 0x86, 0x87, 0x89, 0x88, 0x85];
    let at = grid.windows(association.len()).position(|w| w == association).unwrap();
    grid[at + 4] = 0x86;
    assert!(decode(&grid).is_err());
}
