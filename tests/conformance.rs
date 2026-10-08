//! Bit-exact checks against HM output for files from the Nokia HEIF
//! conformance suite (<https://github.com/nokiatech/heif_conformance>),
//! which is not vendored. Run with:
//! `HEIF_CONFORMANCE_DIR=/path/to/conformance_files cargo test -- --ignored`
use std::{env, fs, path::PathBuf};

#[test]
#[ignore = "requires HEIF_CONFORMANCE_DIR pointing to Nokia conformance_files"]
fn enhancement_layer_nalus_are_not_base_picture_slices() {
    let root = PathBuf::from(env::var_os("HEIF_CONFORMANCE_DIR").expect("set HEIF_CONFORMANCE_DIR"));
    for (name, width, height, expected_hash) in [
        ("multilayer001.heic", 1024, 512, 0x48b788797ba3202cu64),
        ("multilayer002.heic", 512, 256, 0x6dea88498bae984bu64),
        ("multilayer005.heic", 512, 256, 0x64e0d4bdbd9409cbu64),
    ] {
        let data = fs::read(root.join(name)).expect("read fixture");
        let parsed = heic_decoder_rust::isobmff::parse(&data).expect("parse fixture");
        let image = &parsed.images[0];
        let nalus = heic_decoder_rust::isobmff::split_length_prefixed_nalus(
            &image.data, image.hevc_config.length_size_minus_one as usize + 1,
        ).unwrap();
        assert!(nalus.iter().any(|nal| {
            let (header, _) = heic_decoder_rust::hevc::nal::parse_nal_header(nal).unwrap();
            header.nuh_layer_id == 1 && header.nal_unit_type <= 31
        }));
        let img = heic_decoder_rust::decode(&data).expect("decode base layer");
        assert_eq!((img.width, img.height), (width, height), "{name}");
        let hash = img.y.iter().chain(&img.cb).chain(&img.cr).fold(0xcbf29ce484222325u64, |hash, &sample| {
            (hash ^ sample as u64).wrapping_mul(0x100000001b3)
        });
        assert_eq!(hash, expected_hash, "{name}: base-layer Y/Cb/Cr must match HM");
    }
}

#[test]
#[ignore = "requires HEIF_CONFORMANCE_DIR pointing to Nokia conformance_files"]
fn nested_transform_splits_match_hm() {
    let root = PathBuf::from(env::var_os("HEIF_CONFORMANCE_DIR").expect("set HEIF_CONFORMANCE_DIR"));
    let data = fs::read(root.join("C005.heic")).expect("read external fixture");
    let parsed = heic_decoder_rust::isobmff::parse(&data).expect("parse fixture");
    let (sps, pic) = heic_decoder_rust::decode_one_image(&parsed.images[0]).expect("decode nested transform tree");
    assert_eq!((sps.pic_width_in_luma_samples, sps.pic_height_in_luma_samples), (128, 72));
    assert_eq!((sps.bit_depth_luma, sps.bit_depth_chroma), (8, 8));
    let samples: Vec<_> = pic.y.into_samples().into_iter().chain(pic.cb.into_samples()).chain(pic.cr.into_samples()).collect();
    assert_eq!(samples.len(), 13_824);
    // FNV-1a of HM's independently decoded, planar 8-bit Y/Cb/Cr output.
    let hash = samples.into_iter().fold(0xcbf29ce484222325u64, |hash, v| {
        assert!((0..=255).contains(&v));
        (hash ^ v as u64).wrapping_mul(0x100000001b3)
    });
    assert_eq!(hash, 0xeef844d09276f0a5);
}

// A delta signalled after zero-residual CUs must not retroactively change
// those CUs' QPs. Check independent HM output, including loop filters.
#[test]
#[ignore = "requires HEIF_CONFORMANCE_DIR pointing to Nokia conformance_files"]
fn delayed_qp_delta_preserves_earlier_cus() {
    let root = PathBuf::from(env::var_os("HEIF_CONFORMANCE_DIR").expect("set HEIF_CONFORMANCE_DIR"));
    for (name, expected_len, expected_hash) in [
        ("C004.heic", 1382400, 0x38935d9ee744e2a0u64),
        ("C007.heic", 1382400, 0x38935d9ee744e2a0u64),
        ("C012.heic", 1382400, 0x38935d9ee744e2a0u64),
        ("C022.heic", 1382400, 0x38935d9ee744e2a0u64),
        ("C023.heic", 1382400, 0x38935d9ee744e2a0u64),
        ("C045.heic", 1179648, 0x17b7798cb4580f47u64),
        ("C047.heic", 1382400, 0xcf9c70b3701abd46u64),
        ("C048.heic", 1382400, 0xcf9c70b3701abd46u64),
        ("C049.heic", 3110400, 0x5d5515f10d4ad45fu64),
    ] {
        let data = fs::read(root.join(name)).expect("read external fixture");
        let parsed = heic_decoder_rust::isobmff::parse(&data).expect("parse fixture");
        let (_, pic) = heic_decoder_rust::decode_one_image(&parsed.images[0]).expect("decode first image");
        let samples: Vec<_> = pic.y.into_samples().into_iter().chain(pic.cb.into_samples()).chain(pic.cr.into_samples()).collect();
        assert_eq!(samples.len(), expected_len, "{name}");
        let hash = samples.into_iter().fold(0xcbf29ce484222325u64, |hash, v| {
            assert!((0..=255).contains(&v));
            (hash ^ v as u64).wrapping_mul(0x100000001b3)
        });
        assert_eq!(hash, expected_hash, "{name}: must match HM's Y/Cb/Cr samples");
    }
}

#[test]
#[ignore = "requires HEIF_CONFORMANCE_DIR pointing to Nokia conformance_files"]
fn clean_aperture_matches_reference_container_output() {
    let root = PathBuf::from(env::var_os("HEIF_CONFORMANCE_DIR").expect("set HEIF_CONFORMANCE_DIR"));
    // C013 also checked against HM samples cropped at (490,210), 300x300.
    // MIAF007 includes both clean-aperture cropping and rotation.
    for (name, width, height, expected_hash) in [
        ("C013.heic", 300, 300, 0x78fb02c35d5e1a41u64),
        ("MIAF007.heic", 360, 640, 0x6086fbe32892c619u64),
    ] {
        let data = fs::read(root.join(name)).expect("read fixture");
        let img = heic_decoder_rust::decode(&data).expect("decode with crop");
        assert_eq!((img.width, img.height), (width, height), "{name}");
        assert_eq!((img.cb_width, img.cb_height), (width / 2, height / 2));
        assert_eq!(img.y.len() + img.cb.len() + img.cr.len(), width * height * 3 / 2);
        let hash = img.y.iter().chain(&img.cb).chain(&img.cr).fold(0xcbf29ce484222325u64, |hash, &v| {
            (hash ^ v as u64).wrapping_mul(0x100000001b3)
        });
        assert_eq!(hash, expected_hash, "{name}");
    }
}

#[test]
#[ignore = "requires HEIF_CONFORMANCE_DIR pointing to Nokia conformance_files"]
fn identity_items_match_reference_container_output() {
    let root = PathBuf::from(env::var_os("HEIF_CONFORMANCE_DIR").expect("set HEIF_CONFORMANCE_DIR"));
    // Independent heic-main container output, including source and iden transforms.
    for (name, width, height, expected_hash) in [
        ("C008.heic", 720, 1280, 0x231e2c56176b1c82u64),
        ("C014.heic", 300, 300, 0xfa999ff7fcb06c0bu64),
    ] {
        let data = fs::read(root.join(name)).expect("read fixture");
        let img = heic_decoder_rust::decode(&data).expect("decode identity item");
        assert_eq!((img.width, img.height), (width, height), "{name}");
        assert_eq!((img.cb_width, img.cb_height), (width / 2, height / 2));
        let hash = img.y.iter().chain(&img.cb).chain(&img.cr).fold(0xcbf29ce484222325u64, |hash, &v| {
            (hash ^ v as u64).wrapping_mul(0x100000001b3)
        });
        assert_eq!(hash, expected_hash, "{name}");
    }
    let data = fs::read(root.join("C039.heic")).expect("read nested identity fixture");
    let parsed = heic_decoder_rust::isobmff::parse(&data).expect("resolve nested identity chain");
    assert_eq!(parsed.images.len(), 1);
    let img = heic_decoder_rust::decode(&data).expect("nested crop and rotation");
    assert_eq!((img.width, img.height, img.cb_width, img.cb_height), (150, 150, 75, 75));
    let hash = img.y.iter().chain(&img.cb).chain(&img.cr).fold(0xcbf29ce484222325u64, |hash, &v| {
        (hash ^ v as u64).wrapping_mul(0x100000001b3)
    });
    assert_eq!(hash, 0x115b1d7836d2e0ff);
}

#[test]
#[ignore = "requires HEIF_CONFORMANCE_DIR pointing to Nokia conformance_files"]
fn sequence_sync_frames_match_reference() {
    let root = PathBuf::from(env::var_os("HEIF_CONFORMANCE_DIR").expect("set HEIF_CONFORMANCE_DIR"));
    for (name, count, width, height, expected_hash) in [
        ("C026.heic", 8, 1280, 720, 0xf9d8eead53602f78u64),
        ("C027.heic", 16, 1280, 720, 0xbd81bc75d57f0c08u64),
        ("C028.heic", 16, 1280, 720, 0xbd81bc75d57f0c08u64),
        ("C029.heic", 8, 1280, 720, 0xf9d8eead53602f78u64),
        ("C030.heic", 8, 1280, 720, 0xf9d8eead53602f78u64),
        ("C031.heic", 8, 1280, 720, 0xf9d8eead53602f78u64),
        ("C032.heic", 8, 1280, 720, 0xf9d8eead53602f78u64),
        ("C036.heic", 8, 1280, 720, 0xf9d8eead53602f78u64),
        ("C037.heic", 8, 1280, 720, 0xf9d8eead53602f78u64),
        ("C038.heic", 8, 1280, 720, 0xf9d8eead53602f78u64),
        ("C041.heic", 9, 1920, 1080, 0x7ec28c2c2ca7ff80u64),
    ] {
        let data = fs::read(root.join(name)).expect("read fixture");
        let parsed = heic_decoder_rust::isobmff::parse(&data).expect("resolve sequence");
        let frame = parsed.sequence.expect("sequence selection metadata");
        assert_eq!((frame.track_id, frame.sample_number, frame.sample_count), (1, 1, count), "{name}");
        let img = heic_decoder_rust::decode(&data).expect("decode selected sync frame");
        assert_eq!((img.width, img.height), (width, height));
        let hash = img.y.iter().chain(&img.cb).chain(&img.cr).fold(0xcbf29ce484222325u64, |hash, &v| {
            (hash ^ v as u64).wrapping_mul(0x100000001b3)
        });
        assert_eq!(hash, expected_hash, "{name}");
    }
}

#[test]
#[ignore = "requires HEIF_CONFORMANCE_DIR pointing to Nokia conformance_files"]
fn full_resolution_chroma_nxn_matches_hm() {
    let root = PathBuf::from(env::var_os("HEIF_CONFORMANCE_DIR").expect("set HEIF_CONFORMANCE_DIR"));
    let data = fs::read(root.join("MIAF003.heic")).expect("read fixture");
    let img = heic_decoder_rust::decode(&data).expect("decode 4:4:4 NxN image");
    assert_eq!((img.width, img.height, img.cb_width, img.cb_height), (2048, 2048, 2048, 2048));
    assert_eq!((img.chroma_format_idc, img.bit_depth_luma, img.bit_depth_chroma), (3, 8, 8));
    assert_eq!((img.y.len(), img.cb.len(), img.cr.len()), (4_194_304, 4_194_304, 4_194_304));
    // All three full-resolution planes, independently decoded by HM.
    let hash = img.y.iter().chain(&img.cb).chain(&img.cr).fold(0xcbf29ce484222325u64, |hash, &v| {
        assert!((0..=255).contains(&v));
        (hash ^ v as u64).wrapping_mul(0x100000001b3)
    });
    assert_eq!(hash, 0xc427e42b85176a4c);
}

#[test]
#[ignore = "requires HEIF_CONFORMANCE_DIR pointing to Nokia conformance_files"]
fn predictive_primary_matches_hm() {
    let root = PathBuf::from(env::var_os("HEIF_CONFORMANCE_DIR").expect("set HEIF_CONFORMANCE_DIR"));
    let data = fs::read(root.join("C044.heic")).expect("read fixture");
    let parsed = heic_decoder_rust::isobmff::parse(&data).expect("resolve prediction dependency");
    assert!(parsed.reference_image.is_some());
    // The primary cannot be correctly decoded without its reference.
    assert!(heic_decoder_rust::decode_one_image(&parsed.images[0]).is_err());
    let img = heic_decoder_rust::decode(&data).expect("decode predictive primary");
    assert_eq!((img.width, img.height, img.cb_width, img.cb_height), (1280, 720, 640, 360));
    let hash = img.y.iter().chain(&img.cb).chain(&img.cr).fold(0xcbf29ce484222325u64, |hash, &v| {
        (hash ^ v as u64).wrapping_mul(0x100000001b3)
    });
    // HM POC 1 after decoding the IDR dependency, not a standalone P picture.
    assert_eq!(hash, 0x3f6d19e970c1cd10);
}
