//! Inspect a resolved HEIF item and export its coded stream for HM comparison.
//! Usage: reference_image <file.heic> <item-index> <out.hevc> [out.yuv]
use heic_decoder::{bitreader::strip_emulation_prevention, hevc::{nal::parse_nal_header, params::{parse_sps, parse_pps}}, isobmff::{parse, split_length_prefixed_nalus}};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 4 { return Err("usage: reference_image <file.heic> <item-index> <out.hevc> [out.yuv]".into()); }
    let parsed = parse(&std::fs::read(&args[1])?)?;
    println!("grid={:?}, transforms={:?}, items={}", parsed.grid, parsed.transforms, parsed.images.len());
    let image = parsed.images.get(args[2].parse::<usize>()?).ok_or("item index out of range")?;
    let mut annexb = Vec::new();
    for kind in [32, 33, 34] {
        for nal in image.hevc_config.nalus_of_type(kind) {
            let (_, payload) = parse_nal_header(nal)?;
            let rbsp = strip_emulation_prevention(payload);
            if kind == 33 {
                let sps = parse_sps(&rbsp)?;
                println!("SPS: {}x{}, {} / {} bits, chroma={}, CTU={}, transform depth={}, scaling lists={}, SAO={}", sps.pic_width_in_luma_samples, sps.pic_height_in_luma_samples, sps.bit_depth_luma, sps.bit_depth_chroma, sps.chroma_format_idc, 1 << sps.ctb_log2_size_y(), sps.max_transform_hierarchy_depth_intra, sps.scaling_list_enabled, sps.sample_adaptive_offset_enabled);
            }
            if kind == 34 { println!("PPS: {:?}", parse_pps(&rbsp)?); }
            annexb.extend_from_slice(&[0, 0, 0, 1]);
            annexb.extend_from_slice(nal);
        }
    }
    for nal in split_length_prefixed_nalus(&image.data, image.hevc_config.length_size_minus_one as usize + 1)? {
        annexb.extend_from_slice(&[0, 0, 0, 1]);
        annexb.extend_from_slice(&nal);
    }
    std::fs::write(&args[3], annexb)?;
    if let Some(out) = args.get(4) {
        let (sps, pic) = heic_decoder::decode_one_image(image)?;
        let mut raw = Vec::new();
        for (samples, depth) in [(pic.y.into_samples(), sps.bit_depth_luma), (pic.cb.into_samples(), sps.bit_depth_chroma), (pic.cr.into_samples(), sps.bit_depth_chroma)] {
            for sample in samples {
                if depth > 8 { raw.extend_from_slice(&(sample as u16).to_le_bytes()); }
                else { raw.push(sample as u8); }
            }
        }
        std::fs::write(out, raw)?;
    }
    Ok(())
}
