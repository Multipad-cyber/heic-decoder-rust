//! SPS/PPS parsing for still images and single-reference P pictures.
//! Retains short-term reference sets; unsupported long-term and multilayer
//! features are rejected explicitly.

use crate::bitreader::BitReader;
use crate::error::{HeifError, Result};

/// The 12-byte "general" profile/tier/level prefix (clause 7.3.3),
/// sub-layer profile/level data excluded. This decoder requires
/// `sps_max_sub_layers_minus1 == 0` (no temporal sub-layers), which is
/// what every still-image HEIC encoder in practice emits, so the
/// sub-layer PTL loop is never reached.
fn skip_profile_tier_level(r: &mut BitReader<'_>) -> Result<()> {
    r.bits(2)?; // general_profile_space
    r.bits(1)?; // general_tier_flag
    r.bits(5)?; // general_profile_idc
    r.bits(32)?; // general_profile_compatibility_flag[32]
    r.bits(4)?; // progressive/interlaced/non_packed/frame_only constraint flags
    for _ in 0..43 {
        r.bit()?; // profile-specific constraint flags / reserved, fixed at 43 bits either way
    }
    r.bit()?; // general_inbld_flag / reserved_zero_bit
    r.bits(8)?; // general_level_idc
    Ok(())
}

/// `st_ref_pic_set(stRpsIdx)` (clause 7.3.7), parsed only to consume the
/// right number of bits and reach the SPS fields after it — this
/// decoder is intra-only, so the actual reference-picture-set contents
/// are never used. Returns `NumDeltaPocs[stRpsIdx]` (needed by any later
/// set that inter-predicts from this one).
///
/// Real encoders — including Apple's native HEIC encoder, confirmed by
/// decoding its own output — emit a (typically unused, boilerplate)
/// short-term RPS even for a single intra image, so this can't be
/// skipped as "never happens in practice."
/// Short-term references in normative negative-then-positive POC order.
#[derive(Debug, Clone, Default)]
pub struct ShortTermRefs(pub Vec<(i32, bool)>);

pub fn parse_short_term_refs(r: &mut BitReader<'_>, index: usize, sets: &[ShortTermRefs], in_slice: bool) -> Result<ShortTermRefs> {
    let mut values = Vec::new();
    if index != 0 && r.flag()? {
        let back = if in_slice { r.ue()?.checked_add(1).ok_or(HeifError::MalformedHevc("RPS index overflow"))? as usize } else { 1 };
        let previous = index.checked_sub(back).and_then(|i| sets.get(i)).ok_or(HeifError::MalformedHevc("RPS predictor index"))?;
        let negative = r.flag()?;
        let magnitude = i32::try_from(r.ue()?).ok().and_then(|v| v.checked_add(1)).ok_or(HeifError::MalformedHevc("RPS delta overflow"))?;
        let delta = if negative { -magnitude } else { magnitude };
        for &(poc, _) in previous.0.iter().chain(std::iter::once(&(0, false))) {
            let used = r.flag()?;
            let retained = used || r.flag()?;
            let poc = poc.checked_add(delta).ok_or(HeifError::MalformedHevc("RPS POC overflow"))?;
            if retained && poc != 0 { values.push((poc, used)); }
        }
        values.sort_by_key(|&(poc, _)| if poc < 0 { (0, -i64::from(poc)) } else { (1, i64::from(poc)) });
    } else {
        let negative = r.ue()?;
        let positive = r.ue()?;
        if negative > 16 || positive > 16 || negative + positive > 16 { return Err(HeifError::Unsupported("RPS: too many reference pictures")); }
        for (count, sign) in [(negative, -1i32), (positive, 1i32)] {
            let mut poc = 0i32;
            for _ in 0..count {
                let step = i32::try_from(r.ue()?).ok().and_then(|v| v.checked_add(1)).ok_or(HeifError::MalformedHevc("RPS delta overflow"))?;
                poc = poc.checked_add(sign * step).ok_or(HeifError::MalformedHevc("RPS POC overflow"))?;
                values.push((poc, r.flag()?));
            }
        }
    }
    if values.len() > 16 { return Err(HeifError::Unsupported("RPS: too many reference pictures")); }
    Ok(ShortTermRefs(values))
}

pub use super::scaling::ScalingListData;
use super::scaling::parse_scaling_list_data;

#[derive(Debug, Clone)]
pub struct Sps {
    pub log2_max_pic_order_cnt_lsb: u32,
    pub short_term_refs: Vec<ShortTermRefs>,
    pub temporal_mvp_enabled: bool,
    pub chroma_format_idc: u32,
    pub pic_width_in_luma_samples: u32,
    pub pic_height_in_luma_samples: u32,
    pub bit_depth_luma: u32,
    pub bit_depth_chroma: u32,
    pub log2_min_luma_coding_block_size: u32,
    pub log2_diff_max_min_luma_coding_block_size: u32,
    pub log2_min_luma_transform_block_size: u32,
    pub log2_diff_max_min_luma_transform_block_size: u32,
    pub max_transform_hierarchy_depth_inter: u32,
    pub max_transform_hierarchy_depth_intra: u32,
    pub scaling_list_enabled: bool,
    pub scaling_list: Option<ScalingListData>,
    pub amp_enabled: bool,
    pub sample_adaptive_offset_enabled: bool,
    pub pcm_enabled: bool,
    pub strong_intra_smoothing_enabled: bool,
}

impl Sps {
    pub fn ctb_log2_size_y(&self) -> u32 {
        self.log2_min_luma_coding_block_size + self.log2_diff_max_min_luma_coding_block_size
    }

    pub fn min_tb_log2_size_y(&self) -> u32 {
        self.log2_min_luma_transform_block_size
    }

    pub fn max_tb_log2_size_y(&self) -> u32 {
        self.log2_min_luma_transform_block_size + self.log2_diff_max_min_luma_transform_block_size
    }

    pub fn sub_width_c(&self) -> u32 {
        if self.chroma_format_idc == 1 || self.chroma_format_idc == 2 {
            2
        } else {
            1
        }
    }

    pub fn sub_height_c(&self) -> u32 {
        if self.chroma_format_idc == 1 {
            2
        } else {
            1
        }
    }
}

pub fn parse_sps(rbsp: &[u8]) -> Result<Sps> {
    let mut r = BitReader::new(rbsp);
    r.bits(4)?; // sps_video_parameter_set_id
    let max_sub_layers_minus1 = r.bits(3)?;
    if max_sub_layers_minus1 != 0 {
        return Err(HeifError::Unsupported("SPS: temporal sub-layers not supported"));
    }
    r.bit()?; // sps_temporal_id_nesting_flag
    skip_profile_tier_level(&mut r)?;
    r.ue()?; // sps_seq_parameter_set_id
    let chroma_format_idc = r.ue()?;
    if chroma_format_idc == 3 {
        r.bit()?; // separate_colour_plane_flag
    }
    let pic_width_in_luma_samples = r.ue()?;
    let pic_height_in_luma_samples = r.ue()?;
    if pic_width_in_luma_samples == 0 || pic_height_in_luma_samples == 0 {
        return Err(HeifError::MalformedHevc("SPS: zero picture dimension"));
    }
    if u64::from(pic_width_in_luma_samples) * u64::from(pic_height_in_luma_samples) > crate::MAX_IMAGE_PIXELS {
        return Err(HeifError::LimitExceeded("SPS: picture larger than MAX_IMAGE_PIXELS"));
    }
    if r.flag()? {
        // conformance_window_flag: four ue(v) offsets we don't need for
        // decode (they only crop the output).
        r.ue()?;
        r.ue()?;
        r.ue()?;
        r.ue()?;
    }
    // Clause 7.4.3.2.1: bit_depth_*_minus8 is in 0..=8.
    let bit_depth_luma = r.ue()?.checked_add(8).filter(|v| *v <= 16).ok_or(HeifError::MalformedHevc("SPS: luma bit depth"))?;
    let bit_depth_chroma = r.ue()?.checked_add(8).filter(|v| *v <= 16).ok_or(HeifError::MalformedHevc("SPS: chroma bit depth"))?;
    let log2_max_pic_order_cnt_lsb = r.ue()?.checked_add(4).filter(|v| *v <= 16).ok_or(HeifError::MalformedHevc("SPS: POC width"))?;
    r.flag()?; // sps_sub_layer_ordering_info_present_flag
    // With max_sub_layers_minus1 == 0 (enforced above), both possible
    // loop starts are zero and exactly one ordering-information triple
    // follows. The flag changes the start layer, not the number of layers.
    r.ue()?; // sps_max_dec_pic_buffering_minus1[0]
    r.ue()?; // sps_max_num_reorder_pics[0]
    r.ue()?; // sps_max_latency_increase_plus1[0]
    // Clause 7.4.3.2.1 block-size constraints: CtbLog2SizeY in 4..=6,
    // MinTbLog2SizeY < MinCbLog2SizeY, MaxTbLog2SizeY <= Min(CtbLog2SizeY, 5),
    // and transform depths at most CtbLog2SizeY - MinTbLog2SizeY.
    let log2_min_luma_coding_block_size = r.ue()?.checked_add(3).ok_or(HeifError::MalformedHevc("SPS: coding block size"))?;
    let log2_diff_max_min_luma_coding_block_size = r.ue()?;
    let ctb_log2 = log2_min_luma_coding_block_size.checked_add(log2_diff_max_min_luma_coding_block_size)
        .filter(|v| (4..=6).contains(v)).ok_or(HeifError::MalformedHevc("SPS: CTB size"))?;
    let log2_min_luma_transform_block_size = r.ue()?.checked_add(2).ok_or(HeifError::MalformedHevc("SPS: transform block size"))?;
    let log2_diff_max_min_luma_transform_block_size = r.ue()?;
    let max_tb_log2 = log2_min_luma_transform_block_size.checked_add(log2_diff_max_min_luma_transform_block_size)
        .ok_or(HeifError::MalformedHevc("SPS: transform block size"))?;
    if log2_min_luma_transform_block_size >= log2_min_luma_coding_block_size || max_tb_log2 > ctb_log2.min(5) {
        return Err(HeifError::MalformedHevc("SPS: transform block size"));
    }
    let max_depth = ctb_log2 - log2_min_luma_transform_block_size;
    let max_transform_hierarchy_depth_inter = r.ue()?;
    let max_transform_hierarchy_depth_intra = r.ue()?;
    if max_transform_hierarchy_depth_inter > max_depth || max_transform_hierarchy_depth_intra > max_depth {
        return Err(HeifError::MalformedHevc("SPS: transform hierarchy depth"));
    }
    let scaling_list_enabled = r.flag()?;
    let scaling_list = if scaling_list_enabled {
        let present = r.flag()?;
        if present { Some(parse_scaling_list_data(&mut r)?) } else { None }
    } else {
        None
    };
    let amp_enabled = r.flag()?;
    let sample_adaptive_offset_enabled = r.flag()?;
    let pcm_enabled = r.flag()?;
    if pcm_enabled {
        return Err(HeifError::Unsupported("SPS: pcm_enabled_flag not supported"));
    }
    let num_short_term_ref_pic_sets = r.ue()?;
    if num_short_term_ref_pic_sets > 64 {
        return Err(HeifError::Unsupported("SPS: implausible num_short_term_ref_pic_sets"));
    }
    let mut short_term_refs = Vec::new();
    for index in 0..num_short_term_ref_pic_sets as usize {
        short_term_refs.push(parse_short_term_refs(&mut r, index, &short_term_refs, false)?);
    }
    let long_term_ref_pics_present = r.flag()?;
    if long_term_ref_pics_present {
        return Err(HeifError::Unsupported("SPS: long-term reference pictures not supported"));
    }
    let temporal_mvp_enabled = r.flag()?;
    let strong_intra_smoothing_enabled = r.flag()?;

    Ok(Sps {
        log2_max_pic_order_cnt_lsb, short_term_refs, temporal_mvp_enabled,
        chroma_format_idc,
        pic_width_in_luma_samples,
        pic_height_in_luma_samples,
        bit_depth_luma,
        bit_depth_chroma,
        log2_min_luma_coding_block_size,
        log2_diff_max_min_luma_coding_block_size,
        log2_min_luma_transform_block_size,
        log2_diff_max_min_luma_transform_block_size,
        max_transform_hierarchy_depth_inter,
        max_transform_hierarchy_depth_intra,
        scaling_list_enabled,
        scaling_list,
        amp_enabled,
        sample_adaptive_offset_enabled,
        pcm_enabled,
        strong_intra_smoothing_enabled,
    })
}

#[derive(Debug, Clone)]
pub struct Pps {
    pub sao_offset_scale: [u32; 2],
    pub default_l0_references: u32,
    pub weighted_pred: bool,
    pub lists_modification_present: bool,
    pub log2_parallel_merge_level: u32,
    pub dependent_slice_segments_enabled: bool,
    pub output_flag_present: bool,
    pub num_extra_slice_header_bits: u32,
    pub sign_data_hiding_enabled: bool,
    pub cabac_init_present: bool,
    pub init_qp: i32,
    pub constrained_intra_pred: bool,
    pub transform_skip_enabled: bool,
    pub scaling_list: Option<ScalingListData>,
    pub cu_qp_delta_enabled: bool,
    pub diff_cu_qp_delta_depth: u32,
    pub cb_qp_offset: i32,
    pub cr_qp_offset: i32,
    pub slice_chroma_qp_offsets_present: bool,
    pub transquant_bypass_enabled: bool,
    pub tiles_enabled: bool,
    pub entropy_coding_sync_enabled: bool,
    pub num_tile_columns: u32,
    pub num_tile_rows: u32,
    pub loop_filter_across_slices_enabled: bool,
    pub deblocking_filter_override_enabled: bool,
    pub deblocking_filter_disabled: bool,
    pub beta_offset_div2: i32,
    pub tc_offset_div2: i32,
    pub slice_segment_header_extension_present: bool,
}

pub fn parse_pps(rbsp: &[u8]) -> Result<Pps> {
    let mut r = BitReader::new(rbsp);
    r.ue()?; // pps_pic_parameter_set_id
    r.ue()?; // pps_seq_parameter_set_id
    let dependent_slice_segments_enabled = r.flag()?;
    let output_flag_present = r.flag()?;
    let num_extra_slice_header_bits = r.bits(3)?;
    let sign_data_hiding_enabled = r.flag()?;
    let cabac_init_present = r.flag()?;
    let default_l0_references = r.ue()?.checked_add(1).filter(|v| *v <= 15).ok_or(HeifError::MalformedHevc("PPS: reference count"))?;
    r.ue()?; // num_ref_idx_l1_default_active_minus1
    let init_qp = r.se()? + 26;
    let constrained_intra_pred = r.flag()?;
    let transform_skip_enabled = r.flag()?;
    let cu_qp_delta_enabled = r.flag()?;
    let diff_cu_qp_delta_depth = if cu_qp_delta_enabled { r.ue()? } else { 0 };
    let cb_qp_offset = r.se()?;
    let cr_qp_offset = r.se()?;
    let slice_chroma_qp_offsets_present = r.flag()?;
    let weighted_pred = r.flag()?;
    r.bit()?; // weighted_bipred_flag
    let transquant_bypass_enabled = r.flag()?;
    let tiles_enabled = r.flag()?;
    let entropy_coding_sync_enabled = r.flag()?;
    let (mut num_tile_columns, mut num_tile_rows) = (1u32, 1u32);
    if tiles_enabled {
        num_tile_columns = r.ue()? + 1;
        num_tile_rows = r.ue()? + 1;
        let uniform_spacing = r.flag()?;
        if !uniform_spacing {
            for _ in 0..num_tile_columns.saturating_sub(1) {
                r.ue()?;
            }
            for _ in 0..num_tile_rows.saturating_sub(1) {
                r.ue()?;
            }
        }
        r.bit()?; // loop_filter_across_tiles_enabled_flag
    }
    if num_tile_columns != 1 || num_tile_rows != 1 {
        return Err(HeifError::Unsupported("PPS: multi-tile pictures not supported yet"));
    }
    // Wavefront parallel processing (entropy_coding_sync_enabled_flag) IS
    // supported — see `hevc::ctu::decode_slice`'s per-row substream
    // handling. Real encoders, including Apple's, use it routinely for
    // full-resolution photos' grid tiles.

    let loop_filter_across_slices_enabled = r.flag()?;
    let deblocking_filter_control_present = r.flag()?;
    let mut deblocking_filter_override_enabled = false;
    let mut deblocking_filter_disabled = false;
    let mut beta_offset_div2 = 0i32;
    let mut tc_offset_div2 = 0i32;
    if deblocking_filter_control_present {
        deblocking_filter_override_enabled = r.flag()?;
        deblocking_filter_disabled = r.flag()?;
        if !deblocking_filter_disabled {
            beta_offset_div2 = r.se()?;
            tc_offset_div2 = r.se()?;
        }
    }
    let pps_scaling_list_data_present = r.flag()?;
    let scaling_list = if pps_scaling_list_data_present { Some(parse_scaling_list_data(&mut r)?) } else { None };
    let lists_modification_present = r.flag()?;
    let log2_parallel_merge_level = r.ue()?.checked_add(2).filter(|v| *v <= 6).ok_or(HeifError::MalformedHevc("PPS: merge level"))?;
    let slice_segment_header_extension_present = r.flag()?;
    // RExt's SAO offset scales default to zero, including at 12 bits.
    // Bit depth alone does not determine the coded offset's multiplier.
    let mut sao_offset_scale = [0; 2];
    if r.more_rbsp_data() && r.flag()? {
        let extensions = r.bits(8)?;
        if extensions & 0x80 != 0 {
            if transform_skip_enabled && r.ue()? != 0 {
                return Err(HeifError::Unsupported("PPS: transform skip larger than 4x4"));
            }
            if r.flag()? { return Err(HeifError::Unsupported("PPS: cross-component prediction")); }
            if r.flag()? { return Err(HeifError::Unsupported("PPS: chroma QP offset list")); }
            for scale in &mut sao_offset_scale {
                *scale = r.ue()?;
                if *scale > 6 { return Err(HeifError::MalformedHevc("PPS: SAO offset scale")); }
            }
        }
    }

    Ok(Pps {
        sao_offset_scale,
        default_l0_references, weighted_pred, lists_modification_present, log2_parallel_merge_level,
        dependent_slice_segments_enabled,
        output_flag_present,
        num_extra_slice_header_bits,
        sign_data_hiding_enabled,
        cabac_init_present,
        init_qp,
        constrained_intra_pred,
        transform_skip_enabled,
        scaling_list,
        cu_qp_delta_enabled,
        diff_cu_qp_delta_depth,
        cb_qp_offset,
        cr_qp_offset,
        slice_chroma_qp_offsets_present,
        transquant_bypass_enabled,
        tiles_enabled,
        entropy_coding_sync_enabled,
        num_tile_columns,
        num_tile_rows,
        loop_filter_across_slices_enabled,
        deblocking_filter_override_enabled,
        beta_offset_div2,
        tc_offset_div2,
        deblocking_filter_disabled,
        slice_segment_header_extension_present,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-assembled minimal SPS RBSP (profile_idc=1, 16x16 4:2:0 8-bit,
    /// CTB=16 (log2=4), min CB=8 (log2=3), transform 4..16, no exotic
    /// features enabled) — values chosen to be decodable by hand so this
    /// test is a real check of field extraction, not just "it doesn't
    /// crash."
    fn push_bits(bits: &mut Vec<u8>, v: u32, n: u32) {
        for i in (0..n).rev() {
            // `v` is only ever 32 bits wide; a shift of 32+ is a no-op bit
            // (always 0), not a panic.
            let bit = if i < 32 { (v >> i) & 1 } else { 0 };
            bits.push(bit as u8);
        }
    }

    fn push_ue(bits: &mut Vec<u8>, v: u32) {
        let code = v + 1;
        let nbits = 32 - code.leading_zeros();
        for _ in 0..nbits - 1 {
            bits.push(0);
        }
        push_bits(bits, code, nbits);
    }

    /// The SPS fields the validation tests vary; `Default` is the minimal SPS.
    struct SpsFields { width: u32, height: u32, bit_depth_minus8: u32, min_cb_minus3: u32, diff_cb: u32, min_tb_minus2: u32, diff_tb: u32, depth_intra: u32 }

    impl Default for SpsFields {
        fn default() -> Self {
            SpsFields { width: 15, height: 15, bit_depth_minus8: 0, min_cb_minus3: 0, diff_cb: 1, min_tb_minus2: 0, diff_tb: 1, depth_intra: 0 }
        }
    }

    fn minimal_sps_bits(ordering_info_present: bool) -> Vec<u8> {
        sps_bits(ordering_info_present, &SpsFields::default())
    }

    fn sps_bits(ordering_info_present: bool, f: &SpsFields) -> Vec<u8> {
        let mut bits: Vec<u8> = Vec::new();
        push_bits(&mut bits, 0, 4); // sps_video_parameter_set_id
        push_bits(&mut bits, 0, 3); // sps_max_sub_layers_minus1
        push_bits(&mut bits, 0, 1); // sps_temporal_id_nesting_flag
        // profile_tier_level, general part: 2+1+5+32+4+43+1+8 = 96 bits
        push_bits(&mut bits, 0, 96);
        push_ue(&mut bits, 0); // sps_seq_parameter_set_id
        push_ue(&mut bits, 1); // chroma_format_idc = 4:2:0
        push_ue(&mut bits, f.width); // pic_width_in_luma_samples (literal ue(v), not minus-anything)
        push_ue(&mut bits, f.height); // pic_height_in_luma_samples
        push_bits(&mut bits, 0, 1); // conformance_window_flag = 0
        push_ue(&mut bits, f.bit_depth_minus8); // bit_depth_luma_minus8
        push_ue(&mut bits, 0); // bit_depth_chroma_minus8 = 0
        push_ue(&mut bits, 0); // log2_max_pic_order_cnt_lsb_minus4
        push_bits(&mut bits, u32::from(ordering_info_present), 1); // ordering info flag
        push_ue(&mut bits, 0); // sps_max_dec_pic_buffering_minus1 (loop runs once since flag=0)
        push_ue(&mut bits, 0); // sps_max_num_reorder_pics
        push_ue(&mut bits, 0); // sps_max_latency_increase_plus1
        push_ue(&mut bits, f.min_cb_minus3); // log2_min_luma_coding_block_size_minus3
        push_ue(&mut bits, f.diff_cb); // log2_diff_max_min_luma_coding_block_size
        push_ue(&mut bits, f.min_tb_minus2); // log2_min_luma_transform_block_size_minus2
        push_ue(&mut bits, f.diff_tb); // log2_diff_max_min_luma_transform_block_size
        push_ue(&mut bits, 0); // max_transform_hierarchy_depth_inter
        push_ue(&mut bits, f.depth_intra); // max_transform_hierarchy_depth_intra
        push_bits(&mut bits, 0, 1); // scaling_list_enabled_flag = 0
        push_bits(&mut bits, 0, 1); // amp_enabled_flag = 0
        push_bits(&mut bits, 0, 1); // sample_adaptive_offset_enabled_flag = 0
        push_bits(&mut bits, 0, 1); // pcm_enabled_flag = 0
        push_ue(&mut bits, 0); // num_short_term_ref_pic_sets = 0
        push_bits(&mut bits, 0, 1); // long_term_ref_pics_present_flag = 0
        push_bits(&mut bits, 0, 1); // sps_temporal_mvp_enabled_flag = 0
        push_bits(&mut bits, 1, 1); // strong_intra_smoothing_enabled_flag = 1 (exercised by the assertion below)
        // pad to byte boundary with zero bits (fine: parser stops reading before this matters)
        while !bits.len().is_multiple_of(8) {
            bits.push(0);
        }
        let mut out = Vec::with_capacity(bits.len() / 8);
        for chunk in bits.chunks(8) {
            let mut byte = 0u8;
            for (i, &b) in chunk.iter().enumerate() {
                byte |= b << (7 - i);
            }
            out.push(byte);
        }
        out
    }

    #[test]
    fn parses_minimal_sps_fields() {
        let rbsp = minimal_sps_bits(false);
        let sps = parse_sps(&rbsp).expect("minimal synthetic SPS should parse");
        assert_eq!(sps.chroma_format_idc, 1);
        assert_eq!(sps.pic_width_in_luma_samples, 15);
        assert_eq!(sps.pic_height_in_luma_samples, 15);
        assert_eq!(sps.bit_depth_luma, 8);
        assert_eq!(sps.bit_depth_chroma, 8);
        assert_eq!(sps.ctb_log2_size_y(), 4);
        assert!(!sps.pcm_enabled);
        assert!(!sps.scaling_list_enabled);
        assert!(sps.strong_intra_smoothing_enabled);
    }

    /// Header values outside the spec's ranges used to reach unchecked
    /// shifts and allocations; each must now be a plain error.
    #[test]
    fn rejects_out_of_range_sps_values() {
        let cases = [
            ("CTB larger than 64", SpsFields { diff_cb: 61, ..Default::default() }),
            ("CTB smaller than 16", SpsFields { diff_cb: 0, ..Default::default() }),
            ("bit depth above 16", SpsFields { bit_depth_minus8: 40, ..Default::default() }),
            ("transform block not smaller than coding block", SpsFields { min_tb_minus2: 1, ..Default::default() }),
            ("transform block above 32", SpsFields { min_cb_minus3: 1, diff_cb: 2, min_tb_minus2: 0, diff_tb: 4, ..Default::default() }),
            ("transform depth too deep", SpsFields { depth_intra: 3, ..Default::default() }),
            ("zero width", SpsFields { width: 0, ..Default::default() }),
        ];
        for (name, fields) in cases {
            assert!(matches!(parse_sps(&sps_bits(false, &fields)), Err(HeifError::MalformedHevc(_))), "{name}");
        }
        let huge = SpsFields { width: 1 << 15, height: 1 << 15, ..Default::default() };
        assert!(matches!(parse_sps(&sps_bits(false, &huge)), Err(HeifError::LimitExceeded(_))));
        let largest_ctb = SpsFields { min_cb_minus3: 0, diff_cb: 3, diff_tb: 3, depth_intra: 4, ..Default::default() };
        assert_eq!(parse_sps(&sps_bits(false, &largest_ctb)).unwrap().ctb_log2_size_y(), 6);
    }

    #[test]
    fn rejects_truncated_sps() {
        assert!(parse_sps(&[0u8; 2]).is_err());
    }

    #[test]
    fn single_layer_sps_reads_one_ordering_triple_for_either_flag() {
        for present in [false, true] {
            let sps = parse_sps(&minimal_sps_bits(present)).unwrap();
            assert_eq!(sps.ctb_log2_size_y(), 4);
            assert_eq!(sps.min_tb_log2_size_y(), 2);
            assert_eq!(sps.max_tb_log2_size_y(), 3);
            assert!(!sps.scaling_list_enabled);
            assert!(sps.strong_intra_smoothing_enabled);
        }
    }

    #[test]
    fn pps_retains_its_scaling_list_override() {
        let mut bits = Vec::new();
        push_ue(&mut bits, 0); // PPS id
        push_ue(&mut bits, 0); // SPS id
        push_bits(&mut bits, 0, 7); // dependent/output/extra/sign-hiding/CABAC flags
        push_ue(&mut bits, 0); // default L0 references
        push_ue(&mut bits, 0); // default L1 references
        push_bits(&mut bits, 1, 1); // init_qp_minus26: se(0)
        push_bits(&mut bits, 0, 3); // constrained/transform-skip/QP-delta flags
        push_bits(&mut bits, 3, 2); // Cb/Cr offsets: se(0), se(0)
        push_bits(&mut bits, 0, 8); // chroma offsets through deblocking-control flags
        push_bits(&mut bits, 1, 1); // PPS scaling list present
        push_bits(&mut bits, 1, 1); // first matrix uses explicit coefficients
        for _ in 0..16 { push_bits(&mut bits, 1, 1); } // se(0): each coefficient stays 8
        for _ in 1..20 {
            push_bits(&mut bits, 0, 1); // predicted matrix
            push_ue(&mut bits, 0); // default
        }
        push_bits(&mut bits, 0, 1); // lists_modification_present_flag
        push_ue(&mut bits, 0); // parallel merge level
        push_bits(&mut bits, 0, 1); // slice header extension
        let bytes: Vec<u8> = bits.chunks(8).map(|chunk| chunk.iter().enumerate()
            .fold(0, |byte, (i, bit)| byte | (bit << (7 - i)))).collect();
        let pps = parse_pps(&bytes).unwrap();
        let list = pps.scaling_list.unwrap();
        assert_eq!(list.weight(2, 0, 0, 0).unwrap(), 8);
        assert_eq!(list.weight(2, 0, 3, 3).unwrap(), 8);
        assert_eq!(list.weight(2, 1, 0, 0).unwrap(), 16);
    }
}
