//! Slice headers for I pictures and single-reference P pictures.
//! Dependent segments, B pictures and weighted prediction are unsupported.
//! SAO and deblocking parameters are applied after reconstruction.

use super::nal::is_irap_nal;
use super::params::{Pps, Sps};
use crate::bitreader::BitReader;
use crate::error::{HeifError, Result};

#[derive(Debug, Clone)]
pub struct InterSlice {
    pub poc_lsb: u32,
    pub reference_deltas: Vec<i32>,
    pub max_merge_candidates: u32,
    pub cabac_init_b: bool,
    pub temporal_mvp: bool,
}

#[derive(Debug, Clone)]
pub struct SliceHeader {
    pub slice_segment_address: usize,
    pub inter: Option<InterSlice>,
    pub slice_qp_y: i32,
    pub cb_qp_offset: i32,
    pub cr_qp_offset: i32,
    pub sao_luma: bool,
    pub sao_chroma: bool,
    pub deblocking_filter_disabled: bool,
    pub beta_offset_div2: i32,
    pub tc_offset_div2: i32,
    /// Cumulative-sum-able substream lengths in EBSP bytes (one per
    /// substream except the last, which runs to the end of the slice
    /// segment data), empty when there's only one substream.
    pub entry_point_offsets: Vec<u32>,
}

pub fn parse_slice_header(rbsp: &[u8], nal_unit_type: u8, sps: &Sps, pps: &Pps) -> Result<(SliceHeader, usize)> {
    let mut r = BitReader::new(rbsp);
    let first_slice_segment_in_pic = r.flag()?;
    if is_irap_nal(nal_unit_type) {
        r.bit()?; // no_output_of_prior_pics_flag
    }
    r.ue()?; // slice_pic_parameter_set_id

    if pps.dependent_slice_segments_enabled {
        return Err(HeifError::Unsupported("slice header: dependent slice segments not supported"));
    }
    let ctb = 1usize << sps.ctb_log2_size_y();
    let ctb_count = (sps.pic_width_in_luma_samples as usize).div_ceil(ctb)
        * (sps.pic_height_in_luma_samples as usize).div_ceil(ctb);
    let slice_segment_address = if first_slice_segment_in_pic { 0 } else {
        let address = r.bits(usize::BITS - (ctb_count - 1).leading_zeros())? as usize;
        if address == 0 || address >= ctb_count { return Err(HeifError::MalformedHevc("slice: segment address out of range")); }
        address
    };
    for _ in 0..pps.num_extra_slice_header_bits {
        r.bit()?;
    }
    let slice_type = r.ue()?;
    if slice_type != 2 && slice_type != 1 {
        return Err(HeifError::Unsupported("slice header: only I/P slices are supported"));
    }
    if pps.output_flag_present {
        r.bit()?; // pic_output_flag
    }
    let mut poc_lsb = 0;
    let mut reference_deltas = Vec::new();
    let mut temporal_mvp = false;
    if nal_unit_type != 19 && nal_unit_type != 20 {
        poc_lsb = r.bits(sps.log2_max_pic_order_cnt_lsb)?;
        let use_sps_set = r.flag()?;
        let set = if use_sps_set {
            let count = sps.short_term_refs.len();
            if count == 0 { return Err(HeifError::MalformedHevc("slice: no SPS reference set")); }
            let index = if count > 1 { r.bits(usize::BITS - (count - 1).leading_zeros())? as usize } else { 0 };
            sps.short_term_refs.get(index).cloned().ok_or(HeifError::MalformedHevc("slice: RPS index out of range"))?
        } else {
            super::params::parse_short_term_refs(&mut r, sps.short_term_refs.len(), &sps.short_term_refs, true)?
        };
        reference_deltas = set.0.iter().filter_map(|&(delta, used)| used.then_some(delta)).collect();
        if sps.temporal_mvp_enabled { temporal_mvp = r.flag()?; }
    }

    let mut sao_luma = false;
    let mut sao_chroma = false;
    if sps.sample_adaptive_offset_enabled {
        sao_luma = r.flag()?;
        if sps.chroma_format_idc != 0 { sao_chroma = r.flag()?; }
    }

    let inter = if slice_type == 1 {
        let references = if r.flag()? { r.ue()?.checked_add(1).ok_or(HeifError::MalformedHevc("slice: reference count overflow"))? } else { pps.default_l0_references };
        if references != 1 || reference_deltas.len() != 1 {
            return Err(HeifError::Unsupported("P slice: only one short-term reference supported"));
        }
        // List modification syntax is absent with NumPicTotalCurr == 1.
        let cabac_init_b = pps.cabac_init_present && r.flag()?;
        // With one L0 reference, collocated_ref_idx is inferred as zero.
        if pps.weighted_pred { return Err(HeifError::Unsupported("P slice: weighted prediction")); }
        let merge_delta = r.ue()?;
        let max_merge_candidates = 5u32.checked_sub(merge_delta).filter(|&n| n != 0).ok_or(HeifError::MalformedHevc("slice: merge candidate count"))?;
        Some(InterSlice { poc_lsb, reference_deltas, max_merge_candidates, cabac_init_b, temporal_mvp })
    } else { None };

    // SliceQpY, clause 7.4.7.1: PPS's init_qp (already includes the +26
    // offset per `parse_pps`) plus this slice's delta.
    let slice_qp_delta = r.se()?;

    let mut cb_qp_offset = 0;
    let mut cr_qp_offset = 0;
    if pps.slice_chroma_qp_offsets_present {
        cb_qp_offset = r.se()?;
        cr_qp_offset = r.se()?;
    }

    // Deblocking override (clause 7.3.6.1): falls back to the PPS's own
    // disabled flag/offsets unless this slice explicitly overrides them.
    let mut slice_deblocking_filter_disabled = pps.deblocking_filter_disabled;
    let mut beta_offset_div2 = pps.beta_offset_div2;
    let mut tc_offset_div2 = pps.tc_offset_div2;
    if pps.deblocking_filter_override_enabled {
        let deblocking_filter_override = r.flag()?;
        if deblocking_filter_override {
            slice_deblocking_filter_disabled = r.flag()?;
            if !slice_deblocking_filter_disabled {
                beta_offset_div2 = r.se()?;
                tc_offset_div2 = r.se()?;
            }
        }
    }
    if pps.loop_filter_across_slices_enabled && (sao_luma || sao_chroma || !slice_deblocking_filter_disabled) {
        r.bit()?; // slice_loop_filter_across_slices_enabled_flag
    }

    // Entry point offsets (clause 7.3.6.1): present for tiled *or*
    // wavefront-parallel-processing (WPP) pictures. Each value is a
    // substream length in EBSP bytes (i.e. counting emulation-prevention
    // bytes) — `hevc::ctu::decode_slice`'s caller converts these to RBSP
    // positions before seeking, since this function only ever sees the
    // already-emulation-stripped RBSP.
    let mut entry_point_offsets: Vec<u32> = Vec::new();
    if pps.tiles_enabled || pps.entropy_coding_sync_enabled {
        let num_entry_point_offsets = r.ue()?;
        if num_entry_point_offsets > 0 {
            let offset_len = r.ue()? + 1;
            if offset_len > 32 {
                return Err(HeifError::MalformedHevc("slice header: entry point offset length too large"));
            }
            for _ in 0..num_entry_point_offsets {
                entry_point_offsets.push(r.bits(offset_len)? + 1);
            }
        }
    }

    if pps.slice_segment_header_extension_present {
        let ext_len = r.ue()?;
        for _ in 0..ext_len {
            r.bits(8)?;
        }
    }

    // byte_alignment() always consumes alignment_bit_equal_to_one,
    // including when the syntax fields already end on a byte boundary.
    // In that case a whole 0x80 byte precedes CABAC, not zero padding bits.
    if r.bit()? != 1 {
        return Err(HeifError::MalformedHevc("slice header: missing alignment one bit"));
    }
    while !r.bit_pos().is_multiple_of(8) {
        if r.bit()? != 0 {
            return Err(HeifError::MalformedHevc("slice header: nonzero alignment padding"));
        }
    }
    let byte_pos = r.bit_pos() / 8;

    Ok((
        SliceHeader {
            slice_segment_address,
            inter,
            slice_qp_y: pps.init_qp + slice_qp_delta,
            cb_qp_offset,
            cr_qp_offset,
            sao_luma,
            sao_chroma,
            deblocking_filter_disabled: slice_deblocking_filter_disabled,
            beta_offset_div2,
            tc_offset_div2,
            entry_point_offsets,
        },
        byte_pos,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hevc::params::{Pps, Sps};

    fn test_sps() -> Sps {
        Sps {
            log2_max_pic_order_cnt_lsb: 8, short_term_refs: Vec::new(), temporal_mvp_enabled: false,
            chroma_format_idc: 1,
            pic_width_in_luma_samples: 16,
            pic_height_in_luma_samples: 16,
            conformance_window: [0; 4],
            color_info: None, chroma_location: None,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
            log2_min_luma_coding_block_size: 3,
            log2_diff_max_min_luma_coding_block_size: 1,
            log2_min_luma_transform_block_size: 2,
            log2_diff_max_min_luma_transform_block_size: 1,
            max_transform_hierarchy_depth_inter: 0,
            max_transform_hierarchy_depth_intra: 0,
            scaling_list_enabled: false,
            scaling_list: None,
            amp_enabled: false,
            sample_adaptive_offset_enabled: false,
            pcm_enabled: false,
            strong_intra_smoothing_enabled: false,
        }
    }

    fn test_pps() -> Pps {
        Pps {
            sao_offset_scale: [0; 2],
            default_l0_references: 1, weighted_pred: false, lists_modification_present: false, log2_parallel_merge_level: 2,
            dependent_slice_segments_enabled: false,
            output_flag_present: false,
            num_extra_slice_header_bits: 0,
            sign_data_hiding_enabled: false,
            cabac_init_present: false,
            init_qp: 26,
            constrained_intra_pred: false,
            transform_skip_enabled: false,
            scaling_list: None,
            cu_qp_delta_enabled: false,
            diff_cu_qp_delta_depth: 0,
            cb_qp_offset: 0,
            cr_qp_offset: 0,
            slice_chroma_qp_offsets_present: false,
            transquant_bypass_enabled: false,
            tiles_enabled: false,
            entropy_coding_sync_enabled: false,
            num_tile_columns: 1,
            num_tile_rows: 1,
            loop_filter_across_slices_enabled: false,
            deblocking_filter_override_enabled: false,
            deblocking_filter_disabled: false,
            beta_offset_div2: 0,
            tc_offset_div2: 0,
            slice_segment_header_extension_present: false,
        }
    }

    fn push_bits(bits: &mut Vec<u8>, v: u32, n: u32) {
        for i in (0..n).rev() {
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

    fn push_se(bits: &mut Vec<u8>, v: i32) {
        let code_num = if v > 0 { (v as u32) * 2 - 1 } else { (-v as u32) * 2 };
        push_ue(bits, code_num);
    }

    fn pack(bits: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
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
    fn parses_minimal_i_slice_header() {
        let mut bits = Vec::new();
        push_bits(&mut bits, 1, 1); // first_slice_segment_in_pic_flag
        push_bits(&mut bits, 0, 1); // no_output_of_prior_pics_flag (IRAP)
        push_ue(&mut bits, 0); // slice_pic_parameter_set_id
        push_ue(&mut bits, 2); // slice_type = I
        push_se(&mut bits, 3); // slice_qp_delta
        // test_pps() has loop_filter_across_slices_enabled = false, so
        // slice_loop_filter_across_slices_enabled_flag's gate is false
        // and no bit is read for it here.
        bits.push(1); // alignment_bit_equal_to_one
        while !bits.len().is_multiple_of(8) {
            bits.push(0);
        }
        let rbsp = pack(&bits);
        let sps = test_sps();
        let pps = test_pps();
        let (hdr, _byte_pos) = parse_slice_header(&rbsp, 19 /* IDR_W_RADL */, &sps, &pps).expect("minimal synthetic slice header should parse");
        assert_eq!(hdr.slice_qp_y, 29); // init_qp(26) + slice_qp_delta(3)
    }

    #[test]
    fn aligned_slice_header_still_consumes_alignment_byte() {
        let mut pps = test_pps();
        pps.output_flag_present = true;
        let mut bits = Vec::new();
        push_bits(&mut bits, 1, 1); // first slice
        push_bits(&mut bits, 0, 1); // no_output_of_prior_pics
        push_ue(&mut bits, 0); // PPS id
        push_ue(&mut bits, 2); // I slice
        push_bits(&mut bits, 1, 1); // pic_output_flag
        push_se(&mut bits, 0); // QP delta
        assert_eq!(bits.len(), 8);
        let mut rbsp = pack(&bits);
        rbsp.extend_from_slice(&[0x80, 0x2a, 0xb3]); // alignment byte, then CABAC
        let (_, data_offset) = parse_slice_header(&rbsp, 19, &test_sps(), &pps).unwrap();
        assert_eq!(data_offset, 2);
        assert_eq!(&rbsp[data_offset..], &[0x2a, 0xb3]);

        assert!(parse_slice_header(&rbsp[..1], 19, &test_sps(), &pps).is_err());
        rbsp[1] = 0;
        assert!(parse_slice_header(&rbsp, 19, &test_sps(), &pps).is_err());
        rbsp[1] = 0x81;
        assert!(parse_slice_header(&rbsp, 19, &test_sps(), &pps).is_err());
    }

    #[test]
    fn rejects_non_i_slice() {
        let mut bits = Vec::new();
        push_bits(&mut bits, 1, 1);
        push_bits(&mut bits, 0, 1);
        push_ue(&mut bits, 0);
        push_ue(&mut bits, 0); // slice_type = B
        while !bits.len().is_multiple_of(8) {
            bits.push(0);
        }
        let rbsp = pack(&bits);
        let err = parse_slice_header(&rbsp, 19, &test_sps(), &test_pps()).unwrap_err();
        assert!(matches!(err, HeifError::Unsupported(_)));
    }
}
