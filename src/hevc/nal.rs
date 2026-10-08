//! HEVC NAL unit header (clause 7.3.1.2) and the small set of NAL unit
//! types this decoder cares about (clause 7.4.2.2, Table 7-1).

use crate::bitreader::ByteReader;
use crate::error::{HeifError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NalHeader {
    pub nal_unit_type: u8,
    pub nuh_layer_id: u8,
    pub nuh_temporal_id_plus1: u8,
}

pub const NAL_VPS: u8 = 32;
pub const NAL_SPS: u8 = 33;
pub const NAL_PPS: u8 = 34;

/// IRAP (BLA/IDR/CRA) and ordinary slice NAL unit types used by still
/// images (Table 7-1, VCL NAL units 0-31).
pub fn is_slice_nal(nal_unit_type: u8) -> bool {
    nal_unit_type <= 31
}

pub fn is_irap_nal(nal_unit_type: u8) -> bool {
    (16..=23).contains(&nal_unit_type)
}

/// Parses the 2-byte NAL header and returns it along with the remaining
/// payload bytes (still with emulation-prevention bytes present; the
/// caller strips those before bit-level parsing of the RBSP).
pub fn parse_nal_header(nal: &[u8]) -> Result<(NalHeader, &[u8])> {
    let mut r = ByteReader::new(nal);
    let b0 = r.u8()?;
    let b1 = r.u8()?;
    if b0 & 0x80 != 0 {
        return Err(HeifError::MalformedHevc("NAL header: forbidden_zero_bit set"));
    }
    let nal_unit_type = (b0 >> 1) & 0x3F;
    let nuh_layer_id = ((b0 & 0x01) << 5) | (b1 >> 3);
    let nuh_temporal_id_plus1 = b1 & 0x07;
    let payload = r.bytes(r.remaining())?;
    Ok((NalHeader { nal_unit_type, nuh_layer_id, nuh_temporal_id_plus1 }, payload))
}
