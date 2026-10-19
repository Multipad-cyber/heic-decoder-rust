//! Byte-aligned reader for ISOBMFF boxes, and a bit-level reader (with
//! Exp-Golomb) for HEVC RBSP parsing. Every accessor returns `Result`;
//! nothing here indexes or casts without a bounds/range check first.

use crate::error::{HeifError, Result};

/// A byte-aligned cursor over a borrowed buffer, for ISOBMFF box parsing.
pub struct ByteReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn seek(&mut self, pos: usize) -> Result<()> {
        if pos > self.data.len() {
            return Err(HeifError::Truncated { at: pos, needed: pos - self.data.len() });
        }
        self.pos = pos;
        Ok(())
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or(HeifError::MalformedBox("length overflow"))?;
        let slice = self.data.get(self.pos..end).ok_or(HeifError::Truncated { at: self.pos, needed: end.saturating_sub(self.data.len()) })?;
        self.pos = end;
        Ok(slice)
    }

    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    pub fn u16(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_be_bytes([b[0], b[1]]))
    }

    pub fn u24(&mut self) -> Result<u32> {
        let b = self.take(3)?;
        Ok(u32::from_be_bytes([0, b[0], b[1], b[2]]))
    }

    pub fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        let mut arr = [0u8; 8];
        arr.copy_from_slice(b);
        Ok(u64::from_be_bytes(arr))
    }

    pub fn fourcc(&mut self) -> Result<[u8; 4]> {
        let b = self.take(4)?;
        Ok([b[0], b[1], b[2], b[3]])
    }

    pub fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        self.take(n)
    }

    pub fn skip(&mut self, n: usize) -> Result<()> {
        self.take(n)?;
        Ok(())
    }

    /// Null-terminated string, for `hdlr`'s component name.
    pub fn c_string(&mut self) -> Result<&'a [u8]> {
        let start = self.pos;
        loop {
            let b = self.u8()?;
            if b == 0 {
                return self.data.get(start..self.pos - 1).ok_or(HeifError::MalformedBox("c_string"));
            }
        }
    }
}

/// A bit-level reader over HEVC RBSP bytes (emulation-prevention bytes
/// already removed by the caller). Implements the Exp-Golomb codes used
/// throughout the HEVC syntax (clause 9.2).
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Bit position from the start of `data`.
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, bit_pos: 0 }
    }

    pub fn bit_pos(&self) -> usize {
        self.bit_pos
    }

    /// The unread bytes, when the reader is at a byte boundary.
    pub fn rest(&self) -> Option<&'a [u8]> {
        if self.bit_pos.is_multiple_of(8) { self.data.get(self.bit_pos / 8..) } else { None }
    }

    pub fn bits_remaining(&self) -> usize {
        self.data.len().saturating_mul(8).saturating_sub(self.bit_pos)
    }

    /// `u(1)`: a single bit.
    pub fn bit(&mut self) -> Result<u32> {
        let byte_idx = self.bit_pos / 8;
        let bit_idx = 7 - (self.bit_pos % 8);
        let byte = *self.data.get(byte_idx).ok_or(HeifError::Truncated { at: byte_idx, needed: 1 })?;
        self.bit_pos += 1;
        Ok(((byte >> bit_idx) & 1) as u32)
    }

    pub fn flag(&mut self) -> Result<bool> {
        Ok(self.bit()? == 1)
    }

    /// `u(n)`: n <= 32 bits, MSB first.
    pub fn bits(&mut self, n: u32) -> Result<u32> {
        if n == 0 {
            return Ok(0);
        }
        if n > 32 {
            return Err(HeifError::MalformedHevc("bits(n): n > 32"));
        }
        let mut v: u32 = 0;
        for _ in 0..n {
            v = (v << 1) | self.bit()?;
        }
        Ok(v)
    }

    /// `ue(v)`: unsigned Exp-Golomb (clause 9.2.1).
    pub fn ue(&mut self) -> Result<u32> {
        let mut leading_zero_bits: u32 = 0;
        while self.bit()? == 0 {
            leading_zero_bits += 1;
            // A conforming bitstream never has a run this long; bail out
            // rather than spin on adversarial input.
            if leading_zero_bits > 32 {
                return Err(HeifError::MalformedHevc("ue(v): exp-golomb prefix too long"));
            }
        }
        if leading_zero_bits == 0 {
            return Ok(0);
        }
        let suffix = self.bits(leading_zero_bits)?;
        // codeNum = 2^leadingZeroBits - 1 + suffix
        let base = 1u32.checked_shl(leading_zero_bits).ok_or(HeifError::MalformedHevc("ue(v): overflow"))?;
        base.checked_sub(1).and_then(|v| v.checked_add(suffix)).ok_or(HeifError::MalformedHevc("ue(v): overflow"))
    }

    /// `se(v)`: signed Exp-Golomb (clause 9.2.2).
    pub fn se(&mut self) -> Result<i32> {
        let code_num = self.ue()?;
        // (-1)^(k+1) * Ceil(k/2)
        let half = (code_num / 2) as i64 + (code_num % 2) as i64;
        let signed = if code_num % 2 == 1 { half } else { -half };
        i32::try_from(signed).map_err(|_| HeifError::MalformedHevc("se(v): overflow"))
    }

    /// Advances to the next byte boundary (`byte_alignment()` in the spec).
    pub fn align(&mut self) {
        self.bit_pos = self.bit_pos.div_ceil(8) * 8;
    }

    /// `more_rbsp_data()` (clause 7.2): true if anything other than the
    /// rbsp_stop_one_bit followed by zero bits remains.
    pub fn more_rbsp_data(&self) -> bool {
        let total_bits = self.data.len() * 8;
        if self.bit_pos >= total_bits {
            return false;
        }
        // Find the last set bit (the rbsp_stop_one_bit) scanning from the end.
        let mut last_one = None;
        for i in (self.bit_pos..total_bits).rev() {
            let byte_idx = i / 8;
            let bit_idx = 7 - (i % 8);
            let Some(&byte) = self.data.get(byte_idx) else { continue };
            if (byte >> bit_idx) & 1 == 1 {
                last_one = Some(i);
                break;
            }
        }
        match last_one {
            Some(i) => i > self.bit_pos,
            None => false,
        }
    }
}

/// Strips HEVC emulation-prevention bytes (`00 00 03` -> `00 00`) from a NAL
/// unit payload, producing the raw byte sequence payload (RBSP, clause 7.3.1.1).
pub fn strip_emulation_prevention(nal: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(nal.len());
    let mut zero_run = 0u32;
    for &b in nal {
        if zero_run >= 2 && b == 0x03 {
            zero_run = 0;
            continue;
        }
        out.push(b);
        if b == 0 {
            zero_run += 1;
        } else {
            zero_run = 0;
        }
    }
    out
}

/// The EBSP index immediately after `strip_emulation_prevention(ebsp)`
/// would have produced `target_rbsp_len` output bytes. Used to convert a
/// position found by parsing the RBSP (e.g. "the header ends at RBSP
/// byte N") back into the corresponding position in the original,
/// not-yet-stripped NAL bytes — needed because WPP/tile entry-point
/// offsets (clause 7.4.7.1) are defined in EBSP byte space, not RBSP.
pub fn rbsp_len_to_ebsp_pos(ebsp: &[u8], target_rbsp_len: usize) -> Result<usize> {
    let mut produced = 0usize;
    let mut zero_run = 0u32;
    for (i, &b) in ebsp.iter().enumerate() {
        if produced >= target_rbsp_len {
            return Ok(i);
        }
        if zero_run >= 2 && b == 0x03 {
            zero_run = 0;
            continue;
        }
        produced += 1;
        zero_run = if b == 0 { zero_run + 1 } else { 0 };
    }
    if produced == target_rbsp_len { Ok(ebsp.len()) } else { Err(HeifError::Truncated { at: ebsp.len(), needed: target_rbsp_len - produced }) }
}

/// The RBSP length that emulation-prevention removal of the first
/// `ebsp_len` bytes of `ebsp` produces. The inverse direction of
/// [`rbsp_len_to_ebsp_pos`]: converts an EBSP-space entry-point offset
/// into an RBSP byte position.
pub fn ebsp_len_to_rbsp_len(ebsp: &[u8], ebsp_len: usize) -> usize {
    ebsp_lens_to_rbsp_lens(ebsp, &[ebsp_len])[0]
}

/// [`ebsp_len_to_rbsp_len`] for several prefix lengths in one pass over
/// `ebsp`, rather than one pass, and one copy, per prefix: a slice's entry
/// points come one per CTU row. Whether a byte is an emulation prevention
/// byte depends only on the bytes before it, so a prefix's RBSP length is
/// the count of kept bytes within it.
pub fn ebsp_lens_to_rbsp_lens(ebsp: &[u8], ebsp_lens: &[usize]) -> Vec<usize> {
    let mut lens = Vec::with_capacity(ebsp_lens.len());
    let (mut position, mut kept, mut zero_run) = (0usize, 0usize, 0u32);
    for &ebsp_len in ebsp_lens {
        let end = ebsp_len.min(ebsp.len());
        if end < position {
            // Lengths that go back are counted again from the start.
            (position, kept, zero_run) = (0, 0, 0);
        }
        for &b in &ebsp[position..end] {
            if zero_run >= 2 && b == 0x03 {
                zero_run = 0;
                continue;
            }
            kept += 1;
            zero_run = if b == 0 { zero_run + 1 } else { 0 };
        }
        position = end;
        lens.push(kept);
    }
    lens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ue_golomb_known_values() {
        // Spec Table 9-1: bit string -> codeNum.
        let cases: &[(&[u8], u32)] = &[(&[0b1000_0000], 0), (&[0b0100_0000], 1), (&[0b0110_0000], 2), (&[0b0010_0000], 3), (&[0b0010_1000], 4)];
        for (bits, expected) in cases {
            let mut r = BitReader::new(bits);
            assert_eq!(r.ue().unwrap(), *expected);
        }
    }

    #[test]
    fn se_golomb_known_values() {
        let cases: &[(&[u8], i32)] = &[(&[0b1000_0000], 0), (&[0b0100_0000], 1), (&[0b0110_0000], -1), (&[0b0010_0000], 2), (&[0b0010_1000], -2)];
        for (bits, expected) in cases {
            let mut r = BitReader::new(bits);
            assert_eq!(r.se().unwrap(), *expected);
        }
    }

    #[test]
    fn truncated_ue_errs_not_panics() {
        let mut r = BitReader::new(&[0b0000_0000]);
        assert!(r.ue().is_err());
    }

    #[test]
    fn one_pass_rbsp_lengths_match_stripping_each_prefix() {
        // Bytes drawn mostly from 0x00 and 0x03, so emulation prevention
        // sequences are dense, from a fixed linear congruential generator.
        let mut state = 0x2545_f491u32;
        let ebsp: Vec<u8> = (0..400)
            .map(|_| {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                [0x00, 0x00, 0x03, 0x01, 0x00, 0x03, 0xff][(state >> 16) as usize % 7]
            })
            .collect();
        let expected = |len: usize| strip_emulation_prevention(&ebsp[..len.min(ebsp.len())]).len();
        let rising: Vec<usize> = (0..=ebsp.len() + 2).collect();
        assert_eq!(ebsp_lens_to_rbsp_lens(&ebsp, &rising), rising.iter().map(|&len| expected(len)).collect::<Vec<_>>());
        let mixed = [7, 7, 300, 12, 0, 399, 400, 5000, 3];
        assert_eq!(ebsp_lens_to_rbsp_lens(&ebsp, &mixed), mixed.iter().map(|&len| expected(len)).collect::<Vec<_>>());
    }

    #[test]
    fn emulation_prevention_strip() {
        let input = [0x00, 0x00, 0x03, 0x01, 0x00, 0x00, 0x03, 0x02, 0x00, 0x00, 0x01];
        let out = strip_emulation_prevention(&input);
        assert_eq!(out, vec![0x00, 0x00, 0x01, 0x00, 0x00, 0x02, 0x00, 0x00, 0x01]);
    }

    #[test]
    fn ebsp_rbsp_offset_conversion_round_trips() {
        // Same fixture as `emulation_prevention_strip`: EBSP [00,00,03,01,...]
        // strips to RBSP [00,00,01,...] — the emulation-prevention 0x03 at
        // EBSP index 2 is dropped, so RBSP len 3 is reached at EBSP index 4,
        // and stripping EBSP[..4] produces exactly 3 RBSP bytes.
        let input = [0x00, 0x00, 0x03, 0x01, 0x00, 0x00, 0x03, 0x02, 0x00, 0x00, 0x01];
        assert_eq!(rbsp_len_to_ebsp_pos(&input, 3).unwrap(), 4);
        assert_eq!(ebsp_len_to_rbsp_len(&input, 4), 3);
        assert_eq!(rbsp_len_to_ebsp_pos(&input, 0).unwrap(), 0);
    }

    #[test]
    fn rbsp_len_to_ebsp_pos_errs_past_end_not_panics() {
        let input = [0x01, 0x02, 0x03];
        assert!(rbsp_len_to_ebsp_pos(&input, 100).is_err());
    }

    #[test]
    fn byte_reader_bounds_checked() {
        let mut r = ByteReader::new(&[1, 2, 3]);
        assert_eq!(r.u8().unwrap(), 1);
        assert!(r.u32().is_err());
    }
}
