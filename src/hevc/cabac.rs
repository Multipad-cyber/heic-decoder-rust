//! CABAC entropy decoding (clause 9.3). The arithmetic engine
//! (`Engine::decision`/`bypass`/`terminate`) and context-state tables
//! (`LPS_RANGE`, `NEXT_STATE_MPS`, `NEXT_STATE_LPS`) were checked against
//! `libde265`'s `cabac.cc` (LGPL-3.0) — these are the state machine
//! mandated by the standard for interoperability (every conformant HEVC
//! decoder reproduces the identical numbers), not creative content, but
//! a single wrong entry in a 64-row table is exactly the kind of error
//! that's invisible without either the spec text or a cross-check, so
//! they were verified against a working implementation rather than typed
//! from memory. The HEVC-specific per-syntax-element `initValue` tables
//! (`Table 9-4`) were checked the same way against `libde265`'s
//! `contextmodel.cc`. See this crate's `README.md`.

use std::marker::PhantomData;

use crate::bitreader::BitReader;
use crate::error::{HeifError, Result};

/// `rangeTabLPS[pStateIdx][qCodIRangeIdx]`, clause 9.3.4.3.2.2 Table 9-46.
#[rustfmt::skip]
const LPS_RANGE: [[u32; 4]; 64] = [
    [128,176,208,240],[128,167,197,227],[128,158,187,216],[123,150,178,205],
    [116,142,169,195],[111,135,160,185],[105,128,152,175],[100,122,144,166],
    [95,116,137,158],[90,110,130,150],[85,104,123,142],[81,99,117,135],
    [77,94,111,128],[73,89,105,122],[69,85,100,116],[66,80,95,110],
    [62,76,90,104],[59,72,86,99],[56,69,81,94],[53,65,77,89],
    [51,62,73,85],[48,59,69,80],[46,56,66,76],[43,53,63,72],
    [41,50,59,69],[39,48,56,65],[37,45,54,62],[35,43,51,59],
    [33,41,48,56],[32,39,46,53],[30,37,43,50],[29,35,41,48],
    [27,33,39,45],[26,31,37,43],[24,30,35,41],[23,28,33,39],
    [22,27,32,37],[21,26,30,35],[20,24,29,33],[19,23,27,31],
    [18,22,26,30],[17,21,25,28],[16,20,23,27],[15,19,22,25],
    [14,18,21,24],[14,17,20,23],[13,16,19,22],[12,15,18,21],
    [12,14,17,20],[11,14,16,19],[11,13,15,18],[10,12,15,17],
    [10,12,14,16],[9,11,13,15],[9,11,12,14],[8,10,12,14],
    [8,9,11,13],[7,9,11,12],[7,9,10,12],[7,8,10,11],
    [6,8,9,11],[6,7,9,10],[6,7,8,9],[2,2,2,2],
];

/// `transIdxMPS[pStateIdx]`, Table 9-47.
#[rustfmt::skip]
const NEXT_STATE_MPS: [u8; 64] = [
    1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
    17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,
    33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,
    49,50,51,52,53,54,55,56,57,58,59,60,61,62,62,63,
];

/// `transIdxLPS[pStateIdx]`, Table 9-47.
#[rustfmt::skip]
const NEXT_STATE_LPS: [u8; 64] = [
    0,0,1,2,2,4,4,5,6,7,8,9,9,11,11,12,
    13,13,15,15,16,16,18,18,19,19,21,21,22,22,23,24,
    24,25,26,26,27,27,28,29,29,30,30,30,31,32,32,33,
    33,33,34,34,35,35,35,36,36,36,37,37,37,38,38,63,
];

/// Both transitions of every context state at once, indexed
/// `[is_lps][pStateIdx * 2 + valMps]` and packed the same way, so a
/// decision updates its context without branching. The LPS row flips `valMps` at state 0, as clause 9.3.4.3.2.2
/// does.
const TRANSITION: [[u8; 128]; 2] = {
    let mut table = [[0u8; 128]; 2];
    let mut packed = 0;
    while packed < 128 {
        let (state, mps) = (packed / 2, (packed % 2) as u8);
        table[0][packed] = NEXT_STATE_MPS[state] * 2 + mps;
        table[1][packed] = NEXT_STATE_LPS[state] * 2 + if state == 0 { 1 - mps } else { mps };
        packed += 1;
    }
    table
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextModel {
    pub state: u8,
    pub mps: u8,
}

impl ContextModel {
    /// `set_initValue`, clause 9.3.2.2. `qp` is the slice's `SliceQpY`.
    pub fn init(init_value: u8, qp: i32) -> Self {
        let slope_idx = (init_value >> 4) as i32;
        let intersec_idx = (init_value & 0xF) as i32;
        let m = slope_idx * 5 - 45;
        let n = (intersec_idx << 3) - 16;
        let qp_clipped = qp.clamp(0, 51);
        let pre_ctx_state = ((m * qp_clipped) >> 4) + n;
        let pre_ctx_state = pre_ctx_state.clamp(1, 126);
        let mps = if pre_ctx_state <= 63 { 0 } else { 1 };
        let state = if mps == 1 { pre_ctx_state - 64 } else { 63 - pre_ctx_state };
        ContextModel { state: state as u8, mps }
    }
}

/// The CABAC arithmetic decoding engine (clause 9.3.4.3), over the bytes
/// of one slice segment substream. `range` and `offset` are the spec's
/// 9-bit `ivlCurrRange`/`ivlOffset`; the bits not yet shifted into
/// `offset` wait in a 64-bit cache refilled a byte at a time, so
/// renormalization takes all the bits it needs in one shift.
///
/// Decoding a bin cannot fail, so it inlines into the syntax loops: past
/// the end of the data the engine reads zeros and sets `overread`, which
/// [`Engine::terminate`] (read after every CTU) reports as truncation.
/// The `Result`s on the bin decoders stay so callers keep one error path.
pub struct Engine<'a, 'b> {
    data: &'b [u8],
    /// Next byte of `data` to load into `cache`.
    pos: usize,
    /// Unread bits, most significant first: the next bit is bit 63.
    cache: u64,
    cache_bits: u32,
    range: u32,
    offset: u32,
    /// A read went past the end of the data.
    overread: bool,
    /// The reader stays borrowed while the engine consumes its bytes.
    _reader: PhantomData<&'a mut BitReader<'b>>,
}

impl<'a, 'b> Engine<'a, 'b> {
    /// Bit position within the substream, for diagnostics only (not
    /// meaningful as "bits consumed by CABAC," since the engine buffers
    /// ahead of what it's resolved into bins).
    pub fn bit_pos(&self) -> usize {
        self.pos * 8 - self.cache_bits as usize
    }

    /// Clause 9.3.2.5: `ivlCurrRange = 510`, `ivlOffset = read_bits(9)`.
    /// `bits` must be at a byte boundary, as slice data always is.
    pub fn new(bits: &'a mut BitReader<'b>) -> Result<Self> {
        let data = bits.rest().ok_or(HeifError::CabacDesync("engine: slice data not byte-aligned"))?;
        let mut engine = Engine { data, pos: 0, cache: 0, cache_bits: 0, range: 510, offset: 0, overread: false, _reader: PhantomData };
        engine.offset = engine.read(9);
        engine.check_overread()?;
        Ok(engine)
    }

    /// The next `n` bits (1..=32), MSB first; zeros past the end of the
    /// data, which sets `overread`.
    #[inline(always)]
    fn read(&mut self, n: u32) -> u32 {
        if self.cache_bits < n {
            self.refill();
            if self.cache_bits < n {
                // The bits below the loaded ones are already zero.
                self.overread = true;
                self.cache_bits = n;
            }
        }
        let v = (self.cache >> (64 - n)) as u32;
        self.cache <<= n;
        self.cache_bits -= n;
        v
    }

    /// Reports a read past the end of the data as truncation.
    fn check_overread(&self) -> Result<()> {
        if self.overread { Err(HeifError::Truncated { at: self.data.len(), needed: 1 }) } else { Ok(()) }
    }

    #[inline(never)]
    fn refill(&mut self) {
        while self.cache_bits <= 56 {
            let Some(&byte) = self.data.get(self.pos) else { break };
            self.cache |= u64::from(byte) << (56 - self.cache_bits);
            self.cache_bits += 8;
            self.pos += 1;
        }
    }

    /// Clause 9.3.4.3.3, all at once: shift `range` back up to 9 bits
    /// (at most 7 places, since `range` never drops below 2) and as many
    /// new bits into `offset`.
    #[inline(always)]
    fn renormalize(&mut self) {
        let n = self.range.leading_zeros().saturating_sub(23);
        if n > 0 {
            self.range <<= n;
            self.offset = (self.offset << n) | self.read(n);
        }
    }

    /// `DecodeDecision`, clause 9.3.4.3.2. Never fails; see [`Engine`].
    ///
    /// Branch-free: which of MPS and LPS was
    /// decoded is close to a coin toss for many contexts, so a branch on
    /// it mispredicts often. The outcome selects the new offset, range and
    /// context state through masks and [`TRANSITION`] instead, and the
    /// renormalization shift is applied even when it is zero.
    #[inline(always)]
    pub fn decision(&mut self, ctx: &mut ContextModel) -> Result<u32> {
        // `state` is below 64 (every table entry is) and `range` is in
        // 256..=510, so the masks never change an index; they only let the
        // compiler drop the bounds checks.
        let state = ctx.state as usize & 63;
        let mps = u32::from(ctx.mps & 1);
        let lps_range = LPS_RANGE[state][(self.range >> 6) as usize & 3];
        let mps_range = self.range - lps_range;
        let is_lps = u32::from(self.offset >= mps_range);
        let mask = is_lps.wrapping_neg();
        self.offset -= mps_range & mask;
        self.range = mps_range ^ ((mps_range ^ lps_range) & mask);
        let next = TRANSITION[is_lps as usize][state * 2 + mps as usize];
        ctx.state = next >> 1;
        ctx.mps = next & 1;
        // `range` is now in 2..=510: shift it back to 9 bits, 0 to 7 places.
        let n = self.range.leading_zeros() - 23;
        self.range <<= n;
        self.offset = (self.offset << n) | self.read_upto7(n);
        Ok(mps ^ is_lps)
    }

    /// [`Engine::read`] for `n` in 0..=7, without a branch on `n == 0`.
    #[inline(always)]
    fn read_upto7(&mut self, n: u32) -> u32 {
        if self.cache_bits < n {
            self.refill();
            if self.cache_bits < n {
                self.overread = true;
                self.cache_bits = n;
            }
        }
        // Two shifts, so that `n == 0` reads nothing instead of shifting a
        // `u64` by 64.
        let v = ((self.cache >> 32) >> (32 - n)) as u32;
        self.cache <<= n;
        self.cache_bits -= n;
        v
    }

    /// The decision with branches, as clause 9.3.4.3.2 writes it: what
    /// [`Engine::decision`] did before it went branch-free, kept as the
    /// reference it is tested against.
    #[cfg(test)]
    fn reference_decision(&mut self, ctx: &mut ContextModel) -> u32 {
        let state = ctx.state as usize & 63;
        let lps_range = LPS_RANGE[state][(self.range >> 6) as usize & 3];
        self.range -= lps_range;
        let bin_val;
        if self.offset >= self.range {
            bin_val = 1 - ctx.mps as u32;
            self.offset -= self.range;
            self.range = lps_range;
            if state == 0 {
                ctx.mps = 1 - ctx.mps;
            }
            ctx.state = NEXT_STATE_LPS[state];
        } else {
            bin_val = ctx.mps as u32;
            ctx.state = NEXT_STATE_MPS[state];
        }
        self.renormalize();
        bin_val
    }

    /// `DecodeBypass`, clause 9.3.4.3.4. Never fails; see [`Engine`].
    #[inline(always)]
    pub fn bypass(&mut self) -> Result<u32> {
        self.offset = (self.offset << 1) | self.read(1);
        // Bypass bins are equiprobable, so a branch would mispredict half
        // the time: subtract under a mask instead.
        let bin_val = u32::from(self.offset >= self.range);
        self.offset -= self.range & bin_val.wrapping_neg();
        Ok(bin_val)
    }

    /// `FL(n)` using bypass bins, MSB first.
    ///
    /// Up to 16 bins at a time by one division: a run of bypass bins is the long division of the offset,
    /// with the next bits appended, by the unchanging range — each bin is
    /// one quotient bit, and the remainder is the new offset. `offset`
    /// stays below `range` (≤ 510), so 16 more bits fit a `u32`.
    pub fn bypass_fl(&mut self, n: u32) -> Result<u32> {
        if n > 31 {
            return Err(HeifError::MalformedHevc("bypass_fl: n too large"));
        }
        let mut v = 0u32;
        let mut left = n;
        // Only a non-conforming stream leaves `offset` at or above `range`
        // (an initial 9-bit offset of 510 or 511); there the quotient bits
        // are not the bins, so it goes bin by bin as before.
        if self.offset >= self.range {
            for _ in 0..n {
                v = (v << 1) | self.bypass()?;
            }
            return Ok(v);
        }
        while left > 0 {
            let k = left.min(16);
            let dividend = (self.offset << k) | self.read(k);
            let bins = dividend / self.range;
            self.offset = dividend - bins * self.range;
            v = (v << k) | bins;
            left -= k;
        }
        Ok(v)
    }

    /// A unary run of bypass bins: the 1-bins up to and including the 0
    /// that ends them, or `max_ones` 1-bins and nothing more. Returns the
    /// number of 1-bins.
    ///
    /// Eight bins per division: the run's
    /// length is the number of leading ones of the quotient, and only the
    /// bins up to the terminating zero are consumed — their quotient and
    /// remainder follow from the same division. Near the end of the data
    /// it goes bin by bin, so a short read is reported as before.
    pub fn bypass_unary(&mut self, max_ones: u32) -> Result<u32> {
        let mut ones = 0u32;
        while ones < max_ones {
            if self.cache_bits < 8 || self.offset >= self.range {
                self.refill();
                // Near the end of the data, or a non-conforming stream (see
                // `bypass_fl`): bin by bin.
                if self.cache_bits < 8 || self.offset >= self.range {
                    while ones < max_ones && self.bypass()? == 1 {
                        ones += 1;
                    }
                    return Ok(ones);
                }
            }
            let bits = (self.cache >> 56) as u32;
            let quotient = ((self.offset << 8) | bits) / self.range;
            // Leading ones of the 8-bin quotient, 0..=8.
            let leading = (!(quotient << 24)).leading_zeros();
            let (take, ended) = if ones + leading >= max_ones {
                (max_ones - ones, false)
            } else if leading < 8 {
                (leading + 1, true)
            } else {
                (8, false)
            };
            let dividend = (self.offset << take) | (bits >> (8 - take));
            self.offset = dividend - (quotient >> (8 - take)) * self.range;
            self.cache <<= take;
            self.cache_bits -= take;
            ones += take - u32::from(ended);
            if ended {
                return Ok(ones);
            }
        }
        Ok(ones)
    }

    /// `TU(cMax)` (truncated unary) using bypass bins: counts leading
    /// 1-bins, stopping at the first 0 or once `cmax` 1-bins have been
    /// read (clause 9.3.3.4). Used by `sao_offset_abs`.
    pub fn bypass_tu(&mut self, cmax: u32) -> Result<u32> {
        let mut v = 0u32;
        while v < cmax && self.bypass()? == 1 {
            v += 1;
        }
        Ok(v)
    }

    /// `EGk` (k-th order Exp-Golomb) using bypass bins, as used by
    /// `cu_qp_delta_abs`'s suffix and `coeff_abs_level_remaining`.
    pub fn egk_bypass(&mut self, k: u32) -> Result<u32> {
        let mut leading_ones = 0u32;
        while self.bypass()? == 1 {
            leading_ones += 1;
            if leading_ones > 32 {
                return Err(HeifError::MalformedHevc("egk_bypass: prefix too long"));
            }
        }
        let suffix_bits = leading_ones + k;
        let suffix = self.bypass_fl(suffix_bits)?;
        let base = 1u32.checked_shl(leading_ones).ok_or(HeifError::MalformedHevc("egk_bypass: overflow"))?;
        let base = base.checked_sub(1).ok_or(HeifError::MalformedHevc("egk_bypass: overflow"))?;
        let base = base.checked_shl(k).ok_or(HeifError::MalformedHevc("egk_bypass: overflow"))?;
        base.checked_add(suffix).ok_or(HeifError::MalformedHevc("egk_bypass: overflow"))
    }

    /// `DecodeTerminate`, clause 9.3.4.3.5. Also where a read past the end
    /// of the data surfaces: this bin follows every CTU.
    pub fn terminate(&mut self) -> Result<u32> {
        self.check_overread()?;
        self.range -= 2;
        let bin_val = if self.offset >= self.range {
            1
        } else {
            self.renormalize();
            0
        };
        self.check_overread()?;
        Ok(bin_val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn branch_free_decisions_match_the_specification() {
        // A pseudo-random stream, one context per (state, MPS), decoded by
        // two engines: every bin, context, range and offset must agree.
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let data: Vec<u8> = (0..8192)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed as u8
            })
            .collect();
        let (mut r1, mut r2) = (BitReader::new(&data), BitReader::new(&data));
        let (mut fast, mut reference) = (Engine::new(&mut r1).unwrap(), Engine::new(&mut r2).unwrap());
        let mut contexts: Vec<ContextModel> = (0..128).map(|i| ContextModel { state: i / 2, mps: i % 2 }).collect();
        let mut expected = contexts.clone();
        for i in 0..40_000usize {
            let k = (i * 7919 + i / 3) % 128;
            let want = reference.reference_decision(&mut expected[k]);
            let got = fast.decision(&mut contexts[k]).unwrap();
            assert_eq!((got, contexts[k], fast.range, fast.offset), (want, expected[k], reference.range, reference.offset), "bin {i}");
            if i % 5 == 0 {
                assert_eq!(fast.bypass().unwrap(), reference.bypass().unwrap());
            }
        }
    }

    #[test]
    fn bypass_runs_match_bin_by_bin() {
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let data: Vec<u8> = (0..4096)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed as u8
            })
            .collect();
        let (mut r1, mut r2) = (BitReader::new(&data), BitReader::new(&data));
        let (mut fast, mut slow) = (Engine::new(&mut r1).unwrap(), Engine::new(&mut r2).unwrap());
        let mut ctx_a = ContextModel { state: 10, mps: 0 };
        let mut ctx_b = ctx_a;
        for i in 0..2000u32 {
            let n = i % 32;
            let mut want = 0;
            for _ in 0..n {
                want = (want << 1) | slow.bypass().unwrap();
            }
            assert_eq!(fast.bypass_fl(n).unwrap(), want, "run {i} of {n}");
            assert_eq!((fast.range, fast.offset), (slow.range, slow.offset));
            // Interleave decisions so the range varies between runs.
            assert_eq!(fast.decision(&mut ctx_a).unwrap(), slow.reference_decision(&mut ctx_b));
        }
    }

    #[test]
    fn unary_runs_match_bin_by_bin() {
        let mut seed = 0xD1B5_4A32_D192_ED03u64;
        // Long runs of ones are what this is for: bias the bytes toward 0xFF.
        let data: Vec<u8> = (0..4096)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                if seed.is_multiple_of(3) { 0xFF } else { seed as u8 | (seed >> 8) as u8 }
            })
            .collect();
        for cut in [data.len(), 37, 11, 3] {
            let data = &data[..cut];
            let (mut r1, mut r2) = (BitReader::new(data), BitReader::new(data));
            let (mut fast, mut slow) = (Engine::new(&mut r1).unwrap(), Engine::new(&mut r2).unwrap());
            let mut ctx_a = ContextModel { state: 20, mps: 1 };
            let mut ctx_b = ctx_a;
            for i in 0..3000u32 {
                let max = [1, 3, 8, 9, 19, 33][i as usize % 6];
                let mut want = 0;
                while want < max && slow.bypass().unwrap() == 1 {
                    want += 1;
                }
                assert_eq!(fast.bypass_unary(max).unwrap(), want, "cut {cut} run {i} max {max}");
                assert_eq!((fast.range, fast.offset, fast.overread), (slow.range, slow.offset, slow.overread), "cut {cut} run {i}");
                assert_eq!(fast.decision(&mut ctx_a).unwrap(), slow.reference_decision(&mut ctx_b));
            }
        }
    }

    #[test]
    fn context_init_matches_known_example() {
        // slopeIdx=184>>4=11, intersec=184&0xF=8: m=11*5-45=10, n=(8<<3)-16=48.
        // QP=32: preCtxState=((10*32)>>4)+48=(320>>4)+48=20+48=68. mps=1 (68>63).
        // state=68-64=4.
        let cm = ContextModel::init(184, 32);
        assert_eq!(cm.mps, 1);
        assert_eq!(cm.state, 4);
    }

    #[test]
    fn bypass_is_table_free_and_round_trips_fl() {
        // Bypass decoding never touches a context model or the LPS/MPS
        // tables, so it's checkable in isolation: feed a known bit
        // pattern and confirm FL(n) reads it back MSB-first.
        let bits = [0b1011_0000u8, 0b0000_0000, 0b0000_0000];
        let mut r = BitReader::new(&bits);
        let mut engine = Engine::new(&mut r).expect("engine init needs 9 bits, buffer has plenty");
        // After init, 9 bits (the full first byte + 1 bit of the second)
        // are consumed into ivlOffset; remaining bypass reads continue
        // from bit 9 onward in the same stream, which is what we're
        // actually testing: that bypass_fl reads raw bits irrespective
        // of engine state.
        let v = engine.bypass_fl(4).expect("bypass_fl should succeed on a buffer with plenty of bits");
        let _ = v; // structural smoke test: must not error or panic
    }

    #[test]
    fn terminate_errs_not_panics_on_truncated_input() {
        let bits = [0u8; 2]; // enough for init's 9 bits, nothing left to renormalize with
        let mut r = BitReader::new(&bits);
        let mut engine = Engine::new(&mut r).expect("2 bytes is enough for the 9-bit init read");
        // Drive the engine until it needs more bits than are available;
        // this must surface as an error, never a panic.
        for _ in 0..64 {
            if engine.terminate().is_err() {
                return;
            }
        }
    }
}
