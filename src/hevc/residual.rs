//! `residual_coding()` (clause 7.3.8.11 / 9.3.4.2.5–9.3.4.2.8): the
//! last-significant-coefficient position, per-4x4-subblock significance
//! map, greater-than-1/2 flags, signs (with optional sign-data-hiding),
//! and `coeff_abs_level_remaining` Rice/EGk escape coding.
//!
//! Residual decoding uses the base HEVC tools, including basic 4:2:2
//! and 4:4:4 streams. Advanced Range Extensions tools (persistent
//! Rice adaptation, transform-skip context, explicit/implicit RDPCM,
//! bypass alignment) are not implemented here — see the
//! crate README. The context derivation and coefficient reconstruction
//! formulas were checked against `libde265`'s `slice.cc` (LGPL-3.0)
//! rather than typed from memory.

use super::cabac::Engine;
use super::contexts::Contexts;
use super::scan::{Pos, intra_scan_idx, scan_order};
use crate::error::{HeifError, Result};

/// `ctxIdxMap[16]` (clause 9.3.4.2.5), used only for 4x4 ( `log2TrafoSize
/// == 2`) transform blocks.
const CTX_IDX_MAP_4X4: [u8; 16] = [0, 1, 4, 5, 2, 3, 4, 5, 6, 6, 8, 8, 7, 7, 8, 99];

fn decode_last_sig_coeff_prefix(engine: &mut Engine, log2_trafo_size: u32, is_chroma: bool, ctx: &mut [super::cabac::ContextModel]) -> Result<u32> {
    let c_max = (log2_trafo_size * 2).saturating_sub(1);
    let (ctx_offset, ctx_shift) = if !is_chroma {
        (3 * (log2_trafo_size - 2) + ((log2_trafo_size.saturating_sub(1)) >> 2), (log2_trafo_size + 1) >> 2)
    } else {
        (15, log2_trafo_size.saturating_sub(2))
    };
    let mut value = c_max;
    for bin_idx in 0..c_max {
        let ctx_inc = ctx_offset + (bin_idx >> ctx_shift);
        let model = ctx.get_mut(ctx_inc as usize).ok_or(HeifError::MalformedHevc("last_sig_coeff_prefix: context index out of range"))?;
        if engine.decision(model)? == 0 {
            value = bin_idx;
            break;
        }
    }
    Ok(value)
}

fn decode_last_sig_coeff_value(engine: &mut Engine, prefix: u32) -> Result<u32> {
    if prefix > 3 {
        let n_bits = (prefix >> 1) - 1;
        let suffix = engine.bypass_fl(n_bits)?;
        Ok(((2 + (prefix & 1)) << n_bits) + suffix)
    } else {
        Ok(prefix)
    }
}

/// `sigCtx` derivation (clause 9.3.4.2.5), for a coefficient at `(x, y)`
/// within an `nt x nt` block whose containing 4x4 subblock has coded-flag
/// neighbours `prev_csbf` (bit0 = right neighbour coded, bit1 = below).
fn sig_ctx(x: u32, y: u32, sb_width: u32, cidx: usize, scan_idx: u32, prev_csbf: u32) -> u32 {
    if sb_width == 1 {
        let idx = ((y << 2) + x) as usize;
        return CTX_IDX_MAP_4X4.get(idx).copied().unwrap_or(0) as u32;
    }
    if x + y == 0 {
        return 0;
    }
    let x_s = x >> 2;
    let y_s = y >> 2;
    let xp = x & 3;
    let yp = y & 3;
    let mut sig = match prev_csbf {
        0 => {
            if xp + yp >= 3 {
                0
            } else if xp + yp > 0 {
                1
            } else {
                2
            }
        }
        1 => {
            if yp == 0 {
                2
            } else if yp == 1 {
                1
            } else {
                0
            }
        }
        2 => {
            if xp == 0 {
                2
            } else if xp == 1 {
                1
            } else {
                0
            }
        }
        _ => 2,
    };
    if cidx == 0 {
        if x_s + y_s > 0 {
            sig += 3;
        }
        if sb_width == 2 {
            sig += if scan_idx == 0 { 9 } else { 15 };
        } else {
            sig += 21;
        }
    } else if sb_width == 2 {
        sig += 9;
    } else {
        sig += 12;
    }
    sig
}

fn sig_coeff_ctx_inc(x: u32, y: u32, sb_width: u32, cidx: usize, scan_idx: u32, prev_csbf: u32) -> u32 {
    let sig = sig_ctx(x, y, sb_width, cidx, scan_idx, prev_csbf);
    if cidx == 0 { sig } else { 27 + sig }
}

fn decode_coeff_abs_level_remaining(engine: &mut Engine, rice_param: u32) -> Result<u32> {
    const MAX_PREFIX: u32 = 15 + 3;
    let mut prefix = 0u32;
    while engine.bypass()? == 1 {
        prefix += 1;
        if prefix > MAX_PREFIX {
            return Ok(0);
        }
    }
    if prefix <= 3 {
        let codeword = engine.bypass_fl(rice_param)?;
        Ok((prefix << rice_param) + codeword)
    } else {
        let codeword = engine.bypass_fl(prefix - 3 + rice_param)?;
        let base = (1u32.checked_shl(prefix - 3).ok_or(HeifError::MalformedHevc("coeff_abs_level_remaining: overflow"))? + 3 - 1) << rice_param;
        Ok(base + codeword)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn decode_residual(
    engine: &mut Engine,
    ctxs: &mut Contexts,
    log2_trafo_size: u32,
    cidx: usize,
    intra_pred_mode: u32,
    is_luma_or_444: bool,
    sign_data_hiding_enabled: bool,
    coeffs: &mut [i32],
) -> Result<()> {
    let nt = 1usize << log2_trafo_size;
    let coeffs = coeffs.get_mut(..nt * nt).ok_or(HeifError::MalformedHevc("residual: coefficient buffer too small"))?;
    coeffs.fill(0);

    let scan_idx = intra_scan_idx(log2_trafo_size, intra_pred_mode, is_luma_or_444);
    let is_chroma = cidx != 0;

    let last_x_prefix = decode_last_sig_coeff_prefix(engine, log2_trafo_size, is_chroma, &mut ctxs.last_sig_coeff_x_prefix)?;
    let last_y_prefix = decode_last_sig_coeff_prefix(engine, log2_trafo_size, is_chroma, &mut ctxs.last_sig_coeff_y_prefix)?;
    let mut last_x = decode_last_sig_coeff_value(engine, last_x_prefix)?;
    let mut last_y = decode_last_sig_coeff_value(engine, last_y_prefix)?;
    if scan_idx == 2 {
        std::mem::swap(&mut last_x, &mut last_y);
    }

    let sub_scan = scan_order(log2_trafo_size.saturating_sub(2), scan_idx)?;
    let pos_scan = scan_order(2, scan_idx)?;
    let sb_width = 1u32 << log2_trafo_size.saturating_sub(2);

    // Locate the last coefficient's (subBlock, scanPosInSubblock).
    let last_sb_xy = Pos { x: last_x >> 2, y: last_y >> 2 };
    let last_sb = sub_scan.iter().position(|&p| p == last_sb_xy).ok_or(HeifError::MalformedHevc("residual: last coeff subblock not found"))?;
    let last_pos_xy = Pos { x: last_x & 3, y: last_y & 3 };
    let last_scan_pos = pos_scan.iter().position(|&p| p == last_pos_xy).ok_or(HeifError::MalformedHevc("residual: last coeff scan pos not found"))?;

    // At most 8x8 subblocks (a 32x32 transform).
    let mut csbf_buf = [0u8; 64];
    let coded_sub_block_neighbors = csbf_buf.get_mut(..(sb_width * sb_width) as usize).ok_or(HeifError::MalformedHevc("residual: transform too large"))?;
    let mut c1 = 1i32;

    for i in (0..=last_sb).rev() {
        let s = *sub_scan.get(i).ok_or(HeifError::MalformedHevc("residual: subblock scan"))?;
        let sb_idx = (s.x + s.y * sb_width) as usize;
        let prev_csbf = *coded_sub_block_neighbors.get(sb_idx).unwrap_or(&0) as u32;

        let mut infer_sb_dc_sig = false;
        let sub_block_is_coded = if i < last_sb && i > 0 {
            let ctx_inc = (prev_csbf & 1 | (prev_csbf >> 1)) + if is_chroma { 2 } else { 0 };
            let model = ctxs.coded_sub_block_flag.get_mut(ctx_inc as usize).ok_or(HeifError::MalformedHevc("coded_sub_block_flag: context index"))?;
            let coded = engine.decision(model)? == 1;
            infer_sb_dc_sig = true;
            coded
        } else {
            true
        };

        if sub_block_is_coded {
            if s.x > 0
                && let Some(n) = coded_sub_block_neighbors.get_mut((s.x - 1 + s.y * sb_width) as usize)
            {
                *n |= 1;
            }
            if s.y > 0
                && let Some(n) = coded_sub_block_neighbors.get_mut((s.x + (s.y - 1) * sb_width) as usize)
            {
                *n |= 2;
            }
        }
        if !sub_block_is_coded {
            continue;
        }

        // A 4x4 subblock has at most 16 significant positions.
        let mut sig_buf = [0usize; 16];
        let mut n_sig = 0usize;
        let mut push_sig = |pos: usize| -> Result<()> {
            *sig_buf.get_mut(n_sig).ok_or(HeifError::MalformedHevc("residual: too many significant coefficients"))? = pos;
            n_sig += 1;
            Ok(())
        };
        // The last subblock's final coefficient (`last_scan_pos`) was
        // already pushed above as the guaranteed-significant last
        // coefficient, so the AC loop below must start one position
        // *before* it, not re-decode it.
        let last_coeff_in_sb = if i == last_sb { last_scan_pos.saturating_sub(1) } else { 15 };
        if i == last_sb {
            push_sig(last_scan_pos)?;
        }

        for n in (1..=last_coeff_in_sb).rev() {
            let p = *pos_scan.get(n).ok_or(HeifError::MalformedHevc("residual: pos scan"))?;
            let xc = (s.x << 2) + p.x;
            let yc = (s.y << 2) + p.y;
            let ctx_inc = sig_coeff_ctx_inc(xc, yc, sb_width, cidx, scan_idx, prev_csbf);
            let model = ctxs.significant_coeff_flag.get_mut(ctx_inc as usize).ok_or(HeifError::MalformedHevc("significant_coeff_flag: context index"))?;
            if engine.decision(model)? == 1 {
                push_sig(n)?;
                infer_sb_dc_sig = false;
            }
        }
        // DC of this subblock (scan position 0): coded unless this is
        // the lone last-coefficient subblock whose last position *is*
        // the DC (then there's nothing left to code below it).
        if !(i == last_sb && last_scan_pos == 0) {
            if infer_sb_dc_sig {
                push_sig(0)?;
            } else {
                let p0 = *pos_scan.first().ok_or(HeifError::MalformedHevc("residual: pos scan"))?;
                let xc = (s.x << 2) + p0.x;
                let yc = (s.y << 2) + p0.y;
                let ctx_inc = sig_coeff_ctx_inc(xc, yc, sb_width, cidx, scan_idx, prev_csbf);
                let model = ctxs.significant_coeff_flag.get_mut(ctx_inc as usize).ok_or(HeifError::MalformedHevc("significant_coeff_flag: context index"))?;
                if engine.decision(model)? == 1 {
                    push_sig(0)?;
                }
            }
        }

        let sig_positions = &sig_buf[..n_sig];
        if sig_positions.is_empty() {
            continue;
        }

        // `sig_positions` is currently in descending scan-position order
        // (as found above); the spec processes greater1/greater2/signs
        // in that same order (highest scan position, i.e. closest to
        // the last coefficient, first).
        let n_coeff = sig_positions.len();
        let ctx_set_base = if i == 0 || is_chroma { 0 } else { 2 } + if c1 == 0 { 1 } else { 0 };
        c1 = 1;

        let mut level_buf = [1i32; 16];
        let mut has_max_buf = [true; 16];
        let level = &mut level_buf[..n_coeff];
        let has_max_base_level = &mut has_max_buf[..n_coeff];
        let last_g1 = n_coeff.min(8);
        let mut greater1_ctx = 1i32;
        let mut new_last_g1_pos: Option<usize> = None;
        for (c, lvl) in level.iter_mut().enumerate().take(last_g1) {
            let ctx_inc = (ctx_set_base * 4 + greater1_ctx.min(3)) as u32 + if is_chroma { 16 } else { 0 };
            let model =
                ctxs.coeff_abs_level_greater1_flag.get_mut(ctx_inc as usize).ok_or(HeifError::MalformedHevc("coeff_abs_level_greater1_flag: context index"))?;
            let flag = engine.decision(model)?;
            if flag == 1 {
                *lvl += 1;
                c1 = 0;
                if new_last_g1_pos.is_none() {
                    new_last_g1_pos = Some(c);
                }
                greater1_ctx = 0;
            } else {
                if let Some(h) = has_max_base_level.get_mut(c) {
                    *h = false;
                }
                if greater1_ctx > 0 {
                    greater1_ctx += 1;
                }
            }
        }

        if let Some(pos) = new_last_g1_pos {
            let ctx_inc = ctx_set_base as u32 + if is_chroma { 4 } else { 0 };
            let model =
                ctxs.coeff_abs_level_greater2_flag.get_mut(ctx_inc as usize).ok_or(HeifError::MalformedHevc("coeff_abs_level_greater2_flag: context index"))?;
            let flag = engine.decision(model)?;
            if let Some(lvl) = level.get_mut(pos) {
                *lvl += flag as i32;
            }
            if let Some(h) = has_max_base_level.get_mut(pos) {
                *h = flag == 1;
            }
        }

        // `sig_positions` is pushed in descending scan-position order, so
        // `[0]` is the highest (closest to the last coefficient) and
        // `[last]` the lowest (DC-ward) — exactly the spec's
        // `coeff_scan_pos[0]` / `coeff_scan_pos[nCoefficients-1]`.
        let first_scan_pos = *sig_positions.first().ok_or(HeifError::MalformedHevc("residual: sig positions"))?;
        let last_scan_pos_in_sb = *sig_positions.last().ok_or(HeifError::MalformedHevc("residual: sig positions"))?;
        let sign_hidden = first_scan_pos.saturating_sub(last_scan_pos_in_sb) > 3;

        let mut sign_buf = [false; 16];
        let sign = &mut sign_buf[..n_coeff];
        for s in sign.iter_mut().take(n_coeff - 1) {
            *s = engine.bypass()? == 1;
        }
        if (!sign_data_hiding_enabled || !sign_hidden)
            && let Some(s) = sign.get_mut(n_coeff - 1)
        {
            *s = engine.bypass()? == 1;
        }

        let mut rice_param = 0u32;
        let mut sum_abs: i64 = 0;
        for n in 0..n_coeff {
            let base_level = *level.get(n).ok_or(HeifError::MalformedHevc("residual: level"))?;
            let remaining = if *has_max_base_level.get(n).unwrap_or(&false) {
                let r = decode_coeff_abs_level_remaining(engine, rice_param)?;
                if base_level as u32 + r > 3 * (1 << rice_param) {
                    rice_param = (rice_param + 1).min(4);
                }
                r as i32
            } else {
                0
            };
            let mut curr = base_level + remaining;
            if *sign.get(n).ok_or(HeifError::MalformedHevc("residual: sign"))? {
                curr = -curr;
            }
            if sign_data_hiding_enabled && sign_hidden {
                sum_abs += curr as i64;
                if n == n_coeff - 1 && (sum_abs & 1) != 0 {
                    curr = -curr;
                }
            }
            let scan_pos = *sig_positions.get(n).ok_or(HeifError::MalformedHevc("residual: sig positions"))?;
            let p = *pos_scan.get(scan_pos).ok_or(HeifError::MalformedHevc("residual: pos scan"))?;
            let xc = (s.x << 2) + p.x;
            let yc = (s.y << 2) + p.y;
            let out_idx = (xc + (yc << log2_trafo_size)) as usize;
            *coeffs.get_mut(out_idx).ok_or(HeifError::MalformedHevc("residual: coeff out of range"))? = curr.clamp(-32768, 32767);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sig_ctx_4x4_uses_ctx_idx_map() {
        assert_eq!(sig_ctx(0, 0, 1, 0, 0, 0), 0);
        assert_eq!(sig_ctx(3, 3, 1, 0, 0, 0), 99); // last entry of ctxIdxMap
    }

    #[test]
    fn sig_ctx_dc_of_block_is_always_zero() {
        assert_eq!(sig_ctx(0, 0, 2, 0, 0, 0), 0);
        assert_eq!(sig_ctx(0, 0, 2, 1, 1, 3), 0);
    }

    #[test]
    fn coeff_abs_level_remaining_tr_only_path() {
        // prefix<=3 path: bypass_fl should be exercised; smoke test only
        // (bypass bins need no context table, so this is purely a
        // mechanical check that the function doesn't error/panic on a
        // trivial all-zero bypass stream).
        let bits = [0u8; 4];
        let mut r = crate::bitreader::BitReader::new(&bits);
        let mut engine = Engine::new(&mut r).expect("engine init");
        let v = decode_coeff_abs_level_remaining(&mut engine, 0).expect("should decode without error");
        assert_eq!(v, 0); // all-zero bypass bits: prefix terminates immediately at 0
    }
}
