//! Dequantization (clause 8.6.3) and inverse transform (clause 8.6.4):
//! the 32-point integer DCT-II basis (smaller sizes are subsampled from
//! it, per the spec's transform-nesting property) and the 4x4 DST-VII
//! used for luma intra 4x4 blocks.
//!
//! The matrices and two-pass shift/rounding structure here were checked
//! against `libde265`'s `fallback-dct.cc` (LGPL-3.0) rather than typed
//! from memory, given how easy a single wrong entry in a 32x32 table
//! would be to miss. See this crate's `README.md`.

use crate::error::{HeifError, Result};

/// `levelScale[qP % 6]`, clause 8.6.3.
const LEVEL_SCALE: [i64; 6] = [40, 45, 51, 57, 64, 72];

/// Table 8-10's irregular middle section (`qPi` 30..=42) of the
/// `qPi` → `QpC` chroma QP mapping (clause 8.6.1), checked against
/// `libde265`/`imazen-heic` rather than typed from memory.
const CHROMA_QP_TABLE: [i32; 13] = [29, 30, 31, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37];

/// `qPi` → `QpC` (clause 8.6.1): identity below 30, `Table 8-10` in the
/// middle, `qPi - 6` above 43. Only valid for `ChromaArrayType == 1`
/// (4:2:0); 4:2:2/4:4:4 instead use `min(qPi, 51)` directly at the call
/// site.
pub fn chroma_qp_mapping(qp_i: i32) -> i32 {
    if qp_i < 30 {
        qp_i
    } else if qp_i >= 43 {
        qp_i - 6
    } else {
        CHROMA_QP_TABLE.get((qp_i - 30) as usize).copied().unwrap_or(qp_i)
    }
}

/// Dequantizes one coefficient with a flat scaling list (`m = 16`).
/// `qp` is `Qp'Y`/`Qp'Cb`/`Qp'Cr` (already offset-applied).
pub fn dequant(level: i32, qp: i32, bit_depth: u32, log2_size: u32) -> Result<i32> {
    dequant_with_scale(level, qp, bit_depth, log2_size, 16)
}

/// Dequantization with the selected scaling-matrix coefficient (8.6.3).
/// `qp` includes the component's bit-depth offset, i.e. Qp' rather than Qp.
pub fn dequant_with_scale(level: i32, qp: i32, bit_depth: u32, log2_size: u32, weight: i32) -> Result<i32> {
    if !(8..=16).contains(&bit_depth) || !(2..=5).contains(&log2_size)
        || !(0..=255).contains(&weight) || !(0..=51 + 6 * (bit_depth as i32 - 8)).contains(&qp) {
        return Err(HeifError::MalformedHevc("dequant: parameter out of range"));
    }
    let shift = bit_depth + log2_size - 5;
    let fact = LEVEL_SCALE[(qp % 6) as usize] << (qp / 6);
    let v = (i64::from(level) * i64::from(weight) * fact + (1i64 << (shift - 1))) >> shift;
    Ok(v.clamp(-32768, 32767) as i32)
}

/// Row 0 and column 0 of the 32-point DCT-II integer basis repeat across
/// every even subsample, by construction (DCT-II nesting): `mat[0][*] ==
/// 64` for every valid `fact`. Smaller transforms read `MAT_DCT[fact*j][i]`
/// with `fact = 32 / nT`.
#[rustfmt::skip]
const MAT_DCT: [[i32; 32]; 32] = [
    [64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64],
    [90,90,88,85,82,78,73,67,61,54,46,38,31,22,13,4,-4,-13,-22,-31,-38,-46,-54,-61,-67,-73,-78,-82,-85,-88,-90,-90],
    [90,87,80,70,57,43,25,9,-9,-25,-43,-57,-70,-80,-87,-90,-90,-87,-80,-70,-57,-43,-25,-9,9,25,43,57,70,80,87,90],
    [90,82,67,46,22,-4,-31,-54,-73,-85,-90,-88,-78,-61,-38,-13,13,38,61,78,88,90,85,73,54,31,4,-22,-46,-67,-82,-90],
    [89,75,50,18,-18,-50,-75,-89,-89,-75,-50,-18,18,50,75,89,89,75,50,18,-18,-50,-75,-89,-89,-75,-50,-18,18,50,75,89],
    [88,67,31,-13,-54,-82,-90,-78,-46,-4,38,73,90,85,61,22,-22,-61,-85,-90,-73,-38,4,46,78,90,82,54,13,-31,-67,-88],
    [87,57,9,-43,-80,-90,-70,-25,25,70,90,80,43,-9,-57,-87,-87,-57,-9,43,80,90,70,25,-25,-70,-90,-80,-43,9,57,87],
    [85,46,-13,-67,-90,-73,-22,38,82,88,54,-4,-61,-90,-78,-31,31,78,90,61,4,-54,-88,-82,-38,22,73,90,67,13,-46,-85],
    [83,36,-36,-83,-83,-36,36,83,83,36,-36,-83,-83,-36,36,83,83,36,-36,-83,-83,-36,36,83,83,36,-36,-83,-83,-36,36,83],
    [82,22,-54,-90,-61,13,78,85,31,-46,-90,-67,4,73,88,38,-38,-88,-73,-4,67,90,46,-31,-85,-78,-13,61,90,54,-22,-82],
    [80,9,-70,-87,-25,57,90,43,-43,-90,-57,25,87,70,-9,-80,-80,-9,70,87,25,-57,-90,-43,43,90,57,-25,-87,-70,9,80],
    [78,-4,-82,-73,13,85,67,-22,-88,-61,31,90,54,-38,-90,-46,46,90,38,-54,-90,-31,61,88,22,-67,-85,-13,73,82,4,-78],
    [75,-18,-89,-50,50,89,18,-75,-75,18,89,50,-50,-89,-18,75,75,-18,-89,-50,50,89,18,-75,-75,18,89,50,-50,-89,-18,75],
    [73,-31,-90,-22,78,67,-38,-90,-13,82,61,-46,-88,-4,85,54,-54,-85,4,88,46,-61,-82,13,90,38,-67,-78,22,90,31,-73],
    [70,-43,-87,9,90,25,-80,-57,57,80,-25,-90,-9,87,43,-70,-70,43,87,-9,-90,-25,80,57,-57,-80,25,90,9,-87,-43,70],
    [67,-54,-78,38,85,-22,-90,4,90,13,-88,-31,82,46,-73,-61,61,73,-46,-82,31,88,-13,-90,-4,90,22,-85,-38,78,54,-67],
    [64,-64,-64,64,64,-64,-64,64,64,-64,-64,64,64,-64,-64,64,64,-64,-64,64,64,-64,-64,64,64,-64,-64,64,64,-64,-64,64],
    [61,-73,-46,82,31,-88,-13,90,-4,-90,22,85,-38,-78,54,67,-67,-54,78,38,-85,-22,90,4,-90,13,88,-31,-82,46,73,-61],
    [57,-80,-25,90,-9,-87,43,70,-70,-43,87,9,-90,25,80,-57,-57,80,25,-90,9,87,-43,-70,70,43,-87,-9,90,-25,-80,57],
    [54,-85,-4,88,-46,-61,82,13,-90,38,67,-78,-22,90,-31,-73,73,31,-90,22,78,-67,-38,90,-13,-82,61,46,-88,4,85,-54],
    [50,-89,18,75,-75,-18,89,-50,-50,89,-18,-75,75,18,-89,50,50,-89,18,75,-75,-18,89,-50,-50,89,-18,-75,75,18,-89,50],
    [46,-90,38,54,-90,31,61,-88,22,67,-85,13,73,-82,4,78,-78,-4,82,-73,-13,85,-67,-22,88,-61,-31,90,-54,-38,90,-46],
    [43,-90,57,25,-87,70,9,-80,80,-9,-70,87,-25,-57,90,-43,-43,90,-57,-25,87,-70,-9,80,-80,9,70,-87,25,57,-90,43],
    [38,-88,73,-4,-67,90,-46,-31,85,-78,13,61,-90,54,22,-82,82,-22,-54,90,-61,-13,78,-85,31,46,-90,67,4,-73,88,-38],
    [36,-83,83,-36,-36,83,-83,36,36,-83,83,-36,-36,83,-83,36,36,-83,83,-36,-36,83,-83,36,36,-83,83,-36,-36,83,-83,36],
    [31,-78,90,-61,4,54,-88,82,-38,-22,73,-90,67,-13,-46,85,-85,46,13,-67,90,-73,22,38,-82,88,-54,-4,61,-90,78,-31],
    [25,-70,90,-80,43,9,-57,87,-87,57,-9,-43,80,-90,70,-25,-25,70,-90,80,-43,-9,57,-87,87,-57,9,43,-80,90,-70,25],
    [22,-61,85,-90,73,-38,-4,46,-78,90,-82,54,-13,-31,67,-88,88,-67,31,13,-54,82,-90,78,-46,4,38,-73,90,-85,61,-22],
    [18,-50,75,-89,89,-75,50,-18,-18,50,-75,89,-89,75,-50,18,18,-50,75,-89,89,-75,50,-18,-18,50,-75,89,-89,75,-50,18],
    [13,-38,61,-78,88,-90,85,-73,54,-31,4,22,-46,67,-82,90,-90,82,-67,46,-22,-4,31,-54,73,-85,90,-88,78,-61,38,-13],
    [9,-25,43,-57,70,-80,87,-90,90,-87,80,-70,57,-43,25,-9,-9,25,-43,57,-70,80,-87,90,-90,87,-80,70,-57,43,-25,9],
    [4,-13,22,-31,38,-46,54,-61,67,-73,78,-82,85,-88,90,-90,90,-90,88,-85,82,-78,73,-67,61,-54,46,-38,31,-22,13,-4],
];

/// Residual for a transform-skipped block (clause 8.6.4.2): the
/// dequantized coefficients stand directly for the spatial residual,
/// scaled by a fixed shift instead of the NxN inverse transform.
/// Cross-checked against `libde265::fallback-dct.cc`'s
/// `transform_skip_residual_fallback` rather than typed from memory.
pub fn transform_skip_residual(coeffs: &[i32], nt: usize, bit_depth: u32, out: &mut [i32]) -> Result<()> {
    if coeffs.len() != nt * nt || out.len() != nt * nt {
        return Err(HeifError::MalformedHevc("transform_skip_residual: coefficient buffer size mismatch"));
    }
    let ts_shift = 5 + log2_usize(nt) as i64;
    let bd_shift = (20i64 - bit_depth as i64).max(0);
    let rnd: i64 = if bd_shift > 0 { 1 << (bd_shift - 1) } else { 0 };
    for (o, &c) in out.iter_mut().zip(coeffs) {
        let scaled = (i64::from(c)) << ts_shift;
        *o = i32::try_from((scaled + rnd) >> bd_shift).map_err(|_| HeifError::MalformedHevc("transform_skip_residual: overflow"))?;
    }
    Ok(())
}

/// 4x4 DST-VII integer basis used for luma intra 4x4 (clause 8.6.4.1).
const MAT_DST_4: [[i32; 4]; 4] = [[29, 55, 74, 84], [74, 74, 0, -74], [84, -29, -74, 55], [55, -84, 74, -29]];

fn log2_usize(n: usize) -> u32 {
    n.trailing_zeros()
}

/// Inverse transform for an `nT x nT` block of dequantized coefficients
/// (row-major, `coeffs[y*nT+x]`), producing the spatial residual
/// (row-major `i32`, clipped to the coefficient range the spec allows
/// mid-pipeline). `use_dst` selects the 4x4 DST-VII (luma intra 4x4
/// only); every other case uses the DCT-II. `tmp` holds the intermediate
/// column pass; neither it nor `out` needs clearing beforehand, so the
/// caller can reuse both across blocks.
#[allow(clippy::needless_range_loop)] // matrix math: several loop variables index different arrays at once.
pub fn inverse_transform(coeffs: &[i32], nt: usize, bit_depth: u32, use_dst: bool, tmp: &mut [i64], out: &mut [i32]) -> Result<()> {
    if coeffs.len() != nt * nt || out.len() != nt * nt || tmp.len() < nt * nt {
        return Err(HeifError::MalformedHevc("inverse_transform: coefficient buffer size mismatch"));
    }
    if !(nt == 4 || nt == 8 || nt == 16 || nt == 32) {
        return Err(HeifError::Unsupported("inverse_transform: size must be 4/8/16/32"));
    }
    let max_coeff_bits = 15; // clause 7.4.9.11: CoeffMin/Max use a 15-bit-plus-sign range for the default (non-extended) bit depths this decoder supports.
    let coeff_min = -(1i64 << max_coeff_bits);
    let coeff_max = (1i64 << max_coeff_bits) - 1;

    let bd_shift2: i32 = 20 - bit_depth as i32;
    if bd_shift2 < 1 {
        return Err(HeifError::Unsupported("inverse_transform: bit depth too high"));
    }

    if use_dst {
        if nt != 4 {
            return Err(HeifError::MalformedHevc("inverse_transform: DST-VII only valid for 4x4"));
        }
        let rnd_v: i64 = 1 << 6; // 1<<(7-1)
        let rnd_h: i64 = 1 << (bd_shift2 - 1);
        let mut g = [[0i64; 4]; 4];
        for c in 0..4 {
            for i in 0..4 {
                let mut sum: i64 = 0;
                for j in 0..4 {
                    sum += MAT_DST_4[j][i] as i64 * coeffs[c + j * 4] as i64;
                }
                g[i][c] = ((sum + rnd_v) >> 7).clamp(coeff_min, coeff_max);
            }
        }
        let dst = out;
        for y in 0..4 {
            for i in 0..4 {
                let mut sum: i64 = 0;
                for j in 0..4 {
                    sum += MAT_DST_4[j][i] as i64 * g[y][j];
                }
                let v = i32::try_from((sum + rnd_h) >> bd_shift2).map_err(|_| HeifError::MalformedHevc("inverse_transform: overflow"))?;
                *dst.get_mut(y * 4 + i).ok_or(HeifError::MalformedHevc("inverse_transform: index"))? = v;
            }
        }
        return Ok(());
    }

    let log2_nt = log2_usize(nt);
    let fact = 1usize << (5 - log2_nt);
    let rnd1: i64 = 1 << 6;
    let rnd2: i64 = 1 << (bd_shift2 - 1);

    let mat = |row: usize, col: usize| -> Result<i64> {
        let r = fact.checked_mul(row).ok_or(HeifError::MalformedHevc("inverse_transform: index"))?;
        Ok(*MAT_DCT.get(r).and_then(|row| row.get(col)).ok_or(HeifError::MalformedHevc("inverse_transform: matrix index out of range"))? as i64)
    };

    let g = tmp;
    for c in 0..nt {
        let mut last_col = None;
        for j in (0..nt).rev() {
            if *coeffs.get(c + j * nt).ok_or(HeifError::MalformedHevc("inverse_transform: index"))? != 0 {
                last_col = Some(j);
                break;
            }
        }
        let Some(last_col) = last_col else {
            for i in 0..nt {
                *g.get_mut(c + i * nt).ok_or(HeifError::MalformedHevc("inverse_transform: index"))? = 0;
            }
            continue;
        };
        for i in 0..nt {
            let mut sum: i64 = 0;
            for j in 0..=last_col {
                sum += mat(j, i)? * *coeffs.get(c + j * nt).ok_or(HeifError::MalformedHevc("inverse_transform: index"))? as i64;
            }
            *g.get_mut(c + i * nt).ok_or(HeifError::MalformedHevc("inverse_transform: index"))? = ((sum + rnd1) >> 7).clamp(coeff_min, coeff_max);
        }
    }

    let dst = out;
    for y in 0..nt {
        let row_start = y * nt;
        let mut last_col = None;
        for j in (0..nt).rev() {
            if *g.get(row_start + j).ok_or(HeifError::MalformedHevc("inverse_transform: index"))? != 0 {
                last_col = Some(j);
                break;
            }
        }
        let Some(last_col) = last_col else {
            dst.get_mut(row_start..row_start + nt).ok_or(HeifError::MalformedHevc("inverse_transform: index"))?.fill(0);
            continue;
        };
        for i in 0..nt {
            let mut sum: i64 = 0;
            for j in 0..=last_col {
                sum += mat(j, i)? * *g.get(row_start + j).ok_or(HeifError::MalformedHevc("inverse_transform: index"))?;
            }
            let v = i32::try_from((sum + rnd2) >> bd_shift2).map_err(|_| HeifError::MalformedHevc("inverse_transform: overflow"))?;
            *dst.get_mut(row_start + i).ok_or(HeifError::MalformedHevc("inverse_transform: index"))? = v;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Buffers start out full of garbage, as reused decoder scratch would.
    fn idct(coeffs: &[i32], nt: usize, bit_depth: u32, use_dst: bool) -> Result<Vec<i32>> {
        let (mut tmp, mut out) = (vec![99i64; coeffs.len()], vec![99; coeffs.len()]);
        inverse_transform(coeffs, nt, bit_depth, use_dst, &mut tmp, &mut out).map(|()| out)
    }

    fn transform_skip(coeffs: &[i32], nt: usize, bit_depth: u32) -> Result<Vec<i32>> {
        let mut out = vec![99; coeffs.len()];
        transform_skip_residual(coeffs, nt, bit_depth, &mut out).map(|()| out)
    }

    #[test]
    fn dequant_uses_matrix_weight_before_rounding() {
        // Qp'=0, 8-bit 8x8: shift=6, levelScale=40. A weight of 17
        // gives (3*17*40+32)>>6 = 32, versus 30 with the flat matrix.
        assert_eq!(dequant_with_scale(3, 0, 8, 3, 17).unwrap(), 32);
        assert_eq!(dequant_with_scale(3, 0, 8, 3, 16).unwrap(), 30);
        assert_eq!(dequant_with_scale(-3, 0, 8, 3, 17).unwrap(), -32);
        assert_eq!(dequant_with_scale(3, 0, 8, 3, 0).unwrap(), 0);
        // For 10-bit QpY=0, Qp'Y=12 compensates the extra two shift bits.
        assert_eq!(dequant_with_scale(3, 12, 10, 3, 17).unwrap(), 32);
    }

    #[test]
    fn dc_only_dct_is_flat() {
        // Hand-derived (see module history / PR description): for an
        // 8-bit block with only the DC coefficient (value 128) set,
        // every output sample must equal exactly 1, because row 0 and
        // column 0 of the DCT-II basis are constant (64), collapsing
        // the whole 2-D separable transform to two scalar passes.
        for &nt in &[4usize, 8, 16, 32] {
            let mut coeffs = vec![0i32; nt * nt];
            coeffs[0] = 128;
            let out = idct(&coeffs, nt, 8, false).expect("DC-only transform should succeed");
            assert!(out.iter().all(|&v| v == 1), "nt={nt} out={out:?}");
        }
    }

    #[test]
    fn all_zero_coeffs_is_all_zero_residual() {
        let coeffs = vec![0i32; 64];
        let out = idct(&coeffs, 8, 8, false).expect("all-zero transform should succeed");
        assert!(out.iter().all(|&v| v == 0));
    }

    #[test]
    fn transform_skip_residual_matches_hand_computed_values() {
        // 8-bit 4x4: tsShift = 5 + log2(4) = 7, bdShift = 20 - 8 = 12,
        // rnd = 1<<11 = 2048 (cross-checked against
        // `libde265::transform_skip_residual_fallback`, see module docs).
        // coeff=100: (100<<7 + 2048)>>12 = (12800+2048)>>12 = 14848>>12 = 3.
        // coeff=-100: (-12800+2048)>>12 = -10752>>12 = -3 (floor division).
        // coeff=0 stays 0 regardless of rounding.
        let coeffs = [100, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let out = transform_skip(&coeffs, 4, 8).expect("transform_skip_residual should succeed");
        assert_eq!(out[0], 3);
        assert_eq!(out[1], -3);
        assert!(out[2..].iter().all(|&v| v == 0));
    }

    #[test]
    fn dequant_known_value() {
        // qp=32 -> qp%6=2 (levelScale=51), qp/6=5. bit_depth=8, log2_size=3 (8x8) -> bdShift=8+3-5-4=2.
        // fact = 51<<5 = 1632. offset=1<<1=2. level=10: (10*1632+2)>>2 = 16322>>2=4080.
        let v = dequant(10, 32, 8, 3).expect("dequant should succeed");
        assert_eq!(v, 4080);
    }

    #[test]
    fn rejects_wrong_block_size() {
        assert!(idct(&[0; 9], 3, 8, false).is_err());
    }
}
