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
    if !(8..=16).contains(&bit_depth) || !(2..=5).contains(&log2_size) || !(0..=255).contains(&weight) || !(0..=51 + 6 * (bit_depth as i32 - 8)).contains(&qp) {
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

/// The `n`-point DCT-II basis: rows `0, 32/n, 2*32/n, ...` of [`MAT_DCT`], first `n` columns.
const fn dct_matrix<const N: usize>() -> [[i32; N]; N] {
    let mut m = [[0; N]; N];
    let mut j = 0;
    while j < N {
        let mut i = 0;
        while i < N {
            m[j][i] = MAT_DCT[32 / N * j][i];
            i += 1;
        }
        j += 1;
    }
    m
}

const MAT_DCT_4: [[i32; 4]; 4] = dct_matrix();
const MAT_DCT_8: [[i32; 8]; 8] = dct_matrix();
const MAT_DCT_16: [[i32; 16]; 16] = dct_matrix();
const MAT_DCT_32: [[i32; 32]; 32] = dct_matrix();

/// `Clip3(coeffMin, coeffMax, v)` with the 16-bit coefficient range (clause 8.6.4.2).
fn clip_coeff(v: i32) -> i32 {
    v.clamp(-32768, 32767)
}

/// Inverse transform for an `nT x nT` block of dequantized coefficients
/// (row-major, `coeffs[y*nT+x]`), producing the spatial residual
/// (row-major `i32`). `use_dst` selects the 4x4 DST-VII (luma intra 4x4
/// only); every other case uses the DCT-II. `out` needs no clearing
/// beforehand, so the caller can reuse it across blocks.
pub fn inverse_transform(coeffs: &[i32], nt: usize, bit_depth: u32, use_dst: bool, out: &mut [i32]) -> Result<()> {
    if coeffs.len() != nt * nt || out.len() != nt * nt {
        return Err(HeifError::MalformedHevc("inverse_transform: coefficient buffer size mismatch"));
    }
    let bd_shift = match 20u32.checked_sub(bit_depth) {
        Some(shift) if shift >= 1 => shift,
        _ => return Err(HeifError::Unsupported("inverse_transform: bit depth too high")),
    };
    match (nt, use_dst) {
        (4, true) => two_stage(coeffs, &MAT_DST_4, bd_shift, out),
        (_, true) => return Err(HeifError::MalformedHevc("inverse_transform: DST-VII only valid for 4x4")),
        (4, false) => two_stage(coeffs, &MAT_DCT_4, bd_shift, out),
        (8, false) => two_stage(coeffs, &MAT_DCT_8, bd_shift, out),
        (16, false) => two_stage(coeffs, &MAT_DCT_16, bd_shift, out),
        (32, false) => two_stage(coeffs, &MAT_DCT_32, bd_shift, out),
        _ => return Err(HeifError::Unsupported("inverse_transform: size must be 4/8/16/32")),
    }
    Ok(())
}

/// The separable two-stage transform of clause 8.6.4.2 with basis `m`
/// (`m[j][i]`: basis function `j` at sample `i`). `coeffs` and `out` hold
/// exactly `N * N` values.
///
/// Everything fits `i32`: both stages take inputs clipped to 16 bits (the
/// input here too, which dequantization already guarantees), and a sum of
/// 32 products of a 16-bit value with a basis entry of at most 90 stays
/// below 2^27. The loops run over whole fixed-size rows, so the compiler
/// can vectorise them; all-zero coefficient rows, most of a typical
/// block, are skipped.
fn two_stage<const N: usize>(coeffs: &[i32], m: &[[i32; N]; N], bd_shift: u32, out: &mut [i32]) {
    // Stage 1 (vertical): g[i][c] = sum_j m[j][i] * d[j][c], accumulated
    // one coefficient row j at a time so the inner loop runs along c.
    let mut g = [[0i32; N]; N];
    for (row, basis) in coeffs.as_chunks::<N>().0.iter().zip(m) {
        if row.iter().all(|&d| d == 0) {
            continue;
        }
        let d = row.map(clip_coeff);
        for (g_row, &weight) in g.iter_mut().zip(basis) {
            for (acc, &dc) in g_row.iter_mut().zip(&d) {
                *acc += weight * dc;
            }
        }
    }
    for g_row in &mut g {
        for v in g_row.iter_mut() {
            *v = clip_coeff((*v + 64) >> 7);
        }
    }

    // Stage 2 (horizontal): out[y][i] = sum_j m[j][i] * g[y][j].
    let rnd = 1i32 << (bd_shift - 1);
    for (dst, g_row) in out.as_chunks_mut::<N>().0.iter_mut().zip(&g) {
        let mut acc = [0i32; N];
        for (&gj, basis) in g_row.iter().zip(m) {
            if gj == 0 {
                continue;
            }
            for (a, &weight) in acc.iter_mut().zip(basis) {
                *a += weight * gj;
            }
        }
        for (o, a) in dst.iter_mut().zip(acc) {
            *o = (a + rnd) >> bd_shift;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Buffers start out full of garbage, as reused decoder scratch would.
    fn idct(coeffs: &[i32], nt: usize, bit_depth: u32, use_dst: bool) -> Result<Vec<i32>> {
        let mut out = vec![99; coeffs.len()];
        inverse_transform(coeffs, nt, bit_depth, use_dst, &mut out).map(|()| out)
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

    /// The two stages written out directly from clause 8.6.4.2 in `i64`,
    /// indexing the 32-point basis as the spec does.
    fn reference(coeffs: &[i32], nt: usize, bit_depth: u32, use_dst: bool) -> Vec<i32> {
        let m = |j: usize, i: usize| i64::from(if use_dst { MAT_DST_4[j][i] } else { MAT_DCT[32 / nt * j][i] });
        let mut g = vec![0i64; nt * nt];
        for c in 0..nt {
            for i in 0..nt {
                let sum: i64 = (0..nt).map(|j| m(j, i) * i64::from(coeffs[j * nt + c])).sum();
                g[i * nt + c] = ((sum + 64) >> 7).clamp(-32768, 32767);
            }
        }
        let shift = 20 - bit_depth;
        (0..nt * nt)
            .map(|k| {
                let (y, i) = (k / nt, k % nt);
                let sum: i64 = (0..nt).map(|j| m(j, i) * g[y * nt + j]).sum();
                ((sum + (1 << (shift - 1))) >> shift) as i32
            })
            .collect()
    }

    #[test]
    fn matches_the_spec_formula_on_random_blocks() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for (nt, use_dst) in [(4, true), (4, false), (8, false), (16, false), (32, false)] {
            for bit_depth in [8, 10, 12] {
                for _ in 0..200 {
                    // Mostly zeros, like real blocks, with full-range values in between.
                    let coeffs: Vec<i32> = (0..nt * nt)
                        .map(|_| {
                            let r = next();
                            if r % 4 == 0 { (r >> 16) as i32 % 32768 } else { 0 }
                        })
                        .collect();
                    assert_eq!(
                        idct(&coeffs, nt, bit_depth, use_dst).unwrap(),
                        reference(&coeffs, nt, bit_depth, use_dst),
                        "nt={nt} dst={use_dst} bd={bit_depth}"
                    );
                }
            }
        }
    }

    #[test]
    fn extreme_coefficients_do_not_overflow() {
        // Valid streams never exceed 16 bits here (dequantization clips);
        // larger values are clipped on input instead of overflowing `i32`.
        for nt in [4, 8, 16, 32] {
            for v in [32767, -32768, i32::MAX, i32::MIN] {
                let coeffs = vec![v; nt * nt];
                let clipped = vec![v.clamp(-32768, 32767); nt * nt];
                assert_eq!(idct(&coeffs, nt, 8, false).unwrap(), reference(&clipped, nt, 8, false));
            }
        }
    }

    #[test]
    fn rejects_wrong_block_size() {
        assert!(idct(&[0; 9], 3, 8, false).is_err());
    }
}
