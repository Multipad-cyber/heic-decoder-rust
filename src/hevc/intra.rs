//! Intra prediction: reference-sample fetch/substitution/filtering
//! (clause 8.4.4.2.2–8.4.4.2.3) and the three prediction modes — planar,
//! DC, angular (clause 8.4.4.2.4–8.4.4.2.6). The exact arithmetic (shifts,
//! rounding, the boundary-filter special cases for modes 10/26, the
//! strong-intra-smoothing bilinear path) was checked against
//! `libde265`'s `intrapred.h` (LGPL-3.0) rather than typed from memory.
//! See this crate's `README.md`.

use crate::error::{HeifError, Result};

/// `intraPredAngle[predModeIntra]`, Table 8-5, indices 2..=34 valid.
const INTRA_PRED_ANGLE: [i32; 35] =
    [0, 0, 32, 26, 21, 17, 13, 9, 5, 2, 0, -2, -5, -9, -13, -17, -21, -26, -32, -26, -21, -17, -13, -9, -5, -2, 0, 2, 5, 9, 13, 17, 21, 26, 32];

/// `invAngle[predModeIntra]`, Table 8-5, indices 11..=24 valid.
const INV_ANGLE: [i32; 15] = [-4096, -1638, -910, -630, -482, -390, -315, -256, -315, -390, -482, -630, -910, -1638, -4096];

/// The reference sample border for one prediction block: flattened,
/// logically indexed `-2*nt ..= 2*nt` (index 0 of the array is position
/// `-2*nt`; `-1` in spec terms maps to `border[2*nt]` / the "top-left
/// corner", following libde265's `out_border` convention exactly).
pub struct Border {
    data: [i32; MAX_BORDER],
    nt: usize,
}

/// Border length for the largest (32x32) block: `4 * 32 + 1` samples.
const MAX_BORDER: usize = 4 * 32 + 1;

impl Border {
    /// The `4 * nt + 1` samples in use; `nt <= 32` is checked in `build`.
    fn samples(&self) -> &[i32] {
        self.data.get(..4 * self.nt + 1).unwrap_or(&[])
    }

    fn idx(&self, i: isize) -> Result<usize> {
        let base = 2 * self.nt as isize;
        usize::try_from(i + base).map_err(|_| HeifError::MalformedHevc("intra border index out of range"))
    }

    pub fn get(&self, i: isize) -> Result<i32> {
        let idx = self.idx(i)?;
        self.samples().get(idx).copied().ok_or(HeifError::MalformedHevc("intra border index out of range"))
    }

    /// Builds the border for an `nt x nt` block at `(x0, y0)` in the
    /// current plane. `fetch(x, y)` returns `None` when that neighbour
    /// is outside the picture or not yet decoded (clause 8.4.4.2.1/8.4.4.2.2
    /// simplified: the caller supplies only reconstructed samples in
    /// the current independent slice and HEVC tile).
    pub fn build(x0: i64, y0: i64, nt: usize, bit_depth: u32, fetch: impl Fn(i64, i64) -> Option<i32>) -> Result<Self> {
        let n = nt as i64;
        let len = 4 * nt + 1;
        if len > MAX_BORDER {
            return Err(HeifError::MalformedHevc("intra border: block larger than 32x32"));
        }
        let mut avail_buf = [false; MAX_BORDER];
        let mut data_buf = [0i32; MAX_BORDER];
        let avail = &mut avail_buf[..len];
        let data = &mut data_buf[..len];
        let base = 2 * nt as i64;

        // Left column, bottom (y0+2n-1) to top (y0), index -2n..0 (exclusive of 0, which is the corner).
        for k in 0..(2 * nt) {
            let y = y0 + (2 * n - 1 - k as i64);
            let i = -(2 * n) + k as i64;
            if let Some(v) = fetch(x0 - 1, y) {
                let idx = usize::try_from(i + base).map_err(|_| HeifError::MalformedHevc("intra border build index"))?;
                if let Some(slot) = data.get_mut(idx) {
                    *slot = v;
                }
                if let Some(slot) = avail.get_mut(idx) {
                    *slot = true;
                }
            }
        }
        // Top-left corner, index 0.
        if let Some(v) = fetch(x0 - 1, y0 - 1) {
            let idx = usize::try_from(base).map_err(|_| HeifError::MalformedHevc("intra border build index"))?;
            if let Some(slot) = data.get_mut(idx) {
                *slot = v;
            }
            if let Some(slot) = avail.get_mut(idx) {
                *slot = true;
            }
        }
        // Top row, left (x0) to right (x0+2n-1), index 1..=2n.
        for k in 0..(2 * nt) {
            let x = x0 + k as i64;
            let i = 1 + k as i64;
            if let Some(v) = fetch(x, y0 - 1) {
                let idx = usize::try_from(i + base).map_err(|_| HeifError::MalformedHevc("intra border build index"))?;
                if let Some(slot) = data.get_mut(idx) {
                    *slot = v;
                }
                if let Some(slot) = avail.get_mut(idx) {
                    *slot = true;
                }
            }
        }

        let first_available = data.iter().zip(avail.iter()).find(|&(_, &a)| a).map(|(&v, _)| v);
        match first_available {
            None => {
                let v = 1i32 << (bit_depth - 1);
                data.iter_mut().for_each(|d| *d = v);
            }
            Some(first) => {
                let mut last = first;
                for k in 0..len {
                    if *avail.get(k).ok_or(HeifError::MalformedHevc("intra border substitution"))? {
                        last = *data.get(k).ok_or(HeifError::MalformedHevc("intra border substitution"))?;
                    } else {
                        *data.get_mut(k).ok_or(HeifError::MalformedHevc("intra border substitution"))? = last;
                    }
                }
            }
        }

        Ok(Border { data: data_buf, nt })
    }

    /// Reference sample filtering (clause 8.4.4.2.3): the 1:2:1 smoothing
    /// filter, or the strong-intra-smoothing bilinear filter for a flat
    /// 32-wide luma block. No-op unless the mode/size combination calls
    /// for it.
    pub fn filter(&mut self, nt: usize, cidx: usize, mode: u8, bit_depth_luma: u32, strong_intra_smoothing: bool) -> Result<()> {
        if mode == 1 || nt == 4 {
            return Ok(());
        }
        let min_dist_ver_hor = (i64::from(mode) - 26).abs().min((i64::from(mode) - 10).abs());
        let filter_flag = match nt {
            8 => min_dist_ver_hor > 7,
            16 => min_dist_ver_hor > 1,
            32 => min_dist_ver_hor > 0,
            _ => false,
        };
        if !filter_flag {
            return Ok(());
        }

        let nt_i = nt as isize;
        let bi_int_flag = strong_intra_smoothing
            && cidx == 0
            && nt == 32
            && (self.get(0)? + self.get(64)? - 2 * self.get(32)?).abs() < (1 << (bit_depth_luma - 5))
            && (self.get(0)? + self.get(-64)? - 2 * self.get(-32)?).abs() < (1 << (bit_depth_luma - 5));

        let mut filtered = [0i32; MAX_BORDER];
        if bi_int_flag {
            let p0 = self.get(0)?;
            let p_pos = self.get(2 * nt_i)?;
            let p_neg = self.get(-2 * nt_i)?;
            for i in 1..=63i32 {
                let lo = p0 + ((i * (p_neg - p0) + 32) >> 6);
                let hi = p0 + ((i * (p_pos - p0) + 32) >> 6);
                let idx_lo = self.idx(-i as isize)?;
                let idx_hi = self.idx(i as isize)?;
                if let Some(s) = filtered.get_mut(idx_lo) {
                    *s = lo;
                }
                if let Some(s) = filtered.get_mut(idx_hi) {
                    *s = hi;
                }
            }
            let idx0 = self.idx(0)?;
            if let Some(s) = filtered.get_mut(idx0) {
                *s = p0;
            }
            let idx_pos = self.idx(2 * nt_i)?;
            if let Some(s) = filtered.get_mut(idx_pos) {
                *s = p_pos;
            }
            let idx_neg = self.idx(-2 * nt_i)?;
            if let Some(s) = filtered.get_mut(idx_neg) {
                *s = p_neg;
            }
        } else {
            for i in -(2 * nt_i - 1)..=(2 * nt_i - 1) {
                let v = self.get(i + 1)? + 2 * self.get(i)? + self.get(i - 1)? + 2;
                let idx = self.idx(i)?;
                if let Some(s) = filtered.get_mut(idx) {
                    *s = v >> 2;
                }
            }
            let idx_pos = self.idx(2 * nt_i)?;
            if let Some(s) = filtered.get_mut(idx_pos) {
                *s = self.get(2 * nt_i)?;
            }
            let idx_neg = self.idx(-2 * nt_i)?;
            if let Some(s) = filtered.get_mut(idx_neg) {
                *s = self.get(-2 * nt_i)?;
            }
        }
        self.data = filtered;
        Ok(())
    }
}

fn log2(n: usize) -> u32 {
    n.trailing_zeros()
}

/// Planar prediction (clause 8.4.4.2.4). `dst` is row-major `nt x nt`.
pub fn predict_planar(dst: &mut [i32], nt: usize, border: &Border) -> Result<()> {
    let log2_nt = log2(nt);
    for y in 0..nt {
        for x in 0..nt {
            let v = (nt - 1 - x) as i64 * border.get(-1 - y as isize)? as i64
                + (x + 1) as i64 * border.get((1 + nt) as isize)? as i64
                + (nt - 1 - y) as i64 * border.get((1 + x) as isize)? as i64
                + (y + 1) as i64 * border.get(-1 - nt as isize)? as i64
                + nt as i64;
            *dst.get_mut(x + y * nt).ok_or(HeifError::MalformedHevc("predict_planar: index"))? = (v >> (log2_nt + 1)) as i32;
        }
    }
    Ok(())
}

/// DC prediction (clause 8.4.4.2.5).
pub fn predict_dc(dst: &mut [i32], nt: usize, cidx: usize, border: &Border) -> Result<()> {
    let log2_nt = log2(nt);
    let mut dc_val: i64 = 0;
    for i in 0..nt as isize {
        dc_val += border.get(i + 1)? as i64;
        dc_val += border.get(-i - 1)? as i64;
    }
    dc_val += nt as i64;
    dc_val >>= log2_nt + 1;

    if cidx == 0 && nt < 32 {
        *dst.get_mut(0).ok_or(HeifError::MalformedHevc("predict_dc: index"))? = ((border.get(-1)? as i64 + 2 * dc_val + border.get(1)? as i64 + 2) >> 2) as i32;
        for x in 1..nt {
            *dst.get_mut(x).ok_or(HeifError::MalformedHevc("predict_dc: index"))? = ((border.get((x + 1) as isize)? as i64 + 3 * dc_val + 2) >> 2) as i32;
        }
        for y in 1..nt {
            *dst.get_mut(y * nt).ok_or(HeifError::MalformedHevc("predict_dc: index"))? = ((border.get(-(y as isize) - 1)? as i64 + 3 * dc_val + 2) >> 2) as i32;
        }
        for y in 1..nt {
            for x in 1..nt {
                *dst.get_mut(x + y * nt).ok_or(HeifError::MalformedHevc("predict_dc: index"))? = dc_val as i32;
            }
        }
    } else {
        for v in dst.iter_mut().take(nt * nt) {
            *v = dc_val as i32;
        }
    }
    Ok(())
}

/// Angular prediction (clause 8.4.4.2.6). `mode` is `2..=34`.
#[allow(clippy::too_many_arguments)]
pub fn predict_angular(dst: &mut [i32], nt: usize, cidx: usize, mode: u8, bit_depth: u32, disable_boundary_filter: bool, border: &Border) -> Result<()> {
    if !(2..=34).contains(&mode) {
        return Err(HeifError::MalformedHevc("predict_angular: mode out of range"));
    }
    let angle = *INTRA_PRED_ANGLE.get(mode as usize).ok_or(HeifError::MalformedHevc("predict_angular: angle table"))?;
    let nt_i = nt as i64;
    let clip = |v: i64| -> i32 { v.clamp(0, (1i64 << bit_depth) - 1) as i32 };

    // `reff` is indexed `-2*nt ..= 2*nt`, flattened with offset `off`; all
    // index arithmetic here is `i64` and only cast to `usize` at the final
    // array access, to avoid mixing integer types across the formulas.
    let off = 2 * nt_i;
    let put_ref = |reff: &mut [i64], i: i64, v: i64| -> Result<()> {
        let idx = usize::try_from(i + off).map_err(|_| HeifError::MalformedHevc("predict_angular: ref index"))?;
        *reff.get_mut(idx).ok_or(HeifError::MalformedHevc("predict_angular: ref index"))? = v;
        Ok(())
    };
    let get_ref = |reff: &[i64], i: i64| -> Result<i64> {
        let idx = usize::try_from(i + off).map_err(|_| HeifError::MalformedHevc("predict_angular: ref read"))?;
        reff.get(idx).copied().ok_or(HeifError::MalformedHevc("predict_angular: ref read"))
    };

    if mode >= 18 {
        let mut reff_buf = [0i64; MAX_BORDER + 1];
        let reff = reff_buf.get_mut(..4 * nt + 2).ok_or(HeifError::MalformedHevc("predict_angular: block larger than 32x32"))?;
        for x in 0..=nt_i {
            put_ref(reff, x, border.get(x as isize)? as i64)?;
        }
        if angle < 0 {
            let inv_angle = *INV_ANGLE.get((mode - 11) as usize).ok_or(HeifError::MalformedHevc("predict_angular: invAngle table"))?;
            let lo = (nt_i * angle as i64) >> 5;
            if lo < -1 {
                for x in lo..=-1 {
                    let src = -((x * inv_angle as i64 + 128) >> 8);
                    put_ref(reff, x, border.get(src as isize)? as i64)?;
                }
            }
        } else {
            for x in (nt_i + 1)..=(2 * nt_i) {
                put_ref(reff, x, border.get(x as isize)? as i64)?;
            }
        }
        for y in 0..nt {
            let i_idx = ((y as i64 + 1) * angle as i64) >> 5;
            let i_fact = ((y as i64 + 1) * angle as i64) & 31;
            for x in 0..nt {
                let v = if i_fact != 0 {
                    ((32 - i_fact) * get_ref(reff, x as i64 + i_idx + 1)? + i_fact * get_ref(reff, x as i64 + i_idx + 2)? + 16) >> 5
                } else {
                    get_ref(reff, x as i64 + i_idx + 1)?
                };
                *dst.get_mut(x + y * nt).ok_or(HeifError::MalformedHevc("predict_angular: dst index"))? = v as i32;
            }
        }
        if mode == 26 && cidx == 0 && nt < 32 && !disable_boundary_filter {
            for y in 0..nt {
                let v = border.get(1)? as i64 + ((border.get(-1 - y as isize)? as i64 - border.get(0)? as i64) >> 1);
                *dst.get_mut(y * nt).ok_or(HeifError::MalformedHevc("predict_angular: boundary filter index"))? = clip(v);
            }
        }
    } else {
        let mut reff_buf = [0i64; MAX_BORDER + 1];
        let reff = reff_buf.get_mut(..4 * nt + 2).ok_or(HeifError::MalformedHevc("predict_angular: block larger than 32x32"))?;
        for x in 0..=nt_i {
            put_ref(reff, x, border.get(-x as isize)? as i64)?;
        }
        if angle < 0 {
            let inv_angle = *INV_ANGLE.get((mode - 11) as usize).ok_or(HeifError::MalformedHevc("predict_angular: invAngle table"))?;
            let lo = (nt_i * angle as i64) >> 5;
            if lo < -1 {
                for x in lo..=-1 {
                    let src = (x * inv_angle as i64 + 128) >> 8;
                    put_ref(reff, x, border.get(src as isize)? as i64)?;
                }
            }
        } else {
            for x in (nt_i + 1)..=(2 * nt_i) {
                put_ref(reff, x, border.get(-x as isize)? as i64)?;
            }
        }
        for x in 0..nt {
            let i_idx = ((x as i64 + 1) * angle as i64) >> 5;
            let i_fact = ((x as i64 + 1) * angle as i64) & 31;
            for y in 0..nt {
                let v = if i_fact != 0 {
                    ((32 - i_fact) * get_ref(reff, y as i64 + i_idx + 1)? + i_fact * get_ref(reff, y as i64 + i_idx + 2)? + 16) >> 5
                } else {
                    get_ref(reff, y as i64 + i_idx + 1)?
                };
                *dst.get_mut(x + y * nt).ok_or(HeifError::MalformedHevc("predict_angular: dst index"))? = v as i32;
            }
        }
        if mode == 10 && cidx == 0 && nt < 32 && !disable_boundary_filter {
            for x in 0..nt {
                let v = border.get(-1)? as i64 + ((border.get((1 + x) as isize)? as i64 - border.get(0)? as i64) >> 1);
                *dst.get_mut(x).ok_or(HeifError::MalformedHevc("predict_angular: boundary filter index"))? = clip(v);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    impl Border {
        fn from_samples(samples: &[i32], nt: usize) -> Self {
            let mut data = [0; MAX_BORDER];
            data[..samples.len()].copy_from_slice(samples);
            Border { data, nt }
        }
    }

    #[test]
    fn planar_filters_reference_samples_but_dc_does_not() {
        let n = 8;
        let mut samples = vec![32; 4 * n + 1];
        samples[2 * n + 1] = 96; // isolated pulse at the first top sample
        let mut planar = Border::from_samples(&samples, n);
        planar.filter(n, 0, 0, 8, false).unwrap();
        // 1:2:1 smoothing: (32 + 2*96 + 32 + 2) >> 2.
        assert_eq!(planar.get(1).unwrap(), 64);
        let mut predicted = vec![0; n * n];
        predict_planar(&mut predicted, n, &planar).unwrap();
        assert_eq!(predicted[0], 46);

        let mut dc = Border::from_samples(&samples, n);
        dc.filter(n, 0, 1, 8, false).unwrap();
        assert_eq!(dc.samples(), samples.as_slice());
    }

    #[test]
    fn dc_prediction_is_flat_when_cidx_nonzero_or_large() {
        // cIdx!=0 path (chroma) skips the edge-blending special case
        // entirely, so with a uniform border every output sample must
        // equal the border value exactly.
        let border = Border::build(8, 8, 4, 8, |_, _| Some(100)).expect("uniform fetch never fails");
        let mut dst = vec![0i32; 16];
        predict_dc(&mut dst, 4, 1, &border).expect("predict_dc should succeed");
        assert!(dst.iter().all(|&v| v == 100));
    }

    #[test]
    fn border_substitution_fills_uniformly_when_nothing_available() {
        let border = Border::build(0, 0, 4, 8, |_, _| None).expect("all-unavailable build should still succeed");
        for i in -8..=8 {
            assert_eq!(border.get(i).unwrap(), 128); // 1<<(8-1)
        }
    }

    #[test]
    fn planar_prediction_is_flat_for_a_flat_border() {
        let border = Border::build(8, 8, 4, 8, |_, _| Some(50)).expect("uniform fetch never fails");
        let mut dst = vec![0i32; 16];
        predict_planar(&mut dst, 4, &border).expect("predict_planar should succeed");
        assert!(dst.iter().all(|&v| v == 50));
    }
}
