//! Deblocking filter (clause 8.7.2), applied to the whole picture right
//! after CTU decode, before SAO (clause 8.7.1's filtering order).
//!
//! Boundary strength is 2 when either side is intra, 1 for coded luma
//! residuals at transform boundaries or differing single-reference motion,
//! and 0 otherwise. Chroma filtering requires strength 2.
//!
//! The `β′`/`t_C′` tables (Table 8-12), the exact luma/chroma filter
//! arithmetic, and the chroma edge grid's spacing (clause 8.7.2.5.5
//! only ever runs at `bS == 2`, and only on an 8-chroma-sample-mapped
//! grid that's coarser than luma's for 4:2:0) were all checked against
//! `imazen/heic`'s deblocking module rather than typed from memory, the
//! same way this crate's other spec-mandated constant tables are.

use super::ctu::{Picture, Plane};
use super::params::Sps;
use super::transform::chroma_qp_mapping;
use crate::error::{HeifError, Result};

/// `β′` (Table 8-12), indexed by `Q = Clip3(0, 51, qPL + (slice_beta_offset_div2 << 1))`.
#[rustfmt::skip]
static BETA_PRIME: [i32; 52] = [
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 24,
    26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56,
    58, 60, 62, 64,
];

/// `t_C′` (Table 8-12), indexed by `Q = Clip3(0, 53, qPL + 2*(bS-1) + (slice_tc_offset_div2 << 1))`.
#[rustfmt::skip]
static TC_PRIME: [i32; 54] = [
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  3,
     3,  3,  3,  4,  4,  4,  5,  5,  6,  6,  7,  8,  9, 10, 11, 13,
    14, 16, 18, 20, 22, 24,
];

fn clip3(lo: i32, hi: i32, v: i32) -> i32 {
    v.clamp(lo, hi)
}

/// Reads the 4 samples on one side of an edge (`p0..p3` for `side ==
/// -1`, `q0..q3` for `side == 1`) at along-edge offset `k`. `None` if
/// any of them falls outside the plane.
fn read_side(plane: &Plane, x: i64, y: i64, vertical: bool, k: i64, side: i64) -> Option<[i32; 4]> {
    let mut out = [0i32; 4];
    for (i, o) in out.iter_mut().enumerate() {
        let d = side * (i as i64 + 1) - if side < 0 { 0 } else { 1 }; // p: -1,-2,-3,-4; q: 0,1,2,3
        let (sx, sy) = if vertical { (x + d, y + k) } else { (x + k, y + d) };
        *o = plane.sample(usize::try_from(sx).ok()?, usize::try_from(sy).ok()?)?;
    }
    Some(out)
}

fn write_side(plane: &mut Plane, x: i64, y: i64, vertical: bool, k: i64, side: i64, vals: &[i32]) {
    for (i, &v) in vals.iter().enumerate() {
        let d = side * (i as i64 + 1) - if side < 0 { 0 } else { 1 };
        let (sx, sy) = if vertical { (x + d, y + k) } else { (x + k, y + d) };
        if let (Ok(sx), Ok(sy)) = (usize::try_from(sx), usize::try_from(sy)) {
            plane.set_sample(sx, sy, v);
        }
    }
}

// Option::or selects the first present value, including Some(false).
// CU and TU flags must instead be combined with boolean OR.
fn is_filter_edge(pic: &Picture, x: i64, y: i64, vertical: bool) -> bool {
    if vertical {
        pic.is_cu_edge_left(x, y).unwrap_or(false) || pic.is_tu_edge_left(x, y).unwrap_or(false)
    } else {
        pic.is_cu_edge_top(x, y).unwrap_or(false) || pic.is_tu_edge_top(x, y).unwrap_or(false)
    }
}

/// Filters one 4-sample luma edge segment (clause 8.7.2.5.3/.5.6/.5.7).
/// `(x, y)` is the Q-side position of the segment's first (`k == 0`) row/column.
fn filter_luma_segment(pic: &mut Picture, x: usize, y: usize, vertical: bool, beta_offset: i32, tc_offset: i32) -> Result<()> {
    let (x, y) = (x as i64, y as i64);
    let (px, py, qx, qy) = if vertical { (x - 1, y, x, y) } else { (x, y - 1, x, y) };
    let is_edge = is_filter_edge(pic, qx, qy, vertical);
    if !is_edge {
        return Ok(()); // bS == 0: no CU or transform-block edge here at all.
    }
    let bs = pic.boundary_strength(qx, qy, vertical);
    if bs == 0 {
        return Ok(());
    }

    let protect_p = pic.bypass_at(px, py);
    let protect_q = pic.bypass_at(qx, qy);
    let qp_p = pic.qp_y_at(px, py).unwrap_or(0);
    let qp_q = pic.qp_y_at(qx, qy).unwrap_or(0);
    let qp_l = (qp_p + qp_q + 1) >> 1;
    let bit_depth = pic.y.bit_depth as i32;

    let q_beta = clip3(0, 51, qp_l + beta_offset) as usize;
    let beta = BETA_PRIME.get(q_beta).ok_or(HeifError::MalformedHevc("deblock: beta table index"))? << (bit_depth - 8);
    let q_tc = clip3(0, 53, qp_l + 2 * (bs - 1) + tc_offset) as usize;
    let tc = TC_PRIME.get(q_tc).ok_or(HeifError::MalformedHevc("deblock: tc table index"))? << (bit_depth - 8);
    if tc == 0 {
        return Ok(());
    }

    let Some(p_0) = read_side(&pic.y, x, y, vertical, 0, -1) else { return Ok(()) };
    let Some(q_0) = read_side(&pic.y, x, y, vertical, 0, 1) else { return Ok(()) };
    let Some(p_3) = read_side(&pic.y, x, y, vertical, 3, -1) else { return Ok(()) };
    let Some(q_3) = read_side(&pic.y, x, y, vertical, 3, 1) else { return Ok(()) };
    let (p0_0, p1_0, p2_0) = (p_0[0], p_0[1], p_0[2]);
    let (q0_0, q1_0, q2_0) = (q_0[0], q_0[1], q_0[2]);
    let (p0_3, p1_3, p2_3) = (p_3[0], p_3[1], p_3[2]);
    let (q0_3, q1_3, q2_3) = (q_3[0], q_3[1], q_3[2]);

    let dp0 = (p2_0 - 2 * p1_0 + p0_0).abs();
    let dp3 = (p2_3 - 2 * p1_3 + p0_3).abs();
    let dq0 = (q2_0 - 2 * q1_0 + q0_0).abs();
    let dq3 = (q2_3 - 2 * q1_3 + q0_3).abs();
    let dpq0 = dp0 + dq0;
    let dpq3 = dp3 + dq3;
    let dp = dp0 + dp3;
    let dq = dq0 + dq3;
    let d = dpq0 + dpq3;
    if d >= beta {
        return Ok(());
    }

    let strong0 = 2 * dpq0 < (beta >> 2) && (p_0[3] - p0_0).abs() + (q0_0 - q_0[3]).abs() < (beta >> 3) && (p0_0 - q0_0).abs() < ((5 * tc + 1) >> 1);
    let strong3 = 2 * dpq3 < (beta >> 2) && (p_3[3] - p0_3).abs() + (q0_3 - q_3[3]).abs() < (beta >> 3) && (p0_3 - q0_3).abs() < ((5 * tc + 1) >> 1);
    let strong = strong0 && strong3;
    let d_ep = dp < ((beta + (beta >> 1)) >> 3);
    let d_eq = dq < ((beta + (beta >> 1)) >> 3);
    let max_val = (1i32 << bit_depth) - 1;

    for k in 0..4i64 {
        let Some(p) = read_side(&pic.y, x, y, vertical, k, -1) else { continue };
        let Some(q) = read_side(&pic.y, x, y, vertical, k, 1) else { continue };
        let (p0, p1, p2, p3) = (p[0], p[1], p[2], p[3]);
        let (q0, q1, q2, q3) = (q[0], q[1], q[2], q[3]);
        if strong {
            let tc2 = 2 * tc;
            let p0_f = clip3(p0 - tc2, p0 + tc2, (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3).clamp(0, max_val);
            let p1_f = clip3(p1 - tc2, p1 + tc2, (p2 + p1 + p0 + q0 + 2) >> 2).clamp(0, max_val);
            let p2_f = clip3(p2 - tc2, p2 + tc2, (2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3).clamp(0, max_val);
            let q0_f = clip3(q0 - tc2, q0 + tc2, (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3).clamp(0, max_val);
            let q1_f = clip3(q1 - tc2, q1 + tc2, (p0 + q0 + q1 + q2 + 2) >> 2).clamp(0, max_val);
            let q2_f = clip3(q2 - tc2, q2 + tc2, (p0 + q0 + q1 + 3 * q2 + 2 * q3 + 4) >> 3).clamp(0, max_val);
            if !protect_p {
                write_side(&mut pic.y, x, y, vertical, k, -1, &[p0_f, p1_f, p2_f]);
            }
            if !protect_q {
                write_side(&mut pic.y, x, y, vertical, k, 1, &[q0_f, q1_f, q2_f]);
            }
        } else {
            let delta = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4;
            if delta.abs() >= 10 * tc {
                continue;
            }
            let delta = clip3(-tc, tc, delta);
            let p0_f = (p0 + delta).clamp(0, max_val);
            let q0_f = (q0 - delta).clamp(0, max_val);
            let p1_f = if d_ep {
                let delta_p = clip3(-(tc >> 1), tc >> 1, (((p2 + p0 + 1) >> 1) - p1 + delta) >> 1);
                Some((p1 + delta_p).clamp(0, max_val))
            } else {
                None
            };
            let q1_f = if d_eq {
                let delta_q = clip3(-(tc >> 1), tc >> 1, (((q2 + q0 + 1) >> 1) - q1 - delta) >> 1);
                Some((q1 + delta_q).clamp(0, max_val))
            } else {
                None
            };
            if !protect_p {
                match p1_f {
                    Some(p1_f) => write_side(&mut pic.y, x, y, vertical, k, -1, &[p0_f, p1_f]),
                    None => write_side(&mut pic.y, x, y, vertical, k, -1, &[p0_f]),
                }
            }
            if !protect_q {
                match q1_f {
                    Some(q1_f) => write_side(&mut pic.y, x, y, vertical, k, 1, &[q0_f, q1_f]),
                    None => write_side(&mut pic.y, x, y, vertical, k, 1, &[q0_f]),
                }
            }
        }
    }
    Ok(())
}

/// Whether this luma position (the chroma edge's collocated luma
/// position) is a CU/transform-block edge and so, intra-only, `bS ==
/// 2` — chroma deblocking (clause 8.7.2.5.5) only ever runs there — and
/// if so, `t_C′` for it. `None` means "don't filter".
#[allow(clippy::too_many_arguments)]
fn chroma_tc(pic: &Picture, lx: i64, ly: i64, vertical: bool, tc_offset: i32, qp_offset: i32, chroma_format_idc: u32, bit_depth: i32) -> Option<i32> {
    let (px, py, qx, qy) = if vertical { (lx - 1, ly, lx, ly) } else { (lx, ly - 1, lx, ly) };
    let is_edge = is_filter_edge(pic, qx, qy, vertical);
    if !is_edge || pic.boundary_strength(qx, qy, vertical) != 2 {
        return None;
    }
    let qp_p = pic.qp_y_at(px, py).unwrap_or(0);
    let qp_q = pic.qp_y_at(qx, qy).unwrap_or(0);
    let qp_i = ((qp_p + qp_q + 1) >> 1) + qp_offset;
    let qp_c = if chroma_format_idc == 1 { chroma_qp_mapping(qp_i) } else { qp_i.min(51) };
    let q_tc = clip3(0, 53, qp_c + 2 + tc_offset) as usize; // bS == 2 => 2*(bS-1) == 2.
    let tc = TC_PRIME.get(q_tc).copied()? << (bit_depth - 8);
    if tc == 0 { None } else { Some(tc) }
}

/// Filters one chroma sample pair across an edge (clause 8.7.2.5.5):
/// one modified sample on each side, no strong/weak distinction.
fn filter_chroma_pixels(plane: &mut Plane, cx: i64, cy: i64, vertical: bool, tc: i32, protect_p: bool, protect_q: bool) {
    let max_val = (1i32 << plane.bit_depth) - 1;
    let (dxp0, dyp0, dxp1, dyp1, dxq0, dyq0, dxq1, dyq1) = if vertical { (-1, 0, -2, 0, 0, 0, 1, 0) } else { (0, -1, 0, -2, 0, 0, 0, 1) };
    let read = |plane: &Plane, dx: i64, dy: i64| -> Option<i32> { plane.sample(usize::try_from(cx + dx).ok()?, usize::try_from(cy + dy).ok()?) };
    let (Some(p0), Some(p1), Some(q0), Some(q1)) = (read(plane, dxp0, dyp0), read(plane, dxp1, dyp1), read(plane, dxq0, dyq0), read(plane, dxq1, dyq1)) else {
        return;
    };

    let delta = clip3(-tc, tc, (((q0 - p0) << 2) + p1 - q1 + 4) >> 3);
    let p0_f = (p0 + delta).clamp(0, max_val);
    let q0_f = (q0 - delta).clamp(0, max_val);
    if !protect_p && let (Ok(x), Ok(y)) = (usize::try_from(cx + dxp0), usize::try_from(cy + dyp0)) {
        plane.set_sample(x, y, p0_f);
    }
    if !protect_q && let (Ok(x), Ok(y)) = (usize::try_from(cx + dxq0), usize::try_from(cy + dyq0)) {
        plane.set_sample(x, y, q0_f);
    }
}

/// Applies the deblocking filter to the whole picture in-place: every
/// vertical edge first, then every horizontal edge (clause 8.7.2.1),
/// luma then chroma. `beta_offset_div2`/`tc_offset_div2` are the
/// slice's own (or, absent an override, the PPS's) values; `cb_qp_offset`/
/// `cr_qp_offset` are `pps_cb_qp_offset + slice_cb_qp_offset` and its Cr
/// counterpart (clause 8.6.1's `qPi` input).
pub fn deblock_picture(pic: &mut Picture, sps: &Sps, beta_offset_div2: i32, tc_offset_div2: i32, cb_qp_offset: i32, cr_qp_offset: i32) -> Result<()> {
    let (width, height) = (pic.width(), pic.height());
    let (beta_offset, tc_offset) = (beta_offset_div2 * 2, tc_offset_div2 * 2);

    let mut x = 8usize;
    while x < width {
        let mut y = 0usize;
        while y < height {
            filter_luma_segment(pic, x, y, true, beta_offset, tc_offset)?;
            y += 4;
        }
        x += 8;
    }
    let mut y = 8usize;
    while y < height {
        let mut x = 0usize;
        while x < width {
            filter_luma_segment(pic, x, y, false, beta_offset, tc_offset)?;
            x += 4;
        }
        y += 8;
    }

    if sps.chroma_format_idc == 0 {
        return Ok(());
    }
    let bit_depth_chroma = sps.bit_depth_chroma as i32;
    let (sub_w, sub_h) = (sps.sub_width_c() as usize, sps.sub_height_c() as usize);
    // Chroma's own deblocking grid (clause 8.7.2.5.5 is only ever reached
    // at a transform/CU edge that's also on an 8-chroma-sample grid):
    // expressed here in luma coordinates, matching how the edge/QP maps
    // (and `chroma_tc`'s lookups into them) are indexed.
    let (x_step_vert, y_step_vert) = (8 * sub_w, 4 * sub_h);
    let (x_step_horiz, y_step_horiz) = (4 * sub_w, 8 * sub_h);

    // Each segment's `tc` (computed once, at its first luma-space row/
    // column, exactly like luma's 4-sample segments) applies to every
    // chroma sample along the segment, not just that first one — a
    // segment is `y_step_vert`/`x_step_horiz` *luma* rows/columns tall,
    // i.e. that many divided by `sub_h`/`sub_w` *chroma* ones.
    let seg_h_chroma = (y_step_vert / sub_h).max(1);
    let seg_w_chroma = (x_step_horiz / sub_w).max(1);

    for (cb, qp_offset) in [(true, cb_qp_offset), (false, cr_qp_offset)] {
        let mut lx = x_step_vert;
        while lx < width {
            let mut ly = 0usize;
            while ly < height {
                if let Some(tc) = chroma_tc(pic, lx as i64, ly as i64, true, tc_offset, qp_offset, sps.chroma_format_idc, bit_depth_chroma) {
                    let protect_p = pic.bypass_at(lx as i64 - 1, ly as i64);
                    let protect_q = pic.bypass_at(lx as i64, ly as i64);
                    let plane = if cb { &mut pic.cb } else { &mut pic.cr };
                    let cx = (lx / sub_w) as i64;
                    let cy0 = (ly / sub_h) as i64;
                    for dy in 0..seg_h_chroma as i64 {
                        filter_chroma_pixels(plane, cx, cy0 + dy, true, tc, protect_p, protect_q);
                    }
                }
                ly += y_step_vert;
            }
            lx += x_step_vert;
        }
        let mut ly = y_step_horiz;
        while ly < height {
            let mut lx = 0usize;
            while lx < width {
                if let Some(tc) = chroma_tc(pic, lx as i64, ly as i64, false, tc_offset, qp_offset, sps.chroma_format_idc, bit_depth_chroma) {
                    let protect_p = pic.bypass_at(lx as i64, ly as i64 - 1);
                    let protect_q = pic.bypass_at(lx as i64, ly as i64);
                    let plane = if cb { &mut pic.cb } else { &mut pic.cr };
                    let cx0 = (lx / sub_w) as i64;
                    let cy = (ly / sub_h) as i64;
                    for dx in 0..seg_w_chroma as i64 {
                        filter_chroma_pixels(plane, cx0 + dx, cy, false, tc, protect_p, protect_q);
                    }
                }
                lx += x_step_horiz;
            }
            ly += y_step_horiz;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hevc::ctu::Picture;

    fn test_picture(width: usize, height: usize) -> Picture {
        Picture::new(width, height, 2, 2, 8, 8, 3, 4)
    }

    #[test]
    fn bypass_preserves_either_side_without_disabling_the_other() {
        for protect_p in [false, true] {
            let mut pic = test_picture(8, 4);
            for y in 0..4 {
                for x in 0..8 {
                    pic.y.set_sample(x, y, if x < 4 { 100 } else { 110 });
                }
            }
            pic.set_qp_y(0, 0, 8, 37);
            pic.mark_cu_edges(4, 0, 4);
            pic.mark_bypass(if protect_p { 0 } else { 4 }, 0, 4, true);
            filter_luma_segment(&mut pic, 4, 0, true, 0, 0).unwrap();
            assert_eq!(pic.y.sample(3, 0), Some(if protect_p { 100 } else { 104 }));
            assert_eq!(pic.y.sample(4, 0), Some(if protect_p { 106 } else { 110 }));
            let mut other = test_picture(8, 2);
            let plane = &mut other.cb;
            for x in 0..4 {
                plane.set_sample(x, 0, if x < 2 { 100 } else { 110 });
            }
            filter_chroma_pixels(plane, 2, 0, true, 5, protect_p, !protect_p);
            assert_eq!(plane.sample(1, 0), Some(if protect_p { 100 } else { 104 }));
            assert_eq!(plane.sample(2, 0), Some(if protect_p { 106 } else { 110 }));
        }
    }

    /// Flat regions of 100 (p-side) and 110 (q-side) either side of a
    /// marked CU edge at x=4, QP 37 on both sides: hand-computed
    /// against clause 8.7.2.5.3/.5.7 (`beta' = 36`, `tc' = 5` at Q
    /// 37/39 — `BETA_PRIME[37]`/`TC_PRIME[39]`).
    #[test]
    fn strong_filter_matches_hand_computed_values() {
        let mut pic = test_picture(8, 4);
        for y in 0..4 {
            for x in 0..4 {
                pic.y.set_sample(x, y, 100);
            }
            for x in 4..8 {
                pic.y.set_sample(x, y, 110);
            }
        }
        pic.set_qp_y(0, 0, 8, 37);
        pic.mark_cu_edges(4, 0, 4);
        filter_luma_segment(&mut pic, 4, 0, true, 0, 0).expect("filter must not error on a fully in-bounds edge");
        for y in 0..4 {
            assert_eq!(pic.y.sample(0, y), Some(100)); // p3: untouched by the filter
            assert_eq!(pic.y.sample(1, y), Some(101)); // p2'
            assert_eq!(pic.y.sample(2, y), Some(103)); // p1'
            assert_eq!(pic.y.sample(3, y), Some(104)); // p0'
            assert_eq!(pic.y.sample(4, y), Some(106)); // q0'
            assert_eq!(pic.y.sample(5, y), Some(108)); // q1'
            assert_eq!(pic.y.sample(6, y), Some(109)); // q2'
            assert_eq!(pic.y.sample(7, y), Some(110)); // q3: untouched
        }
    }

    #[test]
    fn transform_edges_inside_a_cu_are_filtered() {
        for vertical in [true, false] {
            let mut pic = test_picture(16, 16);
            pic.set_qp_y(0, 0, 16, 37);
            pic.mark_cu_edges(0, 0, 16);
            pic.mark_tu_edges(8, 8, 8);
            for y in 0..16 {
                for x in 0..16 {
                    let value = if (if vertical { x } else { y }) < 8 { 100 } else { 110 };
                    pic.y.set_sample(x, y, value);
                }
            }
            assert_eq!(pic.is_cu_edge_left(8, 8), Some(false));
            assert_eq!(pic.is_cu_edge_top(8, 8), Some(false));
            filter_luma_segment(&mut pic, 8, 8, vertical, 0, 0).unwrap();
            let (px, py) = if vertical { (7, 8) } else { (8, 7) };
            assert_eq!(pic.y.sample(px, py), Some(104));
            assert_eq!(pic.y.sample(8, 8), Some(106));
            assert_eq!(chroma_tc(&pic, 8, 8, vertical, 0, 0, 1, 8), Some(4));
        }
    }

    #[test]
    fn no_marked_edge_means_no_filtering() {
        let mut pic = test_picture(8, 4);
        for y in 0..4 {
            for x in 0..4 {
                pic.y.set_sample(x, y, 100);
            }
            for x in 4..8 {
                pic.y.set_sample(x, y, 110);
            }
        }
        pic.set_qp_y(0, 0, 8, 37);
        // No `mark_cu_edges`/`mark_tu_edges` at x=4: bS stays 0 even
        // though the sample jump would otherwise pass the beta test.
        filter_luma_segment(&mut pic, 4, 0, true, 0, 0).expect("filter must not error");
        for y in 0..4 {
            assert_eq!(pic.y.sample(3, y), Some(100));
            assert_eq!(pic.y.sample(4, y), Some(110));
        }
    }

    /// Hand-computed against clause 8.7.2.5.5: `p1=90, p0=100, q0=110,
    /// q1=120`, `tc=5` → `delta = Clip3(-5,5, ((10<<2)+90-120+4)>>3) =
    /// Clip3(-5,5, 1) = 1`.
    #[test]
    fn chroma_filter_matches_hand_computed_values() {
        let mut pic = test_picture(8, 4);
        pic.cb.set_sample(0, 0, 90);
        pic.cb.set_sample(1, 0, 100);
        pic.cb.set_sample(2, 0, 110);
        pic.cb.set_sample(3, 0, 120);
        filter_chroma_pixels(&mut pic.cb, 2, 0, true, 5, false, false);
        assert_eq!(pic.cb.sample(1, 0), Some(101));
        assert_eq!(pic.cb.sample(2, 0), Some(109));
        // Untouched: chroma deblocking only ever edits p0/q0.
        assert_eq!(pic.cb.sample(0, 0), Some(90));
        assert_eq!(pic.cb.sample(3, 0), Some(120));
    }

    /// Regression test for a real bug (found by diffing against HM, the
    /// JCT-VC reference decoder, on real photos): `deblock_picture`'s
    /// chroma loop computed `tc` once per 4-chroma-row/column segment
    /// (correctly, matching luma's own 4-sample segments) but then
    /// only ever filtered that segment's *first* row/column, leaving
    /// the other 3 chroma samples along the same edge completely
    /// unfiltered. A CU edge spanning 4 chroma rows must filter all 4,
    /// not just the one at the segment's starting `ly`.
    #[test]
    fn chroma_deblocking_filters_every_row_in_a_segment_not_just_the_first() {
        let sps = Sps {
            log2_max_pic_order_cnt_lsb: 8,
            short_term_refs: Vec::new(),
            temporal_mvp_enabled: false,
            chroma_format_idc: 1,
            pic_width_in_luma_samples: 24,
            pic_height_in_luma_samples: 8,
            conformance_window: [0; 4],
            color_info: None,
            chroma_location: None,
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
        };
        let mut pic = test_picture(24, 8);

        // Flat p/q regions either side of a chroma edge at luma x=16
        // (-> chroma x=8), QP 37 throughout (same as the hand-computed
        // single-pixel test above: tc'=4, delta=1, p0'=101, q0'=109).
        for cy in 0..4 {
            pic.cb.set_sample(6, cy, 90); // p1
            pic.cb.set_sample(7, cy, 100); // p0
            pic.cb.set_sample(8, cy, 110); // q0
            pic.cb.set_sample(9, cy, 120); // q1
        }
        pic.set_qp_y(0, 0, 24, 37);
        pic.mark_cu_edges(16, 0, 8); // one CU boundary spanning the whole 8-luma-row (4-chroma-row) height.

        deblock_picture(&mut pic, &sps, 0, 0, 0, 0).expect("deblocking a well-formed picture must not error");

        for cy in 0..4 {
            assert_eq!(pic.cb.sample(7, cy), Some(101), "p0 at chroma row {cy} should be filtered, not just row 0's");
            assert_eq!(pic.cb.sample(8, cy), Some(109), "q0 at chroma row {cy} should be filtered, not just row 0's");
        }
    }
}
