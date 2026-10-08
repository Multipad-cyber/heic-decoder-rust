//! Sample Adaptive Offset filter (clause 8.7.3), applied to the whole
//! picture right after deblocking, per-component (luma/Cb/Cr) and
//! per-CTB using the parameters `Decoder::decode_sao` parsed.
//!
//! Band offset only ever depends on a sample's own value, so applying
//! it can read and write the live plane directly. Edge offset depends
//! on unmodified *neighbour* samples — including ones in another CTB
//! that might itself get a band-offset edit earlier in this same pass
//! — so every component's SAO pass reads from one snapshot of that
//! whole plane taken before any edit in the pass, and only ever writes
//! to the live plane. This mirrors `imazen/heic`'s own SAO module,
//! which keeps the same `src`/`dst` split for edge offset (and applies
//! band offset in place) for the same reason.

use super::ctu::{Picture, SaoCtbParams};
use super::params::Sps;
use crate::error::Result;

/// `(dx0, dy0, dx1, dy1)` of the two neighbours compared for each of
/// the 4 `sao_eo_class` values (clause 8.7.3.2's class-to-direction
/// mapping).
const EO_OFFSETS: [(i32, i32, i32, i32); 4] = [
    (-1, 0, 1, 0),  // 0: horizontal
    (0, -1, 0, 1),  // 1: vertical
    (-1, -1, 1, 1), // 2: 135° diagonal
    (1, -1, -1, 1), // 3: 45° diagonal
];

#[allow(clippy::too_many_arguments)]
fn apply_band(snapshot: &[i32], dst: &mut [i32], width: usize, x0: usize, y0: usize, w: usize, h: usize, band_position: u8, offsets: [i32; 4], bit_depth: u32) {
    let max_val = (1i32 << bit_depth) - 1;
    let band_shift = bit_depth.saturating_sub(5);
    let mut band_table = [0i32; 32];
    for (k, &o) in offsets.iter().enumerate() {
        if let Some(slot) = band_table.get_mut((band_position as usize + k) & 31) {
            *slot = o;
        }
    }
    for y in y0..y0 + h {
        for x in x0..x0 + w {
            let idx = y * width + x;
            let Some(&sample) = snapshot.get(idx) else { continue };
            let band = (sample.clamp(0, max_val) >> band_shift) as usize;
            let offset = band_table.get(band).copied().unwrap_or(0);
            if offset != 0 && let Some(d) = dst.get_mut(idx) {
                *d = (sample + offset).clamp(0, max_val);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_edge(snapshot: &[i32], dst: &mut [i32], width: usize, height: usize, x0: usize, y0: usize, w: usize, h: usize, eo_class: u8, offsets: [i32; 4], bit_depth: u32) {
    let max_val = (1i32 << bit_depth) - 1;
    let (dx0, dy0, dx1, dy1) = EO_OFFSETS[eo_class as usize & 3];
    // Categories 1,2 (table index 0,1) add a positive offset; 3,4
    // (index 2,3) a negative one; "equal to both neighbours" (edge_idx
    // == 2, i.e. sample on a local plateau) is never offset at all.
    let table: [i32; 5] = [offsets[0], offsets[1], 0, -offsets[2], -offsets[3]];

    for y in y0..y0 + h {
        let (ny0, ny1) = (y as i64 + dy0 as i64, y as i64 + dy1 as i64);
        if ny0 < 0 || ny0 as usize >= height || ny1 < 0 || ny1 as usize >= height {
            continue;
        }
        for x in x0..x0 + w {
            let (nx0, nx1) = (x as i64 + dx0 as i64, x as i64 + dx1 as i64);
            if nx0 < 0 || nx0 as usize >= width || nx1 < 0 || nx1 as usize >= width {
                continue;
            }
            let idx = y * width + x;
            let Some(&sample) = snapshot.get(idx) else { continue };
            let Some(&n0) = snapshot.get(ny0 as usize * width + nx0 as usize) else { continue };
            let Some(&n1) = snapshot.get(ny1 as usize * width + nx1 as usize) else { continue };
            let edge_idx = (2 + (sample - n0).signum() + (sample - n1).signum()) as usize;
            let offset = table.get(edge_idx).copied().unwrap_or(0);
            if offset != 0 && let Some(d) = dst.get_mut(idx) {
                *d = (sample + offset).clamp(0, max_val);
            }
        }
    }
}

/// Runs one component's whole-picture SAO pass: snapshots the plane
/// once, then applies every CTB's (already-parsed) band/edge offset
/// over its own area (clipped to the plane's true bounds — the last
/// row/column of CTBs is typically only partially used). Takes the SAO
/// parameter grid as a plain slice (rather than `&Picture`) so this can
/// run concurrently with a mutable borrow of the one plane it's
/// actually editing.
#[allow(clippy::too_many_arguments)]
fn apply_component(sao: &[SaoCtbParams], pic_width_ctbs: usize, pic_height_ctbs: usize, cidx: usize, width: usize, height: usize, ctb_width: usize, ctb_height: usize, bit_depth: u32, snapshot: &[i32], dst: &mut [i32]) {
    for ctb_y in 0..pic_height_ctbs {
        for ctb_x in 0..pic_width_ctbs {
            let Some(&params) = sao.get(ctb_y * pic_width_ctbs + ctb_x) else { continue };
            let Some(&type_idx) = params.type_idx.get(cidx) else { continue };
            if type_idx == 0 {
                continue;
            }
            let (x0, y0) = (ctb_x * ctb_width, ctb_y * ctb_height);
            if x0 >= width || y0 >= height {
                continue;
            }
            let w = ctb_width.min(width - x0);
            let h = ctb_height.min(height - y0);
            let extra = params.extra.get(cidx).copied().unwrap_or(0);
            let offsets = params.offsets.get(cidx).copied().unwrap_or([0; 4]);
            if type_idx == 1 {
                apply_band(snapshot, dst, width, x0, y0, w, h, extra, offsets, bit_depth);
            } else {
                apply_edge(snapshot, dst, width, height, x0, y0, w, h, extra, offsets, bit_depth);
            }
        }
    }
}

/// Applies the SAO filter to the whole picture in-place.
pub fn apply_sao(pic: &mut Picture, sps: &Sps, sao_luma: bool, sao_chroma: bool) -> Result<()> {
    let ctb_pixel = 1usize << pic.ctb_log2;
    let (pic_width_ctbs, pic_height_ctbs) = (pic.pic_width_ctbs(), pic.pic_height_ctbs());
    let sao = pic.sao_params().to_vec();
    let restore_bypass = pic.has_bypass();

    if sao_luma {
        let (width, height) = (pic.y.width, pic.y.height);
        let snapshot = pic.y.samples().to_vec();
        apply_component(&sao, pic_width_ctbs, pic_height_ctbs, 0, width, height, ctb_pixel, ctb_pixel, pic.y.bit_depth, &snapshot, pic.y.samples_mut());
        if restore_bypass { pic.restore_bypass(0, &snapshot, 1, 1); }
    }
    if sao_chroma && sps.chroma_format_idc != 0 {
        let chroma_ctb_width = ctb_pixel / sps.sub_width_c() as usize;
        let chroma_ctb_height = ctb_pixel / sps.sub_height_c() as usize;

        let (cb_width, cb_height) = (pic.cb.width, pic.cb.height);
        let snapshot_cb = pic.cb.samples().to_vec();
        apply_component(&sao, pic_width_ctbs, pic_height_ctbs, 1, cb_width, cb_height, chroma_ctb_width, chroma_ctb_height, pic.cb.bit_depth, &snapshot_cb, pic.cb.samples_mut());
        if restore_bypass { pic.restore_bypass(1, &snapshot_cb, sps.sub_width_c() as usize, sps.sub_height_c() as usize); }

        let (cr_width, cr_height) = (pic.cr.width, pic.cr.height);
        let snapshot_cr = pic.cr.samples().to_vec();
        apply_component(&sao, pic_width_ctbs, pic_height_ctbs, 2, cr_width, cr_height, chroma_ctb_width, chroma_ctb_height, pic.cr.bit_depth, &snapshot_cr, pic.cr.samples_mut());
        if restore_bypass { pic.restore_bypass(2, &snapshot_cr, sps.sub_width_c() as usize, sps.sub_height_c() as usize); }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bypass_samples_keep_their_values_while_neighbours_receive_sao() {
        let mut pic = Picture::new(8, 4, 2, 2, 8, 8, 3, 4);
        pic.y.samples_mut().fill(10);
        pic.mark_bypass(0, 0, 4, true);
        let snapshot = pic.y.samples().to_vec();
        let mut params = SaoCtbParams::default();
        params.type_idx[0] = 1;
        params.extra[0] = 1;
        params.offsets[0] = [3, 0, 0, 0];
        apply_component(&[params], 1, 1, 0, 8, 4, 16, 16, 8, &snapshot, pic.y.samples_mut());
        pic.restore_bypass(0, &snapshot, 1, 1);
        for y in 0..4 { for x in 0..8 { assert_eq!(pic.y.sample(x, y), Some(if x < 4 { 10 } else { 13 })); } }
    }

    /// 8-bit band offset: `band = sample >> 3`, so samples 10/40/70/100
    /// land in bands 1/5/8/12. With `band_position = 1`, only bands
    /// 1..=4 (offsets[0..4]) are non-zero.
    #[test]
    fn band_offset_applies_to_matching_band_only() {
        let snapshot = [10, 40, 70, 100];
        let mut dst = snapshot;
        apply_band(&snapshot, &mut dst, 4, 0, 0, 4, 1, 1, [5, -3, 7, -9], 8);
        assert_eq!(dst, [15, 40, 70, 100]);
    }

    /// Band 0 never maps to an offset slot a 4-offset `sao_band_position`
    /// can reach when `band_position` keeps it out of range, so an
    /// out-of-band sample is left untouched even with all-nonzero
    /// offsets available elsewhere.
    #[test]
    fn band_offset_leaves_out_of_range_bands_untouched() {
        let snapshot = [0i32]; // band 0
        let mut dst = snapshot;
        apply_band(&snapshot, &mut dst, 1, 0, 0, 1, 1, 10, [1, 2, 3, 4], 8); // bands 10..=13 get the offsets
        assert_eq!(dst, [0]);
    }

    /// Clause 8.7.3.2: `edge_idx = 2 + sign(sample-n0) + sign(sample-n1)`.
    /// Sample 50 flanked by 60 and 70 (both higher) is a local minimum
    /// (`edge_idx == 0`), which the table maps to the *positive*
    /// `offsets[0]`. The two boundary samples have an out-of-picture
    /// neighbour and are left untouched.
    #[test]
    fn edge_offset_local_minimum_gets_first_category() {
        let snapshot = [60, 50, 70];
        let mut dst = snapshot;
        apply_edge(&snapshot, &mut dst, 3, 1, 0, 0, 3, 1, 0, [3, 99, 99, 7], 8);
        assert_eq!(dst, [60, 53, 70]);
    }

    /// Sample 70 flanked by 60 and 50 (both lower) is a local maximum
    /// (`edge_idx == 4`), which the table maps to the *negative*
    /// `-offsets[3]`.
    #[test]
    fn edge_offset_local_maximum_gets_last_category() {
        let snapshot = [60, 70, 50];
        let mut dst = snapshot;
        apply_edge(&snapshot, &mut dst, 3, 1, 0, 0, 3, 1, 0, [99, 99, 99, 4], 8);
        assert_eq!(dst, [60, 66, 50]);
    }
}
