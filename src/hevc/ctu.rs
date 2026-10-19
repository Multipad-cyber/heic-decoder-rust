//! The CTU quadtree decode driver: `coding_quadtree` / `coding_unit` /
//! `transform_tree` / `transform_unit` (clause 7.3.8.4–7.3.8.10), tying
//! together CABAC, intra prediction, dequant/transform and residual
//! coding into reconstructed pixels.
//!
//! Intra `PART_2Nx2N`/`PART_NxN` and basic CU transquant bypass are
//! supported; PCM is rejected. The `split_cu_flag` boundary-forced-
//! split rule and the chroma-follows-parent-CU rule at 4x4 luma leaves
//! were checked against `libde265`'s `slice.cc` (LGPL-3.0). See the
//! crate README.

use super::cabac::Engine;
use super::contexts::Contexts;
use super::inter::{Motion, compensate};
use super::intra::{Border, predict_angular, predict_dc, predict_planar};
use super::params::{Pps, Sps};
use super::residual::decode_residual;
use super::scaling::ScalingListData;
use super::slice::InterSlice;
use super::transform::{Dequant, chroma_qp_mapping, inverse_transform, transform_skip_residual};
use crate::bitreader::BitReader;
use crate::error::{HeifError, Result};
use crate::timing::{self, Stage};

pub struct Plane {
    pub width: usize,
    pub height: usize,
    pub bit_depth: u32,
    samples: Vec<i32>,
    decoded: Vec<bool>,
}

impl Plane {
    fn new(width: usize, height: usize, bit_depth: u32) -> Self {
        Plane { width, height, bit_depth, samples: vec![0; width * height], decoded: vec![false; width * height] }
    }

    fn get(&self, x: i64, y: i64) -> Option<i32> {
        if x < 0 || y < 0 || x as usize >= self.width || y as usize >= self.height {
            return None;
        }
        let idx = y as usize * self.width + x as usize;
        if *self.decoded.get(idx)? { self.samples.get(idx).copied() } else { None }
    }

    /// Post-reconstruction passes (deblocking, SAO) only ever run after
    /// every sample in the picture has already been written once by the
    /// CTU decode driver, so the `decoded` gate `get`/`set` enforce
    /// during that decode isn't needed here — every in-bounds position
    /// is guaranteed decoded by construction, not by this accessor.
    pub(crate) fn sample(&self, x: usize, y: usize) -> Option<i32> {
        if x >= self.width || y >= self.height {
            return None;
        }
        self.samples.get(y * self.width + x).copied()
    }

    pub(crate) fn set_sample(&mut self, x: usize, y: usize, v: i32) -> Option<()> {
        if x >= self.width || y >= self.height {
            return None;
        }
        *self.samples.get_mut(y * self.width + x)? = v;
        Some(())
    }

    pub(crate) fn samples(&self) -> &[i32] {
        &self.samples
    }

    pub(crate) fn samples_mut(&mut self) -> &mut [i32] {
        &mut self.samples
    }

    pub(crate) fn set(&mut self, x: usize, y: usize, v: i32) -> Result<()> {
        if x >= self.width || y >= self.height {
            return Err(HeifError::MalformedHevc("plane write out of bounds"));
        }
        let idx = y * self.width + x;
        *self.samples.get_mut(idx).ok_or(HeifError::MalformedHevc("plane write out of bounds"))? = v;
        *self.decoded.get_mut(idx).ok_or(HeifError::MalformedHevc("plane write out of bounds"))? = true;
        Ok(())
    }

    pub fn into_samples(self) -> Vec<i32> {
        self.samples
    }
}

/// One CTB's worth of parsed SAO parameters (clause 7.3.8.3), per
/// colour component (`[0]`=luma, `[1]`=Cb, `[2]`=Cr). `type_idx`: 0=off,
/// 1=band offset, 2=edge offset. `extra`: `band_position` (type 1) or
/// `eo_class` (type 2) — unused (0) when `type_idx == 0`.
#[derive(Clone, Copy, Default)]
pub(crate) struct SaoCtbParams {
    pub type_idx: [u8; 3],
    pub extra: [u8; 3],
    pub offsets: [[i32; 4]; 3],
}

pub struct Picture {
    motions: Vec<Option<Motion>>,
    skips: Vec<bool>,
    luma_residual: Vec<bool>,
    bypass: Vec<bool>,
    pub y: Plane,
    pub cb: Plane,
    pub cr: Plane,
    /// Luma intra mode, one entry per 4x4 min-PU, raster order over the
    /// *picture*'s min-PU grid (used only for MPM derivation).
    intra_mode: Vec<u8>,
    min_pu_width: usize,
    min_pu_height: usize,
    /// Coding-tree depth at which each min-CB-granularity position's CU
    /// was coded (used only for `split_cu_flag`'s neighbour-depth rule).
    ct_depth: Vec<u8>,
    min_cb_log2: u32,
    min_cb_width: usize,
    /// Deblocking bookkeeping, all at the same 4x4 min-PU granularity as
    /// `intra_mode`: the reconstructed `QpY` each position's CU used,
    /// whether that position is a CU's/luma TU's own left/top edge, and
    /// Populated by `Decoder::coding_unit`/`transform_unit`, read only by
    /// `hevc::deblock` (clause 8.7.2.4's boundary-strength derivation,
    /// which is unconditionally `bS == 2` at any CU *or* transform-block
    /// edge for this decoder's intra-only scope: that clause's "either
    /// side is intra" branch always fires first, so the `cbf`-dependent
    /// `bS == 1` branch — relevant only to inter blocks — is never
    /// reached and isn't tracked here).
    qp_y: Vec<i8>,
    cu_edge_left: Vec<bool>,
    cu_edge_top: Vec<bool>,
    tu_edge_left: Vec<bool>,
    tu_edge_top: Vec<bool>,
    /// Parsed per-CTB SAO parameters, populated by `Decoder::decode_sao`
    /// and read only by `hevc::sao`.
    sao: Vec<SaoCtbParams>,
    pub(crate) ctb_log2: u32,
    pic_width_ctbs: usize,
    pic_height_ctbs: usize,
    slice_start_y: usize,
}

impl Picture {
    #[allow(clippy::too_many_arguments)]
    pub fn new(width: usize, height: usize, sub_w: usize, sub_h: usize, bit_depth_luma: u32, bit_depth_chroma: u32, min_cb_log2: u32, ctb_log2: u32) -> Self {
        let cw = if sub_w == 0 { 0 } else { width.div_ceil(sub_w) };
        let ch = if sub_h == 0 { 0 } else { height.div_ceil(sub_h) };
        let min_pu_width = width.div_ceil(4);
        let min_pu_height = height.div_ceil(4);
        let min_cb_width = width.div_ceil(1 << min_cb_log2);
        let min_cb_height = height.div_ceil(1 << min_cb_log2);
        let pic_width_ctbs = width.div_ceil(1 << ctb_log2);
        let pic_height_ctbs = height.div_ceil(1 << ctb_log2);
        let min_pu_count = min_pu_width * min_pu_height;
        Picture {
            slice_start_y: 0,
            motions: vec![None; min_pu_count],
            skips: vec![false; min_pu_count],
            luma_residual: vec![false; min_pu_count],
            bypass: vec![false; min_pu_count],
            y: Plane::new(width, height, bit_depth_luma),
            cb: Plane::new(cw, ch, bit_depth_chroma),
            cr: Plane::new(cw, ch, bit_depth_chroma),
            intra_mode: vec![1; min_pu_count], // INTRA_DC default
            min_pu_width,
            min_pu_height,
            ct_depth: vec![0; min_cb_width * min_cb_height],
            min_cb_log2,
            min_cb_width,
            qp_y: vec![0; min_pu_count],
            cu_edge_left: vec![false; min_pu_count],
            cu_edge_top: vec![false; min_pu_count],
            tu_edge_left: vec![false; min_pu_count],
            tu_edge_top: vec![false; min_pu_count],
            sao: vec![SaoCtbParams::default(); pic_width_ctbs * pic_height_ctbs],
            ctb_log2,
            pic_width_ctbs,
            pic_height_ctbs,
        }
    }

    pub(crate) fn bypass_at(&self, x: i64, y: i64) -> bool {
        self.min_pu_index(x, y).and_then(|i| self.bypass.get(i)).copied().unwrap_or(false)
    }
    pub(crate) fn mark_bypass(&mut self, x: usize, y: usize, size: usize, value: bool) {
        for row in (y..(y + size).min(self.y.height)).step_by(4) {
            for col in (x..(x + size).min(self.y.width)).step_by(4) {
                if let Some(i) = self.min_pu_index(col as i64, row as i64) {
                    self.bypass[i] = value;
                }
            }
        }
    }
    pub(crate) fn has_bypass(&self) -> bool {
        self.bypass.iter().any(|&b| b)
    }
    /// SAO reads the original raster; restore protected samples after its pass.
    pub(crate) fn restore_bypass(&mut self, cidx: usize, snapshot: &[i32], sub_w: usize, sub_h: usize) {
        let width = match cidx {
            0 => self.y.width,
            1 => self.cb.width,
            _ => self.cr.width,
        };
        for (i, &sample) in snapshot.iter().enumerate() {
            if self.bypass_at(((i % width) * sub_w) as i64, ((i / width) * sub_h) as i64) {
                let plane = match cidx {
                    0 => &mut self.y,
                    1 => &mut self.cb,
                    _ => &mut self.cr,
                };
                if let Some(out) = plane.samples_mut().get_mut(i) {
                    *out = sample;
                }
            }
        }
    }

    fn motion_at(&self, x: i64, y: i64) -> Option<Motion> {
        self.min_pu_index(x, y).and_then(|i| self.motions.get(i).copied().flatten())
    }
    fn skip_at(&self, x: i64, y: i64) -> bool {
        self.min_pu_index(x, y).and_then(|i| self.skips.get(i)).copied().unwrap_or(false)
    }
    fn mark_motion(&mut self, x: usize, y: usize, w: usize, h: usize, mv: Motion, skip: bool) {
        for row in (y..y + h).step_by(4) {
            for col in (x..x + w).step_by(4) {
                if let Some(i) = self.min_pu_index(col as i64, row as i64) {
                    self.motions[i] = Some(mv);
                    self.skips[i] = skip;
                    if col == x {
                        self.cu_edge_left[i] = true;
                    }
                    if row == y {
                        self.cu_edge_top[i] = true;
                    }
                }
            }
        }
    }
    fn mark_residual(&mut self, x: usize, y: usize, size: usize, coded: bool) {
        for row in (y..y + size).step_by(4) {
            for col in (x..x + size).step_by(4) {
                if let Some(i) = self.min_pu_index(col as i64, row as i64) {
                    self.luma_residual[i] = coded;
                }
            }
        }
    }
    pub(crate) fn boundary_strength(&self, x: i64, y: i64, vertical: bool) -> i32 {
        let (px, py) = if vertical { (x - 1, y) } else { (x, y - 1) };
        let (Some(a), Some(b)) = (self.motion_at(px, py), self.motion_at(x, y)) else {
            return 2;
        };
        let tu = if vertical { self.is_tu_edge_left(x, y) } else { self.is_tu_edge_top(x, y) }.unwrap_or(false);
        let coded = |xx, yy| self.min_pu_index(xx, yy).map(|i| self.luma_residual[i]).unwrap_or(false);
        if tu && (coded(px, py) || coded(x, y)) || (a.x - b.x).abs() >= 4 || (a.y - b.y).abs() >= 4 { 1 } else { 0 }
    }

    /// Marks `(x0,y0)..(x0+size,y0+size)` as one CU's area: its left
    /// column is a CU boundary (`bS == 2` candidate) for every row it
    /// spans, likewise its top row for every column.
    pub(crate) fn mark_cu_edges(&mut self, x0: usize, y0: usize, size: usize) {
        let (x_pu, y_pu, n) = (x0 / 4, y0 / 4, (size / 4).max(1));
        for dy in 0..n {
            if let Some(f) = self.cu_edge_left.get_mut((y_pu + dy) * self.min_pu_width + x_pu) {
                *f = true;
            }
        }
        for dx in 0..n {
            if let Some(f) = self.cu_edge_top.get_mut(y_pu * self.min_pu_width + (x_pu + dx)) {
                *f = true;
            }
        }
    }

    /// Same as `mark_cu_edges` but for a leaf luma transform block.
    pub(crate) fn mark_tu_edges(&mut self, x0: usize, y0: usize, size: usize) {
        let (x_pu, y_pu, n) = (x0 / 4, y0 / 4, (size / 4).max(1));
        for dy in 0..n {
            if let Some(f) = self.tu_edge_left.get_mut((y_pu + dy) * self.min_pu_width + x_pu) {
                *f = true;
            }
        }
        for dx in 0..n {
            if let Some(f) = self.tu_edge_top.get_mut(y_pu * self.min_pu_width + (x_pu + dx)) {
                *f = true;
            }
        }
    }

    pub(crate) fn set_qp_y(&mut self, x0: usize, y0: usize, size: usize, qp: i32) {
        let (x_pu, y_pu, n) = (x0 / 4, y0 / 4, (size / 4).max(1));
        let qp = qp.clamp(i8::MIN as i32, i8::MAX as i32) as i8;
        for dy in 0..n {
            for dx in 0..n {
                if let Some(q) = self.qp_y.get_mut((y_pu + dy) * self.min_pu_width + (x_pu + dx)) {
                    *q = qp;
                }
            }
        }
    }

    fn min_pu_index(&self, x: i64, y: i64) -> Option<usize> {
        if x < 0 || y < 0 || (x as usize) >= self.min_pu_width * 4 || (y as usize) >= self.min_pu_height * 4 {
            return None;
        }
        Some((y as usize / 4) * self.min_pu_width + (x as usize / 4))
    }

    pub(crate) fn qp_y_at(&self, x: i64, y: i64) -> Option<i32> {
        if y < self.slice_start_y as i64 {
            return None;
        }
        self.min_pu_index(x, y).and_then(|i| self.qp_y.get(i)).map(|&q| q as i32)
    }

    pub(crate) fn is_cu_edge_left(&self, x: i64, y: i64) -> Option<bool> {
        self.min_pu_index(x, y).and_then(|i| self.cu_edge_left.get(i)).copied()
    }

    pub(crate) fn is_cu_edge_top(&self, x: i64, y: i64) -> Option<bool> {
        self.min_pu_index(x, y).and_then(|i| self.cu_edge_top.get(i)).copied()
    }

    pub(crate) fn is_tu_edge_left(&self, x: i64, y: i64) -> Option<bool> {
        self.min_pu_index(x, y).and_then(|i| self.tu_edge_left.get(i)).copied()
    }

    pub(crate) fn is_tu_edge_top(&self, x: i64, y: i64) -> Option<bool> {
        self.min_pu_index(x, y).and_then(|i| self.tu_edge_top.get(i)).copied()
    }

    pub(crate) fn width(&self) -> usize {
        self.y.width
    }

    pub(crate) fn height(&self) -> usize {
        self.y.height
    }

    pub(crate) fn pic_width_ctbs(&self) -> usize {
        self.pic_width_ctbs
    }

    pub(crate) fn pic_height_ctbs(&self) -> usize {
        self.pic_height_ctbs
    }

    pub(crate) fn sao_at(&self, ctb_x: usize, ctb_y: usize) -> Option<SaoCtbParams> {
        self.sao.get(ctb_y * self.pic_width_ctbs + ctb_x).copied()
    }

    pub(crate) fn sao_params(&self) -> &[SaoCtbParams] {
        &self.sao
    }

    pub(crate) fn set_sao_at(&mut self, ctb_x: usize, ctb_y: usize, params: SaoCtbParams) {
        if let Some(slot) = self.sao.get_mut(ctb_y * self.pic_width_ctbs + ctb_x) {
            *slot = params;
        }
    }

    fn set_intra_mode(&mut self, x0: usize, y0: usize, size: usize, mode: u8) {
        let x_pu = x0 / 4;
        let y_pu = y0 / 4;
        let n = size / 4;
        for dy in 0..n {
            for dx in 0..n {
                if let Some(m) = self.intra_mode.get_mut((y_pu + dy) * self.min_pu_width + (x_pu + dx)) {
                    *m = mode;
                }
            }
        }
    }

    fn intra_mode_at(&self, x: i64, y: i64) -> Option<u8> {
        if x < 0 || y < self.slice_start_y as i64 {
            return None;
        }
        let xp = (x / 4) as usize;
        let yp = (y / 4) as usize;
        self.intra_mode.get(yp * self.min_pu_width + xp).copied()
    }

    fn set_ct_depth(&mut self, x0: usize, y0: usize, size: usize, depth: u8) {
        let cb = 1usize << self.min_cb_log2;
        let xg = x0 / cb;
        let yg = y0 / cb;
        let n = (size / cb).max(1);
        for dy in 0..n {
            for dx in 0..n {
                if let Some(d) = self.ct_depth.get_mut((yg + dy) * self.min_cb_width + (xg + dx)) {
                    *d = depth;
                }
            }
        }
    }

    fn ct_depth_at(&self, x: i64, y: i64) -> Option<u8> {
        if x < 0 || y < self.slice_start_y as i64 {
            return None;
        }
        let cb = 1u32 << self.min_cb_log2;
        let xg = (x as u32 / cb) as usize;
        let yg = (y as u32 / cb) as usize;
        self.ct_depth.get(yg * self.min_cb_width + xg).copied()
    }
}

struct Decoder<'e, 'b, 'c> {
    reference: Option<&'c Picture>,
    inter: Option<&'c InterSlice>,
    current_intra: bool,
    transquant_bypass: bool,
    inter_partition_split: bool,
    engine: &'e mut Engine<'b, 'c>,
    ctxs: Contexts,
    sps: Sps,
    pps: Pps,
    scaling_list: Option<&'e ScalingListData>,
    /// Current QG's QpY; at the next QG entry, this is the previous QG's
    /// final QpY used as the fallback for unavailable neighbor predictors.
    qp_y_prev: i32,
    is_cu_qp_delta_coded: bool,
    slice_cb_qp_offset: i32,
    slice_cr_qp_offset: i32,
    scratch: Box<BlockScratch>,
}

/// Working buffers for one transform block, sized for the largest (32x32)
/// and reused for every block instead of allocating per block.
struct BlockScratch {
    pred: [i32; 32 * 32],
    coeffs: [i32; 32 * 32],
    dequantized: [i32; 32 * 32],
    residual: [i32; 32 * 32],
}

/// The leading `nt x nt` part of a scratch buffer.
fn block(buf: &mut [i32], nt: usize) -> Result<&mut [i32]> {
    buf.get_mut(..nt * nt).ok_or(HeifError::MalformedHevc("reconstruct: block larger than 32x32"))
}

impl BlockScratch {
    fn new() -> Box<Self> {
        Box::new(BlockScratch { pred: [0; 32 * 32], coeffs: [0; 32 * 32], dequantized: [0; 32 * 32], residual: [0; 32 * 32] })
    }
}

/// Clause 8.6.1: only left/top neighbors in the current CTB participate.
/// Crossing a CTB edge uses the previous QG's QP instead. The caller resets
/// that fallback to SliceQpY at slice and WPP-row entry.
fn predict_qp(pic: &Picture, x: usize, y: usize, ctb_log2: u32, previous: i32) -> i32 {
    let mask = (1usize << ctb_log2) - 1;
    let a = if x & mask != 0 { pic.qp_y_at(x as i64 - 1, y as i64).unwrap_or(previous) } else { previous };
    let b = if y & mask != 0 { pic.qp_y_at(x as i64, y as i64 - 1).unwrap_or(previous) } else { previous };
    (a + b + 1) >> 1
}

fn apply_qp_delta(predicted: i32, delta: i32, bit_depth: u32) -> Result<i32> {
    if !(8..=16).contains(&bit_depth) {
        return Err(HeifError::Unsupported("QP: unsupported bit depth"));
    }
    let offset = 6 * (bit_depth as i32 - 8);
    Ok(((i64::from(predicted) + i64::from(delta) + i64::from(offset)).rem_euclid(i64::from(52 + offset)) - i64::from(offset)) as i32)
}

/// MPM candidate-list construction (clause 8.4.2).
fn fill_mpm_candidates(a: u8, b: u8) -> [u8; 3] {
    if a == b {
        if a < 2 {
            [0, 1, 26] // PLANAR, DC, ANGULAR_26
        } else {
            [a, 2 + ((a as i32 - 2 - 1 + 32) % 32) as u8, 2 + ((a as i32 - 2 + 1) % 32) as u8]
        }
    } else {
        let third = if a != 0 && b != 0 {
            0
        } else if a != 1 && b != 1 {
            1
        } else {
            26
        };
        [a, b, third]
    }
}

fn decode_mpm_idx(engine: &mut Engine) -> Result<u32> {
    // TR binarization, cMax = 2, all bypass.
    if engine.bypass()? == 0 {
        return Ok(0);
    }
    if engine.bypass()? == 0 {
        return Ok(1);
    }
    Ok(2)
}

fn apply_mpm(mpm_idx: u32, candidates: [u8; 3]) -> u8 {
    *candidates.get(mpm_idx as usize).unwrap_or(&1)
}

fn rem_mode_from_candidates(rem: u8, mut candidates: [u8; 3]) -> u8 {
    candidates.sort_unstable();
    let mut mode = rem;
    for &c in &candidates {
        if mode >= c {
            mode += 1;
        }
    }
    mode
}

impl<'e, 'b, 'c> Decoder<'e, 'b, 'c> {
    /// `sao(rx, ry)` (clause 7.3.8.3), storing the parsed per-CTB
    /// parameters into `pic` for `hevc::sao`'s later filter pass.
    fn decode_sao(&mut self, pic: &mut Picture, ctb_x: usize, ctb_y: usize, sao_luma: bool, sao_chroma: bool) -> Result<()> {
        let merge_left = if ctb_x > 0 {
            let model = self.ctxs.sao_merge_flag.first_mut().ok_or(HeifError::MalformedHevc("sao_merge_flag: context index"))?;
            self.engine.decision(model)? == 1
        } else {
            false
        };
        let merge_up = if ctb_y > 0 && !merge_left {
            let model = self.ctxs.sao_merge_flag.first_mut().ok_or(HeifError::MalformedHevc("sao_merge_flag: context index"))?;
            self.engine.decision(model)? == 1
        } else {
            false
        };
        if merge_left || merge_up {
            // clause 7.3.8.3: merge copies the *entire* neighbouring
            // CTB's SAO parameters (all components), not just its type.
            let (src_x, src_y) = if merge_left { (ctb_x - 1, ctb_y) } else { (ctb_x, ctb_y - 1) };
            let params = pic.sao_at(src_x, src_y).ok_or(HeifError::MalformedHevc("sao: merge neighbour out of range"))?;
            pic.set_sao_at(ctb_x, ctb_y, params);
            return Ok(());
        }

        let n_chroma = if self.sps.chroma_format_idc == 0 { 1 } else { 3 };
        let mut params = SaoCtbParams::default();
        let mut sao_type_idx_cb = 0u32;
        let mut sao_eo_class_cb = 0u32;
        for cidx in 0..n_chroma {
            let applies = (sao_luma && cidx == 0) || (sao_chroma && cidx > 0);
            if !applies {
                continue;
            }
            let sao_type_idx = if cidx <= 1 {
                let model = self.ctxs.sao_type_idx.first_mut().ok_or(HeifError::MalformedHevc("sao_type_idx: context index"))?;
                let t = if self.engine.decision(model)? == 0 {
                    0
                } else if self.engine.bypass()? == 0 {
                    1
                } else {
                    2
                };
                if cidx == 1 {
                    sao_type_idx_cb = t;
                }
                t
            } else {
                // Cr (cIdx==2) reuses Cb's (cIdx==1's) type rather than
                // reading its own bits — clause 7.3.8.3.
                sao_type_idx_cb
            };
            if let Some(slot) = params.type_idx.get_mut(cidx) {
                *slot = sao_type_idx as u8;
            }
            if sao_type_idx == 0 {
                continue;
            }
            let bit_depth = if cidx == 0 { self.sps.bit_depth_luma } else { self.sps.bit_depth_chroma };
            let c_max = (1u32 << (bit_depth.min(10) - 5)) - 1;
            let shift = self.pps.sao_offset_scale[usize::from(cidx != 0)];
            let mut offset_abs = [0i32; 4];
            for o in &mut offset_abs {
                *o = self.engine.bypass_tu(c_max)? as i32;
            }
            if sao_type_idx == 1 {
                for o in &mut offset_abs {
                    if *o != 0 && self.engine.bypass()? == 1 {
                        *o = -*o;
                    }
                }
                let band_position = self.engine.bypass_fl(5)?;
                if let Some(slot) = params.extra.get_mut(cidx) {
                    *slot = band_position as u8;
                }
            } else {
                let eo_class = if cidx <= 1 {
                    let c = self.engine.bypass_fl(2)?;
                    if cidx == 1 {
                        sao_eo_class_cb = c;
                    }
                    c
                } else {
                    // cIdx==2 (Cr) reuses cIdx==1's (Cb's) eo_class
                    // rather than reading its own bits — clause 7.3.8.3.
                    sao_eo_class_cb
                };
                if let Some(slot) = params.extra.get_mut(cidx) {
                    *slot = eo_class as u8;
                }
            }
            if let Some(slot) = params.offsets.get_mut(cidx) {
                for (o, v) in slot.iter_mut().zip(offset_abs) {
                    *o = v << shift;
                }
            }
        }
        pic.set_sao_at(ctb_x, ctb_y, params);
        Ok(())
    }

    fn decode_split_cu_flag(&mut self, pic: &Picture, x0: usize, y0: usize, ct_depth: u8) -> Result<bool> {
        let cond_l = pic.ct_depth_at(x0 as i64 - 1, y0 as i64).is_some_and(|d| d > ct_depth);
        let cond_a = pic.ct_depth_at(x0 as i64, y0 as i64 - 1).is_some_and(|d| d > ct_depth);
        let ctx_inc = cond_l as usize + cond_a as usize;
        let model = self.ctxs.split_cu_flag.get_mut(ctx_inc).ok_or(HeifError::MalformedHevc("split_cu_flag: context index"))?;
        Ok(self.engine.decision(model)? == 1)
    }

    fn coding_quadtree(&mut self, pic: &mut Picture, x0: usize, y0: usize, log2_cb_size: u32, ct_depth: u8) -> Result<()> {
        let size = 1usize << log2_cb_size;
        let fits = x0 + size <= self.sps.pic_width_in_luma_samples as usize && y0 + size <= self.sps.pic_height_in_luma_samples as usize;
        let min_cb_log2 = self.sps.log2_min_luma_coding_block_size;

        let split = if fits && log2_cb_size > min_cb_log2 { self.decode_split_cu_flag(pic, x0, y0, ct_depth)? } else { log2_cb_size > min_cb_log2 };

        // A QG is either an unsplit CU at least as large as the minimum
        // QG, or the whole subtree rooted at that minimum size. Smaller
        // CUs share the predictor and signal only one delta for the group.
        let min_qg_log2 = self.sps.ctb_log2_size_y() - self.pps.diff_cu_qp_delta_depth;
        let starts_qg = self.pps.cu_qp_delta_enabled && log2_cb_size >= min_qg_log2 && (!split || log2_cb_size == min_qg_log2);
        if starts_qg {
            self.is_cu_qp_delta_coded = false;
            self.qp_y_prev = predict_qp(pic, x0, y0, self.sps.ctb_log2_size_y(), self.qp_y_prev);
        }
        if split {
            let half = size / 2;
            let next_log2 = log2_cb_size - 1;
            for (dx, dy) in [(0, 0), (half, 0), (0, half), (half, half)] {
                let (cx, cy) = (x0 + dx, y0 + dy);
                if cx < self.sps.pic_width_in_luma_samples as usize && cy < self.sps.pic_height_in_luma_samples as usize {
                    self.coding_quadtree(pic, cx, cy, next_log2, ct_depth + 1)?;
                }
            }
        } else {
            pic.set_ct_depth(x0, y0, size, ct_depth);
            self.coding_unit(pic, x0, y0, log2_cb_size)?;
        }
        // Keep the QP recorded by each CU. A zero-residual CU preceding
        // this QG's first delta uses the predictor, not the later QP.
        // Repainting the whole QG here changes both deblocking and the
        // neighbor QPs used to predict subsequent groups.
        Ok(())
    }

    fn motion_delta(&mut self) -> Result<Motion> {
        let mut values = [self.engine.decision(&mut self.ctxs.mvd[0])?, self.engine.decision(&mut self.ctxs.mvd[0])?];
        for value in &mut values {
            if *value != 0 {
                *value += self.engine.decision(&mut self.ctxs.mvd[1])?;
            }
        }
        let mut signed = [0; 2];
        for (i, value) in values.iter_mut().enumerate() {
            if *value == 2 {
                let mut bits = 1u32;
                while self.engine.bypass()? != 0 {
                    if bits >= 15 {
                        return Err(HeifError::MalformedHevc("motion delta too large"));
                    }
                    *value += 1 << bits;
                    bits += 1;
                }
                *value += self.engine.bypass_fl(bits)?;
            }
            signed[i] = if *value != 0 && self.engine.bypass()? != 0 { -(*value as i32) } else { *value as i32 };
        }
        Ok(Motion { x: signed[0], y: signed[1] })
    }

    #[allow(clippy::too_many_arguments)]
    fn merge_candidates(&self, pic: &Picture, x: usize, y: usize, w: usize, h: usize, exclude_left: bool, exclude_top: bool) -> Vec<Motion> {
        let (x, y, w, h) = (x as i64, y as i64, w as i64, h as i64);
        let level = self.pps.log2_parallel_merge_level;
        let at = |nx, ny| {
            if nx >> level == x >> level && ny >> level == y >> level { None } else { pic.motion_at(nx, ny) }
        };
        let a1 = if exclude_left { None } else { at(x - 1, y + h - 1) };
        let b1 = if exclude_top { None } else { at(x + w - 1, y - 1) };
        let b0 = at(x + w, y - 1);
        let a0 = at(x - 1, y + h);
        let b2 = at(x - 1, y - 1);
        let mut out = Vec::new();
        if let Some(v) = a1 {
            out.push(v);
        }
        if let Some(v) = b1
            && Some(v) != a1
        {
            out.push(v);
        }
        if let Some(v) = b0
            && Some(v) != b1
        {
            out.push(v);
        }
        if let Some(v) = a0
            && Some(v) != a1
        {
            out.push(v);
        }
        if out.len() < 4
            && let Some(v) = b2
            && Some(v) != a1
            && Some(v) != b1
        {
            out.push(v);
        }
        // The sole reference is an IDR picture, so it has no temporal MV.
        out.resize(5, Motion::default());
        out
    }

    fn motion_predictors(&self, pic: &Picture, x: usize, y: usize, w: usize, h: usize) -> [Motion; 2] {
        let (x, y, w, h) = (x as i64, y as i64, w as i64, h as i64);
        let left = pic.motion_at(x - 1, y + h).or_else(|| pic.motion_at(x - 1, y + h - 1));
        let above = pic.motion_at(x + w, y - 1).or_else(|| pic.motion_at(x + w - 1, y - 1)).or_else(|| pic.motion_at(x - 1, y - 1));
        let mut list = Vec::new();
        if let Some(v) = left {
            list.push(v);
        }
        if let Some(v) = above
            && Some(v) != left
        {
            list.push(v);
        }
        list.resize(2, Motion::default());
        [list[0], list[1]]
    }

    fn inter_unit(&mut self, pic: &mut Picture, x: usize, y: usize, log2: u32, skipped: bool) -> Result<()> {
        let size = 1usize << log2;
        let mut shape = 0usize;
        if !skipped {
            let bins = if log2 == self.sps.log2_min_luma_coding_block_size && size != 8 { 3 } else { 2 };
            while shape < bins {
                if self.engine.decision(&mut self.ctxs.part_mode[shape])? != 0 {
                    break;
                }
                shape += 1;
            }
            if self.sps.amp_enabled
                && log2 > self.sps.log2_min_luma_coding_block_size
                && (shape == 1 || shape == 2)
                && self.engine.decision(&mut self.ctxs.part_mode[3])? == 0
            {
                shape = if shape == 1 { 4 } else { 6 } + self.engine.bypass()? as usize;
            }
        }
        let half = size / 2;
        let quarter = size / 4;
        let partitions = match shape {
            0 => vec![(0, 0, size, size)],
            1 => vec![(0, 0, size, half), (0, half, size, half)],
            2 => vec![(0, 0, half, size), (half, 0, half, size)],
            3 => vec![(0, 0, half, half), (half, 0, half, half), (0, half, half, half), (half, half, half, half)],
            4 => vec![(0, 0, size, quarter), (0, quarter, size, size - quarter)],
            5 => vec![(0, 0, size, size - quarter), (0, size - quarter, size, quarter)],
            6 => vec![(0, 0, quarter, size), (quarter, 0, size - quarter, size)],
            7 => vec![(0, 0, size - quarter, size), (size - quarter, 0, quarter, size)],
            _ => return Err(HeifError::MalformedHevc("inter partition shape")),
        };
        let mut full_merge = false;
        for (index, &(dx, dy, w, h)) in partitions.iter().enumerate() {
            let merged = skipped || self.engine.decision(&mut self.ctxs.merge[0])? != 0;
            full_merge = shape == 0 && merged;
            let mv = if merged {
                let max = self.inter.ok_or(HeifError::MalformedHevc("missing inter slice"))?.max_merge_candidates as usize;
                let mut candidate = 0;
                if max > 1 && self.engine.decision(&mut self.ctxs.merge_idx[0])? != 0 {
                    candidate = 1;
                    while candidate + 1 < max && self.engine.bypass()? != 0 {
                        candidate += 1;
                    }
                }
                // A merge region can cover all PUs in an 8x8 CU.
                let shared = self.pps.log2_parallel_merge_level > 2 && size == 8;
                let (px, py, pw, ph) = if shared { (x, y, size, size) } else { (x + dx, y + dy, w, h) };
                let left = !shared && index == 1 && matches!(shape, 2 | 6 | 7);
                let top = !shared && index == 1 && matches!(shape, 1 | 4 | 5);
                self.merge_candidates(pic, px, py, pw, ph, left, top)[candidate]
            } else {
                let delta = self.motion_delta()?;
                let index = self.engine.decision(&mut self.ctxs.mvp[0])? as usize;
                self.motion_predictors(pic, x + dx, y + dy, w, h)[index].with_delta(delta)
            };
            let reference = self.reference.ok_or(HeifError::Unsupported("inter: no reference picture"))?;
            compensate(&reference.y, &mut pic.y, x + dx, y + dy, w, h, mv, false)?;
            compensate(&reference.cb, &mut pic.cb, (x + dx) / 2, (y + dy) / 2, w / 2, h / 2, mv, true)?;
            compensate(&reference.cr, &mut pic.cr, (x + dx) / 2, (y + dy) / 2, w / 2, h / 2, mv, true)?;
            pic.mark_motion(x + dx, y + dy, w, h, mv, skipped);
        }
        self.inter_partition_split = shape != 0;
        let residual = !skipped && (full_merge || self.engine.decision(&mut self.ctxs.root_cbf[0])? != 0);
        if residual {
            self.transform_tree(pic, x, y, x, y, log2, 0, [true; 2], [true; 2], [0; 4], false)?;
        }
        // Even a zero-residual CU has an implicit root transform boundary.
        // Its neighbour may have coefficients and require bS = 1.
        else {
            pic.mark_tu_edges(x, y, size);
        }
        pic.set_qp_y(x, y, size, self.qp_y_prev);
        pic.mark_cu_edges(x, y, size);
        Ok(())
    }

    fn coding_unit(&mut self, pic: &mut Picture, x0: usize, y0: usize, log2_cb_size: u32) -> Result<()> {
        let size = 1usize << log2_cb_size;

        self.transquant_bypass = self.pps.transquant_bypass_enabled && self.engine.decision(&mut self.ctxs.transquant_bypass[0])? != 0;
        pic.mark_bypass(x0, y0, size, self.transquant_bypass);

        self.current_intra = true;
        self.inter_partition_split = false;
        if self.inter.is_some() {
            let ctx = pic.skip_at(x0 as i64 - 1, y0 as i64) as usize + pic.skip_at(x0 as i64, y0 as i64 - 1) as usize;
            let skipped = self.engine.decision(&mut self.ctxs.skip[ctx])? != 0;
            if skipped || self.engine.decision(&mut self.ctxs.pred_mode[0])? == 0 {
                self.current_intra = false;
                return self.inter_unit(pic, x0, y0, log2_cb_size, skipped);
            }
        }

        let part_nxn = if log2_cb_size == self.sps.log2_min_luma_coding_block_size && self.sps.log2_min_luma_coding_block_size > 2 {
            // part_mode signaled; ctxIdxInc 0 is the only bin this
            // decoder's intra-only scope needs (bit==1 -> 2Nx2N, 0 -> NxN).
            let model = self.ctxs.part_mode.first_mut().ok_or(HeifError::MalformedHevc("part_mode: context index"))?;
            self.engine.decision(model)? == 0
        } else {
            false
        };

        if self.sps.pcm_enabled {
            return Err(HeifError::Unsupported("coding_unit: pcm_enabled SPS not supported"));
        }

        // --- luma intra mode(s) ---
        // clause 7.3.8.5: under PART_NxN, all four prev_intra_luma_pred_flag
        // bins are read first (in raster PU order), *then* all four
        // mpm_idx/rem_intra_luma_pred_mode derivations — not interleaved
        // per PU.
        let luma_mode_0 = if part_nxn {
            let half = size / 2;
            let positions = [(x0, y0), (x0 + half, y0), (x0, y0 + half), (x0 + half, y0 + half)];
            let mut prev_flags = [false; 4];
            for pf in &mut prev_flags {
                let model = self.ctxs.prev_intra_luma_pred_flag.first_mut().ok_or(HeifError::MalformedHevc("prev_intra_luma_pred_flag: context index"))?;
                *pf = self.engine.decision(model)? == 1;
            }
            let mut modes = [0u8; 4];
            for (i, &(px, py)) in positions.iter().enumerate() {
                let mode = self.decode_one_luma_mode(pic, px, py, prev_flags[i])?;
                pic.set_intra_mode(px, py, half, mode);
                modes[i] = mode;
            }
            modes[0]
        } else {
            let model = self.ctxs.prev_intra_luma_pred_flag.first_mut().ok_or(HeifError::MalformedHevc("prev_intra_luma_pred_flag: context index"))?;
            let prev_flag = self.engine.decision(model)? == 1;
            let mode = self.decode_one_luma_mode(pic, x0, y0, prev_flag)?;
            pic.set_intra_mode(x0, y0, size, mode);
            mode
        };

        // 4:4:4 NxN has a chroma prediction mode for each luma PU.
        let mut chroma_modes = [0u8; 4];
        if part_nxn && self.sps.chroma_format_idc == 3 {
            let half = size / 2;
            for (i, (dx, dy)) in [(0, 0), (half, 0), (0, half), (half, half)].into_iter().enumerate() {
                let luma = pic.intra_mode_at((x0 + dx) as i64, (y0 + dy) as i64).ok_or(HeifError::MalformedHevc("chroma: missing luma PU mode"))?;
                chroma_modes[i] = self.decode_chroma_mode(luma)?;
            }
        } else if self.sps.chroma_format_idc != 0 {
            chroma_modes.fill(self.decode_chroma_mode(luma_mode_0)?);
        }

        // Delta presence is tracked across all CUs in the quantization
        // group, initialized by coding_quadtree rather than by each CU.

        self.transform_tree(
            pic,
            x0,
            y0,
            x0,
            y0,
            log2_cb_size,
            0,
            [self.sps.chroma_format_idc != 0; 2],
            [self.sps.chroma_format_idc != 0; 2],
            chroma_modes,
            part_nxn,
        )?;

        // Deblocking bookkeeping (clause 8.7.2.4): this CU's whole area
        // used `self.qp_y_prev`'s final value as its `QpY`, whether or
        // not a `cu_qp_delta` was actually coded for it, and its outer
        // boundary is always a CU edge (`bS == 2` candidate, intra-only).
        pic.set_qp_y(x0, y0, 1usize << log2_cb_size, self.qp_y_prev);
        pic.mark_cu_edges(x0, y0, 1usize << log2_cb_size);
        Ok(())
    }

    /// `mpm_idx`/`rem_intra_luma_pred_mode` derivation for one PU
    /// (clause 8.4.2), given its already-decoded `prev_intra_luma_pred_flag`.
    fn decode_one_luma_mode(&mut self, pic: &Picture, x0: usize, y0: usize, prev_flag: bool) -> Result<u8> {
        let cand_a = pic.intra_mode_at(x0 as i64 - 1, y0 as i64).unwrap_or(1);
        let cand_b = if (y0 as i64 - 1) < ((y0 as i64 >> self.sps.ctb_log2_size_y()) << self.sps.ctb_log2_size_y()) {
            1 // INTRA_DC: neighbour is in a different CTB row, treated as unavailable
        } else {
            pic.intra_mode_at(x0 as i64, y0 as i64 - 1).unwrap_or(1)
        };
        let candidates = fill_mpm_candidates(cand_a, cand_b);
        if prev_flag {
            let mpm_idx = decode_mpm_idx(self.engine)?;
            Ok(apply_mpm(mpm_idx, candidates))
        } else {
            let rem = self.engine.bypass_fl(5)? as u8;
            Ok(rem_mode_from_candidates(rem, candidates))
        }
    }

    /// `cu_qp_delta_abs` + `cu_qp_delta_sign_flag` (clause 7.3.8.10),
    /// applied to `self.qp_y_prev`. Caller gates presence; see
    /// `transform_unit`.
    fn decode_cu_qp_delta(&mut self) -> Result<i32> {
        let model0 = self.ctxs.cu_qp_delta_abs.first_mut().ok_or(HeifError::MalformedHevc("cu_qp_delta_abs: context index"))?;
        if self.engine.decision(model0)? == 0 {
            return Ok(self.qp_y_prev);
        }
        let mut prefix = 1u32;
        for _ in 0..4 {
            let model1 = self.ctxs.cu_qp_delta_abs.get_mut(1).ok_or(HeifError::MalformedHevc("cu_qp_delta_abs: context index"))?;
            if self.engine.decision(model1)? == 0 {
                break;
            }
            prefix += 1;
        }
        let abs_delta = if prefix == 5 { self.engine.egk_bypass(0)? + 5 } else { prefix };
        let sign = if abs_delta != 0 { self.engine.bypass()? } else { 0 };
        let delta = if sign == 1 { -(abs_delta as i32) } else { abs_delta as i32 };
        apply_qp_delta(self.qp_y_prev, delta, self.sps.bit_depth_luma)
    }

    fn decode_chroma_mode(&mut self, luma_mode: u8) -> Result<u8> {
        let model = self.ctxs.intra_chroma_pred_mode.first_mut().ok_or(HeifError::MalformedHevc("intra_chroma_pred_mode: context index"))?;
        let mode = if self.engine.decision(model)? == 0 {
            luma_mode // intra_chroma_pred_mode == 4 maps to "derived from luma"
        } else {
            let idx = self.engine.bypass_fl(2)? as u8;
            const CHROMA_CAND_BASE: [u8; 4] = [0, 26, 10, 1];
            let mut cand = CHROMA_CAND_BASE;
            if let Some(slot) = cand.iter_mut().find(|c| **c == luma_mode) {
                *slot = 34;
            }
            *cand.get(idx as usize).ok_or(HeifError::MalformedHevc("intra_chroma_pred_mode: candidate index"))?
        };

        // Table 8-4 adjusts angular directions for the horizontal-only
        // chroma subsampling. Apply after resolving derived/candidate mode.
        const ANGLES_422: [u8; 35] =
            [0, 1, 2, 2, 2, 2, 3, 5, 7, 8, 10, 12, 13, 15, 17, 18, 19, 20, 21, 22, 23, 23, 24, 24, 25, 25, 26, 27, 27, 28, 28, 29, 29, 30, 31];
        if self.sps.chroma_format_idc == 2 { ANGLES_422.get(mode as usize).copied().ok_or(HeifError::MalformedHevc("4:2:2 chroma mode")) } else { Ok(mode) }
    }

    #[allow(clippy::too_many_arguments)]
    fn transform_tree(
        &mut self,
        pic: &mut Picture,
        x0: usize,
        y0: usize,
        x_base: usize,
        y_base: usize,
        log2_trafo_size: u32,
        trafo_depth: u32,
        cbf_cb: [bool; 2],
        cbf_cr: [bool; 2],
        chroma_modes: [u8; 4],
        intra_split_flag: bool,
    ) -> Result<()> {
        let max_tb_log2 = self.sps.max_tb_log2_size_y();
        let min_tb_log2 = self.sps.min_tb_log2_size_y();
        // `MaxTrafoDepth` (clause 7.4.9.8) is `max_transform_hierarchy_depth_intra
        // + IntraSplitFlag` (1 for a `PART_NxN` CU, so its forced depth-0
        // split doesn't eat into the normally-available split budget).
        let max_trafo_depth = if self.current_intra {
            self.sps.max_transform_hierarchy_depth_intra + intra_split_flag as u32
        } else {
            self.sps.max_transform_hierarchy_depth_inter
        };
        let size = 1usize << log2_trafo_size;

        // Clause 7.3.8.8's exact presence condition for `split_transform_flag`:
        // signal it only when the transform block can still legally split
        // AND the depth limit hasn't been hit AND we're not forced to
        // split anyway (too big, or a PART_NxN CU's mandatory depth-0
        // split to match its 4 PUs); otherwise it's inferred.
        let split = if log2_trafo_size > max_tb_log2
            || (intra_split_flag && trafo_depth == 0)
            || (!self.current_intra && max_trafo_depth == 0 && self.inter_partition_split && trafo_depth == 0)
        {
            true
        } else if log2_trafo_size > min_tb_log2 && trafo_depth < max_trafo_depth {
            let ctx_inc = (5 - log2_trafo_size).clamp(0, 2) as usize;
            let model = self.ctxs.split_transform_flag.get_mut(ctx_inc).ok_or(HeifError::MalformedHevc("split_transform_flag: context index"))?;
            self.engine.decision(model)? == 1
        } else {
            false
        };

        // Only subsampled chroma shares the 8x8 luma parent's transform.
        // In 4:4:4 even a 4x4 leaf has its own chroma flags and residuals.
        let chroma_follows_parent = log2_trafo_size == 2 && self.sps.chroma_format_idc != 3;

        // 4:2:2 leaf chroma is a rectangle made of two stacked square
        // transforms. At an 8x8 luma node their flags are read even when
        // luma splits further: the resulting 4x8 chroma cannot quad-split.
        let chroma_halves = self.sps.chroma_format_idc == 2 && (!split || log2_trafo_size == 3);
        let mut chroma_flags = [cbf_cb, cbf_cr];
        if !chroma_follows_parent && self.sps.chroma_format_idc != 0 {
            for flags in &mut chroma_flags {
                let present = flags.iter().any(|&flag| flag);
                *flags = [false; 2];
                if present {
                    let count = if chroma_halves { 2 } else { 1 };
                    for flag in &mut flags[..count] {
                        let model = self.ctxs.cbf_chroma.get_mut(trafo_depth as usize).ok_or(HeifError::MalformedHevc("cbf_chroma: context index"))?;
                        *flag = self.engine.decision(model)? == 1;
                    }
                }
            }
        }
        let [cbf_cb, cbf_cr] = chroma_flags;
        if split {
            let half = size / 2;
            for (i, (dx, dy)) in [(0, 0), (half, 0), (0, half), (half, half)].into_iter().enumerate() {
                // NxN's mandatory root split matches its four prediction
                // units. Descendants keep their quadrant's chroma mode.
                let modes = if trafo_depth == 0 && intra_split_flag { [chroma_modes[i]; 4] } else { chroma_modes };
                // The child inherits this transform node's origin, not
                // the coding unit's origin. At 4x4 luma leaves it locates
                // the shared chroma block in the immediate 8x8 parent.
                self.transform_tree(pic, x0 + dx, y0 + dy, x0, y0, log2_trafo_size - 1, trafo_depth + 1, cbf_cb, cbf_cr, modes, intra_split_flag)?;
            }
        } else {
            // clause 7.3.8.8: `cbf_luma` is present when
            // `CuPredMode == MODE_INTRA || trafoDepth != 0 || cbf_cb || cbf_cr`.
            // For intra coding cbf_luma is always signalled; an inter
            // root without chroma residual infers it as one.
            let cbf_luma = if !self.current_intra && trafo_depth == 0 && !cbf_cb.iter().any(|&flag| flag) && !cbf_cr.iter().any(|&flag| flag) {
                true
            } else {
                let ctx_inc = (trafo_depth == 0) as usize;
                let model = self.ctxs.cbf_luma.get_mut(ctx_inc).ok_or(HeifError::MalformedHevc("cbf_luma: context index"))?;
                self.engine.decision(model)? == 1
            };
            self.transform_unit(pic, x0, y0, x_base, y_base, log2_trafo_size, trafo_depth, cbf_luma, cbf_cb, cbf_cr, chroma_follows_parent, chroma_modes[0])?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn transform_unit(
        &mut self,
        pic: &mut Picture,
        x0: usize,
        y0: usize,
        x_base: usize,
        y_base: usize,
        log2_trafo_size: u32,
        trafo_depth: u32,
        cbf_luma: bool,
        cbf_cb: [bool; 2],
        cbf_cr: [bool; 2],
        chroma_follows_parent: bool,
        chroma_mode: u8,
    ) -> Result<()> {
        // clause 7.3.8.10: `cu_qp_delta_abs` is read here, at most once
        // per quantization group, and only when this TU actually has
        // some residual. A TU with every cbf flag false (e.g. a
        // perfectly flat block) never signals a QP delta at all.
        if (cbf_luma || cbf_cb.iter().any(|&flag| flag) || cbf_cr.iter().any(|&flag| flag)) && self.pps.cu_qp_delta_enabled && !self.is_cu_qp_delta_coded {
            self.qp_y_prev = self.decode_cu_qp_delta()?;
            self.is_cu_qp_delta_coded = true;
        }

        // Deblocking bookkeeping: this leaf luma TU's own left/top edges
        // (clause 8.7.2.4 — `bS == 2` here too; see `Picture`'s docs).
        pic.mark_tu_edges(x0, y0, 1usize << log2_trafo_size);
        pic.mark_residual(x0, y0, 1usize << log2_trafo_size, cbf_luma);

        // Luma intra mode is looked up per-position rather than threaded
        // as a parameter: under PART_NxN each of the 4 PUs has its own
        // mode, already recorded via `Picture::set_intra_mode` when it
        // was decoded in `coding_unit`, and a plain PART_2Nx2N CU's
        // single mode covers its whole area the same way.
        let luma_mode =
            pic.intra_mode_at(x0 as i64, y0 as i64).ok_or(HeifError::MalformedHevc("transform_unit: no luma intra mode recorded at this position"))?;
        self.predict_and_reconstruct(pic, 0, x0, y0, log2_trafo_size, luma_mode, cbf_luma)?;

        if self.sps.chroma_format_idc == 0 {
            return Ok(());
        }

        let (cx0, cy0, clog2) = if self.sps.chroma_format_idc == 3 {
            (x0, y0, log2_trafo_size)
        } else if chroma_follows_parent {
            (x_base / 2, y_base / self.sps.sub_height_c() as usize, 2u32)
        } else {
            (x0 / 2, y0 / self.sps.sub_height_c() as usize, log2_trafo_size - 1)
        };

        // For 4x4 luma leaves, chroma is only processed once (when we're
        // at the last of the four 4x4 sub-blocks within the parent 8x8,
        // i.e. x0/y0 == x_base/y_base + (4,4) in z-order — simplest
        // correct check: process chroma exactly when (x0,y0) is the
        // bottom-right 4x4 leaf of its parent 8x8 luma block).
        if chroma_follows_parent {
            let is_last_leaf = (x0 - x_base) == 4 && (y0 - y_base) == 4;
            if !is_last_leaf {
                return Ok(());
            }
        }
        let _ = trafo_depth;

        let halves = if self.sps.chroma_format_idc == 2 { 2 } else { 1 };
        // Component order is Cb top/bottom, then Cr top/bottom.
        for (cidx, flags) in [(1, cbf_cb), (2, cbf_cr)] {
            for (half, &cbf) in flags.iter().enumerate().take(halves) {
                self.predict_and_reconstruct(pic, cidx, cx0, cy0 + (half << clog2), clog2, chroma_mode, cbf)?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn predict_and_reconstruct(&mut self, pic: &mut Picture, cidx: usize, x0: usize, y0: usize, log2_size: u32, mode: u8, cbf: bool) -> Result<()> {
        let nt = 1usize << log2_size;
        let bit_depth = if cidx == 0 { self.sps.bit_depth_luma } else { self.sps.bit_depth_chroma };
        let is_luma_or_444 = cidx == 0 || self.sps.chroma_format_idc == 3;

        {
            let plane = match cidx {
                0 => &pic.y,
                1 => &pic.cb,
                _ => &pic.cr,
            };
            let scratch = &mut *self.scratch;
            // Every prediction path below writes all nt*nt samples.
            let pred = block(&mut scratch.pred, nt)?;
            let timer = timing::start(Stage::Intra);
            if self.current_intra {
                let start_y = pic.slice_start_y / if cidx == 0 { 1 } else { self.sps.sub_height_c() as usize };
                let mut border = Border::build(x0 as i64, y0 as i64, nt, bit_depth, |x, y| if y < start_y as i64 { None } else { plane.get(x, y) })?;
                if is_luma_or_444 {
                    border.filter(nt, cidx, mode, bit_depth, self.sps.strong_intra_smoothing_enabled)?;
                }
                match mode {
                    0 => predict_planar(pred, nt, &border)?,
                    1 => predict_dc(pred, nt, cidx, &border)?,
                    _ => predict_angular(pred, nt, cidx, mode, bit_depth, false, &border)?,
                }
            } else {
                for y in 0..nt {
                    for x in 0..nt {
                        pred[y * nt + x] = plane.sample(x0 + x, y0 + y).ok_or(HeifError::MalformedHevc("inter prediction bounds"))?;
                    }
                }
            }
            drop(timer);

            if cbf {
                // clause 7.3.8.8: `transform_skip_flag` is read right
                // before `residual_coding`, once per component, only for
                // 4x4 blocks (`Log2MaxTransformSkipSize` is always 2 for
                // the RExt tools this decoder doesn't implement) and
                // only when the PPS enables it. Reading it unconditionally
                // skipped is a silent, permanent bit-desync the moment an
                // encoder actually uses it on a 4x4 TU — this doesn't
                // show up on any Apple-encoded real photo tested so far,
                // only on non-Apple-encoded HEIF conformance files.
                let timer = timing::start(Stage::Residual);
                let transform_skip = if !self.transquant_bypass && self.pps.transform_skip_enabled && nt == 4 {
                    let ctx_idx = if cidx == 0 { 0 } else { 1 };
                    let model = self.ctxs.transform_skip_flag.get_mut(ctx_idx).ok_or(HeifError::MalformedHevc("transform_skip_flag: context index"))?;
                    self.engine.decision(model)? == 1
                } else {
                    false
                };
                let intra_pred_mode = if self.current_intra { mode as u32 } else { 0 };
                let coeffs = block(&mut scratch.coeffs, nt)?;
                decode_residual(
                    self.engine,
                    &mut self.ctxs,
                    log2_size,
                    cidx,
                    intra_pred_mode,
                    is_luma_or_444,
                    self.pps.sign_data_hiding_enabled && !self.transquant_bypass,
                    coeffs,
                )?;
                drop(timer);
                let timer = timing::start(Stage::Transform);
                let res = block(&mut scratch.residual, nt)?;
                if self.transquant_bypass {
                    res.copy_from_slice(coeffs);
                } else {
                    let qp = if cidx == 0 {
                        self.qp_y_prev + 6 * (self.sps.bit_depth_luma as i32 - 8)
                    } else {
                        // Clause 8.6.1: qPi -> QpCb/QpCr. `QpBdOffsetC` is 0
                        // for the 8-bit-chroma streams this decoder has been
                        // validated against; included anyway since it's a
                        // one-line generalization for higher bit depths.
                        let offset = if cidx == 1 { self.pps.cb_qp_offset + self.slice_cb_qp_offset } else { self.pps.cr_qp_offset + self.slice_cr_qp_offset };
                        let qp_bd_offset_c = 6 * (self.sps.bit_depth_chroma as i32 - 8);
                        let qp_i = (self.qp_y_prev + offset).clamp(-qp_bd_offset_c, 57);
                        let qp_c = if self.sps.chroma_format_idc == 1 { chroma_qp_mapping(qp_i) } else { qp_i.min(51) };
                        qp_c + qp_bd_offset_c
                    };
                    let dequantized = block(&mut scratch.dequantized, nt)?;
                    let scale = Dequant::new(qp, bit_depth, log2_size)?;
                    for (i, (o, &c)) in dequantized.iter_mut().zip(coeffs.iter()).enumerate() {
                        // A zero level dequantizes to zero, and most of a block is zeros.
                        if c == 0 {
                            *o = 0;
                            continue;
                        }
                        let weight = match self.scaling_list {
                            Some(list) => list.weight(log2_size, cidx, i % nt, i / nt)?,
                            None => 16,
                        };
                        *o = scale.apply(c, weight)?;
                    }
                    if transform_skip {
                        transform_skip_residual(dequantized, nt, bit_depth, res)?;
                    } else {
                        let use_dst = self.current_intra && cidx == 0 && nt == 4;
                        inverse_transform(dequantized, nt, bit_depth, use_dst, res)?;
                    }
                }
                drop(timer);
                let _timer = timing::start(Stage::Write);
                for i in 0..nt * nt {
                    let p = *pred.get(i).ok_or(HeifError::MalformedHevc("reconstruct: pred index"))?;
                    let r = *res.get(i).ok_or(HeifError::MalformedHevc("reconstruct: res index"))?;
                    *pred.get_mut(i).ok_or(HeifError::MalformedHevc("reconstruct: pred index"))? = (p + r).clamp(0, (1i32 << bit_depth) - 1);
                }
            }

            let _timer = timing::start(Stage::Write);
            let plane = match cidx {
                0 => &mut pic.y,
                1 => &mut pic.cb,
                _ => &mut pic.cr,
            };
            for y in 0..nt {
                for x in 0..nt {
                    let v = *pred.get(x + y * nt).ok_or(HeifError::MalformedHevc("reconstruct: write index"))?;
                    plane.set(x0 + x, y0 + y, v)?;
                }
            }
        }
        Ok(())
    }
}

/// Decodes one I/P slice covering the whole picture into `pic`.
/// The container path also supports row-aligned independent intra slices.
///
/// `slice_data` is the RBSP bytes starting at `slice_segment_data()`.
/// `substream_starts` holds the RBSP byte position of every substream
/// after the first (already converted from the slice header's
/// EBSP-space entry-point offsets by the caller — see
/// `crate::bitreader::{rbsp_len_to_ebsp_pos, ebsp_len_to_rbsp_len}`).
/// Empty means one continuous substream spanning the whole picture.
///
/// When `substream_starts` has exactly `pic_height_in_ctbs - 1` entries
/// and `pps.entropy_coding_sync_enabled` (true wavefront parallel
/// processing), each substream is exactly one CTB row: the CABAC
/// engine is re-initialized at each substream's own entry point, with
/// its context models restored from the state saved right after CTB
/// column 1 of the *previous* row (clause 9.3.2.4), and the QP
/// predictor resets to `SliceQpY` at each row start (clause 8.6.1).
/// Any other non-empty `substream_starts` (e.g. tiles, which this
/// decoder doesn't support) is rejected rather than mis-decoded.
#[allow(clippy::too_many_arguments)]
pub fn decode_slice(
    pic: &mut Picture,
    slice_data: &[u8],
    substream_starts: &[usize],
    sps: &Sps,
    pps: &Pps,
    slice_qp_y: i32,
    sao_luma: bool,
    sao_chroma: bool,
    slice_cb_qp_offset: i32,
    slice_cr_qp_offset: i32,
    inter: Option<&InterSlice>,
    reference: Option<&Picture>,
) -> Result<()> {
    let rows = (sps.pic_height_in_luma_samples as usize).div_ceil(1 << sps.ctb_log2_size_y());
    decode_slice_rows(
        pic,
        slice_data,
        substream_starts,
        sps,
        pps,
        slice_qp_y,
        sao_luma,
        sao_chroma,
        slice_cb_qp_offset,
        slice_cr_qp_offset,
        inter,
        reference,
        0,
        rows,
    )
}

/// Decode an independent slice covering complete CTU rows. The caller
/// rejects loop filtering across multiple slices until it is implemented.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_slice_rows(
    pic: &mut Picture,
    slice_data: &[u8],
    substream_starts: &[usize],
    sps: &Sps,
    pps: &Pps,
    slice_qp_y: i32,
    sao_luma: bool,
    sao_chroma: bool,
    slice_cb_qp_offset: i32,
    slice_cr_qp_offset: i32,
    inter: Option<&InterSlice>,
    reference: Option<&Picture>,
    first_row: usize,
    end_row: usize,
) -> Result<()> {
    let ctb_log2 = sps.ctb_log2_size_y();
    if pps.diff_cu_qp_delta_depth > sps.log2_diff_max_min_luma_coding_block_size {
        return Err(HeifError::MalformedHevc("QP: quantization group smaller than minimum CU"));
    }
    let scaling_list = if sps.scaling_list_enabled { Some(pps.scaling_list.as_ref().or(sps.scaling_list.as_ref()).cloned().unwrap_or_default()) } else { None };
    let ctb_size = 1usize << ctb_log2;
    let pic_width_ctbs = (sps.pic_width_in_luma_samples as usize).div_ceil(ctb_size);
    let pic_height_ctbs = (sps.pic_height_in_luma_samples as usize).div_ceil(ctb_size);

    if first_row >= end_row || end_row > pic_height_ctbs {
        return Err(HeifError::MalformedHevc("slice: invalid CTU row range"));
    }
    pic.slice_start_y = first_row * ctb_size;
    let slice_rows = end_row - first_row;
    let num_substreams = substream_starts.len() + 1;
    let true_wpp = pps.entropy_coding_sync_enabled && num_substreams == slice_rows && slice_rows > 1;
    if num_substreams > 1 && !true_wpp {
        return Err(HeifError::Unsupported("decode_slice: multiple substreams present but not a simple per-row WPP picture"));
    }
    let rows_per_substream = if true_wpp { 1 } else { slice_rows };
    let effective_substreams = if true_wpp { num_substreams } else { 1 };

    let mut ctxs = if inter.is_some() { Contexts::init_p_slice(slice_qp_y.clamp(0, 51)) } else { Contexts::init_i_slice(slice_qp_y.clamp(0, 51)) };
    let mut qp_y_prev = slice_qp_y;
    let mut is_cu_qp_delta_coded = false;
    let mut wpp_saved_ctx: Option<Contexts> = None;

    for substream_idx in 0..effective_substreams {
        let start = if substream_idx == 0 {
            0
        } else {
            *substream_starts.get(substream_idx - 1).ok_or(HeifError::CabacDesync("decode_slice: missing substream start"))?
        };
        let end = substream_starts.get(substream_idx).copied().unwrap_or(slice_data.len());
        let substream = slice_data.get(start..end).ok_or(HeifError::Truncated { at: start, needed: end.saturating_sub(start) })?;

        if true_wpp && substream_idx > 0 {
            if let Some(saved) = &wpp_saved_ctx {
                ctxs = saved.clone();
            }
            qp_y_prev = slice_qp_y;
            is_cu_qp_delta_coded = false;
        }

        let mut bits = BitReader::new(substream);
        let mut engine = Engine::new(&mut bits)?;
        let mut decoder = Decoder {
            reference,
            inter,
            current_intra: true,
            transquant_bypass: false,
            inter_partition_split: false,
            engine: &mut engine,
            ctxs,
            sps: sps.clone(),
            pps: pps.clone(),
            scaling_list: scaling_list.as_ref(),
            qp_y_prev,
            is_cu_qp_delta_coded,
            slice_cb_qp_offset,
            slice_cr_qp_offset,
            scratch: BlockScratch::new(),
        };

        let row_start = first_row + substream_idx * rows_per_substream;
        let row_end = (row_start + rows_per_substream).min(end_row);

        for ctb_y in row_start..row_end {
            for ctb_x in 0..pic_width_ctbs {
                if sao_luma || sao_chroma {
                    decoder.decode_sao(pic, ctb_x, ctb_y, sao_luma, sao_chroma)?;
                }
                decoder.coding_quadtree(pic, ctb_x * ctb_size, ctb_y * ctb_size, ctb_log2, 0)?;

                if true_wpp && ctb_x == 1 && ctb_y < pic_height_ctbs - 1 {
                    wpp_saved_ctx = Some(decoder.ctxs.clone());
                }

                // Clause 7.3.8.1: `end_of_slice_segment_flag` is read after
                // every CTU. Only when it's 0 AND this CTU is the last one
                // before a tile/WPP-row boundary is a *second*, separate
                // terminate bin read: `end_of_subset_one_bit` (always 1),
                // followed by `byte_alignment()` — which this decoder's
                // per-substream slicing (each substream starts its own
                // fresh engine at its own entry-point byte offset) already
                // accounts for without needing to consume the padding here.
                let is_last_in_slice = ctb_y == end_row - 1 && ctb_x == pic_width_ctbs - 1;
                let is_last_in_substream = ctb_y == row_end - 1 && ctb_x == pic_width_ctbs - 1;
                let end_of_slice_segment_flag = decoder.engine.terminate()?;
                if is_last_in_slice {
                    if end_of_slice_segment_flag != 1 {
                        return Err(HeifError::CabacDesync("decode_slice: missing end_of_slice_segment_flag"));
                    }
                } else {
                    if end_of_slice_segment_flag != 0 {
                        return Err(HeifError::Unsupported("decode_slice: multiple tiles/slice segments not supported"));
                    }
                    if is_last_in_substream {
                        let end_of_subset_one_bit = decoder.engine.terminate()?;
                        if end_of_subset_one_bit != 1 {
                            return Err(HeifError::CabacDesync("decode_slice: missing end_of_subset_one_bit"));
                        }
                    }
                }
            }
        }

        ctxs = decoder.ctxs;
        qp_y_prev = decoder.qp_y_prev;
        is_cu_qp_delta_coded = decoder.is_cu_qp_delta_coded;
    }
    pic.slice_start_y = 0;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inter_boundary_strength_uses_residuals_and_motion() {
        let mut pic = Picture::new(32, 32, 2, 2, 8, 8, 3, 5);
        pic.mark_motion(0, 0, 16, 16, Motion::default(), false);
        pic.mark_motion(16, 0, 16, 16, Motion::default(), true);
        pic.mark_tu_edges(16, 0, 16);
        assert_eq!(pic.boundary_strength(16, 0, true), 0);
        // The skipped Q block still has a TU boundary: P residuals count.
        pic.mark_residual(0, 0, 16, true);
        assert_eq!(pic.boundary_strength(16, 0, true), 1);
        pic.mark_residual(0, 0, 16, false);
        pic.mark_motion(16, 0, 16, 16, Motion { x: 3, y: 0 }, false);
        assert_eq!(pic.boundary_strength(16, 0, true), 0);
        pic.mark_motion(16, 0, 16, 16, Motion { x: 4, y: 0 }, false);
        assert_eq!(pic.boundary_strength(16, 0, true), 1);
        // The lower side is intra (has no motion vector).
        assert_eq!(pic.boundary_strength(16, 16, false), 2);
    }

    #[test]
    fn qp_prediction_averages_neighbors_with_rounding() {
        let mut pic = Picture::new(64, 64, 2, 2, 8, 8, 3, 5);
        pic.set_qp_y(0, 8, 8, 20);
        pic.set_qp_y(8, 0, 8, 31);
        // Previous QG was 15, but both neighbors are available: (20+31+1)/2.
        assert_eq!(predict_qp(&pic, 8, 8, 5, 15), 26);
    }

    #[test]
    fn qp_prediction_uses_fallback_at_ctb_edges() {
        let mut pic = Picture::new(64, 64, 2, 2, 8, 8, 3, 5);
        pic.set_qp_y(24, 8, 8, 50); // left is in another CTB: ignore it
        pic.set_qp_y(32, 0, 8, 30);
        assert_eq!(predict_qp(&pic, 32, 8, 5, 15), 23);
        pic.set_qp_y(8, 24, 8, 50); // top is in another CTB: ignore it
        pic.set_qp_y(0, 32, 8, 20);
        assert_eq!(predict_qp(&pic, 8, 32, 5, 15), 18);
        assert_eq!(predict_qp(&pic, 32, 32, 5, 15), 15);
        // At a WPP row's start the caller supplies SliceQpY as fallback.
        assert_eq!(predict_qp(&pic, 0, 32, 5, 24), 24);
    }

    #[test]
    fn qp_delta_wraps_in_bit_depth_dependent_range() {
        assert_eq!(apply_qp_delta(50, 3, 8).unwrap(), 1);
        assert_eq!(apply_qp_delta(0, -1, 8).unwrap(), 51);
        assert_eq!(apply_qp_delta(-12, -1, 10).unwrap(), 51);
        assert_eq!(apply_qp_delta(51, 1, 10).unwrap(), -12);
    }

    #[test]
    fn mpm_candidates_match_spec_examples() {
        // Equal candidates, both < 2 -> {PLANAR, DC, ANGULAR_26}.
        assert_eq!(fill_mpm_candidates(0, 0), [0, 1, 26]);
        // Equal candidates >= 2 -> the mode plus its two angular neighbours (mod 32, offset by 2).
        assert_eq!(fill_mpm_candidates(10, 10), [10, 9, 11]);
    }

    #[test]
    fn rem_mode_skips_sorted_candidates() {
        let mut cand = [0u8, 1, 26];
        cand.sort_unstable();
        // rem=0 should land on the first mode not in {0,1,26}, i.e. 2.
        assert_eq!(rem_mode_from_candidates(0, cand), 2);
    }
}
