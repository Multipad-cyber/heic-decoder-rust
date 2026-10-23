//! Decode straight to RGB8 without assembling the planar picture.
//!
//! [`crate::decode`] builds the whole canvas as `i32` planes, then
//! [`crate::DecodedImage::to_rgb8`] converts it: ~180 MB at the peak for a
//! 12 MP iPhone photo, 3.5 GB for 200 MP. [`decode_rgb8`] decodes a grid one
//! row of tiles at a time and converts each source row as soon as the chroma
//! rows it reads are decoded, then lets them go. Its output is bit-identical
//! to `decode(..)?.to_rgb8_with_options(..)`: the container's crop, rotation
//! and mirroring are resolved into per-axis index maps with the same chroma
//! geometry [`crate::decode`] would produce, and every pixel goes through the
//! same arithmetic, in the same order, as the planar converter.
//!
//! With [`DecodeRgbOptions::max_edge`] the pixels are averaged into a
//! smaller image as they are produced, so only the output is ever held.
//!
//! Files this does not stream — a predicted image, a grid with uneven or
//! odd-sized tiles, an alpha plane to composite — go through
//! [`crate::decode`] instead, under the same limits.

use std::collections::VecDeque;

use rayon::prelude::*;

use crate::color::{Axis, ChromaSampling, PixelRgb, RgbImage, RgbOptions, alpha8, axis};
use crate::error::{HeifError, Result};
use crate::hevc::nal::NAL_SPS;
use crate::hevc::params::parse_sps;
use crate::isobmff::{self, ResolvedImage, Transform};
use crate::{DecodedImage, MAX_IMAGE_PIXELS, chroma_subsampling, crop_axis, decode_one_image, find_and_parse, image_from_picture};

/// What to do with an alpha plane.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AlphaMode {
    /// Leave it out, as [`crate::DecodedImage::to_rgb8`] does. The alpha
    /// image is then not decoded at all.
    #[default]
    Ignore,
    /// Composite the colour over this background, using alpha scaled as
    /// [`crate::DecodedImage::to_rgba8`] scales it.
    Over([u8; 3]),
}

/// Options for [`decode_rgb8_with`].
#[derive(Debug, Clone, Copy)]
pub struct DecodeRgbOptions {
    /// Longest edge of the output. `None`, or an image that already fits,
    /// keeps the full size; otherwise the image is scaled down by averaging
    /// the full-size pixels that fall in each output pixel.
    pub max_edge: Option<usize>,
    pub alpha: AlphaMode,
    /// Colour override and chroma upsampling, as for `to_rgb8_with_options`.
    pub rgb: RgbOptions,
    /// Largest coded picture (one grid tile, or a whole non-grid image with
    /// its alpha and prediction reference), in luma samples as the SPS
    /// codes it, before any cropping. Checked before anything is allocated.
    pub max_picture_pixels: u64,
    /// Largest grid: the assembled canvas, and the sum of its tiles.
    pub max_total_pixels: u64,
}

impl Default for DecodeRgbOptions {
    fn default() -> Self {
        DecodeRgbOptions {
            max_edge: None,
            alpha: AlphaMode::Ignore,
            rgb: RgbOptions::default(),
            max_picture_pixels: MAX_IMAGE_PIXELS,
            max_total_pixels: MAX_IMAGE_PIXELS,
        }
    }
}

/// The primary image as RGB8, at full size: the same pixels as
/// `decode(bytes)?.to_rgb8()`, without holding the planar picture.
pub fn decode_rgb8(bytes: &[u8]) -> Result<RgbImage<u8>> {
    decode_rgb8_with(bytes, &DecodeRgbOptions::default())
}

/// [`decode_rgb8`] with options: a maximum size, alpha compositing, colour
/// overrides and limits.
pub fn decode_rgb8_with(bytes: &[u8], options: &DecodeRgbOptions) -> Result<RgbImage<u8>> {
    let parsed = isobmff::parse(bytes)?;
    let too_large = || HeifError::LimitExceeded("RGB: picture larger than allowed");
    let wants_alpha = matches!(options.alpha, AlphaMode::Over(_)) && parsed.alpha.is_some();

    let mut tile_pixels = Vec::with_capacity(parsed.images.len());
    for image in &parsed.images {
        let pixels = picture_pixels(image)?;
        if pixels > options.max_picture_pixels {
            return Err(too_large());
        }
        tile_pixels.push(pixels);
    }
    if let Some(grid) = parsed.grid {
        let canvas = u64::from(grid.output_width) * u64::from(grid.output_height);
        if canvas > options.max_total_pixels || tile_pixels.iter().sum::<u64>() > options.max_total_pixels {
            return Err(too_large());
        }
    }
    let alpha_pixels = match &parsed.alpha {
        Some(alpha) if wants_alpha => {
            let mut pixels = 0u64;
            for image in &alpha.image.images {
                pixels = pixels.saturating_add(picture_pixels(image)?);
            }
            if let Some(grid) = alpha.image.grid {
                pixels = pixels.saturating_add(u64::from(grid.output_width) * u64::from(grid.output_height));
            }
            pixels
        }
        _ => 0,
    };
    if alpha_pixels > options.max_picture_pixels {
        return Err(too_large());
    }

    if parsed.reference_image.is_some() {
        let mut pixels = tile_pixels.iter().sum::<u64>();
        if let Some(image) = &parsed.reference_image {
            pixels = pixels.saturating_add(picture_pixels(image)?);
        }
        if pixels.saturating_add(alpha_pixels) > options.max_picture_pixels {
            return Err(too_large());
        }
        return through_decode(bytes, options);
    }
    match stream(&parsed, options)? {
        Some(rgb) => Ok(rgb),
        None => through_decode(bytes, options),
    }
}

fn picture_pixels(image: &ResolvedImage) -> Result<u64> {
    let sps = find_and_parse(image.hevc_config.nalus_of_type(NAL_SPS), NAL_SPS, parse_sps)?;
    Ok(u64::from(sps.pic_width_in_luma_samples) * u64::from(sps.pic_height_in_luma_samples))
}

/// `decode` and the planar converter, for the files [`stream`] leaves out.
fn through_decode(bytes: &[u8], options: &DecodeRgbOptions) -> Result<RgbImage<u8>> {
    let image = crate::decode(bytes)?;
    let rgb = match options.alpha {
        AlphaMode::Over(background) if image.alpha.is_some() => {
            let rgba = image.to_rgba8_with_options(&options.rgb)?;
            let data = rgba.data.as_chunks::<4>().0.iter().flat_map(|&[r, g, b, a]| over([r, g, b], a, background)).collect();
            RgbImage { width: rgba.width, height: rgba.height, data }
        }
        _ => image.to_rgb8_with_options(&options.rgb)?,
    };
    drop(image);
    let full = (rgb.width, rgb.height);
    let out = fit_within(full, options.max_edge);
    if out == full {
        return Ok(rgb);
    }
    let maps = [AxisMap::identity(full.0, out.0, 1), AxisMap::identity(full.1, out.1, out.0)];
    let mut sink = Sink::bins(full, out, maps)?;
    for (y, row) in rgb.data.as_chunks::<3>().0.chunks(full.0).enumerate() {
        for (x, px) in row.iter().enumerate() {
            sink.put(x, y, *px);
        }
    }
    Ok(sink.finish())
}

/// `color` over `background` with 8-bit `alpha`.
fn over(color: [u8; 3], alpha: u8, background: [u8; 3]) -> [u8; 3] {
    let a = u32::from(alpha);
    let blend = |c: u8, b: u8| ((u32::from(c) * a + u32::from(b) * (255 - a) + 127) / 255) as u8;
    [blend(color[0], background[0]), blend(color[1], background[1]), blend(color[2], background[2])]
}

/// Width and height scaled down so the longest edge is at most `max_edge`.
fn fit_within((width, height): (usize, usize), max_edge: Option<usize>) -> (usize, usize) {
    let longest = width.max(height);
    match max_edge {
        Some(max_edge) if longest > max_edge => {
            let scale = |side: usize| ((side * max_edge + longest / 2) / longest).max(1);
            (scale(width), scale(height))
        }
        _ => (width, height),
    }
}

/// Where transformed coordinates come from along one source axis:
/// `source = offset ± t`, on the source's x (`from_y == false`) or y axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Affine {
    from_y: bool,
    flip: bool,
    offset: i64,
}

impl Affine {
    fn eval(self, t: usize) -> i64 {
        if self.flip { self.offset - t as i64 } else { self.offset + t as i64 }
    }
    /// The transformed coordinate source coordinate `s` lands on, if it is
    /// inside `0..len`.
    fn invert(self, s: usize, len: usize) -> Option<usize> {
        let t = if self.flip { self.offset - s as i64 } else { s as i64 - self.offset };
        (0..len as i64).contains(&t).then_some(t as usize)
    }
    /// `self(c - t)`: the axis flipped within `0..=c`.
    fn reflected(self, c: usize) -> Self {
        let c = c as i64;
        if self.flip { Affine { flip: false, offset: self.offset - c, ..self } } else { Affine { flip: true, offset: self.offset + c, ..self } }
    }
    /// `self(t + by)`: the axis shifted by a crop.
    fn shifted(self, by: usize) -> Self {
        let by = by as i64;
        Affine { offset: if self.flip { self.offset - by } else { self.offset + by }, ..self }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Op {
    /// One 90° anticlockwise turn.
    Turn,
    FlipX,
    FlipY,
}

/// A plane's transformed raster: its size, and for each transformed axis
/// (`[x, y]`) where its coordinates come from in the source plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PlaneMap {
    width: usize,
    height: usize,
    axes: [Affine; 2],
}

impl PlaneMap {
    fn new(width: usize, height: usize, ops: &[Op], crop: Option<(usize, usize, usize, usize)>) -> Self {
        let mut axes = [Affine { from_y: false, flip: false, offset: 0 }, Affine { from_y: true, flip: false, offset: 0 }];
        let (mut w, mut h) = (width, height);
        for &op in ops {
            match op {
                // rotate90_ccw_plane: dst(x, y) = src(w - 1 - y, x).
                Op::Turn => {
                    axes = [axes[1], axes[0].reflected(w.saturating_sub(1))];
                    (w, h) = (h, w);
                }
                Op::FlipX => axes[0] = axes[0].reflected(w.saturating_sub(1)),
                Op::FlipY => axes[1] = axes[1].reflected(h.saturating_sub(1)),
            }
        }
        if let Some((left, top, cw, ch)) = crop {
            axes = [axes[0].shifted(left), axes[1].shifted(top)];
            (w, h) = (cw, ch);
        }
        PlaneMap { width: w, height: h, axes }
    }

    /// The transformed axis the source's y axis (`true`) or x axis feeds.
    fn axis_of(&self, from_y: bool) -> usize {
        usize::from(self.axes[1].from_y == from_y)
    }

    fn len(&self, axis: usize) -> usize {
        if axis == 0 { self.width } else { self.height }
    }
}

/// The planar picture `decode` would hand to the converter, as geometry:
/// what `apply_transforms` and `extract_window` do to its size and chroma
/// siting, resolved into index maps instead of moved samples.
#[derive(Debug, Clone, Copy)]
struct Geometry {
    luma: PlaneMap,
    chroma: PlaneMap,
    sampling: ChromaSampling,
}

fn geometry(width: usize, height: usize, cb_width: usize, cb_height: usize, format: u32, coded: ChromaSampling, transforms: &[Transform]) -> Result<Geometry> {
    let (mut sub_w, mut sub_h) = chroma_subsampling(format);
    let (mut w, mut h, mut cw, mut ch) = (width, height, cb_width, cb_height);
    let mut window = (0usize, 0usize, width, height);
    let mut sampling = coded;
    let mut ops = Vec::new();
    // As apply_transforms, step by step.
    for &transform in transforms {
        match transform {
            Transform::CleanAperture(aperture) => {
                let (dx, ww) = crop_axis(window.2, aperture.width, aperture.horizontal_offset)?;
                let (dy, wh) = crop_axis(window.3, aperture.height, aperture.vertical_offset)?;
                window = (window.0 + dx, window.1 + dy, ww, wh);
                continue;
            }
            Transform::Rotate90Ccw(turns) => {
                for _ in 0..turns % 4 {
                    window = (window.1, w - window.0 - window.2, window.3, window.2);
                    let s = sampling;
                    sampling = ChromaSampling {
                        sub_x: s.sub_y,
                        sub_y: s.sub_x,
                        origin_x: s.origin_y,
                        origin_y: (w - 1) as f32 - s.origin_x - cw.saturating_sub(1) as f32 * s.sub_x as f32,
                    };
                    if format == 0 {
                        sampling = ChromaSampling::coded(0, None);
                    }
                    ops.push(Op::Turn);
                    (w, h, cw, ch) = (h, w, ch, cw);
                    std::mem::swap(&mut sub_w, &mut sub_h);
                }
                continue;
            }
            Transform::MirrorHorizontal => {
                window.0 = w - window.0 - window.2;
                sampling.origin_x = (w - 1) as f32 - sampling.origin_x - cw.saturating_sub(1) as f32 * sampling.sub_x as f32;
                ops.push(Op::FlipX);
            }
            Transform::MirrorVertical => {
                window.1 = h - window.1 - window.3;
                sampling.origin_y = (h - 1) as f32 - sampling.origin_y - ch.saturating_sub(1) as f32 * sampling.sub_y as f32;
                ops.push(Op::FlipY);
            }
        }
        if format == 0 {
            sampling = ChromaSampling::coded(0, None);
        }
    }
    let (luma_crop, chroma_crop) = if window == (0, 0, w, h) {
        (None, None)
    } else {
        let (left, top, ww, wh) = window;
        let chroma = if format == 0 { (0, 0, 0, 0) } else { (left / sub_w, top / sub_h, ww.div_ceil(sub_w), wh.div_ceil(sub_h)) };
        sampling.origin_x -= (left % sub_w) as f32;
        sampling.origin_y -= (top % sub_h) as f32;
        (Some(window), Some(chroma))
    };
    Ok(Geometry { luma: PlaneMap::new(width, height, &ops, luma_crop), chroma: PlaneMap::new(cb_width, cb_height, &ops, chroma_crop), sampling })
}

/// The checks the planar converter makes on the picture it is given; a
/// picture failing them goes through `decode`, which then fails the same way.
fn converter_accepts(g: &Geometry, format: u32) -> bool {
    let s = g.sampling;
    let monochrome = format == 0;
    let valid_subsampling = match format {
        0 => (s.sub_x, s.sub_y) == (1, 1),
        1 => (s.sub_x, s.sub_y) == (2, 2),
        2 => matches!((s.sub_x, s.sub_y), (2, 1) | (1, 2)),
        3 => (s.sub_x, s.sub_y) == (1, 1),
        _ => false,
    };
    let shape = if monochrome {
        (g.chroma.width, g.chroma.height) == (0, 0)
    } else {
        g.chroma.width == g.luma.width.div_ceil(s.sub_x as usize)
            && g.chroma.height == g.luma.height.div_ceil(s.sub_y as usize)
            && g.chroma.width > 0
            && g.chroma.height > 0
    };
    g.luma.width > 0
        && g.luma.height > 0
        && (g.luma.width as u64) * (g.luma.height as u64) <= MAX_IMAGE_PIXELS
        && valid_subsampling
        && shape
        && s.origin_x.is_finite()
        && s.origin_y.is_finite()
        && s.origin_x.abs() <= 2.0
        && s.origin_y.abs() <= 2.0
}

/// One source axis, resolved: for each source coordinate, where it lands
/// on the transformed axis it feeds and which chroma samples it reads there.
struct SourceAxis {
    /// The transformed axis this source axis feeds (0 = x).
    axis: usize,
    entries: Vec<Option<Entry>>,
}

#[derive(Clone, Copy)]
struct Entry {
    /// Transformed coordinate.
    t: usize,
    /// Source chroma indices (along this same source axis) of the two
    /// samples the converter interpolates, and the weight of the second.
    lo: usize,
    hi: usize,
    weight: f32,
}

fn source_axis(len: usize, from_y: bool, g: &Geometry, tables: &[Vec<Axis>; 2]) -> SourceAxis {
    let axis = g.luma.axis_of(from_y);
    let luma = g.luma.axes[axis];
    let chroma = g.chroma.axes[axis];
    let tlen = g.luma.len(axis);
    let entries = (0..len)
        .map(|s| {
            let t = luma.invert(s, tlen)?;
            let Some(a) = tables[axis].get(t) else { return Some(Entry { t, lo: 0, hi: 0, weight: 0.0 }) };
            Some(Entry { t, lo: chroma.eval(a.lo) as usize, hi: chroma.eval(a.hi) as usize, weight: a.weight })
        })
        .collect();
    SourceAxis { axis, entries }
}

/// Where converted pixels go: straight into the full-size image, or
/// averaged into a smaller one.
enum Sink {
    Full { width: usize, data: Vec<u8> },
    Bins(Bins),
}

impl Sink {
    fn full(width: usize, height: usize) -> Result<Self> {
        let len = width.checked_mul(height).and_then(|p| p.checked_mul(3)).ok_or(HeifError::LimitExceeded("RGB: output size"))?;
        let mut data = Vec::new();
        data.try_reserve_exact(len).map_err(|_| HeifError::LimitExceeded("RGB: output allocation failed"))?;
        data.resize(len, 0);
        Ok(Sink::Full { width, data })
    }

    fn bins(full: (usize, usize), out: (usize, usize), maps: [AxisMap; 2]) -> Result<Self> {
        Ok(Sink::Bins(Bins::new(full, out, maps)?))
    }

    #[inline(always)]
    fn put(&mut self, x: usize, y: usize, rgb: [u8; 3]) {
        match self {
            Sink::Full { width, data } => {
                let i = (y * *width + x) * 3;
                data[i..i + 3].copy_from_slice(&rgb);
            }
            Sink::Bins(bins) => bins.add(x, y, rgb),
        }
    }

    /// One source row's pixels, in source order. Into bins, neighbouring
    /// pixels that land in the same bin are summed first and added once.
    fn put_row(&mut self, pixels: impl Iterator<Item = (usize, usize, [u8; 3])>) {
        match self {
            Sink::Full { width, data } => {
                for (x, y, rgb) in pixels {
                    let i = (y * *width + x) * 3;
                    data[i..i + 3].copy_from_slice(&rgb);
                }
            }
            Sink::Bins(bins) => {
                let mut run: Option<(usize, [u32; 3])> = None;
                for (x, y, rgb) in pixels {
                    let bin = bins.maps[0].parts[x] + bins.maps[1].parts[y];
                    match &mut run {
                        Some((b, sum)) if *b == bin => sum.iter_mut().zip(rgb).for_each(|(s, c)| *s += u32::from(c)),
                        _ => {
                            if let Some((b, sum)) = run {
                                bins.add_sum(b, sum);
                            }
                            run = Some((bin, rgb.map(u32::from)));
                        }
                    }
                }
                if let Some((b, sum)) = run {
                    bins.add_sum(b, sum);
                }
            }
        }
    }

    fn finish(self) -> RgbImage<u8> {
        match self {
            Sink::Full { width, data } => RgbImage { width, height: data.len() / 3 / width.max(1), data },
            Sink::Bins(bins) => bins.finish(),
        }
    }
}

/// Output bin of each transformed coordinate along one axis, already
/// scaled for its axis (`bin` for x, `bin * out_width` for y).
struct AxisMap {
    parts: Vec<usize>,
    counts: Vec<u32>,
}

impl AxisMap {
    fn identity(full: usize, out: usize, stride: usize) -> Self {
        let mut counts = vec![0u32; out];
        let parts = (0..full)
            .map(|t| {
                let bin = t * out / full;
                counts[bin] += 1;
                bin * stride
            })
            .collect();
        AxisMap { parts, counts }
    }
}

/// Sums per output channel: `u16` while a bin's total cannot overflow it,
/// else `u32`.
enum Sums {
    Narrow(Vec<u16>),
    Wide(Vec<u32>),
}

/// Box-filter accumulator over the transformed image.
struct Bins {
    out: (usize, usize),
    maps: [AxisMap; 2],
    sums: Sums,
}

impl Bins {
    fn new(_full: (usize, usize), out: (usize, usize), maps: [AxisMap; 2]) -> Result<Self> {
        let channels = out.0.checked_mul(out.1).and_then(|p| p.checked_mul(3)).ok_or(HeifError::LimitExceeded("RGB: output size"))?;
        let widest = |map: &AxisMap| map.counts.iter().copied().max().unwrap_or(0);
        let area = u64::from(widest(&maps[0])) * u64::from(widest(&maps[1]));
        let failed = |_| HeifError::LimitExceeded("RGB: output allocation failed");
        let sums = if area * 255 <= u64::from(u16::MAX) {
            let mut sums = Vec::new();
            sums.try_reserve_exact(channels).map_err(failed)?;
            sums.resize(channels, 0);
            Sums::Narrow(sums)
        } else {
            let mut sums = Vec::new();
            sums.try_reserve_exact(channels).map_err(failed)?;
            sums.resize(channels, 0);
            Sums::Wide(sums)
        };
        Ok(Bins { out, maps, sums })
    }

    #[inline(always)]
    fn add(&mut self, x: usize, y: usize, rgb: [u8; 3]) {
        let i = (self.maps[0].parts[x] + self.maps[1].parts[y]) * 3;
        match &mut self.sums {
            Sums::Narrow(s) => s[i..i + 3].iter_mut().zip(rgb).for_each(|(s, c)| *s += u16::from(c)),
            Sums::Wide(s) => s[i..i + 3].iter_mut().zip(rgb).for_each(|(s, c)| *s += u32::from(c)),
        }
    }

    /// Adds a run's summed channels to output pixel `bin`.
    #[inline(always)]
    fn add_sum(&mut self, bin: usize, sum: [u32; 3]) {
        let i = bin * 3;
        match &mut self.sums {
            // A run is part of one bin, so it fits whatever the bin fits.
            Sums::Narrow(s) => s[i..i + 3].iter_mut().zip(sum).for_each(|(s, c)| *s += c as u16),
            Sums::Wide(s) => s[i..i + 3].iter_mut().zip(sum).for_each(|(s, c)| *s += c),
        }
    }

    fn finish(self) -> RgbImage<u8> {
        let (width, height) = self.out;
        let mut data = Vec::with_capacity(width * height * 3);
        for (y, &row) in self.maps[1].counts.iter().enumerate() {
            for (x, &column) in self.maps[0].counts.iter().enumerate() {
                let n = row * column;
                let i = (y * width + x) * 3;
                for c in 0..3 {
                    let sum = match &self.sums {
                        Sums::Narrow(s) => u32::from(s[i + c]),
                        Sums::Wide(s) => s[i + c],
                    };
                    // Every bin covers at least one pixel; white if one did not.
                    data.push((sum + n / 2).checked_div(n).map_or(255, |m| m as u8));
                }
            }
        }
        RgbImage { width, height, data }
    }
}

/// What every tile must share with the first, and what the converter
/// takes from it.
#[derive(Clone, Copy)]
struct TileShape {
    width: usize,
    height: usize,
    format: u32,
    bit_depth_luma: u32,
    bit_depth_chroma: u32,
}

impl TileShape {
    fn of(tile: &DecodedImage) -> Self {
        TileShape {
            width: tile.width,
            height: tile.height,
            format: tile.chroma_format_idc,
            bit_depth_luma: tile.bit_depth_luma,
            bit_depth_chroma: tile.bit_depth_chroma,
        }
    }
}

/// One row of tiles laid out as source rows, as `blit_tiles` places them on
/// the canvas: zeros where no tile reaches. Samples fit `u16` (bit depths
/// are at most 16), half the memory of the decoder's `i32`.
struct Band {
    luma: Vec<Vec<u16>>,
    cb: Vec<Vec<u16>>,
    cr: Vec<Vec<u16>>,
}

/// The canvas layout bands are cut from.
#[derive(Clone, Copy)]
struct Layout {
    tile: TileShape,
    width: usize,
    height: usize,
    cb_width: usize,
    sub_w: usize,
    sub_h: usize,
}

impl Layout {
    fn band(&self, index: usize) -> Band {
        let rows = self.tile.height.min(self.height.saturating_sub(index * self.tile.height));
        let c_rows = if self.tile.format == 0 { 0 } else { rows.div_ceil(self.sub_h) };
        Band { luma: vec![vec![0; self.width]; rows], cb: vec![vec![0; self.cb_width]; c_rows], cr: vec![vec![0; self.cb_width]; c_rows] }
    }

    /// Copies the tile in column `column` into `band`.
    fn place(&self, band: &mut Band, column: usize, tile: &DecodedImage) {
        let x0 = column * self.tile.width;
        let w = self.tile.width.min(self.width.saturating_sub(x0));
        if w == 0 {
            return;
        }
        let narrow = |dst: &mut [u16], src: &[i32]| dst.iter_mut().zip(src).for_each(|(d, &s)| *d = s as u16);
        for (r, row) in band.luma.iter_mut().enumerate() {
            narrow(&mut row[x0..x0 + w], &tile.y[r * tile.width..][..w]);
        }
        let (cx, cw) = (x0 / self.sub_w, w.div_ceil(self.sub_w));
        for (r, (row_cb, row_cr)) in band.cb.iter_mut().zip(band.cr.iter_mut()).enumerate() {
            narrow(&mut row_cb[cx..cx + cw], &tile.cb[r * tile.cb_width..][..cw]);
            narrow(&mut row_cr[cx..cx + cw], &tile.cr[r * tile.cb_width..][..cw]);
        }
    }
}

/// Source rows held for conversion, `first..first + rows.len()`.
struct Rows {
    first: usize,
    rows: VecDeque<Vec<u16>>,
}

impl Rows {
    fn new() -> Self {
        Rows { first: 0, rows: VecDeque::new() }
    }
    fn end(&self) -> usize {
        self.first + self.rows.len()
    }
    fn get(&self, row: usize) -> &[u16] {
        row.checked_sub(self.first).and_then(|i| self.rows.get(i)).map_or(&[], Vec::as_slice)
    }
    fn drop_below(&mut self, row: usize) {
        while self.first < row && !self.rows.is_empty() {
            self.rows.pop_front();
            self.first += 1;
        }
    }
}

fn decode_tile(image: &ResolvedImage) -> Result<DecodedImage> {
    decode_one_image(image).and_then(|(sps, pic)| image_from_picture(&sps, pic))
}

/// Rows converted per parallel batch.
const ROWS_PER_BATCH: usize = 64;

/// Stream-decodes `parsed` to RGB, or `None` when the file needs `decode`'s
/// general path.
fn stream(parsed: &isobmff::ParsedHeif, options: &DecodeRgbOptions) -> Result<Option<RgbImage<u8>>> {
    let (columns, rows) = match parsed.grid {
        Some(grid) => {
            if parsed.images.len() != (grid.rows as usize) * (grid.columns as usize) {
                return Err(HeifError::MalformedBox("grid image count does not match rows*columns"));
            }
            (grid.columns as usize, grid.rows as usize)
        }
        None if parsed.images.len() == 1 => (1, 1),
        None => return Ok(None),
    };
    // The first tile tells the layout; it is placed like the others.
    let first = decode_tile(&parsed.images[0])?;
    let shape = TileShape::of(&first);
    let (width, height) = match parsed.grid {
        Some(grid) => (grid.output_width as usize, grid.output_height as usize),
        None => (first.width, first.height),
    };
    let (sub_w, sub_h) = chroma_subsampling(shape.format);
    // Tiles on chroma boundaries stack into whole chroma rows, as the grid
    // assembly lays them out.
    if !shape.width.is_multiple_of(sub_w) || !shape.height.is_multiple_of(sub_h) {
        return Ok(None);
    }
    let (cb_width, cb_height) = if shape.format == 0 { (0, 0) } else { (width.div_ceil(sub_w), height.div_ceil(sub_h)) };
    let g = geometry(width, height, cb_width, cb_height, shape.format, first.chroma_sampling, &parsed.transforms)?;
    if !converter_accepts(&g, shape.format) {
        return Ok(None);
    }
    let info = options.rgb.color.or(parsed.metadata.color_info).or(first.color_info).unwrap_or_default();
    let pixel = PixelRgb::new(info, shape.format, g.sampling, shape.bit_depth_luma, shape.bit_depth_chroma)?;
    let layout = Layout { tile: shape, width, height, cb_width, sub_w, sub_h };

    // The converter's chroma tables over the transformed picture.
    let tables = if pixel.monochrome() {
        [Vec::new(), Vec::new()]
    } else {
        [
            axis(g.luma.width, g.chroma.width, g.sampling.sub_x, g.sampling.origin_x, options.rgb.upsampling)?,
            axis(g.luma.height, g.chroma.height, g.sampling.sub_y, g.sampling.origin_y, options.rgb.upsampling)?,
        ]
    };
    let columns_axis = source_axis(width, false, &g, &tables);
    let rows_axis = source_axis(height, true, &g, &tables);
    let turned = columns_axis.axis == 1;
    let column_arrays = Columns::new(&columns_axis);

    // Chroma rows each source row reads, and the lowest any later row reads.
    let needs: Vec<Option<(usize, usize)>> =
        if pixel.monochrome() { vec![None; height] } else { rows_axis.entries.iter().map(|e| e.map(|e| (e.lo.min(e.hi), e.lo.max(e.hi)))).collect() };
    let mut keep_from = vec![usize::MAX; height + 1];
    for y in (0..height).rev() {
        keep_from[y] = keep_from[y + 1].min(needs[y].map_or(usize::MAX, |n| n.0));
    }

    let alpha = match (options.alpha, &parsed.alpha) {
        (AlphaMode::Over(background), Some(auxiliary)) => {
            let plane = crate::decode_pixels(&auxiliary.image, true)?;
            if (plane.width, plane.height) != (g.luma.width, g.luma.height) {
                return Err(HeifError::Unsupported("alpha: display dimensions do not match colour image"));
            }
            let max = (1u32 << plane.bit_depth_luma) - 1;
            Some((plane.y, plane.width, max, background))
        }
        _ => None,
    };

    let full = (g.luma.width, g.luma.height);
    let out = fit_within(full, options.max_edge);
    let mut sink = if out == full {
        Sink::full(full.0, full.1)?
    } else {
        Sink::bins(full, out, [AxisMap::identity(full.0, out.0, 1), AxisMap::identity(full.1, out.1, out.0)])?
    };

    let (mut luma, mut cb, mut cr) = (Rows::new(), Rows::new(), Rows::new());
    let mut first = Some(first);
    let threads = rayon::current_num_threads().max(1);
    let mut next_y = 0;
    for band_index in 0..rows {
        // Decode the row of tiles a few at a time (one per thread), each
        // placed in the band as soon as it is decoded.
        let mut band = layout.band(band_index);
        let tiles = &parsed.images[band_index * columns..][..columns];
        let mut column = 0;
        if let Some(tile) = first.take() {
            layout.place(&mut band, 0, &tile);
            column = 1;
        }
        while column < columns {
            let end = (column + threads).min(columns);
            let decoded: Vec<DecodedImage> = tiles[column..end].par_iter().map(decode_tile).collect::<Result<_>>()?;
            for (offset, tile) in decoded.iter().enumerate() {
                let s = TileShape::of(tile);
                if (s.format, s.bit_depth_luma, s.bit_depth_chroma) != (shape.format, shape.bit_depth_luma, shape.bit_depth_chroma) {
                    return Err(HeifError::Unsupported("decode: grid tiles with inconsistent bit depth/chroma format are not supported"));
                }
                if (s.width, s.height) != (shape.width, shape.height) {
                    return Ok(None);
                }
                layout.place(&mut band, column + offset, tile);
            }
            column = end;
        }
        luma.rows.extend(band.luma);
        cb.rows.extend(band.cb);
        cr.rows.extend(band.cr);
        let finished = band_index + 1 == rows;

        // Every source row whose chroma is all here, in order, in batches.
        let mut ready = next_y;
        while ready < luma.end() && needs[ready].is_none_or(|(_, hi)| finished || hi < cb.end()) {
            ready += 1;
        }
        for batch in (next_y..ready).step_by(ROWS_PER_BATCH) {
            let batch_end = (batch + ROWS_PER_BATCH).min(ready);
            let converted: Vec<Vec<[u8; 3]>> =
                (batch..batch_end).into_par_iter().map(|y| convert_row(y, &luma, &cb, &cr, &column_arrays, &rows_axis, turned, &pixel)).collect();
            for (y, row) in (batch..batch_end).zip(converted) {
                let Some(ry) = rows_axis.entries[y] else { continue };
                let pixels = columns_axis.entries.iter().zip(row).filter_map(|(cx, rgb)| {
                    let cx = (*cx)?;
                    let (tx, ty) = if turned { (ry.t, cx.t) } else { (cx.t, ry.t) };
                    let rgb = match &alpha {
                        Some((samples, alpha_width, max, background)) => over(rgb, alpha8(samples[ty * alpha_width + tx] as u32, *max), *background),
                        None => rgb,
                    };
                    Some((tx, ty, rgb))
                });
                sink.put_row(pixels);
            }
        }
        next_y = ready;
        luma.drop_below(next_y);
        let keep = keep_from[next_y.min(height)];
        cb.drop_below(keep.min(cb.end()));
        cr.drop_below(keep.min(cr.end()));
    }
    if next_y != height {
        return Err(HeifError::MalformedHevc("RGB: rows left unconverted"));
    }
    Ok(Some(sink.finish()))
}

/// The source columns as flat arrays, for the row loops: the chroma
/// columns each reads and their weight (0 for a column a crop drops).
struct Columns {
    lo: Vec<usize>,
    hi: Vec<usize>,
    weight: Vec<f32>,
}

impl Columns {
    fn new(axis: &SourceAxis) -> Self {
        let get = |f: fn(&Entry) -> usize| axis.entries.iter().map(|e| e.as_ref().map_or(0, f)).collect();
        Columns { lo: get(|e| e.lo), hi: get(|e| e.hi), weight: axis.entries.iter().map(|e| e.map_or(0.0, |e| e.weight)).collect() }
    }
}

/// One source row through the converter's arithmetic, laid out by rows:
/// the same expressions as `ChromaRows::slot`/`blend` and `Matrix::row`,
/// evaluated once per value. Every index comes from the geometry, not the
/// data, and lies inside the rows held; a miss would read 0, not panic.
#[allow(clippy::too_many_arguments)]
fn convert_row(y: usize, luma: &Rows, cb: &Rows, cr: &Rows, columns: &Columns, rows: &SourceAxis, turned: bool, pixel: &PixelRgb) -> Vec<[u8; 3]> {
    let samples = luma.get(y);
    let Some(ry) = rows.entries[y] else { return Vec::new() };
    let mut out = vec![[0u8; 3]; samples.len()];
    if pixel.monochrome() {
        for (o, &s) in out.iter_mut().zip(samples) {
            *o = pixel.rgb8(i32::from(s), 0.0, 0.0);
        }
        return out;
    }
    let at = |row: &[u16], i: usize| i32::from(row.get(i).copied().unwrap_or(0));
    let (cb_lo, cb_hi, cr_lo, cr_hi) = (cb.get(ry.lo), cb.get(ry.hi), cr.get(ry.lo), cr.get(ry.hi));
    if turned {
        // The transformed picture's chroma rows run along source columns:
        // its x axis comes from this source row, so the "horizontal"
        // interpolation is between this row's two chroma rows, the same
        // for every source chroma column. Then the vertical one, per pixel.
        let across =
            |lo: &[u16], hi: &[u16]| -> Vec<f32> { (0..lo.len().max(hi.len())).map(|c| PixelRgb::lerp_samples(at(lo, c), at(hi, c), ry.weight)).collect() };
        let (h_cb, h_cr) = (across(cb_lo, cb_hi), across(cr_lo, cr_hi));
        let pick = |h: &[f32], i: usize| h.get(i).copied().unwrap_or(0.0);
        for ((((o, &s), &lo), &hi), &w) in out.iter_mut().zip(samples).zip(&columns.lo).zip(&columns.hi).zip(&columns.weight) {
            let cbv = PixelRgb::lerp_rows(pick(&h_cb, lo), pick(&h_cb, hi), w);
            let crv = PixelRgb::lerp_rows(pick(&h_cr, lo), pick(&h_cr, hi), w);
            *o = pixel.rgb8(i32::from(s), cbv, crv);
        }
    } else {
        // Each chroma row interpolated to this row's width, then blended.
        let along = |row: &[u16]| -> Vec<f32> {
            columns.lo.iter().zip(&columns.hi).zip(&columns.weight).map(|((&lo, &hi), &w)| PixelRgb::lerp_samples(at(row, lo), at(row, hi), w)).collect()
        };
        let (cb_t, cb_b, cr_t, cr_b) = (along(cb_lo), along(cb_hi), along(cr_lo), along(cr_hi));
        for (((((o, &s), &ct), &cbm), &rt), &rb) in out.iter_mut().zip(samples).zip(&cb_t).zip(&cb_b).zip(&cr_t).zip(&cr_b) {
            *o = pixel.rgb8(i32::from(s), PixelRgb::lerp_rows(ct, cbm, ry.weight), PixelRgb::lerp_rows(rt, rb, ry.weight));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_maps_follow_turns_flips_and_crops() {
        // 4×2, one anticlockwise turn: transformed (0, 0) is source (3, 0).
        let m = PlaneMap::new(4, 2, &[Op::Turn], None);
        assert_eq!((m.width, m.height), (2, 4));
        let src = |m: &PlaneMap, x: usize, y: usize| {
            let (a, b) = (m.axes[0].eval(x), m.axes[1].eval(y));
            if m.axes[0].from_y { (b, a) } else { (a, b) }
        };
        assert_eq!(src(&m, 0, 0), (3, 0));
        assert_eq!(src(&m, 1, 0), (3, 1));
        assert_eq!(src(&m, 0, 3), (0, 0));
        let m = PlaneMap::new(4, 2, &[Op::FlipX, Op::FlipY], Some((1, 0, 2, 2)));
        assert_eq!(src(&m, 0, 0), (2, 1));
        assert_eq!(m.axes[0].invert(2, 2), Some(0));
        assert_eq!(m.axes[0].invert(0, 2), None);
    }

    #[test]
    fn fits_within_and_never_scales_up() {
        assert_eq!(fit_within((4032, 3024), Some(1568)), (1568, 1176));
        assert_eq!(fit_within((800, 600), Some(1568)), (800, 600));
        assert_eq!(fit_within((800, 600), None), (800, 600));
        assert_eq!(fit_within((100_000, 10), Some(1568)), (1568, 1));
    }

    #[test]
    fn averaging_bins_cover_the_whole_output() {
        let (full, out) = ((5, 3), fit_within((5, 3), Some(2)));
        assert_eq!(out, (2, 1));
        let mut bins = Bins::new(full, out, [AxisMap::identity(5, 2, 1), AxisMap::identity(3, 1, 2)]).unwrap();
        for y in 0..3 {
            for x in 0..5 {
                bins.add(x, y, [if x < 3 { 0 } else { 200 }, 10, 255]);
            }
        }
        assert_eq!(bins.finish().data, [0, 10, 255, 200, 10, 255]);
    }
}
