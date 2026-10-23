//! Optional conversion from native planar samples to interleaved RGB.
//!
//! The matrix/range equations and chroma locations follow ITU-T H.273.
//! Output retains the source transfer function and primaries: this is not
//! an ICC transform, sRGB conversion, linearisation or HDR tone mapping.
use crate::{DecodedImage, HeifError, MAX_IMAGE_PIXELS, Result};
use rayon::prelude::*;

/// Coding-independent colour information from `colr/nclx` or SPS VUI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColorInfo {
    pub color_primaries: u16,
    pub transfer_characteristics: u16,
    pub matrix_coefficients: u16,
    pub full_range: bool,
}

impl Default for ColorInfo {
    /// Unlabelled images use BT.601 limited range, as libheif does (and
    /// MIAF's default matrix). Primaries and transfer remain unspecified;
    /// this does not declare sRGB.
    fn default() -> Self {
        Self { color_primaries: 2, transfer_characteristics: 2, matrix_coefficients: 6, full_range: false }
    }
}

/// Position of the returned chroma raster relative to the luma raster.
/// Positions are in luma pixels, with luma sample (0,0) at the origin.
/// Crops can move the first chroma sample outside the luma output window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChromaSampling {
    pub sub_x: u32,
    pub sub_y: u32,
    pub origin_x: f32,
    pub origin_y: f32,
}

impl ChromaSampling {
    pub(crate) fn coded(format: u32, location: Option<u32>) -> Self {
        let (sub_x, sub_y) = match format {
            1 => (2, 2),
            2 => (2, 1),
            _ => (1, 1),
        };
        // H.265's inferred 4:2:0 location is type 0 (left, vertically centred).
        let (origin_x, origin_y) = if format == 1 {
            match location.unwrap_or(0) {
                1 => (0.5, 0.5),
                2 => (0.0, 0.0),
                3 => (0.5, 0.0),
                4 => (0.0, 1.0),
                5 => (0.5, 1.0),
                _ => (0.0, 0.5),
            }
        } else {
            (0.0, 0.0)
        };
        Self { sub_x, sub_y, origin_x, origin_y }
    }
}

/// Chroma reconstruction filter, with edge samples extended at boundaries.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ChromaUpsampling {
    Nearest,
    #[default]
    Bilinear,
}

/// Optional overrides for applications with external colour information.
#[derive(Debug, Clone, Copy, Default)]
pub struct RgbOptions {
    /// Override the file's matrix and range; `None` uses file information.
    pub color: Option<ColorInfo>,
    pub upsampling: ChromaUpsampling,
}

/// Tightly packed RGB triples, row-major. `u16` values span 0..=65535,
/// preserving higher-depth input without an intermediate 8-bit conversion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbImage<T> {
    pub width: usize,
    pub height: usize,
    pub data: Vec<T>,
}

/// Tightly packed RGBA. Alpha spans the entire destination range, independently
/// of the colour range. `premultiplied` preserves the container's `prem` flag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbaImage<T> {
    pub width: usize,
    pub height: usize,
    pub data: Vec<T>,
    pub premultiplied: bool,
}

impl DecodedImage {
    /// Convert to RGB8 using file colour information and bilinear chroma.
    /// Alpha is omitted; premultiplied colour is not unpremultiplied.
    pub fn to_rgb8(&self) -> Result<RgbImage<u8>> {
        self.to_rgb8_with_options(&RgbOptions::default())
    }
    /// Convert to RGB16 using file colour information and bilinear chroma.
    /// Alpha is omitted; premultiplied colour is not unpremultiplied.
    pub fn to_rgb16(&self) -> Result<RgbImage<u16>> {
        self.to_rgb16_with_options(&RgbOptions::default())
    }
    pub fn to_rgb8_with_options(&self, options: &RgbOptions) -> Result<RgbImage<u8>> {
        convert::<_, 3>(self, options)
    }
    pub fn to_rgb16_with_options(&self, options: &RgbOptions) -> Result<RgbImage<u16>> {
        convert::<_, 3>(self, options)
    }
    /// Convert to RGBA8, using an opaque alpha channel when none is present.
    pub fn to_rgba8(&self) -> Result<RgbaImage<u8>> {
        self.to_rgba8_with_options(&RgbOptions::default())
    }
    /// Convert to RGBA16 without an intermediate 8-bit colour or alpha plane.
    pub fn to_rgba16(&self) -> Result<RgbaImage<u16>> {
        self.to_rgba16_with_options(&RgbOptions::default())
    }
    pub fn to_rgba8_with_options(&self, options: &RgbOptions) -> Result<RgbaImage<u8>> {
        convert_rgba(self, options)
    }
    pub fn to_rgba16_with_options(&self, options: &RgbOptions) -> Result<RgbaImage<u16>> {
        convert_rgba(self, options)
    }
}

trait RgbSample: Copy + Default + Send + Sync {
    fn quantize(value: f32) -> Self;
    fn alpha(sample: u32, max: u32) -> Self;
}
impl RgbSample for u8 {
    fn alpha(sample: u32, max: u32) -> Self {
        ((u64::from(sample) * 255 + u64::from(max / 2)) / u64::from(max)) as u8
    }
    fn quantize(value: f32) -> Self {
        if HAS_ROUND {
            // `as` saturates: below zero and NaN become 0, above 255 becomes 255.
            (value * 255.0).round() as u8
        } else {
            quantize_without_round(value, 255.0, 255) as u8
        }
    }
}
impl RgbSample for u16 {
    fn alpha(sample: u32, max: u32) -> Self {
        ((u64::from(sample) * 65535 + u64::from(max / 2)) / u64::from(max)) as u16
    }
    fn quantize(value: f32) -> Self {
        if HAS_ROUND { (value * 65535.0).round() as u16 } else { quantize_without_round(value, 65535.0, 65535) as u16 }
    }
}

/// Whether `f32::round` is an instruction here. Baseline x86-64 (SSE2) has
/// none, so `round` is a `roundf` call per channel and the loop does not
/// vectorize; there [`quantize_without_round`] is used instead.
const HAS_ROUND: bool = cfg!(any(target_arch = "aarch64", target_feature = "sse4.1"));

/// `value * scale` rounded half away from zero and clamped to `0..=max`,
/// NaN to 0: the same as `(value * scale).round() as u8` (or `u16`) for every
/// `f32`, all of which were compared, on ARM and on x86-64. Neither `round`
/// nor a float-to-integer cast vectorizes with SSE2 only. Instead, adding 2^23
/// rounds the clamped value to an integer (ties to even), which the sum's
/// mantissa then holds; a tie rounded down is moved up.
#[inline(always)]
fn quantize_without_round(value: f32, scale: f32, max: u32) -> u32 {
    const TWO_23: f32 = 8_388_608.0;
    let clamped = (value * scale).max(0.0).min(max as f32);
    let shifted = clamped + TWO_23;
    let nearest = shifted.to_bits() - TWO_23.to_bits();
    nearest + u32::from(clamped - (shifted - TWO_23) == 0.5)
}

struct Axis {
    lo: usize,
    hi: usize,
    weight: f32,
}
fn axis(count: usize, samples: usize, sub: u32, origin: f32, filter: ChromaUpsampling) -> Result<Vec<Axis>> {
    let mut result = Vec::new();
    result.try_reserve_exact(count).map_err(|_| HeifError::LimitExceeded("RGB: sampling allocation failed"))?;
    result.extend((0..count).map(|pixel| {
        // Use f64 for coordinates: f32 rounds large raster indices and
        // could otherwise round the last valid index up to `samples`.
        let position = ((pixel as f64 - f64::from(origin)) / f64::from(sub)).clamp(0.0, (samples - 1) as f64);
        if filter == ChromaUpsampling::Nearest {
            let index = position.round() as usize;
            Axis { lo: index, hi: index, weight: 0.0 }
        } else {
            let lo = position.floor() as usize;
            Axis { lo, hi: (lo + 1).min(samples - 1), weight: (position - lo as f64) as f32 }
        }
    }));
    Ok(result)
}

struct Matrix {
    identity: bool,
    y_offset: f32,
    y_scale: f32,
    c_offset: f32,
    c_scale: f32,
    red_cr: f32,
    blue_cb: f32,
    green_cb: f32,
    green_cr: f32,
}
impl Matrix {
    fn new(info: ColorInfo, y_depth: u32, c_depth: u32) -> Result<Self> {
        let (kr, kb) = match info.matrix_coefficients {
            0 => (0.0, 0.0),
            1 => (0.2126, 0.0722),
            4 => (0.30, 0.11),
            // Unspecified (2) as libheif reads it: BT.601.
            2 | 5 | 6 => (0.299, 0.114),
            7 => (0.212, 0.087),
            9 => (0.2627, 0.0593),
            _ => return Err(HeifError::Unsupported("RGB: unsupported matrix coefficients")),
        };
        let y_shift = (1u32 << (y_depth - 8)) as f32;
        let c_shift = (1u32 << (c_depth - 8)) as f32;
        let (y_offset, y_scale, c_offset, c_scale) = if info.full_range {
            (0.0, 1.0 / ((1u32 << y_depth) - 1) as f32, (1u32 << (c_depth - 1)) as f32, 1.0 / ((1u32 << c_depth) - 1) as f32)
        } else {
            (16.0 * y_shift, 1.0 / (219.0 * y_shift), 128.0 * c_shift, 1.0 / (224.0 * c_shift))
        };
        let identity = info.matrix_coefficients == 0;
        // Identity coding stores G, B, R in the nominal Y, Cb, Cr planes, so
        // Cb and Cr are scaled like luma, at their own depth.
        let (c_offset, c_scale) = match (identity, info.full_range) {
            (false, _) => (c_offset, c_scale),
            (true, true) => (0.0, 1.0 / ((1u32 << c_depth) - 1) as f32),
            (true, false) => (16.0 * c_shift, 1.0 / (219.0 * c_shift)),
        };
        let kg = 1.0 - kr - kb;
        Ok(Self {
            identity,
            y_offset,
            y_scale,
            c_offset,
            c_scale,
            red_cr: 2.0 * (1.0 - kr),
            blue_cb: 2.0 * (1.0 - kb),
            green_cb: -2.0 * kb * (1.0 - kb) / kg,
            green_cr: -2.0 * kr * (1.0 - kr) / kg,
        })
    }
    /// Converts one row from luma and chroma at full width. The identity
    /// test sits outside the loops so that each one vectorizes.
    fn row<T: RgbSample, const CHANNELS: usize>(&self, pixels: &mut [[T; CHANNELS]], luma: &[i32], cb: &[f32], cr: &[f32]) {
        if self.identity {
            for (((rgb, &y), &cb), &cr) in pixels.iter_mut().zip(luma).zip(cb).zip(cr) {
                rgb[0] = T::quantize((cr - self.c_offset) * self.c_scale);
                rgb[1] = T::quantize((y as f32 - self.y_offset) * self.y_scale);
                rgb[2] = T::quantize((cb - self.c_offset) * self.c_scale);
            }
        } else {
            for (((rgb, &y), &cb), &cr) in pixels.iter_mut().zip(luma).zip(cb).zip(cr) {
                let y = (y as f32 - self.y_offset) * self.y_scale;
                let cb = (cb - self.c_offset) * self.c_scale;
                let cr = (cr - self.c_offset) * self.c_scale;
                rgb[0] = T::quantize(y + self.red_cr * cr);
                rgb[1] = T::quantize(y + self.green_cb * cb + self.green_cr * cr);
                rgb[2] = T::quantize(y + self.blue_cb * cb);
            }
        }
    }
}

/// Chroma rows interpolated horizontally to the output width, Cb and Cr,
/// keyed by chroma row. Each output row blends two of them, and neighbouring
/// output rows share theirs, so a band of rows interpolates each chroma row
/// once rather than once for every output row that reads it.
struct ChromaRows {
    keys: [Option<usize>; 3],
    cb: Vec<f32>,
    cr: Vec<f32>,
    /// One output row's chroma, blended from two of the rows above.
    blended: [Vec<f32>; 2],
}
impl ChromaRows {
    fn new(width: usize) -> Self {
        ChromaRows { keys: [None; 3], cb: vec![0.0; 3 * width], cr: vec![0.0; 3 * width], blended: [vec![0.0; width], vec![0.0; width]] }
    }
    /// The slot holding chroma row `row`, interpolating it into a slot other
    /// than `keep`'s if no slot holds it yet. `None` if a sample of the row
    /// exceeds `max`.
    fn slot(&mut self, row: usize, keep: usize, image: &DecodedImage, xs: &[Axis], max: u32) -> Option<usize> {
        if let Some(slot) = self.keys.iter().position(|&key| key == Some(row)) {
            return Some(slot);
        }
        let slot =
            (0..3).filter(|&slot| self.keys[slot] != Some(keep)).max_by_key(|&slot| self.keys[slot].map_or(usize::MAX, |key| key.abs_diff(row))).unwrap_or(0);
        let width = xs.len();
        for (plane, rows) in [(&image.cb, &mut self.cb), (&image.cr, &mut self.cr)] {
            let source = &plane[row * image.cb_width..][..image.cb_width];
            if !in_range(source, max) {
                return None;
            }
            for (o, ax) in rows[slot * width..][..width].iter_mut().zip(xs) {
                *o = source[ax.lo] as f32 * (1.0 - ax.weight) + source[ax.hi] as f32 * ax.weight;
            }
        }
        self.keys[slot] = Some(row);
        Some(slot)
    }
    /// Blends the rows in slots `top` and `bottom`: `top * (1 - weight) + bottom * weight`.
    fn blend(&mut self, top: usize, bottom: usize, weight: f32) -> (&[f32], &[f32]) {
        let width = self.blended[0].len();
        let [cb, cr] = &mut self.blended;
        for (rows, out) in [(&self.cb, &mut *cb), (&self.cr, &mut *cr)] {
            let (t, b) = (&rows[top * width..][..width], &rows[bottom * width..][..width]);
            for ((o, &t), &b) in out.iter_mut().zip(t).zip(b) {
                *o = t * (1.0 - weight) + b * weight;
            }
        }
        (cb, cr)
    }
}

/// Whether every sample lies in `0..=max`; written as a maximum, without an
/// early exit, so that it vectorizes.
fn in_range(samples: &[i32], max: u32) -> bool {
    samples.iter().map(|&v| v as u32).max().is_none_or(|v| v <= max)
}

/// Output rows per parallel task; a band reuses its interpolated chroma rows.
const ROWS_PER_BAND: usize = 16;

fn convert<T: RgbSample, const CHANNELS: usize>(image: &DecodedImage, options: &RgbOptions) -> Result<RgbImage<T>> {
    let _timer = crate::timing::start(crate::timing::Stage::Color);
    let bad = || HeifError::MalformedHevc("RGB: invalid plane geometry or sampling");
    let monochrome = image.chroma_format_idc == 0;
    if image.width == 0 || image.height == 0 || (!monochrome && (image.cb_width == 0 || image.cb_height == 0)) {
        return Err(bad());
    }
    let pixels = image.width.checked_mul(image.height).ok_or_else(bad)?;
    if pixels as u64 > MAX_IMAGE_PIXELS {
        return Err(HeifError::LimitExceeded("RGB: picture larger than MAX_IMAGE_PIXELS"));
    }
    let cpixels = image.cb_width.checked_mul(image.cb_height).ok_or_else(bad)?;
    let s = image.chroma_sampling;
    let valid_subsampling = match image.chroma_format_idc {
        0 => (s.sub_x, s.sub_y) == (1, 1),
        1 => (s.sub_x, s.sub_y) == (2, 2),
        2 => matches!((s.sub_x, s.sub_y), (2, 1) | (1, 2)),
        3 => (s.sub_x, s.sub_y) == (1, 1),
        _ => false,
    };
    let chroma_shape_valid = if monochrome {
        image.cb_width == 0 && image.cb_height == 0
    } else {
        image.cb_width == image.width.div_ceil(s.sub_x.max(1) as usize) && image.cb_height == image.height.div_ceil(s.sub_y.max(1) as usize)
    };
    if !matches!(s.sub_x, 1 | 2)
        || !matches!(s.sub_y, 1 | 2)
        || !s.origin_x.is_finite()
        || !s.origin_y.is_finite()
        || !valid_subsampling
        || s.origin_x.abs() > 2.0
        || s.origin_y.abs() > 2.0
        || !chroma_shape_valid
        || image.y.len() != pixels
        || image.cb.len() != cpixels
        || image.cr.len() != cpixels
        || !(8..=16).contains(&image.bit_depth_luma)
        || !(8..=16).contains(&image.bit_depth_chroma)
    {
        return Err(bad());
    }
    let info = options.color.or(image.color_info).unwrap_or_default();
    let matrix = Matrix::new(if monochrome { ColorInfo { matrix_coefficients: 1, ..info } } else { info }, image.bit_depth_luma, image.bit_depth_chroma)?;
    if matrix.identity && (s.sub_x != 1 || s.sub_y != 1) {
        return Err(HeifError::Unsupported("RGB: identity matrix requires 4:4:4"));
    }
    let xs = if monochrome { Vec::new() } else { axis(image.width, image.cb_width, s.sub_x, s.origin_x, options.upsampling)? };
    let ys = if monochrome { Vec::new() } else { axis(image.height, image.cb_height, s.sub_y, s.origin_y, options.upsampling)? };
    let length = pixels.checked_mul(CHANNELS).ok_or_else(bad)?;
    let mut data = Vec::new();
    data.try_reserve_exact(length).map_err(|_| HeifError::LimitExceeded("RGB: output allocation failed"))?;
    data.resize(length, T::default());
    // Samples are range-checked as the rows are read: luma row by row, and
    // each chroma row as it is interpolated. A chroma row no output row reads
    // is checked here.
    let out_of_range = || HeifError::MalformedHevc("RGB: sample outside coded range");
    let (luma_max, chroma_max) = ((1u32 << image.bit_depth_luma) - 1, (1u32 << image.bit_depth_chroma) - 1);
    let mut read = vec![false; if monochrome { 0 } else { image.cb_height }];
    for ay in &ys {
        read[ay.lo] = true;
        read[ay.hi] = true;
    }
    for (row, _) in read.iter().enumerate().filter(|(_, read)| !**read) {
        for plane in [&image.cb, &image.cr] {
            if !in_range(&plane[row * image.cb_width..][..image.cb_width], chroma_max) {
                return Err(out_of_range());
            }
        }
    }
    let width = image.width;
    let row_length = width * CHANNELS;
    // The arithmetic, and its order, is the same as interpolating chroma
    // pixel by pixel, so the output is too. Alpha comes last.
    data.par_chunks_mut(row_length * ROWS_PER_BAND).enumerate().try_for_each_init(
        || ChromaRows::new(if monochrome { 0 } else { width }),
        |chroma, (band, rows)| {
            for (i, output) in rows.chunks_mut(row_length).enumerate() {
                let row = band * ROWS_PER_BAND + i;
                let pixels = output.as_chunks_mut::<CHANNELS>().0;
                let luma = &image.y[row * width..][..width];
                if !in_range(luma, luma_max) {
                    return Err(out_of_range());
                }
                if monochrome {
                    for (rgb, &y) in pixels.iter_mut().zip(luma) {
                        let value = T::quantize((y as f32 - matrix.y_offset) * matrix.y_scale);
                        rgb[..3].fill(value);
                    }
                } else {
                    let ay = &ys[row];
                    let top = chroma.slot(ay.lo, ay.hi, image, &xs, chroma_max).ok_or_else(out_of_range)?;
                    let bottom = chroma.slot(ay.hi, ay.lo, image, &xs, chroma_max).ok_or_else(out_of_range)?;
                    let (cb, cr) = chroma.blend(top, bottom, ay.weight);
                    matrix.row(pixels, luma, cb, cr);
                }
                if CHANNELS == 4 {
                    match &image.alpha {
                        Some(alpha) => {
                            let max = (1u32 << alpha.bit_depth) - 1;
                            for (rgb, &a) in pixels.iter_mut().zip(&alpha.samples[row * width..][..width]) {
                                rgb[3] = T::alpha(a as u32, max);
                            }
                        }
                        None => pixels.iter_mut().for_each(|rgb| rgb[3] = T::alpha(1, 1)),
                    }
                }
            }
            Ok(())
        },
    )?;
    Ok(RgbImage { width: image.width, height: image.height, data })
}

fn convert_rgba<T: RgbSample>(image: &DecodedImage, options: &RgbOptions) -> Result<RgbaImage<T>> {
    if let Some(alpha) = &image.alpha {
        let pixels = image.width.checked_mul(image.height).ok_or(HeifError::MalformedHevc("RGBA: dimensions overflow"))?;
        if !(8..=16).contains(&alpha.bit_depth) || alpha.samples.len() != pixels {
            return Err(HeifError::MalformedHevc("RGBA: invalid alpha geometry or bit depth"));
        }
        let max = (1i32 << alpha.bit_depth) - 1;
        if alpha.samples.iter().any(|&v| v < 0 || v > max) {
            return Err(HeifError::MalformedHevc("RGBA: alpha outside coded range"));
        }
    }
    let rgb = convert::<T, 4>(image, options)?;
    Ok(RgbaImage { width: rgb.width, height: rgb.height, data: rgb.data, premultiplied: image.alpha.as_ref().is_some_and(|alpha| alpha.premultiplied) })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rgba_uses_full_range_alpha_and_preserves_opaque_rgb() {
        let mut image = planar(vec![16, 128, 235], vec![128; 3], vec![128; 3], 8, info(1, false));
        let opaque = image.to_rgba16().unwrap();
        assert!(opaque.data.as_chunks::<4>().0.iter().all(|pixel| pixel[3] == 65535));
        image.alpha = Some(crate::AlphaPlane { bit_depth: 10, samples: vec![0, 513, 1023], premultiplied: true });
        let rgba = image.to_rgba16().unwrap();
        assert_eq!(rgba.data.as_chunks::<4>().0.iter().map(|p| p[3]).collect::<Vec<_>>(), [0, 32864, 65535]);
        assert!(rgba.premultiplied);
        assert_eq!(rgba.data.as_chunks::<4>().0.iter().flat_map(|p| p[..3].iter().copied()).collect::<Vec<_>>(), image.to_rgb16().unwrap().data);
        image.alpha.as_mut().unwrap().bit_depth = 16;
        image.alpha.as_mut().unwrap().samples = vec![0, 32769, 65535];
        assert_eq!(image.to_rgba16().unwrap().data.as_chunks::<4>().0.iter().map(|p| p[3]).collect::<Vec<_>>(), [0, 32769, 65535]);
    }

    #[test]
    fn invalid_alpha_is_rejected_before_rgba_conversion() {
        let mut image = planar(vec![16], vec![128], vec![128], 8, info(1, false));
        for (depth, samples) in [(0, vec![0]), (17, vec![0]), (8, vec![]), (8, vec![-1]), (8, vec![256])] {
            image.alpha = Some(crate::AlphaPlane { bit_depth: depth, samples, premultiplied: false });
            assert!(image.to_rgba8().is_err());
            assert!(image.to_rgba16().is_err());
            assert!(image.to_rgb8().is_ok(), "explicit RGB conversion omits alpha");
        }
    }
    fn planar(y: Vec<i32>, cb: Vec<i32>, cr: Vec<i32>, depth: u32, color: ColorInfo) -> DecodedImage {
        DecodedImage {
            alpha: None,
            width: y.len(),
            height: 1,
            cb_width: cb.len(),
            cb_height: 1,
            bit_depth_luma: depth,
            bit_depth_chroma: depth,
            chroma_format_idc: 3,
            color_info: Some(color),
            chroma_sampling: ChromaSampling::coded(3, None),
            y,
            cb,
            cr,
            metadata: Default::default(),
        }
    }
    fn info(matrix_coefficients: u16, full_range: bool) -> ColorInfo {
        ColorInfo { matrix_coefficients, full_range, ..Default::default() }
    }
    #[test]
    fn limited_and_full_range_black_white_at_every_depth() {
        for depth in [8, 10, 12, 16] {
            let shift = 1 << (depth - 8);
            let mid = 1 << (depth - 1);
            for full in [false, true] {
                let (black, white) = if full { (0, (1 << depth) - 1) } else { (16 * shift, 235 * shift) };
                let image = planar(vec![black, white], vec![mid; 2], vec![mid; 2], depth, info(1, full));
                assert_eq!(image.to_rgb8().unwrap().data, [0, 0, 0, 255, 255, 255], "depth={depth}, full={full}");
                assert_eq!(image.to_rgb16().unwrap().data, [0, 0, 0, 65535, 65535, 65535]);
            }
        }
    }
    #[test]
    fn matrix_codes_produce_distinct_known_colour_vectors() {
        for (matrix, want) in [(5, [255, 22, 21]), (6, [255, 22, 21]), (1, [255, 46, 18]), (9, [255, 32, 16])] {
            let image = planar(vec![100], vec![90], vec![240], 8, info(matrix, false));
            assert_eq!(image.to_rgb8().unwrap().data, want, "matrix={matrix}");
        }
    }
    #[test]
    fn identity_matrix_has_gbr_order_and_independent_channel_depths() {
        let mut image = planar(vec![128], vec![256], vec![512], 8, info(0, true));
        image.bit_depth_chroma = 10;
        assert_eq!(image.to_rgb8().unwrap().data, [128, 128, 64]);
        let image = planar(vec![16], vec![235], vec![126], 8, info(0, false));
        assert_eq!(image.to_rgb8().unwrap().data, [128, 0, 255]);
    }
    #[test]
    fn sixteen_bit_conversion_preserves_ten_bit_precision() {
        let image = planar(vec![65], vec![321], vec![1000], 10, info(0, true));
        let rgb = image.to_rgb16().unwrap();
        let expected: Vec<_> = [1000u32, 65, 321].iter().map(|v| ((v * 65535 + 511) / 1023) as u16).collect();
        assert_eq!(rgb.data, expected);
        assert!(rgb.data.iter().any(|v| v % 257 != 0));
    }
    #[test]
    fn unlabelled_colour_defaults_and_explicit_override_are_separate() {
        let mut image = planar(vec![100], vec![90], vec![240], 8, info(6, false));
        let options = RgbOptions { color: Some(info(1, false)), ..Default::default() };
        assert_eq!(image.to_rgb8_with_options(&options).unwrap().data, [255, 46, 18]);
        assert_eq!(image.to_rgb8().unwrap().data, [255, 22, 21]);
        // Unlabelled and unspecified both read as BT.601, as in libheif.
        image.color_info = None;
        assert_eq!(image.to_rgb8().unwrap().data, [255, 22, 21]);
        image.color_info = Some(info(2, false));
        assert_eq!(image.to_rgb8().unwrap().data, [255, 22, 21]);
    }
    #[test]
    fn bilinear_upsampling_uses_chroma_locations_and_nearest_is_optional() {
        let mut image = planar(vec![64; 4], vec![128, 160], vec![128; 2], 8, info(1, true));
        image.chroma_format_idc = 2;
        image.chroma_sampling = ChromaSampling::coded(2, None);
        let blue: Vec<_> = image.to_rgb8().unwrap().data.as_chunks::<3>().0.iter().map(|rgb| rgb[2]).collect();
        assert_eq!(blue, [64, 94, 123, 123]);
        image.chroma_sampling.origin_x = 0.5;
        let blue: Vec<_> = image.to_rgb8().unwrap().data.as_chunks::<3>().0.iter().map(|rgb| rgb[2]).collect();
        assert_eq!(blue, [64, 79, 109, 123]);
        let options = RgbOptions { upsampling: ChromaUpsampling::Nearest, ..Default::default() };
        let blue: Vec<_> = image.to_rgb8_with_options(&options).unwrap().data.as_chunks::<3>().0.iter().map(|rgb| rgb[2]).collect();
        assert_eq!(blue, [64, 64, 123, 123]);
    }
    #[test]
    fn rotated_and_mirrored_422_converts_like_transformed_rgb() {
        let image = || DecodedImage {
            alpha: None,
            width: 4,
            height: 3,
            cb_width: 2,
            cb_height: 3,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
            chroma_format_idc: 2,
            color_info: Some(info(1, true)),
            chroma_sampling: ChromaSampling::coded(2, None),
            y: vec![100; 12],
            cb: vec![128, 150, 140, 160, 160, 180],
            cr: vec![128; 6],
            metadata: Default::default(),
        };
        let source = image().to_rgb8().unwrap();
        for transform in [crate::isobmff::Transform::Rotate90Ccw(1), crate::isobmff::Transform::MirrorHorizontal, crate::isobmff::Transform::MirrorVertical] {
            let transformed = crate::apply_transforms(image(), &[transform]).unwrap();
            let rgb = transformed.to_rgb8().unwrap();
            for y in 0..rgb.height {
                for x in 0..rgb.width {
                    let (sx, sy) = match transform {
                        crate::isobmff::Transform::Rotate90Ccw(_) => (3 - y, x),
                        crate::isobmff::Transform::MirrorHorizontal => (3 - x, y),
                        _ => (x, 2 - y),
                    };
                    assert_eq!(
                        &rgb.data[(y * rgb.width + x) * 3..(y * rgb.width + x + 1) * 3],
                        &source.data[(sy * 4 + sx) * 3..(sy * 4 + sx + 1) * 3],
                        "{transform:?}, ({x},{y})"
                    );
                }
            }
        }
    }
    #[test]
    fn odd_crop_tracks_chroma_phase_interior_samples() {
        let image = DecodedImage {
            alpha: None,
            width: 8,
            height: 2,
            cb_width: 4,
            cb_height: 2,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
            chroma_format_idc: 2,
            color_info: Some(info(1, true)),
            chroma_sampling: ChromaSampling::coded(2, None),
            y: vec![100; 16],
            cb: vec![128, 140, 160, 180, 128, 140, 160, 180],
            cr: vec![128; 8],
            metadata: Default::default(),
        };
        let before = image.to_rgb8().unwrap();
        let window = crate::ImageWindow { left: 1, top: 0, width: 6, height: 2 };
        let cropped = crate::extract_window(image, window, 2, 1).unwrap();
        assert_eq!(cropped.chroma_sampling.origin_x, -1.0);
        let rgb = cropped.to_rgb8().unwrap();
        // Samples outside the returned native crop are edge-extended; the
        // interior must still use the original chroma positions.
        assert_eq!(&rgb.data[..12], &before.data[3..15]);
    }
    #[test]
    fn invalid_shapes_ranges_sampling_and_matrices_return_errors() {
        let original = || planar(vec![100], vec![128], vec![128], 8, info(1, false));
        let mut image = original();
        image.cb.clear();
        assert!(image.to_rgb8().is_err());
        let mut image = original();
        image.height = usize::MAX;
        assert!(image.to_rgb8().is_err());
        let mut image = original();
        image.bit_depth_luma = 32;
        assert!(image.to_rgb8().is_err());
        let mut image = original();
        image.y[0] = -1;
        assert!(image.to_rgb8().is_err());
        let mut image = original();
        image.cb[0] = 256;
        assert!(image.to_rgb8().is_err());
        let mut image = original();
        image.chroma_sampling.sub_x = 0;
        assert!(image.to_rgb8().is_err());
        let mut image = original();
        image.chroma_sampling.origin_y = f32::NAN;
        assert!(image.to_rgb8().is_err());
        let mut image = original();
        image.color_info = Some(info(10, false));
        assert!(matches!(image.to_rgb8(), Err(HeifError::Unsupported(_))));
        let mut image = original();
        image.chroma_format_idc = 2;
        image.chroma_sampling = ChromaSampling::coded(2, None);
        image.color_info = Some(info(0, true));
        assert!(image.to_rgb8().is_err());
    }
    #[test]
    fn quantize_matches_rounding_at_the_edges() {
        // Every f32 was compared once; these are the cases that could differ.
        let below_half = f32::from_bits(0.5f32.to_bits() - 1);
        for value in [
            0.0,
            -0.0,
            0.5,
            below_half,
            1.5,
            2.5,
            254.5,
            255.0,
            255.49998,
            255.5,
            -0.5,
            -0.49999997,
            1e10,
            -1e10,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::MAX,
            f32::MIN_POSITIVE,
            16_777_217.0,
            2_147_483_648.0,
        ] {
            for scale in [1.0f32, 1.0 / 255.0, 1.0 / 65535.0] {
                let v = value * scale;
                assert_eq!(quantize_without_round(v, 255.0, 255), u32::from((v * 255.0).round() as u8), "{value} * {scale} to u8");
                assert_eq!(quantize_without_round(v, 65535.0, 65535), u32::from((v * 65535.0).round() as u16), "{value} * {scale} to u16");
            }
        }
    }
    #[test]
    fn a_chroma_row_no_pixel_reads_is_still_range_checked() {
        // With the chroma origin two rows up and nearest sampling, every
        // output row reads chroma row 1, none reads row 0.
        let image = |cb0: i32| DecodedImage {
            width: 2,
            height: 4,
            cb_width: 1,
            cb_height: 2,
            chroma_format_idc: 1,
            chroma_sampling: ChromaSampling { sub_x: 2, sub_y: 2, origin_x: 0.0, origin_y: -2.0 },
            y: vec![100; 8],
            cb: vec![cb0, 128],
            cr: vec![128; 2],
            ..planar(vec![100], vec![128], vec![128], 8, info(1, false))
        };
        let nearest = RgbOptions { upsampling: ChromaUpsampling::Nearest, ..Default::default() };
        assert!(image(255).to_rgb8_with_options(&nearest).is_ok());
        assert!(image(256).to_rgb8_with_options(&nearest).is_err());
        assert!(image(-1).to_rgb8_with_options(&nearest).is_err());
    }
}
