//! Decode straight to a small, opaque RGB8 image without ever holding the
//! full-size picture.
//!
//! [`crate::decode`] assembles the whole canvas as `i32` planes before any
//! conversion: a 12 MP iPhone photo peaks near 180 MB, a 48 MP one past
//! 700 MB. A server that only needs a ~1568 px rendition decodes the grid
//! a few tiles at a time instead, converts each tile to RGB, and averages
//! its pixels into the output bins they land in after the container's
//! crop, rotation and mirroring. Peak memory is then a batch of tiles plus
//! the output, whatever the photo's size.
//!
//! Images that are not grids, or that carry alpha or a prediction
//! reference, take the full [`crate::decode`] path, bounded by
//! [`FitOptions::max_picture_pixels`].

use rayon::prelude::*;

use crate::color::RgbImage;
use crate::error::{HeifError, Result};
use crate::hevc::nal::NAL_SPS;
use crate::hevc::params::parse_sps;
use crate::isobmff::{self, ResolvedImage, Transform};
use crate::{crop_axis, decode_one_image, find_and_parse, image_from_picture};

/// Limits and target size for [`decode_rgb8_fit`].
#[derive(Debug, Clone, Copy)]
pub struct FitOptions {
    /// Longest edge of the output. Smaller images are not scaled up.
    pub max_edge: usize,
    /// Largest coded picture (one grid tile, or a whole non-grid image
    /// with its alpha and reference), in luma samples as the SPS codes
    /// them, before any cropping.
    pub max_picture_pixels: u64,
    /// Largest grid: the assembled canvas, and the sum of its tiles.
    pub max_total_pixels: u64,
    /// Coded luma samples decoded at once. Tiles are decoded in parallel
    /// batches up to this size; one tile always goes, whatever its size.
    pub batch_pixels: u64,
}

/// One container transform with the dimensions it applies to resolved,
/// so a pixel can be mapped through it on its own.
#[derive(Debug, Clone, Copy)]
enum Step {
    Crop {
        left: usize,
        top: usize,
        width: usize,
        height: usize,
    },
    /// One 90° anticlockwise turn of a picture `width` wide.
    Turn {
        width: usize,
    },
    FlipX {
        width: usize,
    },
    FlipY {
        height: usize,
    },
}

/// The steps `transforms` take a `width`×`height` picture through, and
/// the size they leave it at — the same arithmetic as `apply_transforms`.
fn resolve(mut width: usize, mut height: usize, transforms: &[Transform]) -> Result<(Vec<Step>, usize, usize)> {
    let mut steps = Vec::new();
    for &transform in transforms {
        match transform {
            Transform::CleanAperture(aperture) => {
                let (left, w) = crop_axis(width, aperture.width, aperture.horizontal_offset)?;
                let (top, h) = crop_axis(height, aperture.height, aperture.vertical_offset)?;
                steps.push(Step::Crop { left, top, width: w, height: h });
                (width, height) = (w, h);
            }
            Transform::Rotate90Ccw(turns) => {
                for _ in 0..turns % 4 {
                    steps.push(Step::Turn { width });
                    std::mem::swap(&mut width, &mut height);
                }
            }
            Transform::MirrorHorizontal => steps.push(Step::FlipX { width }),
            Transform::MirrorVertical => steps.push(Step::FlipY { height }),
        }
    }
    Ok((steps, width, height))
}

/// Where canvas pixel (`x`, `y`) ends up, or `None` when a crop drops it.
/// The point-by-point reference [`axis_maps`] is tested against.
#[cfg(test)]
fn map(steps: &[Step], mut x: usize, mut y: usize) -> Option<(usize, usize)> {
    for &step in steps {
        (x, y) = match step {
            Step::Crop { left, top, width, height } => {
                let (x, y) = (x.checked_sub(left)?, y.checked_sub(top)?);
                if x >= width || y >= height {
                    return None;
                }
                (x, y)
            }
            // rotate90_ccw_plane: dst(dx, dy) = src(width - 1 - dy, dx).
            Step::Turn { width } => (y, width - 1 - x),
            Step::FlipX { width } => (width - 1 - x, y),
            Step::FlipY { height } => (x, height - 1 - y),
        };
    }
    Some((x, y))
}

/// Width and height scaled down so the longest edge is at most `max_edge`.
fn fit_within(width: usize, height: usize, max_edge: usize) -> (usize, usize) {
    let longest = width.max(height);
    if longest <= max_edge {
        return (width, height);
    }
    let scale = |side: usize| ((side * max_edge + longest / 2) / longest).max(1);
    (scale(width), scale(height))
}

/// Where one canvas axis lands: for each canvas coordinate, its output
/// bin, already scaled for the output axis it feeds (`bin` or
/// `bin * out_width`), or `None` when a crop drops it. Every transform
/// is axis-aligned, so the two canvas axes map independently and a
/// pixel's bin is the sum of its column's and its row's entries.
struct AxisMap {
    parts: Vec<Option<usize>>,
    /// Whether this canvas axis ends up as the output's horizontal one.
    horizontal: bool,
}

/// The two canvas axes through `steps`, binned into `out`.
fn axis_maps(canvas: (usize, usize), steps: &[Step], full: (usize, usize), out: (usize, usize)) -> [AxisMap; 2] {
    // Per canvas axis: current coordinate of each position (None once
    // cropped), and which current axis it is (0 = x, 1 = y).
    let mut axes: [(Vec<Option<usize>>, usize); 2] = [((0..canvas.0).map(Some).collect(), 0), ((0..canvas.1).map(Some).collect(), 1)];
    let along = |axes: &mut [(Vec<Option<usize>>, usize); 2], axis: usize, f: &dyn Fn(usize) -> Option<usize>| {
        for (values, feeds) in axes.iter_mut() {
            if *feeds == axis {
                values.iter_mut().for_each(|v| *v = v.and_then(f));
            }
        }
    };
    for &step in steps {
        match step {
            Step::Crop { left, top, width, height } => {
                along(&mut axes, 0, &|v| v.checked_sub(left).filter(|&v| v < width));
                along(&mut axes, 1, &|v| v.checked_sub(top).filter(|&v| v < height));
            }
            // (x, y) → (y, width - 1 - x): old x becomes the new y, flipped.
            Step::Turn { width } => {
                along(&mut axes, 0, &|v| Some(width - 1 - v));
                for (_, feeds) in axes.iter_mut() {
                    *feeds = 1 - *feeds;
                }
            }
            Step::FlipX { width } => along(&mut axes, 0, &|v| Some(width - 1 - v)),
            Step::FlipY { height } => along(&mut axes, 1, &|v| Some(height - 1 - v)),
        }
    }
    axes.map(|(values, feeds)| {
        let horizontal = feeds == 0;
        let (full, out_len, stride) = if horizontal { (full.0, out.0, 1) } else { (full.1, out.1, out.0) };
        let parts = values.into_iter().map(|v| v.map(|v| v * out_len / full * stride)).collect();
        AxisMap { parts, horizontal }
    })
}

/// How many canvas positions land in each bin of the output axis `map`
/// feeds.
fn bin_counts(map: &AxisMap, out: (usize, usize)) -> Vec<u32> {
    let (len, stride) = if map.horizontal { (out.0, 1) } else { (out.1, out.0) };
    let mut counts = vec![0u32; len];
    for part in map.parts.iter().flatten() {
        counts[part / stride] += 1;
    }
    counts
}

/// Sums per output channel: `u16` while a bin's total cannot overflow it
/// (scale factors up to 16, every phone photo at 1568 px), else `u32`.
enum Sums {
    Narrow(Vec<u16>),
    Wide(Vec<u32>),
}

/// Box-filter accumulator: every source pixel adds into the output pixel
/// whose area it falls in. Counts are not stored: a bin's count is the
/// product of its column's and its row's.
struct Bins {
    out: (usize, usize),
    maps: [AxisMap; 2],
    sums: Sums,
}

impl Bins {
    fn new(canvas: (usize, usize), steps: &[Step], full: (usize, usize), out: (usize, usize)) -> Result<Self> {
        let channels = out.0.checked_mul(out.1).and_then(|p| p.checked_mul(3)).ok_or(HeifError::LimitExceeded("fit: output size"))?;
        let maps = axis_maps(canvas, steps, full, out);
        let widest = |map: &AxisMap| bin_counts(map, out).into_iter().max().unwrap_or(0);
        let area = u64::from(widest(&maps[0])) * u64::from(widest(&maps[1]));
        let failed = |_| HeifError::LimitExceeded("fit: output allocation failed");
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

    /// Adds one row of RGB pixels found at canvas row `y`, from column `x0`.
    fn add_row(&mut self, x0: usize, y: usize, pixels: &[[u8; 3]]) {
        let Some(row) = self.maps[1].parts.get(y).copied().flatten() else { return };
        let columns = self.maps[0].parts.get(x0..).unwrap_or(&[]);
        fn add<T: Copy + std::ops::AddAssign + From<u8>>(sums: &mut [T], row: usize, columns: &[Option<usize>], pixels: &[[u8; 3]]) {
            for (column, px) in columns.iter().zip(pixels) {
                if let Some(column) = column {
                    let i = (row + column) * 3;
                    for (sum, &c) in sums[i..i + 3].iter_mut().zip(px) {
                        *sum += T::from(c);
                    }
                }
            }
        }
        match &mut self.sums {
            Sums::Narrow(sums) => add(sums, row, columns, pixels),
            Sums::Wide(sums) => add(sums, row, columns, pixels),
        }
    }

    fn finish(self) -> RgbImage<u8> {
        let (width, height) = self.out;
        let [x_map, y_map] = &self.maps;
        let (columns, rows) = if x_map.horizontal {
            (bin_counts(x_map, self.out), bin_counts(y_map, self.out))
        } else {
            (bin_counts(y_map, self.out), bin_counts(x_map, self.out))
        };
        let mut data = Vec::with_capacity(width * height * 3);
        for (y, &row) in rows.iter().enumerate() {
            for (x, &column) in columns.iter().enumerate() {
                let n = row * column;
                let i = (y * width + x) * 3;
                for c in 0..3 {
                    let sum = match &self.sums {
                        Sums::Narrow(s) => u32::from(s[i + c]),
                        Sums::Wide(s) => s[i + c],
                    };
                    // An empty bin cannot happen (every bin covers at least
                    // one full-size pixel); white if it ever did.
                    data.push((sum + n / 2).checked_div(n).map_or(255, |m| m as u8));
                }
            }
        }
        RgbImage { width, height, data }
    }
}

fn picture_pixels(image: &ResolvedImage) -> Result<u64> {
    let sps = find_and_parse(image.hevc_config.nalus_of_type(NAL_SPS), NAL_SPS, parse_sps)?;
    Ok(u64::from(sps.pic_width_in_luma_samples) * u64::from(sps.pic_height_in_luma_samples))
}

/// Opaque RGB8: transparency goes over white.
fn over_white(image: &crate::DecodedImage) -> Result<RgbImage<u8>> {
    if image.alpha.is_none() {
        return image.to_rgb8();
    }
    let rgba = image.to_rgba8()?;
    let blend = |c: u8, a: u32| ((u32::from(c) * a + 255 * (255 - a) + 127) / 255) as u8;
    let data = rgba
        .data
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|&[r, g, b, a]| {
            let a = u32::from(a);
            [blend(r, a), blend(g, a), blend(b, a)]
        })
        .collect();
    Ok(RgbImage { width: rgba.width, height: rgba.height, data })
}

/// Decode the primary image to opaque RGB8 (transparency over white),
/// crop/rotation/mirroring applied, its longest edge scaled down to at
/// most `options.max_edge` by averaging.
pub fn decode_rgb8_fit(bytes: &[u8], options: &FitOptions) -> Result<RgbImage<u8>> {
    let parsed = isobmff::parse(bytes)?;
    let too_large = || HeifError::LimitExceeded("fit: picture larger than allowed");
    let grid = match parsed.grid {
        Some(grid) if parsed.alpha.is_none() && parsed.reference_image.is_none() => grid,
        _ => {
            let mut pixels = 0u64;
            for image in parsed.images.iter().chain(&parsed.reference_image) {
                pixels = pixels.saturating_add(picture_pixels(image)?);
            }
            if let Some(alpha) = &parsed.alpha {
                for image in &alpha.image.images {
                    pixels = pixels.saturating_add(picture_pixels(image)?);
                }
            }
            if let Some(grid) = parsed.grid {
                pixels = pixels.saturating_add(u64::from(grid.output_width) * u64::from(grid.output_height));
            }
            if pixels > options.max_picture_pixels {
                return Err(too_large());
            }
            let rgb = over_white(&crate::decode(bytes)?)?;
            let full = (rgb.width, rgb.height);
            let mut bins = Bins::new(full, &[], full, fit_within(full.0, full.1, options.max_edge))?;
            for (y, row) in rgb.data.as_chunks::<3>().0.chunks(full.0).enumerate() {
                bins.add_row(0, y, row);
            }
            return Ok(bins.finish());
        }
    };

    let (width, height) = (grid.output_width as usize, grid.output_height as usize);
    if parsed.images.len() != (grid.rows as usize) * (grid.columns as usize) {
        return Err(HeifError::MalformedBox("grid image count does not match rows*columns"));
    }
    let mut tile_pixels = Vec::with_capacity(parsed.images.len());
    for image in &parsed.images {
        let pixels = picture_pixels(image)?;
        if pixels > options.max_picture_pixels {
            return Err(too_large());
        }
        tile_pixels.push(pixels);
    }
    if (width * height) as u64 > options.max_total_pixels || tile_pixels.iter().sum::<u64>() > options.max_total_pixels {
        return Err(too_large());
    }
    let (steps, full_width, full_height) = resolve(width, height, &parsed.transforms)?;
    let mut bins = Bins::new((width, height), &steps, (full_width, full_height), fit_within(full_width, full_height, options.max_edge))?;
    let color_info = parsed.metadata.color_info;

    let mut tile_size = None;
    let mut next = 0;
    while next < parsed.images.len() {
        let mut end = next + 1;
        let mut batch = tile_pixels[next];
        // As many tiles as there are threads to decode them, so a tile's
        // RGB is accumulated soon after it is decoded: one at a time on
        // one core.
        let threads = rayon::current_num_threads();
        while end < parsed.images.len() && end - next < threads && batch + tile_pixels[end] <= options.batch_pixels {
            batch += tile_pixels[end];
            end += 1;
        }
        let tiles = parsed.images[next..end]
            .par_iter()
            .map(|image| {
                let (sps, pic) = decode_one_image(image)?;
                let mut tile = image_from_picture(&sps, pic)?;
                tile.color_info = color_info.or(tile.color_info);
                let rgb = tile.to_rgb8()?;
                Ok(rgb)
            })
            .collect::<Result<Vec<_>>>()?;
        for (offset, tile) in tiles.iter().enumerate() {
            // Tiles share the first tile's size (ISO/IEC 23008-12 6.6.2.3).
            let (tile_width, tile_height) = *tile_size.get_or_insert((tile.width, tile.height));
            if tile_width * (grid.columns as usize) < width || tile_height * (grid.rows as usize) < height {
                return Err(HeifError::MalformedBox("grid tiles do not cover the output"));
            }
            if (tile.width, tile.height) != (tile_width, tile_height) {
                return Err(HeifError::MalformedBox("grid tiles of different sizes"));
            }
            let index = next + offset;
            let (x0, y0) = (index % grid.columns as usize * tile_width, index / grid.columns as usize * tile_height);
            // A malformed grid may place a tile wholly outside the output.
            let visible = tile_width.min(width.saturating_sub(x0));
            for (row, pixels) in tile.data.as_chunks::<3>().0.chunks(tile_width).take(height.saturating_sub(y0)).enumerate() {
                bins.add_row(x0, y0 + row, &pixels[..visible]);
            }
        }
        next = end;
    }
    Ok(bins.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixels_follow_crop_turn_and_flips() {
        // 4×2, one anticlockwise turn: the top-right corner goes top-left.
        let (steps, w, h) = resolve(4, 2, &[Transform::Rotate90Ccw(1)]).unwrap();
        assert_eq!((w, h), (2, 4));
        assert_eq!(map(&steps, 3, 0), Some((0, 0)));
        assert_eq!(map(&steps, 0, 0), Some((0, 3)));
        assert_eq!(map(&steps, 0, 1), Some((1, 3)));
        let (steps, ..) = resolve(4, 2, &[Transform::MirrorHorizontal, Transform::MirrorVertical]).unwrap();
        assert_eq!(map(&steps, 0, 0), Some((3, 1)));
        let aperture = isobmff::CleanAperture { width: (2, 1), height: (2, 1), horizontal_offset: (0, 1), vertical_offset: (0, 1) };
        let (steps, w, h) = resolve(4, 4, &[Transform::CleanAperture(aperture)]).unwrap();
        assert_eq!((w, h), (2, 2));
        assert_eq!(map(&steps, 1, 1), Some((0, 0)));
        assert_eq!(map(&steps, 0, 0), None);
        assert_eq!(map(&steps, 3, 2), None);
    }

    #[test]
    fn averaging_bins_cover_the_whole_output() {
        let mut bins = Bins::new((5, 3), &[], (5, 3), fit_within(5, 3, 2)).unwrap();
        assert_eq!(bins.out, (2, 1));
        let row: Vec<[u8; 3]> = (0..5).map(|x| [if x < 3 { 0 } else { 200 }, 10, 255]).collect();
        for y in 0..3 {
            bins.add_row(0, y, &row);
        }
        let rgb = bins.finish();
        assert_eq!(rgb.data, [0, 10, 255, 200, 10, 255]);
    }

    #[test]
    fn axis_maps_agree_with_mapping_each_pixel() {
        let aperture = |w, h, dx, dy| {
            Transform::CleanAperture(isobmff::CleanAperture { width: (w, 1), height: (h, 1), horizontal_offset: (dx, 1), vertical_offset: (dy, 1) })
        };
        let chains = [
            vec![],
            vec![Transform::Rotate90Ccw(1)],
            vec![Transform::Rotate90Ccw(3), Transform::MirrorHorizontal],
            vec![aperture(4, 4, -1, 1), Transform::Rotate90Ccw(1), Transform::MirrorVertical],
            vec![Transform::MirrorHorizontal, aperture(6, 4, 0, 0), Transform::Rotate90Ccw(2)],
        ];
        let canvas = (8, 6);
        for transforms in chains {
            let (steps, w, h) = resolve(canvas.0, canvas.1, &transforms).unwrap();
            for out in [(w, h), fit_within(w, h, 3)] {
                let [x_map, y_map] = axis_maps(canvas, &steps, (w, h), out);
                for y in 0..canvas.1 {
                    for x in 0..canvas.0 {
                        let expected = map(&steps, x, y).map(|(tx, ty)| (ty * out.1 / h) * out.0 + tx * out.0 / w);
                        let got = x_map.parts[x].zip(y_map.parts[y]).map(|(a, b)| a + b);
                        assert_eq!(got, expected, "{transforms:?} {out:?} at ({x},{y})");
                    }
                }
            }
        }
    }

    #[test]
    fn narrow_sums_are_used_only_when_they_cannot_overflow() {
        assert!(matches!(Bins::new((64, 64), &[], (64, 64), (4, 4)).unwrap().sums, Sums::Narrow(_)));
        assert!(matches!(Bins::new((64, 64), &[], (64, 64), (2, 2)).unwrap().sums, Sums::Wide(_)));
    }
}
