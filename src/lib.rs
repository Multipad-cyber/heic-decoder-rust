//! A pure-Rust HEIF/HEIC still-image decoder.
//!
//! Decodes HEVC Main/Main10, monochrome and basic 4:2:2/4:4:4 images, including grid photos
//! iPhones produce, to planar YCbCr that is bit-exact against HM, the
//! official HEVC reference decoder. Rotation, mirroring and cropping from
//! the file are applied; the ICC profile, EXIF and XMP are returned as raw
//! bytes. Malformed input returns an error rather than panicking.
//! RGB conversion is optional: use [`DecodedImage::to_rgb8`] or
//! [`DecodedImage::to_rgb16`] when the caller needs interleaved RGB.
//! Auxiliary alpha is returned in [`DecodedImage::alpha`]; use
//! [`DecodedImage::to_rgba8`] or [`DecodedImage::to_rgba16`] to retain it.
//!
//! ```no_run
//! let data = std::fs::read("photo.heic")?;
//! let image = heic_decoder::decode(&data)?;
//! println!("{}x{}, {}-bit", image.width, image.height, image.bit_depth_luma);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! See `README.md` for supported features, limitations and provenance.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

// The parsers and the HEVC decoding stages are not part of the stable API.
// The `internals` feature exposes them for the HM comparison tests and tools.
#[cfg(feature = "internals")]
pub mod bitreader;
#[cfg(not(feature = "internals"))]
#[allow(dead_code)] // parsed fields and helpers used by `internals` users and tests
mod bitreader;
mod color;
mod error;
#[cfg(feature = "internals")]
pub mod hevc;
#[cfg(not(feature = "internals"))]
#[allow(dead_code)] // parsed fields and helpers used by `internals` users and tests
mod hevc;
#[cfg(feature = "internals")]
pub mod isobmff;
#[cfg(not(feature = "internals"))]
#[allow(dead_code)] // parsed fields and helpers used by `internals` users and tests
mod isobmff;
#[cfg(feature = "timing")]
#[doc(hidden)]
pub mod timing;
#[cfg(not(feature = "timing"))]
#[allow(dead_code)]
mod timing;

pub use color::{ChromaSampling, ChromaUpsampling, ColorInfo, RgbImage, RgbOptions, RgbaImage};
pub use error::{HeifError, Result};
pub use isobmff::Metadata;

/// Largest picture, in luma samples, that the decoder will allocate
/// (2^28, about 268 megapixels). Applies to each coded image, to the
/// assembled grid canvas, and to the sum of a grid's tiles. Larger
/// requests return [`HeifError::LimitExceeded`] instead of allocating.
pub const MAX_IMAGE_PIXELS: u64 = 1 << 28;

use bitreader::{ebsp_lens_to_rbsp_lens, rbsp_len_to_ebsp_pos, strip_emulation_prevention};
use hevc::ctu::{Picture, decode_slice_rows};
use hevc::deblock::deblock_picture;
use hevc::nal::{NAL_PPS, NAL_SPS, is_irap_nal, parse_nal_header};
use hevc::params::{Pps, Sps, parse_pps, parse_sps};
use hevc::sao::apply_sao;
use hevc::slice::parse_slice_header;
use isobmff::{ResolvedImage, split_length_prefixed_nalus};
use timing::Stage;

/// Native, full-range opacity: zero is transparent, `(1 << bit_depth) - 1`
/// is opaque. Samples have the same dimensions and orientation as the image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlphaPlane {
    pub bit_depth: u32,
    pub samples: Vec<i32>,
    /// Colour is already multiplied by opacity when the file has a `prem` link.
    pub premultiplied: bool,
}

/// A decoded still image: monochrome or planar YCbCr (4:2:0, 4:2:2 or 4:4:4),
/// `bit_depth` bits per sample, values in `0 ..= (1<<bit_depth)-1`.
/// RGB/RGBA conversion is optional via the `to_rgb*`/`to_rgba*` methods.
/// Clean-aperture output retains native samples: the chroma origin is the
/// cell containing the final luma crop origin, with ceil-divided extents.
/// Odd crop offsets do not trigger chroma interpolation or resampling.
pub struct DecodedImage {
    pub alpha: Option<AlphaPlane>,
    pub width: usize,
    pub height: usize,
    pub bit_depth_luma: u32,
    pub bit_depth_chroma: u32,
    /// Coded chroma format. Rotation also rotates the native chroma grid:
    /// a quarter-turn of 4:2:2 has full-width, half-height chroma planes.
    /// Use `cb_width` and `cb_height` for the returned plane dimensions.
    pub chroma_format_idc: u32,
    /// Effective container/SPS colour description; `None` means unlabelled.
    pub color_info: Option<ColorInfo>,
    /// Chroma positions after cropping, rotation and mirroring.
    pub chroma_sampling: ChromaSampling,
    pub y: Vec<i32>,
    pub cb: Vec<i32>,
    pub cr: Vec<i32>,
    pub cb_width: usize,
    pub cb_height: usize,
    /// ICC profile, EXIF and XMP attached to the image.
    pub metadata: Metadata,
}

fn find_and_parse<T>(nalus: impl Iterator<Item = impl AsRef<[u8]>>, expected_type: u8, parse: impl Fn(&[u8]) -> Result<T>) -> Result<T> {
    for nal in nalus {
        let (header, payload) = parse_nal_header(nal.as_ref())?;
        if header.nal_unit_type == expected_type && header.nuh_layer_id == 0 {
            let rbsp = strip_emulation_prevention(payload);
            return parse(&rbsp);
        }
    }
    Err(HeifError::MalformedHevc("required parameter set NAL not found in hvcC"))
}

/// Decodes one `hvc1` item (whether it's the sole primary image or one
/// tile of a `grid`) into its own picture.
/// The picture retains coded padding and the SPS conformance window for
/// reference prediction and bitstream comparisons. [`decode`] applies the
/// window to each image before grid assembly and container transforms.
/// Unstable: only with the `internals` feature.
#[cfg(feature = "internals")]
pub fn decode_one_image(image: &ResolvedImage) -> Result<(Sps, Picture)> {
    decode_with_reference(image, None)
}

#[cfg(not(feature = "internals"))]
fn decode_one_image(image: &ResolvedImage) -> Result<(Sps, Picture)> {
    decode_with_reference(image, None)
}

fn decode_with_reference(image: &ResolvedImage, reference: Option<&Picture>) -> Result<(Sps, Picture)> {
    let setup = timing::start(Stage::Headers);
    let sps: Sps = find_and_parse(image.hevc_config.nalus_of_type(NAL_SPS), NAL_SPS, parse_sps)?;
    let pps: Pps = find_and_parse(image.hevc_config.nalus_of_type(NAL_PPS), NAL_PPS, parse_pps)?;

    if sps.chroma_format_idc > 3 {
        return Err(HeifError::Unsupported("decode: unsupported chroma format"));
    }
    if pps.sao_offset_scale[0] > sps.bit_depth_luma.saturating_sub(10) || pps.sao_offset_scale[1] > sps.bit_depth_chroma.saturating_sub(10) {
        return Err(HeifError::MalformedHevc("decode: SAO offset scale exceeds bit depth"));
    }
    if pps.tiles_enabled && (pps.num_tile_columns != 1 || pps.num_tile_rows != 1) {
        return Err(HeifError::Unsupported("decode: multi-tile pictures are not supported"));
    }

    let length_size = image.hevc_config.length_size_minus_one as usize + 1;
    let slice_nalus = split_length_prefixed_nalus(&image.data, length_size)?;

    let sub_w = if sps.chroma_format_idc == 0 { 0 } else { sps.sub_width_c() as usize };
    let sub_h = if sps.chroma_format_idc == 0 { 0 } else { sps.sub_height_c() as usize };
    let mut pic = Picture::new(
        sps.pic_width_in_luma_samples as usize,
        sps.pic_height_in_luma_samples as usize,
        sub_w,
        sub_h,
        sps.bit_depth_luma,
        sps.bit_depth_chroma,
        sps.log2_min_luma_coding_block_size,
        sps.ctb_log2_size_y(),
    );

    let mut slices = Vec::new();
    for nal in &slice_nalus {
        let (header, payload) = parse_nal_header(nal)?;
        // Multilayer items can interleave enhancement-layer VCL NALs.
        // This decoder resolves the base layer; only its slices form the
        // independent-slice picture being reconstructed here.
        if !hevc::nal::is_slice_nal(header.nal_unit_type) || header.nuh_layer_id != 0 {
            continue;
        }
        if !is_irap_nal(header.nal_unit_type) && reference.is_none() {
            return Err(HeifError::Unsupported("decode: non-IRAP slice NAL in a still image"));
        }
        let rbsp = strip_emulation_prevention(payload);
        let (slice_header, byte_pos) = parse_slice_header(&rbsp, header.nal_unit_type, &sps, &pps)?;
        slices.push((slice_header, byte_pos, rbsp, payload));
    }
    if slices.is_empty() {
        return Err(HeifError::MalformedHevc("decode: no slice NAL found"));
    }
    let ctb = 1usize << sps.ctb_log2_size_y();
    let columns = (sps.pic_width_in_luma_samples as usize).div_ceil(ctb);
    let rows = (sps.pic_height_in_luma_samples as usize).div_ceil(ctb);
    if slices[0].0.slice_segment_address != 0 {
        return Err(HeifError::MalformedHevc("decode: missing first slice"));
    }
    drop(setup);
    for (index, (slice_header, byte_pos, rbsp, payload)) in slices.iter().enumerate() {
        let byte_pos = *byte_pos;
        let address = slice_header.slice_segment_address;
        let next = slices.get(index + 1).map(|s| s.0.slice_segment_address).unwrap_or(columns * rows);
        if address >= next {
            return Err(HeifError::MalformedHevc("decode: overlapping or unordered slices"));
        }
        if !address.is_multiple_of(columns) || !next.is_multiple_of(columns) {
            return Err(HeifError::Unsupported("decode: slices must cover complete CTU rows"));
        }
        if slices.len() > 1 && (slice_header.inter.is_some() || slice_header.sao_luma || slice_header.sao_chroma || !slice_header.deblocking_filter_disabled) {
            return Err(HeifError::Unsupported("decode: multiple slices require intra coding and disabled loop filters"));
        }
        if let Some(inter) = &slice_header.inter {
            if sps.scaling_list_enabled || pps.scaling_list.is_some() || pps.constrained_intra_pred {
                return Err(HeifError::Unsupported("P picture: scaling lists and constrained intra prediction are not supported"));
            }
            let reference = reference.ok_or(HeifError::Unsupported("P picture requires its prediction dependency"))?;
            if sps.bit_depth_luma != 8
                || sps.bit_depth_chroma != 8
                || sps.chroma_format_idc != 1
                || reference.y.width != pic.y.width
                || reference.y.height != pic.y.height
                || reference.y.bit_depth != 8
                || reference.cb.bit_depth != 8
                || reference.cr.bit_depth != 8
                || reference.cb.width != pic.cb.width
                || reference.cb.height != pic.cb.height
            {
                return Err(HeifError::Unsupported("P picture: requires matching 8-bit 4:2:0 IDR reference"));
            }
            if inter.poc_lsb == 0 || inter.reference_deltas != vec![-(inter.poc_lsb as i32)] || inter.cabac_init_b {
                return Err(HeifError::Unsupported("P picture: requires sole IDR POC-0 reference and P CABAC initialization"));
            }
        }
        let slice_data = rbsp.get(byte_pos..).ok_or(HeifError::MalformedHevc("decode: slice data offset out of range"))?;

        // The slice header's entry-point offsets (WPP/tile substream
        // lengths) are defined in EBSP byte space (clause 7.4.7.1), but
        // `slice_data` is RBSP (emulation-prevention already removed) —
        // convert via `payload`, the original not-yet-stripped bytes.
        let mut substream_starts = Vec::with_capacity(slice_header.entry_point_offsets.len());
        if !slice_header.entry_point_offsets.is_empty() {
            let mut cumulative_ebsp = rbsp_len_to_ebsp_pos(payload, byte_pos)?;
            let ebsp_positions: Vec<usize> = slice_header
                .entry_point_offsets
                .iter()
                .map(|&delta| {
                    cumulative_ebsp = cumulative_ebsp.saturating_add(delta as usize);
                    cumulative_ebsp
                })
                .collect();
            for rbsp_pos_in_full in ebsp_lens_to_rbsp_lens(payload, &ebsp_positions) {
                let rbsp_pos_in_slice_data =
                    rbsp_pos_in_full.checked_sub(byte_pos).ok_or(HeifError::MalformedHevc("decode: entry point before slice data start"))?;
                substream_starts.push(rbsp_pos_in_slice_data);
            }
        }

        let timer = timing::start(Stage::SliceData);
        decode_slice_rows(
            &mut pic,
            slice_data,
            &substream_starts,
            &sps,
            &pps,
            slice_header.slice_qp_y,
            slice_header.sao_luma,
            slice_header.sao_chroma,
            slice_header.cb_qp_offset,
            slice_header.cr_qp_offset,
            slice_header.inter.as_ref(),
            reference,
            address / columns,
            next / columns,
        )?;
        drop(timer);

        if !slice_header.deblocking_filter_disabled {
            let _timer = timing::start(Stage::Deblock);
            deblock_picture(
                &mut pic,
                &sps,
                slice_header.beta_offset_div2,
                slice_header.tc_offset_div2,
                pps.cb_qp_offset + slice_header.cb_qp_offset,
                pps.cr_qp_offset + slice_header.cr_qp_offset,
            )?;
        }
        if slice_header.sao_luma || slice_header.sao_chroma {
            let _timer = timing::start(Stage::Sao);
            apply_sao(&mut pic, &sps, slice_header.sao_luma, slice_header.sao_chroma)?;
        }
    }
    Ok((sps, pic))
}

fn image_from_picture(sps: &Sps, pic: Picture) -> Result<DecodedImage> {
    let _timer = timing::start(Stage::Pack);
    let cb_width = pic.cb.width;
    let cb_height = pic.cb.height;
    let image = DecodedImage {
        alpha: None,
        width: sps.pic_width_in_luma_samples as usize,
        height: sps.pic_height_in_luma_samples as usize,
        bit_depth_luma: sps.bit_depth_luma,
        bit_depth_chroma: sps.bit_depth_chroma,
        chroma_format_idc: sps.chroma_format_idc,
        color_info: sps.color_info,
        chroma_sampling: ChromaSampling::coded(sps.chroma_format_idc, sps.chroma_location),
        y: pic.y.into_samples(),
        cb: pic.cb.into_samples(),
        cr: pic.cr.into_samples(),
        cb_width,
        cb_height,
        metadata: isobmff::Metadata::default(),
    };
    if sps.conformance_window == [0; 4] {
        return Ok(image);
    }
    let [left, right, top, bottom] = sps.conformance_window.map(|v| v as usize);
    let width = image
        .width
        .checked_sub(left)
        .and_then(|v| v.checked_sub(right))
        .filter(|&v| v != 0)
        .ok_or(HeifError::MalformedHevc("SPS: conformance window exceeds picture"))?;
    let height = image
        .height
        .checked_sub(top)
        .and_then(|v| v.checked_sub(bottom))
        .filter(|&v| v != 0)
        .ok_or(HeifError::MalformedHevc("SPS: conformance window exceeds picture"))?;
    let (sub_w, sub_h) = chroma_subsampling(image.chroma_format_idc);
    extract_window(image, ImageWindow { left, top, width, height }, sub_w, sub_h)
}

/// Copies `tile` into `canvas` at the raster position implied by
/// `tile_idx`/`columns`, clamped to the canvas's own dimensions (the
/// last row/column of tiles is typically only partially used — the
/// encoder pads tiles out to a uniform size and the grid descriptor's
/// `output_width`/`output_height` says how much of the last tile is
/// actually part of the picture). Chroma-subsampling-aware.
#[allow(clippy::too_many_arguments)]
/// Where one tile lands in one canvas plane: `width x height` samples from
/// `source` (rows `source_width` apart) at `(x, y)`.
struct Placement<'a> {
    source: &'a [i32],
    source_width: usize,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
}

impl Placement<'_> {
    /// The source and canvas ranges of row `row`.
    fn rows(&self, row: usize, canvas_width: usize) -> (std::ops::Range<usize>, std::ops::Range<usize>) {
        let source = row * self.source_width;
        let canvas = (self.y + row) * canvas_width + self.x;
        (source..source + self.width, canvas..canvas + self.width)
    }
}

/// Copies every tile into the canvas. Each tile covers `tile_width x
/// tile_height` at its grid position, cropped to the canvas.
fn blit_tiles(canvas: &mut DecodedImage, tiles: &[DecodedImage], columns: u32, tile_width: usize, tile_height: usize) -> Result<()> {
    let (sub_w, sub_h) = chroma_subsampling(canvas.chroma_format_idc);
    let mut luma = Vec::with_capacity(tiles.len());
    let mut chroma = Vec::with_capacity(tiles.len());
    for (idx, tile) in tiles.iter().enumerate() {
        let x = idx % columns as usize * tile_width;
        let y = idx / columns as usize * tile_height;
        let width = tile.width.min(canvas.width.saturating_sub(x));
        let height = tile.height.min(canvas.height.saturating_sub(y));
        luma.push(Placement { source: &tile.y, source_width: tile.width, x, y, width, height });
        if canvas.chroma_format_idc != 0 {
            let place = |source| Placement {
                source,
                source_width: tile.cb_width,
                x: x / sub_w,
                y: y / sub_h,
                width: width.div_ceil(sub_w),
                height: height.div_ceil(sub_h),
            };
            chroma.push((place(&tile.cb), place(&tile.cr)));
        }
    }

    // Every row is checked first, tile by tile, luma before chroma, so a bad
    // tile fails as it would copying one tile at a time.
    for (idx, place) in luma.iter().enumerate() {
        for row in 0..place.height {
            let (source, target) = place.rows(row, canvas.width);
            if source.end > place.source.len() {
                return Err(HeifError::MalformedHevc("blit_tile: luma source row out of range"));
            }
            if target.end > canvas.y.len() {
                return Err(HeifError::MalformedHevc("blit_tile: luma dest row out of range"));
            }
        }
        for place in chroma.get(idx).into_iter().flat_map(|(cb, cr)| [cb, cr]) {
            for row in 0..place.height {
                let (source, target) = place.rows(row, canvas.cb_width);
                if source.end > place.source.len() {
                    return Err(HeifError::MalformedHevc("blit_tile: chroma source row out of range"));
                }
                if target.end > canvas.cb.len() {
                    return Err(HeifError::MalformedHevc("blit_tile: chroma dest row out of range"));
                }
            }
        }
    }

    // Canvas rows are filled in parallel. A row takes its tiles in order, so
    // where tiles overlap the later one wins, as copying tile by tile.
    fn fill(plane: &mut [i32], width: usize, places: &[&Placement]) {
        use rayon::prelude::*;
        if width == 0 {
            return;
        }
        plane.par_chunks_mut(width).with_min_len(16).enumerate().for_each(|(y, row)| {
            for place in places.iter().filter(|place| (place.y..place.y + place.height).contains(&y)) {
                let (source, target) = place.rows(y - place.y, width);
                let target = target.start - y * width..target.end - y * width;
                if let (Some(source), Some(target)) = (place.source.get(source), row.get_mut(target)) {
                    target.copy_from_slice(source);
                }
            }
        });
    }
    fill(&mut canvas.y, canvas.width, &luma.iter().collect::<Vec<_>>());
    fill(&mut canvas.cb, canvas.cb_width, &chroma.iter().map(|(cb, _)| cb).collect::<Vec<_>>());
    fill(&mut canvas.cr, canvas.cb_width, &chroma.iter().map(|(_, cr)| cr).collect::<Vec<_>>());
    Ok(())
}

/// Decodes a HEIF/HEIC file's primary image — a single `hvc1` image, or
/// an identity-derived image, or a `grid` of `hvc1` tiles stitched into one picture (what real
/// full-resolution iPhone photos use) — to planar YCbCr.
/// With no still-image metadata, selects one sync sample from an image/video
/// sequence track. This does not implement playback, timing or edit lists.
pub fn decode(bytes: &[u8]) -> Result<DecodedImage> {
    decode_with(bytes, &DecodeOptions::default())
}

/// Settings for [`decode_with`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeOptions {
    /// Apply the container's clean aperture, rotation and mirroring
    /// (`clap`/`irot`/`imir`). With `false`, colour and alpha come back as
    /// coded: grid-assembled and cropped to the SPS conformance window only.
    pub apply_transforms: bool,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        DecodeOptions { apply_transforms: true }
    }
}

/// [`decode`] with explicit settings.
pub fn decode_with(bytes: &[u8], options: &DecodeOptions) -> Result<DecodedImage> {
    let mut parsed = isobmff::parse(bytes)?;
    let metadata = std::mem::take(&mut parsed.metadata);
    let mut image = decode_pixels(&parsed, options.apply_transforms)?;
    if let Some(auxiliary) = &parsed.alpha {
        let alpha = decode_pixels(&auxiliary.image, options.apply_transforms)?;
        if alpha.width != image.width || alpha.height != image.height {
            return Err(HeifError::Unsupported("alpha: display dimensions do not match colour image"));
        }
        image.alpha = Some(AlphaPlane { bit_depth: alpha.bit_depth_luma, samples: alpha.y, premultiplied: auxiliary.premultiplied });
    }
    image.color_info = metadata.color_info.or(image.color_info);
    image.metadata = metadata;
    Ok(image)
}

/// What a file declares, read from the container and parameter sets
/// without decoding any pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageInfo {
    /// Size after the container's crop, rotation and mirroring.
    pub width: usize,
    pub height: usize,
    /// Size before those transforms (after grid assembly and SPS cropping).
    pub coded_width: usize,
    pub coded_height: usize,
    pub bit_depth_luma: u32,
    pub bit_depth_chroma: u32,
    pub chroma_format_idc: u32,
    /// Bit depth of the auxiliary alpha image, if the file has one.
    pub alpha_bit_depth: Option<u32>,
}

/// Reads the primary image's size, bit depth and alpha presence without
/// decoding pixels, so callers can apply their own limits first.
pub fn probe(bytes: &[u8]) -> Result<ImageInfo> {
    let parsed = isobmff::parse(bytes)?;
    let (coded_width, coded_height, sps) = coded_size(&parsed)?;
    let (width, height) = transformed_size(coded_width, coded_height, &parsed.transforms)?;
    let alpha_bit_depth = match &parsed.alpha {
        Some(auxiliary) => Some(coded_size(&auxiliary.image)?.2.bit_depth_luma),
        None => None,
    };
    Ok(ImageInfo {
        width,
        height,
        coded_width,
        coded_height,
        bit_depth_luma: sps.bit_depth_luma,
        bit_depth_chroma: sps.bit_depth_chroma,
        chroma_format_idc: sps.chroma_format_idc,
        alpha_bit_depth,
    })
}

/// Output size before container transforms, and the first image's SPS.
fn coded_size(parsed: &isobmff::ParsedHeif) -> Result<(usize, usize, Sps)> {
    let image = parsed.images.first().ok_or(HeifError::MalformedBox("no image to decode"))?;
    let sps = find_and_parse(image.hevc_config.nalus_of_type(NAL_SPS), NAL_SPS, parse_sps)?;
    if let Some(grid) = parsed.grid {
        return Ok((grid.output_width as usize, grid.output_height as usize, sps));
    }
    let [left, right, top, bottom] = sps.conformance_window.map(|v| v as usize);
    let width = (sps.pic_width_in_luma_samples as usize).checked_sub(left + right).filter(|&v| v != 0);
    let height = (sps.pic_height_in_luma_samples as usize).checked_sub(top + bottom).filter(|&v| v != 0);
    match (width, height) {
        (Some(width), Some(height)) => Ok((width, height, sps)),
        _ => Err(HeifError::MalformedHevc("SPS: conformance window exceeds picture")),
    }
}

/// The size [`apply_transforms`] produces, without touching pixels.
fn transformed_size(mut width: usize, mut height: usize, transforms: &[isobmff::Transform]) -> Result<(usize, usize)> {
    for &transform in transforms {
        match transform {
            isobmff::Transform::CleanAperture(aperture) => {
                width = crop_axis(width, aperture.width, aperture.horizontal_offset)?.1;
                height = crop_axis(height, aperture.height, aperture.vertical_offset)?.1;
            }
            isobmff::Transform::Rotate90Ccw(turns) if turns % 2 == 1 => std::mem::swap(&mut width, &mut height),
            _ => {}
        }
    }
    Ok((width, height))
}

/// Reads the primary image's ICC profile, nclx colour description, EXIF
/// and XMP without decoding any pixels.
pub fn read_metadata(bytes: &[u8]) -> Result<Metadata> {
    Ok(isobmff::parse(bytes)?.metadata)
}

fn decode_pixels(parsed: &isobmff::ParsedHeif, transform: bool) -> Result<DecodedImage> {
    let transforms = if transform { parsed.transforms.as_slice() } else { &[] };
    let reference = parsed
        .reference_image
        .as_ref()
        .map(|image| {
            let nalus = split_length_prefixed_nalus(&image.data, image.hevc_config.length_size_minus_one as usize + 1)?;
            for nal in nalus {
                let (header, _) = parse_nal_header(&nal)?;
                if header.nuh_layer_id == 0 && hevc::nal::is_slice_nal(header.nal_unit_type) && !matches!(header.nal_unit_type, 19 | 20) {
                    return Err(HeifError::Unsupported("prediction dependency must be an IDR picture"));
                }
            }
            decode_one_image(image)
        })
        .transpose()?;

    let Some(grid) = parsed.grid else {
        let image = parsed.images.first().ok_or(HeifError::MalformedBox("no image to decode"))?;
        let (sps, pic) = decode_with_reference(image, reference.as_ref().map(|(_, pic)| pic))?;
        return apply_transforms(image_from_picture(&sps, pic)?, transforms);
    };

    if parsed.images.len() != (grid.rows as usize) * (grid.columns as usize) {
        return Err(HeifError::MalformedBox("grid image count does not match rows*columns"));
    }

    // Every tile is held in memory until the canvas is assembled, so bound
    // their total size before decoding any of them.
    let setup = timing::start(Stage::Headers);
    let mut tile_pixels = 0u64;
    for image in &parsed.images {
        let sps = find_and_parse(image.hevc_config.nalus_of_type(NAL_SPS), NAL_SPS, parse_sps)?;
        tile_pixels = tile_pixels.saturating_add(u64::from(sps.pic_width_in_luma_samples) * u64::from(sps.pic_height_in_luma_samples));
    }
    if tile_pixels > MAX_IMAGE_PIXELS {
        return Err(HeifError::LimitExceeded("grid: tiles larger than MAX_IMAGE_PIXELS"));
    }
    drop(setup);

    // Decode grid tiles in parallel using rayon
    use rayon::prelude::*;
    let tiles: Vec<DecodedImage> = parsed
        .images
        .par_iter()
        .map(|image| {
            let (sps, pic) = decode_one_image(image)?;
            image_from_picture(&sps, pic)
        })
        .collect::<Result<Vec<_>>>()?;
    let assembly = timing::start(Stage::Grid);
    let first = tiles.first().ok_or(HeifError::MalformedBox("grid has no tiles"))?;
    let (tile_width, tile_height) = (first.width, first.height);
    let (bit_depth_luma, bit_depth_chroma, chroma_format_idc) = (first.bit_depth_luma, first.bit_depth_chroma, first.chroma_format_idc);
    for t in &tiles {
        if t.bit_depth_luma != bit_depth_luma || t.bit_depth_chroma != bit_depth_chroma || t.chroma_format_idc != chroma_format_idc {
            return Err(HeifError::Unsupported("decode: grid tiles with inconsistent bit depth/chroma format are not supported"));
        }
    }

    let width = grid.output_width as usize;
    let height = grid.output_height as usize;
    let (sub_w, sub_h) = chroma_subsampling(chroma_format_idc);
    let cb_width = if chroma_format_idc == 0 { 0 } else { width.div_ceil(sub_w) };
    let cb_height = if chroma_format_idc == 0 { 0 } else { height.div_ceil(sub_h) };
    let mut canvas = DecodedImage {
        alpha: None,
        width,
        height,
        bit_depth_luma,
        bit_depth_chroma,
        chroma_format_idc,
        color_info: first.color_info,
        chroma_sampling: first.chroma_sampling,
        y: vec![0; width * height],
        cb: vec![0; cb_width * cb_height],
        cr: vec![0; cb_width * cb_height],
        cb_width,
        cb_height,
        metadata: isobmff::Metadata::default(),
    };
    blit_tiles(&mut canvas, &tiles, grid.columns, tile_width, tile_height)?;
    drop(assembly);
    apply_transforms(canvas, transforms)
}

/// A single plane's `width x height` raster, read via `(x, y)` for
/// clarity in the transform formulas below.
fn plane_get(plane: &[i32], width: usize, x: usize, y: usize) -> i32 {
    plane.get(y * width + x).copied().unwrap_or(0)
}

/// `irot`'s `angle` is a count of 90°-anticlockwise turns (clause
/// 6.5.10.2); rotating by that count once each is simplest and
/// provably equivalent to converting it to an-equivalent-clockwise
/// single rotation, with no separate "which direction does `angle=1`
/// really mean" table to get backwards. One CCW turn: `dst(dx,dy) =
/// src(w-1-dy, dx)`, output `width,height = height,width` — checked by
/// composing it with itself 4 times back to the identity (see tests).
/// Each output row reads one source column, so rows are split across
/// threads in bands and the source is walked in `BLOCK`-square tiles to
/// stay in cache.
fn rotate90_ccw_plane(src: &[i32], width: usize, height: usize) -> (Vec<i32>, usize, usize) {
    use rayon::prelude::*;
    const BLOCK: usize = 32;
    let (nw, nh) = (height, width);
    let mut dst = vec![0i32; nw * nh];
    if nw == 0 {
        return (dst, nw, nh);
    }
    dst.par_chunks_mut(nw * BLOCK).enumerate().for_each(|(band, rows)| {
        for dx0 in (0..nw).step_by(BLOCK) {
            for (r, row) in rows.chunks_mut(nw).enumerate() {
                let dy = band * BLOCK + r;
                for (dx, out) in row.iter_mut().enumerate().skip(dx0).take(BLOCK) {
                    *out = plane_get(src, width, width - 1 - dy, dx);
                }
            }
        }
    });
    (dst, nw, nh)
}

/// `imir` with `axis == 0`: flip top↔bottom.
fn mirror_vertical_plane(src: &[i32], width: usize, height: usize) -> Vec<i32> {
    let mut dst = vec![0i32; width * height];
    for y in 0..height {
        for x in 0..width {
            dst[y * width + x] = plane_get(src, width, x, height - 1 - y);
        }
    }
    dst
}

/// `imir` with `axis == 1`: flip left↔right.
fn mirror_horizontal_plane(src: &[i32], width: usize, height: usize) -> Vec<i32> {
    let mut dst = vec![0i32; width * height];
    for y in 0..height {
        for x in 0..width {
            dst[y * width + x] = plane_get(src, width, width - 1 - x, y);
        }
    }
    dst
}

/// Applies one `Transform` to every plane of a [`DecodedImage`],
/// luma and chroma each at their own (possibly subsampled) dimensions.
// Exact rational arithmetic avoids rounding an aperture onto the wrong pixel.
fn crop_axis(full: usize, extent: (u32, u32), offset: (i32, u32)) -> Result<(usize, usize)> {
    if extent.1 == 0 || offset.1 == 0 || extent.0 == 0 {
        return Err(HeifError::MalformedBox("clap: zero extent or denominator"));
    }
    if !extent.0.is_multiple_of(extent.1) {
        return Err(HeifError::Unsupported("clap: fractional pixel extent"));
    }
    let size = (extent.0 / extent.1) as usize;
    if size > full {
        return Err(HeifError::MalformedBox("clap: aperture exceeds image"));
    }
    let denominator = 2 * i128::from(offset.1);
    let numerator = (full - size) as i128 * i128::from(offset.1) + 2 * i128::from(offset.0);
    if numerator < 0 || numerator > (full - size) as i128 * denominator {
        return Err(HeifError::MalformedBox("clap: aperture outside image"));
    }
    if numerator % denominator != 0 {
        return Err(HeifError::Unsupported("clap: fractional pixel origin"));
    }
    Ok(((numerator / denominator) as usize, size))
}

fn crop_plane(src: &[i32], stride: usize, left: usize, top: usize, width: usize, height: usize) -> Result<Vec<i32>> {
    let mut out = Vec::new();
    for y in top..top + height {
        let start = y * stride + left;
        let row = src.get(start..start + width).ok_or(HeifError::MalformedBox("clap: plane crop out of bounds"))?;
        out.extend_from_slice(row);
    }
    Ok(out)
}

// Keep a luma-space rectangle on the original sampling grid until every
// transform is resolved. Cropping chroma after each step would discard
// the remainder of odd offsets and change the result of nested crops.
#[derive(Clone, Copy)]
struct ImageWindow {
    left: usize,
    top: usize,
    width: usize,
    height: usize,
}

fn chroma_subsampling(format: u32) -> (usize, usize) {
    match format {
        1 => (2, 2),
        2 => (2, 1),
        _ => (1, 1),
    }
}

fn extract_window(img: DecodedImage, window: ImageWindow, sub_w: usize, sub_h: usize) -> Result<DecodedImage> {
    let ImageWindow { left, top, width, height } = window;
    let cb_width = if img.chroma_format_idc == 0 { 0 } else { width.div_ceil(sub_w) };
    let cb_height = if img.chroma_format_idc == 0 { 0 } else { height.div_ceil(sub_h) };
    let y = crop_plane(&img.y, img.width, left, top, width, height)?;
    // Native planar extraction: retain source samples, selecting chroma
    // from the cell containing the final luma origin. No interpolation.
    let cb = if img.chroma_format_idc == 0 { Vec::new() } else { crop_plane(&img.cb, img.cb_width, left / sub_w, top / sub_h, cb_width, cb_height)? };
    let cr = if img.chroma_format_idc == 0 { Vec::new() } else { crop_plane(&img.cr, img.cb_width, left / sub_w, top / sub_h, cb_width, cb_height)? };
    let mut chroma_sampling = img.chroma_sampling;
    chroma_sampling.origin_x -= (left % sub_w) as f32;
    chroma_sampling.origin_y -= (top % sub_h) as f32;
    Ok(DecodedImage { width, height, cb_width, cb_height, y, cb, cr, chroma_sampling, ..img })
}

fn apply_one_transform(img: DecodedImage, t: isobmff::Transform) -> Result<DecodedImage> {
    let mut result = match t {
        isobmff::Transform::CleanAperture(_) => return Err(HeifError::MalformedBox("crop must be resolved as an image window")),
        isobmff::Transform::Rotate90Ccw(count) => {
            let mut img = img;
            for _ in 0..(count % 4) {
                let sampling = img.chroma_sampling;
                let chroma_sampling = ChromaSampling {
                    sub_x: sampling.sub_y,
                    sub_y: sampling.sub_x,
                    origin_x: sampling.origin_y,
                    origin_y: (img.width - 1) as f32 - sampling.origin_x - img.cb_width.saturating_sub(1) as f32 * sampling.sub_x as f32,
                };
                let (y, nw, nh) = rotate90_ccw_plane(&img.y, img.width, img.height);
                let (cb, ncw, nch) = rotate90_ccw_plane(&img.cb, img.cb_width, img.cb_height);
                let (cr, _, _) = rotate90_ccw_plane(&img.cr, img.cb_width, img.cb_height);
                img = DecodedImage { width: nw, height: nh, y, cb, cr, cb_width: ncw, cb_height: nch, chroma_sampling, ..img };
            }
            img
        }
        isobmff::Transform::MirrorVertical => {
            let mut chroma_sampling = img.chroma_sampling;
            chroma_sampling.origin_y =
                (img.height - 1) as f32 - chroma_sampling.origin_y - img.cb_height.saturating_sub(1) as f32 * chroma_sampling.sub_y as f32;
            let y = mirror_vertical_plane(&img.y, img.width, img.height);
            let cb = mirror_vertical_plane(&img.cb, img.cb_width, img.cb_height);
            let cr = mirror_vertical_plane(&img.cr, img.cb_width, img.cb_height);
            DecodedImage { y, cb, cr, chroma_sampling, ..img }
        }
        isobmff::Transform::MirrorHorizontal => {
            let mut chroma_sampling = img.chroma_sampling;
            chroma_sampling.origin_x = (img.width - 1) as f32 - chroma_sampling.origin_x - img.cb_width.saturating_sub(1) as f32 * chroma_sampling.sub_x as f32;
            let y = mirror_horizontal_plane(&img.y, img.width, img.height);
            let cb = mirror_horizontal_plane(&img.cb, img.cb_width, img.cb_height);
            let cr = mirror_horizontal_plane(&img.cr, img.cb_width, img.cb_height);
            DecodedImage { y, cb, cr, chroma_sampling, ..img }
        }
    };
    if result.chroma_format_idc == 0 {
        result.chroma_sampling = ChromaSampling::coded(0, None);
    }
    Ok(result)
}

fn apply_transforms(mut img: DecodedImage, transforms: &[isobmff::Transform]) -> Result<DecodedImage> {
    let _timer = timing::start(Stage::Transforms);
    let (mut sub_w, mut sub_h) = chroma_subsampling(img.chroma_format_idc);
    let mut window = ImageWindow { left: 0, top: 0, width: img.width, height: img.height };
    for &transform in transforms {
        match transform {
            isobmff::Transform::CleanAperture(aperture) => {
                let (dx, width) = crop_axis(window.width, aperture.width, aperture.horizontal_offset)?;
                let (dy, height) = crop_axis(window.height, aperture.height, aperture.vertical_offset)?;
                window = ImageWindow { left: window.left + dx, top: window.top + dy, width, height };
            }
            isobmff::Transform::Rotate90Ccw(turns) => {
                for _ in 0..turns % 4 {
                    window = ImageWindow { left: window.top, top: img.width - window.left - window.width, width: window.height, height: window.width };
                    img = apply_one_transform(img, isobmff::Transform::Rotate90Ccw(1))?;
                    std::mem::swap(&mut sub_w, &mut sub_h);
                }
            }
            isobmff::Transform::MirrorHorizontal => {
                window.left = img.width - window.left - window.width;
                img = apply_one_transform(img, transform)?;
            }
            isobmff::Transform::MirrorVertical => {
                window.top = img.height - window.top - window.height;
                img = apply_one_transform(img, transform)?;
            }
        }
    }
    if window.left == 0 && window.top == 0 && window.width == img.width && window.height == img.height {
        return Ok(img);
    }
    extract_window(img, window, sub_w, sub_h)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Grid assembly as it was before rows were filled in parallel: one
    /// tile at a time, in order.
    fn blit_tile_by_tile(canvas: &mut DecodedImage, tiles: &[DecodedImage], columns: u32, tile_width: usize, tile_height: usize) -> Result<()> {
        for (idx, tile) in tiles.iter().enumerate() {
            let dst_x = idx % columns as usize * tile_width;
            let dst_y = idx / columns as usize * tile_height;
            let copy_w = tile.width.min(canvas.width.saturating_sub(dst_x));
            let copy_h = tile.height.min(canvas.height.saturating_sub(dst_y));
            for row in 0..copy_h {
                let (s, d) = (row * tile.width, (dst_y + row) * canvas.width + dst_x);
                let src = tile.y.get(s..s + copy_w).ok_or(HeifError::MalformedHevc("blit_tile: luma source row out of range"))?;
                let dst = canvas.y.get_mut(d..d + copy_w).ok_or(HeifError::MalformedHevc("blit_tile: luma dest row out of range"))?;
                dst.copy_from_slice(src);
            }
            if canvas.chroma_format_idc == 0 {
                continue;
            }
            let (sub_w, sub_h) = chroma_subsampling(canvas.chroma_format_idc);
            let (c_copy_w, c_copy_h) = (copy_w.div_ceil(sub_w), copy_h.div_ceil(sub_h));
            for row in 0..c_copy_h {
                let (s, d) = (row * tile.cb_width, (dst_y / sub_h + row) * canvas.cb_width + dst_x / sub_w);
                for (src, dst) in [(&tile.cb, &mut canvas.cb), (&tile.cr, &mut canvas.cr)] {
                    let src = src.get(s..s + c_copy_w).ok_or(HeifError::MalformedHevc("blit_tile: chroma source row out of range"))?;
                    let dst = dst.get_mut(d..d + c_copy_w).ok_or(HeifError::MalformedHevc("blit_tile: chroma dest row out of range"))?;
                    dst.copy_from_slice(src);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn grid_rows_filled_in_parallel_match_tile_by_tile_assembly() {
        let image = |width: usize, height: usize, format: u32, seed: i32| {
            let (sub_w, sub_h) = chroma_subsampling(format);
            let (cb_width, cb_height) = if format == 0 { (0, 0) } else { (width.div_ceil(sub_w), height.div_ceil(sub_h)) };
            let plane = |n: usize, k: i32| (0..n as i32).map(|i| (i * 7 + seed * 31 + k) % 256).collect::<Vec<_>>();
            DecodedImage {
                alpha: None,
                width,
                height,
                bit_depth_luma: 8,
                bit_depth_chroma: 8,
                chroma_format_idc: format,
                color_info: None,
                chroma_sampling: ChromaSampling::coded(format, None),
                y: plane(width * height, 0),
                cb: plane(cb_width * cb_height, 1),
                cr: plane(cb_width * cb_height, 2),
                cb_width,
                cb_height,
                metadata: isobmff::Metadata::default(),
            }
        };
        for format in 0..=3 {
            // Canvas size, grid columns, cell size, and each tile's size: tiles
            // that overhang their cell (and overlap the next), fall short of
            // it, or are cropped by the canvas edge.
            type Case = (usize, usize, u32, usize, usize, &'static [(usize, usize)]);
            let cases: [Case; 4] = [
                (40, 24, 2, 20, 12, &[(20, 12); 4]),
                (36, 20, 2, 20, 12, &[(20, 12); 4]),
                (40, 24, 2, 20, 12, &[(24, 16), (20, 12), (16, 8), (24, 14)]),
                (40, 30, 3, 16, 16, &[(16, 16); 6]),
            ];
            for (width, height, columns, tile_width, tile_height, sizes) in cases {
                let tiles = || sizes.iter().enumerate().map(|(i, &(w, h))| image(w, h, format, i as i32)).collect::<Vec<_>>();
                let canvas = || {
                    let mut canvas = image(width, height, format, 0);
                    for plane in [&mut canvas.y, &mut canvas.cb, &mut canvas.cr] {
                        plane.fill(0);
                    }
                    canvas
                };
                let (mut expected, mut actual) = (canvas(), canvas());
                blit_tile_by_tile(&mut expected, &tiles(), columns, tile_width, tile_height).unwrap();
                blit_tiles(&mut actual, &tiles(), columns, tile_width, tile_height).unwrap();
                assert!(
                    (actual.y == expected.y) && (actual.cb == expected.cb) && (actual.cr == expected.cr),
                    "format {format}, canvas {width}x{height}, tiles {sizes:?}"
                );

                // A tile whose planes are too short fails with the same error.
                for plane in 0..3 {
                    let mut short = tiles();
                    let tile = &mut short[2];
                    let target = match plane {
                        0 => &mut tile.y,
                        1 => &mut tile.cb,
                        _ => &mut tile.cr,
                    };
                    if target.is_empty() {
                        continue;
                    }
                    target.truncate(target.len() / 2);
                    let want = blit_tile_by_tile(&mut canvas(), &short, columns, tile_width, tile_height).unwrap_err().to_string();
                    assert_eq!(blit_tiles(&mut canvas(), &short, columns, tile_width, tile_height).unwrap_err().to_string(), want);
                }
            }
        }
    }

    #[test]
    fn rotated_422_crop_retains_vertical_chroma_subsampling() {
        let img = DecodedImage {
            alpha: None,
            width: 8,
            height: 6,
            cb_width: 4,
            cb_height: 6,
            metadata: Default::default(),
            bit_depth_luma: 10,
            bit_depth_chroma: 10,
            chroma_format_idc: 2,
            color_info: None,
            chroma_sampling: ChromaSampling::coded(2, None),
            y: (0..48).collect(),
            cb: (0..24).collect(),
            cr: (100..124).collect(),
        };
        let crop =
            isobmff::Transform::CleanAperture(isobmff::CleanAperture { width: (4, 1), height: (4, 1), horizontal_offset: (0, 1), vertical_offset: (0, 1) });
        let img = apply_transforms(img, &[isobmff::Transform::Rotate90Ccw(1), crop]).unwrap();
        assert_eq!((img.width, img.height, img.cb_width, img.cb_height), (4, 4, 4, 2));
        assert_eq!(img.y, [13, 21, 29, 37, 12, 20, 28, 36, 11, 19, 27, 35, 10, 18, 26, 34]);
        assert_eq!(img.cb, [6, 10, 14, 18, 5, 9, 13, 17]);
        assert_eq!(img.cr, [106, 110, 114, 118, 105, 109, 113, 117]);
    }

    #[test]
    fn nested_odd_crops_keep_the_original_chroma_grid() {
        let image = || DecodedImage {
            alpha: None,
            width: 12,
            height: 12,
            cb_width: 6,
            cb_height: 6,
            metadata: Default::default(),
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
            chroma_format_idc: 1,
            color_info: None,
            chroma_sampling: ChromaSampling::coded(1, None),
            y: (0..144).collect(),
            cb: (0..36).collect(),
            cr: (100..136).collect(),
        };
        let crop = |size| {
            isobmff::Transform::CleanAperture(isobmff::CleanAperture {
                width: (size, 1),
                height: (size, 1),
                horizontal_offset: (0, 1),
                vertical_offset: (0, 1),
            })
        };
        // Two offsets of one luma sample add to one chroma sample.
        // Rounding each offset separately would incorrectly keep chroma (0,0).
        let nested = apply_transforms(image(), &[crop(10), crop(8)]).unwrap();
        let single = apply_transforms(image(), &[crop(8)]).unwrap();
        assert_eq!(nested.y, single.y);
        assert_eq!(nested.cb, single.cb);
        assert_eq!(nested.cr, single.cr);
        assert_eq!(nested.y[0], 26);
        assert_eq!(nested.cb[0], 7);
        // An odd final origin selects the enclosing native chroma cell.
        let odd = apply_transforms(image(), &[crop(10)]).unwrap();
        assert_eq!((odd.width, odd.cb_width), (10, 5));
        assert_eq!(odd.y[0], 13);
        assert_eq!(odd.cb[0], 0);
    }

    #[test]
    fn crop_window_tracks_rotation_and_mirroring() {
        let img = DecodedImage {
            alpha: None,
            width: 12,
            height: 8,
            cb_width: 6,
            cb_height: 4,
            metadata: Default::default(),
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
            chroma_format_idc: 1,
            color_info: None,
            chroma_sampling: ChromaSampling::coded(1, None),
            y: (0..96).collect(),
            cb: (0..24).collect(),
            cr: (100..124).collect(),
        };
        let crop = |w, h| {
            isobmff::Transform::CleanAperture(isobmff::CleanAperture { width: (w, 1), height: (h, 1), horizontal_offset: (0, 1), vertical_offset: (0, 1) })
        };
        let img = apply_transforms(img, &[crop(10, 6), isobmff::Transform::Rotate90Ccw(1), crop(4, 8), isobmff::Transform::MirrorHorizontal]).unwrap();
        assert_eq!((img.width, img.height, img.cb_width, img.cb_height), (4, 8, 2, 4));
        for y in 0..8 {
            for x in 0..4 {
                assert_eq!(img.y[y * 4 + x], ((5 - x) * 12 + 9 - y) as i32);
            }
        }
        for y in 0..4 {
            for x in 0..2 {
                let expected = ((2 - x) * 6 + 4 - y) as i32;
                assert_eq!(img.cb[y * 2 + x], expected);
                assert_eq!(img.cr[y * 2 + x], expected + 100);
            }
        }
    }

    #[test]
    fn clean_aperture_geometry_uses_signed_rational_offsets() {
        assert_eq!(crop_axis(1280, (300, 1), (0, 1)).unwrap(), (490, 300));
        assert_eq!(crop_axis(720, (300, 1), (0, 1)).unwrap(), (210, 300));
        assert_eq!(crop_axis(10, (4, 1), (-2, 1)).unwrap(), (1, 4));
        assert_eq!(crop_axis(9, (4, 1), (-1, 2)).unwrap(), (2, 4));
        assert!(crop_axis(10, (4, 0), (0, 1)).is_err());
        assert!(crop_axis(10, (4, 1), (0, 0)).is_err());
        assert!(crop_axis(10, (11, 1), (0, 1)).is_err());
        assert!(crop_axis(10, (0, 1), (0, 1)).is_err());
        assert!(crop_axis(10, (4, 1), (-4, 1)).is_err());
        assert!(crop_axis(10, (4, 1), (4, 1)).is_err());
        assert!(crop_axis(10, (3, 2), (0, 1)).is_err());
        assert!(crop_axis(10, (4, 1), (1, 2)).is_err());
    }

    #[test]
    fn crop_precedes_rotation_and_preserves_chroma_planes() {
        let img = DecodedImage {
            alpha: None,
            width: 8,
            height: 4,
            cb_width: 4,
            cb_height: 2,
            metadata: Default::default(),
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
            chroma_format_idc: 1,
            color_info: None,
            chroma_sampling: ChromaSampling::coded(1, None),
            y: (0..32).collect(),
            cb: (100..108).collect(),
            cr: (200..208).collect(),
        };
        let aperture = isobmff::CleanAperture { width: (4, 1), height: (4, 1), horizontal_offset: (0, 1), vertical_offset: (0, 1) };
        let img = apply_transforms(img, &[isobmff::Transform::CleanAperture(aperture), isobmff::Transform::Rotate90Ccw(1)]).unwrap();
        assert_eq!((img.width, img.height, img.cb_width, img.cb_height), (4, 4, 2, 2));
        assert_eq!(img.y, vec![5, 13, 21, 29, 4, 12, 20, 28, 3, 11, 19, 27, 2, 10, 18, 26]);
        assert_eq!(img.cb, vec![102, 106, 101, 105]);
        assert_eq!(img.cr, vec![202, 206, 201, 205]);
    }

    /// A 2x2 plane `[[1,2],[3,4]]` (row-major `[1,2,3,4]`) rotated 90°
    /// anticlockwise becomes `[[2,4],[1,3]]` — hand-traced from the
    /// rotation's own `dst(dx,dy) = src(w-1-dy, dx)` definition.
    #[test]
    fn rotate90_ccw_matches_hand_traced_example() {
        let (dst, w, h) = rotate90_ccw_plane(&[1, 2, 3, 4], 2, 2);
        assert_eq!((w, h), (2, 2));
        assert_eq!(dst, vec![2, 4, 1, 3]);
    }

    #[test]
    fn rotate90_ccw_four_times_is_identity() {
        let src = vec![1, 2, 3, 4, 5, 6]; // 3x2
        let (mut cur, mut w, mut h) = (src.clone(), 3, 2);
        for _ in 0..4 {
            let (d, nw, nh) = rotate90_ccw_plane(&cur, w, h);
            cur = d;
            w = nw;
            h = nh;
        }
        assert_eq!((w, h), (3, 2));
        assert_eq!(cur, src);
    }

    #[test]
    fn mirror_vertical_flips_top_and_bottom_rows() {
        let src = vec![1, 2, 3, 4, 5, 6]; // 2x3: rows [1,2] [3,4] [5,6]
        assert_eq!(mirror_vertical_plane(&src, 2, 3), vec![5, 6, 3, 4, 1, 2]);
    }

    #[test]
    fn mirror_horizontal_flips_left_and_right_columns() {
        let src = vec![1, 2, 3, 4, 5, 6]; // 2x3: rows [1,2] [3,4] [5,6]
        assert_eq!(mirror_horizontal_plane(&src, 2, 3), vec![2, 1, 4, 3, 6, 5]);
    }
}
