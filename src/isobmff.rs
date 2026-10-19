//! ISOBMFF container parsing, scoped to what a HEIF still image needs:
//! `ftyp`, `meta` (`hdlr`, `pitm`, `iinf`/`infe`, `iloc`, `iprp`/`ipco`/`ipma`,
//! `idat`) and resolving the primary item's HEVC decoder configuration
//! (`hvcC`, ISO/IEC 14496-15) plus its coded sample bytes from `mdat`.
//!
//! Box layout (ISO/IEC 14496-12 clause 4.2) and the HEVC sample entry
//! (ISO/IEC 14496-15 clause 8.3.3) are implemented from the public
//! standards' well-known, widely-published structure — not copied from any
//! other implementation's source.

use crate::bitreader::ByteReader;
use crate::error::{HeifError, Result};
use std::collections::BTreeMap;

mod sequence;

const MAX_BOX_RECURSION: u32 = 32;

struct BoxHeader {
    kind: [u8; 4],
    /// Byte range of the box's *body* (after the header) within the buffer
    /// passed to the parser that found it.
    body_start: usize,
    body_end: usize,
}

/// Reads one box header at the reader's current position and returns it,
/// leaving the reader positioned at the start of the box body.
fn read_box_header(r: &mut ByteReader<'_>) -> Result<BoxHeader> {
    let box_start = r.pos();
    let size32 = r.u32()?;
    let kind = r.fourcc()?;
    let header_len: u64 = 8;
    let size: u64 = match size32 {
        0 => {
            // Extends to end of the enclosing buffer; caller clamps.
            u64::MAX
        }
        1 => r.u64()?,
        n => n as u64,
    };
    let extra_header = if size32 == 1 { 8 } else { 0 };
    let body_start = r.pos();
    let total_len = header_len + extra_header;
    if size != u64::MAX {
        let body_len = size.checked_sub(total_len).ok_or(HeifError::MalformedBox("box size smaller than its header"))?;
        let body_end = (box_start as u64).checked_add(size).ok_or(HeifError::MalformedBox("box size overflow"))?;
        let body_end = usize::try_from(body_end).map_err(|_| HeifError::MalformedBox("box size overflow"))?;
        let _ = body_len;
        Ok(BoxHeader { kind, body_start, body_end })
    } else {
        Ok(BoxHeader { kind, body_start, body_end: usize::MAX })
    }
}

/// Iterates sibling boxes inside `data[range]`, calling `f(kind, body_bytes)`
/// for each. `body_end == usize::MAX` (a size-0 "to EOF" box) is clamped to
/// the slice's own end.
fn for_each_child<'a>(data: &'a [u8], start: usize, end: usize, depth: u32, mut f: impl FnMut(&[u8; 4], &'a [u8]) -> Result<()>) -> Result<()> {
    if depth > MAX_BOX_RECURSION {
        return Err(HeifError::MalformedBox("box nesting too deep"));
    }
    let end = end.min(data.len());
    let slice = data.get(start..end).ok_or(HeifError::MalformedBox("child range out of bounds"))?;
    let mut r = ByteReader::new(slice);
    while r.remaining() >= 8 {
        let before = r.pos();
        let hdr = read_box_header(&mut r)?;
        // `hdr`'s offsets come from a reader constructed fresh over `slice`
        // (see below), so they're already local to it — no further
        // adjustment by `start` here.
        let body_end_local = if hdr.body_end == usize::MAX { slice.len() } else { hdr.body_end };
        let body_start_local = hdr.body_start;
        let body = slice.get(body_start_local..body_end_local).ok_or(HeifError::MalformedBox("box body out of bounds"))?;
        f(&hdr.kind, body)?;
        if body_end_local <= before {
            return Err(HeifError::MalformedBox("non-advancing box"));
        }
        r.seek(body_end_local)?;
    }
    Ok(())
}

/// A "full box" (`version` + 24-bit `flags`) header, clause 4.2.
fn full_box_header(r: &mut ByteReader<'_>) -> Result<(u8, u32)> {
    let version = r.u8()?;
    let flags = r.u24()?;
    Ok((version, flags))
}

#[derive(Debug, Clone)]
pub struct HevcNalArray {
    pub nal_unit_type: u8,
    pub nalus: Vec<Vec<u8>>,
}

#[derive(Debug, Clone, Default)]
pub struct HevcDecoderConfig {
    pub general_profile_idc: u8,
    pub general_level_idc: u8,
    pub chroma_format_idc: u8,
    pub bit_depth_luma_minus8: u8,
    pub bit_depth_chroma_minus8: u8,
    pub length_size_minus_one: u8,
    pub arrays: Vec<HevcNalArray>,
}

impl HevcDecoderConfig {
    pub fn nalus_of_type(&self, nal_unit_type: u8) -> impl Iterator<Item = &Vec<u8>> {
        self.arrays.iter().filter(move |a| a.nal_unit_type == nal_unit_type).flat_map(|a| a.nalus.iter())
    }
}

/// Parses an `hvcC` box body (ISO/IEC 14496-15 clause 8.3.3.1).
fn parse_hvcc(body: &[u8]) -> Result<HevcDecoderConfig> {
    let mut r = ByteReader::new(body);
    let _configuration_version = r.u8()?;
    let b1 = r.u8()?; // profile_space(2) tier_flag(1) profile_idc(5)
    let general_profile_idc = b1 & 0x1F;
    let _general_profile_compat = r.u32()?;
    let _general_constraint_indicator = {
        let hi = r.u32()?;
        let lo = r.u16()?;
        ((hi as u64) << 16) | lo as u64
    };
    let general_level_idc = r.u8()?;
    let min_spatial_seg = r.u16()?;
    let _min_spatial_segmentation_idc = min_spatial_seg & 0x0FFF;
    let parallelism = r.u8()?;
    let _parallelism_type = parallelism & 0x03;
    let chroma = r.u8()?;
    let chroma_format_idc = chroma & 0x03;
    let bdl = r.u8()?;
    let bit_depth_luma_minus8 = bdl & 0x07;
    let bdc = r.u8()?;
    let bit_depth_chroma_minus8 = bdc & 0x07;
    let _avg_frame_rate = r.u16()?;
    let last = r.u8()?; // constFrameRate(2) numTemporalLayers(3) temporalIdNested(1) lengthSizeMinusOne(2)
    let length_size_minus_one = last & 0x03;
    let num_arrays = r.u8()?;
    let mut arrays = Vec::new();
    for _ in 0..num_arrays {
        let hdr = r.u8()?;
        let nal_unit_type = hdr & 0x3F;
        let num_nalus = r.u16()?;
        let mut nalus = Vec::new();
        for _ in 0..num_nalus {
            let len = r.u16()? as usize;
            let data = r.bytes(len)?;
            nalus.push(data.to_vec());
        }
        arrays.push(HevcNalArray { nal_unit_type, nalus });
    }
    Ok(HevcDecoderConfig {
        general_profile_idc,
        general_level_idc,
        chroma_format_idc,
        bit_depth_luma_minus8,
        bit_depth_chroma_minus8,
        length_size_minus_one,
        arrays,
    })
}

#[derive(Debug, Clone, Copy)]
struct IlocExtent {
    offset: u64,
    length: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::enum_variant_names)] // each variant names what the iloc offset is relative to; the shared suffix is clarifying, not redundant.
enum ConstructionMethod {
    FileOffset,
    IdatOffset,
    ItemOffset,
}

struct IlocEntry {
    construction_method: ConstructionMethod,
    base_offset: u64,
    extents: Vec<IlocExtent>,
}

/// Parses an `iloc` box body (ISO/IEC 14496-12 clause 8.11.3).
fn parse_iloc(body: &[u8]) -> Result<BTreeMap<u32, IlocEntry>> {
    let mut r = ByteReader::new(body);
    let (version, _flags) = full_box_header(&mut r)?;
    let sizes = r.u16()?;
    let offset_size = (sizes >> 12) & 0xF;
    let length_size = (sizes >> 8) & 0xF;
    let base_offset_size = (sizes >> 4) & 0xF;
    let index_size = if version == 1 || version == 2 { sizes & 0xF } else { 0 };

    let read_sized = |r: &mut ByteReader<'_>, n: u16| -> Result<u64> {
        match n {
            0 => Ok(0),
            4 => Ok(r.u32()? as u64),
            8 => r.u64(),
            // 2-byte "size" fields also occur in practice; be permissive.
            2 => Ok(r.u16()? as u64),
            _ => Err(HeifError::Unsupported("iloc: unsupported field size")),
        }
    };

    let item_count = if version < 2 { r.u16()? as u32 } else { r.u32()? };
    let mut out = BTreeMap::new();
    for _ in 0..item_count {
        let item_id = if version < 2 { r.u16()? as u32 } else { r.u32()? };
        let construction_method = if version == 1 || version == 2 {
            let v = r.u16()? & 0xF;
            match v {
                0 => ConstructionMethod::FileOffset,
                1 => ConstructionMethod::IdatOffset,
                2 => ConstructionMethod::ItemOffset,
                _ => return Err(HeifError::Unsupported("iloc: unknown construction_method")),
            }
        } else {
            ConstructionMethod::FileOffset
        };
        let _data_reference_index = r.u16()?;
        let base_offset = read_sized(&mut r, base_offset_size)?;
        let extent_count = r.u16()?;
        let mut extents = Vec::new();
        for _ in 0..extent_count {
            if index_size > 0 {
                let _extent_index = read_sized(&mut r, index_size)?;
            }
            let offset = read_sized(&mut r, offset_size)?;
            let length = read_sized(&mut r, length_size)?;
            extents.push(IlocExtent { offset, length });
        }
        out.insert(item_id, IlocEntry { construction_method, base_offset, extents });
    }
    Ok(out)
}

/// A single `ipco` property: its box type and raw body.
struct Property<'a> {
    kind: [u8; 4],
    body: &'a [u8],
}

fn parse_ipco<'a>(body: &'a [u8]) -> Result<Vec<Property<'a>>> {
    let mut props = Vec::new();
    for_each_child(body, 0, body.len(), 0, |kind, data| {
        props.push(Property { kind: *kind, body: data });
        Ok(())
    })?;
    Ok(props)
}

/// `ipma` associations: item_id -> 1-based indices into `ipco` (clause 8.11.14).
fn parse_ipma(body: &[u8]) -> Result<BTreeMap<u32, Vec<u32>>> {
    let mut r = ByteReader::new(body);
    let (version, flags) = full_box_header(&mut r)?;
    let entry_count = r.u32()?;
    let large_index = flags & 1 == 1;
    let mut out = BTreeMap::new();
    for _ in 0..entry_count {
        let item_id = if version < 1 { r.u16()? as u32 } else { r.u32()? };
        let assoc_count = r.u8()?;
        let mut indices = Vec::new();
        for _ in 0..assoc_count {
            let idx = if large_index {
                let v = r.u16()?;
                (v & 0x7FFF) as u32
            } else {
                let v = r.u8()?;
                (v & 0x7F) as u32
            };
            indices.push(idx);
        }
        out.insert(item_id, indices);
    }
    Ok(out)
}

/// `iref` (item reference box, clause 8.11.12): a full-box header
/// followed by one child box per reference *type* (its box type is the
/// 4-byte reference type, e.g. `dimg`), each child being a
/// `SingleItemTypeReferenceBox`: `from_item_id` + `reference_count` +
/// that many `to_item_id`s. Returns, for the given `(from_item_id,
/// ref_type)`, the ordered list of referenced item IDs (order matters:
/// for `dimg` on a `grid` item, it's raster order, top-left first).
fn parse_iref(body: &[u8], want_from: u32, want_type: &[u8; 4]) -> Result<Vec<u32>> {
    let mut r = ByteReader::new(body);
    let (version, _flags) = full_box_header(&mut r)?;
    let children_start = r.pos();
    let mut result = Vec::new();
    for_each_child(body, children_start, body.len(), 0, |ref_type, data| {
        if ref_type != want_type {
            return Ok(());
        }
        let mut cr = ByteReader::new(data);
        while cr.remaining() > 0 {
            let from_id = if version == 0 { cr.u16()? as u32 } else { cr.u32()? };
            let ref_count = cr.u16()?;
            let mut to_ids = Vec::with_capacity(ref_count as usize);
            for _ in 0..ref_count {
                let to_id = if version == 0 { cr.u16()? as u32 } else { cr.u32()? };
                to_ids.push(to_id);
            }
            if from_id == want_from {
                result = to_ids;
            }
        }
        Ok(())
    })?;
    Ok(result)
}

#[derive(Debug, Clone, Copy)]
pub struct GridInfo {
    pub rows: u32,
    pub columns: u32,
    pub output_width: u32,
    pub output_height: u32,
}

/// `ImageGrid` (ISO/IEC 23008-12 clause 6.6.2.3.2): NOT a full box — just
/// `version`(8) + `flags`(8) + `rows_minus_one`(8) + `columns_minus_one`(8),
/// then `output_width`/`output_height` as 16-bit fields, or 32-bit if
/// `flags & 1`.
fn parse_grid_descriptor(body: &[u8]) -> Result<GridInfo> {
    let mut r = ByteReader::new(body);
    let _version = r.u8()?;
    let flags = r.u8()?;
    let rows = r.u8()? as u32 + 1;
    let columns = r.u8()? as u32 + 1;
    let (output_width, output_height) = if flags & 1 != 0 { (r.u32()?, r.u32()?) } else { (r.u16()? as u32, r.u16()? as u32) };
    if output_width == 0 || output_height == 0 {
        return Err(HeifError::MalformedBox("grid: zero output dimension"));
    }
    if u64::from(output_width) * u64::from(output_height) > crate::MAX_IMAGE_PIXELS {
        return Err(HeifError::LimitExceeded("grid: output larger than MAX_IMAGE_PIXELS"));
    }
    Ok(GridInfo { rows, columns, output_width, output_height })
}

#[derive(Debug, Clone)]
pub struct ResolvedImage {
    pub hevc_config: HevcDecoderConfig,
    /// Same ISOBMFF length-prefixed "sample" form as `ParsedHeif`
    /// previously documented for `primary_item_data`.
    pub data: Vec<u8>,
}

/// The single sync sample selected from a sequence, not animation playback.
#[derive(Debug, Clone, Copy)]
pub struct SequenceFrameInfo {
    pub track_id: u32,
    /// One-based sample number in decoding order.
    pub sample_number: u32,
    pub sample_count: u32,
}

/// Resolved auxiliary alpha image. Its own transforms are applied separately.
#[derive(Debug, Clone)]
pub struct AuxiliaryAlpha {
    pub image: Box<ParsedHeif>,
    /// `prem` links the colour image to this alpha image.
    pub premultiplied: bool,
}

#[derive(Debug, Clone)]
pub struct ParsedHeif {
    pub alpha: Option<AuxiliaryAlpha>,
    /// Single coded reference image for the restricted IDR-to-P path.
    pub reference_image: Option<ResolvedImage>,
    /// Present when returning a sequence preview rather than a still item.
    pub sequence: Option<SequenceFrameInfo>,
    /// Primary still-item ID, or the selected track ID for a sequence preview.
    pub primary_item_id: u32,
    /// `None` when the primary item is a plain `hvc1` image (`images`
    /// then holds exactly one entry); `Some` when it's a `grid` of
    /// `hvc1` tiles (`images` then holds `rows*columns` entries, in
    /// raster order: row-major, top-left first).
    pub grid: Option<GridInfo>,
    pub images: Vec<ResolvedImage>,
    /// Source and enclosing identity items' `clap`/`irot`/`imir` properties,
    /// applied source-first to the fully assembled picture.
    pub transforms: Vec<Transform>,
    pub metadata: Metadata,
}

/// Metadata attached to the primary image, as stored in the file. Parsing
/// EXIF or XMP is left to dedicated crates. Metadata is best-effort: a
/// malformed metadata item is reported as absent rather than failing the
/// decode.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Metadata {
    /// Coding-independent colour description from `colr/nclx`, if present.
    /// ICC profiles are retained separately; they do not replace the YCbCr matrix.
    pub color_info: Option<crate::color::ColorInfo>,
    /// ICC colour profile from the image's `colr` property (`prof` or `rICC`).
    pub icc_profile: Option<Vec<u8>>,
    /// EXIF data starting at its TIFF header (`II*\0` or `MM\0*`).
    pub exif: Option<Vec<u8>>,
    /// XMP packet (UTF-8 XML).
    pub xmp: Option<Vec<u8>>,
}

/// Splits ISOBMFF length-prefixed NAL sample data into individual NAL units.
pub fn split_length_prefixed_nalus(data: &[u8], length_size: usize) -> Result<Vec<Vec<u8>>> {
    if !(1..=4).contains(&length_size) {
        return Err(HeifError::Unsupported("split_length_prefixed_nalus: length_size out of range"));
    }
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < data.len() {
        let hdr = data.get(pos..pos + length_size).ok_or(HeifError::Truncated { at: pos, needed: length_size })?;
        let mut len: u64 = 0;
        for &b in hdr {
            len = (len << 8) | b as u64;
        }
        pos += length_size;
        let len = usize::try_from(len).map_err(|_| HeifError::MalformedHevc("NAL length overflow"))?;
        let nal = data.get(pos..pos + len).ok_or(HeifError::Truncated { at: pos, needed: len })?;
        out.push(nal.to_vec());
        pos += len;
    }
    Ok(out)
}

/// Parses a HEIF file, resolving the primary image item's HEVC
/// configuration and coded sample bytes.
pub fn parse(file: &[u8]) -> Result<ParsedHeif> {
    let _timer = crate::timing::start(crate::timing::Stage::Container);
    parse_selected(file, None, true)
}

fn parse_selected(file: &[u8], selected: Option<u32>, include_alpha: bool) -> Result<ParsedHeif> {
    let mut saw_ftyp = false;
    let mut meta_body_range: Option<(usize, usize)> = None;
    let mut moov_body_range = None;

    {
        let mut r = ByteReader::new(file);
        while r.remaining() >= 8 {
            let before = r.pos();
            let hdr = read_box_header(&mut r)?;
            let body_end = if hdr.body_end == usize::MAX { file.len() } else { hdr.body_end };
            if &hdr.kind == b"ftyp" {
                saw_ftyp = true;
            }
            if &hdr.kind == b"moov" {
                moov_body_range = Some((hdr.body_start, body_end));
            }
            if &hdr.kind == b"meta" {
                meta_body_range = Some((hdr.body_start, body_end));
            }
            if body_end <= before {
                return Err(HeifError::MalformedBox("non-advancing top-level box"));
            }
            r.seek(body_end.min(file.len()))?;
        }
    }

    if !saw_ftyp {
        return Err(HeifError::NotHeif("missing ftyp box"));
    }
    if meta_body_range.is_none()
        && let Some((start, end)) = moov_body_range
    {
        let moov = file.get(start..end).ok_or(HeifError::MalformedBox("moov outside file"))?;
        return sequence::parse(file, moov);
    }
    let (meta_start, meta_end) = meta_body_range.ok_or(HeifError::NotHeif("missing meta box"))?;
    let meta_body = file.get(meta_start..meta_end).ok_or(HeifError::MalformedBox("meta box out of bounds"))?;

    // `meta` is itself a full box (version/flags) followed by children.
    let mut mr = ByteReader::new(meta_body);
    full_box_header(&mut mr)?;
    let children_start = mr.pos();

    let mut primary_item_id: Option<u32> = None;
    let mut iloc: Option<BTreeMap<u32, IlocEntry>> = None;
    let mut item_types: BTreeMap<u32, [u8; 4]> = BTreeMap::new();
    let mut xmp_items: Vec<u32> = Vec::new();
    let mut ipco_bodies: Vec<Vec<u8>> = Vec::new();
    let mut ipco_kinds: Vec<[u8; 4]> = Vec::new();
    let mut ipma: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    let mut idat: Vec<u8> = Vec::new();
    let mut iref_body: Option<Vec<u8>> = None;

    for_each_child(meta_body, children_start, meta_body.len(), 0, |kind, body| {
        match kind {
            b"pitm" => {
                let mut r = ByteReader::new(body);
                let (version, _flags) = full_box_header(&mut r)?;
                primary_item_id = Some(if version == 0 { r.u16()? as u32 } else { r.u32()? });
            }
            b"iloc" => {
                iloc = Some(parse_iloc(body)?);
            }
            b"iinf" => {
                let mut r = ByteReader::new(body);
                let (version, _flags) = full_box_header(&mut r)?;
                let entry_count = if version == 0 { r.u16()? as u32 } else { r.u32()? };
                let rest_start = r.pos();
                let mut seen = 0u32;
                for_each_child(body, rest_start, body.len(), 0, |ckind, cbody| {
                    if ckind != b"infe" || seen >= entry_count {
                        return Ok(());
                    }
                    seen += 1;
                    let mut ir = ByteReader::new(cbody);
                    let (iversion, _iflags) = full_box_header(&mut ir)?;
                    if iversion < 2 {
                        return Ok(()); // versions 0/1 predate item_type; not used for images here.
                    }
                    let item_id = if iversion == 2 { ir.u16()? as u32 } else { ir.u32()? };
                    let _protection_index = ir.u16()?;
                    let item_type = ir.fourcc()?;
                    item_types.insert(item_id, item_type);
                    // `mime` items name their content type after `item_name`.
                    if &item_type == b"mime" && ir.c_string().and_then(|_| ir.c_string()).is_ok_and(|t| t == b"application/rdf+xml") {
                        xmp_items.push(item_id);
                    }
                    Ok(())
                })?;
            }
            b"iprp" => {
                for_each_child(body, 0, body.len(), 0, |pkind, pbody| {
                    if pkind == b"ipco" {
                        for prop in parse_ipco(pbody)? {
                            ipco_kinds.push(prop.kind);
                            ipco_bodies.push(prop.body.to_vec());
                        }
                    } else if pkind == b"ipma" {
                        for (item_id, indices) in parse_ipma(pbody)? {
                            ipma.insert(item_id, indices);
                        }
                    }
                    Ok(())
                })?;
            }
            b"idat" => {
                idat = body.to_vec();
            }
            b"iref" => {
                iref_body = Some(body.to_vec());
            }
            _ => {}
        }
        Ok(())
    })?;

    let primary_item_id = selected.or(primary_item_id).ok_or(HeifError::NotHeif("missing pitm box"))?;
    let iloc = iloc.ok_or(HeifError::NotHeif("missing iloc box"))?;

    let chain = identity_chain(&item_types, iref_body.as_deref(), primary_item_id)?;
    let alpha = if include_alpha { resolve_alpha(file, &item_types, &ipma, &ipco_kinds, &ipco_bodies, iref_body.as_deref(), &chain)? } else { None };
    let source_id = *chain.last().ok_or(HeifError::MalformedBox("empty image reference chain"))?;
    // Source transforms apply first, then each enclosing identity item.
    let mut transforms = Vec::new();
    for &item_id in chain.iter().rev() {
        transforms.extend(resolve_transforms(&ipma, &ipco_kinds, &ipco_bodies, item_id)?);
    }
    let reference_ids = match iref_body.as_deref() {
        Some(body) => parse_iref(body, source_id, b"pred")?,
        None => Vec::new(),
    };
    let reference_image = if reference_ids.is_empty() {
        None
    } else {
        let [reference_id] = reference_ids.as_slice() else {
            return Err(HeifError::Unsupported("prediction: multiple dependency images"));
        };
        if *reference_id == source_id {
            return Err(HeifError::MalformedBox("prediction: self reference"));
        }
        if item_types.get(&source_id) != Some(b"hvc1") || item_types.get(reference_id) != Some(b"hvc1") {
            return Err(HeifError::Unsupported("prediction: dependencies must be hvc1 images"));
        }
        if let Some(body) = iref_body.as_deref()
            && !parse_iref(body, *reference_id, b"pred")?.is_empty()
        {
            return Err(HeifError::Unsupported("prediction: chained reference images"));
        }
        Some(ResolvedImage {
            hevc_config: resolve_hvcc(&ipma, &ipco_kinds, &ipco_bodies, *reference_id)?,
            data: resolve_item_bytes(file, &idat, &iloc, *reference_id)?,
        })
    };
    let exif_items: Vec<u32> = item_types.iter().filter(|(_, t)| *t == b"Exif").map(|(id, _)| *id).collect();
    let metadata_item =
        |candidates: &[u32]| describing_item(candidates, iref_body.as_deref(), &chain).and_then(|id| resolve_item_bytes(file, &idat, &iloc, id).ok());
    let mut metadata = Metadata {
        color_info: resolve_color_info(&ipma, &ipco_kinds, &ipco_bodies, &chain)?,
        icc_profile: resolve_icc(&ipma, &ipco_kinds, &ipco_bodies, &chain),
        exif: metadata_item(&exif_items).and_then(|payload| exif_tiff(&payload)),
        xmp: metadata_item(&xmp_items),
    };
    match item_types.get(&source_id) {
        Some(b"hvc1") => {
            let hevc_config = resolve_hvcc(&ipma, &ipco_kinds, &ipco_bodies, source_id)?;
            let data = resolve_item_bytes(file, &idat, &iloc, source_id)?;
            Ok(ParsedHeif {
                alpha,
                reference_image,
                sequence: None,
                primary_item_id,
                grid: None,
                images: vec![ResolvedImage { hevc_config, data }],
                transforms,
                metadata,
            })
        }
        Some(b"grid") => {
            let grid_bytes = resolve_item_bytes(file, &idat, &iloc, source_id)?;
            let grid_info = parse_grid_descriptor(&grid_bytes)?;
            let iref_body = iref_body.ok_or(HeifError::MalformedBox("grid item but no iref box"))?;
            let tile_ids = parse_iref(&iref_body, source_id, b"dimg")?;
            let expected_tiles =
                (grid_info.rows as usize).checked_mul(grid_info.columns as usize).ok_or(HeifError::MalformedBox("grid dimensions overflow"))?;
            if tile_ids.is_empty() || tile_ids.len() != expected_tiles {
                return Err(HeifError::MalformedBox("grid tile count does not match rows*columns"));
            }
            let mut images = Vec::with_capacity(tile_ids.len());
            for &tile_id in &tile_ids {
                match item_types.get(&tile_id) {
                    Some(b"hvc1") => {}
                    Some(_) => return Err(HeifError::Unsupported("grid tile is not type 'hvc1' (mixed-codec grids not supported)")),
                    None => return Err(HeifError::MalformedBox("grid tile has no iinf entry")),
                }
                let hevc_config = resolve_hvcc(&ipma, &ipco_kinds, &ipco_bodies, tile_id)?;
                let data = resolve_item_bytes(file, &idat, &iloc, tile_id)?;
                images.push(ResolvedImage { hevc_config, data });
            }
            // Some writers attach the profile to the tiles rather than the grid.
            if metadata.icc_profile.is_none() {
                metadata.icc_profile = resolve_icc(&ipma, &ipco_kinds, &ipco_bodies, &tile_ids);
            }
            if metadata.color_info.is_none() {
                metadata.color_info = resolve_color_info(&ipma, &ipco_kinds, &ipco_bodies, &tile_ids)?;
            }
            Ok(ParsedHeif { alpha, reference_image, sequence: None, primary_item_id, grid: Some(grid_info), images, transforms, metadata })
        }
        Some(_) => Err(HeifError::Unsupported("resolved image item is not type 'hvc1' or 'grid'")),
        None => Err(HeifError::MalformedBox("primary item has no iinf entry")),
    }
}

fn auxiliary_links(body: &[u8], wanted: &[u8; 4]) -> Result<BTreeMap<u32, Vec<u32>>> {
    let mut r = ByteReader::new(body);
    let (version, _) = full_box_header(&mut r)?;
    if version > 1 {
        return Err(HeifError::Unsupported("iref: version"));
    }
    let mut links = BTreeMap::new();
    for_each_child(body, r.pos(), body.len(), 0, |kind, data| {
        if kind != wanted {
            return Ok(());
        }
        let mut entry = ByteReader::new(data);
        while entry.remaining() > 0 {
            let from = if version == 0 { u32::from(entry.u16()?) } else { entry.u32()? };
            let count = entry.u16()?;
            let mut targets = Vec::with_capacity(count as usize);
            for _ in 0..count {
                targets.push(if version == 0 { u32::from(entry.u16()?) } else { entry.u32()? });
            }
            if links.insert(from, targets).is_some() {
                return Err(HeifError::MalformedBox("iref: duplicate auxiliary reference"));
            }
        }
        Ok(())
    })?;
    Ok(links)
}

// Alpha is identified by its auxiliary type AND an auxl link to the selected
// image/identity chain. Depth maps and unrelated auxiliary images are ignored.
#[allow(clippy::too_many_arguments)]
fn resolve_alpha(
    file: &[u8],
    types: &BTreeMap<u32, [u8; 4]>,
    ipma: &BTreeMap<u32, Vec<u32>>,
    kinds: &[[u8; 4]],
    bodies: &[Vec<u8>],
    references: Option<&[u8]>,
    chain: &[u32],
) -> Result<Option<AuxiliaryAlpha>> {
    let Some(references) = references else {
        return Ok(None);
    };
    let links = auxiliary_links(references, b"auxl")?;
    let mut selected = None;
    for (&id, targets) in &links {
        if !targets.iter().any(|target| chain.contains(target)) {
            continue;
        }
        let mut is_alpha = false;
        for &property in ipma.get(&id).into_iter().flatten() {
            let Some(index) = (property as usize).checked_sub(1) else {
                continue;
            };
            if kinds.get(index) != Some(b"auxC") {
                continue;
            }
            let body = bodies.get(index).ok_or(HeifError::MalformedBox("auxC: property out of bounds"))?;
            let mut r = ByteReader::new(body);
            let (version, _) = full_box_header(&mut r)?;
            if version != 0 {
                return Err(HeifError::Unsupported("auxC: version"));
            }
            let urn = r.c_string()?;
            is_alpha |= matches!(urn, b"urn:mpeg:hevc:2015:auxid:1" | b"urn:mpeg:avc:2015:auxid:1" | b"urn:mpeg:mpegB:cicp:systems:auxiliary:alpha");
        }
        if !is_alpha {
            continue;
        }
        if chain.contains(&id) {
            return Err(HeifError::MalformedBox("alpha: self reference"));
        }
        if selected.replace(id).is_some() {
            return Err(HeifError::Unsupported("alpha: multiple auxiliary alpha images"));
        }
    }
    let Some(id) = selected else {
        return Ok(None);
    };
    if identity_chain(types, Some(references), id)?.iter().any(|source| chain.contains(source)) {
        return Err(HeifError::MalformedBox("alpha: cyclic identity reference"));
    }
    let prem_links = auxiliary_links(references, b"prem")?;
    let premultiplied = chain.iter().any(|master| prem_links.get(master).is_some_and(|targets| targets.contains(&id)));
    // One bounded reparse reuses the same grid/identity validation. Nested
    // auxiliaries are deliberately not traversed, avoiding recursive cycles.
    let image = parse_selected(file, Some(id), false)?;
    Ok(Some(AuxiliaryAlpha { image: Box::new(image), premultiplied }))
}

pub(super) fn parse_nclx(body: &[u8]) -> Result<Option<crate::color::ColorInfo>> {
    if !body.starts_with(b"nclx") {
        return Ok(None);
    }
    let mut r = ByteReader::new(body);
    r.skip(4)?;
    Ok(Some(crate::color::ColorInfo {
        color_primaries: r.u16()?,
        transfer_characteristics: r.u16()?,
        matrix_coefficients: r.u16()?,
        full_range: r.u8()? & 0x80 != 0,
    }))
}

fn resolve_color_info(ipma: &BTreeMap<u32, Vec<u32>>, kinds: &[[u8; 4]], bodies: &[Vec<u8>], items: &[u32]) -> Result<Option<crate::color::ColorInfo>> {
    for &index in items.iter().filter_map(|item| ipma.get(item)).flatten() {
        let Some(i) = (index as usize).checked_sub(1) else {
            continue;
        };
        if kinds.get(i) == Some(b"colr") {
            let body = bodies.get(i).ok_or(HeifError::MalformedBox("colr: property out of bounds"))?;
            if let Some(color) = parse_nclx(body)? {
                return Ok(Some(color));
            }
        }
    }
    Ok(None)
}

/// The first `colr` ICC profile (`prof` or `rICC`) associated with `items`.
fn resolve_icc(ipma: &BTreeMap<u32, Vec<u32>>, ipco_kinds: &[[u8; 4]], ipco_bodies: &[Vec<u8>], items: &[u32]) -> Option<Vec<u8>> {
    items.iter().filter_map(|item| ipma.get(item)).flatten().find_map(|&idx| {
        let i = (idx as usize).checked_sub(1)?;
        if ipco_kinds.get(i) != Some(b"colr") {
            return None;
        }
        let (colour_type, profile) = ipco_bodies.get(i)?.split_first_chunk::<4>()?;
        (colour_type == b"prof" || colour_type == b"rICC").then(|| profile.to_vec())
    })
}

/// Picks the metadata item that describes the image (`cdsc` reference,
/// ISO/IEC 23008-12 clause 6.4.2). Falls back to an item with no `cdsc`
/// references at all, since some writers omit them.
fn describing_item(candidates: &[u32], references: Option<&[u8]>, image_items: &[u32]) -> Option<u32> {
    let targets = |id| references.map_or(Ok(Vec::new()), |body| parse_iref(body, id, b"cdsc"));
    candidates
        .iter()
        .copied()
        .find(|&id| targets(id).is_ok_and(|t| t.iter().any(|t| image_items.contains(t))))
        .or_else(|| candidates.iter().copied().find(|&id| targets(id).is_ok_and(|t| t.is_empty())))
}

/// An `Exif` item starts with a 32-bit offset to the TIFF header
/// (ISO/IEC 23008-12 Annex A.2.1); return the bytes from that header on.
/// Older files (e.g. conformance file C034) store the TIFF data directly.
fn exif_tiff(payload: &[u8]) -> Option<Vec<u8>> {
    let is_tiff = |b: &[u8]| b.starts_with(b"II*\0") || b.starts_with(b"MM\0*");
    if is_tiff(payload) {
        return Some(payload.to_vec());
    }
    let (offset, rest) = payload.split_first_chunk::<4>()?;
    let tiff = rest.get(usize::try_from(u32::from_be_bytes(*offset)).ok()?..)?;
    is_tiff(tiff).then(|| tiff.to_vec())
}

/// Follow identity derivations without recursion. Keep every identity so
/// its transforms can be applied after those of its source.
fn identity_chain(types: &BTreeMap<u32, [u8; 4]>, references: Option<&[u8]>, primary: u32) -> Result<Vec<u32>> {
    let mut chain = Vec::new();
    let mut current = primary;
    loop {
        if chain.contains(&current) {
            return Err(HeifError::MalformedBox("iden: cyclic image reference"));
        }
        if chain.len() >= 32 {
            return Err(HeifError::Unsupported("iden: image reference chain too deep"));
        }
        chain.push(current);
        match types.get(&current) {
            Some(b"iden") => {
                let body = references.ok_or(HeifError::MalformedBox("iden: missing iref box"))?;
                let sources = parse_iref(body, current, b"dimg")?;
                let [source] = sources.as_slice() else {
                    return Err(HeifError::MalformedBox("iden: expected exactly one dimg source"));
                };
                current = *source;
            }
            Some(_) => return Ok(chain),
            None => return Err(HeifError::MalformedBox("image reference has no iinf entry")),
        }
    }
}

/// Resolves one item's coded bytes via its `iloc` entry (concatenating
/// all of its extents), following `construction_method 0`/`1` (file- or
/// `idat`-relative offsets).
fn resolve_item_bytes(file: &[u8], idat: &[u8], iloc: &BTreeMap<u32, IlocEntry>, item_id: u32) -> Result<Vec<u8>> {
    let entry = iloc.get(&item_id).ok_or(HeifError::MalformedBox("item has no iloc entry"))?;
    let mut data = Vec::new();
    for extent in &entry.extents {
        let start = entry.base_offset.checked_add(extent.offset).ok_or(HeifError::MalformedBox("iloc offset overflow"))?;
        let start = usize::try_from(start).map_err(|_| HeifError::MalformedBox("iloc offset overflow"))?;
        let len = usize::try_from(extent.length).map_err(|_| HeifError::MalformedBox("iloc length overflow"))?;
        let bytes = match entry.construction_method {
            ConstructionMethod::FileOffset => file
                .get(start..start.checked_add(len).ok_or(HeifError::MalformedBox("iloc end overflow"))?)
                .ok_or(HeifError::Truncated { at: start, needed: len })?,
            ConstructionMethod::IdatOffset => idat
                .get(start..start.checked_add(len).ok_or(HeifError::MalformedBox("iloc end overflow"))?)
                .ok_or(HeifError::Truncated { at: start, needed: len })?,
            ConstructionMethod::ItemOffset => return Err(HeifError::Unsupported("iloc construction_method 2 (item offset) not supported")),
        };
        data.extend_from_slice(bytes);
    }
    Ok(data)
}

/// An orientation-changing item property (ISO/IEC 23008-12 clause
/// 6.5.10 `irot` / 6.5.12 `imir`) associated with the primary item,
/// applied, in `ipma` listing order, to the fully assembled picture
/// (after grid stitching, if any) to get the pixels as they're meant
/// to be displayed.
/// Rational dimensions and signed center offsets from a `clap` property.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CleanAperture {
    pub width: (u32, u32),
    pub height: (u32, u32),
    pub horizontal_offset: (i32, u32),
    pub vertical_offset: (i32, u32),
}

fn parse_clean_aperture(body: &[u8]) -> Result<CleanAperture> {
    let mut r = ByteReader::new(body);
    let aperture = CleanAperture {
        width: (r.u32()?, r.u32()?),
        height: (r.u32()?, r.u32()?),
        horizontal_offset: (r.u32()? as i32, r.u32()?),
        vertical_offset: (r.u32()? as i32, r.u32()?),
    };
    if aperture.width.1 == 0 || aperture.height.1 == 0 || aperture.horizontal_offset.1 == 0 || aperture.vertical_offset.1 == 0 {
        return Err(HeifError::MalformedBox("clap: zero denominator"));
    }
    Ok(aperture)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transform {
    CleanAperture(CleanAperture),
    /// `irot`'s `angle` field is a count of 90°-*anticlockwise* turns
    /// (0..=3); `0` is a no-op and is never pushed to the list.
    Rotate90Ccw(u8),
    /// `imir` with `axis == 0`: flip top↔bottom.
    MirrorVertical,
    /// `imir` with `axis == 1`: flip left↔right.
    MirrorHorizontal,
}

/// Resolves one item's `clap`/`irot`/`imir` properties via its `ipma`
/// association, in listing order (clause 8.11.14: when an item has
/// more than one transformative property, they're applied in the
/// order they're associated).
fn resolve_transforms(ipma: &BTreeMap<u32, Vec<u32>>, ipco_kinds: &[[u8; 4]], ipco_bodies: &[Vec<u8>], item_id: u32) -> Result<Vec<Transform>> {
    let Some(prop_indices) = ipma.get(&item_id) else {
        return Ok(Vec::new());
    };
    let mut transforms = Vec::new();
    for &idx in prop_indices {
        let zero_based = idx.checked_sub(1).ok_or(HeifError::MalformedBox("ipma: zero property index"))? as usize;
        let kind = ipco_kinds.get(zero_based);
        let body = ipco_bodies.get(zero_based).ok_or(HeifError::MalformedBox("ipma: index out of range"))?;
        if kind == Some(b"clap") {
            transforms.push(Transform::CleanAperture(parse_clean_aperture(body)?));
        } else if kind == Some(b"irot") {
            // Clause 6.5.10.2: not a full box — one byte, `angle` in
            // the low 2 bits, anticlockwise 90°-turn count.
            let &[byte, ..] = body.as_slice() else {
                return Err(HeifError::MalformedBox("irot: empty property body"));
            };
            let count = byte & 0x03;
            if count != 0 {
                transforms.push(Transform::Rotate90Ccw(count));
            }
        } else if kind == Some(b"imir") {
            // Clause 6.5.12.2: likewise not a full box — one byte,
            // `axis` in the low bit.
            let &[byte, ..] = body.as_slice() else {
                return Err(HeifError::MalformedBox("imir: empty property body"));
            };
            transforms.push(if byte & 0x01 == 0 { Transform::MirrorVertical } else { Transform::MirrorHorizontal });
        }
    }
    Ok(transforms)
}

/// Resolves one item's `hvcC` property via its `ipma` association.
fn resolve_hvcc(ipma: &BTreeMap<u32, Vec<u32>>, ipco_kinds: &[[u8; 4]], ipco_bodies: &[Vec<u8>], item_id: u32) -> Result<HevcDecoderConfig> {
    let prop_indices = ipma.get(&item_id).ok_or(HeifError::MalformedBox("item has no ipma association"))?;
    let mut hevc_config: Option<HevcDecoderConfig> = None;
    for &idx in prop_indices {
        // ipma indices are 1-based.
        let zero_based = idx.checked_sub(1).ok_or(HeifError::MalformedBox("ipma: zero property index"))? as usize;
        if ipco_kinds.get(zero_based) == Some(b"hvcC") {
            let body = ipco_bodies.get(zero_based).ok_or(HeifError::MalformedBox("ipma: index out of range"))?;
            hevc_config = Some(parse_hvcc(body)?);
        }
    }
    hevc_config.ok_or(HeifError::MalformedBox("item has no hvcC property"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn box_(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let size = (8 + body.len()) as u32;
        out.extend_from_slice(&size.to_be_bytes());
        out.extend_from_slice(kind);
        out.extend_from_slice(body);
        out
    }

    fn full_box_body(version: u8, flags: u32, rest: &[u8]) -> Vec<u8> {
        let mut out = vec![version];
        out.extend_from_slice(&flags.to_be_bytes()[1..4]);
        out.extend_from_slice(rest);
        out
    }

    #[test]
    fn auxiliary_alpha_rejects_self_cycles_duplicates_and_multiple_planes() {
        let reference = |kind: &[u8; 4], from: u16, to: u16| box_(kind, &[from.to_be_bytes(), 1u16.to_be_bytes(), to.to_be_bytes()].concat());
        let types = BTreeMap::from([(1, *b"hvc1"), (2, *b"iden"), (3, *b"hvc1")]);
        let ipma = BTreeMap::from([(1, vec![1]), (2, vec![1]), (3, vec![1])]);
        let kinds = [*b"auxC"];
        let bodies = [full_box_body(0, 0, b"urn:mpeg:hevc:2015:auxid:1\0")];
        for entries in [
            reference(b"auxl", 1, 1),
            [reference(b"auxl", 2, 1), reference(b"dimg", 2, 1)].concat(),
            [reference(b"auxl", 2, 1), reference(b"auxl", 2, 1)].concat(),
            [reference(b"auxl", 2, 1), reference(b"auxl", 3, 1)].concat(),
        ] {
            let body = full_box_body(0, 0, &entries);
            assert!(resolve_alpha(&[], &types, &ipma, &kinds, &bodies, Some(&body), &[1]).is_err());
        }
        // Version 1 references use 32-bit IDs rather than silently truncating.
        let body =
            full_box_body(1, 0, &box_(b"auxl", &[65537u32.to_be_bytes().as_slice(), 1u16.to_be_bytes().as_slice(), 1u32.to_be_bytes().as_slice()].concat()));
        assert_eq!(auxiliary_links(&body, b"auxl").unwrap().get(&65537), Some(&vec![1]));
    }

    /// Builds a minimal, synthetic (not real-world) HEIC-shaped file with one
    /// `hvc1` primary item whose coded data is a single fake NAL, purely to
    /// exercise the container parser end to end.
    fn build_minimal_heic(coded: &[u8]) -> Vec<u8> {
        let ftyp = box_(b"ftyp", b"heicheic\0\0\0\0mif1heic");

        let hdlr = box_(
            b"hdlr",
            &full_box_body(0, 0, &{
                let mut b = vec![0, 0, 0, 0]; // pre_defined
                b.extend_from_slice(b"pict");
                b.extend_from_slice(&[0u8; 12]); // reserved
                b.push(0); // name (empty c-string)
                b
            }),
        );

        let pitm = box_(b"pitm", &full_box_body(0, 0, &1u16.to_be_bytes()));

        let infe = box_(
            b"infe",
            &full_box_body(2, 0, &{
                let mut b = 1u16.to_be_bytes().to_vec(); // item_id
                b.extend_from_slice(&0u16.to_be_bytes()); // protection_index
                b.extend_from_slice(b"hvc1"); // item_type
                b.push(0); // item_name (empty c-string)
                b
            }),
        );
        let iinf_body = full_box_body(0, 0, &{
            let mut b = 1u16.to_be_bytes().to_vec(); // entry_count
            b.extend_from_slice(&infe);
            b
        });
        let iinf = box_(b"iinf", &iinf_body);

        // iloc v0: 4-bit nibble field sizes: offset=4,length=4,base_offset=4,index=0
        let iloc_body = full_box_body(0, 0, &{
            let mut b = Vec::new();
            b.extend_from_slice(&0x4440u16.to_be_bytes());
            b.extend_from_slice(&1u16.to_be_bytes()); // item_count
            b.extend_from_slice(&1u16.to_be_bytes()); // item_id
            b.extend_from_slice(&0u16.to_be_bytes()); // data_reference_index
            b.extend_from_slice(&0u32.to_be_bytes()); // base_offset (set below, file-relative)
            b.extend_from_slice(&1u16.to_be_bytes()); // extent_count
            b.extend_from_slice(&0u32.to_be_bytes()); // extent_offset (patched below)
            b.extend_from_slice(&(coded.len() as u32).to_be_bytes()); // extent_length
            b
        });
        let iloc = box_(b"iloc", &iloc_body);

        // hvcC: minimal fixed header + one VPS-type array with zero NALUs,
        // enough to exercise parse_hvcc's field layout.
        let hvcc_body = {
            let mut b = vec![1u8]; // configurationVersion
            b.push(0b0000_0001); // profile_space(2)=0 tier(1)=0 profile_idc(5)=1
            b.extend_from_slice(&0u32.to_be_bytes()); // profile_compat
            b.extend_from_slice(&[0u8; 6]); // constraint flags (48 bits)
            b.push(93); // level_idc
            b.extend_from_slice(&0xF000u16.to_be_bytes()); // reserved+min_spatial_seg
            b.push(0xFC); // reserved+parallelism
            b.push(0xFC | 1); // reserved+chroma_format_idc=1 (4:2:0)
            b.push(0xF8); // reserved+bit_depth_luma_minus8=0
            b.push(0xF8); // reserved+bit_depth_chroma_minus8=0
            b.extend_from_slice(&0u16.to_be_bytes()); // avgFrameRate
            b.push(0x03); // constFrameRate(2)=0 numTemporalLayers(3)=0 nested(1)=0 lengthSizeMinusOne(2)=3
            b.push(0); // numOfArrays = 0
            b
        };
        let hvcc = box_(b"hvcC", &hvcc_body);
        let ipco = box_(b"ipco", &hvcc);
        let ipma_body = full_box_body(0, 0, &{
            let mut b = 1u32.to_be_bytes().to_vec(); // entry_count
            b.extend_from_slice(&1u16.to_be_bytes()); // item_id
            b.push(1); // association_count
            b.push(0x81); // essential=1, property_index=1
            b
        });
        let ipma = box_(b"ipma", &ipma_body);
        let mut iprp_body = Vec::new();
        iprp_body.extend_from_slice(&ipco);
        iprp_body.extend_from_slice(&ipma);
        let iprp = box_(b"iprp", &iprp_body);

        let mut meta_children = Vec::new();
        meta_children.extend_from_slice(&hdlr);
        meta_children.extend_from_slice(&pitm);
        meta_children.extend_from_slice(&iinf);
        meta_children.extend_from_slice(&iloc);
        meta_children.extend_from_slice(&iprp);
        let meta = box_(b"meta", &full_box_body(0, 0, &meta_children));

        let mdat_offset_in_file = (ftyp.len() + meta.len() + 8) as u32; // after mdat's own header

        // `extent_offset` sits at a fixed, hand-tracked position inside
        // `iloc_body`: sizes(2) + item_count(2) + item_id(2) +
        // data_reference_index(2) + base_offset(4) + extent_count(2) = 14
        // bytes in, then the 4-byte extent_offset field itself. iloc_body
        // is wrapped by a 4-byte full-box header, then an 8-byte box
        // header, then meta_children is prefixed by hdlr+pitm+iinf, then
        // meta_body by a 4-byte full-box header, then meta by an 8-byte
        // box header, then ftyp precedes all of it.
        let extent_offset_pos = ftyp.len() + 8 + 4 + hdlr.len() + pitm.len() + iinf.len() + 8 + 4 + 14;
        let mut file = Vec::new();
        file.extend_from_slice(&ftyp);
        file.extend_from_slice(&meta);
        file[extent_offset_pos..extent_offset_pos + 4].copy_from_slice(&mdat_offset_in_file.to_be_bytes());
        let mdat = box_(b"mdat", coded);
        file.extend_from_slice(&mdat);
        file
    }

    fn identity_references(edges: &[(u16, &[u16])]) -> Vec<u8> {
        let mut children = Vec::new();
        for &(from, to) in edges {
            let mut body = from.to_be_bytes().to_vec();
            body.extend_from_slice(&(to.len() as u16).to_be_bytes());
            for id in to {
                body.extend_from_slice(&id.to_be_bytes());
            }
            children.extend_from_slice(&box_(b"dimg", &body));
        }
        full_box_body(0, 0, &children)
    }

    #[test]
    fn nested_identity_chain_preserves_source_first_transform_order() {
        let types = BTreeMap::from([(1, *b"hvc1"), (2, *b"iden"), (3, *b"iden")]);
        let refs = identity_references(&[(3, &[2]), (2, &[1])]);
        let chain = identity_chain(&types, Some(&refs), 3).unwrap();
        assert_eq!(chain, vec![3, 2, 1]);
        let ipma = BTreeMap::from([(1, vec![1]), (2, vec![2]), (3, vec![3])]);
        let kinds = vec![*b"irot", *b"imir", *b"irot"];
        let bodies = vec![vec![1], vec![1], vec![2]];
        let mut transforms = Vec::new();
        for id in chain.into_iter().rev() {
            transforms.extend(resolve_transforms(&ipma, &kinds, &bodies, id).unwrap());
        }
        assert_eq!(transforms, vec![Transform::Rotate90Ccw(1), Transform::MirrorHorizontal, Transform::Rotate90Ccw(2)]);
    }

    #[test]
    fn identity_chain_rejects_invalid_graphs() {
        let types = BTreeMap::from([(1, *b"hvc1"), (2, *b"iden"), (3, *b"iden")]);
        assert!(identity_chain(&types, None, 2).is_err());
        for edges in [vec![(2, vec![])], vec![(2, vec![1, 3])], vec![(2, vec![9])], vec![(2, vec![2])], vec![(2, vec![3]), (3, vec![2])]] {
            let edges: Vec<_> = edges.iter().map(|(from, to)| (*from, to.as_slice())).collect();
            let refs = identity_references(&edges);
            assert!(identity_chain(&types, Some(&refs), 2).is_err());
        }
        let types: BTreeMap<_, _> = (1..=33).map(|id| (id, if id == 33 { *b"hvc1" } else { *b"iden" })).collect();
        let targets: Vec<_> = (2u16..=33).map(|id| vec![id]).collect();
        let edges: Vec<_> = targets.iter().enumerate().map(|(i, to)| (i as u16 + 1, to.as_slice())).collect();
        let refs = identity_references(&edges);
        assert!(identity_chain(&types, Some(&refs), 1).is_err());
    }

    #[test]
    fn clean_aperture_is_parsed_in_association_order() {
        let words = [300u32, 1, 300, 1, (-20i32) as u32, 1, 0, 1];
        let body: Vec<u8> = words.into_iter().flat_map(u32::to_be_bytes).collect();
        let properties = vec![body.clone(), vec![1]];
        let kinds = vec![*b"clap", *b"irot"];
        let associations = BTreeMap::from([(1, vec![2, 1])]);
        let transforms = resolve_transforms(&associations, &kinds, &properties, 1).unwrap();
        assert_eq!(transforms[0], Transform::Rotate90Ccw(1));
        let Transform::CleanAperture(aperture) = transforms[1] else { panic!("missing crop") };
        assert_eq!(aperture.horizontal_offset, (-20, 1));
        assert_eq!(aperture.width, (300, 1));
        assert!(parse_clean_aperture(&body[..31]).is_err());
        let mut invalid = body;
        invalid[4..8].fill(0);
        assert!(parse_clean_aperture(&invalid).is_err());
    }

    #[test]
    fn exif_payload_with_and_without_offset_field() {
        let tiff = b"MM\0*\0\0\0\x08".to_vec();
        let with_offset = [&6u32.to_be_bytes()[..], b"Exif\0\0", &tiff].concat();
        assert_eq!(exif_tiff(&with_offset), Some(tiff.clone()));
        assert_eq!(exif_tiff(&tiff), Some(tiff));
        assert_eq!(exif_tiff(&[0, 0, 0, 0, b'n', b'o']), None);
        assert_eq!(exif_tiff(&[0xFF, 0xFF, 0xFF, 0xFF]), None);
    }

    #[test]
    fn icc_profile_comes_from_colr_prof_only() {
        let kinds = vec![*b"colr", *b"colr", *b"irot"];
        let bodies = vec![b"nclx\0\x01\0\x0d\0\x01\x80".to_vec(), b"profICCDATA".to_vec(), vec![1]];
        let ipma = BTreeMap::from([(1, vec![1, 3]), (2, vec![1, 2])]);
        assert_eq!(resolve_icc(&ipma, &kinds, &bodies, &[1]), None);
        assert_eq!(resolve_icc(&ipma, &kinds, &bodies, &[1, 2]), Some(b"ICCDATA".to_vec()));
    }

    #[test]
    fn nclx_colour_uses_item_precedence_and_rejects_truncated_data() {
        let kinds = vec![*b"colr", *b"colr", *b"colr"];
        let bodies = vec![b"profICC".to_vec(), b"nclx\0\x09\0\x10\0\x09\x80".to_vec(), b"nclx\0\x01\0\x0d\0\x06\0".to_vec()];
        let associations = BTreeMap::from([(1, vec![1, 2]), (2, vec![3])]);
        let colour = resolve_color_info(&associations, &kinds, &bodies, &[2, 1]).unwrap().unwrap();
        assert_eq!((colour.color_primaries, colour.transfer_characteristics, colour.matrix_coefficients, colour.full_range), (1, 13, 6, false));
        assert_eq!(resolve_icc(&associations, &kinds, &bodies, &[2, 1]), Some(b"ICC".to_vec()));
        for length in 4..11 {
            assert!(parse_nclx(&bodies[1][..length]).is_err());
        }
        assert_eq!(parse_nclx(b"profICC").unwrap(), None);
        // Keep unsupported code points visible to the optional converter;
        // the planar decoder does not need a colour matrix.
        assert_eq!(parse_nclx(b"nclx\0\x02\0\x02\0\x0e\x80").unwrap().unwrap().matrix_coefficients, 14);
    }

    #[test]
    fn metadata_item_must_describe_the_primary_image() {
        // As in iPhone photos: Exif (65) describes the primary (49), XMP (64)
        // describes the HDR gain map (63) and must not be reported.
        let mut children = box_(b"cdsc", &[&64u16.to_be_bytes()[..], &1u16.to_be_bytes(), &63u16.to_be_bytes()].concat());
        children.extend(box_(b"cdsc", &[&65u16.to_be_bytes()[..], &1u16.to_be_bytes(), &49u16.to_be_bytes()].concat()));
        let refs = full_box_body(0, 0, &children);
        assert_eq!(describing_item(&[65], Some(&refs), &[49]), Some(65));
        assert_eq!(describing_item(&[64], Some(&refs), &[49]), None);
        // An item without any cdsc reference is accepted as a fallback.
        assert_eq!(describing_item(&[70], Some(&refs), &[49]), Some(70));
        assert_eq!(describing_item(&[70], None, &[49]), Some(70));
    }

    #[test]
    fn grid_output_size_is_bounded() {
        // version, flags (bit 0: 32-bit dimensions), rows-1, columns-1, width, height
        let grid = |w: u32, h: u32| [&[0u8, 1, 0, 0][..], &w.to_be_bytes(), &h.to_be_bytes()].concat();
        assert!(parse_grid_descriptor(&grid(8064, 6048)).is_ok());
        assert!(matches!(parse_grid_descriptor(&grid(u32::MAX, u32::MAX)), Err(HeifError::LimitExceeded(_))));
        assert!(matches!(parse_grid_descriptor(&grid(0, 512)), Err(HeifError::MalformedBox(_))));
    }

    #[test]
    fn parses_minimal_synthetic_container() {
        let coded = vec![0x26, 0x01, 0xAF, 0x08]; // arbitrary stand-in NAL bytes
        let file = build_minimal_heic(&coded);
        let parsed = parse(&file).expect("minimal synthetic container should parse");
        assert_eq!(parsed.primary_item_id, 1);
        assert!(parsed.grid.is_none());
        assert_eq!(parsed.images.len(), 1);
        assert_eq!(parsed.images[0].hevc_config.chroma_format_idc, 1);
        assert_eq!(parsed.images[0].hevc_config.length_size_minus_one, 3);
        assert_eq!(parsed.images[0].data, coded);
        assert!(parsed.transforms.is_empty());
    }

    #[test]
    fn rejects_non_heif_input() {
        let err = parse(b"not a heif file at all, just plain bytes").unwrap_err();
        assert!(matches!(err, HeifError::NotHeif(_)));
    }

    #[test]
    fn length_prefixed_nalu_split() {
        let data = [0, 0, 0, 2, 0xAA, 0xBB, 0, 0, 0, 1, 0xCC];
        let nalus = split_length_prefixed_nalus(&data, 4).unwrap();
        assert_eq!(nalus, vec![vec![0xAA, 0xBB], vec![0xCC]]);
    }

    #[test]
    fn truncated_box_header_errs() {
        assert!(parse(&[0, 0, 0]).is_err());
    }
}
