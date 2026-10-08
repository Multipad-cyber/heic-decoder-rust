//! Resolve one sync sample from a non-fragmented image-sequence track.
//! Table traversal is separate from HEVC decoding; no video frame state is kept.
use super::*;

fn children<'a>(data: &'a [u8], kind: &[u8; 4]) -> Result<Vec<&'a [u8]>> {
    let mut found = Vec::new();
    for_each_child(data, 0, data.len(), 0, |tag, body| {
        if tag == kind { found.push(body); }
        Ok(())
    })?;
    Ok(found)
}

fn one<'a>(data: &'a [u8], kind: &[u8; 4]) -> Result<&'a [u8]> {
    let found = children(data, kind)?;
    let [body] = found.as_slice() else {
        return Err(HeifError::MalformedBox("sequence: missing or duplicate required box"));
    };
    Ok(body)
}

fn table(body: &[u8], row_bytes: usize) -> Result<ByteReader<'_>> {
    let mut r = ByteReader::new(body);
    let (version, _) = full_box_header(&mut r)?;
    if version != 0 { return Err(HeifError::Unsupported("sequence: table version")); }
    let count = r.u32()? as usize;
    if count.checked_mul(row_bytes) != Some(r.remaining()) {
        return Err(HeifError::MalformedBox("sequence: table count/length mismatch"));
    }
    Ok(r)
}

struct Sizes<'a> { count: u32, uniform: u32, entries: &'a [u8] }
impl<'a> Sizes<'a> {
    fn parse(body: &'a [u8]) -> Result<Self> {
        let mut r = ByteReader::new(body);
        let (version, _) = full_box_header(&mut r)?;
        if version != 0 { return Err(HeifError::Unsupported("sequence: stsz version")); }
        let uniform = r.u32()?;
        let count = r.u32()?;
        let bytes = if uniform == 0 { (count as usize).checked_mul(4) } else { Some(0) };
        if count == 0 || bytes != Some(r.remaining()) {
            return Err(HeifError::MalformedBox("sequence: invalid sample sizes"));
        }
        Ok(Self { count, uniform, entries: &body[r.pos()..] })
    }
    fn size(&self, index: u32) -> Result<u32> {
        if index >= self.count { return Err(HeifError::MalformedBox("sequence: sample index out of range")); }
        if self.uniform != 0 { return Ok(self.uniform); }
        let mut r = ByteReader::new(self.entries);
        r.skip(index as usize * 4)?;
        r.u32()
    }
}

// Return file offset, length and sample-description index for a zero-based sample.
fn locate_sample(stsc: &[u8], offsets: &[u8], wide: bool, sizes: &Sizes<'_>, sample: u32) -> Result<(u64, u32, u32)> {
    let mut r = table(stsc, 12)?;
    let mut runs = Vec::new();
    while r.remaining() != 0 {
        let row = (r.u32()?, r.u32()?, r.u32()?);
        if row.0 == 0 || row.1 == 0 || row.2 == 0 || runs.last().is_some_and(|prev: &(u32, u32, u32)| prev.0 >= row.0) {
            return Err(HeifError::MalformedBox("sequence: invalid stsc run"));
        }
        runs.push(row);
    }
    if runs.first().map(|r| r.0) != Some(1) || sample >= sizes.count {
        return Err(HeifError::MalformedBox("sequence: missing sample-to-chunk mapping"));
    }
    let mut chunks = table(offsets, if wide { 8 } else { 4 })?;
    let chunk_count = chunks.remaining() / if wide { 8 } else { 4 };
    if runs.last().is_some_and(|r| r.0 as usize > chunk_count) {
        return Err(HeifError::MalformedBox("sequence: stsc run outside chunk table"));
    }
    let mut run = 0;
    let mut first_sample = 0u64;
    for chunk in 1..=chunk_count {
        let mut offset = if wide { chunks.u64()? } else { u64::from(chunks.u32()?) };
        if run + 1 < runs.len() && chunk == runs[run + 1].0 as usize { run += 1; }
        let (_, count, description) = runs[run];
        let end = first_sample.checked_add(u64::from(count)).ok_or(HeifError::MalformedBox("sequence: sample count overflow"))?;
        if u64::from(sample) < end {
            if sizes.uniform != 0 {
                let skip = (u64::from(sample) - first_sample) * u64::from(sizes.uniform);
                offset = offset.checked_add(skip).ok_or(HeifError::MalformedBox("sequence: sample offset overflow"))?;
            } else {
                for index in first_sample..u64::from(sample) {
                    offset = offset.checked_add(u64::from(sizes.size(index as u32)?)).ok_or(HeifError::MalformedBox("sequence: sample offset overflow"))?;
                }
            }
            return Ok((offset, sizes.size(sample)?, description));
        }
        first_sample = end;
    }
    Err(HeifError::MalformedBox("sequence: sample not covered by chunks"))
}

pub(super) fn parse(file: &[u8], moov: &[u8]) -> Result<ParsedHeif> {
    let tracks = children(moov, b"trak")?;
    let mut selected = None;
    // Prefer image tracks regardless of their order relative to video tracks.
    for wanted in [b"pict", b"vide"] {
        for &track in &tracks {
            let mdia = one(track, b"mdia")?;
            let hdlr = one(mdia, b"hdlr")?;
            if hdlr.get(8..12) == Some(wanted.as_slice()) { selected = Some((track, mdia)); break; }
        }
        if selected.is_some() { break; }
    }
    let (track, mdia) = selected.ok_or(HeifError::Unsupported("sequence: no image/video track"))?;
    let mut tkhd = ByteReader::new(one(track, b"tkhd")?);
    let (version, _) = full_box_header(&mut tkhd)?;
    match version { 0 => tkhd.skip(8)?, 1 => tkhd.skip(16)?, _ => return Err(HeifError::Unsupported("sequence: tkhd version")) }
    let track_id = tkhd.u32()?;
    tkhd.skip(if version == 0 { 24 } else { 28 })?;
    for expected in [0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x40000000] {
        if tkhd.u32()? != expected { return Err(HeifError::Unsupported("sequence: track matrix transform")); }
    }
    let minf = one(mdia, b"minf")?;
    let stbl = one(minf, b"stbl")?;
    let sizes = Sizes::parse(one(stbl, b"stsz")?)?;
    let sync = children(stbl, b"stss")?;
    let sample_number = match sync.as_slice() {
        [] => 1,
        [body] => {
            let mut r = table(body, 4)?;
            if r.remaining() == 0 { return Err(HeifError::Unsupported("sequence: no sync samples")); }
            let first = r.u32()?;
            let mut previous = first;
            while r.remaining() > 0 {
                let next = r.u32()?;
                if next <= previous || next > sizes.count { return Err(HeifError::MalformedBox("sequence: invalid sync sample list")); }
                previous = next;
            }
            first
        }
        _ => return Err(HeifError::MalformedBox("sequence: duplicate stss")),
    };
    if sample_number == 0 || sample_number > sizes.count { return Err(HeifError::MalformedBox("sequence: invalid sync sample")); }
    let small = children(stbl, b"stco")?;
    let large = children(stbl, b"co64")?;
    let (offsets, wide) = match (small.as_slice(), large.as_slice()) {
        ([body], []) => (*body, false), ([], [body]) => (*body, true),
        _ => return Err(HeifError::MalformedBox("sequence: missing/ambiguous chunk offsets")),
    };
    let (offset, length, description) = locate_sample(one(stbl, b"stsc")?, offsets, wide, &sizes, sample_number - 1)?;
    let stsd = one(stbl, b"stsd")?;
    let mut r = ByteReader::new(stsd);
    let (version, _) = full_box_header(&mut r)?;
    if version != 0 { return Err(HeifError::Unsupported("sequence: stsd version")); }
    let expected = r.u32()? as usize;
    let mut entries = Vec::new();
    for_each_child(stsd, r.pos(), stsd.len(), 0, |kind, body| { entries.push((*kind, body)); Ok(()) })?;
    if entries.len() != expected { return Err(HeifError::MalformedBox("sequence: stsd count mismatch")); }
    let (kind, entry) = entries.get(description as usize - 1).ok_or(HeifError::MalformedBox("sequence: sample description out of range"))?;
    if kind != b"hvc1" { return Err(HeifError::Unsupported("sequence: sample entry is not hvc1")); }
    let mut header = ByteReader::new(entry);
    header.skip(6)?;
    let data_reference = header.u16()?;
    let dref = one(one(minf, b"dinf")?, b"dref")?;
    let mut dr = ByteReader::new(dref);
    let (version, _) = full_box_header(&mut dr)?;
    if version != 0 { return Err(HeifError::Unsupported("sequence: dref version")); }
    let count = dr.u32()? as usize;
    let mut references = Vec::new();
    for_each_child(dref, dr.pos(), dref.len(), 0, |kind, body| { references.push((*kind, body)); Ok(()) })?;
    if references.len() != count || data_reference == 0 { return Err(HeifError::MalformedBox("sequence: invalid data reference")); }
    let (kind, body) = references.get(data_reference as usize - 1).ok_or(HeifError::MalformedBox("sequence: missing data reference"))?;
    let mut dr = ByteReader::new(body);
    let (version, flags) = full_box_header(&mut dr)?;
    if kind != b"url " || version != 0 || flags != 1 {
        return Err(HeifError::Unsupported("sequence: external data reference"));
    }
    // VisualSampleEntry has a 78-byte fixed body before child properties.
    let extension = entry.get(78..).ok_or(HeifError::MalformedBox("sequence: short visual sample entry"))?;
    let hevc_config = parse_hvcc(one(extension, b"hvcC")?)?;
    let mut transforms = Vec::new();
    let mut metadata = Metadata::default();
    for_each_child(extension, 0, extension.len(), 0, |kind, body| {
        if kind == b"clap" { transforms.push(Transform::CleanAperture(parse_clean_aperture(body)?)); }
        if kind == b"colr" && metadata.color_info.is_none() { metadata.color_info = parse_nclx(body)?; }
        Ok(())
    })?;
    let start = usize::try_from(offset).map_err(|_| HeifError::MalformedBox("sequence: offset overflow"))?;
    let end = start.checked_add(length as usize).ok_or(HeifError::MalformedBox("sequence: extent overflow"))?;
    let data = file.get(start..end).ok_or(HeifError::MalformedBox("sequence: sample outside file"))?.to_vec();
    Ok(ParsedHeif {
        alpha: None,
        reference_image: None,
        primary_item_id: track_id, grid: None, transforms, metadata,
        images: vec![ResolvedImage { hevc_config, data }],
        sequence: Some(SequenceFrameInfo { track_id, sample_number, sample_count: sizes.count }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn words(values: &[u32]) -> Vec<u8> { values.iter().flat_map(|v| v.to_be_bytes()).collect() }

    #[test]
    fn sample_location_handles_run_changes_and_description_selection() {
        let data = words(&[0, 0, 6, 10, 20, 30, 40, 50, 60]);
        let sizes = Sizes::parse(&data).unwrap();
        let runs = words(&[0, 2, 1, 2, 1, 3, 1, 2]);
        let offsets = words(&[0, 4, 100, 500, 900, 1000]);
        assert_eq!(locate_sample(&runs, &offsets, false, &sizes, 1).unwrap(), (110, 20, 1));
        assert_eq!(locate_sample(&runs, &offsets, false, &sizes, 3).unwrap(), (530, 40, 1));
        assert_eq!(locate_sample(&runs, &offsets, false, &sizes, 4).unwrap(), (900, 50, 2));
        assert_eq!(locate_sample(&runs, &offsets, false, &sizes, 5).unwrap(), (1000, 60, 2));
    }

    #[test]
    fn wide_offsets_and_uniform_sizes_do_not_truncate() {
        let data = words(&[0, 7, 3]);
        let sizes = Sizes::parse(&data).unwrap();
        let runs = words(&[0, 1, 1, 3, 1]);
        let mut offsets = words(&[0, 1]);
        offsets.extend_from_slice(&(u64::from(u32::MAX) + 100).to_be_bytes());
        assert_eq!(locate_sample(&runs, &offsets, true, &sizes, 2).unwrap(), (u64::from(u32::MAX) + 114, 7, 1));
        offsets[8..].copy_from_slice(&u64::MAX.to_be_bytes());
        assert!(locate_sample(&runs, &offsets, true, &sizes, 2).is_err());
    }

    #[test]
    fn malformed_sample_tables_are_rejected() {
        assert!(Sizes::parse(&words(&[0, 0, u32::MAX])).is_err());
        assert!(Sizes::parse(&words(&[0, 0, 0])).is_err());
        let data = words(&[0, 7, 3]);
        let sizes = Sizes::parse(&data).unwrap();
        let offsets = words(&[0, 1, 100]);
        for runs in [vec![0,1,2,3,1], vec![0,1,1,0,1], vec![0,1,1,3,0], vec![0,2,1,3,1,1,2,1], vec![0,1]] {
            assert!(locate_sample(&words(&runs), &offsets, false, &sizes, 0).is_err());
        }
        assert!(locate_sample(&words(&[0,1,1,1,1]), &offsets, false, &sizes, 2).is_err());
        assert!(table(&words(&[0,u32::MAX]), 4).is_err());
    }
}
