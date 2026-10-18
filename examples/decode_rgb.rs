//! Write optional RGB8/RGB16 conversion as binary PPM (no ICC metadata).
use std::io::{BufWriter, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let input = args.next().ok_or("usage: decode_rgb <input.heic> <output.ppm> [8|16]")?;
    let output = args.next().ok_or("usage: decode_rgb <input.heic> <output.ppm> [8|16]")?;
    let depth = args.next().unwrap_or_else(|| "8".to_owned());
    if depth != "8" && depth != "16" {
        return Err("output depth must be 8 or 16".into());
    }
    let image = heic_decoder::decode(&std::fs::read(input)?)?;
    let mut file = BufWriter::new(std::fs::File::create(output)?);
    writeln!(file, "P6\n{} {}\n{}", image.width, image.height, if depth == "8" { 255 } else { 65535 })?;
    if depth == "8" {
        file.write_all(&image.to_rgb8()?.data)?;
    } else {
        let rgb = image.to_rgb16()?;
        for row in rgb.data.chunks(image.width * 3) {
            let bytes: Vec<_> = row.iter().flat_map(|v| v.to_be_bytes()).collect();
            file.write_all(&bytes)?;
        }
    }
    file.flush()?;
    Ok(())
}
