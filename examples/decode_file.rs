//! Manual smoke-test driver: `cargo run -p heic-decoder-rust --example decode_file -- <path.heic> [out.yuv]`.
//! Not part of the library's public API or test suite.

use std::env;
use std::fs;
use std::process::ExitCode;

fn main() -> ExitCode {
    let Some(path) = env::args().nth(1) else {
        eprintln!("usage: decode_file <path.heic> [out.yuv]");
        return ExitCode::FAILURE;
    };
    let bytes = match fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("read error: {e}");
            return ExitCode::FAILURE;
        }
    };

    match heic_decoder::decode(&bytes) {
        Ok(img) => {
            println!(
                "DECODED: {}x{} bit_depth={}/{} chroma_format_idc={}",
                img.width, img.height, img.bit_depth_luma, img.bit_depth_chroma, img.chroma_format_idc
            );
            if let Some(out) = env::args().nth(2) {
                let mut raw = Vec::new();
                for (samples, depth) in [(&img.y, img.bit_depth_luma), (&img.cb, img.bit_depth_chroma), (&img.cr, img.bit_depth_chroma)] {
                    for &v in samples {
                        if depth > 8 {
                            raw.extend_from_slice(&(v as u16).to_le_bytes());
                        } else {
                            raw.push(v as u8);
                        }
                    }
                }
                fs::write(&out, &raw).expect("write yuv");
                println!("wrote raw planar YUV (Y then Cb then Cr, no padding) to {out}");
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            println!("decode error: {e}");
            ExitCode::FAILURE
        }
    }
}
