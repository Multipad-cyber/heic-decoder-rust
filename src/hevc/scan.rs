//! Coefficient scan orders (clause 6.5.3) and the intra-mode -> scanIdx
//! rule (clause 8.4.4.2.9 / Table 9-... as implemented in `residual_coding`).
//! Generation algorithm cross-checked against `libde265`'s `scan.cc`
//! (LGPL-3.0): see this crate's `README.md`.

use crate::error::{HeifError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pos {
    pub x: u32,
    pub y: u32,
}

/// Largest `log2_block_size` a scan table is built for (32x32).
const MAX_LOG2_SCAN: u32 = 5;

/// `scanIdx`: 0 = up-right diagonal, 1 = horizontal, 2 = vertical. The
/// tables are built once and shared, since residual decoding asks for
/// them on every transform block.
pub fn scan_order(log2_block_size: u32, scan_idx: u32) -> Result<&'static [Pos]> {
    static TABLES: std::sync::OnceLock<Vec<Vec<Pos>>> = std::sync::OnceLock::new();
    if scan_idx > 2 {
        return Err(HeifError::MalformedHevc("scan_order: scanIdx must be 0..=2"));
    }
    if log2_block_size > MAX_LOG2_SCAN {
        return Err(HeifError::MalformedHevc("scan_order: block too large"));
    }
    let tables = TABLES.get_or_init(|| (0..=MAX_LOG2_SCAN).flat_map(|log2| (0..3).map(move |idx| build_scan_order(log2, idx))).collect());
    tables.get((log2_block_size * 3 + scan_idx) as usize).map(Vec::as_slice).ok_or(HeifError::MalformedHevc("scan_order: table index"))
}

fn build_scan_order(log2_block_size: u32, scan_idx: u32) -> Vec<Pos> {
    let size = 1usize << log2_block_size;
    let mut out = Vec::with_capacity(size * size);
    match scan_idx {
        1 => {
            for y in 0..size {
                for x in 0..size {
                    out.push(Pos { x: x as u32, y: y as u32 });
                }
            }
        }
        2 => {
            for x in 0..size {
                for y in 0..size {
                    out.push(Pos { x: x as u32, y: y as u32 });
                }
            }
        }
        _ => {
            // Up-right diagonal scan (6.5.3): walk anti-diagonals,
            // each one from bottom-left to top-right.
            let mut x = 0i64;
            let mut y = 0i64;
            while out.len() < size * size {
                while y >= 0 {
                    if x < size as i64 && y < size as i64 {
                        out.push(Pos { x: x as u32, y: y as u32 });
                    }
                    y -= 1;
                    x += 1;
                }
                y = x;
                x = 0;
            }
        }
    }
    out
}

/// `get_intra_scan_idx` (used by `residual_coding` to pick the scan for
/// an intra-predicted transform block): mode-dependent scan only applies
/// to small luma/4:4:4-chroma blocks (clause 8.4.4.2.9).
pub fn intra_scan_idx(log2_trafo_size: u32, intra_pred_mode: u32, is_luma_or_444: bool) -> u32 {
    if log2_trafo_size == 2 || (log2_trafo_size == 3 && is_luma_or_444) {
        if (6..=14).contains(&intra_pred_mode) {
            2
        } else if (22..=30).contains(&intra_pred_mode) {
            1
        } else {
            0
        }
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn horizontal_scan_is_row_major() {
        let s = scan_order(2, 1).unwrap();
        assert_eq!(s[0], Pos { x: 0, y: 0 });
        assert_eq!(s[1], Pos { x: 1, y: 0 });
        assert_eq!(s[4], Pos { x: 0, y: 1 });
    }

    #[test]
    fn vertical_scan_is_col_major() {
        let s = scan_order(2, 2).unwrap();
        assert_eq!(s[0], Pos { x: 0, y: 0 });
        assert_eq!(s[1], Pos { x: 0, y: 1 });
        assert_eq!(s[4], Pos { x: 1, y: 0 });
    }

    #[test]
    fn diagonal_scan_4x4_known_order() {
        // Hand-verified against the spec's up-right-diagonal description:
        // DC first, then each anti-diagonal bottom-left to top-right.
        let s = scan_order(2, 0).unwrap();
        let expected = [
            Pos { x: 0, y: 0 },
            Pos { x: 0, y: 1 },
            Pos { x: 1, y: 0 },
            Pos { x: 0, y: 2 },
            Pos { x: 1, y: 1 },
            Pos { x: 2, y: 0 },
            Pos { x: 0, y: 3 },
            Pos { x: 1, y: 2 },
            Pos { x: 2, y: 1 },
            Pos { x: 3, y: 0 },
        ];
        assert_eq!(&s[..10], &expected);
        assert_eq!(s.len(), 16);
    }

    #[test]
    fn intra_scan_idx_matches_mode_bands() {
        assert_eq!(intra_scan_idx(2, 10, true), 2); // 6..=14 -> vertical-mode band
        assert_eq!(intra_scan_idx(2, 26, true), 1); // 22..=30 -> horizontal-mode band
        assert_eq!(intra_scan_idx(2, 0, true), 0);
        assert_eq!(intra_scan_idx(4, 10, true), 0); // only applies to log2size 2, or 3 for luma/444
    }
}
