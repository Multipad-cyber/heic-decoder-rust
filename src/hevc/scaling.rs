//! Scaling lists (7.3.4, 7.4.5 and 8.6.3). Coefficients are stored in
//! raster order; the separately signaled DC replaces only position (0,0)
//! after the 8x8 matrix has been expanded for 16x16/32x32 transforms.

use super::scan::scan_order;
use crate::bitreader::BitReader;
use crate::error::{HeifError, Result};

// H.265 Tables 7-5 and 7-6, in raster order.
const DEFAULT_INTRA: [i32; 64] = [
    16,16,16,16,17,18,21,24, 16,16,16,16,17,19,22,25,
    16,16,17,18,20,22,25,29, 16,16,18,21,24,27,31,36,
    17,17,20,24,30,35,41,47, 18,19,22,27,35,44,54,65,
    21,22,25,31,41,54,70,88, 24,25,29,36,47,65,88,115,
];
const DEFAULT_INTER: [i32; 64] = [
    16,16,16,16,17,18,20,24, 16,16,16,17,18,20,24,25,
    16,16,17,18,20,24,25,28, 16,17,18,20,24,25,28,33,
    17,18,20,24,25,28,33,41, 18,20,24,25,28,33,41,54,
    20,24,25,28,33,41,54,71, 24,25,28,33,41,54,71,91,
];

#[derive(Debug, Clone)]
pub struct ScalingListData {
    /// `[sizeId][matrixId]`: 16 coefficients for 4x4, otherwise 64.
    pub coeffs: [[Vec<i32>; 6]; 4],
    /// `[sizeId - 2][matrixId]` for 16x16 and 32x32 transforms.
    pub dc: [[i32; 6]; 2],
}

impl Default for ScalingListData {
    fn default() -> Self {
        let coeffs = std::array::from_fn(|size| std::array::from_fn(|matrix| {
            if size == 0 { vec![16; 16] }
            else if matrix < 3 { DEFAULT_INTRA.to_vec() }
            else { DEFAULT_INTER.to_vec() }
        }));
        Self { coeffs, dc: [[16; 6]; 2] }
    }
}

impl ScalingListData {
    pub fn weight(&self, log2_size: u32, matrix: usize, x: usize, y: usize) -> Result<i32> {
        if !(2..=5).contains(&log2_size) || matrix >= 6 {
            return Err(HeifError::MalformedHevc("scaling list: invalid size or matrix"));
        }
        let size = (log2_size - 2) as usize;
        let nt = 1usize << log2_size;
        if x >= nt || y >= nt {
            return Err(HeifError::MalformedHevc("scaling list: coefficient out of range"));
        }
        // For 4:4:4, 32x32 chroma matrices are inferred from the 16x16
        // chroma lists (only matrix IDs 0 and 3 are signaled at sizeId 3).
        let stored_size = if size == 3 && !matrix.is_multiple_of(3) { 2 } else { size };
        if size >= 2 && x == 0 && y == 0 {
            return Ok(self.dc[stored_size - 2][matrix]);
        }
        let side = nt.min(8);
        let step = nt / side;
        self.coeffs[stored_size][matrix].get((y / step) * side + x / step)
            .copied().ok_or(HeifError::MalformedHevc("scaling list: incomplete matrix"))
    }
}

pub(crate) fn parse_scaling_list_data(r: &mut BitReader<'_>) -> Result<ScalingListData> {
    let mut lists = ScalingListData::default();
    for size in 0..4 {
        let scan = scan_order(if size == 0 { 2 } else { 3 }, 0)?;
        let step = if size == 3 { 3 } else { 1 };
        for matrix in (0..6).step_by(step) {
            if !r.flag()? {
                let delta = r.ue()? as usize;
                if delta > matrix / step {
                    return Err(HeifError::MalformedHevc("scaling list: invalid predictor"));
                }
                if delta != 0 {
                    let source = matrix - delta * step;
                    lists.coeffs[size][matrix] = lists.coeffs[size][source].clone();
                    if size >= 2 { lists.dc[size - 2][matrix] = lists.dc[size - 2][source]; }
                }
                // delta == 0 selects the default matrix already installed.
            } else {
                let mut next = 8;
                if size >= 2 {
                    let delta = r.se()?;
                    if !(-7..=247).contains(&delta) {
                        return Err(HeifError::MalformedHevc("scaling list: invalid DC"));
                    }
                    next = delta + 8;
                    lists.dc[size - 2][matrix] = next;
                }
                let side = if size == 0 { 4 } else { 8 };
                for pos in scan {
                    let delta = r.se()?;
                    if !(-128..=127).contains(&delta) {
                        return Err(HeifError::MalformedHevc("scaling list: invalid delta"));
                    }
                    next = (next + delta).rem_euclid(256);
                    lists.coeffs[size][matrix][pos.y as usize * side + pos.x as usize] = next;
                }
            }
        }
    }
    Ok(lists)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ue(bits: &mut Vec<u8>, value: u32) {
        let code = value + 1;
        let n = 32 - code.leading_zeros();
        bits.extend(std::iter::repeat_n(0, (n - 1) as usize));
        bits.extend((0..n).rev().map(|i| ((code >> i) & 1) as u8));
    }

    fn se(bits: &mut Vec<u8>, value: i32) {
        ue(bits, if value > 0 { (2 * value - 1) as u32 } else { (-2 * value) as u32 });
    }

    fn pack(bits: &[u8]) -> Vec<u8> {
        bits.chunks(8).map(|chunk| chunk.iter().enumerate()
            .fold(0, |byte, (i, bit)| byte | (bit << (7 - i)))).collect()
    }

    #[test]
    fn default_matrices_expand_by_replication() {
        let list = ScalingListData::default();
        assert_eq!(list.weight(2, 0, 3, 3).unwrap(), 16);
        assert_eq!(list.weight(3, 0, 7, 7).unwrap(), 115);
        assert_eq!(list.weight(3, 3, 7, 7).unwrap(), 91);
        assert_eq!(list.weight(4, 0, 14, 14).unwrap(), 115);
        assert_eq!(list.weight(4, 0, 15, 15).unwrap(), 115);
        assert_eq!(list.weight(5, 0, 31, 31).unwrap(), 115);
        assert_eq!(list.weight(5, 0, 0, 0).unwrap(), 16);
    }

    #[test]
    fn explicit_lists_use_diagonal_scan_and_preserve_separate_dc() {
        let mut bits = Vec::new();
        for size in 0..4 {
            for matrix in (0..6).step_by(if size == 3 { 3 } else { 1 }) {
                if size == 2 && matrix == 0 {
                    bits.push(1); // explicit 16x16 intra-luma matrix
                    se(&mut bits, 12); // DC = 20
                    for i in 0..64 { se(&mut bits, if i < 2 { 1 } else { 0 }); }
                } else {
                    bits.push(0);
                    // Copy both the DC and matrix into the next slot.
                    ue(&mut bits, u32::from(size == 2 && matrix == 1));
                }
            }
        }
        let bytes = pack(&bits);
        let lists = parse_scaling_list_data(&mut BitReader::new(&bytes)).unwrap();
        for matrix in [0, 1] {
            assert_eq!(lists.weight(4, matrix, 0, 0).unwrap(), 20);
            assert_eq!(lists.weight(4, matrix, 1, 0).unwrap(), 21);
            assert_eq!(lists.weight(4, matrix, 0, 1).unwrap(), 21);
            assert_eq!(lists.weight(4, matrix, 0, 2).unwrap(), 22);
        }
        // SizeId 3 chroma uses the corresponding sizeId 2 list.
        assert_eq!(lists.weight(5, 1, 0, 0).unwrap(), 20);
        assert_eq!(lists.weight(5, 1, 3, 3).unwrap(), 21);
        assert_eq!(lists.weight(5, 1, 0, 4).unwrap(), 22);
    }

    #[test]
    fn explicit_asymmetric_matrix_is_not_left_in_scan_order() {
        let mut bits = vec![1];
        for _ in 0..16 { se(&mut bits, 1); } // 9,10,11,... in scan order
        for _ in 1..20 { bits.push(0); ue(&mut bits, 0); }
        let bytes = pack(&bits);
        let lists = parse_scaling_list_data(&mut BitReader::new(&bytes)).unwrap();
        assert_eq!(lists.weight(2, 0, 0, 0).unwrap(), 9);
        assert_eq!(lists.weight(2, 0, 0, 1).unwrap(), 10);
        assert_eq!(lists.weight(2, 0, 1, 0).unwrap(), 11);
    }

    #[test]
    fn rejects_forward_predictor_and_out_of_range_delta() {
        let bytes = pack(&[0, 0, 1, 0]); // first matrix: predictor delta=1
        assert!(parse_scaling_list_data(&mut BitReader::new(&bytes)).is_err());
        let mut bits = vec![1];
        se(&mut bits, 128);
        let bytes = pack(&bits);
        assert!(parse_scaling_list_data(&mut BitReader::new(&bytes)).is_err());
    }

    #[test]
    fn size_32_predictor_copies_matrix_zero_and_its_dc() {
        let mut bits = Vec::new();
        for _ in 0..18 { bits.push(0); ue(&mut bits, 0); }
        bits.push(1); // sizeId=3, matrixId=0: explicit
        se(&mut bits, 15); // DC=23
        for _ in 0..64 { se(&mut bits, 1); }
        bits.push(0); // sizeId=3, matrixId=3: predicted
        ue(&mut bits, 1); // step of 3 -> copy matrixId 0
        let bytes = pack(&bits);
        let lists = parse_scaling_list_data(&mut BitReader::new(&bytes)).unwrap();
        assert_eq!(lists.weight(5, 3, 0, 0).unwrap(), 23);
        assert_eq!(lists.weight(5, 3, 1, 0).unwrap(), 24);
        assert_eq!(lists.weight(5, 3, 0, 4).unwrap(), 25);
        assert_eq!(lists.weight(5, 3, 31, 31).unwrap(), 87);
    }
}
