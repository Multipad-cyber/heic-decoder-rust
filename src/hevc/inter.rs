//! Single-reference 8-bit motion compensation (H.265 8.5.3).
//! Filter coefficients are the normative HEVC luma/chroma interpolation tables.
use super::ctu::Plane;
use crate::error::{HeifError, Result};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Motion { pub x: i32, pub y: i32 }
impl Motion {
    pub fn with_delta(self, delta: Self) -> Self {
        Self { x: (self.x + delta.x) as i16 as i32, y: (self.y + delta.y) as i16 as i32 }
    }
}

const LUMA: [[i32; 8]; 4] = [
    [0,0,0,64,0,0,0,0], [-1,4,-10,58,17,-5,1,0],
    [-1,4,-11,40,40,-11,4,-1], [0,1,-5,17,58,-10,4,-1],
];
const CHROMA: [[i32; 4]; 8] = [
    [0,64,0,0], [-2,58,10,-2], [-4,54,16,-2], [-6,46,28,-4],
    [-4,36,36,-4], [-4,28,46,-6], [-2,16,54,-4], [-2,10,58,-2],
];

#[allow(clippy::too_many_arguments)]
pub fn compensate(reference: &Plane, dest: &mut Plane, x: usize, y: usize, width: usize, height: usize, mv: Motion, chroma: bool) -> Result<()> {
    if reference.bit_depth != 8 || dest.bit_depth != 8 { return Err(HeifError::Unsupported("motion compensation: only 8-bit")); }
    let denominator = if chroma { 8 } else { 4 };
    let phase_x = mv.x.rem_euclid(denominator) as usize;
    let phase_y = mv.y.rem_euclid(denominator) as usize;
    let integer_x = mv.x.div_euclid(denominator);
    let integer_y = mv.y.div_euclid(denominator);
    let (fx, fy, origin): (&[i32], &[i32], i32) = if chroma { (&CHROMA[phase_x], &CHROMA[phase_y], 1) } else { (&LUMA[phase_x], &LUMA[phase_y], 3) };
    for row in 0..height {
        for col in 0..width {
            let mut sum = 0i32;
            for (j, &vy) in fy.iter().enumerate() {
                if vy == 0 { continue; }
                for (i, &vx) in fx.iter().enumerate() {
                    if vx == 0 { continue; }
                    let sx = (x as i32 + col as i32 + integer_x + i as i32 - origin).clamp(0, reference.width as i32 - 1);
                    let sy = (y as i32 + row as i32 + integer_y + j as i32 - origin).clamp(0, reference.height as i32 - 1);
                    sum += vx * vy * reference.sample(sx as usize, sy as usize).ok_or(HeifError::MalformedHevc("motion reference bounds"))?;
                }
            }
            dest.set(x + col, y + row, ((sum + 2048) >> 12).clamp(0, 255))?;
        }
    }
    Ok(())
}
