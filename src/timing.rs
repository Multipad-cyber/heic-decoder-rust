//! Stage timers for profiling, behind the `timing` feature.
//!
//! Each stage of a decode adds its elapsed time and call count to process-wide
//! totals, which [`take`] reads and resets. Times are summed over threads, so
//! with parallel tiles a stage's total is CPU time, not wall time. Without the
//! feature, [`start`] returns a zero-sized guard and compiles away.
//!
//! The fine stages split slice decoding into its parts. They run once per
//! transform block, so their own clock reads cost time; they are off until
//! [`set_fine`] turns them on.

/// A stage of the decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// Box parsing and locating item data.
    Container,
    /// Parameter sets, NAL splitting, emulation prevention, slice headers.
    Headers,
    /// `slice_segment_data()`: CABAC parsing and reconstruction.
    SliceData,
    /// Fine, inside `SliceData`: `residual_coding()`.
    Residual,
    /// Fine, inside `SliceData`: reference samples, smoothing, prediction.
    Intra,
    /// Fine, inside `SliceData`: dequantisation and inverse transform.
    Transform,
    /// Fine, inside `SliceData`: prediction plus residual into the picture.
    Write,
    /// Deblocking filter.
    Deblock,
    /// Sample adaptive offset.
    Sao,
    /// Picture to output planes: cropping, sample type conversion.
    Pack,
    /// Assembling grid tiles into one picture.
    Grid,
    /// YCbCr to RGB.
    Color,
    /// `clap`, `irot` and `imir`.
    Transforms,
}

/// Every stage with its name, in report order.
pub const STAGES: [(Stage, &str); 13] = [
    (Stage::Container, "container"),
    (Stage::Headers, "headers"),
    (Stage::SliceData, "slice data"),
    (Stage::Residual, "  residual"),
    (Stage::Intra, "  intra"),
    (Stage::Transform, "  dequant+transform"),
    (Stage::Write, "  write"),
    (Stage::Deblock, "deblock"),
    (Stage::Sao, "sao"),
    (Stage::Pack, "pack"),
    (Stage::Grid, "grid"),
    (Stage::Color, "color"),
    (Stage::Transforms, "transforms"),
];

impl Stage {
    fn is_fine(self) -> bool {
        matches!(self, Stage::Residual | Stage::Intra | Stage::Transform | Stage::Write)
    }
}

#[cfg(feature = "timing")]
mod imp {
    use super::{STAGES, Stage};
    use std::cell::Cell;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
    use std::time::Instant;

    const N: usize = STAGES.len();
    static FINE: AtomicBool = AtomicBool::new(false);
    static NANOS: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
    static CALLS: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];

    std::thread_local! {
        // Fine stages add up here, without atomics, until the coarse stage around them ends.
        static LOCAL: [Cell<(u64, u64)>; N] = const { [const { Cell::new((0, 0)) }; N] };
    }

    /// Records its stage's time when dropped.
    pub struct Timer {
        stage: Stage,
        start: Option<Instant>,
    }

    /// Starts timing `stage` until the returned guard is dropped.
    #[inline]
    pub fn start(stage: Stage) -> Timer {
        let on = !stage.is_fine() || FINE.load(Relaxed);
        Timer { stage, start: on.then(Instant::now) }
    }

    impl Drop for Timer {
        #[inline]
        fn drop(&mut self) {
            let Some(start) = self.start else { return };
            let nanos = start.elapsed().as_nanos() as u64;
            let i = self.stage as usize;
            if self.stage.is_fine() {
                LOCAL.with(|local| {
                    let (n, c) = local[i].get();
                    local[i].set((n + nanos, c + 1));
                });
                return;
            }
            NANOS[i].fetch_add(nanos, Relaxed);
            CALLS[i].fetch_add(1, Relaxed);
            LOCAL.with(|local| {
                for (j, cell) in local.iter().enumerate() {
                    let (n, c) = cell.replace((0, 0));
                    if c != 0 {
                        NANOS[j].fetch_add(n, Relaxed);
                        CALLS[j].fetch_add(c, Relaxed);
                    }
                }
            });
        }
    }

    /// Turns the fine stages on or off.
    pub fn set_fine(on: bool) {
        FINE.store(on, Relaxed);
    }

    /// Every stage's name, total nanoseconds and call count since the last
    /// call, in report order; resets the totals.
    pub fn take() -> [(&'static str, u64, u64); N] {
        STAGES.map(|(stage, name)| (name, NANOS[stage as usize].swap(0, Relaxed), CALLS[stage as usize].swap(0, Relaxed)))
    }
}

#[cfg(not(feature = "timing"))]
mod imp {
    use super::Stage;

    pub struct Timer;

    // So that ending a stage early with `drop` reads the same in both builds.
    impl Drop for Timer {
        #[inline(always)]
        fn drop(&mut self) {}
    }

    #[inline(always)]
    pub fn start(_: Stage) -> Timer {
        Timer
    }
}

pub use imp::*;
