//! Per-syntax-element CABAC context arrays and their `initValue` tables
//! (clause 9.3.2.2, Table 9-4), scoped to I-slices (`initType` is always
//! 0 for an I-slice, clause 9.3.2.2, so the `initType` 1/2 columns the
//! spec defines for P/B slices are never reached and aren't stored).
//!
//! The numeric `initValue` tables were checked against `libde265`'s
//! `contextmodel.cc` (LGPL-3.0) rather than typed from memory. See this
//! crate's `README.md`.

use super::cabac::ContextModel;

fn init_array<const N: usize>(values: [u8; N], qp: i32) -> [ContextModel; N] {
    let mut out = [ContextModel { state: 0, mps: 0 }; N];
    for (o, v) in out.iter_mut().zip(values.iter()) {
        *o = ContextModel::init(*v, qp);
    }
    out
}

#[derive(Clone)]
pub struct Contexts {
    pub skip: [ContextModel; 3],
    pub pred_mode: [ContextModel; 1],
    pub merge: [ContextModel; 1],
    pub merge_idx: [ContextModel; 1],
    pub mvp: [ContextModel; 1],
    pub mvd: [ContextModel; 2],
    pub root_cbf: [ContextModel; 1],
    pub split_cu_flag: [ContextModel; 3],
    pub part_mode: [ContextModel; 4],
    pub prev_intra_luma_pred_flag: [ContextModel; 1],
    pub intra_chroma_pred_mode: [ContextModel; 1],
    pub cbf_luma: [ContextModel; 2],
    pub cbf_chroma: [ContextModel; 4],
    pub split_transform_flag: [ContextModel; 3],
    pub last_sig_coeff_x_prefix: [ContextModel; 18],
    pub last_sig_coeff_y_prefix: [ContextModel; 18],
    pub coded_sub_block_flag: [ContextModel; 4],
    pub significant_coeff_flag: [ContextModel; 42],
    pub coeff_abs_level_greater1_flag: [ContextModel; 24],
    pub coeff_abs_level_greater2_flag: [ContextModel; 6],
    pub cu_qp_delta_abs: [ContextModel; 2],
    pub transform_skip_flag: [ContextModel; 2],
    pub sao_merge_flag: [ContextModel; 1],
    pub sao_type_idx: [ContextModel; 1],
}

impl Contexts {
    /// P-slice initialization constants (H.265 Table 9-4), cross-checked
    /// against HM ContextTables.h. Unchanged syntax groups reuse I init.
    pub fn init_p_slice(qp: i32) -> Self {
        let mut c = Self::init_i_slice(qp);
        c.skip = init_array([197,185,201], qp);
        c.pred_mode = init_array([149], qp);
        c.merge = init_array([110], qp);
        c.merge_idx = init_array([122], qp);
        c.mvp = init_array([168], qp);
        c.mvd = init_array([140,198], qp);
        c.root_cbf = init_array([79], qp);
        c.split_cu_flag = init_array([107,139,126], qp);
        c.part_mode = init_array([154,139,154,154], qp);
        c.prev_intra_luma_pred_flag = init_array([154], qp);
        c.intra_chroma_pred_mode = init_array([152], qp);
        c.cbf_luma = init_array([153,111], qp);
        c.cbf_chroma = init_array([149,107,167,154], qp);
        c.split_transform_flag = init_array([124,138,94], qp);
        c.last_sig_coeff_x_prefix = init_array([125,110,94,110,95,79,125,111,110,78,110,111,111,95,94,108,123,108], qp);
        c.last_sig_coeff_y_prefix = c.last_sig_coeff_x_prefix;
        c.coded_sub_block_flag = init_array([121,140,61,154], qp);
        c.significant_coeff_flag = init_array([155,154,139,153,139,123,123,63,153,166,183,140,136,153,154,166,183,140,136,153,154,166,183,140,136,153,154,170,153,123,123,107,121,107,121,167,151,183,140,151,183,140], qp);
        c.coeff_abs_level_greater1_flag = init_array([154,196,196,167,154,152,167,182,182,134,149,136,153,121,136,137,169,194,166,167,154,167,137,182], qp);
        c.coeff_abs_level_greater2_flag = init_array([107,167,91,122,107,167], qp);
        c.sao_type_idx = init_array([185], qp);
        c
    }

    /// `initType = 0` (I-slice) initialization, clause 9.3.2.2, using
    /// `SliceQpY` as the QP input to `set_initValue`.
    pub fn init_i_slice(qp: i32) -> Self {
        Contexts {
            skip: init_array([154; 3], qp), pred_mode: init_array([154], qp),
            merge: init_array([154], qp), merge_idx: init_array([154], qp),
            mvp: init_array([154], qp), mvd: init_array([154; 2], qp), root_cbf: init_array([154], qp),
            split_cu_flag: init_array([139, 141, 157], qp),
            part_mode: init_array([184, 154, 139, 154], qp),
            prev_intra_luma_pred_flag: init_array([184], qp),
            intra_chroma_pred_mode: init_array([63], qp),
            cbf_luma: init_array([111, 141], qp),
            cbf_chroma: init_array([94, 138, 182, 154], qp),
            split_transform_flag: init_array([153, 138, 138], qp),
            last_sig_coeff_x_prefix: init_array([110, 110, 124, 125, 140, 153, 125, 127, 140, 109, 111, 143, 127, 111, 79, 108, 123, 63], qp),
            last_sig_coeff_y_prefix: init_array([110, 110, 124, 125, 140, 153, 125, 127, 140, 109, 111, 143, 127, 111, 79, 108, 123, 63], qp),
            coded_sub_block_flag: init_array([91, 171, 134, 141], qp),
            significant_coeff_flag: init_array(
                [
                    111, 111, 125, 110, 110, 94, 124, 108, 124, 107, 125, 141, 179, 153, 125, 107, 125, 141, 179, 153, 125, 107, 125, 141, 179, 153, 125, 140, 139, 182, 182, 152, 136, 152, 136, 153,
                    136, 139, 111, 136, 139, 111,
                ],
                qp,
            ),
            coeff_abs_level_greater1_flag: init_array(
                [140, 92, 137, 138, 140, 152, 138, 139, 153, 74, 149, 92, 139, 107, 122, 152, 140, 179, 166, 182, 140, 227, 122, 197],
                qp,
            ),
            coeff_abs_level_greater2_flag: init_array([138, 153, 136, 167, 152, 152], qp),
            cu_qp_delta_abs: init_array([154, 154], qp),
            transform_skip_flag: init_array([139, 139], qp),
            sao_merge_flag: init_array([153], qp),
            sao_type_idx: init_array([200], qp),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_produces_the_expected_count_of_each_context_group() {
        let c = Contexts::init_i_slice(32);
        assert_eq!(c.significant_coeff_flag.len(), 42);
        assert_eq!(c.coeff_abs_level_greater1_flag.len(), 24);
        assert_eq!(c.coeff_abs_level_greater2_flag.len(), 6);
    }

    #[test]
    fn last_sig_coeff_x_and_y_start_equal_but_are_independent_storage() {
        let mut c = Contexts::init_i_slice(32);
        assert_eq!(c.last_sig_coeff_x_prefix, c.last_sig_coeff_y_prefix);
        c.last_sig_coeff_x_prefix[0].state = 7;
        assert_ne!(c.last_sig_coeff_x_prefix[0], c.last_sig_coeff_y_prefix[0]);
    }
}
