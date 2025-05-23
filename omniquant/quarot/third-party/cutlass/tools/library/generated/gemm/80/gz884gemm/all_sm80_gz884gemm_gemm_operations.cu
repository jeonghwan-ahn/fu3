
/*
 Generated by manifest.py - Do not edit.
*/

#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_cutlass_tensorop_gz884gemm_64x64_8x3_nn_align1(Manifest &manifest);
void initialize_cutlass_tensorop_gz884gemm_64x64_8x3_cn_align1(Manifest &manifest);
void initialize_cutlass_tensorop_gz884gemm_64x64_8x3_nc_align1(Manifest &manifest);
void initialize_cutlass_tensorop_gz884gemm_64x64_8x3_cc_align1(Manifest &manifest);
void initialize_cutlass_tensorop_gz884gemm_64x64_8x3_nt_align1(Manifest &manifest);
void initialize_cutlass_tensorop_gz884gemm_64x64_8x3_ct_align1(Manifest &manifest);
void initialize_cutlass_tensorop_gz884gemm_64x64_8x3_nh_align1(Manifest &manifest);
void initialize_cutlass_tensorop_gz884gemm_64x64_8x3_ch_align1(Manifest &manifest);
void initialize_cutlass_tensorop_gz884gemm_64x64_8x3_tn_align1(Manifest &manifest);
void initialize_cutlass_tensorop_gz884gemm_64x64_8x3_hn_align1(Manifest &manifest);
void initialize_cutlass_tensorop_gz884gemm_64x64_8x3_tc_align1(Manifest &manifest);
void initialize_cutlass_tensorop_gz884gemm_64x64_8x3_hc_align1(Manifest &manifest);
void initialize_cutlass_tensorop_gz884gemm_64x64_8x3_tt_align1(Manifest &manifest);
void initialize_cutlass_tensorop_gz884gemm_64x64_8x3_ht_align1(Manifest &manifest);
void initialize_cutlass_tensorop_gz884gemm_64x64_8x3_th_align1(Manifest &manifest);
void initialize_cutlass_tensorop_gz884gemm_64x64_8x3_hh_align1(Manifest &manifest);


//
// Entry point to construct operations
//
void initialize_all_sm80_gz884gemm_gemm_operations(Manifest &manifest) {
  initialize_cutlass_tensorop_gz884gemm_64x64_8x3_nn_align1(manifest);
  initialize_cutlass_tensorop_gz884gemm_64x64_8x3_cn_align1(manifest);
  initialize_cutlass_tensorop_gz884gemm_64x64_8x3_nc_align1(manifest);
  initialize_cutlass_tensorop_gz884gemm_64x64_8x3_cc_align1(manifest);
  initialize_cutlass_tensorop_gz884gemm_64x64_8x3_nt_align1(manifest);
  initialize_cutlass_tensorop_gz884gemm_64x64_8x3_ct_align1(manifest);
  initialize_cutlass_tensorop_gz884gemm_64x64_8x3_nh_align1(manifest);
  initialize_cutlass_tensorop_gz884gemm_64x64_8x3_ch_align1(manifest);
  initialize_cutlass_tensorop_gz884gemm_64x64_8x3_tn_align1(manifest);
  initialize_cutlass_tensorop_gz884gemm_64x64_8x3_hn_align1(manifest);
  initialize_cutlass_tensorop_gz884gemm_64x64_8x3_tc_align1(manifest);
  initialize_cutlass_tensorop_gz884gemm_64x64_8x3_hc_align1(manifest);
  initialize_cutlass_tensorop_gz884gemm_64x64_8x3_tt_align1(manifest);
  initialize_cutlass_tensorop_gz884gemm_64x64_8x3_ht_align1(manifest);
  initialize_cutlass_tensorop_gz884gemm_64x64_8x3_th_align1(manifest);
  initialize_cutlass_tensorop_gz884gemm_64x64_8x3_hh_align1(manifest);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

