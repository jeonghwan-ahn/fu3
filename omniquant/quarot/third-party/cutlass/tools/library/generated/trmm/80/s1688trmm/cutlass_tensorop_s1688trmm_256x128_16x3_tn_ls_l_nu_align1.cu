
/*
  Generated by trmm_operation.py - Do not edit.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

#include "library_internal.h"
#include "trmm_operation.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


// Trmm operator cutlass_tensorop_s1688trmm_256x128_16x3_tn_ls_l_nu_align1
using Operation_cutlass_tensorop_s1688trmm_256x128_16x3_tn_ls_l_nu_align1 =
  typename cutlass::gemm::device::Trmm<
    float, cutlass::layout::RowMajor,
    cutlass::SideMode::kLeft, cutlass::FillMode::kLower, cutlass::DiagType::kNonUnit,
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      1,
      float,
      float,
      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    1,
    1,
    false,
    cutlass::arch::OpMultiplyAddFastF32
>;


///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_cutlass_tensorop_s1688trmm_256x128_16x3_tn_ls_l_nu_align1(Manifest &manifest) {



  manifest.append(new TrmmOperation<
    Operation_cutlass_tensorop_s1688trmm_256x128_16x3_tn_ls_l_nu_align1
  >("cutlass_tensorop_s1688trmm_256x128_16x3_tn_ls_l_nu_align1"));



}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

