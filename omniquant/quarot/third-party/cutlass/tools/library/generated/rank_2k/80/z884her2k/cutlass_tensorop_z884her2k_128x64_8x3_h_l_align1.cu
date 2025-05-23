
/*
  Generated by rank_2k_operation.py - Do not edit.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

#include "library_internal.h"
#include "rank_2k_operation.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


// Rank K operator cutlass_tensorop_z884her2k_128x64_8x3_h_l_align1
using Operation_cutlass_tensorop_z884her2k_128x64_8x3_h_l_align1 =
  typename cutlass::gemm::device::Rank2K<
    cutlass::complex<double>, cutlass::layout::RowMajor,
    cutlass::complex<double>, cutlass::layout::RowMajor,
    cutlass::complex<double>, cutlass::layout::ColumnMajor, cutlass::FillMode::kLower,
    cutlass::complex<double>,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 8>,
    cutlass::gemm::GemmShape<32, 32, 8>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::complex<double>,
      1,
      cutlass::complex<double>,
      cutlass::complex<double>
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    1,
    1,
    false,
    cutlass::arch::OpMultiplyAddComplex,
    cutlass::ComplexTransform::kConjugate,
    cutlass::ComplexTransform::kConjugate,
    cutlass::BlasMode::kHermitian
>;


///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_cutlass_tensorop_z884her2k_128x64_8x3_h_l_align1(Manifest &manifest) {



  manifest.append(new Rank2KOperation<
    Operation_cutlass_tensorop_z884her2k_128x64_8x3_h_l_align1
  >("cutlass_tensorop_z884her2k_128x64_8x3_h_l_align1"));



}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

