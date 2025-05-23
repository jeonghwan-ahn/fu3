
/*
  Generated by conv2d_operation.py - Do not edit.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

#include "library_internal.h"
#include "conv2d_operation.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


  // Conv2dFprop Fixed_channels kernel instance "cutlass_tensorop_f16_s16816fprop_fixed_channels_f16_256x128_32x3_nhwc_align4"
  using cutlass_tensorop_f16_s16816fprop_fixed_channels_f16_256x128_32x3_nhwc_align4_base =
  typename cutlass::conv::kernel::DefaultConv2dFprop<
    cutlass::half_t,
    cutlass::layout::TensorNHWC,
    cutlass::half_t,
    cutlass::layout::TensorNHWC,
    cutlass::half_t,
    cutlass::layout::TensorNHWC,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32 >,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      4,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
    3,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kFixedChannels,
    cutlass::conv::StrideSupport::kStrided,
    4,
    4
  >::Kernel;


// Derived class
struct cutlass_tensorop_f16_s16816fprop_fixed_channels_f16_256x128_32x3_nhwc_align4 :
  public cutlass_tensorop_f16_s16816fprop_fixed_channels_f16_256x128_32x3_nhwc_align4_base { };

///////////////////////////////////////////////////////////////////////////////////////////////////



namespace cutlass {
namespace library {

// Initialize all instances
void initialize_cutlass_tensorop_f16_s16816fprop_fixed_channels_f16_256x128_32x3_nhwc_align4(Manifest &manifest) {


  using Operation_cutlass_tensorop_f16_s16816fprop_fixed_channels_f16_256x128_32x3_nhwc_align4 = cutlass::conv::device::ImplicitGemmConvolution<
    cutlass_tensorop_f16_s16816fprop_fixed_channels_f16_256x128_32x3_nhwc_align4>;

  manifest.append(new cutlass::library::Conv2dOperation<
    Operation_cutlass_tensorop_f16_s16816fprop_fixed_channels_f16_256x128_32x3_nhwc_align4>(
      "cutlass_tensorop_f16_s16816fprop_fixed_channels_f16_256x128_32x3_nhwc_align4"));


}


///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

