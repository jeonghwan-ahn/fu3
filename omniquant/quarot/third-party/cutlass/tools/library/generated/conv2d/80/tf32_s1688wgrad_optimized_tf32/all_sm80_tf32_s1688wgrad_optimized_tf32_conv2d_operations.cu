
/*
 Generated by manifest.py - Do not edit.
*/

#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_cutlass_tensorop_tf32_s1688wgrad_optimized_tf32_256x128_16x3_nhwc_align4(Manifest &manifest);


//
// Entry point to construct operations
//
void initialize_all_sm80_tf32_s1688wgrad_optimized_tf32_conv2d_operations(Manifest &manifest) {
  initialize_cutlass_tensorop_tf32_s1688wgrad_optimized_tf32_256x128_16x3_nhwc_align4(manifest);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

