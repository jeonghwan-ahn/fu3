
/*
 Generated by manifest.py - Do not edit.
*/

#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_cutlass_tensorop_i8816fprop_optimized_u8_256x128_64x2_nhwc_align16(Manifest &manifest);


//
// Entry point to construct operations
//
void initialize_all_sm75_i8816fprop_optimized_u8_conv2d_operations(Manifest &manifest) {
  initialize_cutlass_tensorop_i8816fprop_optimized_u8_256x128_64x2_nhwc_align16(manifest);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

