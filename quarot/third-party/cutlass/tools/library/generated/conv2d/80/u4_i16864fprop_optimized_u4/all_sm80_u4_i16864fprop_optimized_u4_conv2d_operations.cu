
/*
 Generated by manifest.py - Do not edit.
*/

#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_cutlass_tensorop_u4_i16864fprop_optimized_u4_256x128_128x3_nhwc_align32(Manifest &manifest);
void initialize_cutlass_tensorop_u4_i16864fprop_optimized_u4_256x128_128x3_nhwc_single_group_align32(Manifest &manifest);
void initialize_cutlass_tensorop_u4_i16864fprop_optimized_u4_256x128_128x3_nc64hw64_align32(Manifest &manifest);


//
// Entry point to construct operations
//
void initialize_all_sm80_u4_i16864fprop_optimized_u4_conv2d_operations(Manifest &manifest) {
  initialize_cutlass_tensorop_u4_i16864fprop_optimized_u4_256x128_128x3_nhwc_align32(manifest);
  initialize_cutlass_tensorop_u4_i16864fprop_optimized_u4_256x128_128x3_nhwc_single_group_align32(manifest);
  initialize_cutlass_tensorop_u4_i16864fprop_optimized_u4_256x128_128x3_nc64hw64_align32(manifest);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

