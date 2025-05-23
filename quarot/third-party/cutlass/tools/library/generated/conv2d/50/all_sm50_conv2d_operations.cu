
/*
 Generated by manifest.py - Do not edit.
*/

#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_all_sm50_cf32_cdgrad_optimized_cf32_conv2d_operations(Manifest &manifest);
void initialize_all_sm50_cf32_cfprop_optimized_cf32_conv2d_operations(Manifest &manifest);
void initialize_all_sm50_cf32_cwgrad_optimized_cf32_conv2d_operations(Manifest &manifest);
void initialize_all_sm50_sdgrad_optimized_conv2d_operations(Manifest &manifest);
void initialize_all_sm50_sfprop_optimized_conv2d_operations(Manifest &manifest);
void initialize_all_sm50_swgrad_optimized_conv2d_operations(Manifest &manifest);


//
// Entry point to construct operations
//
void initialize_all_sm50__conv2d_operations(Manifest &manifest) {
  initialize_all_sm50_cf32_cdgrad_optimized_cf32_conv2d_operations(manifest);
  initialize_all_sm50_cf32_cfprop_optimized_cf32_conv2d_operations(manifest);
  initialize_all_sm50_cf32_cwgrad_optimized_cf32_conv2d_operations(manifest);
  initialize_all_sm50_sdgrad_optimized_conv2d_operations(manifest);
  initialize_all_sm50_sfprop_optimized_conv2d_operations(manifest);
  initialize_all_sm50_swgrad_optimized_conv2d_operations(manifest);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

