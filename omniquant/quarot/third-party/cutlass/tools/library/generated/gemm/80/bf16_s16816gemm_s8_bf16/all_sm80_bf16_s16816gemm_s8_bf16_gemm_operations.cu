
/*
 Generated by manifest.py - Do not edit.
*/

#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_cutlass_tensorop_bf16_s16816gemm_s8_bf16_128x128_64x4_tn_align16(Manifest &manifest);


//
// Entry point to construct operations
//
void initialize_all_sm80_bf16_s16816gemm_s8_bf16_gemm_operations(Manifest &manifest) {
  initialize_cutlass_tensorop_bf16_s16816gemm_s8_bf16_128x128_64x4_tn_align16(manifest);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

