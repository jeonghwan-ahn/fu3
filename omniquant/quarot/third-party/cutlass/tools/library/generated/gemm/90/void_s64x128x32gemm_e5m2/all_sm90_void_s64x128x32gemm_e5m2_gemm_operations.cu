
/*
 Generated by manifest.py - Do not edit.
*/

#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_cutlass3x_sm90_tensorop_s64x128x32gemm_e5m2_e5m2_f32_void_e5m2_256x128x128_1x2x1_0_tnn_align16_warpspecialized_cooperative_epi_tma(Manifest &manifest);
void initialize_cutlass3x_sm90_tensorop_s64x128x32gemm_e5m2_e5m2_f32_void_e5m2_256x128x128_1x2x1_0_tnn_align16_warpspecialized_cooperative_fp8_fastaccum_epi_tma(Manifest &manifest);
void initialize_cutlass3x_sm90_tensorop_s64x128x32gemm_e5m2_e5m2_f32_void_e4m3_256x128x128_1x2x1_0_tnn_align16_warpspecialized_cooperative_epi_tma(Manifest &manifest);
void initialize_cutlass3x_sm90_tensorop_s64x128x32gemm_e5m2_e5m2_f32_void_e4m3_256x128x128_1x2x1_0_tnn_align16_warpspecialized_cooperative_fp8_fastaccum_epi_tma(Manifest &manifest);


//
// Entry point to construct operations
//
void initialize_all_sm90_void_s64x128x32gemm_e5m2_gemm_operations(Manifest &manifest) {
  initialize_cutlass3x_sm90_tensorop_s64x128x32gemm_e5m2_e5m2_f32_void_e5m2_256x128x128_1x2x1_0_tnn_align16_warpspecialized_cooperative_epi_tma(manifest);
  initialize_cutlass3x_sm90_tensorop_s64x128x32gemm_e5m2_e5m2_f32_void_e5m2_256x128x128_1x2x1_0_tnn_align16_warpspecialized_cooperative_fp8_fastaccum_epi_tma(manifest);
  initialize_cutlass3x_sm90_tensorop_s64x128x32gemm_e5m2_e5m2_f32_void_e4m3_256x128x128_1x2x1_0_tnn_align16_warpspecialized_cooperative_epi_tma(manifest);
  initialize_cutlass3x_sm90_tensorop_s64x128x32gemm_e5m2_e5m2_f32_void_e4m3_256x128x128_1x2x1_0_tnn_align16_warpspecialized_cooperative_fp8_fastaccum_epi_tma(manifest);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

