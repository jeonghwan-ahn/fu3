
/*
 Generated by manifest.py - Do not edit.
*/

#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_all_sm50_cgemm_gemm_operations(Manifest &manifest);
void initialize_all_sm50_dgemm_gemm_operations(Manifest &manifest);
void initialize_all_sm50_sgemm_gemm_operations(Manifest &manifest);


//
// Entry point to construct operations
//
void initialize_all_sm50__gemm_operations(Manifest &manifest) {
  initialize_all_sm50_cgemm_gemm_operations(manifest);
  initialize_all_sm50_dgemm_gemm_operations(manifest);
  initialize_all_sm50_sgemm_gemm_operations(manifest);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

