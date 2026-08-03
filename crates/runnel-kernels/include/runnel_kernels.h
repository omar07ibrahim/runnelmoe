#ifndef RUNNEL_KERNELS_H
#define RUNNEL_KERNELS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Stable status values for the versioned native ABI. */
enum runnel_kernel_status {
    RUNNEL_KERNEL_STATUS_OK = 0,
    RUNNEL_KERNEL_STATUS_NULL_POINTER = 1,
    RUNNEL_KERNEL_STATUS_INVALID_DIMENSION = 2,
    RUNNEL_KERNEL_STATUS_SIZE_OVERFLOW = 3,
    RUNNEL_KERNEL_STATUS_LENGTH_MISMATCH = 4,
    RUNNEL_KERNEL_STATUS_ALIGNMENT_FAILURE = 5,
    RUNNEL_KERNEL_STATUS_ADDRESS_OVERFLOW = 6,
    RUNNEL_KERNEL_STATUS_RANGE_OVERLAP = 7,
    RUNNEL_KERNEL_STATUS_UNAVAILABLE_ISA = 8
};

/*
 * Compute output[row] = sum(widen_bf16(weights[row, column]) * input[column]).
 *
 * The three byte lengths must exactly match rows and columns. All ranges must
 * be naturally aligned and pairwise disjoint. Each declared extent must name
 * live storage: readable for weights/input and writable for output. The caller
 * owns that storage and must keep weights/input immutable and output
 * exclusively writable for the complete call. Every nonzero status, including
 * unavailable ISA, leaves output unchanged.
 */
int32_t runnel_bf16_gemv_avx2_v1(
    const void *weights,
    size_t weight_bytes,
    const void *input,
    size_t input_bytes,
    void *output,
    size_t output_bytes,
    size_t rows,
    size_t columns);

#ifdef __cplusplus
}
#endif

#endif /* RUNNEL_KERNELS_H */
