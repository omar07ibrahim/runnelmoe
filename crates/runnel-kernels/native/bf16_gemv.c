#include "runnel_kernels.h"

#include <immintrin.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if !defined(__x86_64__) || !defined(__GNUC__)
#error "the native AVX2 kernel requires a GNU-compatible x86_64 compiler"
#endif

struct runnel_byte_range {
    uintptr_t start;
    uintptr_t end;
};

static bool runnel_checked_multiply(
    size_t left,
    size_t right,
    size_t *result) {
    if (left != 0U && right > SIZE_MAX / left) {
        return false;
    }
    *result = left * right;
    return true;
}

static bool runnel_checked_range(
    const void *pointer,
    size_t byte_length,
    struct runnel_byte_range *range) {
    const uintptr_t start = (uintptr_t)pointer;
    if (byte_length > UINTPTR_MAX || start > UINTPTR_MAX - byte_length) {
        return false;
    }
    range->start = start;
    range->end = start + byte_length;
    return true;
}

static bool runnel_ranges_overlap(
    const struct runnel_byte_range *left,
    const struct runnel_byte_range *right) {
    return left->start < right->end && right->start < left->end;
}

static float runnel_widen_bf16(uint16_t value) {
    const uint32_t bits = (uint32_t)value << 16U;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

/*
 * This function is the only code emitted with AVX2 enabled. The public entry
 * point remains baseline-safe and calls it only after a runtime capability
 * check. Explicit intrinsics provide the vector body; the tail is written as
 * a dependency chain of scalar operations so no masked or overreading load is
 * needed. AVX2 does not imply FMA, and the build additionally disables FP
 * contraction.
 */
__attribute__((target("avx2"), noinline))
static void runnel_bf16_gemv_avx2(
    const uint16_t *weights,
    const float *input,
    float *output,
    size_t rows,
    size_t columns) {
    const size_t vector_columns = columns - (columns % 8U);

    for (size_t row = 0U; row < rows; ++row) {
        const uint16_t *const row_weights = weights + row * columns;
        __m256 accumulator = _mm256_setzero_ps();

        size_t column = 0U;
        for (; column < vector_columns; column += 8U) {
            const __m128i packed =
                _mm_loadu_si128((const __m128i *)(const void *)(row_weights + column));
            __m256i widened_bits = _mm256_cvtepu16_epi32(packed);
            widened_bits = _mm256_slli_epi32(widened_bits, 16);
            const __m256 widened_weights = _mm256_castsi256_ps(widened_bits);
            const __m256 input_values = _mm256_loadu_ps(input + column);
            const __m256 products = _mm256_mul_ps(widened_weights, input_values);
            accumulator = _mm256_add_ps(accumulator, products);
        }

        float lanes[8];
        _mm256_storeu_ps(lanes, accumulator);
        float sum = lanes[0];
        sum += lanes[1];
        sum += lanes[2];
        sum += lanes[3];
        sum += lanes[4];
        sum += lanes[5];
        sum += lanes[6];
        sum += lanes[7];

        const size_t remaining = columns - column;
        if (remaining >= 1U) {
            sum += runnel_widen_bf16(row_weights[column]) * input[column];
        }
        if (remaining >= 2U) {
            sum += runnel_widen_bf16(row_weights[column + 1U]) * input[column + 1U];
        }
        if (remaining >= 3U) {
            sum += runnel_widen_bf16(row_weights[column + 2U]) * input[column + 2U];
        }
        if (remaining >= 4U) {
            sum += runnel_widen_bf16(row_weights[column + 3U]) * input[column + 3U];
        }
        if (remaining >= 5U) {
            sum += runnel_widen_bf16(row_weights[column + 4U]) * input[column + 4U];
        }
        if (remaining >= 6U) {
            sum += runnel_widen_bf16(row_weights[column + 5U]) * input[column + 5U];
        }
        if (remaining >= 7U) {
            sum += runnel_widen_bf16(row_weights[column + 6U]) * input[column + 6U];
        }

        output[row] = sum;
    }

    _mm256_zeroupper();
}

int32_t runnel_bf16_gemv_avx2_v1(
    const void *weights,
    size_t weight_bytes,
    const void *input,
    size_t input_bytes,
    void *output,
    size_t output_bytes,
    size_t rows,
    size_t columns) {
    if (weights == NULL || input == NULL || output == NULL) {
        return RUNNEL_KERNEL_STATUS_NULL_POINTER;
    }
    if (rows == 0U || columns == 0U) {
        return RUNNEL_KERNEL_STATUS_INVALID_DIMENSION;
    }

    size_t element_count;
    size_t expected_weight_bytes;
    size_t expected_input_bytes;
    size_t expected_output_bytes;
    if (!runnel_checked_multiply(rows, columns, &element_count) ||
        !runnel_checked_multiply(element_count, sizeof(uint16_t), &expected_weight_bytes) ||
        !runnel_checked_multiply(columns, sizeof(float), &expected_input_bytes) ||
        !runnel_checked_multiply(rows, sizeof(float), &expected_output_bytes)) {
        return RUNNEL_KERNEL_STATUS_SIZE_OVERFLOW;
    }

    if (weight_bytes != expected_weight_bytes || input_bytes != expected_input_bytes ||
        output_bytes != expected_output_bytes) {
        return RUNNEL_KERNEL_STATUS_LENGTH_MISMATCH;
    }

    const uintptr_t weight_address = (uintptr_t)weights;
    const uintptr_t input_address = (uintptr_t)input;
    const uintptr_t output_address = (uintptr_t)output;
    if (weight_address % _Alignof(uint16_t) != 0U ||
        input_address % _Alignof(float) != 0U ||
        output_address % _Alignof(float) != 0U) {
        return RUNNEL_KERNEL_STATUS_ALIGNMENT_FAILURE;
    }

    struct runnel_byte_range weight_range;
    struct runnel_byte_range input_range;
    struct runnel_byte_range output_range;
    if (!runnel_checked_range(weights, weight_bytes, &weight_range) ||
        !runnel_checked_range(input, input_bytes, &input_range) ||
        !runnel_checked_range(output, output_bytes, &output_range)) {
        return RUNNEL_KERNEL_STATUS_ADDRESS_OVERFLOW;
    }

    if (runnel_ranges_overlap(&weight_range, &input_range) ||
        runnel_ranges_overlap(&weight_range, &output_range) ||
        runnel_ranges_overlap(&input_range, &output_range)) {
        return RUNNEL_KERNEL_STATUS_RANGE_OVERLAP;
    }

    if (__builtin_cpu_supports("avx2") == 0) {
        return RUNNEL_KERNEL_STATUS_UNAVAILABLE_ISA;
    }

    runnel_bf16_gemv_avx2(
        (const uint16_t *)weights,
        (const float *)input,
        (float *)output,
        rows,
        columns);
    return RUNNEL_KERNEL_STATUS_OK;
}
