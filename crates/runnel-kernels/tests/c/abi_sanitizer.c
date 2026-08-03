#include "runnel_kernels.h"

#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RUNNEL_TEST_MAX_COLUMNS 33U
#define RUNNEL_TEST_ROWS 3U
#define RUNNEL_TEST_WEIGHT_OFFSETS 16U
#define RUNNEL_TEST_FLOAT_OFFSETS 8U
#define RUNNEL_TEST_THREADS 4U
#define RUNNEL_TEST_CONCURRENT_ROWS 17U
#define RUNNEL_TEST_CONCURRENT_COLUMNS 33U
#define RUNNEL_TEST_CONCURRENT_REPETITIONS 500U

struct validation_fixture {
    _Alignas(32) uint16_t weights[8];
    _Alignas(32) float input[8];
    _Alignas(32) float output[8];
};

struct concurrent_case {
    const uint16_t *weights;
    const float *input;
    const float *expected;
    float output[RUNNEL_TEST_CONCURRENT_ROWS];
    int failure;
};

static float widen_bf16(uint16_t value) {
    const uint32_t bits = (uint32_t)value << 16U;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static float reference_row(
    const uint16_t *weights,
    const float *input,
    size_t columns) {
    float result = 0.0F;
    for (size_t column = 0U; column < columns; ++column) {
        result += widen_bf16(weights[column]) * input[column];
    }
    return result;
}

static bool nearly_equal(float left, float right) {
    const float scale = 1.0F + fabsf(right);
    return fabsf(left - right) <= 0.00002F * scale;
}

static void initialize_validation_fixture(struct validation_fixture *fixture) {
    static const uint16_t weight_values[8] = {
        UINT16_C(0x3f80), UINT16_C(0xbf80), UINT16_C(0x4000), UINT16_C(0xc000),
        UINT16_C(0x3f00), UINT16_C(0xbf00), UINT16_C(0x3e80), UINT16_C(0xbe80)};
    static const float input_values[8] = {
        2.0F, -1.0F, 0.5F, -0.25F, 4.0F, -2.0F, 0.125F, -0.5F};

    memcpy(fixture->weights, weight_values, sizeof(weight_values));
    memcpy(fixture->input, input_values, sizeof(input_values));
    for (size_t index = 0U; index < 8U; ++index) {
        fixture->output[index] = -12345.0F - (float)index;
    }
}

static int expect_status_and_unchanged(
    struct validation_fixture *fixture,
    const void *weights,
    size_t weight_bytes,
    const void *input,
    size_t input_bytes,
    void *output,
    size_t output_bytes,
    size_t rows,
    size_t columns,
    int32_t expected_status,
    const char *name) {
    float output_snapshot[8];
    memcpy(output_snapshot, fixture->output, sizeof(output_snapshot));

    const int32_t actual_status = runnel_bf16_gemv_avx2_v1(
        weights,
        weight_bytes,
        input,
        input_bytes,
        output,
        output_bytes,
        rows,
        columns);
    if (actual_status != expected_status) {
        (void)fprintf(
            stderr,
            "%s: expected status %d, received %d\n",
            name,
            (int)expected_status,
            (int)actual_status);
        return 1;
    }
    if (memcmp(output_snapshot, fixture->output, sizeof(output_snapshot)) != 0) {
        (void)fprintf(stderr, "%s: rejected call changed output storage\n", name);
        return 1;
    }
    return 0;
}

static int test_validation(void) {
    struct validation_fixture fixture;
    initialize_validation_fixture(&fixture);
    int failures = 0;

    failures += expect_status_and_unchanged(
        &fixture,
        NULL,
        sizeof(uint16_t),
        fixture.input,
        sizeof(float),
        fixture.output,
        sizeof(float),
        1U,
        1U,
        RUNNEL_KERNEL_STATUS_NULL_POINTER,
        "null weights");
    failures += expect_status_and_unchanged(
        &fixture,
        fixture.weights,
        sizeof(uint16_t),
        NULL,
        sizeof(float),
        fixture.output,
        sizeof(float),
        1U,
        1U,
        RUNNEL_KERNEL_STATUS_NULL_POINTER,
        "null input");
    failures += expect_status_and_unchanged(
        &fixture,
        fixture.weights,
        sizeof(uint16_t),
        fixture.input,
        sizeof(float),
        NULL,
        sizeof(float),
        1U,
        1U,
        RUNNEL_KERNEL_STATUS_NULL_POINTER,
        "null output");
    failures += expect_status_and_unchanged(
        &fixture,
        NULL,
        0U,
        NULL,
        0U,
        NULL,
        0U,
        0U,
        0U,
        RUNNEL_KERNEL_STATUS_NULL_POINTER,
        "null precedes zero dimensions");

    failures += expect_status_and_unchanged(
        &fixture,
        fixture.weights,
        0U,
        fixture.input,
        sizeof(float),
        fixture.output,
        0U,
        0U,
        1U,
        RUNNEL_KERNEL_STATUS_INVALID_DIMENSION,
        "zero rows");
    failures += expect_status_and_unchanged(
        &fixture,
        fixture.weights,
        0U,
        fixture.input,
        0U,
        fixture.output,
        sizeof(float),
        1U,
        0U,
        RUNNEL_KERNEL_STATUS_INVALID_DIMENSION,
        "zero columns");
    failures += expect_status_and_unchanged(
        &fixture,
        fixture.weights,
        0U,
        fixture.input,
        0U,
        fixture.output,
        0U,
        0U,
        SIZE_MAX,
        RUNNEL_KERNEL_STATUS_INVALID_DIMENSION,
        "zero dimension precedes size overflow");

    failures += expect_status_and_unchanged(
        &fixture,
        fixture.weights,
        0U,
        fixture.input,
        0U,
        fixture.output,
        0U,
        SIZE_MAX,
        2U,
        RUNNEL_KERNEL_STATUS_SIZE_OVERFLOW,
        "weight element-count overflow");
    failures += expect_status_and_unchanged(
        &fixture,
        fixture.weights,
        0U,
        fixture.input,
        0U,
        fixture.output,
        0U,
        1U,
        SIZE_MAX / sizeof(float) + 1U,
        RUNNEL_KERNEL_STATUS_SIZE_OVERFLOW,
        "input byte-count overflow");
    failures += expect_status_and_unchanged(
        &fixture,
        fixture.weights,
        0U,
        fixture.input,
        0U,
        fixture.output,
        0U,
        SIZE_MAX / sizeof(float) + 1U,
        1U,
        RUNNEL_KERNEL_STATUS_SIZE_OVERFLOW,
        "output byte-count overflow");

    failures += expect_status_and_unchanged(
        &fixture,
        fixture.weights,
        0U,
        fixture.input,
        sizeof(float),
        fixture.output,
        sizeof(float),
        1U,
        1U,
        RUNNEL_KERNEL_STATUS_LENGTH_MISMATCH,
        "weight length mismatch");
    failures += expect_status_and_unchanged(
        &fixture,
        fixture.weights,
        sizeof(uint16_t),
        fixture.input,
        0U,
        fixture.output,
        sizeof(float),
        1U,
        1U,
        RUNNEL_KERNEL_STATUS_LENGTH_MISMATCH,
        "input length mismatch");
    failures += expect_status_and_unchanged(
        &fixture,
        fixture.weights,
        sizeof(uint16_t),
        fixture.input,
        sizeof(float),
        fixture.output,
        0U,
        1U,
        1U,
        RUNNEL_KERNEL_STATUS_LENGTH_MISMATCH,
        "output length mismatch");
    failures += expect_status_and_unchanged(
        &fixture,
        (const unsigned char *)(const void *)fixture.weights + 1U,
        0U,
        fixture.input,
        sizeof(float),
        fixture.output,
        sizeof(float),
        1U,
        1U,
        RUNNEL_KERNEL_STATUS_LENGTH_MISMATCH,
        "length mismatch precedes alignment");

    failures += expect_status_and_unchanged(
        &fixture,
        (const unsigned char *)(const void *)fixture.weights + 1U,
        sizeof(uint16_t),
        fixture.input,
        sizeof(float),
        fixture.output,
        sizeof(float),
        1U,
        1U,
        RUNNEL_KERNEL_STATUS_ALIGNMENT_FAILURE,
        "misaligned weights");
    failures += expect_status_and_unchanged(
        &fixture,
        fixture.weights,
        sizeof(uint16_t),
        (const unsigned char *)(const void *)fixture.input + 1U,
        sizeof(float),
        fixture.output,
        sizeof(float),
        1U,
        1U,
        RUNNEL_KERNEL_STATUS_ALIGNMENT_FAILURE,
        "misaligned input");
    failures += expect_status_and_unchanged(
        &fixture,
        fixture.weights,
        sizeof(uint16_t),
        fixture.input,
        sizeof(float),
        (unsigned char *)(void *)fixture.output + 1U,
        sizeof(float),
        1U,
        1U,
        RUNNEL_KERNEL_STATUS_ALIGNMENT_FAILURE,
        "misaligned output");
    failures += expect_status_and_unchanged(
        &fixture,
        (const void *)(uintptr_t)UINTPTR_MAX,
        sizeof(uint16_t),
        fixture.input,
        sizeof(float),
        fixture.output,
        sizeof(float),
        1U,
        1U,
        RUNNEL_KERNEL_STATUS_ALIGNMENT_FAILURE,
        "alignment precedes address overflow");

    failures += expect_status_and_unchanged(
        &fixture,
        (const void *)(uintptr_t)(UINTPTR_MAX - 1U),
        sizeof(uint16_t),
        fixture.input,
        sizeof(float),
        fixture.output,
        sizeof(float),
        1U,
        1U,
        RUNNEL_KERNEL_STATUS_ADDRESS_OVERFLOW,
        "weight address overflow");
    failures += expect_status_and_unchanged(
        &fixture,
        fixture.weights,
        sizeof(uint16_t),
        (const void *)(uintptr_t)(UINTPTR_MAX - 3U),
        sizeof(float),
        fixture.output,
        sizeof(float),
        1U,
        1U,
        RUNNEL_KERNEL_STATUS_ADDRESS_OVERFLOW,
        "input address overflow");
    failures += expect_status_and_unchanged(
        &fixture,
        fixture.weights,
        sizeof(uint16_t),
        fixture.input,
        sizeof(float),
        (void *)(uintptr_t)(UINTPTR_MAX - 3U),
        sizeof(float),
        1U,
        1U,
        RUNNEL_KERNEL_STATUS_ADDRESS_OVERFLOW,
        "output address overflow");

    _Alignas(32) unsigned char overlap_storage[32];
    unsigned char overlap_snapshot[32];
    memset(overlap_storage, 0xA5, sizeof(overlap_storage));
    failures += expect_status_and_unchanged(
        &fixture,
        overlap_storage,
        sizeof(uint16_t),
        overlap_storage,
        sizeof(float),
        fixture.output,
        sizeof(float),
        1U,
        1U,
        RUNNEL_KERNEL_STATUS_RANGE_OVERLAP,
        "weights overlap input");

    memcpy(overlap_snapshot, overlap_storage, sizeof(overlap_snapshot));
    failures += expect_status_and_unchanged(
        &fixture,
        overlap_storage,
        sizeof(uint16_t),
        fixture.input,
        sizeof(float),
        overlap_storage,
        sizeof(float),
        1U,
        1U,
        RUNNEL_KERNEL_STATUS_RANGE_OVERLAP,
        "weights overlap output");
    if (memcmp(overlap_snapshot, overlap_storage, sizeof(overlap_snapshot)) != 0) {
        (void)fprintf(stderr, "weights-overlap-output rejection changed declared output\n");
        ++failures;
    }

    memset(overlap_storage, 0x5A, sizeof(overlap_storage));
    memcpy(overlap_snapshot, overlap_storage, sizeof(overlap_snapshot));
    failures += expect_status_and_unchanged(
        &fixture,
        fixture.weights,
        sizeof(uint16_t),
        overlap_storage,
        sizeof(float),
        overlap_storage,
        sizeof(float),
        1U,
        1U,
        RUNNEL_KERNEL_STATUS_RANGE_OVERLAP,
        "input overlaps output");
    if (memcmp(overlap_snapshot, overlap_storage, sizeof(overlap_snapshot)) != 0) {
        (void)fprintf(stderr, "input-overlap-output rejection changed declared output\n");
        ++failures;
    }

    return failures;
}

static int test_valid_probe(bool *avx2_available) {
    struct validation_fixture fixture;
    initialize_validation_fixture(&fixture);
    float output_before[8];
    memcpy(output_before, fixture.output, sizeof(output_before));

    const int32_t status = runnel_bf16_gemv_avx2_v1(
        fixture.weights,
        sizeof(uint16_t),
        fixture.input,
        sizeof(float),
        fixture.output,
        sizeof(float),
        1U,
        1U);
    if (status == RUNNEL_KERNEL_STATUS_UNAVAILABLE_ISA) {
        *avx2_available = false;
        if (memcmp(output_before, fixture.output, sizeof(output_before)) != 0) {
            (void)fprintf(stderr, "unavailable ISA changed output storage\n");
            return 1;
        }
        return 0;
    }
    if (status != RUNNEL_KERNEL_STATUS_OK) {
        (void)fprintf(stderr, "valid probe returned status %d\n", (int)status);
        return 1;
    }
    *avx2_available = true;
    if (fixture.output[0] != 2.0F ||
        memcmp(&output_before[1], &fixture.output[1], 7U * sizeof(float)) != 0) {
        (void)fprintf(stderr, "valid probe produced an invalid value or overwrite\n");
        return 1;
    }
    return 0;
}

static int test_tails_offsets_and_canaries(void) {
    enum {
        WEIGHT_STORAGE_COUNT =
            RUNNEL_TEST_WEIGHT_OFFSETS + RUNNEL_TEST_ROWS * RUNNEL_TEST_MAX_COLUMNS + 8U,
        INPUT_STORAGE_COUNT = RUNNEL_TEST_FLOAT_OFFSETS + RUNNEL_TEST_MAX_COLUMNS + 8U,
        OUTPUT_PREFIX_COUNT = 8U,
        OUTPUT_STORAGE_COUNT =
            OUTPUT_PREFIX_COUNT + RUNNEL_TEST_FLOAT_OFFSETS + RUNNEL_TEST_ROWS + 8U
    };
    _Alignas(64) uint16_t weight_storage[WEIGHT_STORAGE_COUNT];
    _Alignas(64) uint16_t weight_snapshot[WEIGHT_STORAGE_COUNT];
    _Alignas(64) float input_storage[INPUT_STORAGE_COUNT];
    _Alignas(64) float input_snapshot[INPUT_STORAGE_COUNT];
    _Alignas(64) float output_storage[OUTPUT_STORAGE_COUNT];
    _Alignas(64) float output_snapshot[OUTPUT_STORAGE_COUNT];
    static const uint16_t weight_values[8] = {
        UINT16_C(0x3f80), UINT16_C(0xbf80), UINT16_C(0x4000), UINT16_C(0xc000),
        UINT16_C(0x3f00), UINT16_C(0xbf00), UINT16_C(0x3e80), UINT16_C(0xbe80)};
    static const float input_values[8] = {
        1.0F, -2.0F, 0.5F, -0.25F, 4.0F, -1.0F, 0.125F, -0.5F};

    for (size_t columns = 1U; columns <= RUNNEL_TEST_MAX_COLUMNS; ++columns) {
        for (size_t weight_offset = 0U; weight_offset < RUNNEL_TEST_WEIGHT_OFFSETS;
             ++weight_offset) {
            for (size_t input_offset = 0U; input_offset < RUNNEL_TEST_FLOAT_OFFSETS;
                 ++input_offset) {
                for (size_t output_offset = 0U; output_offset < RUNNEL_TEST_FLOAT_OFFSETS;
                     ++output_offset) {
                    for (size_t index = 0U; index < WEIGHT_STORAGE_COUNT; ++index) {
                        weight_storage[index] = UINT16_C(0x3555);
                    }
                    for (size_t index = 0U; index < INPUT_STORAGE_COUNT; ++index) {
                        input_storage[index] = 9999.0F + (float)index;
                    }
                    for (size_t index = 0U; index < OUTPUT_STORAGE_COUNT; ++index) {
                        output_storage[index] = -7777.0F - (float)index;
                    }

                    uint16_t *const weights = weight_storage + weight_offset;
                    float *const input = input_storage + input_offset;
                    const size_t output_start = OUTPUT_PREFIX_COUNT + output_offset;
                    float *const output = output_storage + output_start;
                    for (size_t row = 0U; row < RUNNEL_TEST_ROWS; ++row) {
                        for (size_t column = 0U; column < columns; ++column) {
                            const size_t value_index = (row * 3U + column) % 8U;
                            weights[row * columns + column] = weight_values[value_index];
                        }
                    }
                    for (size_t column = 0U; column < columns; ++column) {
                        input[column] = input_values[column % 8U];
                    }

                    if ((uintptr_t)weights % 32U != weight_offset * sizeof(uint16_t) ||
                        (uintptr_t)input % 32U != input_offset * sizeof(float) ||
                        (uintptr_t)output % 32U != output_offset * sizeof(float)) {
                        (void)fprintf(stderr, "test allocation did not realize requested offsets\n");
                        return 1;
                    }

                    memcpy(weight_snapshot, weight_storage, sizeof(weight_storage));
                    memcpy(input_snapshot, input_storage, sizeof(input_storage));
                    memcpy(output_snapshot, output_storage, sizeof(output_storage));

                    const int32_t status = runnel_bf16_gemv_avx2_v1(
                        weights,
                        RUNNEL_TEST_ROWS * columns * sizeof(uint16_t),
                        input,
                        columns * sizeof(float),
                        output,
                        RUNNEL_TEST_ROWS * sizeof(float),
                        RUNNEL_TEST_ROWS,
                        columns);
                    if (status != RUNNEL_KERNEL_STATUS_OK) {
                        (void)fprintf(
                            stderr,
                            "valid offset call failed: columns=%zu weight=%zu input=%zu output=%zu "
                            "status=%d\n",
                            columns,
                            weight_offset,
                            input_offset,
                            output_offset,
                            (int)status);
                        return 1;
                    }
                    if (memcmp(weight_snapshot, weight_storage, sizeof(weight_storage)) != 0 ||
                        memcmp(input_snapshot, input_storage, sizeof(input_storage)) != 0) {
                        (void)fprintf(stderr, "kernel changed immutable input storage\n");
                        return 1;
                    }
                    for (size_t index = 0U; index < OUTPUT_STORAGE_COUNT; ++index) {
                        const bool is_output =
                            index >= output_start && index < output_start + RUNNEL_TEST_ROWS;
                        if (!is_output &&
                            memcmp(&output_storage[index], &output_snapshot[index], sizeof(float)) !=
                                0) {
                            (void)fprintf(stderr, "output canary changed at index %zu\n", index);
                            return 1;
                        }
                    }
                    for (size_t row = 0U; row < RUNNEL_TEST_ROWS; ++row) {
                        const float expected = reference_row(weights + row * columns, input, columns);
                        if (!nearly_equal(output[row], expected)) {
                            (void)fprintf(
                                stderr,
                                "numeric mismatch: columns=%zu row=%zu got=%g expected=%g\n",
                                columns,
                                row,
                                (double)output[row],
                                (double)expected);
                            return 1;
                        }
                    }
                }
            }
        }
    }
    return 0;
}

static void *run_concurrent_case(void *opaque) {
    struct concurrent_case *const test_case = (struct concurrent_case *)opaque;
    test_case->failure = 0;
    for (size_t iteration = 0U; iteration < RUNNEL_TEST_CONCURRENT_REPETITIONS; ++iteration) {
        for (size_t row = 0U; row < RUNNEL_TEST_CONCURRENT_ROWS; ++row) {
            test_case->output[row] = -9999.0F;
        }
        const int32_t status = runnel_bf16_gemv_avx2_v1(
            test_case->weights,
            RUNNEL_TEST_CONCURRENT_ROWS * RUNNEL_TEST_CONCURRENT_COLUMNS * sizeof(uint16_t),
            test_case->input,
            RUNNEL_TEST_CONCURRENT_COLUMNS * sizeof(float),
            test_case->output,
            RUNNEL_TEST_CONCURRENT_ROWS * sizeof(float),
            RUNNEL_TEST_CONCURRENT_ROWS,
            RUNNEL_TEST_CONCURRENT_COLUMNS);
        if (status != RUNNEL_KERNEL_STATUS_OK) {
            test_case->failure = 1;
            return NULL;
        }
        for (size_t row = 0U; row < RUNNEL_TEST_CONCURRENT_ROWS; ++row) {
            if (!nearly_equal(test_case->output[row], test_case->expected[row])) {
                test_case->failure = 1;
                return NULL;
            }
        }
    }
    return NULL;
}

static int test_concurrency(void) {
    _Alignas(64) uint16_t
        weights[RUNNEL_TEST_CONCURRENT_ROWS * RUNNEL_TEST_CONCURRENT_COLUMNS];
    _Alignas(64) float input[RUNNEL_TEST_CONCURRENT_COLUMNS];
    float expected[RUNNEL_TEST_CONCURRENT_ROWS];
    static const uint16_t weight_values[8] = {
        UINT16_C(0x3f80), UINT16_C(0xbf80), UINT16_C(0x4000), UINT16_C(0xc000),
        UINT16_C(0x3f00), UINT16_C(0xbf00), UINT16_C(0x3e80), UINT16_C(0xbe80)};

    for (size_t row = 0U; row < RUNNEL_TEST_CONCURRENT_ROWS; ++row) {
        for (size_t column = 0U; column < RUNNEL_TEST_CONCURRENT_COLUMNS; ++column) {
            weights[row * RUNNEL_TEST_CONCURRENT_COLUMNS + column] =
                weight_values[(row + column * 3U) % 8U];
        }
    }
    for (size_t column = 0U; column < RUNNEL_TEST_CONCURRENT_COLUMNS; ++column) {
        input[column] = (float)((int)(column % 9U) - 4) * 0.25F;
    }
    for (size_t row = 0U; row < RUNNEL_TEST_CONCURRENT_ROWS; ++row) {
        expected[row] = reference_row(
            weights + row * RUNNEL_TEST_CONCURRENT_COLUMNS,
            input,
            RUNNEL_TEST_CONCURRENT_COLUMNS);
    }

    pthread_t threads[RUNNEL_TEST_THREADS];
    struct concurrent_case cases[RUNNEL_TEST_THREADS];
    size_t created = 0U;
    for (size_t index = 0U; index < RUNNEL_TEST_THREADS; ++index) {
        cases[index].weights = weights;
        cases[index].input = input;
        cases[index].expected = expected;
        cases[index].failure = 1;
        const int result = pthread_create(&threads[index], NULL, run_concurrent_case, &cases[index]);
        if (result != 0) {
            (void)fprintf(stderr, "pthread_create failed: %s\n", strerror(result));
            break;
        }
        ++created;
    }

    int failure = created == RUNNEL_TEST_THREADS ? 0 : 1;
    for (size_t index = 0U; index < created; ++index) {
        const int result = pthread_join(threads[index], NULL);
        if (result != 0) {
            (void)fprintf(stderr, "pthread_join failed: %s\n", strerror(result));
            failure = 1;
        }
        if (cases[index].failure != 0) {
            (void)fprintf(stderr, "concurrent kernel call failed for worker %zu\n", index);
            failure = 1;
        }
    }
    return failure;
}

int main(void) {
    int failures = test_validation();
    bool avx2_available = false;
    failures += test_valid_probe(&avx2_available);
    if (failures != 0) {
        return EXIT_FAILURE;
    }
    if (!avx2_available) {
        (void)puts("runnel native ABI validation passed; AVX2 compute tests skipped");
        return EXIT_SUCCESS;
    }

    failures += test_tails_offsets_and_canaries();
    failures += test_concurrency();
    if (failures != 0) {
        return EXIT_FAILURE;
    }
    (void)puts("runnel native ABI sanitizer harness passed");
    return EXIT_SUCCESS;
}
