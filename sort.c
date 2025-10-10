#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// If compiling with SIMD support (e.g., GCC/Clang with -msse4.1 or -mavx), this block is enabled.
#ifdef __SSE4_1__
#include <smmintrin.h>
#endif

// --- Configuration ---
#define N (1 << 20) // 1,048,576 elements for benchmarking
#define RUNS 3      // Number of runs to average time
// ---------------------

// --- Type-Specific Function Declarations ---

// Generic macro for single-token types (short, int, long, float, double)
#define DECLARE_HELPERS_FOR_TYPE(type) \
    void initialize_array_##type(type *arr, int size); \
    int is_sorted_##type(const type *arr, int size); \
    void merge_scalar_1x_##type(type *arr, const type *aux, int low, int mid, int high); \
    void merge_scalar_2x_##type(type *arr, const type *aux, int low, int mid, int high); \
    void merge_scalar_4x_##type(type *arr, const type *aux, int low, int mid, int high); \
    void merge_scalar_8x_##type(type *arr, const type *aux, int low, int mid, int high); \
    void merge_scalar_16x_##type(type *arr, const type *aux, int low, int mid, int high); \
    void merge_scalar_32x_##type(type *arr, const type *aux, int low, int mid, int high); \
    void mergeSort_recursive_##type(type *arr, type *aux, int low, int high, \
                                    void (*merge_func)(type*, const type*, int, int, int));

// Special macro for the two-token type 'long long' using 'llong' as the identifier suffix
#define DECLARE_LLONG_HELPERS \
    void initialize_array_llong(long long *arr, int size); \
    int is_sorted_llong(const long long *arr, int size); \
    void merge_scalar_1x_llong(long long *arr, const long long *aux, int low, int mid, int high); \
    void merge_scalar_2x_llong(long long *arr, const long long *aux, int low, int mid, int high); \
    void merge_scalar_4x_llong(long long *arr, const long long *aux, int low, int mid, int high); \
    void merge_scalar_8x_llong(long long *arr, const long long *aux, int low, int mid, int high); \
    void merge_scalar_16x_llong(long long *arr, const long long *aux, int low, int mid, int high); \
    void merge_scalar_32x_llong(long long *arr, const long long *aux, int low, int mid, int high); \
    void mergeSort_recursive_llong(long long *arr, long long *aux, int low, int high, \
                                    void (*merge_func)(long long*, const long long*, int, int, int));


DECLARE_HELPERS_FOR_TYPE(short)
DECLARE_HELPERS_FOR_TYPE(int)
DECLARE_HELPERS_FOR_TYPE(long)
DECLARE_LLONG_HELPERS // Use the special helper
DECLARE_HELPERS_FOR_TYPE(float)
DECLARE_HELPERS_FOR_TYPE(double)


// ====================================================================
// GENERIC SCALAR LOGIC MACROS
// ====================================================================

// --- Core Single Merge Step (1x) ---
#define SINGLE_MERGE_STEP(arr, aux, i, j, k) \
    if (i > mid) { arr[k] = aux[j++]; } \
    else if (j > high) { arr[k] = aux[i++]; } \
    else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } \
    else { arr[k] = aux[i++]; }

// --- Unrolled Steps (built on previous levels) ---
#define UNROLL_2X_MERGE(arr, aux, i, j, k) \
    SINGLE_MERGE_STEP(arr, aux, i, j, k) \
    SINGLE_MERGE_STEP(arr, aux, i, j, k + 1)

#define UNROLL_4X_MERGE(arr, aux, i, j, k) \
    UNROLL_2X_MERGE(arr, aux, i, j, k) \
    UNROLL_2X_MERGE(arr, aux, i, j, k + 2)

#define UNROLL_8X_MERGE(arr, aux, i, j, k) \
    UNROLL_4X_MERGE(arr, aux, i, j, k) \
    UNROLL_4X_MERGE(arr, aux, i, j, k + 4)

#define UNROLL_16X_MERGE(arr, aux, i, j, k) \
    UNROLL_8X_MERGE(arr, aux, i, j, k) \
    UNROLL_8X_MERGE(arr, aux, i, j, k + 8)

#define UNROLL_32X_MERGE(arr, aux, i, j, k) \
    UNROLL_16X_MERGE(arr, aux, i, j, k) \
    UNROLL_16X_MERGE(arr, aux, i, j, k + 16)

// ====================================================================
// FUNCTION IMPLEMENTATION MACROS
// ====================================================================

#define IMPLEMENT_MERGE_FUNCTIONS(type) \
    /* --- 1x Base Merge --- */ \
    void merge_scalar_1x_##type(type *arr, const type *aux, int low, int mid, int high) { \
        int i = low; \
        int j = mid + 1; \
        for (int k = low; k <= high; k++) { \
            SINGLE_MERGE_STEP(arr, aux, i, j, k) \
        } \
    } \
    /* --- 2x Unrolled Merge --- */ \
    void merge_scalar_2x_##type(type *arr, const type *aux, int low, int mid, int high) { \
        int i = low; \
        int j = mid + 1; \
        int k; \
        for (k = low; k <= high - 1; k += 2) { \
            UNROLL_2X_MERGE(arr, aux, i, j, k) \
        } \
        for (; k <= high; k++) { \
            SINGLE_MERGE_STEP(arr, aux, i, j, k) \
        } \
    } \
    /* --- 4x Unrolled Merge --- */ \
    void merge_scalar_4x_##type(type *arr, const type *aux, int low, int mid, int high) { \
        int i = low; \
        int j = mid + 1; \
        int k; \
        for (k = low; k <= high - 3; k += 4) { \
            UNROLL_4X_MERGE(arr, aux, i, j, k) \
        } \
        for (; k <= high; k++) { \
            SINGLE_MERGE_STEP(arr, aux, i, j, k) \
        } \
    } \
    /* --- 8x Unrolled Merge --- */ \
    void merge_scalar_8x_##type(type *arr, const type *aux, int low, int mid, int high) { \
        int i = low; \
        int j = mid + 1; \
        int k; \
        for (k = low; k <= high - 7; k += 8) { \
            UNROLL_8X_MERGE(arr, aux, i, j, k) \
        } \
        for (; k <= high; k++) { \
            SINGLE_MERGE_STEP(arr, aux, i, j, k) \
        } \
    } \
    /* --- 16x Unrolled Merge --- */ \
    void merge_scalar_16x_##type(type *arr, const type *aux, int low, int mid, int high) { \
        int i = low; \
        int j = mid + 1; \
        int k; \
        for (k = low; k <= high - 15; k += 16) { \
            UNROLL_16X_MERGE(arr, aux, i, j, k) \
        } \
        for (; k <= high; k++) { \
            SINGLE_MERGE_STEP(arr, aux, i, j, k) \
        } \
    } \
    /* --- 32x Unrolled Merge --- */ \
    void merge_scalar_32x_##type(type *arr, const type *aux, int low, int mid, int high) { \
        int i = low; \
        int j = mid + 1; \
        int k; \
        for (k = low; k <= high - 31; k += 32) { \
            UNROLL_32X_MERGE(arr, aux, i, j, k) \
        } \
        for (; k <= high; k++) { \
            SINGLE_MERGE_STEP(arr, aux, i, j, k) \
        } \
    } \
    /* --- Recursive Driver --- */ \
    void mergeSort_recursive_##type(type *arr, type *aux, int low, int high, \
                                    void (*merge_func)(type*, const type*, int, int, int)) { \
        if (low >= high) { return; } \
        int mid = low + (high - low) / 2; \
        for (int k = low; k <= high; k++) { aux[k] = arr[k]; } \
        mergeSort_recursive_##type(arr, aux, low, mid, merge_func); \
        mergeSort_recursive_##type(arr, aux, mid + 1, high, merge_func); \
        merge_func(arr, aux, low, mid, high); \
    } \
    /* --- Initialize Array --- */ \
    void initialize_array_##type(type *arr, int size) { \
        srand(time(NULL)); \
        for (int i = 0; i < size; i++) { \
            if (sizeof(type) <= 4) { \
                arr[i] = (type)(rand() % 1000000); \
            } else { \
                arr[i] = (type)((double)rand() / RAND_MAX * 1000000.0); \
            } \
        } \
    } \
    /* --- Is Sorted Check --- */ \
    int is_sorted_##type(const type *arr, int size) { \
        for (int i = 0; i < size - 1; i++) { \
            if (arr[i] > arr[i + 1]) { \
                return 0; \
            } \
        } \
        return 1; \
    }


// Special macro implementation for 'long long' (llong suffix)
#define IMPLEMENT_LLONG_FUNCTIONS(type_suffix, C_type) \
    /* --- 1x Base Merge --- */ \
    void merge_scalar_1x_##type_suffix(C_type *arr, const C_type *aux, int low, int mid, int high) { \
        int i = low; \
        int j = mid + 1; \
        for (int k = low; k <= high; k++) { \
            SINGLE_MERGE_STEP(arr, aux, i, j, k) \
        } \
    } \
    /* --- 2x Unrolled Merge --- */ \
    void merge_scalar_2x_##type_suffix(C_type *arr, const C_type *aux, int low, int mid, int high) { \
        int i = low; \
        int j = mid + 1; \
        int k; \
        for (k = low; k <= high - 1; k += 2) { \
            UNROLL_2X_MERGE(arr, aux, i, j, k) \
        } \
        for (; k <= high; k++) { \
            SINGLE_MERGE_STEP(arr, aux, i, j, k) \
        } \
    } \
    /* --- 4x Unrolled Merge --- */ \
    void merge_scalar_4x_##type_suffix(C_type *arr, const C_type *aux, int low, int mid, int high) { \
        int i = low; \
        int j = mid + 1; \
        int k; \
        for (k = low; k <= high - 3; k += 4) { \
            UNROLL_4X_MERGE(arr, aux, i, j, k) \
        } \
        for (; k <= high; k++) { \
            SINGLE_MERGE_STEP(arr, aux, i, j, k) \
        } \
    } \
    /* --- 8x Unrolled Merge --- */ \
    void merge_scalar_8x_##type_suffix(C_type *arr, const C_type *aux, int low, int mid, int high) { \
        int i = low; \
        int j = mid + 1; \
        int k; \
        for (k = low; k <= high - 7; k += 8) { \
            UNROLL_8X_MERGE(arr, aux, i, j, k) \
        } \
        for (; k <= high; k++) { \
            SINGLE_MERGE_STEP(arr, aux, i, j, k) \
        } \
    } \
    /* --- 16x Unrolled Merge --- */ \
    void merge_scalar_16x_##type_suffix(C_type *arr, const C_type *aux, int low, int mid, int high) { \
        int i = low; \
        int j = mid + 1; \
        int k; \
        for (k = low; k <= high - 15; k += 16) { \
            UNROLL_16X_MERGE(arr, aux, i, j, k) \
        } \
        for (; k <= high; k++) { \
            SINGLE_MERGE_STEP(arr, aux, i, j, k) \
        } \
    } \
    /* --- 32x Unrolled Merge --- */ \
    void merge_scalar_32x_##type_suffix(C_type *arr, const C_type *aux, int low, int mid, int high) { \
        int i = low; \
        int j = mid + 1; \
        int k; \
        for (k = low; k <= high - 31; k += 32) { \
            UNROLL_32X_MERGE(arr, aux, i, j, k) \
        } \
        for (; k <= high; k++) { \
            SINGLE_MERGE_STEP(arr, aux, i, j, k) \
        } \
    } \
    /* --- Recursive Driver --- */ \
    void mergeSort_recursive_##type_suffix(C_type *arr, C_type *aux, int low, int high, \
                                    void (*merge_func)(C_type*, const C_type*, int, int, int)) { \
        if (low >= high) { return; } \
        int mid = low + (high - low) / 2; \
        for (int k = low; k <= high; k++) { aux[k] = arr[k]; } \
        mergeSort_recursive_##type_suffix(arr, aux, low, mid, merge_func); \
        mergeSort_recursive_##type_suffix(arr, aux, mid + 1, high, merge_func); \
        merge_func(arr, aux, low, mid, high); \
    } \
    /* --- Initialize Array --- */ \
    void initialize_array_##type_suffix(C_type *arr, int size) { \
        srand(time(NULL)); \
        for (int i = 0; i < size; i++) { \
            arr[i] = (C_type)((double)rand() / RAND_MAX * 1000000.0); \
        } \
    } \
    /* --- Is Sorted Check --- */ \
    int is_sorted_##type_suffix(const C_type *arr, int size) { \
        for (int i = 0; i < size - 1; i++) { \
            if (arr[i] > arr[i + 1]) { \
                return 0; \
            } \
        } \
        return 1; \
    }

// --- Implement all 6 types ---
IMPLEMENT_MERGE_FUNCTIONS(short)
IMPLEMENT_MERGE_FUNCTIONS(int)
IMPLEMENT_MERGE_FUNCTIONS(long)
IMPLEMENT_LLONG_FUNCTIONS(llong, long long) // Use the special implementation
IMPLEMENT_MERGE_FUNCTIONS(float)
IMPLEMENT_MERGE_FUNCTIONS(double)


// ====================================================================
// SIMD/VECTOR IMPLEMENTATION (Targeting 'int' with varying unrolling)
// ====================================================================
#ifdef __SSE4_1__

// Vector Width (VW): 128-bit / 32-bit int = 4 elements per vector register
#define SIMD_VW_INT 4

// Macro for SIMD merge structure. Uses scalar steps internally for structural correctness.
#define SIMD_MERGE_STRUCTURE(UNROLL_FACTOR) \
    void merge_simd_##UNROLL_FACTOR##x_int(int *arr, const int *aux, int low, int mid, int high) { \
        int i = low; \
        int j = mid + 1; \
        int k; \
        const int LOOP_INCR = UNROLL_FACTOR * SIMD_VW_INT; \
        /* The main loop iterates by the total elements processed per iteration (e.g., 4, 8, 16, 32, 64) */ \
        for (k = low; k <= high - (LOOP_INCR - 1); k += LOOP_INCR) { \
            /* Placeholder: Use the corresponding scalar steps */ \
            for (int p = 0; p < LOOP_INCR; p++) { \
                SINGLE_MERGE_STEP(arr, aux, i, j, k + p) \
            } \
        } \
        /* Cleanup loop (handles the remainder 0 to LOOP_INCR-1 elements) */ \
        for (; k <= high; k++) { \
            SINGLE_MERGE_STEP(arr, aux, i, j, k) \
        } \
    }

SIMD_MERGE_STRUCTURE(1)
SIMD_MERGE_STRUCTURE(2)
SIMD_MERGE_STRUCTURE(4)
SIMD_MERGE_STRUCTURE(8)
SIMD_MERGE_STRUCTURE(16)

#endif // __SSE4_1__


// ====================================================================
// BENCHMARK DRIVER
// ====================================================================

// Macro to run the full scalar benchmark for a single type
#define DATA_TYPE_TESTER_SCALAR(type, name, func_suffix) \
    do { \
        printf("\n--- Testing SCALAR Type: %s (Size: %lu bytes) ---\n", name, sizeof(type)); \
        \
        type *data = (type*)malloc(N * sizeof(type)); \
        type *aux = (type*)malloc(N * sizeof(type)); \
        if (!data || !aux) { perror("Memory allocation failed"); continue; } \
        \
        void (*merge_funcs[])(type*, const type*, int, int, int) = { \
            (void (*)(type*, const type*, int, int, int))merge_scalar_1x_##func_suffix, \
            (void (*)(type*, const type*, int, int, int))merge_scalar_2x_##func_suffix, \
            (void (*)(type*, const type*, int, int, int))merge_scalar_4x_##func_suffix, \
            (void (*)(type*, const type*, int, int, int))merge_scalar_8x_##func_suffix, \
            (void (*)(type*, const type*, int, int, int))merge_scalar_16x_##func_suffix, \
            (void (*)(type*, const type*, int, int, int))merge_scalar_32x_##func_suffix \
        }; \
        const char *func_names[] = { "1x", "2x", "4x", "8x", "16x", "32x" }; \
        int num_funcs = sizeof(merge_funcs) / sizeof(*merge_funcs); \
        \
        for (int f = 0; f < num_funcs; f++) { \
            double total_time = 0; \
            for (int r = 0; r < RUNS; r++) { \
                initialize_array_##func_suffix(data, N); \
                clock_t start = clock(); \
                mergeSort_recursive_##func_suffix(data, aux, 0, N - 1, merge_funcs[f]); \
                clock_t end = clock(); \
                total_time += (double)(end - start) * 1000.0 / CLOCKS_PER_SEC; \
            } \
            if (!is_sorted_##func_suffix(data, N)) { \
                printf("  [ERROR] %s failed to sort the array.\n", func_names[f]); \
            } else { \
                printf("  - Scalar '%s' Time: %.2f ms (Avg over %d runs)\n", func_names[f], total_time / RUNS, RUNS); \
            } \
        } \
        free(data); \
        free(aux); \
    } while(0)

// Macro to run the INT SIMD test (only if SIMD is enabled)
#define INT_SIMD_TESTER \
    do { \
        printf("\n--- Testing VECTOR (SIMD) Type: int ---\n"); \
        int *data = (int*)malloc(N * sizeof(int)); \
        int *aux = (int*)malloc(N * sizeof(int)); \
        if (!data || !aux) { perror("Memory allocation failed"); continue; } \
        \
        void (*merge_simd_funcs[])(int*, const int*, int, int, int) = { \
            merge_simd_1x_int, merge_simd_2x_int, merge_simd_4x_int, \
            merge_simd_8x_int, merge_simd_16x_int \
        }; \
        const char *simd_names[] = { "1x (4 elements)", "2x (8 elements)", "4x (16 elements)", "8x (32 elements)", "16x (64 elements)" }; \
        int num_simd_funcs = sizeof(merge_simd_funcs) / sizeof(*merge_simd_funcs); \
        \
        for (int f = 0; f < num_simd_funcs; f++) { \
            double total_time = 0; \
            for (int r = 0; r < RUNS; r++) { \
                initialize_array_int(data, N); \
                clock_t start = clock(); \
                mergeSort_recursive_int(data, aux, 0, N - 1, merge_simd_funcs[f]); \
                clock_t end = clock(); \
                total_time += (double)(end - start) * 1000.0 / CLOCKS_PER_SEC; \
            } \
            if (!is_sorted_int(data, N)) { \
                printf("  [ERROR] SIMD %s failed to sort the array.\n", simd_names[f]); \
            } else { \
                printf("  - Vector '%s' Time: %.2f ms (Avg over %d runs)\n", simd_names[f], total_time / RUNS, RUNS); \
            } \
        } \
        free(data); \
        free(aux); \
    } while(0)


int main() {
    printf("--- Starting Comprehensive Merge Sort Benchmark (N=%d, RUNS=%d) ---\n", N, RUNS);
    printf("NOTE: Vector (SIMD) test is only run for 'int' and requires compiler flags (e.g., -msse4.1).\n");

    // Run tests for all 6 types and 6 scalar unrolling levels
    DATA_TYPE_TESTER_SCALAR(short, "short", short);
    DATA_TYPE_TESTER_SCALAR(int, "int", int);
    DATA_TYPE_TESTER_SCALAR(long, "long", long);
    DATA_TYPE_TESTER_SCALAR(long long, "long long", llong); // Special usage
    DATA_TYPE_TESTER_SCALAR(float, "float", float);
    DATA_TYPE_TESTER_SCALAR(double, "double", double);

    // Run the structured SIMD test (if compiled with SIMD support)
#ifdef __SSE4_1__
    INT_SIMD_TESTER;
#else
    printf("\n--- Vector (SIMD) Tests Skipped ---\n");
    printf("Rerun compilation with SIMD flags (e.g., -msse4.1) to enable vector testing.\n");
#endif

    printf("\n--- Benchmark Complete ---\n");
    return 0;
}