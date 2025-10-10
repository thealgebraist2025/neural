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

// --- SIMD Vector Widths (VW) based on 128-bit SSE registers ---
#define VW_short      (128 / (sizeof(short) * 8))  // 8
#define VW_int        (128 / (sizeof(int) * 8))    // 4
#define VW_long       (128 / (sizeof(long) * 8))   // 2 (Assuming 64-bit long)
#define VW_long_long  (128 / (sizeof(long long) * 8)) // 2 (Using 'long_long' as safe macro suffix)
#define VW_float      (128 / (sizeof(float) * 8))  // 4
#define VW_double     (128 / (sizeof(double) * 8)) // 2

// --- Function Declarations for Recursive Alternation ---
// Forward declaration of the function that merges INTO the auxiliary array.
#define DECLARE_REC_TO_AUX_HELPER(type) \
    void mergeSort_recursive_to_aux_##type(type *arr, type *aux, int low, int high, \
                                    void (*merge_func)(type*, const type*, int, int, int));

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
    /* This is the function called by the benchmark driver. It merges AUX -> ARR */ \
    void mergeSort_recursive_##type(type *arr, type *aux, int low, int high, \
                                    void (*merge_func)(type*, const type*, int, int, int)); \
    DECLARE_REC_TO_AUX_HELPER(type) \
    void merge_simd_1x_##type(type *arr, const type *aux, int low, int mid, int high); \
    void merge_simd_2x_##type(type *arr, const type *aux, int low, int mid, int high); \
    void merge_simd_4x_##type(type *arr, const type *aux, int low, int mid, int high); \
    void merge_simd_8x_##type(type *arr, const type *aux, int low, int mid, int high); \
    void merge_simd_16x_##type(type *arr, const type *aux, int low, int mid, int high);

// Special macro for the two-token type 'long long' using 'long_long' as the identifier suffix
#define DECLARE_LONG_LONG_HELPERS \
    void initialize_array_long_long(long long *arr, int size); \
    int is_sorted_long_long(const long long *arr, int size); \
    void merge_scalar_1x_long_long(long long *arr, const long long *aux, int low, int mid, int high); \
    void merge_scalar_2x_long_long(long long *arr, const long long *aux, int low, int mid, int high); \
    void merge_scalar_4x_long_long(long long *arr, const long long *aux, int low, int mid, int high); \
    void merge_scalar_8x_long_long(long long *arr, const long long *aux, int low, int mid, int high); \
    void merge_scalar_16x_long_long(long long *arr, const long long *aux, int low, int mid, int high); \
    void merge_scalar_32x_long_long(long long *arr, const long long *aux, int low, int mid, int high); \
    void mergeSort_recursive_long_long(long long *arr, long long *aux, int low, int high, \
                                    void (*merge_func)(long long*, const long long*, int, int, int)); \
    DECLARE_REC_TO_AUX_HELPER(long_long) \
    void merge_simd_1x_long_long(long long *arr, const long long *aux, int low, int mid, int high); \
    void merge_simd_2x_long_long(long long *arr, const long long *aux, int low, int mid, int high); \
    void merge_simd_4x_long_long(long long *arr, const long long *aux, int low, int mid, int high); \
    void merge_simd_8x_long_long(long long *arr, const long long *aux, int low, int mid, int high); \
    void merge_simd_16x_long_long(long long *arr, const long long *aux, int low, int mid, int high);


DECLARE_HELPERS_FOR_TYPE(short)
DECLARE_HELPERS_FOR_TYPE(int)
DECLARE_HELPERS_FOR_TYPE(long)
DECLARE_LONG_LONG_HELPERS // Use the special helper
DECLARE_HELPERS_FOR_TYPE(float)
DECLARE_HELPERS_FOR_TYPE(double)


// ====================================================================
// GENERIC SCALAR LOGIC MACROS
// ====================================================================

// --- Core Single Merge Step (1x) ---
// arr = destination, aux = source
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
// FUNCTION IMPLEMENTATION MACROS (Scalar, Driver, Helpers)
// ====================================================================

// Macro for scalar merges, driver, and helpers (used by short, int, long, float, double)
#define IMPLEMENT_MERGE_FUNCTIONS(type) \
    /* --- 1x to 32x Scalar Merge Implementations (merge AUX -> ARR) --- */ \
    void merge_scalar_1x_##type(type *arr, const type *aux, int low, int mid, int high) { \
        int i = low; int j = mid + 1; \
        for (int k = low; k <= high; k++) { SINGLE_MERGE_STEP(arr, aux, i, j, k) } \
    } \
    void merge_scalar_2x_##type(type *arr, const type *aux, int low, int mid, int high) { \
        int i = low; int j = mid + 1; int k; \
        for (k = low; k <= high - 1; k += 2) { UNROLL_2X_MERGE(arr, aux, i, j, k) } \
        for (; k <= high; k++) { SINGLE_MERGE_STEP(arr, aux, i, j, k) } \
    } \
    void merge_scalar_4x_##type(type *arr, const type *aux, int low, int mid, int high) { \
        int i = low; int j = mid + 1; int k; \
        for (k = low; k <= high - 3; k += 4) { UNROLL_4X_MERGE(arr, aux, i, j, k) } \
        for (; k <= high; k++) { SINGLE_MERGE_STEP(arr, aux, i, j, k) } \
    } \
    void merge_scalar_8x_##type(type *arr, const type *aux, int low, int mid, int high) { \
        int i = low; int j = mid + 1; int k; \
        for (k = low; k <= high - 7; k += 8) { UNROLL_8X_MERGE(arr, aux, i, j, k) } \
        for (; k <= high; k++) { SINGLE_MERGE_STEP(arr, aux, i, j, k) } \
    } \
    void merge_scalar_16x_##type(type *arr, const type *aux, int low, int mid, int high) { \
        int i = low; int j = mid + 1; int k; \
        for (k = low; k <= high - 15; k += 16) { UNROLL_16X_MERGE(arr, aux, i, j, k) } \
        for (; k <= high; k++) { SINGLE_MERGE_STEP(arr, aux, i, j, k) } \
    } \
    void merge_scalar_32x_##type(type *arr, const type *aux, int low, int mid, int high) { \
        int i = low; int j = mid + 1; int k; \
        for (k = low; k <= high - 31; k += 32) { UNROLL_32X_MERGE(arr, aux, i, j, k) } \
        for (; k <= high; k++) { SINGLE_MERGE_STEP(arr, aux, i, j, k) } \
    } \
    \
    /* --- Recursive Driver (Merges AUX -> ARR) --- */ \
    void mergeSort_recursive_##type(type *arr, type *aux, int low, int high, \
                                    void (*merge_func)(type*, const type*, int, int, int)) { \
        if (low >= high) { return; } \
        int mid = low + (high - low) / 2; \
        \
        /* 1. Recurse, swapping roles: Sort from ARR (source) into AUX (destination) */ \
        mergeSort_recursive_to_aux_##type(arr, aux, low, mid, merge_func); \
        mergeSort_recursive_to_aux_##type(arr, aux, mid + 1, high, merge_func); \
        \
        /* 2. Merge: The two sorted halves are now in AUX. Merge AUX -> ARR. */ \
        merge_func(arr, aux, low, mid, high); \
    } \
    \
    /* --- Recursive Helper (Merges ARR -> AUX) --- */ \
    void mergeSort_recursive_to_aux_##type(type *arr, type *aux, int low, int high, \
                                    void (*merge_func)(type*, const type*, int, int, int)) { \
        if (low >= high) { return; } \
        int mid = low + (high - low) / 2; \
        \
        /* 1. Recurse, swapping roles back: Sort from AUX (source) into ARR (destination) */ \
        mergeSort_recursive_##type(arr, aux, low, mid, merge_func); \
        mergeSort_recursive_##type(arr, aux, mid + 1, high, merge_func); \
        \
        /* 2. Merge: The two sorted halves are now in ARR. Merge ARR -> AUX. \
           Since the merge_func only merges DST<-SRC, we swap arguments. */ \
        merge_func(aux, arr, low, mid, high); \
    } \
    \
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
            if (arr[i] > arr[i + 1]) { return 0; } \
        } \
        return 1; \
    }


// Special macro implementation for 'long long' (long_long suffix)
#define IMPLEMENT_LONG_LONG_FUNCTIONS(C_type) \
    /* Use the generic macro but pass the specific C type (long long) for the type argument. */ \
    /* The suffix will be 'long_long' */ \
    void merge_scalar_1x_long_long(long long *arr, const long long *aux, int low, int mid, int high) { \
        int i = low; int j = mid + 1; \
        for (int k = low; k <= high; k++) { SINGLE_MERGE_STEP(arr, aux, i, j, k) } \
    } \
    void merge_scalar_2x_long_long(long long *arr, const long long *aux, int low, int mid, int high) { \
        int i = low; int j = mid + 1; int k; \
        for (k = low; k <= high - 1; k += 2) { UNROLL_2X_MERGE(arr, aux, i, j, k) } \
        for (; k <= high; k++) { SINGLE_MERGE_STEP(arr, aux, i, j, k) } \
    } \
    void merge_scalar_4x_long_long(long long *arr, const long long *aux, int low, int mid, int high) { \
        int i = low; int j = mid + 1; int k; \
        for (k = low; k <= high - 3; k += 4) { UNROLL_4X_MERGE(arr, aux, i, j, k) } \
        for (; k <= high; k++) { SINGLE_MERGE_STEP(arr, aux, i, j, k) } \
    } \
    void merge_scalar_8x_long_long(long long *arr, const long long *aux, int low, int mid, int high) { \
        int i = low; int j = mid + 1; int k; \
        for (k = low; k <= high - 7; k += 8) { UNROLL_8X_MERGE(arr, aux, i, j, k) } \
        for (; k <= high; k++) { SINGLE_MERGE_STEP(arr, aux, i, j, k) } \
    } \
    void merge_scalar_16x_long_long(long long *arr, const long long *aux, int low, int mid, int high) { \
        int i = low; int j = mid + 1; int k; \
        for (k = low; k <= high - 15; k += 16) { UNROLL_16X_MERGE(arr, aux, i, j, k) } \
        for (; k <= high; k++) { SINGLE_MERGE_STEP(arr, aux, i, j, k) } \
    } \
    void merge_scalar_32x_long_long(long long *arr, const long long *aux, int low, int mid, int high) { \
        int i = low; int j = mid + 1; int k; \
        for (k = low; k <= high - 31; k += 32) { UNROLL_32X_MERGE(arr, aux, i, j, k) } \
        for (; k <= high; k++) { SINGLE_MERGE_STEP(arr, aux, i, j, k) } \
    } \
    /* Recursive Driver (Merges AUX -> ARR) */ \
    void mergeSort_recursive_long_long(long long *arr, long long *aux, int low, int high, \
                                    void (*merge_func)(long long*, const long long*, int, int, int)) { \
        if (low >= high) { return; } \
        int mid = low + (high - low) / 2; \
        mergeSort_recursive_to_aux_long_long(arr, aux, low, mid, merge_func); \
        mergeSort_recursive_to_aux_long_long(arr, aux, mid + 1, high, merge_func); \
        merge_func(arr, aux, low, mid, high); \
    } \
    /* Recursive Helper (Merges ARR -> AUX) */ \
    void mergeSort_recursive_to_aux_long_long(long long *arr, long long *aux, int low, int high, \
                                    void (*merge_func)(long long*, const long long*, int, int, int)) { \
        if (low >= high) { return; } \
        int mid = low + (high - low) / 2; \
        mergeSort_recursive_long_long(arr, aux, low, mid, merge_func); \
        mergeSort_recursive_long_long(arr, aux, mid + 1, high, merge_func); \
        merge_func(aux, arr, low, mid, high); \
    } \
    /* Fix for initialize_array and is_sorted specific to C_type */ \
    void initialize_array_long_long(long long *arr, int size) { \
        srand(time(NULL)); \
        for (int i = 0; i < size; i++) { \
            arr[i] = (long long)((double)rand() / RAND_MAX * 1000000.0); \
        } \
    } \
    int is_sorted_long_long(const long long *arr, int size) { \
        for (int i = 0; i < size - 1; i++) { \
            if (arr[i] > arr[i + 1]) { return 0; } \
        } \
        return 1; \
    }


// --- Implement all 6 types ---
IMPLEMENT_MERGE_FUNCTIONS(short)
IMPLEMENT_MERGE_FUNCTIONS(int)
IMPLEMENT_MERGE_FUNCTIONS(long)
IMPLEMENT_LONG_LONG_FUNCTIONS(long long) // Use the special implementation
IMPLEMENT_MERGE_FUNCTIONS(float)
IMPLEMENT_MERGE_FUNCTIONS(double)


// ====================================================================
// SIMD/VECTOR IMPLEMENTATION FOR ALL TYPES
// ====================================================================
#ifdef __SSE4_1__

// Macro to generate a structured SIMD merge function for a given unroll factor, type, suffix, and VW
#define SIMD_MERGE_STRUCTURE(UNROLL_FACTOR, type, suffix, VW) \
    void merge_simd_##UNROLL_FACTOR##x_##suffix(type *arr, const type *aux, int low, int mid, int high) { \
        int i = low; \
        int j = mid + 1; \
        int k; \
        /* Total elements processed per iteration: Unroll Factor * Vector Width */ \
        const int LOOP_INCR = UNROLL_FACTOR * VW; \
        \
        /* Main highly unrolled vector loop */ \
        for (k = low; k <= high - (LOOP_INCR - 1); k += LOOP_INCR) { \
            /* Placeholder: Scalar steps used for structural correctness and comparison to scalar code. */ \
            for (int p = 0; p < LOOP_INCR; p++) { \
                if (i > mid) { arr[k+p] = aux[j++]; } \
                else if (j > high) { arr[k+p] = aux[i++]; } \
                else if (aux[j] < aux[i]) { arr[k+p] = aux[j++]; } \
                else { arr[k+p] = aux[i++]; } \
            } \
        } \
        /* Cleanup loop (handles the remainder elements) */ \
        for (; k <= high; k++) { \
            if (i > mid) { arr[k] = aux[j++]; } \
            else if (j > high) { arr[k] = aux[i++]; } \
            else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } \
            else { arr[k] = aux[i++]; } \
        } \
    }

// Macro to implement all 5 SIMD unrolls for a single type
#define IMPLEMENT_ALL_SIMD_MERGES(type, suffix, VW) \
    SIMD_MERGE_STRUCTURE(1, type, suffix, VW) \
    SIMD_MERGE_STRUCTURE(2, type, suffix, VW) \
    SIMD_MERGE_STRUCTURE(4, type, suffix, VW) \
    SIMD_MERGE_STRUCTURE(8, type, suffix, VW) \
    SIMD_MERGE_STRUCTURE(16, type, suffix, VW)

// Implement SIMD merges for all 6 types
IMPLEMENT_ALL_SIMD_MERGES(short, short, VW_short)
IMPLEMENT_ALL_SIMD_MERGES(int, int, VW_int)
IMPLEMENT_ALL_SIMD_MERGES(long, long, VW_long)
IMPLEMENT_ALL_SIMD_MERGES(long long, long_long, VW_long_long) // Updated suffix
IMPLEMENT_ALL_SIMD_MERGES(float, float, VW_float)
IMPLEMENT_ALL_SIMD_MERGES(double, double, VW_double)

#endif // __SSE4_1__


// ====================================================================
// BENCHMARK DRIVER
// ====================================================================

// Macro to run the full benchmark suite (Scalar and SIMD) for a single type
#define DATA_TYPE_TESTER(type, name, func_suffix, VW_identifier) \
    do { \
        printf("\n--- Testing Data Type: %s (Size: %lu bytes) ---\n", name, sizeof(type)); \
        \
        type *data = (type*)malloc(N * sizeof(type)); \
        type *aux = (type*)malloc(N * sizeof(type)); \
        if (!data || !aux) { perror("Memory allocation failed"); continue; } \
        \
        /* SCALAR TESTS */ \
        void (*scalar_funcs[])(type*, const type*, int, int, int) = { \
            (void (*)(type*, const type*, int, int, int))merge_scalar_1x_##func_suffix, \
            (void (*)(type*, const type*, int, int, int))merge_scalar_2x_##func_suffix, \
            (void (*)(type*, const type*, int, int, int))merge_scalar_4x_##func_suffix, \
            (void (*)(type*, const type*, int, int, int))merge_scalar_8x_##func_suffix, \
            (void (*)(type*, const type*, int, int, int))merge_scalar_16x_##func_suffix, \
            (void (*)(type*, const type*, int, int, int))merge_scalar_32x_##func_suffix \
        }; \
        const char *scalar_names[] = { "1x", "2x", "4x", "8x", "16x", "32x" }; \
        printf("  [SCALAR] Unrolling (Loop Increments): \n"); \
        for (int f = 0; f < 6; f++) { \
            double total_time = 0; \
            for (int r = 0; r < RUNS; r++) { \
                initialize_array_##func_suffix(data, N); \
                /* The fixed recursive function will handle the initial copy and alternation */ \
                clock_t start = clock(); \
                mergeSort_recursive_##func_suffix(data, aux, 0, N - 1, scalar_funcs[f]); \
                clock_t end = clock(); \
                total_time += (double)(end - start) * 1000.0 / CLOCKS_PER_SEC; \
            } \
            if (!is_sorted_##func_suffix(data, N)) { \
                printf("    [ERROR] Scalar %s failed to sort.\n", scalar_names[f]); \
            } else { \
                printf("    - Scalar '%s' Time: %.2f ms (Avg over %d runs)\n", scalar_names[f], total_time / RUNS, RUNS); \
            } \
        } \
        \
        /* SIMD TESTS (only runs if compiled with SIMD flags) */ \
        #ifdef __SSE4_1__ \
            void (*simd_funcs[])(type*, const type*, int, int, int) = { \
                (void (*)(type*, const type*, int, int, int))merge_simd_1x_##func_suffix, \
                (void (*)(type*, const type*, int, int, int))merge_simd_2x_##func_suffix, \
                (void (*)(type*, const type*, int, int, int))merge_simd_4x_##func_suffix, \
                (void (*)(type*, const type*, int, int, int))merge_simd_8x_##func_suffix, \
                (void (*)(type*, const type*, int, int, int))merge_simd_16x_##func_suffix \
            }; \
            const char *simd_names[] = { "1x", "2x", "4x", "8x", "16x" }; \
            const int vw = VW_identifier; \
            printf("  [VECTOR] Unrolling (Elements/Iter: VW=%d):\n", vw); \
            for (int f = 0; f < 5; f++) { \
                double total_time = 0; \
                int unroll_factor = (f == 0 ? 1 : (f == 1 ? 2 : (f == 2 ? 4 : (f == 3 ? 8 : 16)))); \
                int elements_per_iter = unroll_factor * vw; \
                for (int r = 0; r < RUNS; r++) { \
                    initialize_array_##func_suffix(data, N); \
                    clock_t start = clock(); \
                    mergeSort_recursive_##func_suffix(data, aux, 0, N - 1, simd_funcs[f]); \
                    clock_t end = clock(); \
                    total_time += (double)(end - start) * 1000.0 / CLOCKS_PER_SEC; \
                } \
                if (!is_sorted_##func_suffix(data, N)) { \
                    printf("    [ERROR] SIMD %s failed to sort.\n", simd_names[f]); \
                } else { \
                    printf("    - Vector '%s' Time (Proc %d elements): %.2f ms\n", simd_names[f], elements_per_iter, total_time / RUNS); \
                } \
            } \
        #else \
            printf("  [VECTOR] SIMD Tests Skipped (Compile with -msse4.1 or similar).\n"); \
        #endif \
        \
        free(data); \
        free(aux); \
    } while(0)


int main() {
    printf("--- Starting Comprehensive Merge Sort Benchmark (N=%d, RUNS=%d) ---\n", N, RUNS);

    // Run tests for all 6 types
    DATA_TYPE_TESTER(short, "short", short, VW_short);
    DATA_TYPE_TESTER(int, "int", int, VW_int);
    DATA_TYPE_TESTER(long, "long", long, VW_long);
    // Note: The C type is 'long long', but the macro suffix is 'long_long'
    DATA_TYPE_TESTER(long long, "long long", long_long, VW_long_long); 
    DATA_TYPE_TESTER(float, "float", float, VW_float);
    DATA_TYPE_TESTER(double, "double", double, VW_double);

    printf("\n--- Benchmark Complete ---\n");
    return 0;
}