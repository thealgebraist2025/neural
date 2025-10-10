#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// --- Configuration ---
#define N (1 << 20) // 1,048,576 elements for benchmarking
#define RUNS 3      // Number of runs to average time
// ---------------------

// --- Type-Specific Function Definitions ---
// Forward declarations for clarity
#define DEFINE_HELPERS_FOR_TYPE(type) \
    void initialize_array_##type(type *arr, int size); \
    int is_sorted_##type(const type *arr, int size); \
    void merge_base_##type(type *arr, const type *aux, int low, int mid, int high); \
    void merge_unroll_4x_##type(type *arr, const type *aux, int low, int mid, int high); \
    void merge_unroll_8x_##type(type *arr, const type *aux, int low, const type *aux, int mid, int high); \
    void mergeSort_recursive_##type(type *arr, type *aux, int low, int high, \
                                    void (*merge_func)(type*, const type*, int, int, int));

DEFINE_HELPERS_FOR_TYPE(short)
DEFINE_HELPERS_FOR_TYPE(int)
DEFINE_HELPERS_FOR_TYPE(long)
DEFINE_HELPERS_FOR_TYPE(long long)
DEFINE_HELPERS_FOR_TYPE(float)
DEFINE_HELPERS_FOR_TYPE(double)


// ====================================================================
// GENERIC LOGIC MACROS (Used for Function Generation)
// ====================================================================

// --- Core Single Merge Step ---
// Compares aux[i] and aux[j] and writes the smaller element to arr[k]
#define SINGLE_MERGE_STEP(arr, aux, i, j, k) \
    if (i > mid) { arr[k] = aux[j++]; } \
    else if (j > high) { arr[k] = aux[i++]; } \
    else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } \
    else { arr[k] = aux[i++]; }

// --- Unrolled 4x Merge Steps ---
#define UNROLL_4X_MERGE(arr, aux, i, j, k) \
    /* k */ SINGLE_MERGE_STEP(arr, aux, i, j, k) \
    /* k+1 */ SINGLE_MERGE_STEP(arr, aux, i, j, k + 1) \
    /* k+2 */ SINGLE_MERGE_STEP(arr, aux, i, j, k + 2) \
    /* k+3 */ SINGLE_MERGE_STEP(arr, aux, i, j, k + 3)

// --- Unrolled 8x Merge Steps ---
#define UNROLL_8X_MERGE(arr, aux, i, j, k) \
    UNROLL_4X_MERGE(arr, aux, i, j, k) \
    UNROLL_4X_MERGE(arr, aux, i, j, k + 4)


// ====================================================================
// FUNCTION GENERATION MACRO
// ====================================================================

#define IMPLEMENT_MERGE_FUNCTIONS(type) \
    /* ---------------------------------------------------- */ \
    /* 1. Standard (1x) Merge Function */ \
    /* ---------------------------------------------------- */ \
    void merge_base_##type(type *arr, const type *aux, int low, int mid, int high) { \
        int i = low; \
        int j = mid + 1; \
        for (int k = low; k <= high; k++) { \
            SINGLE_MERGE_STEP(arr, aux, i, j, k) \
        } \
    } \
    \
    /* ---------------------------------------------------- */ \
    /* 2. 4x Unrolled Merge Function */ \
    /* ---------------------------------------------------- */ \
    void merge_unroll_4x_##type(type *arr, const type *aux, int low, int mid, int high) { \
        int i = low; \
        int j = mid + 1; \
        int k; \
        for (k = low; k <= high - 3; k += 4) { \
            UNROLL_4X_MERGE(arr, aux, i, j, k) \
        } \
        /* Cleanup loop */ \
        for (; k <= high; k++) { \
            SINGLE_MERGE_STEP(arr, aux, i, j, k) \
        } \
    } \
    \
    /* ---------------------------------------------------- */ \
    /* 3. 8x Unrolled Merge Function */ \
    /* ---------------------------------------------------- */ \
    void merge_unroll_8x_##type(type *arr, const type *aux, int low, int mid, int high) { \
        int i = low; \
        int j = mid + 1; \
        int k; \
        for (k = low; k <= high - 7; k += 8) { \
            UNROLL_8X_MERGE(arr, aux, i, j, k) \
        } \
        /* Cleanup loop */ \
        for (; k <= high; k++) { \
            SINGLE_MERGE_STEP(arr, aux, i, j, k) \
        } \
    } \
    \
    /* ---------------------------------------------------- */ \
    /* 4. Recursive Driver */ \
    /* ---------------------------------------------------- */ \
    void mergeSort_recursive_##type(type *arr, type *aux, int low, int high, \
                                    void (*merge_func)(type*, const type*, int, int, int)) { \
        if (low >= high) { return; } \
        int mid = low + (high - low) / 2; \
        \
        /* Copy data from arr to aux for merge source */ \
        for (int k = low; k <= high; k++) { \
            aux[k] = arr[k]; \
        } \
        \
        /* Recurse */ \
        mergeSort_recursive_##type(arr, aux, low, mid, merge_func); \
        mergeSort_recursive_##type(arr, aux, mid + 1, high, merge_func); \
        \
        /* Merge from aux back into arr */ \
        merge_func(arr, aux, low, mid, high); \
    } \
    \
    /* ---------------------------------------------------- */ \
    /* 5. Initialize Array */ \
    /* ---------------------------------------------------- */ \
    void initialize_array_##type(type *arr, int size) { \
        srand(time(NULL)); \
        for (int i = 0; i < size; i++) { \
            /* Use rand() and scale/cast appropriately */ \
            if (sizeof(type) <= 4) { \
                arr[i] = (type)(rand() % 1000000); \
            } else { \
                arr[i] = (type)((double)rand() / RAND_MAX * 1000000.0); \
            } \
        } \
    } \
    \
    /* ---------------------------------------------------- */ \
    /* 6. Is Sorted Check */ \
    /* ---------------------------------------------------- */ \
    int is_sorted_##type(const type *arr, int size) { \
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
IMPLEMENT_MERGE_FUNCTIONS(long long)
IMPLEMENT_MERGE_FUNCTIONS(float)
IMPLEMENT_MERGE_FUNCTIONS(double)


// ====================================================================
// BENCHMARK DRIVER
// ====================================================================

// Macro to encapsulate the full benchmark run for a single type
#define DATA_TYPE_TESTER(type, name) \
    do { \
        printf("\n--- Testing Data Type: %s (Size: %lu bytes) ---\n", name, sizeof(type)); \
        \
        type *data = (type*)malloc(N * sizeof(type)); \
        type *aux = (type*)malloc(N * sizeof(type)); \
        if (!data || !aux) { perror("Memory allocation failed"); continue; } \
        \
        /* Array of function pointers for the current type */ \
        void (*merge_funcs[])(type*, const type*, int, int, int) = { \
            merge_base_##type, \
            merge_unroll_4x_##type, \
            merge_unroll_8x_##type \
        }; \
        const char *func_names[] = { "1x Base Merge", "4x Unrolled", "8x Unrolled" }; \
        int num_funcs = sizeof(merge_funcs) / sizeof(*merge_funcs); \
        \
        for (int f = 0; f < num_funcs; f++) { \
            double total_time = 0; \
            for (int r = 0; r < RUNS; r++) { \
                initialize_array_##type(data, N); \
                clock_t start = clock(); \
                mergeSort_recursive_##type(data, aux, 0, N - 1, merge_funcs[f]); \
                clock_t end = clock(); \
                total_time += (double)(end - start) * 1000.0 / CLOCKS_PER_SEC; \
            } \
            \
            if (!is_sorted_##type(data, N)) { \
                printf("  [ERROR] %s failed to sort the array.\n", func_names[f]); \
            } else { \
                printf("  - Variant '%s' Time: %.2f ms (Avg over %d runs)\n", func_names[f], total_time / RUNS, RUNS); \
            } \
        } \
        \
        free(data); \
        free(aux); \
    } while(0)


int main() {
    printf("--- Starting Comprehensive Merge Sort Benchmark (N=%d, RUNS=%d) ---\n", N, RUNS);

    DATA_TYPE_TESTER(short, "short");
    DATA_TYPE_TESTER(int, "int");
    DATA_TYPE_TESTER(long, "long");
    DATA_TYPE_TESTER(long long, "long long");
    DATA_TYPE_TESTER(float, "float");
    DATA_TYPE_TESTER(double, "double");

    printf("\n--- Benchmark Complete ---\n");
    return 0;
}