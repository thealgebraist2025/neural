#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// --- Configuration ---
#define N (1 << 20) // 1,048,576 elements for benchmarking
#define RUNS 3      // Number of runs to average time
// ---------------------

// --- Type-Agnostic Function Declarations ---
// We use void* and size_t to handle different data types (short, int, double, etc.) generically.

typedef void (*MergeFunction)(void*, const void*, int, int, int, size_t);

// Forward declaration of the recursive driver
void mergeSort_recursive(void *arr, void *aux, int low, int high, size_t element_size, MergeFunction merge_func);


// ====================================================================
// GENERIC HELPER FUNCTIONS
// ====================================================================

/**
 * @brief Performs a type-agnostic comparison (assumes numeric types).
 */
int compare(const void *a, const void *b, size_t element_size) {
    // This is a simplification; a production-level benchmark would need a type-specific comparison.
    // We assume the data types are simple numerics for direct byte-level comparison.
    if (element_size == sizeof(int) || element_size == sizeof(short)) {
        if (*(int*)a < *(int*)b) return -1;
        if (*(int*)a > *(int*)b) return 1;
        return 0;
    } else { // Handle larger types like long/double
        if (*(double*)a < *(double*)b) return -1;
        if (*(double*)a > *(double*)b) return 1;
        return 0;
    }
}

/**
 * @brief Checks if the array is sorted.
 */
int is_sorted(const void *arr, int size, size_t element_size) {
    const char *byte_arr = (const char *)arr;
    for (int i = 0; i < size - 1; i++) {
        const void *a = byte_arr + (i * element_size);
        const void *b = byte_arr + ((i + 1) * element_size);
        if (compare(a, b, element_size) > 0) {
            return 0; // Not sorted
        }
    }
    return 1;
}

/**
 * @brief Initializes an array with random values (simplified, assumes small types or casts).
 */
void initialize_array(void *arr, int size, size_t element_size) {
    srand(time(NULL));
    char *byte_arr = (char *)arr;

    for (int i = 0; i < size; i++) {
        // Use a generic int value and cast to the target type
        int rand_val = rand() % 1000000;
        void *target = byte_arr + (i * element_size);
        
        if (element_size == sizeof(short)) { *(short*)target = (short)rand_val; }
        else if (element_size == sizeof(int)) { *(int*)target = rand_val; }
        else if (element_size == sizeof(long)) { *(long*)target = (long)rand_val; }
        else if (element_size == sizeof(long long)) { *(long long*)target = (long long)rand_val; }
        else if (element_size == sizeof(float)) { *(float*)target = (float)rand_val + ((float)rand() / RAND_MAX); }
        else if (element_size == sizeof(double)) { *(double*)target = (double)rand_val + ((double)rand() / RAND_MAX); }
    }
}


// ====================================================================
// MERGE FUNCTIONS (Type-Agnostic, using memcpy)
// ====================================================================

// Utility macro for copying an element from aux[source_index] to arr[dest_index]
#define COPY_ELEMENT(dest_arr, src_arr, dest_idx, src_idx, elem_size) \
    memcpy((char*)dest_arr + (dest_idx * elem_size), (const char*)src_arr + (src_idx++ * elem_size), elem_size)

// Utility macro for comparison
#define COMPARE_ELEMENTS(arr, idx_i, idx_j, elem_size) \
    (compare((const char*)arr + (idx_j * elem_size), (const char*)arr + (idx_i * elem_size), elem_size) < 0)


/**
 * @brief Standard (1x unrolled) merge function.
 */
void merge_base(void *arr, const void *aux, int low, int mid, int high, size_t element_size) {
    int i = low;
    int j = mid + 1;

    for (int k = low; k <= high; k++) {
        if (i > mid) {
            COPY_ELEMENT(arr, aux, k, j, element_size);
        } else if (j > high) {
            COPY_ELEMENT(arr, aux, k, i, element_size);
        } else if (COMPARE_ELEMENTS(aux, i, j, element_size)) {
            COPY_ELEMENT(arr, aux, k, j, element_size);
        } else {
            COPY_ELEMENT(arr, aux, k, i, element_size);
        }
    }
}


/**
 * @brief 4x Unrolled merge function.
 */
void merge_unroll_4x(void *arr, const void *aux, int low, int mid, int high, size_t element_size) {
    int i = low;
    int j = mid + 1;
    int k;

    // --- 1. Main Unrolled Loop (Processes blocks of 4) ---
    for (k = low; k <= high - 3; k += 4) {
        // Element k
        if (i > mid) { COPY_ELEMENT(arr, aux, k, j, element_size); } 
        else if (j > high) { COPY_ELEMENT(arr, aux, k, i, element_size); } 
        else if (COMPARE_ELEMENTS(aux, i, j, element_size)) { COPY_ELEMENT(arr, aux, k, j, element_size); } 
        else { COPY_ELEMENT(arr, aux, k, i, element_size); }

        // Element k+1
        if (i > mid) { COPY_ELEMENT(arr, aux, k+1, j, element_size); } 
        else if (j > high) { COPY_ELEMENT(arr, aux, k+1, i, element_size); } 
        else if (COMPARE_ELEMENTS(aux, i, j, element_size)) { COPY_ELEMENT(arr, aux, k+1, j, element_size); } 
        else { COPY_ELEMENT(arr, aux, k+1, i, element_size); }
        
        // Element k+2
        if (i > mid) { COPY_ELEMENT(arr, aux, k+2, j, element_size); } 
        else if (j > high) { COPY_ELEMENT(arr, aux, k+2, i, element_size); } 
        else if (COMPARE_ELEMENTS(aux, i, j, element_size)) { COPY_ELEMENT(arr, aux, k+2, j, element_size); } 
        else { COPY_ELEMENT(arr, aux, k+2, i, element_size); }

        // Element k+3
        if (i > mid) { COPY_ELEMENT(arr, aux, k+3, j, element_size); } 
        else if (j > high) { COPY_ELEMENT(arr, aux, k+3, i, element_size); } 
        else if (COMPARE_ELEMENTS(aux, i, j, element_size)) { COPY_ELEMENT(arr, aux, k+3, j, element_size); } 
        else { COPY_ELEMENT(arr, aux, k+3, i, element_size); }
    }
    
    // --- 2. Cleanup Loop (Handles the remaining 0, 1, 2, or 3 elements) ---
    for (; k <= high; k++) {
        if (i > mid) {
            COPY_ELEMENT(arr, aux, k, j, element_size);
        } else if (j > high) {
            COPY_ELEMENT(arr, aux, k, i, element_size);
        } else if (COMPARE_ELEMENTS(aux, i, j, element_size)) {
            COPY_ELEMENT(arr, aux, k, j, element_size);
        } else {
            COPY_ELEMENT(arr, aux, k, i, element_size);
        }
    }
}


/**
 * @brief 8x Unrolled merge function.
 */
void merge_unroll_8x(void *arr, const void *aux, int low, int mid, int high, size_t element_size) {
    int i = low;
    int j = mid + 1;
    int k;

    // --- 1. Main Unrolled Loop (Processes blocks of 8) ---
    for (k = low; k <= high - 7; k += 8) {
        // Run the 8 comparisons/copies in sequence. This is boilerplate 
        // extension of the 4x logic, ensuring all 8 steps check bounds.
        
        // Elements k to k+3 (First half of the 8x block)
        #define MERGE_STEP(offset) \
            if (i > mid) { COPY_ELEMENT(arr, aux, k + offset, j, element_size); } \
            else if (j > high) { COPY_ELEMENT(arr, aux, k + offset, i, element_size); } \
            else if (COMPARE_ELEMENTS(aux, i, j, element_size)) { COPY_ELEMENT(arr, aux, k + offset, j, element_size); } \
            else { COPY_ELEMENT(arr, aux, k + offset, i, element_size); }

        MERGE_STEP(0); MERGE_STEP(1); MERGE_STEP(2); MERGE_STEP(3);
        
        // Elements k+4 to k+7 (Second half of the 8x block)
        MERGE_STEP(4); MERGE_STEP(5); MERGE_STEP(6); MERGE_STEP(7);
        
        #undef MERGE_STEP
    }
    
    // --- 2. Cleanup Loop (Handles the remaining 0 to 7 elements) ---
    // The standard 1x merge is used for the remainder.
    for (; k <= high; k++) {
        if (i > mid) {
            COPY_ELEMENT(arr, aux, k, j, element_size);
        } else if (j > high) {
            COPY_ELEMENT(arr, aux, k, i, element_size);
        } else if (COMPARE_ELEMENTS(aux, i, j, element_size)) {
            COPY_ELEMENT(arr, aux, k, j, element_size);
        } else {
            COPY_ELEMENT(arr, aux, k, i, element_size);
        }
    }
}

// --------------------------------------------------------------------
// MERGE SORT RECURSIVE DRIVER (Type-Agnostic)
// --------------------------------------------------------------------

void mergeSort_recursive(void *arr, void *aux, int low, int high, size_t element_size, MergeFunction merge_func) {
    if (low >= high) {
        return;
    }

    int mid = low + (high - low) / 2;

    // Use a simple copy-based approach for safety:
    // 1. Copy current data from 'arr' to 'aux' for the merge source
    char *arr_byte = (char *)arr;
    char *aux_byte = (char *)aux;
    
    for (int k = low; k <= high; k++) {
        memcpy(aux_byte + (k * element_size), arr_byte + (k * element_size), element_size);
    }
    
    // Recurse on the two halves (which will ultimately write back to 'arr')
    mergeSort_recursive(arr, aux, low, mid, element_size, merge_func);
    mergeSort_recursive(arr, aux, mid + 1, high, element_size, merge_func);

    // Merge from 'aux' back into 'arr' (the destination)
    merge_func(arr, aux, low, mid, high, element_size);
}

// ====================================================================
// BENCHMARK DRIVER
// ====================================================================

// Macro to encapsulate the full benchmark run for a single type
#define DATA_TYPE_TESTER(type, name) \
    do { \
        printf("\n--- Testing Data Type: %s (Size: %lu bytes) ---\n", name, sizeof(type)); \
        size_t element_size = sizeof(type); \
        size_t total_bytes = N * element_size; \
        void *data = malloc(total_bytes); \
        void *aux = malloc(total_bytes); \
        if (!data || !aux) { perror("Memory allocation failed"); continue; } \
        \
        MergeFunction funcs[] = { merge_base, merge_unroll_4x, merge_unroll_8x }; \
        const char *func_names[] = { "1x Base Merge", "4x Unrolled", "8x Unrolled" }; \
        int num_funcs = sizeof(funcs) / sizeof(MergeFunction); \
        \
        for (int f = 0; f < num_funcs; f++) { \
            double total_time = 0; \
            for (int r = 0; r < RUNS; r++) { \
                initialize_array(data, N, element_size); \
                clock_t start = clock(); \
                mergeSort_recursive(data, aux, 0, N - 1, element_size, funcs[f]); \
                clock_t end = clock(); \
                total_time += (double)(end - start) * 1000.0 / CLOCKS_PER_SEC; \
            } \
            if (!is_sorted(data, N, element_size)) { \
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

    // Run tests for all requested data types
    DATA_TYPE_TESTER(short, "short");
    DATA_TYPE_TESTER(int, "int");
    DATA_TYPE_TESTER(long, "long");
    DATA_TYPE_TESTER(long long, "long long");
    DATA_TYPE_TESTER(float, "float");
    DATA_TYPE_TESTER(double, "double");

    printf("\n--- Benchmark Complete ---\n");
    return 0;
}