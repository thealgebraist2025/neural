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
// These are kept as constants for use in the main function logic.
const int VW_short      = (128 / (sizeof(short) * 8));
const int VW_int        = (128 / (sizeof(int) * 8));
const int VW_long       = (128 / (sizeof(long) * 8));
const int VW_long_long  = (128 / (sizeof(long long) * 8));
const int VW_float      = (128 / (sizeof(float) * 8));
const int VW_double     = (128 / (sizeof(double) * 8));


// ====================================================================
// CORE MERGE LOGIC HELPERS (Scalar)
// Since macros are removed, this repeated block is used inline in the merge functions.
// Note: This is NOT a macro, it's a comment showing the logic being repeated.
/*
#define SINGLE_MERGE_STEP(arr, aux, i, j, k) \
    if (i > mid) { arr[k] = aux[j++]; } \
    else if (j > high) { arr[k] = aux[i++]; } \
    else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } \
    else { arr[k] = aux[i++]; }
*/
// ====================================================================


// ====================================================================
// SECTION 1: TYPE 'short' IMPLEMENTATIONS
// ====================================================================

// --- Function Declarations for short ---
void initialize_array_short(short *arr, int size);
int is_sorted_short(const short *arr, int size);
void merge_scalar_1x_short(short *arr, const short *aux, int low, int mid, int high);
void merge_scalar_2x_short(short *arr, const short *aux, int low, int mid, int high);
void merge_scalar_4x_short(short *arr, const short *aux, int low, int mid, int high);
void merge_scalar_8x_short(short *arr, const short *aux, int low, int mid, int high);
void merge_scalar_16x_short(short *arr, const short *aux, int low, int mid, int high);
void merge_scalar_32x_short(short *arr, const short *aux, int low, int mid, int high);
void mergeSort_recursive_short(short *arr, short *aux, int low, int high, void (*merge_func)(short*, const short*, int, int, int));
void mergeSort_recursive_to_aux_short(short *arr, short *aux, int low, int high, void (*merge_func)(short*, const short*, int, int, int));
#ifdef __SSE4_1__
void merge_simd_1x_short(short *arr, const short *aux, int low, int mid, int high);
void merge_simd_2x_short(short *arr, const short *aux, int low, int mid, int high);
void merge_simd_4x_short(short *arr, const short *aux, int low, int mid, int high);
void merge_simd_8x_short(short *arr, const short *aux, int low, int mid, int high);
void merge_simd_16x_short(short *arr, const short *aux, int low, int mid, int high);
#endif

// --- Helper Functions short ---
void initialize_array_short(short *arr, int size) {
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        arr[i] = (short)(rand() % 1000000);
    }
}
int is_sorted_short(const short *arr, int size) {
    for (int i = 0; i < size - 1; i++) {
        if (arr[i] > arr[i + 1]) { return 0; }
    }
    return 1;
}

// --- Merge Functions (Scalar) short ---
void merge_scalar_1x_short(short *arr, const short *aux, int low, int mid, int high) {
    int i = low; int j = mid + 1;
    for (int k = low; k <= high; k++) {
        if (i > mid) { arr[k] = aux[j++]; }
        else if (j > high) { arr[k] = aux[i++]; }
        else if (aux[j] < aux[i]) { arr[k] = aux[j++]; }
        else { arr[k] = aux[i++]; }
    }
}
void merge_scalar_2x_short(short *arr, const short *aux, int low, int mid, int high) {
    int i = low; int j = mid + 1; int k;
    for (k = low; k <= high - 1; k += 2) {
        // UNROLL 1
        if (i > mid) { arr[k] = aux[j++]; } else if (j > high) { arr[k] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } else { arr[k] = aux[i++]; }
        // UNROLL 2
        if (i > mid) { arr[k+1] = aux[j++]; } else if (j > high) { arr[k+1] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k+1] = aux[j++]; } else { arr[k+1] = aux[i++]; }
    }
    for (; k <= high; k++) { // CLEANUP
        if (i > mid) { arr[k] = aux[j++]; } else if (j > high) { arr[k] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } else { arr[k] = aux[i++]; }
    }
}
void merge_scalar_4x_short(short *arr, const short *aux, int low, int mid, int high) {
    int i = low; int j = mid + 1; int k;
    for (k = low; k <= high - 3; k += 4) {
        // UNROLL 1
        if (i > mid) { arr[k] = aux[j++]; } else if (j > high) { arr[k] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } else { arr[k] = aux[i++]; }
        // UNROLL 2
        if (i > mid) { arr[k+1] = aux[j++]; } else if (j > high) { arr[k+1] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k+1] = aux[j++]; } else { arr[k+1] = aux[i++]; }
        // UNROLL 3
        if (i > mid) { arr[k+2] = aux[j++]; } else if (j > high) { arr[k+2] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k+2] = aux[j++]; } else { arr[k+2] = aux[i++]; }
        // UNROLL 4
        if (i > mid) { arr[k+3] = aux[j++]; } else if (j > high) { arr[k+3] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k+3] = aux[j++]; } else { arr[k+3] = aux[i++]; }
    }
    for (; k <= high; k++) { // CLEANUP
        if (i > mid) { arr[k] = aux[j++]; } else if (j > high) { arr[k] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } else { arr[k] = aux[i++]; }
    }
}
void merge_scalar_8x_short(short *arr, const short *aux, int low, int mid, int high) {
    int i = low; int j = mid + 1; int k;
    for (k = low; k <= high - 7; k += 8) {
        for (int p = 0; p < 8; p++) {
            if (i > mid) { arr[k+p] = aux[j++]; } else if (j > high) { arr[k+p] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k+p] = aux[j++]; } else { arr[k+p] = aux[i++]; }
        }
    }
    for (; k <= high; k++) { // CLEANUP
        if (i > mid) { arr[k] = aux[j++]; } else if (j > high) { arr[k] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } else { arr[k] = aux[i++]; }
    }
}
void merge_scalar_16x_short(short *arr, const short *aux, int low, int mid, int high) {
    int i = low; int j = mid + 1; int k;
    for (k = low; k <= high - 15; k += 16) {
        for (int p = 0; p < 16; p++) {
            if (i > mid) { arr[k+p] = aux[j++]; } else if (j > high) { arr[k+p] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k+p] = aux[j++]; } else { arr[k+p] = aux[i++]; }
        }
    }
    for (; k <= high; k++) { // CLEANUP
        if (i > mid) { arr[k] = aux[j++]; } else if (j > high) { arr[k] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } else { arr[k] = aux[i++]; }
    }
}
void merge_scalar_32x_short(short *arr, const short *aux, int low, int mid, int high) {
    int i = low; int j = mid + 1; int k;
    for (k = low; k <= high - 31; k += 32) {
        for (int p = 0; p < 32; p++) {
            if (i > mid) { arr[k+p] = aux[j++]; } else if (j > high) { arr[k+p] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k+p] = aux[j++]; } else { arr[k+p] = aux[i++]; }
        }
    }
    for (; k <= high; k++) { // CLEANUP
        if (i > mid) { arr[k] = aux[j++]; } else if (j > high) { arr[k] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } else { arr[k] = aux[i++]; }
    }
}

// --- Recursive Sort Functions short ---
// Merges ARR (source) -> AUX (destination)
void mergeSort_recursive_to_aux_short(short *arr, short *aux, int low, int high, void (*merge_func)(short*, const short*, int, int, int)) {
    if (low >= high) { return; }
    int mid = low + (high - low) / 2;
    mergeSort_recursive_short(arr, aux, low, mid, merge_func);      // Sorts low->mid into ARR
    mergeSort_recursive_short(arr, aux, mid + 1, high, merge_func); // Sorts mid+1->high into ARR
    merge_func(aux, arr, low, mid, high); // Merge ARR -> AUX
}
// Merges AUX (source) -> ARR (destination)
void mergeSort_recursive_short(short *arr, short *aux, int low, int high, void (*merge_func)(short*, const short*, int, int, int)) {
    if (low >= high) { return; }
    int mid = low + (high - low) / 2;
    mergeSort_recursive_to_aux_short(arr, aux, low, mid, merge_func);      // Sorts low->mid into AUX
    mergeSort_recursive_to_aux_short(arr, aux, mid + 1, high, merge_func); // Sorts mid+1->high into AUX
    merge_func(arr, aux, low, mid, high); // Merge AUX -> ARR
}

// --- SIMD Merge Functions short (using scalar fallback placeholders) ---
#ifdef __SSE4_1__
void merge_simd_1x_short(short *arr, const short *aux, int low, int mid, int high) { merge_scalar_1x_short(arr, aux, low, mid, high); }
void merge_simd_2x_short(short *arr, const short *aux, int low, int mid, int high) { merge_scalar_2x_short(arr, aux, low, mid, high); }
void merge_simd_4x_short(short *arr, const short *aux, int low, int mid, int high) { merge_scalar_4x_short(arr, aux, low, mid, high); }
void merge_simd_8x_short(short *arr, const short *aux, int low, int mid, int high) { merge_scalar_8x_short(arr, aux, low, mid, high); }
void merge_simd_16x_short(short *arr, const short *aux, int low, int mid, int high) { merge_scalar_16x_short(arr, aux, low, mid, high); }
#endif

// ====================================================================
// SECTION 2: TYPE 'int' IMPLEMENTATIONS
// ====================================================================

// --- Function Declarations for int ---
void initialize_array_int(int *arr, int size);
int is_sorted_int(const int *arr, int size);
void merge_scalar_1x_int(int *arr, const int *aux, int low, int mid, int high);
void merge_scalar_2x_int(int *arr, const int *aux, int low, int mid, int high);
void merge_scalar_4x_int(int *arr, const int *aux, int low, int mid, int high);
void merge_scalar_8x_int(int *arr, const int *aux, int low, int mid, int high);
void merge_scalar_16x_int(int *arr, const int *aux, int low, int mid, int high);
void merge_scalar_32x_int(int *arr, const int *aux, int low, int mid, int high);
void mergeSort_recursive_int(int *arr, int *aux, int low, int high, void (*merge_func)(int*, const int*, int, int, int));
void mergeSort_recursive_to_aux_int(int *arr, int *aux, int low, int high, void (*merge_func)(int*, const int*, int, int, int));
#ifdef __SSE4_1__
void merge_simd_1x_int(int *arr, const int *aux, int low, int mid, int high);
void merge_simd_2x_int(int *arr, const int *aux, int low, int mid, int high);
void merge_simd_4x_int(int *arr, const int *aux, int low, int mid, int high);
void merge_simd_8x_int(int *arr, const int *aux, int low, int mid, int high);
void merge_simd_16x_int(int *arr, const int *aux, int low, int mid, int high);
#endif

// --- Helper Functions int ---
void initialize_array_int(int *arr, int size) {
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        arr[i] = (int)(rand() % 1000000);
    }
}
int is_sorted_int(const int *arr, int size) {
    for (int i = 0; i < size - 1; i++) {
        if (arr[i] > arr[i + 1]) { return 0; }
    }
    return 1;
}

// --- Merge Functions (Scalar) int ---
void merge_scalar_1x_int(int *arr, const int *aux, int low, int mid, int high) {
    int i = low; int j = mid + 1;
    for (int k = low; k <= high; k++) {
        if (i > mid) { arr[k] = aux[j++]; }
        else if (j > high) { arr[k] = aux[i++]; }
        else if (aux[j] < aux[i]) { arr[k] = aux[j++]; }
        else { arr[k] = aux[i++]; }
    }
}
void merge_scalar_2x_int(int *arr, const int *aux, int low, int mid, int high) {
    int i = low; int j = mid + 1; int k;
    for (k = low; k <= high - 1; k += 2) {
        // UNROLL 1
        if (i > mid) { arr[k] = aux[j++]; } else if (j > high) { arr[k] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } else { arr[k] = aux[i++]; }
        // UNROLL 2
        if (i > mid) { arr[k+1] = aux[j++]; } else if (j > high) { arr[k+1] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k+1] = aux[j++]; } else { arr[k+1] = aux[i++]; }
    }
    for (; k <= high; k++) { // CLEANUP
        if (i > mid) { arr[k] = aux[j++]; } else if (j > high) { arr[k] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } else { arr[k] = aux[i++]; }
    }
}
void merge_scalar_4x_int(int *arr, const int *aux, int low, int mid, int high) {
    int i = low; int j = mid + 1; int k;
    for (k = low; k <= high - 3; k += 4) {
        for (int p = 0; p < 4; p++) {
            if (i > mid) { arr[k+p] = aux[j++]; } else if (j > high) { arr[k+p] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k+p] = aux[j++]; } else { arr[k+p] = aux[i++]; }
        }
    }
    for (; k <= high; k++) { // CLEANUP
        if (i > mid) { arr[k] = aux[j++]; } else if (j > high) { arr[k] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } else { arr[k] = aux[i++]; }
    }
}
void merge_scalar_8x_int(int *arr, const int *aux, int low, int mid, int high) {
    int i = low; int j = mid + 1; int k;
    for (k = low; k <= high - 7; k += 8) {
        for (int p = 0; p < 8; p++) {
            if (i > mid) { arr[k+p] = aux[j++]; } else if (j > high) { arr[k+p] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k+p] = aux[j++]; } else { arr[k+p] = aux[i++]; }
        }
    }
    for (; k <= high; k++) { // CLEANUP
        if (i > mid) { arr[k] = aux[j++]; } else if (j > high) { arr[k] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } else { arr[k] = aux[i++]; }
    }
}
void merge_scalar_16x_int(int *arr, const int *aux, int low, int mid, int high) {
    int i = low; int j = mid + 1; int k;
    for (k = low; k <= high - 15; k += 16) {
        for (int p = 0; p < 16; p++) {
            if (i > mid) { arr[k+p] = aux[j++]; } else if (j > high) { arr[k+p] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k+p] = aux[j++]; } else { arr[k+p] = aux[i++]; }
        }
    }
    for (; k <= high; k++) { // CLEANUP
        if (i > mid) { arr[k] = aux[j++]; } else if (j > high) { arr[k] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } else { arr[k] = aux[i++]; }
    }
}
void merge_scalar_32x_int(int *arr, const int *aux, int low, int mid, int high) {
    int i = low; int j = mid + 1; int k;
    for (k = low; k <= high - 31; k += 32) {
        for (int p = 0; p < 32; p++) {
            if (i > mid) { arr[k+p] = aux[j++]; } else if (j > high) { arr[k+p] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k+p] = aux[j++]; } else { arr[k+p] = aux[i++]; }
        }
    }
    for (; k <= high; k++) { // CLEANUP
        if (i > mid) { arr[k] = aux[j++]; } else if (j > high) { arr[k] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } else { arr[k] = aux[i++]; }
    }
}

// --- Recursive Sort Functions int ---
void mergeSort_recursive_to_aux_int(int *arr, int *aux, int low, int high, void (*merge_func)(int*, const int*, int, int, int)) {
    if (low >= high) { return; }
    int mid = low + (high - low) / 2;
    mergeSort_recursive_int(arr, aux, low, mid, merge_func);
    mergeSort_recursive_int(arr, aux, mid + 1, high, merge_func);
    merge_func(aux, arr, low, mid, high); // Merge ARR -> AUX
}
void mergeSort_recursive_int(int *arr, int *aux, int low, int high, void (*merge_func)(int*, const int*, int, int, int)) {
    if (low >= high) { return; }
    int mid = low + (high - low) / 2;
    mergeSort_recursive_to_aux_int(arr, aux, low, mid, merge_func);
    mergeSort_recursive_to_aux_int(arr, aux, mid + 1, high, merge_func);
    merge_func(arr, aux, low, mid, high); // Merge AUX -> ARR
}

// --- SIMD Merge Functions int (using scalar fallback placeholders) ---
#ifdef __SSE4_1__
void merge_simd_1x_int(int *arr, const int *aux, int low, int mid, int high) { merge_scalar_1x_int(arr, aux, low, mid, high); }
void merge_simd_2x_int(int *arr, const int *aux, int low, int mid, int high) { merge_scalar_2x_int(arr, aux, low, mid, high); }
void merge_simd_4x_int(int *arr, const int *aux, int low, int mid, int high) { merge_scalar_4x_int(arr, aux, low, mid, high); }
void merge_simd_8x_int(int *arr, const int *aux, int low, int mid, int high) { merge_scalar_8x_int(arr, aux, low, mid, high); }
void merge_simd_16x_int(int *arr, const int *aux, int low, int mid, int high) { merge_scalar_16x_int(arr, aux, low, mid, high); }
#endif

// ====================================================================
// SECTION 3: TYPE 'long' IMPLEMENTATIONS
// (NOTE: Assumes 64-bit long)
// ====================================================================

// --- Function Declarations for long ---
void initialize_array_long(long *arr, int size);
int is_sorted_long(const long *arr, int size);
void merge_scalar_1x_long(long *arr, const long *aux, int low, int mid, int high);
void merge_scalar_2x_long(long *arr, const long *aux, int low, int mid, int high);
void merge_scalar_4x_long(long *arr, const long *aux, int low, int mid, int high);
void merge_scalar_8x_long(long *arr, const long *aux, int low, int mid, int high);
void merge_scalar_16x_long(long *arr, const long *aux, int low, int mid, int high);
void merge_scalar_32x_long(long *arr, const long *aux, int low, int mid, int high);
void mergeSort_recursive_long(long *arr, long *aux, int low, int high, void (*merge_func)(long*, const long*, int, int, int));
void mergeSort_recursive_to_aux_long(long *arr, long *aux, int low, int high, void (*merge_func)(long*, const long*, int, int, int));
#ifdef __SSE4_1__
void merge_simd_1x_long(long *arr, const long *aux, int low, int mid, int high);
void merge_simd_2x_long(long *arr, const long *aux, int low, int mid, int high);
void merge_simd_4x_long(long *arr, const long *aux, int low, int mid, int high);
void merge_simd_8x_long(long *arr, const long *aux, int low, int mid, int high);
void merge_simd_16x_long(long *arr, const long *aux, int low, int mid, int high);
#endif

// --- Helper Functions long ---
void initialize_array_long(long *arr, int size) {
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        arr[i] = (long)((double)rand() / RAND_MAX * 1000000.0);
    }
}
int is_sorted_long(const long *arr, int size) {
    for (int i = 0; i < size - 1; i++) {
        if (arr[i] > arr[i + 1]) { return 0; }
    }
    return 1;
}

// --- Merge Functions (Scalar) long ---
void merge_scalar_1x_long(long *arr, const long *aux, int low, int mid, int high) {
    int i = low; int j = mid + 1;
    for (int k = low; k <= high; k++) {
        if (i > mid) { arr[k] = aux[j++]; }
        else if (j > high) { arr[k] = aux[i++]; }
        else if (aux[j] < aux[i]) { arr[k] = aux[j++]; }
        else { arr[k] = aux[i++]; }
    }
}
void merge_scalar_2x_long(long *arr, const long *aux, int low, int mid, int high) {
    int i = low; int j = mid + 1; int k;
    for (k = low; k <= high - 1; k += 2) {
        // UNROLL 1
        if (i > mid) { arr[k] = aux[j++]; } else if (j > high) { arr[k] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } else { arr[k] = aux[i++]; }
        // UNROLL 2
        if (i > mid) { arr[k+1] = aux[j++]; } else if (j > high) { arr[k+1] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k+1] = aux[j++]; } else { arr[k+1] = aux[i++]; }
    }
    for (; k <= high; k++) { // CLEANUP
        if (i > mid) { arr[k] = aux[j++]; } else if (j > high) { arr[k] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } else { arr[k] = aux[i++]; }
    }
}
void merge_scalar_4x_long(long *arr, const long *aux, int low, int mid, int high) {
    int i = low; int j = mid + 1; int k;
    for (k = low; k <= high - 3; k += 4) {
        for (int p = 0; p < 4; p++) {
            if (i > mid) { arr[k+p] = aux[j++]; } else if (j > high) { arr[k+p] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k+p] = aux[j++]; } else { arr[k+p] = aux[i++]; }
        }
    }
    for (; k <= high; k++) { // CLEANUP
        if (i > mid) { arr[k] = aux[j++]; } else if (j > high) { arr[k] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } else { arr[k] = aux[i++]; }
    }
}
void merge_scalar_8x_long(long *arr, const long *aux, int low, int mid, int high) {
    int i = low; int j = mid + 1; int k;
    for (k = low; k <= high - 7; k += 8) {
        for (int p = 0; p < 8; p++) {
            if (i > mid) { arr[k+p] = aux[j++]; } else if (j > high) { arr[k+p] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k+p] = aux[j++]; } else { arr[k+p] = aux[i++]; }
        }
    }
    for (; k <= high; k++) { // CLEANUP
        if (i > mid) { arr[k] = aux[j++]; } else if (j > high) { arr[k] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } else { arr[k] = aux[i++]; }
    }
}
void merge_scalar_16x_long(long *arr, const long *aux, int low, int mid, int high) {
    int i = low; int j = mid + 1; int k;
    for (k = low; k <= high - 15; k += 16) {
        for (int p = 0; p < 16; p++) {
            if (i > mid) { arr[k+p] = aux[j++]; } else if (j > high) { arr[k+p] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k+p] = aux[j++]; } else { arr[k+p] = aux[i++]; }
        }
    }
    for (; k <= high; k++) { // CLEANUP
        if (i > mid) { arr[k] = aux[j++]; } else if (j > high) { arr[k] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } else { arr[k] = aux[i++]; }
    }
}
void merge_scalar_32x_long(long *arr, const long *aux, int low, int mid, int high) {
    int i = low; int j = mid + 1; int k;
    for (k = low; k <= high - 31; k += 32) {
        for (int p = 0; p < 32; p++) {
            if (i > mid) { arr[k+p] = aux[j++]; } else if (j > high) { arr[k+p] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k+p] = aux[j++]; } else { arr[k+p] = aux[i++]; }
        }
    }
    for (; k <= high; k++) { // CLEANUP
        if (i > mid) { arr[k] = aux[j++]; } else if (j > high) { arr[k] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } else { arr[k] = aux[i++]; }
    }
}

// --- Recursive Sort Functions long ---
void mergeSort_recursive_to_aux_long(long *arr, long *aux, int low, int high, void (*merge_func)(long*, const long*, int, int, int)) {
    if (low >= high) { return; }
    int mid = low + (high - low) / 2;
    mergeSort_recursive_long(arr, aux, low, mid, merge_func);
    mergeSort_recursive_long(arr, aux, mid + 1, high, merge_func);
    merge_func(aux, arr, low, mid, high); // Merge ARR -> AUX
}
void mergeSort_recursive_long(long *arr, long *aux, int low, int high, void (*merge_func)(long*, const long*, int, int, int)) {
    if (low >= high) { return; }
    int mid = low + (high - low) / 2;
    mergeSort_recursive_to_aux_long(arr, aux, low, mid, merge_func);
    mergeSort_recursive_to_aux_long(arr, aux, mid + 1, high, merge_func);
    merge_func(arr, aux, low, mid, high); // Merge AUX -> ARR
}

// --- SIMD Merge Functions long (using scalar fallback placeholders) ---
#ifdef __SSE4_1__
void merge_simd_1x_long(long *arr, const long *aux, int low, int mid, int high) { merge_scalar_1x_long(arr, aux, low, mid, high); }
void merge_simd_2x_long(long *arr, const long *aux, int low, int mid, int high) { merge_scalar_2x_long(arr, aux, low, mid, high); }
void merge_simd_4x_long(long *arr, const long *aux, int low, int mid, int high) { merge_scalar_4x_long(arr, aux, low, mid, high); }
void merge_simd_8x_long(long *arr, const long *aux, int low, int mid, int high) { merge_scalar_8x_long(arr, aux, low, mid, high); }
void merge_simd_16x_long(long *arr, const long *aux, int low, int mid, int high) { merge_scalar_16x_long(arr, aux, low, mid, high); }
#endif


// ====================================================================
// SECTION 4: TYPE 'long long' IMPLEMENTATIONS
// ====================================================================

// --- Function Declarations for long long ---
void initialize_array_long_long(long long *arr, int size);
int is_sorted_long_long(const long long *arr, int size);
void merge_scalar_1x_long_long(long long *arr, const long long *aux, int low, int mid, int high);
void merge_scalar_2x_long_long(long long *arr, const long long *aux, int low, int mid, int high);
void merge_scalar_4x_long_long(long long *arr, const long long *aux, int low, int mid, int high);
void merge_scalar_8x_long_long(long long *arr, const long long *aux, int low, int mid, int high);
void merge_scalar_16x_long_long(long long *arr, const long long *aux, int low, int mid, int high);
void merge_scalar_32x_long_long(long long *arr, const long long *aux, int low, int mid, int high);
void mergeSort_recursive_long_long(long long *arr, long long *aux, int low, int high, void (*merge_func)(long long*, const long long*, int, int, int));
void mergeSort_recursive_to_aux_long_long(long long *arr, long long *aux, int low, int high, void (*merge_func)(long long*, const long long*, int, int, int));
#ifdef __SSE4_1__
void merge_simd_1x_long_long(long long *arr, const long long *aux, int low, int mid, int high);
void merge_simd_2x_long_long(long long *arr, const long long *aux, int low, int mid, int high);
void merge_simd_4x_long_long(long long *arr, const long long *aux, int low, int mid, int high);
void merge_simd_8x_long_long(long long *arr, const long long *aux, int low, int mid, int high);
void merge_simd_16x_long_long(long long *arr, const long long *aux, int low, int mid, int high);
#endif

// --- Helper Functions long long ---
void initialize_array_long_long(long long *arr, int size) {
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        arr[i] = (long long)((double)rand() / RAND_MAX * 1000000.0);
    }
}
int is_sorted_long_long(const long long *arr, int size) {
    for (int i = 0; i < size - 1; i++) {
        if (arr[i] > arr[i + 1]) { return 0; }
    }
    return 1;
}

// --- Merge Functions (Scalar) long long ---
void merge_scalar_1x_long_long(long long *arr, const long long *aux, int low, int mid, int high) {
    int i = low; int j = mid + 1;
    for (int k = low; k <= high; k++) {
        if (i > mid) { arr[k] = aux[j++]; }
        else if (j > high) { arr[k] = aux[i++]; }
        else if (aux[j] < aux[i]) { arr[k] = aux[j++]; }
        else { arr[k] = aux[i++]; }
    }
}
void merge_scalar_2x_long_long(long long *arr, const long long *aux, int low, int mid, int high) {
    int i = low; int j = mid + 1; int k;
    for (k = low; k <= high - 1; k += 2) {
        // UNROLL 1
        if (i > mid) { arr[k] = aux[j++]; } else if (j > high) { arr[k] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } else { arr[k] = aux[i++]; }
        // UNROLL 2
        if (i > mid) { arr[k+1] = aux[j++]; } else if (j > high) { arr[k+1] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k+1] = aux[j++]; } else { arr[k+1] = aux[i++]; }
    }
    for (; k <= high; k++) { // CLEANUP
        if (i > mid) { arr[k] = aux[j++]; } else if (j > high) { arr[k] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } else { arr[k] = aux[i++]; }
    }
}
void merge_scalar_4x_long_long(long long *arr, const long long *aux, int low, int mid, int high) {
    int i = low; int j = mid + 1; int k;
    for (k = low; k <= high - 3; k += 4) {
        for (int p = 0; p < 4; p++) {
            if (i > mid) { arr[k+p] = aux[j++]; } else if (j > high) { arr[k+p] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k+p] = aux[j++]; } else { arr[k+p] = aux[i++]; }
        }
    }
    for (; k <= high; k++) { // CLEANUP
        if (i > mid) { arr[k] = aux[j++]; } else if (j > high) { arr[k] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } else { arr[k] = aux[i++]; }
    }
}
void merge_scalar_8x_long_long(long long *arr, const long long *aux, int low, int mid, int high) {
    int i = low; int j = mid + 1; int k;
    for (k = low; k <= high - 7; k += 8) {
        for (int p = 0; p < 8; p++) {
            if (i > mid) { arr[k+p] = aux[j++]; } else if (j > high) { arr[k+p] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k+p] = aux[j++]; } else { arr[k+p] = aux[i++]; }
        }
    }
    for (; k <= high; k++) { // CLEANUP
        if (i > mid) { arr[k] = aux[j++]; } else if (j > high) { arr[k] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } else { arr[k] = aux[i++]; }
    }
}
void merge_scalar_16x_long_long(long long *arr, const long long *aux, int low, int mid, int high) {
    int i = low; int j = mid + 1; int k;
    for (k = low; k <= high - 15; k += 16) {
        for (int p = 0; p < 16; p++) {
            if (i > mid) { arr[k+p] = aux[j++]; } else if (j > high) { arr[k+p] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k+p] = aux[j++]; } else { arr[k+p] = aux[i++]; }
        }
    }
    for (; k <= high; k++) { // CLEANUP
        if (i > mid) { arr[k] = aux[j++]; } else if (j > high) { arr[k] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } else { arr[k] = aux[i++]; }
    }
}
void merge_scalar_32x_long_long(long long *arr, const long long *aux, int low, int mid, int high) {
    int i = low; int j = mid + 1; int k;
    for (k = low; k <= high - 31; k += 32) {
        for (int p = 0; p < 32; p++) {
            if (i > mid) { arr[k+p] = aux[j++]; } else if (j > high) { arr[k+p] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k+p] = aux[j++]; } else { arr[k+p] = aux[i++]; }
        }
    }
    for (; k <= high; k++) { // CLEANUP
        if (i > mid) { arr[k] = aux[j++]; } else if (j > high) { arr[k] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } else { arr[k] = aux[i++]; }
    }
}

// --- Recursive Sort Functions long long ---
void mergeSort_recursive_to_aux_long_long(long long *arr, long long *aux, int low, int high, void (*merge_func)(long long*, const long long*, int, int, int)) {
    if (low >= high) { return; }
    int mid = low + (high - low) / 2;
    mergeSort_recursive_long_long(arr, aux, low, mid, merge_func);
    mergeSort_recursive_long_long(arr, aux, mid + 1, high, merge_func);
    merge_func(aux, arr, low, mid, high); // Merge ARR -> AUX
}
void mergeSort_recursive_long_long(long long *arr, long long *aux, int low, int high, void (*merge_func)(long long*, const long long*, int, int, int)) {
    if (low >= high) { return; }
    int mid = low + (high - low) / 2;
    mergeSort_recursive_to_aux_long_long(arr, aux, low, mid, merge_func);
    mergeSort_recursive_to_aux_long_long(arr, aux, mid + 1, high, merge_func);
    merge_func(arr, aux, low, mid, high); // Merge AUX -> ARR
}

// --- SIMD Merge Functions long long (using scalar fallback placeholders) ---
#ifdef __SSE4_1__
void merge_simd_1x_long_long(long long *arr, const long long *aux, int low, int mid, int high) { merge_scalar_1x_long_long(arr, aux, low, mid, high); }
void merge_simd_2x_long_long(long long *arr, const long long *aux, int low, int mid, int high) { merge_scalar_2x_long_long(arr, aux, low, mid, high); }
void merge_simd_4x_long_long(long long *arr, const long long *aux, int low, int mid, int high) { merge_scalar_4x_long_long(arr, aux, low, mid, high); }
void merge_simd_8x_long_long(long long *arr, const long long *aux, int low, int mid, int high) { merge_scalar_8x_long_long(arr, aux, low, mid, high); }
void merge_simd_16x_long_long(long long *arr, const long long *aux, int low, int mid, int high) { merge_scalar_16x_long_long(arr, aux, low, mid, high); }
#endif

// ====================================================================
// SECTION 5: TYPE 'float' IMPLEMENTATIONS
// ====================================================================

// --- Function Declarations for float ---
void initialize_array_float(float *arr, int size);
int is_sorted_float(const float *arr, int size);
void merge_scalar_1x_float(float *arr, const float *aux, int low, int mid, int high);
void merge_scalar_2x_float(float *arr, const float *aux, int low, int mid, int high);
void merge_scalar_4x_float(float *arr, const float *aux, int low, int mid, int high);
void merge_scalar_8x_float(float *arr, const float *aux, int low, int mid, int high);
void merge_scalar_16x_float(float *arr, const float *aux, int low, int mid, int high);
void merge_scalar_32x_float(float *arr, const float *aux, int low, int mid, int high);
void mergeSort_recursive_float(float *arr, float *aux, int low, int high, void (*merge_func)(float*, const float*, int, int, int));
void mergeSort_recursive_to_aux_float(float *arr, float *aux, int low, int high, void (*merge_func)(float*, const float*, int, int, int));
#ifdef __SSE4_1__
void merge_simd_1x_float(float *arr, const float *aux, int low, int mid, int high);
void merge_simd_2x_float(float *arr, const float *aux, int low, int mid, int high);
void merge_simd_4x_float(float *arr, const float *aux, int low, int mid, int high);
void merge_simd_8x_float(float *arr, const float *aux, int low, int mid, int high);
void merge_simd_16x_float(float *arr, const float *aux, int low, int mid, int high);
#endif

// --- Helper Functions float ---
void initialize_array_float(float *arr, int size) {
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        arr[i] = (float)((double)rand() / RAND_MAX * 1000000.0);
    }
}
int is_sorted_float(const float *arr, int size) {
    for (int i = 0; i < size - 1; i++) {
        if (arr[i] > arr[i + 1]) { return 0; }
    }
    return 1;
}

// --- Merge Functions (Scalar) float ---
void merge_scalar_1x_float(float *arr, const float *aux, int low, int mid, int high) {
    int i = low; int j = mid + 1;
    for (int k = low; k <= high; k++) {
        if (i > mid) { arr[k] = aux[j++]; }
        else if (j > high) { arr[k] = aux[i++]; }
        else if (aux[j] < aux[i]) { arr[k] = aux[j++]; }
        else { arr[k] = aux[i++]; }
    }
}
void merge_scalar_2x_float(float *arr, const float *aux, int low, int mid, int high) {
    int i = low; int j = mid + 1; int k;
    for (k = low; k <= high - 1; k += 2) {
        // UNROLL 1
        if (i > mid) { arr[k] = aux[j++]; } else if (j > high) { arr[k] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } else { arr[k] = aux[i++]; }
        // UNROLL 2
        if (i > mid) { arr[k+1] = aux[j++]; } else if (j > high) { arr[k+1] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k+1] = aux[j++]; } else { arr[k+1] = aux[i++]; }
    }
    for (; k <= high; k++) { // CLEANUP
        if (i > mid) { arr[k] = aux[j++]; } else if (j > high) { arr[k] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } else { arr[k] = aux[i++]; }
    }
}
void merge_scalar_4x_float(float *arr, const float *aux, int low, int mid, int high) {
    int i = low; int j = mid + 1; int k;
    for (k = low; k <= high - 3; k += 4) {
        for (int p = 0; p < 4; p++) {
            if (i > mid) { arr[k+p] = aux[j++]; } else if (j > high) { arr[k+p] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k+p] = aux[j++]; } else { arr[k+p] = aux[i++]; }
        }
    }
    for (; k <= high; k++) { // CLEANUP
        if (i > mid) { arr[k] = aux[j++]; } else if (j > high) { arr[k] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } else { arr[k] = aux[i++]; }
    }
}
void merge_scalar_8x_float(float *arr, const float *aux, int low, int mid, int high) {
    int i = low; int j = mid + 1; int k;
    for (k = low; k <= high - 7; k += 8) {
        for (int p = 0; p < 8; p++) {
            if (i > mid) { arr[k+p] = aux[j++]; } else if (j > high) { arr[k+p] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k+p] = aux[j++]; } else { arr[k+p] = aux[i++]; }
        }
    }
    for (; k <= high; k++) { // CLEANUP
        if (i > mid) { arr[k] = aux[j++]; } else if (j > high) { arr[k] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } else { arr[k] = aux[i++]; }
    }
}
void merge_scalar_16x_float(float *arr, const float *aux, int low, int mid, int high) {
    int i = low; int j = mid + 1; int k;
    for (k = low; k <= high - 15; k += 16) {
        for (int p = 0; p < 16; p++) {
            if (i > mid) { arr[k+p] = aux[j++]; } else if (j > high) { arr[k+p] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k+p] = aux[j++]; } else { arr[k+p] = aux[i++]; }
        }
    }
    for (; k <= high; k++) { // CLEANUP
        if (i > mid) { arr[k] = aux[j++]; } else if (j > high) { arr[k] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } else { arr[k] = aux[i++]; }
    }
}
void merge_scalar_32x_float(float *arr, const float *aux, int low, int mid, int high) {
    int i = low; int j = mid + 1; int k;
    for (k = low; k <= high - 31; k += 32) {
        for (int p = 0; p < 32; p++) {
            if (i > mid) { arr[k+p] = aux[j++]; } else if (j > high) { arr[k+p] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k+p] = aux[j++]; } else { arr[k+p] = aux[i++]; }
        }
    }
    for (; k <= high; k++) { // CLEANUP
        if (i > mid) { arr[k] = aux[j++]; } else if (j > high) { arr[k] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } else { arr[k] = aux[i++]; }
    }
}

// --- Recursive Sort Functions float ---
void mergeSort_recursive_to_aux_float(float *arr, float *aux, int low, int high, void (*merge_func)(float*, const float*, int, int, int)) {
    if (low >= high) { return; }
    int mid = low + (high - low) / 2;
    mergeSort_recursive_float(arr, aux, low, mid, merge_func);
    mergeSort_recursive_float(arr, aux, mid + 1, high, merge_func);
    merge_func(aux, arr, low, mid, high); // Merge ARR -> AUX
}
void mergeSort_recursive_float(float *arr, float *aux, int low, int high, void (*merge_func)(float*, const float*, int, int, int)) {
    if (low >= high) { return; }
    int mid = low + (high - low) / 2;
    mergeSort_recursive_to_aux_float(arr, aux, low, mid, merge_func);
    mergeSort_recursive_to_aux_float(arr, aux, mid + 1, high, merge_func);
    merge_func(arr, aux, low, mid, high); // Merge AUX -> ARR
}

// --- SIMD Merge Functions float (using scalar fallback placeholders) ---
#ifdef __SSE4_1__
void merge_simd_1x_float(float *arr, const float *aux, int low, int mid, int high) { merge_scalar_1x_float(arr, aux, low, mid, high); }
void merge_simd_2x_float(float *arr, const float *aux, int low, int mid, int high) { merge_scalar_2x_float(arr, aux, low, mid, high); }
void merge_simd_4x_float(float *arr, const float *aux, int low, int mid, int high) { merge_scalar_4x_float(arr, aux, low, mid, high); }
void merge_simd_8x_float(float *arr, const float *aux, int low, int mid, int high) { merge_scalar_8x_float(arr, aux, low, mid, high); }
void merge_simd_16x_float(float *arr, const float *aux, int low, int mid, int high) { merge_scalar_16x_float(arr, aux, low, mid, high); }
#endif

// ====================================================================
// SECTION 6: TYPE 'double' IMPLEMENTATIONS
// ====================================================================

// --- Function Declarations for double ---
void initialize_array_double(double *arr, int size);
int is_sorted_double(const double *arr, int size);
void merge_scalar_1x_double(double *arr, const double *aux, int low, int mid, int high);
void merge_scalar_2x_double(double *arr, const double *aux, int low, int mid, int high);
void merge_scalar_4x_double(double *arr, const double *aux, int low, int mid, int high);
void merge_scalar_8x_double(double *arr, const double *aux, int low, int mid, int high);
void merge_scalar_16x_double(double *arr, const double *aux, int low, int mid, int high);
void merge_scalar_32x_double(double *arr, const double *aux, int low, int mid, int high);
void mergeSort_recursive_double(double *arr, double *aux, int low, int high, void (*merge_func)(double*, const double*, int, int, int));
void mergeSort_recursive_to_aux_double(double *arr, double *aux, int low, int high, void (*merge_func)(double*, const double*, int, int, int));
#ifdef __SSE4_1__
void merge_simd_1x_double(double *arr, const double *aux, int low, int mid, int high);
void merge_simd_2x_double(double *arr, const double *aux, int low, int mid, int high);
void merge_simd_4x_double(double *arr, const double *aux, int low, int mid, int high);
void merge_simd_8x_double(double *arr, const double *aux, int low, int mid, int high);
void merge_simd_16x_double(double *arr, const double *aux, int low, int mid, int high);
#endif

// --- Helper Functions double ---
void initialize_array_double(double *arr, int size) {
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        arr[i] = (double)((double)rand() / RAND_MAX * 1000000.0);
    }
}
int is_sorted_double(const double *arr, int size) {
    for (int i = 0; i < size - 1; i++) {
        if (arr[i] > arr[i + 1]) { return 0; }
    }
    return 1;
}

// --- Merge Functions (Scalar) double ---
void merge_scalar_1x_double(double *arr, const double *aux, int low, int mid, int high) {
    int i = low; int j = mid + 1;
    for (int k = low; k <= high; k++) {
        if (i > mid) { arr[k] = aux[j++]; }
        else if (j > high) { arr[k] = aux[i++]; }
        else if (aux[j] < aux[i]) { arr[k] = aux[j++]; }
        else { arr[k] = aux[i++]; }
    }
}
void merge_scalar_2x_double(double *arr, const double *aux, int low, int mid, int high) {
    int i = low; int j = mid + 1; int k;
    for (k = low; k <= high - 1; k += 2) {
        // UNROLL 1
        if (i > mid) { arr[k] = aux[j++]; } else if (j > high) { arr[k] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } else { arr[k] = aux[i++]; }
        // UNROLL 2
        if (i > mid) { arr[k+1] = aux[j++]; } else if (j > high) { arr[k+1] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k+1] = aux[j++]; } else { arr[k+1] = aux[i++]; }
    }
    for (; k <= high; k++) { // CLEANUP
        if (i > mid) { arr[k] = aux[j++]; } else if (j > high) { arr[k] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } else { arr[k] = aux[i++]; }
    }
}
void merge_scalar_4x_double(double *arr, const double *aux, int low, int mid, int high) {
    int i = low; int j = mid + 1; int k;
    for (k = low; k <= high - 3; k += 4) {
        for (int p = 0; p < 4; p++) {
            if (i > mid) { arr[k+p] = aux[j++]; } else if (j > high) { arr[k+p] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k+p] = aux[j++]; } else { arr[k+p] = aux[i++]; }
        }
    }
    for (; k <= high; k++) { // CLEANUP
        if (i > mid) { arr[k] = aux[j++]; } else if (j > high) { arr[k] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } else { arr[k] = aux[i++]; }
    }
}
void merge_scalar_8x_double(double *arr, const double *aux, int low, int mid, int high) {
    int i = low; int j = mid + 1; int k;
    for (k = low; k <= high - 7; k += 8) {
        for (int p = 0; p < 8; p++) {
            if (i > mid) { arr[k+p] = aux[j++]; } else if (j > high) { arr[k+p] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k+p] = aux[j++]; } else { arr[k+p] = aux[i++]; }
        }
    }
    for (; k <= high; k++) { // CLEANUP
        if (i > mid) { arr[k] = aux[j++]; } else if (j > high) { arr[k] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } else { arr[k] = aux[i++]; }
    }
}
void merge_scalar_16x_double(double *arr, const double *aux, int low, int mid, int high) {
    int i = low; int j = mid + 1; int k;
    for (k = low; k <= high - 15; k += 16) {
        for (int p = 0; p < 16; p++) {
            if (i > mid) { arr[k+p] = aux[j++]; } else if (j > high) { arr[k+p] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k+p] = aux[j++]; } else { arr[k+p] = aux[i++]; }
        }
    }
    for (; k <= high; k++) { // CLEANUP
        if (i > mid) { arr[k] = aux[j++]; } else if (j > high) { arr[k] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } else { arr[k] = aux[i++]; }
    }
}
void merge_scalar_32x_double(double *arr, const double *aux, int low, int mid, int high) {
    int i = low; int j = mid + 1; int k;
    for (k = low; k <= high - 31; k += 32) {
        for (int p = 0; p < 32; p++) {
            if (i > mid) { arr[k+p] = aux[j++]; } else if (j > high) { arr[k+p] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k+p] = aux[j++]; } else { arr[k+p] = aux[i++]; }
        }
    }
    for (; k <= high; k++) { // CLEANUP
        if (i > mid) { arr[k] = aux[j++]; } else if (j > high) { arr[k] = aux[i++]; } else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } else { arr[k] = aux[i++]; }
    }
}

// --- Recursive Sort Functions double ---
void mergeSort_recursive_to_aux_double(double *arr, double *aux, int low, int high, void (*merge_func)(double*, const double*, int, int, int)) {
    if (low >= high) { return; }
    int mid = low + (high - low) / 2;
    mergeSort_recursive_double(arr, aux, low, mid, merge_func);
    mergeSort_recursive_double(arr, aux, mid + 1, high, merge_func);
    merge_func(aux, arr, low, mid, high); // Merge ARR -> AUX
}
void mergeSort_recursive_double(double *arr, double *aux, int low, int high, void (*merge_func)(double*, const double*, int, int, int)) {
    if (low >= high) { return; }
    int mid = low + (high - low) / 2;
    mergeSort_recursive_to_aux_double(arr, aux, low, mid, merge_func);
    mergeSort_recursive_to_aux_double(arr, aux, mid + 1, high, merge_func);
    merge_func(arr, aux, low, mid, high); // Merge AUX -> ARR
}

// --- SIMD Merge Functions double (using scalar fallback placeholders) ---
#ifdef __SSE4_1__
void merge_simd_1x_double(double *arr, const double *aux, int low, int mid, int high) { merge_scalar_1x_double(arr, aux, low, mid, high); }
void merge_simd_2x_double(double *arr, const double *aux, int low, int mid, int high) { merge_scalar_2x_double(arr, aux, low, mid, high); }
void merge_simd_4x_double(double *arr, const double *aux, int low, int mid, int high) { merge_scalar_4x_double(arr, aux, low, mid, high); }
void merge_simd_8x_double(double *arr, const double *aux, int low, int mid, int high) { merge_scalar_8x_double(arr, aux, low, mid, high); }
void merge_simd_16x_double(double *arr, const double *aux, int low, int mid, int high) { merge_scalar_16x_double(arr, aux, low, mid, high); }
#endif

// ====================================================================
// BENCHMARK DRIVER (Main Function)
// ====================================================================

// Generic function to handle the benchmark test for a specific data type
void run_benchmark_test(void* data_ptr, void* aux_ptr, size_t data_size,
                        const char* type_name, int vw,
                        void (*init_func)(void*, int),
                        int (*is_sorted_func)(const void*, int),
                        void (*recursive_sort_func)(void*, void*, int, int, void(*)(void*, const void*, int, int, int)),
                        void* scalar_funcs[], const char* scalar_names[], int scalar_count,
                        void* simd_funcs[], const char* simd_names[], int simd_count) {

    printf("\n--- Testing Data Type: %s (Size: %lu bytes) ---\n", type_name, data_size);

    // SCALAR TESTS
    printf("  [SCALAR] Unrolling (Loop Increments): \n");
    for (int f = 0; f < scalar_count; f++) {
        double total_time = 0;
        void (*merge_func)(void*, const void*, int, int, int) = scalar_funcs[f];
        for (int r = 0; r < RUNS; r++) {
            init_func(data_ptr, N);
            // Copy data to aux before starting the first recursive call if the recursive_sort_func starts with AUX -> ARR
            // The macro implementation handles this via the alternating recursive calls.
            clock_t start = clock();
            recursive_sort_func(data_ptr, aux_ptr, 0, N - 1, merge_func);
            clock_t end = clock();
            total_time += (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;
        }
        if (!is_sorted_func(data_ptr, N)) {
            printf("    [ERROR] Scalar %s failed to sort.\n", scalar_names[f]);
        } else {
            printf("    - Scalar '%s' Time: %.2f ms (Avg over %d runs)\n", scalar_names[f], total_time / RUNS, RUNS);
        }
    }

    // SIMD TESTS
    #ifdef __SSE4_1__
        printf("  [VECTOR] Unrolling (Elements/Iter: VW=%d):\n", vw);
        for (int f = 0; f < simd_count; f++) {
            double total_time = 0;
            void (*merge_func)(void*, const void*, int, int, int) = simd_funcs[f];
            int unroll_factor = (f == 0 ? 1 : (f == 1 ? 2 : (f == 2 ? 4 : (f == 3 ? 8 : 16))));
            int elements_per_iter = unroll_factor * vw;
            for (int r = 0; r < RUNS; r++) {
                init_func(data_ptr, N);
                clock_t start = clock();
                recursive_sort_func(data_ptr, aux_ptr, 0, N - 1, merge_func);
                clock_t end = clock();
                total_time += (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;
            }
            if (!is_sorted_func(data_ptr, N)) {
                printf("    [ERROR] SIMD %s failed to sort.\n", simd_names[f]);
            } else {
                printf("    - Vector '%s' Time (Proc %d elements): %.2f ms\n", simd_names[f], elements_per_iter, total_time / RUNS);
            }
        }
    #else
        printf("  [VECTOR] SIMD Tests Skipped (Compile with -msse4.1 or similar).\n");
    #endif
}


int main() {
    printf("--- Starting Comprehensive Merge Sort Benchmark (N=%d, RUNS=%d) ---\n", N, RUNS);

    const char *scalar_names[] = { "1x", "2x", "4x", "8x", "16x", "32x" };
    const char *simd_names[] = { "1x", "2x", "4x", "8x", "16x" };
    const int scalar_count = 6;
    const int simd_count = 5;


    // --- Test short ---
    short *data_short = (short*)malloc(N * sizeof(short));
    short *aux_short = (short*)malloc(N * sizeof(short));
    if (!data_short || !aux_short) { perror("Memory allocation failed for short"); return 1; }

    void (*scalar_funcs_short[])(short*, const short*, int, int, int) = {
        merge_scalar_1x_short, merge_scalar_2x_short, merge_scalar_4x_short,
        merge_scalar_8x_short, merge_scalar_16x_short, merge_scalar_32x_short
    };
    #ifdef __SSE4_1__
    void (*simd_funcs_short[])(short*, const short*, int, int, int) = {
        merge_simd_1x_short, merge_simd_2x_short, merge_simd_4x_short,
        merge_simd_8x_short, merge_simd_16x_short
    };
    #endif
    run_benchmark_test(data_short, aux_short, sizeof(short), "short", VW_short,
                       (void (*)(void*, int))initialize_array_short,
                       (int (*)(const void*, int))is_sorted_short,
                       (void (*)(void*, void*, int, int, void(*)(void*, const void*, int, int, int)))mergeSort_recursive_short,
                       (void**)scalar_funcs_short, scalar_names, scalar_count,
                       (void**)simd_funcs_short, simd_names, simd_count);
    free(data_short); free(aux_short);


    // --- Test int ---
    int *data_int = (int*)malloc(N * sizeof(int));
    int *aux_int = (int*)malloc(N * sizeof(int));
    if (!data_int || !aux_int) { perror("Memory allocation failed for int"); return 1; }

    void (*scalar_funcs_int[])(int*, const int*, int, int, int) = {
        merge_scalar_1x_int, merge_scalar_2x_int, merge_scalar_4x_int,
        merge_scalar_8x_int, merge_scalar_16x_int, merge_scalar_32x_int
    };
    #ifdef __SSE4_1__
    void (*simd_funcs_int[])(int*, const int*, int, int, int) = {
        merge_simd_1x_int, merge_simd_2x_int, merge_simd_4x_int,
        merge_simd_8x_int, merge_simd_16x_int
    };
    #endif
    run_benchmark_test(data_int, aux_int, sizeof(int), "int", VW_int,
                       (void (*)(void*, int))initialize_array_int,
                       (int (*)(const void*, int))is_sorted_int,
                       (void (*)(void*, void*, int, int, void(*)(void*, const void*, int, int, int)))mergeSort_recursive_int,
                       (void**)scalar_funcs_int, scalar_names, scalar_count,
                       (void**)simd_funcs_int, simd_names, simd_count);
    free(data_int); free(aux_int);


    // --- Test long ---
    long *data_long = (long*)malloc(N * sizeof(long));
    long *aux_long = (long*)malloc(N * sizeof(long));
    if (!data_long || !aux_long) { perror("Memory allocation failed for long"); return 1; }

    void (*scalar_funcs_long[])(long*, const long*, int, int, int) = {
        merge_scalar_1x_long, merge_scalar_2x_long, merge_scalar_4x_long,
        merge_scalar_8x_long, merge_scalar_16x_long, merge_scalar_32x_long
    };
    #ifdef __SSE4_1__
    void (*simd_funcs_long[])(long*, const long*, int, int, int) = {
        merge_simd_1x_long, merge_simd_2x_long, merge_simd_4x_long,
        merge_simd_8x_long, merge_simd_16x_long
    };
    #endif
    run_benchmark_test(data_long, aux_long, sizeof(long), "long", VW_long,
                       (void (*)(void*, int))initialize_array_long,
                       (int (*)(const void*, int))is_sorted_long,
                       (void (*)(void*, void*, int, int, void(*)(void*, const void*, int, int, int)))mergeSort_recursive_long,
                       (void**)scalar_funcs_long, scalar_names, scalar_count,
                       (void**)simd_funcs_long, simd_names, simd_count);
    free(data_long); free(aux_long);


    // --- Test long long ---
    long long *data_long_long = (long long*)malloc(N * sizeof(long long));
    long long *aux_long_long = (long long*)malloc(N * sizeof(long long));
    if (!data_long_long || !aux_long_long) { perror("Memory allocation failed for long long"); return 1; }

    void (*scalar_funcs_long_long[])(long long*, const long long*, int, int, int) = {
        merge_scalar_1x_long_long, merge_scalar_2x_long_long, merge_scalar_4x_long_long,
        merge_scalar_8x_long_long, merge_scalar_16x_long_long, merge_scalar_32x_long_long
    };
    #ifdef __SSE4_1__
    void (*simd_funcs_long_long[])(long long*, const long long*, int, int, int) = {
        merge_simd_1x_long_long, merge_simd_2x_long_long, merge_simd_4x_long_long,
        merge_simd_8x_long_long, merge_simd_16x_long_long
    };
    #endif
    run_benchmark_test(data_long_long, aux_long_long, sizeof(long long), "long long", VW_long_long,
                       (void (*)(void*, int))initialize_array_long_long,
                       (int (*)(const void*, int))is_sorted_long_long,
                       (void (*)(void*, void*, int, int, void(*)(void*, const void*, int, int, int)))mergeSort_recursive_long_long,
                       (void**)scalar_funcs_long_long, scalar_names, scalar_count,
                       (void**)simd_funcs_long_long, simd_names, simd_count);
    free(data_long_long); free(aux_long_long);


    // --- Test float ---
    float *data_float = (float*)malloc(N * sizeof(float));
    float *aux_float = (float*)malloc(N * sizeof(float));
    if (!data_float || !aux_float) { perror("Memory allocation failed for float"); return 1; }

    void (*scalar_funcs_float[])(float*, const float*, int, int, int) = {
        merge_scalar_1x_float, merge_scalar_2x_float, merge_scalar_4x_float,
        merge_scalar_8x_float, merge_scalar_16x_float, merge_scalar_32x_float
    };
    #ifdef __SSE4_1__
    void (*simd_funcs_float[])(float*, const float*, int, int, int) = {
        merge_simd_1x_float, merge_simd_2x_float, merge_simd_4x_float,
        merge_simd_8x_float, merge_simd_16x_float
    };
    #endif
    run_benchmark_test(data_float, aux_float, sizeof(float), "float", VW_float,
                       (void (*)(void*, int))initialize_array_float,
                       (int (*)(const void*, int))is_sorted_float,
                       (void (*)(void*, void*, int, int, void(*)(void*, const void*, int, int, int)))mergeSort_recursive_float,
                       (void**)scalar_funcs_float, scalar_names, scalar_count,
                       (void**)simd_funcs_float, simd_names, simd_count);
    free(data_float); free(aux_float);


    // --- Test double ---
    double *data_double = (double*)malloc(N * sizeof(double));
    double *aux_double = (double*)malloc(N * sizeof(double));
    if (!data_double || !aux_double) { perror("Memory allocation failed for double"); return 1; }

    void (*scalar_funcs_double[])(double*, const double*, int, int, int) = {
        merge_scalar_1x_double, merge_scalar_2x_double, merge_scalar_4x_double,
        merge_scalar_8x_double, merge_scalar_16x_double, merge_scalar_32x_double
    };
    #ifdef __SSE4_1__
    void (*simd_funcs_double[])(double*, const double*, int, int, int) = {
        merge_simd_1x_double, merge_simd_2x_double, merge_simd_4x_double,
        merge_simd_8x_double, merge_simd_16x_double
    };
    #endif
    run_benchmark_test(data_double, aux_double, sizeof(double), "double", VW_double,
                       (void (*)(void*, int))initialize_array_double,
                       (int (*)(const void*, int))is_sorted_double,
                       (void (*)(void*, void*, int, int, void(*)(void*, const void*, int, int, int)))mergeSort_recursive_double,
                       (void**)scalar_funcs_double, scalar_names, scalar_count,
                       (void**)simd_funcs_double, simd_names, simd_count);
    free(data_double); free(aux_double);


    printf("\n--- Benchmark Complete ---\n");
    return 0;
}