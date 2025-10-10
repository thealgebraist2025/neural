#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// --- Configuration ---
#define N (1 << 20) // 1,048,576 elements for a typical benchmark size
typedef int DataType;
// ---------------------

// Type definition for the merge function pointer
typedef void (*MergeFunction)(DataType*, const DataType*, int, int, int);

// ====================================================================
// STANDARD MERGE FUNCTION (Baseline)
// ====================================================================

/**
 * @brief Standard (non-optimized) merge function.
 * * @param arr The destination array (final merge result is written here).
 * @param aux The auxiliary array (contains the two sorted subarrays to be merged).
 * @param low The starting index of the merge section.
 * @param mid The midpoint index.
 * @param high The ending index of the merge section.
 */
void merge_base(DataType *arr, const DataType *aux, int low, int mid, int high) {
    int i = low;
    int j = mid + 1;

    // 1. Copy data from arr to aux (This is done in the recursive driver 
    //    but often done here in standard in-place merge sort variations.)
    //    We'll skip the copy here, assuming the recursive driver handles swapping/copying
    //    or that 'aux' already holds the data to be merged.
    
    // For simplicity and correctness with a single aux array:
    // We assume the data to be merged is currently in 'arr', 
    // and we copy it to 'aux' before merging back into 'arr'.

    // Step 1: Copy data to aux array (critical for in-place Merge Sort variants)
    // NOTE: This copy should logically happen in the recursive call BEFORE the merge.
    // However, to satisfy the function signature where 'aux' holds the data to merge, 
    // we assume the data from 'low' to 'high' is copied to 'aux' before this call.
    // For this example, we'll assume the driver function ensures aux holds the data.

    for (int k = low; k <= high; k++) {
        // Section 1: left half is exhausted (i > mid)
        if (i > mid) {
            arr[k] = aux[j++];
        } 
        // Section 2: right half is exhausted (j > high)
        else if (j > high) {
            arr[k] = aux[i++];
        } 
        // Section 3: aux[j] is smaller
        else if (aux[j] < aux[i]) {
            arr[k] = aux[j++];
        } 
        // Section 4: aux[i] is smaller or equal
        else {
            arr[k] = aux[i++];
        }
    }
}


// ====================================================================
// FIXED UNROLLED MERGE FUNCTION (4x)
// ====================================================================

/**
 * @brief Cache-optimized 4x unrolled merge function (FIXED).
 * * The crash occurs because 'k' exceeds 'high', causing an illegal write to 'arr[k]'.
 * This fixed version ensures the unrolled loop (k <= high - 3) strictly avoids 
 * the final 0, 1, 2, or 3 elements, which are then handled by the robust 
 * standard merge logic in the cleanup loop.
 * * @param arr The destination array.
 * @param aux The auxiliary array containing the data to be merged.
 * @param low The starting index.
 * @param mid The midpoint index.
 * @param high The ending index.
 */
void merge_unroll_4x(DataType *arr, const DataType *aux, int low, int mid, int high) {
    // Indices for the two halves in the auxiliary array
    int i = low;
    int j = mid + 1;
    int k; // index for writing back to 'arr'
    
    // --- 1. Main Unrolled Loop (Processes blocks of 4) ---
    // The loop condition is strictly 'k <= high - 3'. 
    // This is safe because it guarantees that there are at least 4 more slots 
    // available to be written to in the 'arr' array before 'high'.
    // The previous bug likely occurred because the inner block of 4 comparisons 
    // failed to check for the exhaustion of 'i' or 'j' for all 4 steps,
    // or the 'k' index ran too far in the cleanup.
    
    // We must ensure the boundary checks (i > mid) and (j > high) 
    // are robustly checked for *each* of the 4 elements being processed.
    
    for (k = low; k <= high - 3; k += 4) {
        
        // --- Element k (Original line 83 crash location) ---
        if (i > mid) { arr[k] = aux[j++]; } 
        else if (j > high) { arr[k] = aux[i++]; } 
        else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } 
        else { arr[k] = aux[i++]; }

        // --- Element k+1 ---
        if (i > mid) { arr[k+1] = aux[j++]; } 
        else if (j > high) { arr[k+1] = aux[i++]; } 
        else if (aux[j] < aux[i]) { arr[k+1] = aux[j++]; } 
        else { arr[k+1] = aux[i++]; }
        
        // --- Element k+2 ---
        if (i > mid) { arr[k+2] = aux[j++]; } 
        else if (j > high) { arr[k+2] = aux[i++]; } 
        else if (aux[j] < aux[i]) { arr[k+2] = aux[j++]; } 
        else { arr[k+2] = aux[i++]; }

        // --- Element k+3 ---
        if (i > mid) { arr[k+3] = aux[j++]; } 
        else if (j > high) { arr[k+3] = aux[i++]; } 
        else if (aux[j] < aux[i]) { arr[k+3] = aux[j++]; } 
        else { arr[k+3] = aux[i++]; }
    }
    
    // --- 2. Cleanup Loop (Handles the remaining 0, 1, 2, or 3 elements) ---
    // This loop uses the standard, robust merge logic until 'k' reaches 'high'.
    // Because the main loop guaranteed k < high - 3 before the last increment, 
    // k will be at most high - 3 + 4 = high + 1 when it starts the final cleanup.
    // The condition 'k <= high' is now safe.
    for (; k <= high; k++) {
        if (i > mid) {
            arr[k] = aux[j++];
        } else if (j > high) {
            arr[k] = aux[i++];
        } else if (aux[j] < aux[i]) {
            arr[k] = aux[j++];
        } else {
            arr[k] = aux[i++];
        }
    }
}


// ====================================================================
// MERGE SORT RECURSIVE DRIVER
// ====================================================================

/**
 * @brief Recursive driver for Merge Sort.
 * * @param arr The array currently holding the data to be copied and merged into.
 * @param aux The auxiliary array used for temporary storage.
 * @param low Starting index.
 * @param high Ending index.
 * @param merge_func The specific merge function to use (base or unrolled).
 */
void mergeSort_recursive(DataType *arr, DataType *aux, int low, int high, MergeFunction merge_func) {
    if (low >= high) {
        return;
    }

    int mid = low + (high - low) / 2;

    // 1. Copy data from 'arr' (current state) to 'aux' for the two recursive calls
    // Note: The roles of 'arr' and 'aux' swap in some optimized versions to avoid copies.
    // For this simple version, we copy 'arr' to 'aux' only once at the start of the recursion
    // and rely on the merge function to copy back from 'aux' to 'arr'.
    
    // Standard implementation: Copy data from 'arr' to 'aux' for the merge step.
    for(int k = low; k <= high; k++) {
        aux[k] = arr[k];
    }
    
    // Recurse on the left half. (Note: Here we swap roles for efficiency)
    // To implement the copy-free swap (Knuth's method), the recursive calls should swap arr/aux:
    // mergeSort_recursive(aux, arr, low, mid, merge_func); // left half written to arr
    // mergeSort_recursive(aux, arr, mid + 1, high, merge_func); // right half written to arr
    // Then the final merge should be from aux to arr.

    // To keep it simple and safe (and match the structure of the LLDB trace):
    // Recurse on current array, which relies on the merge step to write to 'arr'
    mergeSort_recursive(arr, aux, low, mid, merge_func);
    mergeSort_recursive(arr, aux, mid + 1, high, merge_func);

    // Merge the two sorted halves (now in 'arr') back into 'arr' using 'aux' as source
    // Since the recursive calls returned, the current data to be merged is in 'arr'.
    // We must copy it to 'aux' before merging, as the merge function expects 'aux' as source.
    for (int k = low; k <= high; k++) {
        aux[k] = arr[k];
    }

    merge_func(arr, aux, low, mid, high);
}

// ====================================================================
// MAIN DRIVER AND BENCHMARK
// ====================================================================

/**
 * @brief Initializes an array with random values.
 * @param arr The array to initialize.
 * @param size The size of the array.
 */
void initialize_array(DataType *arr, int size) {
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % 1000000;
    }
}

/**
 * @brief Checks if the array is sorted.
 */
int is_sorted(const DataType *arr, int size) {
    for (int i = 0; i < size - 1; i++) {
        if (arr[i] > arr[i + 1]) {
            return 0;
        }
    }
    return 1;
}

int main() {
    DataType *data = (DataType*)malloc(N * sizeof(DataType));
    DataType *aux = (DataType*)malloc(N * sizeof(DataType));

    if (!data || !aux) {
        perror("Memory allocation failed");
        return 1;
    }
    
    printf("--- Starting Cache Unrolling Optimization Benchmark ---\n");

    // --- Benchmark 1: Base Merge ---
    initialize_array(data, N);
    
    clock_t start = clock();
    mergeSort_recursive(data, aux, 0, N - 1, merge_base);
    clock_t end = clock();
    
    double time_base = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;

    if (!is_sorted(data, N)) {
        printf("Error: Base Merge failed to sort the array!\n");
    }
    printf("- Variant '1x Base Merge' Time: %.2f ms\n", time_base);

    // --- Benchmark 2: Unrolled Merge (FIXED) ---
    initialize_array(data, N);
    
    start = clock();
    // This call should now be safe and pass without segfault
    mergeSort_recursive(data, aux, 0, N - 1, merge_unroll_4x);
    end = clock();
    
    double time_unrolled = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;

    if (!is_sorted(data, N)) {
        printf("Error: Unrolled Merge failed to sort the array!\n");
    }
    printf("- Variant '4x Unrolled Merge' Time: %.2f ms (Fixed)\n", time_unrolled);


    free(data);
    free(aux);
    
    return 0;
}