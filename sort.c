#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <string.h> // For memcpy
// Define POSIX feature test macro to ensure availability of clock_gettime and CLOCK_MONOTONIC
#define _POSIX_C_SOURCE 199309L

// Define the clock to use for high-resolution timing
#define TIME_CLOCK CLOCK_MONOTONIC
// Use UL suffix to ensure the literal is treated as size_t compatible type (unsigned long)
#define ARRAY_SIZE 500000UL 

// --- Function Pointer for Optimal Merge Selection ---
typedef void (*MergeFunc)(int[], int[], size_t, size_t, size_t);

// --- Timing Utilities ---

/**
 * @brief Measures time difference in milliseconds.
 */
double time_diff_ms(const struct timespec *start, const struct timespec *end) {
    long seconds = end->tv_sec - start->tv_sec;
    long nanoseconds = end->tv_nsec - start->tv_nsec;
    
    if (nanoseconds < 0) {
        seconds--;
        nanoseconds += 1000000000;
    }
    
    return (double)seconds * 1000.0 + (double)nanoseconds / 1000000.0;
}


// --- Merge Implementations for Benchmarking ---

/**
 * @brief Base Merge function (1x unrolled/standard loop).
 */
static void merge_base(int arr[], int aux[], size_t low, size_t mid, size_t high) {
    // 1. Copy to auxiliary array
    for (size_t k = low; k <= high; k++) {
        aux[k] = arr[k];
    }

    size_t i = low;      // Left half index
    size_t j = mid + 1;  // Right half index
    
    // 2. Merge back from aux to arr
    for (size_t k = low; k <= high; k++) {
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

/**
 * @brief 4x Loop Unrolled Merge function.
 * * Optimization attempt to reduce loop overhead and increase data flow.
 * * Cache Awareness: By processing 4 elements at a time, we increase instruction
 * level parallelism and maximize the usage of data already brought into the cache.
 */
static void merge_unroll_4x(int arr[], int aux[], size_t low, size_t mid, size_t high) {
    // Copy to auxiliary array - simple loop is often best here
    for (size_t k = low; k <= high; k++) {
        aux[k] = arr[k];
    }

    size_t i = low;
    size_t j = mid + 1;
    size_t k = low;

    // The main loop processes blocks of 4
    for (; k <= high - 3; k += 4) {
        // Step 1
        if (i > mid) { arr[k] = aux[j++]; } 
        else if (j > high) { arr[k] = aux[i++]; } 
        else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } 
        else { arr[k] = aux[i++]; }
        
        // Step 2
        if (i > mid) { arr[k+1] = aux[j++]; } 
        else if (j > high) { arr[k+1] = aux[i++]; } 
        else if (aux[j] < aux[i]) { arr[k+1] = aux[j++]; } 
        else { arr[k+1] = aux[i++]; }

        // Step 3
        if (i > mid) { arr[k+2] = aux[j++]; } 
        else if (j > high) { arr[k+2] = aux[i++]; } 
        else if (aux[j] < aux[i]) { arr[k+2] = aux[j++]; } 
        else { arr[k+2] = aux[i++]; }
        
        // Step 4
        if (i > mid) { arr[k+3] = aux[j++]; } 
        else if (j > high) { arr[k+3] = aux[i++]; } 
        else if (aux[j] < aux[i]) { arr[k+3] = aux[j++]; } 
        else { arr[k+3] = aux[i++]; }
    }
    
    // Handle the remaining elements (tail part)
    for (; k <= high; k++) {
        if (i > mid) { arr[k] = aux[j++]; } 
        else if (j > high) { arr[k] = aux[i++]; } 
        else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } 
        else { arr[k] = aux[i++]; }
    }
}


/**
 * @brief The common recursive Merge Sort function used for both benchmarking and final sort.
 * * This function takes the specific merge implementation (MergeFunc) to use.
 * @param arr The main array.
 * @param aux The shared auxiliary array (cache optimization).
 * @param low The starting index.
 * @param high The ending index.
 * @param merge_func The function pointer to the selected merge variant.
 */
static void mergeSort_recursive(int arr[], int aux[], size_t low, size_t high, MergeFunc merge_func) {
    if (high <= low) return;
    
    size_t mid = low + (high - low) / 2;
    
    // Recurse, passing the merge function down
    mergeSort_recursive(arr, aux, low, mid, merge_func);
    mergeSort_recursive(arr, aux, mid + 1, high, merge_func);
    
    // Perform the merge using the specific variant
    merge_func(arr, aux, low, mid, high);
}


/**
 * @brief Utility to run a function and measure its time.
 * @param m_func The MergeFunc to test.
 * @param data Array to sort.
 * @param aux Auxiliary array.
 * @param size Size of the array.
 * @return double Execution time in milliseconds.
 */
double measure_sort_time(MergeFunc m_func, int *data, int *aux, size_t size) {
    struct timespec start, end;
    
    if (clock_gettime(TIME_CLOCK, &start) == -1) {
        perror("clock_gettime start failed");
        return -1.0;
    }
    
    // Call the standard recursive sort function with the merge variant to be tested
    mergeSort_recursive(data, aux, 0, size - 1, m_func); 

    if (clock_gettime(TIME_CLOCK, &end) == -1) {
        perror("clock_gettime end failed");
        return -1.0;
    }
    
    return time_diff_ms(&start, &end);
}


/**
 * @brief Public interface for the final cache-aware Merge Sort execution.
 * @param arr The array to be sorted.
 * @param n The number of elements in the array.
 * @param optimal_merge_func The function pointer selected by the benchmark.
 */
void mergeSort_final(int arr[], size_t n, MergeFunc optimal_merge_func) {
    if (n < 2) return;
    
    // Allocate the auxiliary array once (The key cache-aware optimization)
    int *aux = (int *)malloc(n * sizeof(int));
    if (aux == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for auxiliary array.\n");
        return;
    }
    
    // Start the recursive sort using the optimal function
    mergeSort_recursive(arr, aux, 0, n - 1, optimal_merge_func);
    
    free(aux);
}

// --- Main Execution and Benchmarking ---

int main() {
    // The master array, which will hold the final sorted data
    int *master_data = (int *)malloc(ARRAY_SIZE * sizeof(int));
    // A temporary array for benchmarking each variant
    int *test_data = (int *)malloc(ARRAY_SIZE * sizeof(int));
    int *aux_array = (int *)malloc(ARRAY_SIZE * sizeof(int));
    
    if (master_data == NULL || test_data == NULL || aux_array == NULL) {
        fprintf(stderr, "Error: Failed to allocate necessary arrays.\n");
        return 1;
    }

    // Initialize master array with random data
    printf("Initializing master array with %zu random integers for testing...\n", ARRAY_SIZE);
    srand((unsigned int)time(NULL));
    for (size_t i = 0; i < ARRAY_SIZE; i++) {
        master_data[i] = rand();
    }
    
    printf("\n--- Starting Cache Unrolling Optimization Benchmark ---\n");

    // Array of merge variants to test
    struct {
        const char *name;
        MergeFunc func;
        double time_ms;
    } merge_variants[] = {
        {"1x Base Merge", merge_base, 0.0},
        {"4x Unroll Merge", merge_unroll_4x, 0.0}
        // Add more unrolling variants here (e.g., 8x)
    };
    
    size_t num_variants = sizeof(merge_variants) / sizeof(merge_variants[0]);
    MergeFunc optimal_merge_func = NULL;
    double best_time = 999999.0;
    
    for (size_t i = 0; i < num_variants; i++) {
        // Copy the pristine master data to the test array
        memcpy(test_data, master_data, ARRAY_SIZE * sizeof(int));
        
        // Measure time for the current variant
        merge_variants[i].time_ms = measure_sort_time(
            merge_variants[i].func, 
            test_data, 
            aux_array, 
            ARRAY_SIZE
        );
        
        printf("- Variant '%s' Time: %.2f ms\n", 
               merge_variants[i].name, 
               merge_variants[i].time_ms);
        
        // Track the fastest one
        if (merge_variants[i].time_ms < best_time) {
            best_time = merge_variants[i].time_ms;
            optimal_merge_func = merge_variants[i].func;
        }
    }
    
    const char *best_name = (optimal_merge_func == merge_base) ? "1x Base Merge" : "4x Unroll Merge";

    printf("\n--- Optimal Merge Variant Selected ---\n");
    printf("Fastest variant: %s (%.2f ms)\n", best_name, best_time);
    printf("Starting final sort using the optimal function...\n");

    // --- Final Execution ---
    
    struct timespec final_start, final_end;
    if (clock_gettime(TIME_CLOCK, &final_start) == -1) {
        perror("clock_gettime final start failed");
        goto cleanup;
    }
    
    // Execute the final sort on the master data using the determined optimal function
    mergeSort_final(master_data, ARRAY_SIZE, optimal_merge_func);

    if (clock_gettime(TIME_CLOCK, &final_end) == -1) {
        perror("clock_gettime final end failed");
        goto cleanup;
    }
    
    double final_duration_ms = time_diff_ms(&final_start, &final_end);

    printf("Final Master Sort Complete.\n");
    printf("--- Final Cache-Aware Timing Result ---\n");
    printf("Elements Sorted: %zu\n", ARRAY_SIZE);
    printf("Total Execution Time (using %s): %.2f ms\n", best_name, final_duration_ms);
    printf("---------------------------------------\n");
    
cleanup:
    free(master_data);
    free(test_data);
    free(aux_array);
    return 0;
}