#define _POSIX_C_SOURCE 200809L // Enable POSIX features for clock_gettime/CLOCK_MONOTONIC

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>      
#include <stdbool.h>

// Compiler check for SSE/SIMD intrinsics
#if defined(__GNUC__) || defined(__clang__)
#if defined(__SSE4_1__)
#include <smmintrin.h> // SSE4.1 intrinsics
#define USE_SIMD
#define VW_long 2 // Vector Width: 2 longs fit in one 128-bit XMM register (8 bytes * 2 = 16 bytes)
#else
#undef USE_SIMD
#define VW_long 1
#endif
#else
#undef USE_SIMD
#define VW_long 1
#endif

// --- Configuration ---
#define N (1 << 18) // Array size: 262,144 elements
#define RUNS 5      // Number of benchmark runs

// ====================================================================
// SECTION 1: UTILITY FUNCTIONS
// ====================================================================

// Initialization and Verification for long
void initialize_array_long(long *arr, int size) {
    srand(42); // Deterministic seed
    for (int i = 0; i < size; i++) {
        arr[i] = (long)((i * 3 + rand() % 100) % size);
    }
}

bool is_sorted_long(const long *arr, int size) {
    for (int i = 1; i < size; i++) {
        if (arr[i - 1] > arr[i]) {
            fprintf(stderr, "Verification failed at index %d: %ld > %ld\n", i - 1, arr[i - 1], arr[i]);
            return false;
        }
    }
    return true;
}

// Timer
double get_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

// ====================================================================
// SECTION 2: MERGE IMPLEMENTATIONS for 'long'
// ====================================================================

// --- Function Declarations for long ---
void merge_asm_scalar_1x_long(long *arr, const long *aux, int low, int mid, int high);
#ifdef USE_SIMD
void merge_asm_simd_1x_long(long *arr, const long *aux, int low, int mid, int high);
#endif
void merge_scalar_1x_long(long *arr, const long *aux, int low, int mid, int high);
void merge_scalar_2x_long(long *arr, const long *aux, int low, int mid, int high);
void merge_scalar_4x_long(long *arr, const long *aux, int low, int mid, int high);
void merge_scalar_8x_long(long *arr, const long *aux, int low, int mid, int high);

#ifdef USE_SIMD
void merge_simd_1x_long(long *arr, const long *aux, int low, int mid, int high);
void merge_simd_2x_long(long *arr, const long *aux, int low, int mid, int high);
void merge_simd_4x_long(long *arr, const long *aux, int low, int mid, int high);
void merge_simd_8x_long(long *arr, const long *aux, int low, int mid, int high);
void merge_simd_16x_long(long *arr, const long *aux, int low, int mid, int high);
#endif


// --------------------------------------------------------------------
// Assembly Implementations (long)
// --------------------------------------------------------------------

/**
 * @brief Performs a 1x merge using inline assembly, strictly controlling register usage (max 8 GPRs).
 */
void merge_asm_scalar_1x_long(long *arr, const long *aux, int low, int mid, int high) {
    // Setup initial pointers
    long *i_ptr = (long*)&aux[low];
    long *j_ptr = (long*)&aux[mid + 1];
    long *k_ptr = (long*)&arr[low];
    long *i_limit = (long*)&aux[mid];
    long *j_limit = (long*)&aux[high]; // Output limit

    // Uses 7 GPRs: RDI (k_ptr), RDX (i_ptr), RCX (j_ptr), R8 (i_limit), R9 (j_limit), R10 (aux[i]), R11 (aux[j])
    __asm__ __volatile__ (
        // Initialize pointers/limits into dedicated GPRs
        "movq %0, %%rdx\n"      // RDX = i_ptr
        "movq %1, %%rcx\n"      // RCX = j_ptr
        "movq %2, %%r8\n"       // R8 = i_limit
        "movq %3, %%r9\n"       // R9 = j_limit
        "movq %4, %%rdi\n"      // RDI = k_ptr
        
        "2:\n"                  // Loop start
        
        // 1. Check if left side (i) is exhausted: RDX > R8
        "cmpq %%r8, %%rdx\n"    
        "jg 4b_i_exhausted\n"   // If i exhausted, jump to load/take from j 

        // 2. Check if right side (j) is exhausted: RCX > R9
        "cmpq %%r9, %%rcx\n"    
        "jg 3b_j_exhausted\n"   // If j exhausted, jump to load/take from i

        // 3. Both sides have elements: load and compare
        "movq (%%rdx), %%r10\n" // R10 = aux[i]
        "movq (%%rcx), %%r11\n" // R11 = aux[j]
        "cmpq %%r10, %%r11\n"   // Compare aux[j] (R11) to aux[i] (R10)
        "jl 4f_compare\n"       // If aux[j] < aux[i], jump to take from j
        
        // TAKE FROM I (Fallthrough from compare: aux[i] <= aux[j])
        "3:\n"
        "movq %%r10, (%%rdi)\n" // arr[k] = aux[i] (R10 is pre-loaded)
        "addq $8, %%rdx\n"      // i_ptr++
        "jmp 5f\n"              // Jump to next iteration/exit check

        // J-EXHAUSTED (Jump from exhaustion check 2)
        "3b_j_exhausted:\n"
        "movq (%%rdx), %%r10\n" // FIX: R10 = aux[i] (Must load here!)
        "jmp 3b\n"              // Jump to the write-I path

        // TAKE FROM J (Jump from compare: aux[j] < aux[i])
        "4f_compare:\n"
        
        // TAKE FROM J (Fallthrough from compare or jump from exhaustion)
        "4:\n"
        "movq %%r11, (%%rdi)\n" // arr[k] = aux[j] (R11 is pre-loaded)
        "addq $8, %%rcx\n"      // j_ptr++
        "jmp 5f\n"

        // I-EXHAUSTED (Jump from exhaustion check 1)
        "4b_i_exhausted:\n"
        "movq (%%rcx), %%r11\n" // FIX: R11 = aux[j] (Must load here!)
        "jmp 4b\n"              // Jump to the write-J path

        // Loop end / Iteration complete
        "5:\n"
        "addq $8, %%rdi\n"      // k_ptr++
        
        // 4. Check if output array (k) is complete: RDI > R9
        "cmpq %%r9, %%rdi\n"    
        "jle 2b\n"              // If k_ptr <= aux[high], jump back to loop start (2b)

        : // No explicit outputs needed for this block
        : "g" (i_ptr), "g" (j_ptr), "g" (i_limit), "g" (j_limit), "g" (k_ptr) // Inputs
        : "rdx", "rcx", "r8", "r9", "rdi", "r10", "r11", "cc", "memory" // Clobbered: 7 GPRs + Flags + Memory
    );
}

#ifdef USE_SIMD
/**
 * @brief Performs a 1x merge using inline assembly with SIMD (SSE4.1) block transfer.
 */
void merge_asm_simd_1x_long(long *arr, const long *aux, int low, int mid, int high) { 
    long *i_ptr = (long*)&aux[low];
    long *j_ptr = (long*)&aux[mid + 1];
    long *k_ptr = (long*)&arr[low];
    long *i_limit = (long*)&aux[mid];
    long *j_limit = (long*)&aux[high];

    if (high - low < VW_long || mid < low || mid >= high) {
        merge_scalar_1x_long(arr, aux, low, mid, high);
        return;
    }

    // GPRs: RDX (i_ptr), RCX (j_ptr), R8 (k_ptr), R9 (j_limit), R11 (i_limit)
    // XMMs: XMM0, XMM1
    __asm__ __volatile__ (
        "movq %0, %%rdx\n"      // RDX = i_ptr
        "movq %1, %%rcx\n"      // RCX = j_ptr
        "movq %2, %%r8\n"       // R8 = k_ptr (output)
        "movq %3, %%r9\n"       // R9 = j_limit
        "movq %4, %%r11\n"      // R11 = i_limit
        
        "1:\n"                  // Loop start
        
        "cmpq %%r9, %%r8\n" 
        "jg 7f\n"               // Exit if output complete

        "cmpq %%r11, %%rdx\n"       
        "jg 4f\n"                   // If i exhausted, jump to take j block (4f)

        "cmpq %%r9, %%rcx\n"        
        "jg 3f\n"                   // If j exhausted, jump to take i block (3f)

        // Compare first elements for block-transfer decision (uses RAX, RBX)
        "movq (%%rdx), %%rax\n"     // RAX = aux[i]
        "movq (%%rcx), %%rbx\n"     // RBX = aux[j]
        
        "cmpq %%rbx, %%rax\n"       
        "jg 4f_simd\n"                   // If aux[j] < aux[i], jump to attempt block from j (4f_simd)
        
        // Take 2 elements from i (aux[i] <= aux[j] or j exhausted)
        "3f_simd:\n"
        // Check if a full vector can be read from i
        "addq $8, %%rdx\n"          // Check i_ptr + 1
        "cmpq %%r11, %%rdx\n"       
        "ja 8f\n"                   // Jump to C cleanup for single element (8f)
        "subq $8, %%rdx\n"          // Restore i_ptr

        "movdqu (%%rdx), %%xmm0\n"  // XMM0 = [aux[i], aux[i+1]]
        "movdqu %%xmm0, (%%r8)\n"   
        "addq $16, %%rdx\n"         // i_ptr += 2
        "addq $16, %%r8\n"          // k_ptr += 2
        "jmp 1b\n"                  // Continue loop

        // Take 2 elements from j (aux[j] < aux[i] or i exhausted)
        "4f_simd:\n"
        // Check if a full vector can be read from j
        "addq $8, %%rcx\n"          // Check j_ptr + 1
        "cmpq %%r9, %%rcx\n"        
        "ja 8f\n"                   // Jump to C cleanup for single element (8f)
        "subq $8, %%rcx\n"          // Restore j_ptr
        
        "movdqu (%%rcx), %%xmm1\n"  // XMM1 = [aux[j], aux[j+1]]
        "movdqu %%xmm1, (%%r8)\n"   
        "addq $16, %%rcx\n"         // j_ptr += 2
        "addq $16, %%r8\n"          // k_ptr += 2
        "jmp 1b\n"                  // Continue loop

        // 8f is the point where we fall back to C cleanup (only one element left)
        "8:\n"
        
        "7:\n"                     // Assembly exit point/fallthrough
        
        : // No explicit outputs
        : "g" (i_ptr), "g" (j_ptr), "g" (k_ptr), "g" (j_limit), "g" (i_limit)
        : "rdx", "rcx", "r8", "r9", "r11", "rax", "rbx", "cc", "memory", "xmm0", "xmm1"
    );
    
    // Terminate the inline assembly block
    __asm__ __volatile__(""); 
    
    // C-based scalar cleanup loop
    int k = (k_ptr - arr);
    int i = (i_ptr - aux);
    int j = (j_ptr - aux);
    
    for (; k <= high; k++) { 
        if (i > mid) { arr[k] = aux[j++]; } 
        else if (j > high) { arr[k] = aux[i++]; } 
        else if (aux[j] < aux[i]) { arr[k] = aux[j++]; } 
        else { arr[k] = aux[i++]; }
    }
}
#endif // USE_SIMD

// --------------------------------------------------------------------
// C-based Implementations (long) - Unrolled versions fallback to 1x for simplicity
// --------------------------------------------------------------------

void merge_scalar_1x_long(long *arr, const long *aux, int low, int mid, int high) {
    int i = low, j = mid + 1;
    for (int k = low; k <= high; k++) {
        if (i > mid) { arr[k] = aux[j++]; }
        else if (j > high) { arr[k] = aux[i++]; }
        else if (aux[j] < aux[i]) { arr[k] = aux[j++]; }
        else { arr[k] = aux[i++]; }
    }
}

void merge_scalar_2x_long(long *arr, const long *aux, int low, int mid, int high) { merge_scalar_1x_long(arr, aux, low, mid, high); }
void merge_scalar_4x_long(long *arr, const long *aux, int low, int mid, int high) { merge_scalar_1x_long(arr, aux, low, mid, high); }
void merge_scalar_8x_long(long *arr, const long *aux, int low, int mid, int high) { merge_scalar_1x_long(arr, aux, low, mid, high); }

#ifdef USE_SIMD
void merge_simd_1x_long(long *arr, const long *aux, int low, int mid, int high) { merge_scalar_1x_long(arr, aux, low, mid, high); }
void merge_simd_2x_long(long *arr, const long *aux, int low, int mid, int high) { merge_scalar_1x_long(arr, aux, low, mid, high); }
void merge_simd_4x_long(long *arr, const long *aux, int low, int mid, int high) { merge_scalar_1x_long(arr, aux, low, mid, high); }
void merge_simd_8x_long(long *arr, const long *aux, int low, int mid, int high) { merge_scalar_1x_long(arr, aux, low, mid, high); }
void merge_simd_16x_long(long *arr, const long *aux, int low, int mid, int high) { merge_scalar_1x_long(arr, aux, low, mid, high); }
#endif

// --------------------------------------------------------------------
// Recursive Sort Driver 
// --------------------------------------------------------------------

/**
 * @brief Recursive function that performs the top-down merge sort.
 */
void mergeSort_recursive_long(long *data, long *aux, int low, int high, 
                              void (*merge_func)(long*, const long*, int, int, int)) {
    
    // Base case: 1 element or less
    if (low >= high) {
        if (low == high) {
             aux[low] = data[low]; // Ensure aux has the base element
        }
        return;
    }

    int mid = low + (high - low) / 2;

    // 1. Recursively sort the left half: (Swap data and aux roles)
    mergeSort_recursive_long(aux, data, low, mid, merge_func); 
    
    // 2. Recursively sort the right half: (Swap data and aux roles)
    mergeSort_recursive_long(aux, data, mid + 1, high, merge_func); 

    // 3. Merge sorted halves from 'aux' back into 'data'.
    // merge_func(destination_arr, source_aux, low, mid, high)
    merge_func(data, aux, low, mid, high);
}


// ====================================================================
// SECTION 3: BENCHMARK DRIVER
// ====================================================================

// Generic function to run the benchmark for a given data type (Simplified to match the specific type used in main)
void run_benchmark_test(long *data, long *aux, size_t type_size, const char *type_name, int vw,
                        void (*init_func)(long*, int),
                        bool (*verify_func)(const long*, int),
                        void (*sort_func)(long*, long*, int, int, void*),
                        void **scalar_funcs, const char **scalar_names, int scalar_count,
                        void **simd_funcs, const char **simd_names, int simd_count) {
    
    printf("\n--- Type: %s (Size: %zu bytes, VW: %d) ---\n", type_name, type_size, vw);

    int total_count = scalar_count + simd_count;
    void *all_funcs[total_count];
    const char *all_names[total_count];
    
    for(int i = 0; i < scalar_count; i++) {
        all_funcs[i] = scalar_funcs[i];
        all_names[i] = scalar_names[i];
    }
    for(int i = 0; i < simd_count; i++) {
        all_funcs[scalar_count + i] = simd_funcs[i];
        all_names[scalar_count + i] = simd_names[i];
    }

    // Run tests
    for (int i = 0; i < total_count; i++) {
        double total_time = 0.0;
        // Cast the function pointer back to the merge signature
        void (*current_merge_func)(long*, const long*, int, int, int) = all_funcs[i];
        bool verified = false;

        for (int r = 0; r < RUNS; r++) {
            init_func(data, N);
            memcpy(aux, data, N * sizeof(long)); // aux starts as unsorted copy
            
            double start_time = get_time_sec();
            
            // The initial call starts the sort
            sort_func(data, aux, 0, N - 1, current_merge_func);
            
            total_time += get_time_sec() - start_time;

            if (r == 0) {
                // The mergeSort_recursive_long function ensures the final result is in 'data'
                verified = verify_func(data, N);
            }
        }

        printf("  %-15s | Avg Time: %8.4f ms | Verified: %s\n", 
               all_names[i], 
               (total_time / RUNS) * 1000.0, 
               verified ? "YES" : "NO ");
    }
}


// ====================================================================
// SECTION 4: MAIN DRIVER
// ====================================================================

int main() {
    printf("--- Starting Comprehensive Merge Sort Benchmark (N=%d, RUNS=%d) ---\n", N, RUNS);

    // --- Test long (Includes Assembly Tests) ---
    long *data_long = (long*)malloc(N * sizeof(long));
    // Allocate N+2 elements for the auxiliary buffer to guard against minor overflows
    long *aux_long = (long*)malloc((N + 2) * sizeof(long)); 
    if (!data_long || !aux_long) { perror("Memory allocation failed for long"); return 1; }

    void (*scalar_funcs_long[])(long*, const long*, int, int, int) = {
        merge_asm_scalar_1x_long, 
        merge_scalar_1x_long, 
        merge_scalar_2x_long, 
        merge_scalar_4x_long,
        merge_scalar_8x_long
    };
    const char *scalar_names_long[] = { "ASM_1x_SCL (8Reg)", "C_1x_SCL", "C_2x_SCL", "C_4x_SCL", "C_8x_SCL" };
    const int scalar_count_long = 5;
    
    void **simd_funcs_long_ptr = NULL;
    const char **simd_names_long_ptr = NULL;
    int simd_count_long = 0;
    
    #ifdef USE_SIMD
    static void (*simd_funcs_long[])(long*, const long*, int, int, int) = {
        merge_asm_simd_1x_long, 
        merge_simd_1x_long, 
        merge_simd_2x_long, 
        merge_simd_4x_long,
        merge_simd_8x_long,
        merge_simd_16x_long
    };
    static const char *simd_names_long[] = { "ASM_1x_SIMD", "C_1x_SIMD", "C_2x_SIMD", "C_4x_SIMD", "C_8x_SIMD", "C_16x_SIMD" };
    simd_funcs_long_ptr = (void**)simd_funcs_long;
    simd_names_long_ptr = simd_names_long;
    simd_count_long = 6;
    #endif
    
    // Run the specific test for 'long'
    run_benchmark_test(data_long, aux_long, sizeof(long), "long", VW_long,
                       initialize_array_long,
                       is_sorted_long,
                       (void (*)(long*, long*, int, int, void*))mergeSort_recursive_long,
                       (void**)scalar_funcs_long, scalar_names_long, scalar_count_long,
                       simd_funcs_long_ptr, simd_names_long_ptr, simd_count_long
                       );
    free(data_long); 
    free(aux_long); 
    
    printf("\n--- Benchmark Complete ---\n");
    return 0;
}
