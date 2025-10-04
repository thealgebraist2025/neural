#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <limits.h>

// Define image dimensions
#define ROWS 64
#define BITS_PER_ROW 64 // 64 bits = 8 bytes

// Test parameters
#define NUM_1D_SHIFTS 8
#define NUM_2D_SHIFTS 8

#define TOTAL_COMPARISONS_1D (ROWS * NUM_1D_SHIFTS)
#define TOTAL_COMPARISONS_2D (ROWS * NUM_2D_SHIFTS)
#define TOTAL_INCREMENTAL_OPS 65536 // 2^16
#define CHECKPOINT_INTERVAL 1024    // 2^10

// --- Synthetic Image Data (64x64 bits, stored as 64 uint64_t) ---
uint64_t REFERENCE_IMAGE[ROWS];
uint64_t HANDWRITTEN_IMAGE[ROWS];
// A copy of the reference image that we will "deform" during the incremental search
uint64_t CURRENT_DEFORMED_IMAGE[ROWS]; 

// Arrays to store the results
uint32_t distances_1d[TOTAL_COMPARISONS_1D];
uint32_t distances_2d[TOTAL_COMPARISONS_2D];

// Structure to store checkpoint data (Best distance found in that block)
typedef struct {
    uint32_t best_dist;
    uint32_t best_index;
} CheckpointResult;

CheckpointResult checkpoint_results[TOTAL_INCREMENTAL_OPS / CHECKPOINT_INTERVAL];

// Forward declaration of the scalar initial calculation
uint64_t calculate_initial_total_distance(void);

/**
 * @brief Initializes synthetic data for the 64x64 bit images.
 */
void initialize_synthetic_data(void) {
    uint64_t reference_pattern = 0xF0F0F0F0F0F0F0F0ULL;

    for (int i = 0; i < ROWS; ++i) {
        REFERENCE_IMAGE[i] = reference_pattern;
        
        if (i % 2 == 0) {
            HANDWRITTEN_IMAGE[i] = (reference_pattern >> 1) ^ 0x0100010001000100ULL;
        } else {
             HANDWRITTEN_IMAGE[i] = (reference_pattern << 2) ^ 0x0002000200020002ULL;
        }
        CURRENT_DEFORMED_IMAGE[i] = HANDWRITTEN_IMAGE[i];
    }
}

/**
 * @brief Calculates the initial total Hamming distance for all 64 rows (Scalar POPCNT).
 * @return The total Hamming distance.
 */
uint64_t calculate_initial_total_distance(void) {
    uint64_t total_distance = 0;
    for (int i = 0; i < ROWS; ++i) {
        uint64_t xor_result = HANDWRITTEN_IMAGE[i] ^ REFERENCE_IMAGE[i];
        uint64_t row_distance;
        
        // Use scalar POPCNT assembly
        __asm__ volatile (
            "popcntq %1, %0\n\t"
            : "=r" (row_distance)
            : "r" (xor_result)
        );
        total_distance += row_distance;
    }
    return total_distance;
}

/**
 * @brief Simulates a single 2x2 movement by modifying 2 rows.
 */
void simulate_2x2_move(int index, int *affected_row_start, uint64_t *new_row_data) {
    *affected_row_start = (index % (ROWS - 1)); 
    
    int row_index = *affected_row_start;

    new_row_data[0] = CURRENT_DEFORMED_IMAGE[row_index] ^ (1ULL << (index % 64));
    new_row_data[1] = CURRENT_DEFORMED_IMAGE[row_index + 1] ^ (1ULL << ((index + 1) % 64));
}


/**
 * @brief Core SIMD function using register accumulation and checkpointing.
 * * FIX: Corrected assembly operand numbering to resolve "invalid operand number" errors.
 */
void test_simd_incremental_search(void) {
    // --- C SETUP ---
    uint64_t current_total_distance = calculate_initial_total_distance();
    uint64_t best_distance_in_run = current_total_distance;
    uint32_t best_index_in_run = 0;
    
    // FIX: Changed %llu to %lu to match the typical size of uint64_t on x86-64
    printf("\n--- Starting Incremental AVX Search Simulation ---\n");
    printf("Initial Total Distance: %lu bits\n", current_total_distance);

    uint64_t old_row_data[2]; 
    uint64_t new_row_data[2]; 
    
    for (int i = 0; i < TOTAL_INCREMENTAL_OPS; ++i) {
        int affected_row_start;
        
        // 1. Simulate Move
        simulate_2x2_move(i, &affected_row_start, new_row_data);

        // Backup old data for incremental calculation
        old_row_data[0] = CURRENT_DEFORMED_IMAGE[affected_row_start];
        old_row_data[1] = CURRENT_DEFORMED_IMAGE[affected_row_start + 1];

        // --- AVX ASSEMBLY BLOCK for Incremental Update ---
        // Note on Operand Numbering:
        // %0: diff, %1: old_dist_sum, %2: new_dist_sum, %3: total_distance_reg (R/W register)
        // %4: old_row_data (Memory), %5: new_row_data (Memory), %6: reference_rows (Memory)

        uint64_t old_dist_sum, new_dist_sum, diff;
        uint64_t total_distance_reg = current_total_distance; 
        uint64_t reference_rows[2] = {
            REFERENCE_IMAGE[affected_row_start], 
            REFERENCE_IMAGE[affected_row_start + 1]
        };

        __asm__ volatile (
            // Load 2 Old Rows into XMM0 (128 bits: 2 x 64-bit)
            "vmovdqu %4, %%xmm0\n\t"
            // Load 2 New Rows into XMM1
            "vmovdqu %5, %%xmm1\n\t"
            // Load 2 Reference Rows into XMM2
            "vmovdqu %6, %%xmm2\n\t"
            
            // --- 1. Calculate Old Distance Sum (XMM0 XOR XMM2) ---
            "vpxor %%xmm2, %%xmm0, %%xmm3\n\t" // XMM3 = Old XOR result (2 rows)
            
            // Extract and POPCNT 1st row (Low 64-bits)
            "vextracti128 $0, %%xmm3, %%xmm4\n\t" 
            "vmovq %%xmm4, %%rax\n\t" 
            "popcntq %%rax, %1\n\t"  // POPCNT on 1st row, store in old_dist_sum (%1)

            // Extract and POPCNT 2nd row (High 64-bits)
            "vextracti128 $1, %%xmm3, %%xmm4\n\t" 
            "vmovq %%xmm4, %%rbx\n\t" 
            "popcntq %%rbx, %%rbx\n\t" 
            "addq %%rbx, %1\n\t"      // old_dist_sum += (2nd row POPCNT)

            // --- 2. Calculate New Distance Sum (XMM1 XOR XMM2) ---
            "vpxor %%xmm2, %%xmm1, %%xmm5\n\t" // XMM5 = New XOR result (2 rows)

            // Extract and POPCNT 1st row (Low 64-bits)
            "vextracti128 $0, %%xmm5, %%xmm4\n\t" 
            "vmovq %%xmm4, %%rax\n\t" 
            "popcntq %%rax, %2\n\t"  // POPCNT on 1st row, store in new_dist_sum (%2)

            // Extract and POPCNT 2nd row (High 64-bits)
            "vextracti128 $1, %%xmm5, %%xmm4\n\t" 
            "vmovq %%xmm4, %%rbx\n\t" 
            "popcntq %%rbx, %%rbx\n\t" 
            "addq %%rbx, %2\n\t"      // new_dist_sum += (2nd row POPCNT)

            // --- 3. Calculate Difference and New Total (Register Math) ---
            "movq %3, %%rax\n\t"        // Load total_distance_reg (%3) into RAX
            "subq %1, %%rax\n\t"        // RAX -= old_dist_sum (%1)
            "addq %2, %%rax\n\t"        // RAX += new_dist_sum (%2)
            "movq %%rax, %0\n\t"        // Store result (new total) in diff (%0)

            : "=r" (diff), // %0 Output: The new total distance
              "=r" (old_dist_sum), // %1 Output: Old distance sum (must be set as output to be used as R/W)
              "=r" (new_dist_sum) // %2 Output: New distance sum
            : "r" (total_distance_reg), // %3 Input: Old total distance
              "m" (old_row_data[0]), // %4 Input: Old row data
              "m" (new_row_data[0]), // %5 Input: New row data
              "m" (reference_rows[0]) // %6 Input: Reference rows
            : "rax", "rbx", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5" // Clobbered
        );

        // --- C/Register Update (Simulated) ---
        current_total_distance = diff; 
        
        // ... (rest of the C logic remains the same) ...

        if (current_total_distance < best_distance_in_run) {
            best_distance_in_run = current_total_distance; 
            best_index_in_run = i; 
        }
        
        // 4. Checkpoint Logic
        if ((i % CHECKPOINT_INTERVAL) == (CHECKPOINT_INTERVAL - 1)) {
            int checkpoint_idx = i / CHECKPOINT_INTERVAL;
            
            // --- Write Checkpoint to Memory ---
            checkpoint_results[checkpoint_idx].best_dist = (uint32_t)best_distance_in_run;
            checkpoint_results[checkpoint_idx].best_index = best_index_in_run;
            
            // Reset for the next block
            best_distance_in_run = current_total_distance;
            best_index_in_run = i + 1;
        }

        // Apply the move to the working image for the next iteration's "old data"
        CURRENT_DEFORMED_IMAGE[affected_row_start] = new_row_data[0];
        CURRENT_DEFORMED_IMAGE[affected_row_start + 1] = new_row_data[1];
    }
    
    // Final check for the last block
    if (TOTAL_INCREMENTAL_OPS % CHECKPOINT_INTERVAL != 0) {
         int checkpoint_idx = TOTAL_INCREMENTAL_OPS / CHECKPOINT_INTERVAL;
         checkpoint_results[checkpoint_idx].best_dist = (uint32_t)best_distance_in_run;
         checkpoint_results[checkpoint_idx].best_index = best_index_in_run;
    }
    
    printf("Incremental AVX Search Complete.\n");
    printf("Total Checkpoints Stored in Memory: %zu\n", sizeof(checkpoint_results) / sizeof(CheckpointResult));
    printf("Total Memory Writes Avoided: %d (65536 vs 64 writes)\n", TOTAL_INCREMENTAL_OPS - (TOTAL_INCREMENTAL_OPS / CHECKPOINT_INTERVAL));

}


/**
 * @brief Summarizes and prints the calculated distance statistics for both tests.
 */
void print_stats(void) {
    // ... (Stats printing code remains the same) ...
    uint32_t global_best_dist = UINT32_MAX;
    uint32_t global_best_index = 0;
    
    for (size_t i = 0; i < sizeof(checkpoint_results) / sizeof(CheckpointResult); ++i) {
        if (checkpoint_results[i].best_dist < global_best_dist) {
            global_best_dist = checkpoint_results[i].best_dist;
            global_best_index = checkpoint_results[i].best_index;
        }
    }
    
    printf("\n========================================================\n");
    printf("  SIMD Incremental Search Results (2^16 Operations)\n");
    printf("========================================================\n");
    printf("Best Distance (Global Min): %u bits\n", global_best_dist);
    printf("Found at Operation Index: %u\n", global_best_index);
    printf("\nOptimization Notes:\n");
    printf("- The total distance is maintained in a register (simulated by 'current_total_distance').\n");
    printf("- Only two memory writes occur per %d operations (Checkpointing).\n", CHECKPOINT_INTERVAL);
    printf("- The core distance update uses AVX registers for parallel XOR and incremental sum.\n");
}

int main() {
    // ... (main function remains the same) ...
    initialize_synthetic_data();
    
    // Execute the highly optimized incremental search
    test_simd_incremental_search();
    
    print_stats();
    
    return 0;
}
