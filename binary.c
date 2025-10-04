#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <limits.h>

// Note: Including <immintrin.h> is often required for AVX intrinsics,
// but for pure inline assembly with Clang/GCC, we rely on the compiler 
// being informed via compilation flags (like -mavx2).

// Define image dimensions
#define ROWS 64
#define BITS_PER_ROW 64 // 64 bits = 8 bytes

// Test parameters
#define NUM_1D_SHIFTS 8
#define NUM_2D_SHIFTS 8

#define TOTAL_COMPARISONS_1D (ROWS * NUM_1D_SHIFTS)
#define TOTAL_COMPARISONS_2D (ROWS * NUM_2D_SHIFTS)

// --- Synthetic Image Data (64x64 bits, stored as 64 uint64_t) ---
uint64_t REFERENCE_IMAGE[ROWS];
uint64_t HANDWRITTEN_IMAGE[ROWS];

// Arrays to store the results
uint32_t distances_1d[TOTAL_COMPARISONS_1D];
uint32_t distances_2d[TOTAL_COMPARISONS_2D];

/**
 * @brief Initializes synthetic data for the 64x64 bit images.
 *
 * The pattern is chosen to ensure non-trivial XOR distances when shifted.
 */
void initialize_synthetic_data(void) {
    // Pattern: 0xF0F0F0F0F0F0F0F0 (Alternating 1s and 0s every 4 bits)
    uint64_t reference_pattern = 0xF0F0F0F0F0F0F0F0ULL;

    for (int i = 0; i < ROWS; ++i) {
        REFERENCE_IMAGE[i] = reference_pattern;

        // Create a slightly deformed handwritten image for testing
        if (i % 2 == 0) {
            // Even rows: Shifted right by 1 bit, plus some noise
            HANDWRITTEN_IMAGE[i] = (reference_pattern >> 1) ^ 0x0100010001000100ULL;
        } else {
             // Odd rows: Shifted left by 2 bits, plus some noise
             HANDWRITTEN_IMAGE[i] = (reference_pattern << 2) ^ 0x0002000200020002ULL;
        }
    }
}

/**
 * @brief Calculates the Hamming distance for 8 1D horizontal deformations per row,
 * using AVX (256-bit) assembly to process 4 rows in parallel.
 * * FIX: The AVX shift instructions (vpsllq, vpsrlq) require the shift count to be 
 * provided via an XMM register, not the CL register. This block has been corrected.
 */
void calculate_1d_distances(void) {
    int result_index = 0;
    // Shifts allowed: -4, -3, -2, -1 (Left), 1, 2, 3, 4 (Right)
    const int shifts[] = { -4, -3, -2, -1, 1, 2, 3, 4 };
    
    // We iterate in chunks of 4 rows, since AVX 256-bit registers hold 4 x 64-bit integers.
    for (int i = 0; i < ROWS; i += 4) {
        // Pointers to the current block of 4 rows
        const uint64_t *h_ptr = &HANDWRITTEN_IMAGE[i];
        const uint64_t *r_ptr = &REFERENCE_IMAGE[i];
        
        // Array to hold the 4 resulting 64-bit XOR results from the YMM register
        uint64_t four_xored_results[4]; 

        for (int j = 0; j < NUM_1D_SHIFTS; ++j) {
            int shift_amount = shifts[j];
            
            if (shift_amount < 0) {
                // --- AVX ASSEMBLY BLOCK for Parallel LEFT SHIFT (vpsllq) ---
                int absolute_shift = -shift_amount;
                // Store the scalar shift amount in a 64-bit variable 
                // so we can pass its memory address to the assembly (constraint %3).
                uint64_t shift_reg_val = absolute_shift;
                
                __asm__ volatile (
                    // Load 4 handwritten rows into YMM0 (256 bits)
                    "vmovdqu %1, %%ymm0\n\t"
                    // Load 4 reference rows into YMM1
                    "vmovdqu %2, %%ymm1\n\t"
                    
                    // Load scalar shift amount (64-bit) into XMM2 using vmovq. (%3 is the memory location of shift_reg_val)
                    "vmovq %3, %%xmm2\n\t" 
                    
                    // Parallel Shift Left (vpsllq uses the shift count from the lowest 64 bits of XMM2)
                    "vpsllq %%xmm2, %%ymm0, %%ymm0\n\t" 
                    
                    // Parallel XOR (YMM0 = YMM0 XOR YMM1)
                    "vpxor %%ymm1, %%ymm0, %%ymm0\n\t"
                    
                    // Store the 4 x 64-bit XOR results back to memory
                    "vmovdqu %%ymm0, %0\n\t"
                    
                    : "=m" (four_xored_results[0]) // %0
                    : "m" (h_ptr[0]), // %1
                      "m" (r_ptr[0]), // %2
                      "m" (shift_reg_val) // %3 (The new shift count constraint)
                    : "ymm0", "ymm1", "xmm2" // Clobbered registers: added xmm2
                );

            } else {
                // --- AVX ASSEMBLY BLOCK for Parallel RIGHT SHIFT (vpsrlq) ---
                uint64_t shift_reg_val = shift_amount;
                
                __asm__ volatile (
                    "vmovdqu %1, %%ymm0\n\t"
                    "vmovdqu %2, %%ymm1\n\t"
                    
                    // Load scalar shift amount (64-bit) into XMM2
                    "vmovq %3, %%xmm2\n\t"
                    
                    // Parallel Shift Right
                    "vpsrlq %%xmm2, %%ymm0, %%ymm0\n\t" 
                    
                    "vpxor %%ymm1, %%ymm0, %%ymm0\n\t"
                    
                    "vmovdqu %%ymm0, %0\n\t"
                    
                    : "=m" (four_xored_results[0])
                    : "m" (h_ptr[0]),
                      "m" (r_ptr[0]),
                      "m" (shift_reg_val) 
                    : "ymm0", "ymm1", "xmm2" // Clobbered registers: added xmm2
                );
            }

            // After parallel shift and XOR, we perform the scalar POPCNT on the 4 results.
            for (int k = 0; k < 4; ++k) {
                uint64_t distance_val = 0;
                
                // --- Scalar POPCNT ASSEMBLY BLOCK ---
                __asm__ volatile (
                    "popcntq %1, %0\n\t" // Count bits in the 64-bit element
                    : "=r" (distance_val)
                    : "r" (four_xored_results[k])
                );
                distances_1d[result_index++] = (uint32_t)distance_val;
            }
        }
    }
}


/**
 * @brief Calculates the Hamming distance for 8 2D shifts (Up, Down, Diagonals, etc.).
 *
 * This function remains scalar (non-SIMD) for comparison.
 */
void calculate_2d_distances(void) {
    int result_index = 0;
    // Shifts defined as (dy, dx): (Row Offset, Bit Shift)
    const int shifts_2d[NUM_2D_SHIFTS][2] = {
        {0, 1},   // Right
        {0, -1},  // Left
        {-1, 0},  // Up
        {1, 0},   // Down
        {-1, 1},  // Up-Right
        {-1, -1}, // Up-Left
        {1, 1},   // Down-Right
        {1, -1}   // Down-Left
    };

    for (int i = 0; i < ROWS; ++i) {
        uint64_t handwritten_row = HANDWRITTEN_IMAGE[i];

        for (int k = 0; k < NUM_2D_SHIFTS; ++k) {
            int dy = shifts_2d[k][0]; // Vertical (Row) offset
            int dx = shifts_2d[k][1]; // Horizontal (Bit) shift

            int reference_index = i + dy;
            
            // Handle boundary conditions: Pad with 0s (all black or all white depending on context, 0 is safer)
            uint64_t reference_row;
            if (reference_index < 0 || reference_index >= ROWS) {
                reference_row = 0ULL; 
            } else {
                reference_row = REFERENCE_IMAGE[reference_index];
            }

            uint64_t distance_val = 0; 
            
            // --- ASSEMBLY BLOCK for 2D SHIFT (Applied to Reference Row) ---
            
            if (dx == 0) {
                 // No horizontal shift required, just XOR and POPCNT.
                 // This covers Up and Down shifts.
                 __asm__ volatile (
                    "xorq %1, %2\n\t"        // XOR handwritten_row with reference_row (result in %2)
                    "popcntq %2, %0\n\t"     // Count set bits, store in distance_val
                    : "=r" (distance_val)
                    : "r" (handwritten_row),
                      "r" (reference_row)
                    : 
                 );

            } else if (dx > 0) { // Right Shift or Down-Right/Up-Right Diagonal
                int shift_amount = dx;
                __asm__ volatile (
                    "movq %1, %%rax\n\t"   
                    "shrq %%cl, %%rax\n\t" // Shift Reference Row Right
                    "xorq %2, %%rax\n\t"   
                    "popcntq %%rax, %0\n\t"
                    : "=r" (distance_val)
                    : "r" (reference_row), 
                      "r" (handwritten_row),
                      "c" ((uint8_t)shift_amount) 
                    : "rax"
                );
            } else { // Left Shift or Down-Left/Up-Left Diagonal (dx < 0)
                int absolute_shift = -dx;
                 __asm__ volatile (
                    "movq %1, %%rax\n\t"   
                    "shlq %%cl, %%rax\n\t" // Shift Reference Row Left
                    "xorq %2, %%rax\n\t"   
                    "popcntq %%rax, %0\n\t"
                    : "=r" (distance_val)
                    : "r" (reference_row), 
                      "r" (handwritten_row),
                      "c" ((uint8_t)absolute_shift) 
                    : "rax"
                );
            }

            distances_2d[result_index++] = (uint32_t)distance_val;
        }
    }
}


/**
 * @brief Helper to calculate and print stats for a given distance array.
 */
void summarize_stats(const char *test_name, const uint32_t *data, int count) {
    uint32_t min_dist = UINT32_MAX;
    uint32_t max_dist = 0;
    uint64_t sum_dist = 0;

    printf("\n--- %s Statistics ---\n", test_name);
    
    for (int i = 0; i < count; ++i) {
        uint32_t dist = data[i];
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
        sum_dist += dist;
    }

    double avg_dist = (double)sum_dist / count;

    printf("Total Comparisons: %d\n", count);
    printf("Minimum Distance Found: %u bits\n", min_dist);
    printf("Maximum Distance Found: %u bits\n", max_dist);
    printf("Average Distance: %.2f bits\n", avg_dist);
}

/**
 * @brief Summarizes and prints the calculated distance statistics for both tests.
 */
void print_stats(void) {
    printf("========================================================\n");
    printf("        Image Deformation Distance Analysis (64x64)\n");
    printf("========================================================\n");
    
    // --- 1D Shift Results ---
    summarize_stats("1D Horizontal Row Shifts (AVX Vectorized)", distances_1d, TOTAL_COMPARISONS_1D);

    // --- 2D Shift Results ---
    summarize_stats("2D Local/Global Translation Shifts (Scalar)", distances_2d, TOTAL_COMPARISONS_2D);

    printf("\n--- Detailed 2D Test Results (First 4 Rows) ---\n");
    const char *shift_names[] = {"R+1", "L-1", "U-1", "D+1", "UR", "UL", "DR", "DL"};

    for (int i = 0; i < ROWS && i < 4; ++i) {
        for (int k = 0; k < NUM_2D_SHIFTS; ++k) {
            uint32_t dist = distances_2d[(i * NUM_2D_SHIFTS) + k];
            printf("Row %2d, Shift %4s: Distance = %u bits\n", i + 1, shift_names[k], dist);
        }
    }
    
    printf("\nNote: 1D calculation uses AVX (256-bit) SIMD instructions for parallel XOR and Shift.\n");
    printf("Compilation requires AVX and POPCNT support (e.g., -mavx2 -msse4.2 flags).\n");
}

int main() {
    initialize_synthetic_data();
    
    // Run both calculation tests
    calculate_1d_distances();
    calculate_2d_distances();
    
    print_stats();
    
    return 0;
}