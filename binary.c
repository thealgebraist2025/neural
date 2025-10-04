#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <limits.h>

// Define image dimensions
#define ROWS 64
#define BITS_PER_ROW 64 // 64 bits = 8 bytes
#define TOTAL_COMPARISONS (ROWS * 8)

// --- Synthetic Image Data (64x64 bits, stored as 64 uint64_t) ---
// Note: Since we don't have the actual '2' images, we use synthetic data
// to simulate the comparison process.

// 1. Reference Image (Perfect 2): A repeating pattern (e.g., '11110000' byte pattern)
uint64_t REFERENCE_IMAGE[ROWS];

// 2. Handwritten Image (Deformed 2): A slightly shifted/corrupted version
uint64_t HANDWRITTEN_IMAGE[ROWS];

// Array to store the results: 64 rows * 8 deformations
uint32_t distances[TOTAL_COMPARISONS];

/**
 * @brief Initializes synthetic data for the 64x64 bit images.
 */
void initialize_synthetic_data(void) {
    // Pattern: 0xF0F0F0F0F0F0F0F0 (Alternating 1s and 0s)
    uint64_t reference_pattern = 0xF0F0F0F0F0F0F0F0ULL;

    for (int i = 0; i < ROWS; ++i) {
        REFERENCE_IMAGE[i] = reference_pattern;

        // Handwritten image is mostly the reference pattern, but with
        // an intentional small shift and some noise for non-zero distances.
        if (i % 2 == 0) {
            // Shifted right by 1 bit (0x7878...)
            HANDWRITTEN_IMAGE[i] = (reference_pattern >> 1) ^ 0x0100010001000100ULL;
        } else {
             // Shifted left by 2 bits (0xC0C0...)
             HANDWRITTEN_IMAGE[i] = (reference_pattern << 2) ^ 0x0002000200020002ULL;
        }
    }
}

/**
 * @brief Calculates the Hamming distance for 8 deformations per row using
 * embedded x86-64 assembly instructions.
 *
 * The core logic (shift, XOR, POPCNT) is performed using inline assembly for
 * speed and to meet the user's specific requirement.
 */
void calculate_distances(void) {
    int result_index = 0;
    // Shifts allowed: -4, -3, -2, -1 (Left), 1, 2, 3, 4 (Right)
    const int shifts[] = { -4, -3, -2, -1, 1, 2, 3, 4 };
    const int num_shifts = sizeof(shifts) / sizeof(shifts[0]);

    for (int i = 0; i < ROWS; ++i) {
        uint64_t handwritten_row = HANDWRITTEN_IMAGE[i];
        uint64_t reference_row = REFERENCE_IMAGE[i];

        for (int j = 0; j < num_shifts; ++j) {
            int shift_amount = shifts[j];
            uint64_t distance_val = 0; // POPCNT output will be stored here

            if (shift_amount < 0) {
                // --- ASSEMBLY BLOCK for LEFT SHIFT (SHL) ---
                int absolute_shift = -shift_amount;
                
                // Constraints:
                // "=r" (distance_val): Output variable, in a general-purpose register
                // "r" (handwritten_row): Input 1 (the source row)
                // "r" (reference_row): Input 2 (the target for XOR)
                // "c" ((uint8_t)absolute_shift): Input 3, forces shift amount into CL register
                // Clobbers: "rax" (used for intermediate calculation)
                __asm__ volatile (
                    "movq %1, %%rax\n\t"   // Load handwritten_row into RAX (Input %1)
                    "shlq %%cl, %%rax\n\t" // Left Shift RAX by absolute_shift (in CL)
                    "xorq %2, %%rax\n\t"   // XOR RAX with reference_row (Input %2)
                    "popcntq %%rax, %0\n\t"// Count set bits (Hamming Dist) in RAX, store in distance_val (Output %0)
                    : "=r" (distance_val)
                    : "r" (handwritten_row),
                      "r" (reference_row),
                      "c" ((uint8_t)absolute_shift) 
                    : "rax"
                );

            } else {
                // --- ASSEMBLY BLOCK for RIGHT SHIFT (SHR) ---
                
                // Constraints are the same as above. CL contains the positive shift_amount.
                __asm__ volatile (
                    "movq %1, %%rax\n\t"   // Load handwritten_row into RAX (Input %1)
                    "shrq %%cl, %%rax\n\t" // Logical Right Shift RAX by shift_amount (in CL)
                    "xorq %2, %%rax\n\t"   // XOR RAX with reference_row (Input %2)
                    "popcntq %%rax, %0\n\t"// Count set bits (Hamming Dist) in RAX, store in distance_val (Output %0)
                    : "=r" (distance_val)
                    : "r" (handwritten_row),
                      "r" (reference_row),
                      "c" ((uint8_t)shift_amount) 
                    : "rax"
                );
            }

            distances[result_index++] = (uint32_t)distance_val;
        }
    }
}

/**
 * @brief Summarizes and prints the calculated distance statistics.
 */
void print_stats(void) {
    uint32_t min_dist = UINT32_MAX;
    uint32_t max_dist = 0;
    uint64_t sum_dist = 0;
    int count = TOTAL_COMPARISONS;

    printf("--- Deformation Distance Summary ---\n");
    printf("Image Size: %d x %d bits (%d total rows)\n", BITS_PER_ROW, ROWS, ROWS);
    printf("Deformations Tested per Row: 8 (Shifts of +/- 1, 2, 3, 4 bits)\n");
    printf("Total Comparisons: %d\n\n", count);

    // Calculate overall statistics and print detailed results for the first few rows
    for (int i = 0; i < count; ++i) {
        uint32_t dist = distances[i];
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
        sum_dist += dist;

        if (i < 8 * 4) { // Print details for the first 4 rows
            // Calculate the shift value for printing based on index
            int shift_value = (i % 8) < 4 ? -4 + (i % 8) : (i % 8) - 3; 
            printf("Row %2d, Shift %+d: Distance = %u bits\n", 
                   (i / 8) + 1, shift_value, dist);
        }
    }
    
    if (count > 8 * 4) {
        printf("[... %d more results not shown ...]\n", count - (8 * 4));
    }

    double avg_dist = (double)sum_dist / count;

    printf("\n--- Overall Statistics (Hamming Distance) ---\n");
    printf("Minimum Distance Found: %u bits\n", min_dist);
    printf("Maximum Distance Found: %u bits\n", max_dist);
    printf("Average Distance: %.2f bits\n", avg_dist);
    printf("\nNote: This code requires a compiler that supports embedded assembly (like Clang/GCC) \nand the POPCNT instruction (requires -msse4.2 flag or equivalent on Intel x86-64).\n");
}

int main() {
    initialize_synthetic_data();
    
    calculate_distances();
    
    print_stats();
    
    return 0;
}