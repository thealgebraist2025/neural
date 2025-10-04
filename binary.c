#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

// FIX: Define M_PI manually as it is not guaranteed to be included in <math.h>
#define M_PI 3.14159265358979323846

// Define image dimensions
#define SIZE 64
#define BITS_TOTAL (SIZE * SIZE)
#define BYTES_PER_ROW (SIZE / 8)

// --- Image Data ---
// 64 rows of 64 bits (8 bytes) each.
uint64_t SOURCE_IMAGE[SIZE];
uint64_t DEST_IMAGE[SIZE];

// --- Lookup Table (LUT) ---
// The rotation map: For every source bit index (0 to 4095), this array stores
// the global destination bit index (0 to 4095) after a 45-degree rotation.
uint16_t ROTATION_MAP[BITS_TOTAL];


/**
 * @brief Pre-calculates the 45-degree rotation map.
 */
void precalculate_rotation_map(void) {
    const double angle_rad = M_PI / 4.0; // 45 degrees
    const double cos_a = cos(angle_rad);
    const double sin_a = sin(angle_rad);
    const double center = (double)(SIZE - 1) / 2.0;

    for (int y = 0; y < SIZE; ++y) {
        for (int x = 0; x < SIZE; ++x) {
            // Translate origin to center
            double dx = x - center;
            double dy = y - center;

            // Apply rotation matrix
            double x_prime = dx * cos_a - dy * sin_a;
            double y_prime = dx * sin_a + dy * cos_a;

            // Translate back and round to nearest integer pixel
            int x_out = (int)round(x_prime + center);
            int y_out = (int)round(y_prime + center);

            // Clamp coordinates to stay within the 64x64 bounds
            if (x_out >= 0 && x_out < SIZE && y_out >= 0 && y_out < SIZE) {
                // Source index: i = y * 64 + x
                int source_index = y * SIZE + x;
                // Destination index: D_i = y_out * 64 + x_out
                int dest_index = y_out * SIZE + x_out;
                ROTATION_MAP[source_index] = (uint16_t)dest_index;
            } else {
                // Map out-of-bounds bits to 0
                int source_index = y * SIZE + x;
                ROTATION_MAP[source_index] = 0; 
            }
        }
    }
}

/**
 * @brief Initializes the source image with a recognizable pattern (a diagonal line).
 */
void initialize_synthetic_data(void) {
    for (int i = 0; i < SIZE; ++i) {
        // Create a synthetic diagonal line pattern (Bit i in row i is set)
        SOURCE_IMAGE[i] = (1ULL << i) | (1ULL << (SIZE - 1 - i));
    }
}

/**
 * @brief Rotates the image 45 degrees using the pre-calculated map and 
 * optimized Bit Test/Set assembly instructions.
 */
void rotate_image_45_degrees(void) {
    // Ensure the destination image is cleared before starting the rotation
    for (int i = 0; i < SIZE; ++i) {
        DEST_IMAGE[i] = 0ULL;
    }

    uint64_t *src_ptr = SOURCE_IMAGE;
    uint64_t *dst_ptr = DEST_IMAGE;
    uint16_t *map_ptr = ROTATION_MAP;

    // Loop through all 4096 source bits
    for (int i = 0; i < BITS_TOTAL; ++i) {
        uint16_t dest_index = map_ptr[i];

        // FIX: Introduce 64-bit temporaries to force Clang to assign 64-bit registers
        // to the source index inputs, resolving the MOVQ/MOVL conflict error.
        uint64_t i_64 = i; 
        uint64_t dest_index_64 = dest_index; 

        // --- Optimized Assembly Block (Minimal Bitwise Operations) ---
        // Constraint mapping (Input list starts at %0):
        // %0: i_64 (source index, uint64_t)
        // %1: src_ptr (Source pointer)
        // %2: dst_ptr (Destination pointer)
        // %3: dest_index_64 (Destination index, uint64_t)

        __asm__ volatile (
            // 1. Setup Source Bit Index (i) and Destination Bit Index (dest_index)
            // Use MOVQ with the 64-bit temporary variables.
            "movq %0, %%rdx\n\t"        // RDX = i_64 (source bit index 0-4095)
            "movq %3, %%rdi\n\t"        // RDI = dest_index_64 (destination bit index 0-4095)

            // --- Get Source Bit Value (using BTQ) ---
            // Calculate R_src offset: R_src = RDX / 64. R8 = 8 * R_src.
            "movq %%rdx, %%r8\n\t"
            "shrq $6, %%r8\n\t"         // R8 holds R_src (Row Index)
            "imulq $8, %%r8, %%r8\n\t"  // R8 holds byte offset of the source word
            
            // BTQ requires a 64-bit memory operand. RDX holds the index (implicit modulo 64).
            "btq (%1, %%r8), %%rdx\n\t"   // Test bit RDX % 64 at address src_ptr + R8
            
            // --- Check and Set Destination Bit Value (using JNC/BTSQ) ---
            "jnc 1f\n\t"                // Jump if Carry Flag is NOT set (bit was 0)

            // Bit was 1, so we must set the destination bit.
            // Calculate R_dst offset: R_dst = RDI / 64. R9 = 8 * R_dst.
            "movq %%rdi, %%r9\n\t"
            "shrq $6, %%r9\n\t"         // R9 holds R_dst (Row Index)
            "imulq $8, %%r9, %%r9\n\t"  // R9 holds byte offset of the destination word
            
            // BTSQ requires a 64-bit memory operand. RDI holds the index (implicit modulo 64).
            "btsq (%2, %%r9), %%rdi\n\t" 

            "1:\n\t" // Label for jump instruction (if bit was 0)

            : // No outputs modified in this block
            : "r" (i_64),           // %0 Source index (i)
              "r" (src_ptr),        // %1 Source pointer
              "r" (dst_ptr),        // %2 Destination pointer
              "r" (dest_index_64)   // %3 Destination index (dest_index)
            : "rdx", "rdi", "r8", "r9", "cc", "memory" // Clobbered: CC for Carry Flag and memory
        );
    }
}

/**
 * @brief Prints the image as 0s and 1s for verification (only first 8 rows).
 */
void print_image(const char *name, const uint64_t image[SIZE]) {
    printf("\n--- %s (First 8 Rows) ---\n", name);
    for (int r = 0; r < 8; ++r) {
        printf("Row %2d: ", r);
        // Print 64 bits from MSB to LSB
        for (int c = SIZE - 1; c >= 0; --c) {
            if ((image[r] >> c) & 1ULL) {
                printf("1");
            } else {
                printf("0");
            }
        }
        printf("\n");
    }
}

int main() {
    // 1. Pre-calculation (Computational Heavy, Done Once)
    precalculate_rotation_map();
    initialize_synthetic_data();

    // 2. Runtime Rotation (Minimal Bitwise/Arithmetic Instructions)
    rotate_image_45_degrees();

    // 3. Verification
    print_image("Source Image", SOURCE_IMAGE);
    print_image("Rotated Image (45 Degrees)", DEST_IMAGE);

    printf("\nSuccess. The rotation function minimizes explicit bitwise operations\n");
    printf("by relying on x86-64 assembly instructions BT (Bit Test) and BTS (Bit Test and Set).\n");

    return 0;
}