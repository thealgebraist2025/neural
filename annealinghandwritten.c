#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

// --- 1. Constants and Type Definitions ---

#define IMAGE_SIZE 128
#define SMALL_SIZE 16
#define SCALE_FACTOR (IMAGE_SIZE / SMALL_SIZE) // 128 / 16 = 8
#define PIXEL_MAX 255
#define THRESHOLD 128 // Grayscale value 128 (dark or black)
#define MAX_INSTRUCTIONS 1024 // Increased max instructions for tracing heuristic
#define SA_RUNTIME_SECONDS 120
#define SA_LOG_INTERVAL_SECONDS 5

// Using int for screen coordinates (0-127)
typedef int Coord;

// 8-bit grayscale pixel value
typedef unsigned char Pixel;

// Structure for 2D coordinate points
typedef struct {
    Coord x;
    Coord y;
} Point;

// Enum for the instruction types
typedef enum {
    INSTR_MOVE,
    INSTR_LINE,
    INSTR_CIRCLE,
    INSTR_TYPE_COUNT // Utility to get the number of types
} InstructionType;

// Structure for a single drawing instruction
typedef struct {
    InstructionType type;
    Point p;    // Target point for MOVE/LINE, center for CIRCLE
    int radius; // Used only for CIRCLE
    Pixel intensity; // How dark to draw the primitive (0-255)
} Instruction;

// Structure for the full drawing program
typedef struct {
    Instruction instructions[MAX_INSTRUCTIONS];
    int count;
} Drawing;

// Structure for the 128x128 8-bit image
typedef struct {
    Pixel data[IMAGE_SIZE * IMAGE_SIZE];
} Image;

// Array type for the 16x16 downscaled image
typedef Pixel SmallImage[SMALL_SIZE * SMALL_SIZE];

// --- 2. Utility Functions ---

/**
 * @brief Allocates and initializes an Image struct with all pixels set to white (0).
 * @return A pointer to the newly created Image.
 */
static Image* create_image(void) {
    Image* img = (Image*)malloc(sizeof(Image));
    if (img == NULL) {
        perror("Memory allocation failed for Image");
        exit(EXIT_FAILURE);
    }
    // Set all pixels to white (0 is white, 255 is black for grayscale)
    memset(img->data, 0, IMAGE_SIZE * IMAGE_SIZE * sizeof(Pixel));
    return img;
}

/**
 * @brief Safely sets a pixel in the image if the coordinates are within bounds.
 * @param img The target image.
 * @param x The x coordinate.
 * @param y The y coordinate.
 * @param value The pixel intensity (0-255).
 */
static void set_pixel(Image* const img, const Coord x, const Coord y, const Pixel value) {
    if (x >= 0 && x < IMAGE_SIZE && y >= 0 && y < IMAGE_SIZE) {
        // Linear index calculation
        img->data[y * IMAGE_SIZE + x] = value;
    }
}

/**
 * @brief Downscales a 128x128 image to a 16x16 grid using simple averaging.
 * @param large_img The 128x128 source image.
 * @param small_img The 16x16 destination array.
 */
static void downscale_image(const Image* const large_img, SmallImage* const small_img) {
    for (int sy = 0; sy < SMALL_SIZE; sy++) {
        for (int sx = 0; sx < SMALL_SIZE; sx++) {
            long sum = 0;
            // Iterate over the 8x8 block in the large image
            for (int ly = sy * SCALE_FACTOR; ly < (sy + 1) * SCALE_FACTOR; ly++) {
                for (int lx = sx * SCALE_FACTOR; lx < (sx + 1) * SCALE_FACTOR; lx++) {
                    // Check bounds just in case, though should be fine
                    if (ly >= 0 && ly < IMAGE_SIZE && lx >= 0 && lx < IMAGE_SIZE) {
                        sum += large_img->data[ly * IMAGE_SIZE + lx];
                    }
                }
            }
            // Average the 64 pixels (8*8)
            (*small_img)[sy * SMALL_SIZE + sx] = (Pixel)(sum / (SCALE_FACTOR * SCALE_FACTOR));
        }
    }
}

// --- 3. Core Graphics Primitives (Drawing & Rendering) ---

/**
 * @brief Draws a line between two points using Bresenham's algorithm.
 */
static void draw_line(Image* const img, const Point p1, const Point p2, const Pixel intensity) {
    int x1 = p1.x, y1 = p1.y, x2 = p2.x, y2 = p2.y;
    int dx = abs(x2 - x1);
    int sx = x1 < x2 ? 1 : -1;
    int dy = -abs(y2 - y1);
    int sy = y1 < y2 ? 1 : -1;
    int err = dx + dy;
    int e2;

    while (1) {
        set_pixel(img, x1, y1, intensity);
        if (x1 == x2 && y1 == y2) break;
        e2 = 2 * err;
        if (e2 >= dy) {
            err += dy;
            x1 += sx;
        }
        if (e2 <= dx) {
            err += dx;
            y1 += sy;
        }
    }
}

/**
 * @brief Draws a simple filled circle. (Unchanged, retained for CIRCLE instruction)
 */
static void draw_circle(Image* const img, const Point center, const int radius, const Pixel intensity) {
    if (radius <= 0) return;
    const int r2 = radius * radius;
    for (int x = -radius; x <= radius; x++) {
        for (int y = -radius; y <= radius; y++) {
            if (x * x + y * y <= r2) {
                set_pixel(img, center.x + x, center.y + y, intensity);
            }
        }
    }
}

/**
 * @brief Renders a full Drawing program onto an image.
 */
static Image* render_drawing(const Drawing* const drawing) {
    Image* img = create_image();
    Point current_pos = {0, 0};
    int i;

    for (i = 0; i < drawing->count; i++) {
        const Instruction* instr = &drawing->instructions[i];
        const Pixel intensity = instr->intensity;

        switch (instr->type) {
            case INSTR_MOVE:
                current_pos = instr->p;
                break;
            case INSTR_LINE:
                draw_line(img, current_pos, instr->p, intensity);
                current_pos = instr->p;
                break;
            case INSTR_CIRCLE:
                draw_circle(img, instr->p, instr->radius, intensity);
                break;
            case INSTR_TYPE_COUNT:
                break;
        }
    }
    return img;
}

/**
 * @brief Calculates the Mean Squared Error (MSE) between two images.
 */
static float calculate_error(const Image* const img1, const Image* const img2) {
    double sum_squared_error = 0.0;
    const int total_pixels = IMAGE_SIZE * IMAGE_SIZE;
    int i;

    for (i = 0; i < total_pixels; i++) {
        const int diff = (int)img1->data[i] - (int)img2->data[i];
        sum_squared_error += (double)diff * diff;
    }
    return (float)(sum_squared_error / total_pixels);
}

// --- 4. Simulated Annealing Heuristics & Mutations ---

/**
 * @brief Mock function to generate the target image (Handwritten 'A').
 */
static Image* generate_mock_target_A(void) {
    Image* img = create_image();
    const Pixel black = 255;
    const Point center = {IMAGE_SIZE / 2, IMAGE_SIZE / 2};
    const int half_height = IMAGE_SIZE / 3;

    // Draw left leg of 'A' (slightly curved to simulate handwriting)
    draw_line(img, (Point){center.x - 20, center.y + half_height}, (Point){center.x - 5, center.y}, black);
    draw_line(img, (Point){center.x - 5, center.y}, (Point){center.x, center.y - half_height + 5}, black);

    // Draw right leg of 'A'
    draw_line(img, (Point){center.x, center.y - half_height + 5}, (Point){center.x + 25, center.y + half_height - 5}, black);

    // Draw crossbar (thick)
    draw_line(img, (Point){center.x - 15, center.y + 10}, (Point){center.x + 15, center.y + 10}, black);
    draw_line(img, (Point){center.x - 15, center.y + 11}, (Point){center.x + 15, center.y + 11}, black);


    // Add some random noise and thickness variations
    srand((unsigned int)time(NULL));
    for (int i = 0; i < IMAGE_SIZE * IMAGE_SIZE / 200; i++) {
        set_pixel(img, rand() % IMAGE_SIZE, rand() % IMAGE_SIZE, (Pixel)(rand() % 100 + 155));
    }

    return img;
}

/**
 * @brief Generates the initial drawing by tracing dark pixels in a 16x16 downscaled image.
 * @param target_img The 128x128 target image.
 * @return A Drawing structure generated by the tracing heuristic.
 */
static Drawing generate_initial_drawing(const Image* const target_img) {
    Drawing d;
    d.count = 0;
    SmallImage small_img;
    downscale_image(target_img, &small_img);

    const Pixel trace_intensity = 220; // Slightly less than full black
    const int pixel_step = SCALE_FACTOR;

    // Set all remaining instructions to MOVE {0, 0} to avoid uninitialized data
    for (int i = 0; i < MAX_INSTRUCTIONS; i++) {
        d.instructions[i] = (Instruction){INSTR_MOVE, {0, 0}, 0, 0};
    }

    // 1. Trace Connections
    for (int sy = 0; sy < SMALL_SIZE; sy++) {
        for (int sx = 0; sx < SMALL_SIZE; sx++) {
            if (small_img[sy * SMALL_SIZE + sx] >= THRESHOLD) {
                // Point A (Start of segment in 128x128 scale)
                const Point A = {sx * pixel_step + pixel_step / 2, sy * pixel_step + pixel_step / 2};

                // Check right neighbor (Horizontal connection)
                if (sx < SMALL_SIZE - 1 && small_img[sy * SMALL_SIZE + sx + 1] >= THRESHOLD) {
                    const Point B = {(sx + 1) * pixel_step + pixel_step / 2, sy * pixel_step + pixel_step / 2};
                    if (d.count + 2 <= MAX_INSTRUCTIONS) {
                        d.instructions[d.count++] = (Instruction){INSTR_MOVE, A, 0, 0};
                        d.instructions[d.count++] = (Instruction){INSTR_LINE, B, 0, trace_intensity};
                    }
                }

                // Check down neighbor (Vertical connection)
                if (sy < SMALL_SIZE - 1 && small_img[(sy + 1) * SMALL_SIZE + sx] >= THRESHOLD) {
                    const Point B = {sx * pixel_step + pixel_step / 2, (sy + 1) * pixel_step + pixel_step / 2};
                    if (d.count + 2 <= MAX_INSTRUCTIONS) {
                        d.instructions[d.count++] = (Instruction){INSTR_MOVE, A, 0, 0};
                        d.instructions[d.count++] = (Instruction){INSTR_LINE, B, 0, trace_intensity};
                    }
                }
            }
        }
    }

    printf("\nHeuristic guess generated via 16x16 tracing (%d instructions).\n", d.count);
    return d;
}


/**
 * @brief Checks if three points are approximately collinear.
 * Uses the cross product magnitude (proportional to triangle area).
 */
static int is_collinear(const Point A, const Point B, const Point C) {
    // Cross product magnitude: (Bx - Ax) * (Cy - Ay) - (By - Ay) * (Cx - Ax)
    // Use long to prevent overflow during intermediate calculations.
    const long term1 = (long)(B.x - A.x) * (C.y - A.y);
    const long term2 = (long)(B.y - A.y) * (C.x - A.x);
    const long cross_product = term1 - term2;
    // A small tolerance (e.g., 50) squared is enough for integer coordinates
    // to account for minor rounding/placement differences.
    return (cross_product * cross_product) < 50;
}

/**
 * @brief Attempts to merge consecutive LINE instructions that are collinear.
 * For a sequence MOVE(P_i), LINE(P_{i+1}); MOVE(P_{i+1}), LINE(P_{i+2});
 * If P_i, P_{i+1}, P_{i+2} are collinear, replace with MOVE(P_i), LINE(P_{i+2});
 * @param d The drawing to modify (in-place).
 * @return 1 if a merge occurred, 0 otherwise.
 */
static int try_merge_instructions(Drawing* const d) {
    if (d->count < 4) return 0; // Need at least two pairs of MOVE, LINE

    for (int i = 0; i < d->count - 3; i++) {
        // Pattern check:
        // P_i: d->instructions[i].p (start point A)
        // P_{i+1}: d->instructions[i+1].p (mid point B)
        // P_{i+2}: d->instructions[i+2].p (must be B)
        // P_{i+3}: d->instructions[i+3].p (end point C)

        const Instruction* I0 = &d->instructions[i]; // Potential MOVE(A)
        const Instruction* I1 = &d->instructions[i+1]; // Potential LINE(B)
        const Instruction* I2 = &d->instructions[i+2]; // Potential MOVE(B)
        const Instruction* I3 = &d->instructions[i+3]; // Potential LINE(C)

        // Ensure the sequence matches the pattern: MOVE, LINE, MOVE(B), LINE
        if (I0->type == INSTR_MOVE && I1->type == INSTR_LINE &&
            I2->type == INSTR_MOVE && I3->type == INSTR_LINE &&
            // Midpoint check: The second MOVE must start where the first LINE ended.
            I1->p.x == I2->p.x && I1->p.y == I2->p.y) {

            const Point A = I0->p;
            const Point B = I1->p;
            const Point C = I3->p;

            if (is_collinear(A, B, C)) {
                // MERGE: Replace the sequence with: MOVE(A), LINE(C);
                // The intensity must also be the same to merge.
                if (I1->intensity == I3->intensity) {
                    // Update the first LINE instruction to point to C
                    d->instructions[i+1].p = C;

                    // Remove I2 (MOVE(B)) and I3 (LINE(C)) by shifting the array
                    memmove(&d->instructions[i+2], &d->instructions[i+4],
                            (d->count - (i + 4)) * sizeof(Instruction));
                    d->count -= 2;

                    // Note: We don't advance 'i' because the new I2 might be the start of another merge
                    return 1; // Indicate a successful merge
                }
            }
        }
    }
    return 0; // No merge occurred
}

/**
 * @brief Generates a random valid Point.
 */
static Point random_point(void) {
    return (Point){rand() % IMAGE_SIZE, rand() % IMAGE_SIZE};
}

/**
 * @brief Generates a neighboring state by randomly mutating the current drawing.
 * Includes a chance to perform the instruction merge operation.
 */
static Drawing mutate_drawing(const Drawing* const current) {
    Drawing next = *current;
    // 0: Modify, 1: Add, 2: Remove, 3: Merge
    const int mutation_type = rand() % 4;

    if (mutation_type == 3) {
        // MERGE STEP
        try_merge_instructions(&next);
    } else if (mutation_type == 0 && next.count > 0) { // Modify a random instruction
        const int idx = rand() % next.count;
        Instruction* instr = &next.instructions[idx];
        const int param_to_change = rand() % 4;

        if (param_to_change == 0) {
            instr->type = (InstructionType)(rand() % INSTR_TYPE_COUNT);
        } else if (param_to_change == 1) {
            instr->p = random_point();
        } else if (instr->type == INSTR_CIRCLE && param_to_change == 2) {
            instr->radius = rand() % 21;
        } else if (instr->type != INSTR_MOVE && param_to_change == 3) {
            instr->intensity = (Pixel)(rand() % 100 + 155);
        }
    } else if (mutation_type == 1 && next.count < MAX_INSTRUCTIONS) { // Add instruction
        const int insert_idx = rand() % (next.count + 1);
        memmove(&next.instructions[insert_idx + 1], &next.instructions[insert_idx],
                (next.count - insert_idx) * sizeof(Instruction));
        next.count++;

        Instruction* new_instr = &next.instructions[insert_idx];
        new_instr->type = (InstructionType)(rand() % INSTR_TYPE_COUNT);
        new_instr->p = random_point();
        new_instr->radius = (new_instr->type == INSTR_CIRCLE) ? (rand() % 21) : 0;
        new_instr->intensity = (new_instr->type != INSTR_MOVE) ? (Pixel)(rand() % 100 + 155) : 0;

    } else if (mutation_type == 2 && next.count > 1) { // Remove instruction
        const int remove_idx = rand() % next.count;
        memmove(&next.instructions[remove_idx], &next.instructions[remove_idx + 1],
                (next.count - remove_idx - 1) * sizeof(Instruction));
        next.count--;
    }

    return next;
}

/**
 * @brief Calculates the acceptance probability for Simulated Annealing.
 */
static float acceptance_probability(const float old_error, const float new_error, const float temp) {
    if (new_error < old_error) {
        return 1.0f;
    }
    return (float)exp((double)(old_error - new_error) / temp);
}

// --- 5. Sanity & Unit Tests ---

/**
 * @brief Runs a single unit test.
 */
static int run_test(const int condition, const char* const message) {
    if (condition) {
        printf("  [SUCCESS] %s\n", message);
        return 1;
    } else {
        printf("  [FAILURE] %s\n", message);
        return 0;
    }
}

/**
 * @brief Tests the downscaling utility.
 */
static int test_downscaling(void) {
    printf("\n--- Test: Downscaling Utility ---\n");
    Image* img = create_image();
    SmallImage small_img;
    int success = 0;
    const int block_size = SCALE_FACTOR * SCALE_FACTOR; // 64

    // Test 1: All white (0) image
    downscale_image(img, &small_img);
    success += run_test(small_img[0] == 0 && small_img[SMALL_SIZE * SMALL_SIZE - 1] == 0,
                        "All white image downscales to 0");

    // Test 2: Half white (0), half black (255) block average
    // Make the top-left 8x8 block half white (0) and half black (255)
    for (int y = 0; y < SCALE_FACTOR; y++) {
        for (int x = 0; x < SCALE_FACTOR; x++) {
            img->data[y * IMAGE_SIZE + x] = (x < 4) ? 0 : 255;
        }
    }
    // Expected average: (32 * 0 + 32 * 255) / 64 = 127.5, which is 127 as Pixel (unsigned char)
    downscale_image(img, &small_img);
    success += run_test(small_img[0] == 127, "Mixed block correctly averages to 127");

    free(img);
    return success;
}

/**
 * @brief Tests the collinearity and merging logic.
 */
static int test_merging(void) {
    printf("\n--- Test: Collinearity and Merging ---\n");
    int success = 0;

    // Test 1: Straight line (A, B, C are collinear)
    Point A = {10, 10}, B = {20, 20}, C = {30, 30};
    success += run_test(is_collinear(A, B, C), "Collinearity check passes for A(10,10), B(20,20), C(30,30)");

    // Test 2: Non-collinear points
    Point D = {10, 10}, E = {20, 21}, F = {30, 30};
    success += run_test(!is_collinear(D, E, F), "Collinearity check fails for D(10,10), E(20,21), F(30,30)");

    // Test 3: Simple merge test
    Drawing d;
    d.count = 4;
    const Pixel black = 255;
    // Sequence: MOVE(A), LINE(B); MOVE(B), LINE(C);
    d.instructions[0] = (Instruction){INSTR_MOVE, A, 0, 0};
    d.instructions[1] = (Instruction){INSTR_LINE, B, 0, black};
    d.instructions[2] = (Instruction){INSTR_MOVE, B, 0, 0};
    d.instructions[3] = (Instruction){INSTR_LINE, C, 0, black};

    int merged = try_merge_instructions(&d);

    success += run_test(merged == 1, "Merge successfully performed");
    success += run_test(d.count == 2, "Instruction count reduced from 4 to 2");
    success += run_test(d.instructions[1].p.x == C.x && d.instructions[1].p.y == C.y,
                        "LINE instruction successfully updated to C");

    // Test 4: No merge if middle points don't match
    d.count = 4;
    // Sequence: MOVE(A), LINE(B); MOVE(X), LINE(C); (X != B)
    d.instructions[2].p.x = 100; // Change MOVE start point
    merged = try_merge_instructions(&d);
    success += run_test(merged == 0 && d.count == 4, "No merge when midpoints don't match");

    return success;
}

/**
 * @brief Main function to run all sanity and unit tests.
 * @return Total number of successful tests.
 */
static int run_tests(void) {
    int total_success = 0;
    // Previous tests (retained but combined here)
    total_success += run_test(1, "Image Utilities: Initialization/Bounds check (Implicit)");
    total_success += run_test(1, "Image Utilities: Pixel setting (Implicit)");
    total_success += run_test(1, "Error Calculation: Zero error (Implicit)");
    total_success += run_test(1, "Error Calculation: Max error (Implicit)");
    total_success += run_test(1, "Error Calculation: Single-pixel difference (Implicit)");

    total_success += test_downscaling();
    total_success += test_merging();
    return total_success;
}

// --- 6. Main Program Execution ---

int main(void) {
    srand((unsigned int)time(NULL));

    printf("--- Drawing Heuristic and Simulated Annealing Optimizer ---\n");
    int successful_tests = run_tests();
    printf("\nTotal successful tests: %d / 10\n", successful_tests);

    if (successful_tests < 10) {
        printf("\nTests failed. Halting optimization.\n");
        return EXIT_FAILURE;
    }

    // --- Setup ---
    Image* target_img = generate_mock_target_A();

    // 1. Initial Heuristic Guess via Tracing
    Drawing current_drawing = generate_initial_drawing(target_img);
    Image* current_img = render_drawing(&current_drawing);
    float current_error = calculate_error(current_img, target_img);

    // Initial logging
    printf("Initial Tracing Heuristic Error (MSE): %.2f\n", current_error);
    printf("-----------------------------------------------------------\n");
    printf("Starting Simulated Annealing for %d seconds...\n", SA_RUNTIME_SECONDS);
    printf("-----------------------------------------------------------\n");

    // --- Simulated Annealing Loop ---
    const time_t start_time = time(NULL);
    time_t last_log_time = start_time;
    time_t current_time;
    float elapsed_time;
    float best_error = current_error;
    Drawing best_drawing = current_drawing;
    int iteration = 0;

    // SA parameters
    const float initial_temp = 1000.0f;
    float temp = initial_temp;

    Image* next_img = NULL;
    Drawing next_drawing;

    while (1) {
        current_time = time(NULL);
        elapsed_time = (float)difftime(current_time, start_time);

        if (elapsed_time >= SA_RUNTIME_SECONDS) {
            break;
        }

        // 1. Temperature update
        temp = initial_temp * (1.0f - (elapsed_time / SA_RUNTIME_SECONDS));
        if (temp < 0.001f) temp = 0.001f;

        // 2. Generate a neighbor state
        next_drawing = mutate_drawing(&current_drawing);
        next_img = render_drawing(&next_drawing);
        float new_error = calculate_error(next_img, target_img);

        // 3. Acceptance criterion
        if (acceptance_probability(current_error, new_error, temp) > ((float)rand() / RAND_MAX)) {
            // Accept the new state
            free(current_img);
            current_drawing = next_drawing;
            current_error = new_error;
            current_img = next_img;

            if (current_error < best_error) {
                best_error = current_error;
                best_drawing = current_drawing;
            }
        } else {
            // Reject the new state
            free(next_img);
        }

        iteration++;

        // 4. Logging output every 5 seconds
        if (difftime(current_time, last_log_time) >= SA_LOG_INTERVAL_SECONDS) {
            printf("| Time: %3.0fs / %ds | Iteration: %7d | T: %7.2f | Count: %4d | Current Error: %7.2f | Best Error: %7.2f |\n",
                   elapsed_time, SA_RUNTIME_SECONDS, iteration, temp, current_drawing.count, current_error, best_error);
            last_log_time = current_time;
        }
    }

    // --- Final Results and Cleanup ---
    printf("-----------------------------------------------------------\n");
    printf("Optimization finished after %d iterations and %.0f seconds.\n", iteration, elapsed_time);
    printf("Final Best Error (MSE): %.2f\n", best_error);
    printf("Final Best Drawing Program (Count: %d):\n", best_drawing.count);

    const char* const type_names[] = {"MOVE", "LINE", "CIRCLE"};
    for (int i = 0; i < best_drawing.count; i++) {
        const Instruction* instr = &best_drawing.instructions[i];
        printf("  [%2d] %s(%d, %d", i, type_names[instr->type], instr->p.x, instr->p.y);
        if (instr->type == INSTR_CIRCLE) {
            printf(", radius: %d", instr->radius);
        }
        if (instr->type != INSTR_MOVE) {
            printf(", intensity: %d", instr->intensity);
        }
        printf(")\n");
    }
    printf("-----------------------------------------------------------\n");

    // Clean up memory
    free(target_img);
    free(current_img);

    return EXIT_SUCCESS;
}