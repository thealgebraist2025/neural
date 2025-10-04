#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

// --- 1. Constants and Type Definitions (OCaml-style clarity) ---

#define IMAGE_SIZE 128
#define MAX_INSTRUCTIONS 15
#define PIXEL_MAX 255
#define NUM_TRAINING_IMAGES 16
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

// Enum for the three instruction types
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
    int radius; // Used only for CIRCLE (max 20 to keep it manageable)
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
 * @brief Clamps a value between a minimum and maximum.
 * @param val The value to clamp.
 * @param min The minimum allowed value.
 * @param max The maximum allowed value.
 * @return The clamped value.
 */
static inline int clamp_int(const int val, const int min, const int max) {
    if (val < min) return min;
    if (val > max) return max;
    return val;
}

// --- 3. Core Graphics Primitives (Drawing) ---

/**
 * @brief Draws a line between two points using Bresenham's algorithm.
 * @param img The image to draw on.
 * @param p1 Starting point.
 * @param p2 Ending point.
 * @param intensity The pixel value to draw.
 */
static void draw_line(Image* const img, const Point p1, const Point p2, const Pixel intensity) {
    int x1 = p1.x, y1 = p1.y, x2 = p2.x, y2 = p2.y;
    int dx = abs(x2 - x1);
    int sx = x1 < x2 ? 1 : -1;
    int dy = -abs(y2 - y1);
    int sy = y1 < y2 ? 1 : -1;
    int err = dx + dy; // error
    int e2;

    while (1) {
        set_pixel(img, x1, y1, intensity);
        if (x1 == x2 && y1 == y2) break;
        e2 = 2 * err;
        if (e2 >= dy) { // x step
            err += dy;
            x1 += sx;
        }
        if (e2 <= dx) { // y step
            err += dx;
            y1 += sy;
        }
    }
}

/**
 * @brief Draws a simple filled circle.
 * @param img The image to draw on.
 * @param center The center point.
 * @param radius The radius of the circle.
 * @param intensity The pixel value to draw.
 */
static void draw_circle(Image* const img, const Point center, const int radius, const Pixel intensity) {
    if (radius <= 0) return;
    const int r2 = radius * radius;
    // Iterate over the bounding box
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
 * @param drawing The drawing program (list of instructions).
 * @return A newly rendered Image.
 */
static Image* render_drawing(const Drawing* const drawing) {
    Image* img = create_image();
    Point current_pos = {0, 0}; // Starts at top-left
    int i;

    for (i = 0; i < drawing->count; i++) {
        const Instruction* instr = &drawing->instructions[i];
        const Pixel intensity = instr->intensity;

        switch (instr->type) {
            case INSTR_MOVE:
                // Only updates the current position
                current_pos = instr->p;
                break;
            case INSTR_LINE:
                // Draws a line from current_pos to instr->p
                draw_line(img, current_pos, instr->p, intensity);
                current_pos = instr->p; // Update current position
                break;
            case INSTR_CIRCLE:
                // Draws a circle centered at instr->p
                draw_circle(img, instr->p, instr->radius, intensity);
                // NOTE: Circle does not change the drawing current_pos in this model
                break;
            case INSTR_TYPE_COUNT:
                // Should not happen
                break;
        }
    }
    return img;
}

/**
 * @brief Calculates the Mean Squared Error (MSE) between two images.
 * @param img1 First image (e.g., rendered).
 * @param img2 Second image (e.g., target).
 * @return The MSE as a float.
 */
static float calculate_error(const Image* const img1, const Image* const img2) {
    double sum_squared_error = 0.0;
    const int total_pixels = IMAGE_SIZE * IMAGE_SIZE;
    int i;

    for (i = 0; i < total_pixels; i++) {
        // Difference is clamped to -255 to 255. Squared difference up to 65025.
        const int diff = (int)img1->data[i] - (int)img2->data[i];
        sum_squared_error += (double)diff * diff;
    }
    // MSE is the average of the squared errors
    return (float)(sum_squared_error / total_pixels);
}

// --- 4. Simulated Annealing Logic ---

/**
 * @brief Mock function to generate the target image (Handwritten 'A').
 * Note: In a real scenario, this would load data from disk.
 * @return A pointer to a mock target Image.
 */
static Image* generate_mock_target_A(void) {
    Image* img = create_image();
    const Pixel black = 255; // Black is 255
    const Point center = {IMAGE_SIZE / 2, IMAGE_SIZE / 2};
    const int half_height = IMAGE_SIZE / 3;

    // Draw left leg of 'A'
    draw_line(img, (Point){center.x - 20, center.y + half_height}, (Point){center.x, center.y - half_height}, black);
    // Draw right leg of 'A'
    draw_line(img, (Point){center.x, center.y - half_height}, (Point){center.x + 20, center.y + half_height}, black);
    // Draw crossbar
    draw_line(img, (Point){center.x - 15, center.y + 10}, (Point){center.x + 15, center.y + 10}, black);

    // Add some random noise to simulate "handwritten" variations
    srand((unsigned int)time(NULL));
    for (int i = 0; i < IMAGE_SIZE * IMAGE_SIZE / 100; i++) {
        set_pixel(img, rand() % IMAGE_SIZE, rand() % IMAGE_SIZE, (Pixel)(rand() % 100 + 155));
    }

    return img;
}

/**
 * @brief Simple heuristic to generate a starting 'A' drawing.
 * @return A Drawing structure approximating the letter 'A'.
 */
static Drawing generate_initial_drawing(void) {
    Drawing d;
    d.count = 5;
    const Pixel black = 255;
    const Coord cx = IMAGE_SIZE / 2;
    const Coord cy = IMAGE_SIZE / 2;
    const int hh = IMAGE_SIZE / 3;

    // 1. Move to start of left leg
    d.instructions[0] = (Instruction){INSTR_MOVE, {cx - 20, cy + hh}, 0, 0};
    // 2. Draw left leg up
    d.instructions[1] = (Instruction){INSTR_LINE, {cx, cy - hh}, 0, black};
    // 3. Draw right leg down
    d.instructions[2] = (Instruction){INSTR_LINE, {cx + 20, cy + hh}, 0, black};
    // 4. Move to crossbar start
    d.instructions[3] = (Instruction){INSTR_MOVE, {cx - 15, cy + 10}, 0, 0};
    // 5. Draw crossbar
    d.instructions[4] = (Instruction){INSTR_LINE, {cx + 15, cy + 10}, 0, black};

    // Initialize remaining instructions to MOVE to prevent null-pointer issues later
    for (int i = d.count; i < MAX_INSTRUCTIONS; i++) {
        d.instructions[i] = (Instruction){INSTR_MOVE, {0, 0}, 0, 0};
    }

    printf("\nHeuristic guess generated (5 instructions).\n");
    return d;
}

/**
 * @brief Generates a random valid Point.
 * @return A random Point.
 */
static Point random_point(void) {
    return (Point){rand() % IMAGE_SIZE, rand() % IMAGE_SIZE};
}

/**
 * @brief Generates a neighboring state by randomly mutating the current drawing.
 * @param current The current drawing state.
 * @return A new Drawing (neighbor state).
 */
static Drawing mutate_drawing(const Drawing* const current) {
    Drawing next = *current;
    const int mutation_type = rand() % 3; // 0: Modify, 1: Add, 2: Remove

    if (mutation_type == 0 && next.count > 0) { // Modify a random instruction
        const int idx = rand() % next.count;
        Instruction* instr = &next.instructions[idx];
        const int param_to_change = rand() % 4; // 0:type, 1:p, 2:radius, 3:intensity

        if (param_to_change == 0) {
            // Change instruction type (skip MOVE, as it's often a necessary start)
            instr->type = (InstructionType)((rand() % (INSTR_TYPE_COUNT - 1)) + 1);
        } else if (param_to_change == 1) {
            // Change target point
            instr->p = random_point();
        } else if (instr->type == INSTR_CIRCLE && param_to_change == 2) {
            // Change radius (only for CIRCLE)
            instr->radius = rand() % 21; // Max radius 20
        } else if (param_to_change == 3) {
            // Change intensity (must be dark)
            instr->intensity = (Pixel)(rand() % 100 + 155); // Keep it dark
        }
    } else if (mutation_type == 1 && next.count < MAX_INSTRUCTIONS) { // Add instruction
        const int insert_idx = rand() % (next.count + 1);
        // Shift existing instructions down
        memmove(&next.instructions[insert_idx + 1], &next.instructions[insert_idx],
                (next.count - insert_idx) * sizeof(Instruction));
        next.count++;

        // Insert a new random instruction
        Instruction* new_instr = &next.instructions[insert_idx];
        new_instr->type = (InstructionType)(rand() % INSTR_TYPE_COUNT);
        new_instr->p = random_point();
        new_instr->radius = (new_instr->type == INSTR_CIRCLE) ? (rand() % 21) : 0;
        new_instr->intensity = (new_instr->type != INSTR_MOVE) ? (Pixel)(rand() % 100 + 155) : 0;

    } else if (mutation_type == 2 && next.count > 1) { // Remove instruction (keep at least one)
        const int remove_idx = rand() % next.count;
        // Shift existing instructions up
        memmove(&next.instructions[remove_idx], &next.instructions[remove_idx + 1],
                (next.count - remove_idx - 1) * sizeof(Instruction));
        next.count--;
    }

    return next;
}

/**
 * @brief Calculates the acceptance probability for Simulated Annealing.
 * @param old_error Current error (cost).
 * @param new_error Neighbor error (cost).
 * @param temp Current temperature.
 * @return Probability (0.0 to 1.0).
 */
static float acceptance_probability(const float old_error, const float new_error, const float temp) {
    if (new_error < old_error) {
        return 1.0f; // Always accept better solutions
    }
    // Accept worse solutions with a decreasing probability over time
    // The exponent is (old - new) / temp
    return (float)exp((double)(old_error - new_error) / temp);
}

// --- 5. Sanity & Unit Tests ---

/**
 * @brief Runs a single unit test.
 * @param condition The condition to test.
 * @param message The message to print on success/failure.
 * @return 1 on success, 0 on failure.
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
 * @brief Tests the image creation and pixel setting utility.
 */
static int test_image_utilities(void) {
    printf("\n--- Test: Image Utilities ---\n");
    Image* img = create_image();
    int success = 0;

    // Test 1: Initialization to white (0)
    success += run_test(img->data[0] == 0 && img->data[IMAGE_SIZE * IMAGE_SIZE - 1] == 0,
                        "Image initialized to 0 (white)");

    // Test 2: Set pixel within bounds
    set_pixel(img, 10, 20, 255);
    success += run_test(img->data[20 * IMAGE_SIZE + 10] == 255,
                        "Pixel set successfully within bounds");

    // Test 3: Set pixel outside bounds (should not change anything)
    set_pixel(img, IMAGE_SIZE + 1, 10, 100);
    success += run_test(img->data[10 * IMAGE_SIZE + IMAGE_SIZE + 1] != 100,
                        "Pixel set ignored outside bounds");

    free(img);
    return success;
}

/**
 * @brief Tests the error calculation function.
 */
static int test_error_calculation(void) {
    printf("\n--- Test: Error Calculation ---\n");
    Image* img1 = create_image();
    Image* img2 = create_image();
    int success = 0;

    // Test 1: Identical images (error should be 0)
    float error1 = calculate_error(img1, img2);
    success += run_test(fabs(error1 - 0.0f) < 1e-6, "Error is 0 for identical images");

    // Test 2: Completely black vs completely white images (max error)
    memset(img1->data, 255, IMAGE_SIZE * IMAGE_SIZE * sizeof(Pixel));
    // Max diff is 255, max squared diff is 255*255 = 65025
    float expected_max_error = 65025.0f;
    float error2 = calculate_error(img1, img2);
    success += run_test(fabs(error2 - expected_max_error) < 1e-6, "Error is MAX for completely different images");

    // Test 3: Simple one-pixel difference
    memset(img1->data, 0, IMAGE_SIZE * IMAGE_SIZE * sizeof(Pixel));
    img1->data[1] = 10;
    float expected_error3 = (10.0f * 10.0f) / (IMAGE_SIZE * IMAGE_SIZE);
    float error3 = calculate_error(img1, img2);
    success += run_test(fabs(error3 - expected_error3) < 1e-6, "Error calculated correctly for single pixel change");

    free(img1);
    free(img2);
    return success;
}

/**
 * @brief Main function to run all sanity and unit tests.
 * @return Total number of successful tests.
 */
static int run_tests(void) {
    int total_success = 0;
    total_success += test_image_utilities();
    total_success += test_error_calculation();
    return total_success;
}

// --- 6. Main Program Execution ---

int main(void) {
    // Seed the random number generator
    srand((unsigned int)time(NULL));

    printf("--- Drawing Heuristic and Simulated Annealing Optimizer ---\n");
    int successful_tests = run_tests();
    printf("\nTotal successful tests: %d / 6\n", successful_tests);

    if (successful_tests < 6) {
        printf("\nTests failed. Halting optimization.\n");
        return EXIT_FAILURE;
    }

    // --- Setup ---
    Image* target_img = generate_mock_target_A(); // The ideal 'A' we aim for

    // 1. Initial Heuristic Guess
    Drawing current_drawing = generate_initial_drawing();
    Image* current_img = render_drawing(&current_drawing);
    float current_error = calculate_error(current_img, target_img);

    // Initial logging
    printf("Initial Heuristic Error (MSE): %.2f\n", current_error);
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
    const float cooling_rate = 0.9999f;
    float temp = initial_temp;

    // Use a small buffer to manage the drawing and image pointers
    Image* next_img = NULL;
    Drawing next_drawing;

    while (1) {
        current_time = time(NULL);
        elapsed_time = (float)difftime(current_time, start_time);

        if (elapsed_time >= SA_RUNTIME_SECONDS) {
            break; // Stop after the allotted time
        }

        // 1. Temperature update (Exponential decay)
        temp = initial_temp * (1.0f - (elapsed_time / SA_RUNTIME_SECONDS));
        if (temp < 0.001f) temp = 0.001f;

        // 2. Generate a neighbor state
        next_drawing = mutate_drawing(&current_drawing);
        next_img = render_drawing(&next_drawing);
        float new_error = calculate_error(next_img, target_img);

        // 3. Acceptance criterion
        if (acceptance_probability(current_error, new_error, temp) > ((float)rand() / RAND_MAX)) {
            // Accept the new state
            free(current_img); // Clean up the old image
            current_drawing = next_drawing;
            current_error = new_error;
            current_img = next_img;

            // Check for new best solution
            if (current_error < best_error) {
                best_error = current_error;
                best_drawing = current_drawing;
            }
        } else {
            // Reject the new state, clean up the temporary image
            free(next_img);
        }

        iteration++;

        // 4. Logging output every 5 seconds
        if (difftime(current_time, last_log_time) >= SA_LOG_INTERVAL_SECONDS) {
            printf("| Time: %3.0fs / %ds | Iteration: %7d | T: %7.2f | Current Error: %7.2f | Best Error: %7.2f |\n",
                   elapsed_time, SA_RUNTIME_SECONDS, iteration, temp, current_error, best_error);
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
    free(current_img); // current_img holds the image of the last accepted state (or best if rejected)

    return EXIT_SUCCESS;
}