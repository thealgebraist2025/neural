#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

// --- LIBPNG DEPENDENCY ---
// Explicitly include libpng for PNG file generation.
#include <png.h>

// --- 1. Constants and Type Definitions ---

#define IMAGE_SIZE 128
#define SMALL_SIZE 16
#define SCALE_FACTOR (IMAGE_SIZE / SMALL_SIZE) // 128 / 16 = 8
#define PIXEL_MAX 255
#define THRESHOLD 128 // Grayscale value 128 (dark or black)
#define MAX_INSTRUCTIONS 1024
#define SA_RUNTIME_SECONDS 120
#define SA_LOG_INTERVAL_SECONDS 5
#define PNG_FILENAME "annealing.png" // PNG file extension

// Using int for screen coordinates (0-127)
typedef int Coord;

// 8-bit grayscale pixel value (0=White, 255=Black)
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
    INSTR_TYPE_COUNT
} InstructionType;

// Structure for a single drawing instruction
typedef struct {
    InstructionType type;
    Point p;
    int radius;
    Pixel intensity;
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

// Forward declaration for drawing primitive
static void draw_line(Image* const img, const Point p1, const Point p2, const Pixel intensity);

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
 */
static void set_pixel(Image* const img, const Coord x, const Coord y, const Pixel value) {
    if (x >= 0 && x < IMAGE_SIZE && y >= 0 && y < IMAGE_SIZE) {
        img->data[y * IMAGE_SIZE + x] = value;
    }
}

/**
 * @brief Downscales a 128x128 image to a 16x16 grid using simple averaging.
 */
static void downscale_image(const Image* const large_img, SmallImage* const small_img) {
    for (int sy = 0; sy < SMALL_SIZE; sy++) {
        for (int sx = 0; sx < SMALL_SIZE; sx++) {
            long sum = 0;
            const int block_pixels = SCALE_FACTOR * SCALE_FACTOR;

            for (int ly = sy * SCALE_FACTOR; ly < (sy + 1) * SCALE_FACTOR; ly++) {
                for (int lx = sx * SCALE_FACTOR; lx < (sx + 1) * SCALE_FACTOR; lx++) {
                    sum += large_img->data[ly * IMAGE_SIZE + lx];
                }
            }
            (*small_img)[sy * SMALL_SIZE + sx] = (Pixel)(sum / block_pixels);
        }
    }
}

/**
 * @brief Hardcoded 128x128 image simulating a handwritten 'A'.
 * This acts as the target image for optimization.
 */
static Image* generate_handwritten_A_target(void) {
    Image* img = create_image();
    const Pixel black = 255;
    const Pixel gray = 180;
    const Point center = {IMAGE_SIZE / 2, IMAGE_SIZE / 2};
    const int hh = IMAGE_SIZE / 3;

    // Left leg (wavy)
    draw_line(img, (Point){center.x - 25, center.y + hh + 5}, (Point){center.x - 10, center.y}, black);
    draw_line(img, (Point){center.x - 10, center.y + 1}, (Point){center.x, center.y - hh + 5}, gray);

    // Right leg (thick)
    draw_line(img, (Point){center.x + 3, center.y - hh + 7}, (Point){center.x + 30, center.y + hh}, black);
    draw_line(img, (Point){center.x + 4, center.y - hh + 7}, (Point){center.x + 31, center.y + hh}, black);

    // Crossbar (slanted)
    draw_line(img, (Point){center.x - 18, center.y + 15}, (Point){center.x + 8, center.y + 8}, black);

    // Add some random noise and thickness variations
    srand((unsigned int)time(NULL));
    for (int i = 0; i < IMAGE_SIZE * IMAGE_SIZE / 200; i++) {
        set_pixel(img, rand() % IMAGE_SIZE, rand() % IMAGE_SIZE, (Pixel)(rand() % 80 + 175));
    }

    return img;
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
 * @brief Draws a simple filled circle.
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
 * @return A newly rendered Image.
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
 * @brief Generates the initial drawing by tracing dark pixels in a 16x16 downscaled image.
 */
static Drawing generate_initial_drawing(const Image* const target_img) {
    Drawing d;
    d.count = 0;
    SmallImage small_img;
    downscale_image(target_img, &small_img);

    const Pixel trace_intensity = 220;
    const int pixel_step = SCALE_FACTOR;

    // Set all remaining instructions to MOVE {0, 0} to avoid uninitialized data
    for (int i = 0; i < MAX_INSTRUCTIONS; i++) {
        d.instructions[i] = (Instruction){INSTR_MOVE, {0, 0}, 0, 0};
    }

    // Trace Connections (4-neighborhood: Right, Down)
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

    return d;
}

/**
 * @brief Checks if three points are approximately collinear.
 */
static int is_collinear(const Point A, const Point B, const Point C) {
    // Cross product magnitude (proportional to triangle area)
    const long term1 = (long)(B.x - A.x) * (C.y - A.y);
    const long term2 = (long)(B.y - A.y) * (C.x - A.x);
    const long cross_product = term1 - term2;
    // Tolerance (50) squared to account for small deviations
    return (cross_product * cross_product) < 50;
}

/**
 * @brief Attempts to merge consecutive LINE instructions that are collinear.
 * @return 1 if a merge occurred, 0 otherwise.
 */
static int try_merge_instructions(Drawing* const d) {
    if (d->count < 4) return 0;

    for (int i = 0; i < d->count - 3; i++) {
        const Instruction* I0 = &d->instructions[i];   // MOVE(A)
        const Instruction* I1 = &d->instructions[i+1]; // LINE(B)
        const Instruction* I2 = &d->instructions[i+2]; // MOVE(B)
        const Instruction* I3 = &d->instructions[i+3]; // LINE(C)

        if (I0->type == INSTR_MOVE && I1->type == INSTR_LINE &&
            I2->type == INSTR_MOVE && I3->type == INSTR_LINE &&
            I1->p.x == I2->p.x && I1->p.y == I2->p.y &&
            I1->intensity == I3->intensity) { // Check midpoints and intensity

            const Point A = I0->p;
            const Point B = I1->p;
            const Point C = I3->p;

            if (is_collinear(A, B, C)) {
                // MERGE: Replace sequence with MOVE(A), LINE(C)
                d->instructions[i+1].p = C;

                // Remove I2 (MOVE(B)) and I3 (LINE(C))
                memmove(&d->instructions[i+2], &d->instructions[i+4],
                        (d->count - (i + 4)) * sizeof(Instruction));
                d->count -= 2;

                return 1; // Indicate a successful merge
            }
        }
    }
    return 0;
}

/**
 * @brief Generates a random valid Point.
 */
static Point random_point(void) {
    return (Point){rand() % IMAGE_SIZE, rand() % IMAGE_SIZE};
}

/**
 * @brief Generates a neighboring state by randomly mutating the current drawing.
 */
static Drawing mutate_drawing(const Drawing* const current) {
    Drawing next = *current;
    // 0: Modify, 1: Add, 2: Remove, 3: Merge
    const int mutation_type = rand() % 4;

    if (mutation_type == 3) {
        // MERGE STEP (Prioritize merging to reduce complexity)
        try_merge_instructions(&next);
    } else if (mutation_type == 0 && next.count > 0) { // Modify
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
    } else if (mutation_type == 1 && next.count < MAX_INSTRUCTIONS) { // Add
        const int insert_idx = rand() % (next.count + 1);
        memmove(&next.instructions[insert_idx + 1], &next.instructions[insert_idx],
                (next.count - insert_idx) * sizeof(Instruction));
        next.count++;

        Instruction* new_instr = &next.instructions[insert_idx];
        new_instr->type = (InstructionType)(rand() % INSTR_TYPE_COUNT);
        new_instr->p = random_point();
        new_instr->radius = (new_instr->type == INSTR_CIRCLE) ? (rand() % 21) : 0;
        new_instr->intensity = (new_instr->type != INSTR_MOVE) ? (Pixel)(rand() % 100 + 155) : 0;

    } else if (mutation_type == 2 && next.count > 1) { // Remove
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

// --- 5. PNG Output using libpng ---

/**
 * @brief Converts the two 128x128 grayscale images into a single 128x256 RGB buffer.
 */
static unsigned char* convert_to_rgb_buffer(const Image* const target, const Image* const rendered) {
    const int total_height = IMAGE_SIZE * 2;
    const int total_pixels = IMAGE_SIZE * total_height;
    // 3 color components (RGB) per pixel
    unsigned char* buffer = (unsigned char*)malloc(total_pixels * 3);
    if (!buffer) {
        perror("Failed to allocate RGB buffer");
        exit(EXIT_FAILURE);
    }

    // Pointers for buffer access
    unsigned char* ptr = buffer;

    // 1. Convert Target Image (Top half)
    for (int i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++) {
        // Grayscale pixel value (0=White, 255=Black) is used for R, G, B
        const Pixel gray = target->data[i];
        *ptr++ = gray; // R
        *ptr++ = gray; // G
        *ptr++ = gray; // B
    }

    // 2. Convert Rendered Image (Bottom half)
    for (int i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++) {
        // Grayscale pixel value (0=White, 255=Black) is used for R, G, B
        const Pixel gray = rendered->data[i];
        *ptr++ = gray; // R
        *ptr++ = gray; // G
        *ptr++ = gray; // B
    }

    return buffer;
}

/**
 * @brief Saves the composite RGB buffer to a PNG file using libpng.
 */
static void save_png_composite(const Image* const target, const Image* const rendered,
                               const Drawing* const best_drawing, const float final_error) {
    FILE *fp = fopen(PNG_FILENAME, "wb");
    if (!fp) {
        printf("[ERROR] Could not open file %s for binary writing.\n", PNG_FILENAME);
        return;
    }

    png_structp png_ptr = NULL;
    png_infop info_ptr = NULL;
    unsigned char* rgb_buffer = NULL;
    png_bytep *row_pointers = NULL;

    // 1. Setup PNG structures
    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) { goto cleanup; }

    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) { goto cleanup; }

    // 2. Set error handling (required for libpng)
    if (setjmp(png_jmpbuf(png_ptr))) {
        printf("[ERROR] Error during libpng I/O operation.\n");
        goto cleanup;
    }

    // 3. Init I/O
    png_init_io(png_ptr, fp);

    // 4. Set IHDR (Image Header)
    const int width = IMAGE_SIZE;
    const int height = IMAGE_SIZE * 2;
    const int bit_depth = 8;
    const int color_type = PNG_COLOR_TYPE_RGB; // 3 components (R, G, B)

    png_set_IHDR(png_ptr, info_ptr, width, height,
                 bit_depth, color_type, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    // 5. Write Info
    png_write_info(png_ptr, info_ptr);

    // 6. Prepare pixel data and row pointers
    rgb_buffer = convert_to_rgb_buffer(target, rendered);
    row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    if (!rgb_buffer || !row_pointers) { goto cleanup; }

    for (int y = 0; y < height; y++) {
        // Each row starts at y * (width * 3 bytes) offset in the RGB buffer
        row_pointers[y] = rgb_buffer + y * width * 3;
    }

    // 7. Write Data
    png_write_image(png_ptr, row_pointers);

    // 8. End Write
    png_write_end(png_ptr, NULL);

    printf("\nSuccessfully saved composite image to %s (using libpng).\n", PNG_FILENAME);
    printf("--- Final Results ---\n");
    printf("Final Mean Squared Error (MSE): %.2f\n", final_error);
    printf("Final Instruction Count: %d\n", best_drawing->count);
    printf("Final Instructions:\n");

    const char* const type_names[] = {"MOVE", "LINE", "CIRCLE"};
    for (int i = 0; i < best_drawing->count; i++) {
        const Instruction* instr = &best_drawing->instructions[i];
        printf("  [%2d] %s(%d, %d", i, type_names[instr->type], instr->p.x, instr->p.y);
        if (instr->type == INSTR_CIRCLE) {
            printf(", radius: %d", instr->radius);
        }
        if (instr->type != INSTR_MOVE) {
            printf(", intensity: %d", instr->intensity);
        }
        printf(")\n");
    }

cleanup:
    // 9. Cleanup
    if (fp) fclose(fp);
    if (rgb_buffer) free(rgb_buffer);
    if (row_pointers) free(row_pointers);
    if (png_ptr) {
        // png_destroy_write_struct safely handles both png_ptr and info_ptr
        png_destroy_write_struct(&png_ptr, &info_ptr);
    }
}


// --- 6. Sanity & Unit Tests (Retained for robustness) ---

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

    // Test 1: Mixed block correctly averages
    const int block_pixels = SCALE_FACTOR * SCALE_FACTOR; // 64
    for (int y = 0; y < SCALE_FACTOR; y++) {
        for (int x = 0; x < SCALE_FACTOR; x++) {
            img->data[y * IMAGE_SIZE + x] = (x < 4) ? 0 : 255;
        }
    }
    downscale_image(img, &small_img);
    success += run_test(small_img[0] == 127, "Mixed block averages to 127");

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
    const Point A = {10, 10}, B = {20, 20}, C = {30, 30};
    success += run_test(is_collinear(A, B, C), "Collinearity passes for A(10,10), B(20,20), C(30,30)");

    // Test 2: Simple merge test
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

    return success;
}

/**
 * @brief Main function to run all sanity and unit tests.
 */
static int run_tests(void) {
    int total_success = 0;
    total_success += test_downscaling(); // 1 test
    total_success += test_merging(); // 2 tests
    return total_success;
}

// --- 7. Main Program Execution ---

int main(void) {
    srand((unsigned int)time(NULL));

    printf("--- Drawing Heuristic and Simulated Annealing Optimizer (libpng Output) ---\n");
    int successful_tests = run_tests();
    printf("\nTotal successful tests: %d / 3\n", successful_tests);

    if (successful_tests < 3) {
        printf("\nTests failed. Halting optimization.\n");
        return EXIT_FAILURE;
    }

    // --- Setup ---
    Image* target_img = generate_handwritten_A_target();

    // 1. Initial Heuristic Guess via Tracing
    Drawing current_drawing = generate_initial_drawing(target_img);
    Image* current_img = render_drawing(&current_drawing);
    float current_error = calculate_error(current_img, target_img);

    printf("Initial Tracing Heuristic Error (MSE): %.2f\n", current_error);
    printf("Initial Instruction Count: %d\n", current_drawing.count);
    printf("-----------------------------------------------------------\n");
    printf("Starting Simulated Annealing for %d seconds...\n", SA_RUNTIME_SECONDS);
    printf("-----------------------------------------------------------\n");

    // --- Simulated Annealing Loop ---
    const time_t start_time = time(NULL);
    time_t last_log_time = start_time;
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
        time_t current_time = time(NULL);
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
        const float prob = (float)rand() / RAND_MAX;
        if (acceptance_probability(current_error, new_error, temp) > prob) {
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

    // Render the final best drawing to get the image for output
    Image* best_img = render_drawing(&best_drawing);

    // Use the libpng function to write the PNG file
    save_png_composite(target_img, best_img, &best_drawing, best_error);

    // Clean up memory
    free(target_img);
    free(current_img);
    free(best_img);

    return EXIT_SUCCESS;
}