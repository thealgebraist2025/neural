#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <setjmp.h> // Required for libjpeg error handling

// Include the external libjpeg library headers
#include <jpeglib.h>

// --- 1. Constants and Type Definitions ---

#define IMAGE_SIZE 128          // The primary dimension for the image (128x128)
#define SMALL_GRID_SIZE 16      // Size for the downscaled grid used for initial tracing
#define PIXEL_MAX 255
#define THRESHOLD 128           // Grayscale value for tracing heuristics
#define MAX_INSTRUCTIONS 1024
#define SA_RUNTIME_SECONDS 180  // Optimization runtime
#define SA_LOG_INTERVAL_SECONDS 5
#define JPEG_OUTPUT_FILENAME "generated.jpg" // Updated output filename
#define JPEG_INPUT_FILENAME "a.jpg" // Mandatory input file name

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

/**
 * @brief Structure for the 128x128 8-bit image (The final target and rendered output)
 */
typedef struct {
    Pixel data[IMAGE_SIZE * IMAGE_SIZE];
} Image;

// Structure for image data loaded from JPEG before resizing
typedef struct {
    Pixel *data;
    int width;
    int height;
    int components; // 1 for grayscale, 3 for RGB
} LoadedImage;

// Array type for the 16x16 grid used internally for tracing/sampling
typedef Pixel SmallImage[SMALL_GRID_SIZE * SMALL_GRID_SIZE];

// Forward declarations
static void draw_line(Image* const img, const Point p1, const Point p2, const Pixel intensity);

// --- 2. JPEG Error Handling ---

// Custom error handling structure for libjpeg
struct my_error_mgr {
    struct jpeg_error_mgr pub;
    jmp_buf setjmp_buffer;
};

typedef struct my_error_mgr * my_error_ptr;

// Error exit routine that uses setjmp/longjmp
METHODDEF(void)
my_error_exit (j_common_ptr cinfo)
{
  my_error_ptr myerr = (my_error_ptr) cinfo->err;
  (*cinfo->err->output_message) (cinfo);
  longjmp(myerr->setjmp_buffer, 1);
}

// --- 3. Utility Functions ---

/**
 * @brief Allocates and initializes an Image struct with all pixels set to white (0).
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
 * @brief Loads a JPEG file, decompresses it, and resizes it to the target IMAGE_SIZE x IMAGE_SIZE.
 * @return The 128x128 Image struct, or NULL on failure.
 */
static Image* load_and_resize_target(const char* filename) {
    struct jpeg_decompress_struct cinfo;
    struct my_error_mgr jerr;
    FILE *infile;
    JSAMPROW row_pointer[1];
    Image* target_img = NULL;
    LoadedImage raw_img = {NULL, 0, 0, 0};

    // Step 1: Open the input file
    if ((infile = fopen(filename, "rb")) == NULL) {
        fprintf(stderr, "[ERROR] Can't open %s\n", filename);
        return NULL;
    }

    // Step 2: Initialize the JPEG decompression object
    // FIX for MemorySanitizer: Zero-initialize the structure before libjpeg uses it.
    memset(&cinfo, 0, sizeof(cinfo));
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = my_error_exit;

    if (setjmp(jerr.setjmp_buffer)) {
        // If we get here, the JPEG code has signaled an error.
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        if (raw_img.data) free(raw_img.data);
        if (target_img) free(target_img);
        return NULL;
    }

    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);

    // Step 3: Read file parameters
    (void)jpeg_read_header(&cinfo, TRUE);

    // Force output to grayscale for simplicity (1 component)
    cinfo.out_color_space = JCS_GRAYSCALE;
    
    // Step 4: Start decompressor
    (void)jpeg_start_decompress(&cinfo);
    
    raw_img.width = cinfo.output_width;
    raw_img.height = cinfo.output_height;
    raw_img.components = cinfo.output_components;
    long total_pixels = (long)raw_img.width * raw_img.height * raw_img.components;

    // Allocate buffer for the entire raw image data
    raw_img.data = (Pixel*)malloc(total_pixels * sizeof(Pixel));
    if (!raw_img.data) {
        perror("[ERROR] Failed to allocate memory for raw image data");
        longjmp(jerr.setjmp_buffer, 1); // Jump to error exit
    }

    int row_stride = raw_img.width * raw_img.components;
    Pixel* buffer_ptr = raw_img.data;

    // Step 5: Read scanlines
    while (cinfo.output_scanline < cinfo.output_height) {
        row_pointer[0] = buffer_ptr + (cinfo.output_scanline * row_stride);
        (void)jpeg_read_scanlines(&cinfo, row_pointer, 1);
    }

    // Step 6: Finish decompression and cleanup
    (void)jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    
    // --- Step 7: Resize/Resample to 128x128 using averaging ---
    target_img = create_image();
    const float scale_x = (float)raw_img.width / IMAGE_SIZE;
    const float scale_y = (float)raw_img.height / IMAGE_SIZE;

    for (int ty = 0; ty < IMAGE_SIZE; ty++) {
        for (int tx = 0; tx < IMAGE_SIZE; tx++) {
            long sum = 0;
            long count = 0;

            // Define the block in the source image corresponding to target pixel (tx, ty)
            const int sx_start = (int)(tx * scale_x);
            const int sy_start = (int)(ty * scale_y);
            // The current code implements a fixed-size block averaging which is robust:
            const int block_w = (int)scale_x > 0 ? (int)scale_x : 1;
            const int block_h = (int)scale_y > 0 ? (int)scale_y : 1;

            for (int sy = sy_start; sy < sy_start + block_h && sy < raw_img.height; sy++) {
                for (int sx = sx_start; sx < sx_start + block_w && sx < raw_img.width; sx++) {
                    sum += raw_img.data[sy * raw_img.width + sx];
                    count++;
                }
            }
            
            // Set the target pixel to the average value
            if (count > 0) {
                // Since the input A is white background/black A, we keep 0=white, 255=black.
                target_img->data[ty * IMAGE_SIZE + tx] = (Pixel)(sum / count);
            }
        }
    }
    
    printf("[INFO] Successfully loaded and resized %s from %dx%d (Grayscale) to %dx%d.\n",
           filename, raw_img.width, raw_img.height, IMAGE_SIZE, IMAGE_SIZE);

    free(raw_img.data);
    return target_img;
}

/**
 * @brief Downscales a 128x128 image to a 16x16 grid using simple averaging.
 */
static void downscale_image(const Image* const large_img, SmallImage* const small_img) {
    const int TARGET_SIZE = SMALL_GRID_SIZE;
    const int SCALE_FACTOR = IMAGE_SIZE / TARGET_SIZE; // 8

    for (int sy = 0; sy < TARGET_SIZE; sy++) {
        for (int sx = 0; sx < TARGET_SIZE; sx++) {
            long sum = 0;
            const int block_pixels = SCALE_FACTOR * SCALE_FACTOR;

            for (int ly = sy * SCALE_FACTOR; ly < (sy + 1) * SCALE_FACTOR; ly++) {
                for (int lx = sx * SCALE_FACTOR; lx < (sx + 1) * SCALE_FACTOR; lx++) {
                    sum += large_img->data[ly * IMAGE_SIZE + lx];
                }
            }
            // Use simple integer division for averaging
            (*small_img)[sy * TARGET_SIZE + sx] = (Pixel)(sum / block_pixels);
        }
    }
}

// --- 4. Core Graphics Primitives (Drawing & Rendering) ---

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

// --- 5. Simulated Annealing Heuristics & Mutations ---

/**
 * @brief Generates the initial drawing by tracing dark pixels in a 16x16 downscaled image.
 */
static Drawing generate_initial_drawing(const Image* const target_img) {
    Drawing d;

    // Explicitly zero the entire drawing structure
    memset(&d, 0, sizeof(Drawing));

    d.count = 0;
    SmallImage small_img;
    downscale_image(target_img, &small_img);

    const int TARGET_SIZE = SMALL_GRID_SIZE;
    const int SCALE_FACTOR = IMAGE_SIZE / TARGET_SIZE; // 8

    const Pixel trace_intensity = 220;
    const int pixel_step = SCALE_FACTOR;

    // Trace Connections (4-neighborhood: Right, Down)
    for (int sy = 0; sy < TARGET_SIZE; sy++) {
        for (int sx = 0; sx < TARGET_SIZE; sx++) {
            if (small_img[sy * TARGET_SIZE + sx] >= THRESHOLD) {
                // Point A (Start of segment in 128x128 scale)
                const Point A = {sx * pixel_step + pixel_step / 2, sy * pixel_step + pixel_step / 2};

                // Check right neighbor (Horizontal connection)
                if (sx < TARGET_SIZE - 1 && small_img[sy * TARGET_SIZE + sx + 1] >= THRESHOLD) {
                    const Point B = {(sx + 1) * pixel_step + pixel_step / 2, sy * pixel_step + pixel_step / 2};
                    if (d.count + 2 <= MAX_INSTRUCTIONS) {
                        d.instructions[d.count++] = (Instruction){INSTR_MOVE, A, 0, 0};
                        d.instructions[d.count++] = (Instruction){INSTR_LINE, B, 0, trace_intensity};
                    }
                }

                // Check down neighbor (Vertical connection)
                if (sy < TARGET_SIZE - 1 && small_img[(sy + 1) * TARGET_SIZE + sx] >= THRESHOLD) {
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
            // Change instruction type
            instr->type = (InstructionType)(rand() % INSTR_TYPE_COUNT);
        } else if (param_to_change == 1) {
            // Change point location
            instr->p = random_point();
        } else if (instr->type == INSTR_CIRCLE && param_to_change == 2) {
            // Change circle radius
            instr->radius = rand() % 21;
        } else if (instr->type != INSTR_MOVE && param_to_change == 3) {
            // Change intensity (non-MOVE instructions only)
            instr->intensity = (Pixel)(rand() % 100 + 155);
        }
    } else if (mutation_type == 1 && next.count < MAX_INSTRUCTIONS) { // Add
        // Insert a new random instruction
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
        // Remove a random instruction
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
    // E^(delta_E / T)
    return (float)exp((double)(old_error - new_error) / temp);
}

// --- 6. JPEG Output (Using libjpeg) ---

/**
 * @brief Saves a grayscale Image to a high-quality JPEG file using libjpeg.
 */
static void save_jpeg_grayscale(const Image* const img, const Drawing* const best_drawing, const float final_error) {
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE *outfile;
    JSAMPROW row_pointer[1];
    int row_stride; 

    const int quality = 90; // High quality setting

    // Step 1: Initialize JPEG compression object
    // Initialize the compression struct to zero for safety.
    memset(&cinfo, 0, sizeof(cinfo));
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    // Step 2: Open the output file
    if ((outfile = fopen(JPEG_OUTPUT_FILENAME, "wb")) == NULL) {
        fprintf(stderr, "[ERROR] Can't open %s for writing\n", JPEG_OUTPUT_FILENAME);
        return;
    }
    jpeg_stdio_dest(&cinfo, outfile);

    // Step 3: Set compression parameters
    cinfo.image_width = IMAGE_SIZE;
    cinfo.image_height = IMAGE_SIZE;
    cinfo.input_components = 1; // Grayscale image
    cinfo.in_color_space = JCS_GRAYSCALE;

    // Calculate row_stride
    row_stride = IMAGE_SIZE * cinfo.input_components; // Bytes per row for grayscale (1 byte/pixel)

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE); // Set the desired quality

    // Step 4: Start compressor
    jpeg_start_compress(&cinfo, TRUE);

    // Step 5: Write scanlines
    while (cinfo.next_scanline < cinfo.image_height) {
        // FIX 2: Add explicit cast to JSAMPROW to resolve 'discards qualifiers' warning
        row_pointer[0] = (JSAMPROW)&img->data[cinfo.next_scanline * row_stride];
        (void)jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    // Step 6: Finish compression and close file
    jpeg_finish_compress(&cinfo);
    fclose(outfile);

    // Step 7: Release JPEG compression object
    jpeg_destroy_compress(&cinfo);


    printf("\nSuccessfully saved best rendered image to %s (JPEG, Quality %d).\n", JPEG_OUTPUT_FILENAME, quality);
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
}


// --- 7. Main Program Execution ---

int main(void) {
    // Seed the random number generator
    srand((unsigned int)time(NULL));

    printf("--- Drawing Heuristic and Simulated Annealing Optimizer (libjpeg) ---\n");
    printf("--- Target image loading from: %s ---\n", JPEG_INPUT_FILENAME);

    // --- Setup ---
    // Generate the target image by loading and resizing the input JPEG
    Image* target_img = load_and_resize_target(JPEG_INPUT_FILENAME);
    
    if (target_img == NULL) {
        fprintf(stderr, "[FATAL] Failed to load target image. Exiting.\n");
        return EXIT_FAILURE;
    }

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
            break; // Time limit reached
        }

        // 1. Temperature update (Linear cooling schedule)
        temp = initial_temp * (1.0f - (elapsed_time / SA_RUNTIME_SECONDS));
        if (temp < 0.001f) temp = 0.001f;

        // 2. Generate a neighbor state
        next_drawing = mutate_drawing(&current_drawing);
        next_img = render_drawing(&next_drawing);
        float new_error = calculate_error(next_img, target_img);

        // 3. Acceptance criterion
        const float prob = (float)rand() / (float)RAND_MAX;
        if (acceptance_probability(current_error, new_error, temp) > prob) {
            // Accept the new state (Accept a better state, or a worse state based on probability)
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

        // 4. Logging output every SA_LOG_INTERVAL_SECONDS
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

    // Use the libjpeg writer function
    save_jpeg_grayscale(best_img, &best_drawing, best_error);

    // Clean up memory
    free(target_img);
    free(current_img);
    free(best_img);

    return EXIT_SUCCESS;
}