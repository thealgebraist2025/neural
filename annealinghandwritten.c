#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <setjmp.h> 

// Include the external libjpeg library headers
#include <jpeglib.h>

// --- 1. Constants and Type Definitions ---

#define IMAGE_SIZE 128          // The primary dimension for the image (128x128)
#define SMALL_GRID_SIZE 16      // Size for the downscaled grid (16x16)
#define PIXEL_MAX 255
#define TRACE_THRESHOLD 64      // Grayscale value for pixel segmentation (pixels > 64 are "dark")
#define A_TEMPLATE_THRESHOLD 128 // Grayscale value for template comparison
#define MAX_INSTRUCTIONS 50     // Instruction limit
#define SA_RUNTIME_SECONDS 60   // Optimization runtime
#define SA_LOG_INTERVAL_SECONDS 5
#define JPEG_OUTPUT_FILENAME "generated.jpg"
#define JPEG_INPUT_FILENAME "a.jpg" // Mandatory input file name
#define A_TEMPLATE_WEIGHT 0.5f  // Weight for the 'A' shape template error

// Using int for screen coordinates (0-127)
typedef int Coord;

// 8-bit grayscale pixel value (0=White, 255=Black)
typedef unsigned char Pixel;

// Structure for 2D coordinate points
typedef struct {
    Coord x;
    Coord y;
} Point;

// Enum for the instruction types (Only MOVE and LINE)
typedef enum {
    INSTR_MOVE,
    INSTR_LINE,
    INSTR_TYPE_COUNT
} InstructionType;

// Structure for a single drawing instruction
typedef struct {
    InstructionType type;
    Point p;
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

// Hardcoded 16x16 template for the letter 'A' (1=Dark/Black, 0=Light/White)
const int A_TEMPLATE[SMALL_GRID_SIZE * SMALL_GRID_SIZE] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,
    0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, // Crossbar
    1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
    1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
    1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
    1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
    1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
    1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
    1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
};


// --- 2. JPEG Error Handling ---

struct my_error_mgr {
    struct jpeg_error_mgr pub;
    jmp_buf setjmp_buffer;
};
typedef struct my_error_mgr * my_error_ptr;

METHODDEF(void)
my_error_exit (j_common_ptr cinfo)
{
  my_error_ptr myerr = (my_error_ptr) cinfo->err;
  (*cinfo->err->output_message) (cinfo);
  longjmp(myerr->setjmp_buffer, 1);
}

// Forward declarations
static void draw_line(Image* const img, const Point p1, const Point p2, const Pixel intensity);
static Image* render_drawing(const Drawing* const drawing);
static float calculate_template_error(const Image* const rendered_img);
static void downscale_image(const Image* const large_img, SmallImage* const small_img);

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
        // We use the maximum value if the intensity is high to ensure we draw dark lines
        if (value > img->data[y * IMAGE_SIZE + x]) {
             img->data[y * IMAGE_SIZE + x] = value;
        }
    }
}

/**
 * @brief Downscales a 128x128 image to a 16x16 grid using simple averaging.
 */
static void downscale_image(const Image* const large_img, SmallImage* const small_img) {
    const int GRID_SIZE = SMALL_GRID_SIZE; // 16
    const int BLOCK_SIZE = IMAGE_SIZE / GRID_SIZE; // 8

    for (int sy = 0; sy < GRID_SIZE; sy++) {
        for (int sx = 0; sx < GRID_SIZE; sx++) {
            long sum = 0;
            const int block_pixels = BLOCK_SIZE * BLOCK_SIZE;

            // Define starting pixel in 128x128 grid
            const int start_y = sy * BLOCK_SIZE;
            const int start_x = sx * BLOCK_SIZE;

            for (int dy = 0; dy < BLOCK_SIZE; dy++) {
                const int ly = start_y + dy;
                for (int dx = 0; dx < BLOCK_SIZE; dx++) {
                    const int lx = start_x + dx;
                    
                    // Access is safe: max ly/lx is 120 + 7 = 127.
                    sum += large_img->data[ly * IMAGE_SIZE + lx];
                }
            }
            
            // Set the target pixel to the average value
            (*small_img)[sy * GRID_SIZE + sx] = (Pixel)(sum / block_pixels);
        }
    }
}

/**
 * @brief Loads a JPEG file, resizes it, and inverts colors to ensure black lines on white background.
 */
static Image* load_and_resize_target(const char* filename) {
    struct jpeg_decompress_struct cinfo;
    struct my_error_mgr jerr;
    FILE *infile;
    JSAMPROW row_pointer[1];
    Image* target_img = NULL;
    LoadedImage raw_img = {NULL, 0, 0, 0};

    if ((infile = fopen(filename, "rb")) == NULL) {
        fprintf(stderr, "[ERROR] Can't open %s\n", filename);
        return NULL;
    }
    memset(&cinfo, 0, sizeof(cinfo));
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = my_error_exit;

    if (setjmp(jerr.setjmp_buffer)) {
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        if (raw_img.data) free(raw_img.data);
        if (target_img) free(target_img);
        return NULL;
    }

    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    (void)jpeg_read_header(&cinfo, TRUE);
    cinfo.out_color_space = JCS_GRAYSCALE;
    (void)jpeg_start_decompress(&cinfo);
    
    raw_img.width = cinfo.output_width;
    raw_img.height = cinfo.output_height;
    raw_img.components = cinfo.output_components;
    long total_pixels_raw = (long)raw_img.width * raw_img.height * raw_img.components;

    raw_img.data = (Pixel*)malloc(total_pixels_raw * sizeof(Pixel));
    if (!raw_img.data) {
        perror("[ERROR] Failed to allocate memory for raw image data");
        longjmp(jerr.setjmp_buffer, 1);
    }

    int row_stride = raw_img.width * raw_img.components;
    Pixel* buffer_ptr = raw_img.data;

    while (cinfo.output_scanline < cinfo.output_height) {
        row_pointer[0] = buffer_ptr + (cinfo.output_scanline * row_stride);
        (void)jpeg_read_scanlines(&cinfo, row_pointer, 1);
    }

    (void)jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    
    target_img = create_image();
    const float scale_x = (float)raw_img.width / IMAGE_SIZE;
    const float scale_y = (float)raw_img.height / IMAGE_SIZE;
    
    // Resize with averaging
    for (int ty = 0; ty < IMAGE_SIZE; ty++) {
        for (int tx = 0; tx < IMAGE_SIZE; tx++) {
            long sum = 0;
            long count = 0;
            const int sx_start = (int)(tx * scale_x);
            const int int_scale_x = (int)scale_x;
            const int int_scale_y = (int)scale_y;
            const int block_w = int_scale_x > 0 ? int_scale_x : 1;
            const int block_h = int_scale_y > 0 ? int_scale_y : 1;
            const int sy_start = (int)(ty * scale_y);

            for (int sy = sy_start; sy < sy_start + block_h && sy < raw_img.height; sy++) {
                for (int sx = sx_start; sx < sx_start + block_w && sx < raw_img.width; sx++) {
                    sum += raw_img.data[sy * raw_img.width + sx];
                    count++;
                }
            }
            
            if (count > 0) {
                target_img->data[ty * IMAGE_SIZE + tx] = (Pixel)(sum / count);
            }
        }
    }
    
    // --- COLOR INVERSION LOGIC ---
    long total_brightness = 0;
    const int total_pixels_resized = IMAGE_SIZE * IMAGE_SIZE;
    for (int i = 0; i < total_pixels_resized; i++) {
        total_brightness += target_img->data[i];
    }
    const long average_brightness = total_brightness / total_pixels_resized;

    if (average_brightness > PIXEL_MAX / 2) {
        // Average brightness is high (light image), INVERT for black-on-white target!
        for (int i = 0; i < total_pixels_resized; i++) {
            target_img->data[i] = PIXEL_MAX - target_img->data[i];
        }
        printf("[INFO] Image is light-on-dark. Colors inverted for black-on-white target.\n");
    } else {
        printf("[INFO] Image is dark-on-light, no inversion.\n");
    }
    // --- END COLOR INVERSION LOGIC ---


    printf("[INFO] Successfully loaded and resized %s from %dx%d (Grayscale) to %dx%d.\n",
           filename, raw_img.width, raw_img.height, IMAGE_SIZE, IMAGE_SIZE);

    free(raw_img.data);
    return target_img;
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
                // Only LINE instructions have non-zero intensity
                if (intensity > 0) { 
                    draw_line(img, current_pos, instr->p, intensity);
                }
                current_pos = instr->p;
                break;
            case INSTR_TYPE_COUNT:
                break; 
        }
    }
    return img;
}

/**
 * @brief Calculates the Mean Squared Error (MSE) between two 128x128 images.
 */
static float calculate_mse_image(const Image* const img1, const Image* const img2) {
    double sum_squared_error = 0.0;
    const int total_pixels = IMAGE_SIZE * IMAGE_SIZE;
    int i;

    for (i = 0; i < total_pixels; i++) {
        const int diff = (int)img1->data[i] - (int)img2->data[i];
        sum_squared_error += (double)diff * diff;
    }
    return (float)(sum_squared_error / total_pixels);
}

/**
 * @brief Calculates the Mean Squared Error (MSE) between the rendered 16x16 grid and the A-Template.
 */
static float calculate_template_error(const Image* const rendered_img) {
    SmallImage small_img;
    downscale_image(rendered_img, &small_img);

    double sum_squared_error = 0.0;
    const int total_pixels = SMALL_GRID_SIZE * SMALL_GRID_SIZE;
    int i;

    for (i = 0; i < total_pixels; i++) {
        // Map grayscale rendered pixel (0=White, 255=Black) to a binary value (0 or 1)
        const int rendered_binary = small_img[i] > A_TEMPLATE_THRESHOLD ? 1 : 0;
        
        // Template value is 1 (dark/black) or 0 (light/white)
        const int template_value = A_TEMPLATE[i];

        // We want (rendered_binary - template_value) to be 0 for a match
        const int diff = rendered_binary - template_value;
        sum_squared_error += (double)diff * diff;
    }

    return (float)(sum_squared_error / total_pixels);
}

/**
 * @brief Calculates the combined fitness score for the Simulated Annealing.
 * $F = MSE_{Image} + (\text{Weight} \times MSE_{Template})$
 */
static float calculate_combined_fitness(const Image* const rendered_img, const Image* const target_img) {
    const float mse_image = calculate_mse_image(rendered_img, target_img);
    const float mse_template = calculate_template_error(rendered_img);

    // The fitness function prioritizes both matching the original image and looking like an 'A'
    return mse_image + (A_TEMPLATE_WEIGHT * mse_template);
}


// --- 5. Initial Drawing Heuristics (Component-based Vectorization) ---

#define MAX_COMPONENTS 10
#define MAX_PIXELS_PER_COMPONENT (SMALL_GRID_SIZE * SMALL_GRID_SIZE)

// Structure to hold a single connected component (a cluster of dark pixels)
typedef struct {
    Point pixels[MAX_PIXELS_PER_COMPONENT]; // Store coordinates (0-15)
    int count;
} Component;


/**
 * @brief Depth-First Search (DFS) to find all connected pixels in a component.
 * Uses 8-way adjacency.
 * @param small_img The 16x16 downscaled image.
 * @param visited A 16x16 array tracking visited pixels.
 * @param x Current x-coordinate (0-15).
 * @param y Current y-coordinate (0-15).
 * @param current_component Pointer to the component being built.
 */
static void dfs_find_component(const SmallImage* const small_img, int visited[SMALL_GRID_SIZE][SMALL_GRID_SIZE], int x, int y, Component* const current_component) {
    // Check bounds, check visited, check pixel darkness (value > TRACE_THRESHOLD)
    if (x < 0 || x >= SMALL_GRID_SIZE || y < 0 || y >= SMALL_GRID_SIZE || 
        visited[y][x] || (*small_img)[y * SMALL_GRID_SIZE + x] <= TRACE_THRESHOLD) {
        return;
    }

    visited[y][x] = 1;
    current_component->pixels[current_component->count++] = (Point){x, y};

    // Recursively check 8 neighbors
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx != 0 || dy != 0) { // Exclude self
                dfs_find_component(small_img, visited, x + dx, y + dy, current_component);
            }
        }
    }
}

/**
 * @brief Splits the 16x16 target image into disconnected dark pixel clusters.
 * @return The number of components found.
 */
static int find_connected_components(const SmallImage* const small_img, Component components[MAX_COMPONENTS]) {
    int visited[SMALL_GRID_SIZE][SMALL_GRID_SIZE] = {0};
    int component_count = 0;

    for (int y = 0; y < SMALL_GRID_SIZE; y++) {
        for (int x = 0; x < SMALL_GRID_SIZE; x++) {
            if (!visited[y][x] && (*small_img)[y * SMALL_GRID_SIZE + x] > TRACE_THRESHOLD) {
                if (component_count >= MAX_COMPONENTS) {
                    // Safety break
                    fprintf(stderr, "[WARNING] Max components reached (%d).\n", MAX_COMPONENTS);
                    return component_count;
                }
                
                // Initialize the new component
                components[component_count].count = 0;

                // Start DFS to find all connected pixels
                dfs_find_component(small_img, visited, x, y, &components[component_count]);
                component_count++;
            }
        }
    }
    return component_count;
}

/**
 * @brief Simplifies a cluster of pixels (a Component) into a single LINE instruction.
 * This is done by finding the two points that are furthest apart in the component.
 * @param component The cluster of pixels.
 * @param line_p1 Output for the start point (128x128 coords).
 * @param line_p2 Output for the end point (128x128 coords).
 */
static void simplify_component_to_lines(const Component* const component, Point* const line_p1, Point* const line_p2) {
    if (component->count == 0) return;
    if (component->count == 1) {
        // Handle single pixel component by making a tiny line (or just setting one point)
        // For simplicity in the initial drawing, we'll just set the line to be a single point.
        // This component will be filtered out by the main drawing function anyway.
        *line_p1 = component->pixels[0];
        *line_p2 = component->pixels[0];
        return;
    }

    long max_dist_sq = -1;
    int best_p1_idx = 0;
    int best_p2_idx = 0;

    // Find the pair of points (pixels) that are farthest from each other
    for (int i = 0; i < component->count; i++) {
        for (int j = i + 1; j < component->count; j++) {
            const Point pA = component->pixels[i];
            const Point pB = component->pixels[j];

            // Squared Euclidean distance
            const long dx = pA.x - pB.x;
            const long dy = pA.y - pB.y;
            const long dist_sq = dx * dx + dy * dy;

            if (dist_sq > max_dist_sq) {
                max_dist_sq = dist_sq;
                best_p1_idx = i;
                best_p2_idx = j;
            }
        }
    }
    
    // Convert 16x16 coordinates to 128x128 center points
    const int SCALE_FACTOR = IMAGE_SIZE / SMALL_GRID_SIZE; // 8
    const int offset = SCALE_FACTOR / 2; // 4

    Point p1_16 = component->pixels[best_p1_idx];
    Point p2_16 = component->pixels[best_p2_idx];

    // Map the 16x16 grid coordinate (0-15) to the center of the 8x8 block in the 128x128 grid
    line_p1->x = p1_16.x * SCALE_FACTOR + offset;
    line_p1->y = p1_16.y * SCALE_FACTOR + offset;

    line_p2->x = p2_16.x * SCALE_FACTOR + offset;
    line_p2->y = p2_16.y * SCALE_FACTOR + offset;
}

/**
 * @brief Generates the initial Drawing program by segmenting the target image
 * and approximating each segment with a single line.
 */
static Drawing generate_initial_drawing(const Image* const target_img) {
    Drawing d;
    memset(&d, 0, sizeof(Drawing));

    SmallImage small_img;
    downscale_image(target_img, &small_img);
    
    Component components[MAX_COMPONENTS];
    const int component_count = find_connected_components(&small_img, components);

    const Pixel trace_intensity = 220;
    int instructions_added = 0;

    for (int i = 0; i < component_count; i++) {
        if (instructions_added + 2 > MAX_INSTRUCTIONS) {
            fprintf(stderr, "[WARNING] Max instructions hit during component processing.\n");
            break;
        }

        Component* comp = &components[i];
        
        // Ignore very small components (e.g., noise)
        if (comp->count < 2) continue;

        Point p1_128, p2_128;
        simplify_component_to_lines(comp, &p1_128, &p2_128);

        // Add MOVE and LINE instructions
        d.instructions[d.count++] = (Instruction){INSTR_MOVE, p1_128, 0};
        d.instructions[d.count++] = (Instruction){INSTR_LINE, p2_128, trace_intensity};
        instructions_added += 2;
    }

    printf("[INFO] Component vectorization generated %d instructions from %d components.\n", d.count, component_count);
    return d;
}


// --- 6. Simulated Annealing Heuristics & Mutations ---

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
    
    // 0: Modify, 1: Add, 2: Remove
    const int mutation_type = rand() % 3;

    if (mutation_type == 0 && next.count > 0) { // Modify
        const int idx = rand() % next.count;
        Instruction* instr = &next.instructions[idx];
        const int param_to_change = rand() % 3; // 0: Type, 1: Point, 2: Intensity

        if (param_to_change == 0) {
            // Change instruction type (MOVE or LINE)
            instr->type = (InstructionType)(rand() % INSTR_TYPE_COUNT);
        } else if (param_to_change == 1) {
            // Change point location
            instr->p = random_point();
        } else if (instr->type == INSTR_LINE && param_to_change == 2) {
            // Change intensity (LINE instructions only)
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
        new_instr->intensity = (new_instr->type == INSTR_LINE) ? (Pixel)(rand() % 100 + 155) : 0;

    } else if (mutation_type == 2 && next.count > 1) { // Remove
        // Remove a random instruction, ensuring we keep at least one instruction
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
static float acceptance_probability(const float old_fitness, const float new_fitness, const float temp) {
    if (new_fitness < old_fitness) {
        return 1.0f;
    }
    // $e^{\Delta F / T}$
    return (float)exp((double)(old_fitness - new_fitness) / temp);
}

// --- 7. Unit Tests ---

/**
 * @brief Helper to check if a calculated line approximation matches expected 16x16 coordinates.
 * @param p1 Calculated start point (128x128).
 * @param p2 Calculated end point (128x128).
 * @param expected_p1_16 Expected 16x16 start point.
 * @param expected_p2_16 Expected 16x16 end point.
 * @return 1 if matched, 0 otherwise.
 */
static int check_line_approx(Point p1, Point p2, Point expected_p1_16, Point expected_p2_16) {
    const int SCALE_FACTOR = IMAGE_SIZE / SMALL_GRID_SIZE; // 8
    const int offset = SCALE_FACTOR / 2; // 4

    // Convert 128x128 back to 16x16 grid coordinate indices (0-15)
    Point p1_16_calc = {(p1.x - offset) / SCALE_FACTOR, (p1.y - offset) / SCALE_FACTOR};
    Point p2_16_calc = {(p2.x - offset) / SCALE_FACTOR, (p2.y - offset) / SCALE_FACTOR};

    // Check both (p1, p2) and (p2, p1) because line direction is arbitrary
    int match_1 = (p1_16_calc.x == expected_p1_16.x && p1_16_calc.y == expected_p1_16.y &&
                   p2_16_calc.x == expected_p2_16.x && p2_16_calc.y == expected_p2_16.y);
                   
    int match_2 = (p1_16_calc.x == expected_p2_16.x && p1_16_calc.y == expected_p2_16.y &&
                   p2_16_calc.x == expected_p1_16.x && p2_16_calc.y == expected_p1_16.y);
    
    return match_1 || match_2;
}

/**
 * @brief Unit test to verify component detection and line simplification for various line types.
 */
static void test_line_simplification_accuracy(void) {
    printf("\n--- Running Unit Test: Line Simplification Accuracy ---\n");
    SmallImage test_small_img;
    memset(test_small_img, 0, sizeof(test_small_img));
    
    // 1. Vertical Line (x=2, y=1..7) -> Component 1
    for (int y = 1; y < 8; y++) test_small_img[y * 16 + 2] = 100;
    
    // 2. Horizontal Line (y=10, x=9..14) -> Component 2
    for (int x = 9; x < 15; x++) test_small_img[10 * 16 + x] = 100;

    // 3. Diagonal Line (x=1, y=1 to x=5, y=5) -> Component 3
    for (int i = 0; i <= 4; i++) test_small_img[(1 + i) * 16 + (1 + i)] = 100;
    
    // 4. Single pixel (x=15, y=15) -> Component 4 (should be ignored by main function)
    test_small_img[15 * 16 + 15] = 100;

    Component components[MAX_COMPONENTS];
    const int component_count = find_connected_components(&test_small_img, components);

    if (component_count == 4) {
        printf("[SUCCESS] Component Detection Passed: Found 4 components.\n");
    } else {
        printf("[FAILURE] Component Detection Failed: Found %d components (Expected 4).\n", component_count);
    }
    
    Point p1_128, p2_128;

    // --- Test 1: Vertical Line (16x16: (2, 1) to (2, 7)) ---
    simplify_component_to_lines(&components[0], &p1_128, &p2_128);
    if (check_line_approx(p1_128, p2_128, (Point){2, 1}, (Point){2, 7})) {
        printf("[SUCCESS] Vertical Line Simplification: Correctly identified endpoints.\n");
    } else {
         printf("[FAILURE] Vertical Line Simplification: Expected (2,1)-(2,7). Got: (%d,%d)-(%d,%d) (16x16 indices).\n",
            (p1_128.x-4)/8, (p1_128.y-4)/8, (p2_128.x-4)/8, (p2_128.y-4)/8);
    }

    // --- Test 2: Horizontal Line (16x16: (9, 10) to (14, 10)) ---
    simplify_component_to_lines(&components[1], &p1_128, &p2_128);
    if (check_line_approx(p1_128, p2_128, (Point){9, 10}, (Point){14, 10})) {
        printf("[SUCCESS] Horizontal Line Simplification: Correctly identified endpoints.\n");
    } else {
         printf("[FAILURE] Horizontal Line Simplification: Expected (9,10)-(14,10). Got: (%d,%d)-(%d,%d) (16x16 indices).\n",
            (p1_128.x-4)/8, (p1_128.y-4)/8, (p2_128.x-4)/8, (p2_128.y-4)/8);
    }

    // --- Test 3: Diagonal Line (16x16: (1, 1) to (5, 5)) ---
    simplify_component_to_lines(&components[2], &p1_128, &p2_128);
    if (check_line_approx(p1_128, p2_128, (Point){1, 1}, (Point){5, 5})) {
        printf("[SUCCESS] Diagonal Line Simplification: Correctly identified endpoints.\n");
    } else {
         printf("[FAILURE] Diagonal Line Simplification: Expected (1,1)-(5,5). Got: (%d,%d)-(%d,%d) (16x16 indices).\n",
            (p1_128.x-4)/8, (p1_128.y-4)/8, (p2_128.x-4)/8, (p2_128.y-4)/8);
    }

    printf("--------------------------------------------------\n");
}


/**
 * @brief Unit test to verify that a simple 3-line 'A' drawing matches the hardcoded A_TEMPLATE.
 */
static void test_a_template_match(void) {
    printf("\n--- Running Unit Test: A-Template Match (Control) ---\n");
    Drawing test_drawing = {0};
    
    // Create a 3-line 'A' using coordinates that map to the center of the 16x16 grid blocks:
    const Pixel intensity = 255; // Solid black lines

    // 1. Left leg (Top-center to Bottom-left)
    test_drawing.instructions[test_drawing.count++] = (Instruction){INSTR_MOVE, {64, 16}, 0};
    test_drawing.instructions[test_drawing.count++] = (Instruction){INSTR_LINE, {16, 112}, intensity};
    
    // 2. Right leg (Top-center to Bottom-right)
    test_drawing.instructions[test_drawing.count++] = (Instruction){INSTR_MOVE, {64, 16}, 0};
    test_drawing.instructions[test_drawing.count++] = (Instruction){INSTR_LINE, {112, 112}, intensity};
    
    // 3. Crossbar (Middle-left to Middle-right)
    test_drawing.instructions[test_drawing.count++] = (Instruction){INSTR_MOVE, {32, 80}, 0};
    test_drawing.instructions[test_drawing.count++] = (Instruction){INSTR_LINE, {96, 80}, intensity};

    Image* rendered_a = render_drawing(&test_drawing);
    float template_error = calculate_template_error(rendered_a);
    
    const float max_acceptable_error = 0.08f; 

    if (template_error < max_acceptable_error) {
        printf("[SUCCESS] Template Match Test Passed! Error: %.4f (Expected < %.4f)\n", 
               template_error, max_acceptable_error);
    } else {
        printf("[FAILURE] Template Match Test Failed! Error: %.4f (Expected < %.4f)\n", 
               template_error, max_acceptable_error);
    }
    
    free(rendered_a);
    printf("--------------------------------------------------\n");
}


// --- 8. JPEG Output (Using libjpeg) ---

/**
 * @brief Saves a grayscale Image to a high-quality JPEG file using libjpeg.
 */
static void save_jpeg_grayscale(const Image* const img, const Drawing* const best_drawing, const float final_fitness) {
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE *outfile;
    JSAMPROW row_pointer[1];
    int row_stride; 

    const int quality = 90; // High quality setting

    memset(&cinfo, 0, sizeof(cinfo));
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    if ((outfile = fopen(JPEG_OUTPUT_FILENAME, "wb")) == NULL) {
        fprintf(stderr, "[ERROR] Can't open %s for writing\n", JPEG_OUTPUT_FILENAME);
        return;
    }
    jpeg_stdio_dest(&cinfo, outfile);

    cinfo.image_width = IMAGE_SIZE;
    cinfo.image_height = IMAGE_SIZE;
    cinfo.input_components = 1;
    cinfo.in_color_space = JCS_GRAYSCALE;
    row_stride = IMAGE_SIZE * cinfo.input_components;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);
    jpeg_start_compress(&cinfo, TRUE);

    while (cinfo.next_scanline < cinfo.image_height) {
        row_pointer[0] = (JSAMPROW)&img->data[cinfo.next_scanline * row_stride];
        (void)jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo);
    fclose(outfile);
    jpeg_destroy_compress(&cinfo);


    printf("\nSuccessfully saved best rendered image to %s (JPEG, Quality %d).\n", JPEG_OUTPUT_FILENAME, quality);
    printf("--- Final Results ---\n");
    printf("Final Combined Fitness Score: %.2f\n", final_fitness);
    printf("Final Instruction Count: %d\n", best_drawing->count);
    printf("Final Instructions:\n");

    const char* const type_names[] = {"MOVE", "LINE"};
    for (int i = 0; i < best_drawing->count; i++) {
        const Instruction* instr = &best_drawing->instructions[i];
        printf("  [%2d] %s(%d, %d", i, type_names[instr->type], instr->p.x, instr->p.y);
        if (instr->type == INSTR_LINE) {
            printf(", intensity: %d", instr->intensity);
        }
        printf(")\n");
    }
}


// --- 9. Main Program Execution ---

int main(void) {
    srand((unsigned int)time(NULL));

    // Run all unit tests
    test_a_template_match();
    test_line_simplification_accuracy();


    printf("--- Drawing Optimizer with 'A' Constraint ---\n");
    printf("--- Target image loading from: %s ---\n", JPEG_INPUT_FILENAME);
    printf("--- Constraint Weight (A-Template): %.1f ---\n", A_TEMPLATE_WEIGHT);

    Image* target_img = load_and_resize_target(JPEG_INPUT_FILENAME);
    
    if (target_img == NULL) {
        fprintf(stderr, "[FATAL] Failed to load target image. Exiting.\n");
        return EXIT_FAILURE;
    }

    // 1. Initial Heuristic Guess via Component Vectorization
    Drawing current_drawing = generate_initial_drawing(target_img);
    Image* current_img = render_drawing(&current_drawing);
    float current_fitness = calculate_combined_fitness(current_img, target_img);

    printf("Initial Component Vectorization Fitness: %.2f\n", current_fitness);
    printf("Initial Instruction Count: %d\n", current_drawing.count);
    printf("-----------------------------------------------------------\n");
    printf("Starting Simulated Annealing for %d seconds...\n", SA_RUNTIME_SECONDS);
    printf("-----------------------------------------------------------\n");

    // --- Simulated Annealing Loop ---
    const time_t start_time = time(NULL);
    time_t last_log_time = start_time;
    float elapsed_time;
    float best_fitness = current_fitness;
    Drawing best_drawing = current_drawing;
    int iteration = 0;

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

        // 1. Temperature update (Linear cooling)
        temp = initial_temp * (1.0f - (elapsed_time / SA_RUNTIME_SECONDS));
        if (temp < 0.001f) temp = 0.001f;

        // 2. Generate a neighbor state
        next_drawing = mutate_drawing(&current_drawing);
        next_img = render_drawing(&next_drawing);
        float new_fitness = calculate_combined_fitness(next_img, target_img);

        // 3. Acceptance criterion
        const float prob = (float)rand() / (float)RAND_MAX;
        if (acceptance_probability(current_fitness, new_fitness, temp) > prob) {
            // Accept the new state
            free(current_img);
            current_drawing = next_drawing;
            current_fitness = new_fitness;
            current_img = next_img;

            if (current_fitness < best_fitness) {
                best_fitness = current_fitness;
                best_drawing = current_drawing;
            }
        } else {
            // Reject the new state
            free(next_img);
        }

        iteration++;

        // 4. Logging output
        if (difftime(current_time, last_log_time) >= SA_LOG_INTERVAL_SECONDS) {
            printf("| Time: %3.0fs / %ds | Iteration: %7d | T: %7.2f | Count: %4d | Current Fitness: %7.2f | Best Fitness: %7.2f |\n",
                   elapsed_time, SA_RUNTIME_SECONDS, iteration, temp, current_drawing.count, current_fitness, best_fitness);
            last_log_time = current_time;
        }
    }

    // --- Final Results and Cleanup ---
    printf("-----------------------------------------------------------\n");
    printf("Optimization finished after %d iterations and %.0f seconds.\n", iteration, elapsed_time);

    Image* best_img = render_drawing(&best_drawing);
    save_jpeg_grayscale(best_img, &best_drawing, best_fitness);

    free(target_img);
    free(current_img);
    free(best_img);

    return EXIT_SUCCESS;
}