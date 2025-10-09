#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h> 

// --- STB Image Write Configuration ---
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// --- STB Image Read Configuration (Placeholder) ---
// We can't actually include this and load an image, but define the implementation guard
// to show where it would be used if possible.
// #define STB_IMAGE_IMPLEMENTATION 
// #include "stb_image.h" // Assuming stbi_load function is here

// Define M_PI explicitly
#define M_PI 3.14159265358979323846

// --- Global Configuration ---
#define GRID_SIZE 32        // 32x32 resolution for recognition
#define NUM_DEFORMATIONS 2  
#define NUM_VECTORS 16      
#define NUM_BINS 32         
#define NUM_FEATURES (NUM_VECTORS * NUM_BINS) 
#define PIXEL_LOSS_WEIGHT 2.5 
#define NUM_POINTS 200
#define ITERATIONS 500      
#define GRADIENT_EPSILON 0.01 
#define NUM_IDEAL_CHARS 36  
#define TESTS_PER_CHAR 8    
#define NUM_TESTS (NUM_IDEAL_CHARS * TESTS_PER_CHAR) // 36 * 8 = 288 total tests
#define NUM_CONTROL_POINTS 9 
#define MAX_PIXEL_ERROR (GRID_SIZE * GRID_SIZE) 
#define TIME_LIMIT_SECONDS 240.0 // 4 minutes limit

// Loss history configuration
#define LOSS_HISTORY_STEP 5 
#define LOSS_HISTORY_SIZE (ITERATIONS / LOSS_HISTORY_STEP + 1) 

// Segmentation limits
#define MAX_SEGMENTS 100 // Maximum number of letters we expect to find

// --- Data Structures ---

typedef struct { double x; double y; } Point;
typedef struct { const Point control_points[NUM_CONTROL_POINTS]; } Ideal_Curve_Params; 
typedef struct { double alpha[NUM_DEFORMATIONS]; } Deformation_Coefficients;
typedef double Generated_Image[GRID_SIZE][GRID_SIZE]; 
typedef double Feature_Vector[NUM_FEATURES]; 

typedef struct {
    int start; 
    int end; 
} Boundary;

typedef struct {
    int x_start, x_end; 
    int y_start, y_end; 
    Generated_Image resized_img; // The 32x32 image segment for recognition
    int best_match_index; 
    double final_loss;
    double estimated_alpha[NUM_DEFORMATIONS];
} SegmentResult;

typedef struct { 
    double estimated_alpha[NUM_DEFORMATIONS]; 
    double final_loss; 
    double loss_history[LOSS_HISTORY_SIZE]; 
} EstimationResult;

typedef struct {
    int id; int true_char_index; int best_match_index;
    double true_alpha[NUM_DEFORMATIONS];
    Generated_Image true_image; Generated_Image observed_image;
    EstimationResult classification_results[NUM_IDEAL_CHARS]; 
    Generated_Image best_estimated_image; Generated_Image best_diff_image;
} TestResult;

// --- Memory Tracking Globals (Unchanged from previous context) ---
size_t total_allocated_bytes = 0;
size_t total_freed_bytes = 0;
size_t memory_history[NUM_TESTS + 1]; 
int memory_history_index = 0;

TestResult *all_results = NULL; 
int tests_completed_before_limit = 0; 


// --- Fixed Ideal Curves (A-Z, 0-9) ---
// (The full IDEAL_TEMPLATES array with the fixed 'O' from the last context is omitted for brevity)
const char *CHAR_NAMES[NUM_IDEAL_CHARS] = {
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", 
    "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", 
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
};
const Ideal_Curve_Params IDEAL_TEMPLATES[NUM_IDEAL_CHARS] = {
    // A
    [0] = {.control_points = {{.x = 0.5, .y = 0.1}, {.x = 0.3, .y = 0.3}, {.x = 0.2, .y = 0.5}, {.x = 0.3, .y = 0.6}, 
        {.x = 0.7, .y = 0.6}, {.x = 0.8, .y = 0.5}, {.x = 0.7, .y = 0.3}, {.x = 0.5, .y = 0.1}, 
        {.x = 0.7, .y = 0.9} 
    }},
    // B... Z, 0...9 (All other definitions are assumed to be present and correct)
    // ...
    // O (FIXED: Duplicated closing point to prevent superfluous internal lines)
    [14] = {.control_points = {{.x = 0.5, .y = 0.1}, {.x = 0.85, .y = 0.3}, {.x = 0.85, .y = 0.7}, 
        {.x = 0.5, .y = 0.9}, {.x = 0.15, .y = 0.7}, {.x = 0.15, .y = 0.3}, 
        {.x = 0.5, .y = 0.1}, // P6: Closes the loop
        {.x = 0.5, .y = 0.1}, // P7: Zero-length segment
        {.x = 0.5, .y = 0.1}  // P8: Zero-length segment
    }},
    // ... all 36 characters are here ...
    [35] = {.control_points = {{.x = 0.5, .y = 0.1}, {.x = 0.8, .y = 0.2}, {.x = 0.5, .y = 0.4}, {.x = 0.2, .y = 0.2}, 
        {.x = 0.5, .y = 0.1}, {.x = 0.8, .y = 0.4}, {.x = 0.8, .y = 0.9}, {.x = 0.5, .y = 0.7}, 
        {.x = 0.8, .y = 0.9} 
    }}
};


// --- Utility Functions (Memory, Curve Drawing, Loss, Optimization - Omitted for brevity, assumed correct) ---

// (safe_malloc, safe_free, record_memory_usage, apply_deformation, get_deformed_point, 
// draw_curve, extract_geometric_features, calculate_feature_loss_L2, calculate_pixel_loss_L2, 
// calculate_combined_loss, calculate_gradient, generate_target_image, calculate_pixel_error_sum, 
// run_optimization, calculate_difference_image - All functions from previous context are assumed present)


// --- Image Loading and Preprocessing ---

/**
 * @brief Placeholder for loading an image from file.
 * NOTE: This function cannot be truly implemented/tested in this environment.
 * @param filename The name of the file to load (e.g., "test1.jpg").
 * @param data_out Pointer to an array to store pixel data (dynamically allocated).
 * @param width_out Pointer to store image width.
 * @param height_out Pointer to store image height.
 * @return 1 on success, 0 on failure.
 */
int load_image_stb(const char *filename, double **data_out, int *width_out, int *height_out) {
    printf("--- WARNING: Skipping actual image load (stbi_load not available) ---\n");
    printf("Simulating image load with a small, all-white image.\n");

    // Simulation: A 100x100 grayscale image
    *width_out = 100;
    *height_out = 100;
    size_t size = (size_t)(*width_out) * (*height_out);
    *data_out = (double *)safe_malloc(sizeof(double) * size);
    
    if (*data_out == NULL) return 0;

    // Simulate some content for a simple test (e.g., a "T" shape)
    for (int i = 0; i < *height_out; i++) {
        for (int j = 0; j < *width_out; j++) {
            // Background is black (0.0)
            (*data_out)[i * (*width_out) + j] = 0.0; 
        }
    }
    
    // Simulate a letter 'T' around the center (10-80 range)
    for (int j = 10; j <= 80; j++) { // Horizontal bar
        (*data_out)[15 * (*width_out) + j] = 1.0;
        (*data_out)[16 * (*width_out) + j] = 1.0;
    }
    for (int i = 15; i <= 80; i++) { // Vertical bar
        (*data_out)[i * (*width_out) + 45] = 1.0;
        (*data_out)[i * (*width_out) + 46] = 1.0;
    }


    return 1;
}

/**
 * @brief Converts a segment of a larger image into a 32x32 Generated_Image by resampling.
 */
void resize_segment(const double *full_data, int full_width, int full_height, 
                    int x_start, int x_end, int y_start, int y_end, 
                    Generated_Image segment_out) {
    
    int segment_w = x_end - x_start;
    int segment_h = y_end - y_start;

    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            
            // Map the 32x32 cell (i, j) to a block in the original image segment
            // This is a simple nearest-neighbor or box-sampling approach
            int src_y = y_start + (int)round((double)i / GRID_SIZE * segment_h);
            int src_x = x_start + (int)round((double)j / GRID_SIZE * segment_w);

            // Ensure coordinates are within the full image bounds
            src_y = fmax(0, fmin(full_height - 1, src_y));
            src_x = fmax(0, fmin(full_width - 1, src_x));

            segment_out[i][j] = full_data[src_y * full_width + src_x];
        }
    }
}


// --- Histogram Segmentation Logic ---

/**
 * @brief Projects the image intensity onto an axis.
 * @param full_data The full image data.
 * @param width The image width.
 * @param height The image height.
 * @param orientation 0 for horizontal (project to Y-axis), 1 for vertical (project to X-axis).
 * @param hist_out Array to store the histogram.
 */
void project_histogram(const double *full_data, int width, int height, int orientation, double *hist_out) {
    int hist_size = (orientation == 0) ? height : width; 
    int other_dim = (orientation == 0) ? width : height; 

    for (int i = 0; i < hist_size; i++) {
        hist_out[i] = 0.0;
    }

    if (orientation == 0) { // Horizontal projection (sum rows -> Y-axis)
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                hist_out[i] += full_data[i * width + j];
            }
        }
    } else { // Vertical projection (sum columns -> X-axis)
        for (int j = 0; j < width; j++) {
            for (int i = 0; i < height; i++) {
                hist_out[j] += full_data[i * width + j];
            }
        }
    }
}

/**
 * @brief Finds intervals in the histogram with near-zero values.
 * @param hist The histogram array.
 * @param size The size of the histogram.
 * @param min_zero_length The minimum number of consecutive zeros required for a boundary.
 * @param threshold The maximum value to be considered "zero."
 * @param boundaries_out Array to store the found Boundary structures.
 * @return The number of boundaries found.
 */
int find_zero_intervals(const double *hist, int size, int min_zero_length, double threshold, Boundary *boundaries_out) {
    int count = 0;
    int zero_start = -1;
    
    // Find gaps between non-zero areas
    for (int i = 0; i < size; i++) {
        if (hist[i] < threshold) {
            if (zero_start == -1) {
                zero_start = i;
            }
        } else {
            if (zero_start != -1) {
                int zero_length = i - zero_start;
                if (zero_length >= min_zero_length) {
                    // Boundary is found at the *center* of the zero interval
                    // For line/letter segmentation, we want the boundary to be the edge of the content.
                    // The content starts *after* the zero-end and ends *before* the zero-start.
                    
                    // The boundary of content starts at the end of the zero-gap
                    boundaries_out[count].start = zero_start + zero_length / 2;
                    boundaries_out[count].end = zero_start + zero_length / 2; // Single point in a zero-gap
                    
                    if (count < MAX_SEGMENTS) count++;
                }
                zero_start = -1;
            }
        }
    }

    // Process the last interval if it ends with zeros
    if (zero_start != -1 && (size - zero_start) >= min_zero_length) {
        if (count < MAX_SEGMENTS) {
            boundaries_out[count].start = zero_start + (size - zero_start) / 2;
            boundaries_out[count].end = zero_start + (size - zero_start) / 2;
            count++;
        }
    }

    // Now, convert the gap-centers into content-edges
    Boundary content_boundaries[MAX_SEGMENTS + 1];
    int content_count = 0;
    
    // Initial content boundary (start of image)
    content_boundaries[content_count++].start = 0;

    // Boundaries are between the zero gaps
    for (int i = 0; i < count; i++) {
        content_boundaries[content_count++].start = boundaries_out[i].start;
    }

    // Final content boundary (end of image)
    content_boundaries[content_count++].start = size - 1;

    // Now, pair them up
    int final_count = 0;
    for (int i = 0; i < content_count - 1; i++) {
        // Only keep segments that have a minimum size (to filter noise)
        int current_start = content_boundaries[i].start;
        int current_end = content_boundaries[i+1].start;

        // Skip intervals that are too small (likely just noise)
        if (current_end - current_start < 5) continue;
        
        // Refine boundary to the *first* non-zero/last non-zero pixel in the region
        int refined_start = current_start;
        while(refined_start < current_end && hist[refined_start] < threshold) {
            refined_start++;
        }
        int refined_end = current_end;
        while(refined_end > refined_start && hist[refined_end] < threshold) {
            refined_end--;
        }
        
        if (refined_end - refined_start > 5) {
            boundaries_out[final_count].start = refined_start;
            boundaries_out[final_count].end = refined_end;
            if (final_count < MAX_SEGMENTS) final_count++;
        }
    }

    return final_count;
}

/**
 * @brief Performs naive histogram-based segmentation (Line -> Letter).
 * @param full_data The full image data.
 * @param full_width The image width.
 * @param full_height The image height.
 * @param segments_out Array to store the SegmentResult structures.
 * @return The total number of letter segments found.
 */
int segment_image_naive(const double *full_data, int full_width, int full_height, SegmentResult *segments_out) {
    int total_segments = 0;
    
    // 1. Vertical Histogram Projection (Find Lines - Y-axis)
    double *h_hist = (double *)safe_malloc(sizeof(double) * full_height);
    Boundary line_boundaries[MAX_SEGMENTS];
    project_histogram(full_data, full_width, full_height, 0, h_hist); // 0 for horizontal projection (Y-axis)
    
    // Line segmentation parameters: Min zero length (gap size), Max zero value (threshold)
    int num_lines = find_zero_intervals(h_hist, full_height, full_height / 15, 0.01, line_boundaries);
    safe_free(h_hist, sizeof(double) * full_height);
    
    printf("Segmentation: Found %d line(s).\n", num_lines);

    // 2. Horizontal Histogram Projection (Find Letters - X-axis)
    for (int l = 0; l < num_lines; l++) {
        int line_y_start = line_boundaries[l].start;
        int line_y_end = line_boundaries[l].end;
        int line_height = line_y_end - line_y_start;
        
        // Calculate vertical histogram for the current line segment
        double *v_hist = (double *)safe_malloc(sizeof(double) * full_width);
        
        // Manual vertical projection for a sub-region
        for (int j = 0; j < full_width; j++) {
            v_hist[j] = 0.0;
            for (int i = line_y_start; i <= line_y_end; i++) {
                v_hist[j] += full_data[i * full_width + j];
            }
        }
        
        Boundary letter_boundaries[MAX_SEGMENTS];
        // Letter segmentation parameters: Min zero length (gap size), Max zero value (threshold)
        int num_letters = find_zero_intervals(v_hist, full_width, full_width / 20, 0.01, letter_boundaries);
        safe_free(v_hist, sizeof(double) * full_width);
        
        printf("  Line %d: Found %d letter(s).\n", l + 1, num_letters);

        // 3. Process each letter segment
        for (int c = 0; c < num_letters; c++) {
            if (total_segments >= MAX_SEGMENTS) {
                printf("  WARNING: Reached maximum segment limit (%d).\n", MAX_SEGMENTS);
                return total_segments;
            }
            
            SegmentResult *seg = &segments_out[total_segments];
            seg->x_start = letter_boundaries[c].start;
            seg->x_end = letter_boundaries[c].end;
            seg->y_start = line_y_start;
            seg->y_end = line_y_end;
            
            // Resize and store the segment as a 32x32 image
            resize_segment(full_data, full_width, full_height, 
                           seg->x_start, seg->x_end, seg->y_start, seg->y_end, 
                           seg->resized_img);
                           
            total_segments++;
        }
    }
    
    return total_segments;
}


// --- Recognition Function ---

void recognize_segment(SegmentResult *segment) {
    Feature_Vector observed_features;
    extract_geometric_features(segment->resized_img, observed_features);
    
    double min_feature_loss = HUGE_VAL;
    int best_match_index = -1;

    // Optimization is only run against a single "observed" image (the segment)
    // We only need the EstimatedResult struct for loss and alpha.
    EstimationResult current_result; 
    
    for (int i = 0; i < NUM_IDEAL_CHARS; i++) {
        // We reuse run_optimization but it needs a full set of EstimationResult. 
        // We'll run a local version that only returns the best fit for this char.
        run_optimization(segment->resized_img, observed_features, i, &current_result); 

        if (current_result.final_loss < min_feature_loss) {
            min_feature_loss = current_result.final_loss;
            best_match_index = i;
            memcpy(segment->estimated_alpha, current_result.estimated_alpha, sizeof(double) * NUM_DEFORMATIONS);
        }
    }
    
    segment->best_match_index = best_match_index;
    segment->final_loss = min_feature_loss;
}


// --- PNG Rendering for Segmentation Output ---

#define SEG_ROW_HEIGHT (IMG_SIZE + SET_SPACING) 
#define SEG_PNG_WIDTH (IMG_SIZE * 2 + IMG_SPACING * 3 + GRAPH_WIDTH + SET_SPACING * 2) 

void draw_segment_info_box(unsigned char *buffer, int buf_width, int buf_height, int x, int y, int width, int height, const SegmentResult *seg) {
    char info[100];
    const char* char_name = (seg->best_match_index != -1) ? CHAR_NAMES[seg->best_match_index] : "N/A";
    
    // Placeholder text in the box
    sprintf(info, "Match: %s | Loss: %.2f | $a_1$:%.2f", char_name, seg->final_loss, seg->estimated_alpha[0]);
    draw_text_placeholder_box(buffer, buf_width, buf_height, x, y, width, height, 200, 200, 255);
}

void render_segment_to_png(unsigned char *buffer, int buf_width, int buf_height, const SegmentResult *seg, int seg_index, int x_set, int y_set) {
    int current_x = x_set;
    
    // 1. Resized Segment Image (Observed)
    render_single_image_to_png(buffer, buf_width, buf_height, seg->resized_img, current_x, y_set + TEXT_HEIGHT, 0); 
    current_x += IMG_SIZE + IMG_SPACING;
    
    // 2. Best Estimated Image
    Generated_Image estimated_img;
    if (seg->best_match_index != -1) {
        draw_curve(seg->estimated_alpha, estimated_img, &IDEAL_TEMPLATES[seg->best_match_index]);
    } else {
        memset(estimated_img, 0, sizeof(Generated_Image));
    }
    render_single_image_to_png(buffer, buf_width, buf_height, estimated_img, current_x, y_set + TEXT_HEIGHT, 0);
    current_x += IMG_SIZE + IMG_SPACING;

    // 3. Info Box Placeholder
    draw_segment_info_box(buffer, buf_width, buf_height, x_set, y_set + 2, IMG_SIZE * 2 + IMG_SPACING, TEXT_HEIGHT - 4, seg);
}

void generate_segment_png(const SegmentResult *segments, int num_segments, const double *full_data, int full_width, int full_height) {
    if (num_segments == 0) {
        printf("\nWARNING: No segments found. Skipping segment PNG generation.\n");
        return;
    }
    
    const int num_cols = 4; // Full Image + 2 Segment Images + Info/Loss
    const int MAX_SEGMENTS_PER_ROW = 5;
    const int SEGMENTS_PER_ROW = fmin(MAX_SEGMENTS_PER_ROW, num_segments);
    
    // Calculate required height: Full Image + All Segments
    int full_img_row_height = IMG_SIZE + TEXT_HEIGHT + SET_SPACING;
    int seg_row_height = IMG_SIZE + TEXT_HEIGHT + SET_SPACING;
    int num_seg_rows = (num_segments + SEGMENTS_PER_ROW - 1) / SEGMENTS_PER_ROW; 
    
    int png_height = full_img_row_height + num_seg_rows * seg_row_height + SET_SPACING;
    int png_width = SEG_PNG_WIDTH * SEGMENTS_PER_ROW + SET_SPACING;
    
    // Allocate buffer
    long buffer_size = (long)png_width * png_height * NUM_CHANNELS;
    unsigned char *buffer = (unsigned char *)safe_malloc(buffer_size);
    if (buffer == NULL) {
        printf("\nERROR: Failed to allocate buffer for segment PNG.\n");
        return;
    }
    memset(buffer, 0, buffer_size);

    int x_set = SET_SPACING;
    int y_set = SET_SPACING;

    // 1. Draw Full Image (as a segment for visualization)
    draw_text_placeholder_box(buffer, png_width, png_height, x_set, y_set + 2, full_width * PIXEL_SIZE, TEXT_HEIGHT - 4, 150, 150, 255);
    
    // We need to render the full image into the 3-channel buffer, scaled by PIXEL_SIZE.
    unsigned char r, g, b;
    for (int i = 0; i < full_height; i++) {
        for (int j = 0; j < full_width; j++) {
            get_pixel_color(full_data[i * full_width + j], 0, &r, &g, &b);
            for (int py = 0; py < PIXEL_SIZE; py++) {
                for (int px = 0; px < PIXEL_SIZE; px++) {
                    set_pixel(buffer, 
                              x_set + j * PIXEL_SIZE + px, 
                              y_set + TEXT_HEIGHT + i * PIXEL_SIZE + py, 
                              png_width, png_height, r, g, b);
                }
            }
        }
    }
    y_set += full_img_row_height;

    // 2. Draw Segment Boundaries on the Full Image
    for (int k = 0; k < num_segments; k++) {
        const SegmentResult *seg = &segments[k];
        // Draw segment box (blue outline)
        for (int y = seg->y_start * PIXEL_SIZE; y < (seg->y_end) * PIXEL_SIZE; y++) {
            set_pixel(buffer, x_set + seg->x_start * PIXEL_SIZE, y_set + y, png_width, png_height, 0, 0, 255);
            set_pixel(buffer, x_set + seg->x_end * PIXEL_SIZE, y_set + y, png_width, png_height, 0, 0, 255);
        }
        for (int x = seg->x_start * PIXEL_SIZE; x < (seg->x_end) * PIXEL_SIZE; x++) {
            set_pixel(buffer, x_set + x, y_set + seg->y_start * PIXEL_SIZE, png_width, png_height, 0, 0, 255);
            set_pixel(buffer, x_set + x, y_set + seg->y_end * PIXEL_SIZE, png_width, png_height, 0, 0, 255);
        }
    }
    y_set += SET_SPACING; // Move below the full image drawing

    // 3. Draw Letter Segments
    for (int k = 0; k < num_segments; k++) {
        int row_index = k / SEGMENTS_PER_ROW;
        int col_index = k % SEGMENTS_PER_ROW;
        
        int seg_x = SET_SPACING + col_index * SEG_PNG_WIDTH;
        int seg_y = y_set + row_index * seg_row_height;
        
        render_segment_to_png(buffer, png_width, png_height, &segments[k], k, seg_x, seg_y);
    }

    int success = stbi_write_png("segmentation_output.png", png_width, png_height, NUM_CHANNELS, buffer, png_width * NUM_CHANNELS);

    safe_free(buffer, buffer_size);

    if (success) {
        printf("\nSegmentation Output Complete: segmentation_output.png created.\n");
        printf("Size: %d x %d pixels. Total Segments: %d.\n", png_width, png_height, num_segments);
    } else {
        printf("\nERROR: Failed to write segmentation_output.png.\n");
    }
}


// --- Main Execution ---

int main(void) {
    srand(42); 

    // --- Image Segmentation/Recognition Test ---
    const char *input_filename = "test1.jpg";
    double *full_image_data = NULL;
    int full_width = 0;
    int full_height = 0;

    // Load Image (Uses Placeholder)
    if (!load_image_stb(input_filename, &full_image_data, &full_width, &full_height) || full_image_data == NULL) {
        fprintf(stderr, "Fatal Error: Failed to load image or memory allocation failed.\n");
        return 1;
    }
    
    // Allocate space for segments
    SegmentResult *segments = (SegmentResult *)safe_malloc(sizeof(SegmentResult) * MAX_SEGMENTS);
    if (segments == NULL) {
        fprintf(stderr, "Fatal Error: Failed to allocate memory for segments.\n");
        safe_free(full_image_data, sizeof(double) * full_width * full_height);
        return 1;
    }
    
    // 1. Segment Image
    int num_segments = segment_image_naive(full_image_data, full_width, full_height, segments);

    // 2. Recognize Each Segment
    printf("Starting recognition for %d segments...\n", num_segments);
    for (int i = 0; i < num_segments; i++) {
        recognize_segment(&segments[i]);
        const char* char_name = (segments[i].best_match_index != -1) ? CHAR_NAMES[segments[i].best_match_index] : "N/A";
        printf("  Segment %d: Match='%s' (Loss: %.4f, Bounds: X:[%d,%d] Y:[%d,%d])\n", 
               i + 1, char_name, segments[i].final_loss, 
               segments[i].x_start, segments[i].x_end, segments[i].y_start, segments[i].y_end);
    }
    
    // 3. Generate PNG Output
    generate_segment_png(segments, num_segments, full_image_data, full_width, full_height);

    // Free resources
    safe_free(segments, sizeof(SegmentResult) * MAX_SEGMENTS);
    safe_free(full_image_data, sizeof(double) * full_width * full_height);

    // --- End of Segmentation Test ---
    printf("\n--- Segmentation and Recognition Test Complete ---\n");
    
    return 0;
}
