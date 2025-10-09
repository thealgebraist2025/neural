#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// --- STB Image Write Configuration ---
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Define M_PI explicitly
#define M_PI 3.14159265358979323846

// --- Global Configuration ---
#define GRID_SIZE 16
#define NUM_DEFORMATIONS 2  
#define NUM_VECTORS 16      
#define NUM_BINS 32         
#define NUM_FEATURES (NUM_VECTORS * NUM_BINS) 
#define PIXEL_LOSS_WEIGHT 5.0 
#define NUM_POINTS 200
#define ITERATIONS 1000     
#define GRADIENT_EPSILON 0.01 
#define NUM_IDEAL_CHARS 36  
#define NUM_TESTS 36        
#define NUM_CONTROL_POINTS 9 
#define MAX_PIXEL_ERROR (GRID_SIZE * GRID_SIZE) 

// --- Data Structures (Same as before) ---
typedef struct { double x; double y; } Point;
typedef struct { const Point control_points[NUM_CONTROL_POINTS]; } Ideal_Curve_Params; 
typedef struct { double alpha[NUM_DEFORMATIONS]; } Deformation_Coefficients;
typedef double Generated_Image[GRID_SIZE][GRID_SIZE]; 
typedef double Feature_Vector[NUM_FEATURES]; 
typedef struct { double estimated_alpha[NUM_DEFORMATIONS]; double final_loss; } EstimationResult;
typedef struct {
    int id; int true_char_index; int best_match_index;
    double true_alpha[NUM_DEFORMATIONS];
    Generated_Image true_image; Generated_Image observed_image;
    EstimationResult classification_results[NUM_IDEAL_CHARS]; 
    Generated_Image best_estimated_image; Generated_Image best_diff_image;
} TestResult;

TestResult all_results[NUM_TESTS];

// --- Fixed Ideal Curves (A-Z, 0-9) ---
// (Templates and CHAR_NAMES omitted for brevity, assume they are present)
const char *CHAR_NAMES[NUM_IDEAL_CHARS] = {
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", 
    "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", 
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
};
const Ideal_Curve_Params IDEAL_TEMPLATES[NUM_IDEAL_CHARS] = {
    // ... all 36 templates defined here ...
    [0] = {.control_points = {{.x = 0.5, .y = 0.1}, {.x = 0.3, .y = 0.3}, {.x = 0.2, .y = 0.5}, {.x = 0.3, .y = 0.6}, 
        {.x = 0.7, .y = 0.6}, {.x = 0.8, .y = 0.5}, {.x = 0.7, .y = 0.3}, {.x = 0.5, .y = 0.1}, 
        {.x = 0.7, .y = 0.9} 
    }},
    [1] = {.control_points = {{.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.9}, {.x = 0.2, .y = 0.1}, 
        {.x = 0.8, .y = 0.1}, {.x = 0.8, .y = 0.3}, {.x = 0.2, .y = 0.5}, 
        {.x = 0.8, .y = 0.5}, {.x = 0.8, .y = 0.7}, {.x = 0.2, .y = 0.9} 
    }},
    [30] = {.control_points = {{.x = 0.2, .y = 0.1}, {.x = 0.3, .y = 0.2}, {.x = 0.4, .y = 0.3}, {.x = 0.5, .y = 0.4}, 
        {.x = 0.2, .y = 0.55}, {.x = 0.8, .y = 0.55}, {.x = 0.8, .y = 0.7}, {.x = 0.8, .y = 0.85}, 
        {.x = 0.8, .y = 0.9} 
    }},
    // ... all other templates ...
};

// --- Function Prototypes (Assume all mathematical and optimization functions 
// like apply_deformation, get_deformed_point, draw_curve, extract_geometric_features, 
// calculate_loss, calculate_gradient, run_optimization, run_classification_test are defined)
void apply_deformation(Point *point, const double alpha[NUM_DEFORMATIONS]);
Point get_deformed_point(const double t, const Ideal_Curve_Params *const params, const double alpha[NUM_DEFORMATIONS]);
void draw_curve(const double alpha[NUM_DEFORMATIONS], Generated_Image img, const Ideal_Curve_Params *const ideal_params);
void extract_geometric_features(const Generated_Image img, Feature_Vector features_out);
double calculate_feature_loss_L2(const Feature_Vector generated, const Feature_Vector observed);
double calculate_pixel_loss_L2(const Generated_Image generated, const Generated_Image observed);
double calculate_combined_loss(const Generated_Image generated_img, const Feature_Vector generated_features, const Generated_Image observed_img, const Feature_Vector observed_features);
void calculate_gradient(const Generated_Image observed_img, const Feature_Vector observed_features, const Deformation_Coefficients *const alpha, const double loss_base, double grad_out[NUM_DEFORMATIONS], const Ideal_Curve_Params *const ideal_params);
void generate_target_image(Generated_Image image_out, const double true_alpha[NUM_DEFORMATIONS], const Ideal_Curve_Params *const ideal_params, int add_noise);
double calculate_pixel_error_sum(const Generated_Image obs, const Generated_Image est);
void run_optimization(const Generated_Image observed_image, const Feature_Vector observed_features, int ideal_char_index, EstimationResult *result, int print_trace);
void calculate_difference_image(const Generated_Image obs, const Generated_Image est, Generated_Image diff);
void run_classification_test(int test_id, int true_char_index, const double true_alpha[NUM_DEFORMATIONS], TestResult *result);
void summarize_results_console();


// --- PNG Rendering Constants ---
#define PIXEL_SIZE 5    
#define IMG_SIZE (GRID_SIZE * PIXEL_SIZE) // 80
#define IMG_SPACING 5   
#define TEXT_HEIGHT 15  
#define SET_SPACING 40  
#define SET_WIDTH (4 * IMG_SIZE + 3 * IMG_SPACING + SET_SPACING) 
#define PNG_RENDER_LIMIT 10 // Number of test sets per row
#define PNG_WIDTH (PNG_RENDER_LIMIT * SET_WIDTH + SET_SPACING) 
#define PNG_HEIGHT (IMG_SIZE + 2 * TEXT_HEIGHT + SET_SPACING) // Image height + Title + Label + Spacing
#define NUM_CHANNELS 3 // RGB (24-bit color)

// --- PNG Rendering Functions ---

/**
 * @brief Sets a pixel color in the 24-bit RGB buffer.
 */
void set_pixel(unsigned char *buffer, int x, int y, int width, unsigned char r, unsigned char g, unsigned char b) {
    if (x >= 0 && x < width && y >= 0 && y < PNG_HEIGHT) {
        long index = (long)y * width * NUM_CHANNELS + x * NUM_CHANNELS;
        buffer[index] = r;
        buffer[index + 1] = g;
        buffer[index + 2] = b;
    }
}

/**
 * @brief Determines the 8-bit RGB color for a given normalized intensity and image type.
 */
void get_pixel_color(double intensity, int is_error_map, unsigned char *r, unsigned char *g, unsigned char *b) {
    double clamped_intensity = fmax(0.0, fmin(1.0, intensity));

    if (is_error_map) {
        // Error map: Red/Orange for high error, Black for low/zero
        if (clamped_intensity > 0.3) {
            *r = 255; *g = 50; *b = 50; // High Error: Red
        } else if (clamped_intensity > 0.1) {
            *r = 255; *g = 150; *b = 0; // Medium Error: Orange
        } else {
            *r = 0; *g = 0; *b = 0; // Low Error: Black
        }
    } else {
        // Regular image: White/Yellow on Black background
        if (clamped_intensity > 0.6) {
            *r = 255; *g = 255; *b = 100; // Bright Yellowish-White
        } else if (clamped_intensity > 0.3) {
            *r = 100; *g = 100; *b = 100; // Gray
        } else {
            *r = 0; *g = 0; *b = 0; // Black
        }
    }
}

/**
 * @brief Renders a single 16x16 image onto the PNG buffer.
 */
void render_single_image_to_png(unsigned char *buffer, int buf_width, const Generated_Image img, int x_offset, int y_offset, int is_error_map) {
    unsigned char r, g, b;
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            get_pixel_color(img[i][j], is_error_map, &r, &g, &b);
            
            // Draw the 5x5 pixel block
            for (int py = 0; py < PIXEL_SIZE; py++) {
                for (int px = 0; px < PIXEL_SIZE; px++) {
                    set_pixel(buffer, 
                              x_offset + j * PIXEL_SIZE + px, 
                              y_offset + i * PIXEL_SIZE + py, 
                              buf_width, r, g, b);
                }
            }
        }
    }
}

/**
 * @brief Simple function to "draw" text onto the PNG buffer.
 * In a real application, a font rendering library would be used. 
 * For this example, we just color a block where text should be.
 */
void draw_text_placeholder(unsigned char *buffer, int buf_width, int x, int y, const char* text, unsigned char r, unsigned char g, unsigned char b) {
    // This is a placeholder. Real text rendering is complex.
    // For now, we draw a single white pixel to indicate the text start position.
    set_pixel(buffer, x, y, buf_width, r, g, b);

    // If we wanted to draw simple text, we'd iterate over characters and a simple font map.
    // We will skip complex text rendering to keep the example focused on the stb_image usage.
}

/**
 * @brief Renders the 4-image comparison set for one test case onto the PNG buffer.
 */
void render_test_to_png(unsigned char *buffer, int buf_width, const TestResult *r, int x_set, int y_set) {
    // Text Label: T:'A' (x,y) | P:'B' | Error:XX.XX% (NO)
    const EstimationResult *best_fit = &r->classification_results[r->best_match_index];
    char label[150];
    double pixel_error_sum = calculate_pixel_error_sum(r->observed_image, r->best_estimated_image);
    double pixel_error_percent = (pixel_error_sum / MAX_PIXEL_ERROR) * 100.0;
    
    sprintf(label, "T:'%s' (%.2f,%.2f) | P:'%s' | Error:%.2f%% (%s)", 
            CHAR_NAMES[r->true_char_index], r->true_alpha[0], r->true_alpha[1], 
            CHAR_NAMES[r->best_match_index], pixel_error_percent, 
            (r->true_char_index == r->best_match_index) ? "YES" : "NO");
    
    // 1. Draw Images
    int img_step = IMG_SIZE + IMG_SPACING;
    
    // TRUE CLEAN
    render_single_image_to_png(buffer, buf_width, r->true_image, x_set, y_set + TEXT_HEIGHT, 0); 
    
    // OBSERVED NOISY
    render_single_image_to_png(buffer, buf_width, r->observed_image, x_set + img_step, y_set + TEXT_HEIGHT, 0);
    
    // BEST ESTIMATED
    render_single_image_to_png(buffer, buf_width, r->best_estimated_image, x_set + 2 * img_step, y_set + TEXT_HEIGHT, 0);
    
    // ERROR DIFF
    render_single_image_to_png(buffer, buf_width, r->best_diff_image, x_set + 3 * img_step, y_set + TEXT_HEIGHT, 1);
    
    // 2. Draw Text Placeholder (at y_set + TEXT_HEIGHT + IMG_SIZE + 5)
    // NOTE: True text drawing is omitted, use a single pixel marker.
    draw_text_placeholder(buffer, buf_width, x_set, y_set + TEXT_HEIGHT + IMG_SIZE + 5, label, 255, 255, 255);
}

/**
 * @brief Generates the final PNG file using stb_image_write.
 */
void generate_png_file() {
    // Allocate the pixel buffer (Width * Height * 3 bytes per pixel)
    long buffer_size = (long)PNG_WIDTH * PNG_HEIGHT * NUM_CHANNELS;
    unsigned char *buffer = (unsigned char *)calloc(buffer_size, 1);
    if (buffer == NULL) {
        printf("\nERROR: Failed to allocate buffer for PNG.\n");
        return;
    }

    // Fill background black (already done by calloc, but good practice)
    // memset(buffer, 0, buffer_size);

    // Draw column titles (simple white pixel marker)
    int initial_x = SET_SPACING + IMG_SIZE / 2;
    int initial_y = TEXT_HEIGHT / 2;
    int img_step = IMG_SIZE + IMG_SPACING;
    draw_text_placeholder(buffer, PNG_WIDTH, initial_x, initial_y, "TRUE CLEAN", 255, 255, 255);
    draw_text_placeholder(buffer, PNG_WIDTH, initial_x + img_step, initial_y, "OBSERVED NOISY", 255, 255, 255);
    draw_text_placeholder(buffer, PNG_WIDTH, initial_x + 2 * img_step, initial_y, "BEST ESTIMATED", 255, 255, 255);
    draw_text_placeholder(buffer, PNG_WIDTH, initial_x + 3 * img_step, initial_y, "ERROR DIFF", 255, 255, 255);
    
    // Render the test sets in a single row
    int actual_render_limit = (NUM_TESTS < PNG_RENDER_LIMIT) ? NUM_TESTS : PNG_RENDER_LIMIT;
    int y_set = TEXT_HEIGHT; // Leave space for the column titles
    
    for (int k = 0; k < actual_render_limit; k++) {
        int x_set = SET_SPACING + k * SET_WIDTH;
        render_test_to_png(buffer, PNG_WIDTH, &all_results[k], x_set, y_set, 0);
    }

    // Save the buffer to a PNG file
    int success = stbi_write_png("network.png", PNG_WIDTH, PNG_HEIGHT, NUM_CHANNELS, buffer, PNG_WIDTH * NUM_CHANNELS);

    free(buffer);

    if (success) {
        printf("\n\n======================================================\n");
        printf("PNG Output Complete: network.png created (First %d tests).\n", actual_render_limit);
        printf("Size: %d x %d pixels.\n", PNG_WIDTH, PNG_HEIGHT);
        printf("======================================================\n");
    } else {
        printf("\nERROR: Failed to write network.png using stb_image_write.\n");
    }
}


// --- Main Execution (Simplified for this answer) ---

// Placeholder functions for the optimizer to compile
void apply_deformation(Point *point, const double alpha[NUM_DEFORMATIONS]) { /* ... */ }
Point get_deformed_point(const double t, const Ideal_Curve_Params *const params, const double alpha[NUM_DEFORMATIONS]) { return (Point){0}; }
void draw_curve(const double alpha[NUM_DEFORMATIONS], Generated_Image img, const Ideal_Curve_Params *const ideal_params) { /* ... */ }
void extract_geometric_features(const Generated_Image img, Feature_Vector features_out) { /* ... */ }
double calculate_feature_loss_L2(const Feature_Vector generated, const Feature_Vector observed) { return 0.0; }
double calculate_pixel_loss_L2(const Generated_Image generated, const Generated_Image observed) { return 0.0; }
double calculate_combined_loss(const Generated_Image generated_img, const Feature_Vector generated_features, const Generated_Image observed_img, const Feature_Vector observed_features) { return 0.0; }
void calculate_gradient(const Generated_Image observed_img, const Feature_Vector observed_features, const Deformation_Coefficients *const alpha, const double loss_base, double grad_out[NUM_DEFORMATIONS], const Ideal_Curve_Params *const ideal_params) { /* ... */ }
void generate_target_image(Generated_Image image_out, const double true_alpha[NUM_DEFORMATIONS], const Ideal_Curve_Params *const ideal_params, int add_noise) { /* ... */ }
double calculate_pixel_error_sum(const Generated_Image obs, const Generated_Image est) { return 0.0; }
void run_optimization(const Generated_Image observed_image, const Feature_Vector observed_features, int ideal_char_index, EstimationResult *result, int print_trace) { /* ... */ }
void calculate_difference_image(const Generated_Image obs, const Generated_Image est, Generated_Image diff) { /* ... */ }
void summarize_results_console() { /* ... */ }


void initialize_dummy_results() {
    // Fill all_results with dummy data for image rendering demonstration
    for (int i = 0; i < PNG_RENDER_LIMIT; i++) {
        all_results[i].id = i + 1;
        all_results[i].true_char_index = i; // 'A', 'B', 'C', ...
        all_results[i].best_match_index = (i == 3) ? 1 : i; // Dummy error: Test 4 ('D') misclassified as 'B'
        all_results[i].true_alpha[0] = 0.14; all_results[i].true_alpha[1] = -0.05;

        // Populate images with simple grayscale patterns for demonstration
        for (int r = 0; r < GRID_SIZE; r++) {
            for (int c = 0; c < GRID_SIZE; c++) {
                double val = (r + c) / (2.0 * GRID_SIZE);
                all_results[i].true_image[r][c] = val * 0.5;
                all_results[i].observed_image[r][c] = val * 0.5 + 0.1;
                all_results[i].best_estimated_image[r][c] = val * 0.5;
                all_results[i].best_diff_image[r][c] = 0.2 + (r % 2) * 0.1;
            }
        }
    }
}


int main(void) {
    // Replaced the actual computation with a dummy initializer for demonstration
    // In a final setup, you would call your optimization loop here:
    // for (int i = 0; i < NUM_TESTS; i++) { run_classification_test(...) }
    
    initialize_dummy_results();

    // 1. (Omitted) Print the console-based summary

    // 2. Generate the PNG file 
    generate_png_file();

    return 0;
}
