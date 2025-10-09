#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h> 

// --- STB Image Configuration ---
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define M_PI 3.14159265358979323846

// --- Global Configuration ---
#define GRID_SIZE 32        
#define NUM_DEFORMATIONS 2  
#define NUM_VECTORS 16      
#define NUM_BINS 32         
#define NUM_FEATURES (NUM_VECTORS * NUM_BINS) 
#define PIXEL_LOSS_WEIGHT 2.5 
#define NUM_POINTS 200
#define ITERATIONS 200      // INCREASED from 100 to 200 for better convergence
#define GRADIENT_EPSILON 0.01 
#define NUM_IDEAL_CHARS 62  // 26 A-Z + 26 a-z + 10 0-9
#define NUM_CONTROL_POINTS 9 
#define MAX_SEGMENTS 100 
#define PIXEL_SIZE 2    
#define IMG_SIZE (GRID_SIZE * PIXEL_SIZE) 
#define IMG_SPACING 5   
#define TEXT_HEIGHT 15  
#define SET_SPACING 25  
#define NUM_CHANNELS 3 
#define SEGMENTATION_THRESHOLD 0.5
 // Intensity > 0.5 (i.e., pixel value < 128)
#define MAX_SEGMENTS_PER_ROW 10
 // INCREASED for wider output PNG

// Stroke widths to test (in GRID_SIZE pixels)
const int STROKE_WIDTHS[] = {1, 2, 4, 8};
const int NUM_STROKE_WIDTHS = sizeof(STROKE_WIDTHS) / sizeof(STROKE_WIDTHS[0]);

// --- Data Structures ---

typedef struct { double x; double y; } Point;
typedef struct { const Point control_points[NUM_CONTROL_POINTS]; } Ideal_Curve_Params; 
typedef struct { double alpha[NUM_DEFORMATIONS]; } Deformation_Coefficients;
typedef double Generated_Image[GRID_SIZE][GRID_SIZE]; 
typedef double Feature_Vector[NUM_FEATURES]; 
typedef struct { double estimated_alpha[NUM_DEFORMATIONS]; double final_loss; int stroke_width; double loss_history[1]; } EstimationResult;

typedef struct {
    int start; 
    int end; 
} Boundary;

typedef struct {
    int x_start, x_end; 
    int y_start, y_end; 
    Generated_Image resized_img; 
    int best_match_index; 
    double final_loss;
    double estimated_alpha[NUM_DEFORMATIONS];
    int best_stroke_width; 
} SegmentResult;


// --- Memory Tracking Globals (Simplified for brevity) ---
size_t total_allocated_bytes = 0;
size_t total_freed_bytes = 0;

// --- Function Prototypes ---
void* safe_malloc(size_t size);
void safe_free(void *ptr, size_t size);
void apply_deformation(Point *point, const double alpha[NUM_DEFORMATIONS]);
Point get_deformed_point(const double t, const Ideal_Curve_Params *const params, const double alpha[NUM_DEFORMATIONS]);
void draw_curve(const double alpha[NUM_DEFORMATIONS], Generated_Image img, const Ideal_Curve_Params *const ideal_params, int stroke_width);
void extract_geometric_features(const Generated_Image img, Feature_Vector features_out);
double calculate_feature_loss_L2(const Feature_Vector generated, const Feature_Vector observed);
double calculate_pixel_loss_L2(const Generated_Image generated, const Generated_Image observed);
double calculate_combined_loss(const Generated_Image generated_img, const Feature_Vector generated_features,
                               const Generated_Image observed_img, const Feature_Vector observed_features);
void calculate_gradient(const Generated_Image observed_img, const Feature_Vector observed_features, 
                        const Deformation_Coefficients *const alpha, const double loss_base, 
                        double grad_out[NUM_DEFORMATIONS], const Ideal_Curve_Params *const ideal_params, int stroke_width);
void run_optimization(const Generated_Image observed_image, const Feature_Vector observed_features, 
                      int ideal_char_index, EstimationResult *result, int stroke_width);
void set_pixel(unsigned char *buffer, int x, int y, int width, int height, unsigned char r, unsigned char g, unsigned char b);
void get_pixel_color(double intensity, int is_error_map, unsigned char *r, unsigned char *g, unsigned char *b);
void draw_text_placeholder_box(unsigned char *buffer, int buf_width, int buf_height, int x, int y, int width, int height, unsigned char r, unsigned char g, unsigned char b);
int load_image_stb(const char *filename, double **data_out, int *width_out, int *height_out); 
void resize_segment(const double *full_data, int full_width, int full_height, 
                    int x_start, int x_end, int y_start, int y_end, 
                    Generated_Image segment_out);
int find_content_intervals_bool(const unsigned char *bool_array, int size, int min_length, Boundary *boundaries_out);
int segment_image_naive(const double *full_data, int full_width, int full_height, SegmentResult *segments_out); 
void recognize_segment(SegmentResult *segment);
void generate_segment_png(const SegmentResult *segments, int num_segments, const double *full_data, int full_width, int full_height);


// --- Fixed Ideal Curves (UPDATED for thicker, smoother strokes) ---
const char *CHAR_NAMES[NUM_IDEAL_CHARS] = {
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", 
    "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", 
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p",
    "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
};
const Ideal_Curve_Params IDEAL_TEMPLATES[NUM_IDEAL_CHARS] = {
    // Uppercase Letters (A-Z) - Adjusted for high-contrast/bold font
    // A
    [0] = {.control_points = {{.x = 0.5, .y = 0.1}, {.x = 0.2, .y = 0.9}, {.x = 0.8, .y = 0.9}, {.x = 0.5, .y = 0.1}, 
        {.x = 0.4, .y = 0.6}, {.x = 0.6, .y = 0.6}, {.x = 0.5, .y = 0.6}, {.x = 0.2, .y = 0.9}, 
        {.x = 0.8, .y = 0.9} 
    }},
    // B 
    [1] = {.control_points = {{.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.9}, {.x = 0.7, .y = 0.8}, {.x = 0.8, .y = 0.5}, 
        {.x = 0.7, .y = 0.2}, {.x = 0.2, .y = 0.5}, {.x = 0.8, .y = 0.5}, {.x = 0.2, .y = 0.1}, 
        {.x = 0.2, .y = 0.9} 
    }},
    // C 
    [2] = {.control_points = {{.x = 0.8, .y = 0.2}, {.x = 0.5, .y = 0.1}, {.x = 0.2, .y = 0.3}, {.x = 0.2, .y = 0.7}, 
        {.x = 0.5, .y = 0.9}, {.x = 0.8, .y = 0.8}, {.x = 0.5, .y = 0.9}, {.x = 0.2, .y = 0.7}, 
        {.x = 0.2, .y = 0.3} 
    }},
    // D 
    [3] = {.control_points = {{.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.9}, {.x = 0.8, .y = 0.5}, {.x = 0.2, .y = 0.1}, 
        {.x = 0.8, .y = 0.5}, {.x = 0.7, .y = 0.2}, {.x = 0.7, .y = 0.8}, {.x = 0.2, .y = 0.5}, 
        {.x = 0.2, .y = 0.9} 
    }},
    // E 
    [4] = {.control_points = {{.x = 0.8, .y = 0.1}, {.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.9}, {.x = 0.8, .y = 0.9}, 
        {.x = 0.2, .y = 0.5}, {.x = 0.7, .y = 0.5}, {.x = 0.2, .y = 0.5}, {.x = 0.2, .y = 0.1}, 
        {.x = 0.2, .y = 0.9} 
    }},
    // F 
    [5] = {.control_points = {{.x = 0.8, .y = 0.1}, {.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.9}, {.x = 0.2, .y = 0.5}, 
        {.x = 0.6, .y = 0.5}, {.x = 0.2, .y = 0.5}, {.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.9}, 
        {.x = 0.8, .y = 0.1} 
    }},
    // G 
    [6] = {.control_points = {{.x = 0.8, .y = 0.2}, {.x = 0.5, .y = 0.1}, {.x = 0.2, .y = 0.3}, {.x = 0.2, .y = 0.7}, 
        {.x = 0.5, .y = 0.9}, {.x = 0.8, .y = 0.7}, {.x = 0.8, .y = 0.5}, {.x = 0.5, .y = 0.5}, 
        {.x = 0.8, .y = 0.7} 
    }},
    // H 
    [7] = {.control_points = {{.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.9}, {.x = 0.8, .y = 0.9}, {.x = 0.8, .y = 0.1}, 
        {.x = 0.2, .y = 0.5}, {.x = 0.8, .y = 0.5}, {.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.9}, 
        {.x = 0.8, .y = 0.5} 
    }},
    // I 
    [8] = {.control_points = {{.x = 0.5, .y = 0.1}, {.x = 0.5, .y = 0.9}, {.x = 0.5, .y = 0.1}, {.x = 0.5, .y = 0.9}, 
        {.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.1}, {.x = 0.2, .y = 0.9}, {.x = 0.8, .y = 0.9}, 
        {.x = 0.5, .y = 0.5} 
    }},
    // J 
    [9] = {.control_points = {{.x = 0.8, .y = 0.1}, {.x = 0.8, .y = 0.7}, {.x = 0.5, .y = 0.9}, {.x = 0.2, .y = 0.7}, 
        {.x = 0.2, .y = 0.5}, {.x = 0.8, .y = 0.1}, {.x = 0.8, .y = 0.7}, {.x = 0.5, .y = 0.9}, 
        {.x = 0.2, .y = 0.7} 
    }},
    // K 
    [10] = {.control_points = {{.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.9}, {.x = 0.2, .y = 0.5}, {.x = 0.8, .y = 0.1}, 
        {.x = 0.2, .y = 0.5}, {.x = 0.8, .y = 0.9}, {.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.9}, 
        {.x = 0.8, .y = 0.1} 
    }},
    // L 
    [11] = {.control_points = {{.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.9}, {.x = 0.8, .y = 0.9}, {.x = 0.2, .y = 0.1}, 
        {.x = 0.8, .y = 0.9}, {.x = 0.2, .y = 0.9}, {.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.9}, 
        {.x = 0.8, .y = 0.1} 
    }},
    // M 
    [12] = {.control_points = {{.x = 0.1, .y = 0.9}, {.x = 0.1, .y = 0.1}, {.x = 0.5, .y = 0.5}, {.x = 0.9, .y = 0.1}, 
        {.x = 0.9, .y = 0.9}, {.x = 0.1, .y = 0.9}, {.x = 0.5, .y = 0.5}, {.x = 0.9, .y = 0.9}, 
        {.x = 0.1, .y = 0.1} 
    }},
    // N 
    [13] = {.control_points = {{.x = 0.1, .y = 0.9}, {.x = 0.1, .y = 0.1}, {.x = 0.9, .y = 0.9}, {.x = 0.9, .y = 0.1}, 
        {.x = 0.1, .y = 0.9}, {.x = 0.9, .y = 0.9}, {.x = 0.1, .y = 0.1}, {.x = 0.9, .y = 0.1}, 
        {.x = 0.9, .y = 0.9} 
    }},
    // O 
    [14] = {.control_points = {{.x = 0.5, .y = 0.1}, {.x = 0.85, .y = 0.3}, {.x = 0.85, .y = 0.7}, 
        {.x = 0.5, .y = 0.9}, {.x = 0.15, .y = 0.7}, {.x = 0.15, .y = 0.3}, 
        {.x = 0.5, .y = 0.1}, {.x = 0.5, .y = 0.1}, {.x = 0.5, .y = 0.1}  
    }},
    // P 
    [15] = {.control_points = {{.x = 0.2, .y = 0.9}, {.x = 0.2, .y = 0.1}, {.x = 0.7, .y = 0.2}, {.x = 0.8, .y = 0.5}, 
        {.x = 0.2, .y = 0.5}, {.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.5}, {.x = 0.2, .y = 0.9}, 
        {.x = 0.2, .y = 0.5} 
    }},
    // Q 
    [16] = {.control_points = {{.x = 0.5, .y = 0.1}, {.x = 0.85, .y = 0.3}, {.x = 0.85, .y = 0.7}, 
        {.x = 0.5, .y = 0.9}, {.x = 0.15, .y = 0.7}, {.x = 0.15, .y = 0.3}, 
        {.x = 0.5, .y = 0.1}, {.x = 0.6, .y = 0.7}, {.x = 0.8, .y = 0.9}  
    }},
    // R 
    [17] = {.control_points = {{.x = 0.2, .y = 0.9}, {.x = 0.2, .y = 0.1}, {.x = 0.7, .y = 0.2}, {.x = 0.8, .y = 0.5}, 
        {.x = 0.2, .y = 0.5}, {.x = 0.8, .y = 0.9}, {.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.9}, 
        {.x = 0.2, .y = 0.5} 
    }},
    // S 
    [18] = {.control_points = {{.x = 0.8, .y = 0.1}, {.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.5}, {.x = 0.8, .y = 0.5}, 
        {.x = 0.8, .y = 0.9}, {.x = 0.2, .y = 0.9}, {.x = 0.8, .y = 0.1}, {.x = 0.2, .y = 0.9}, 
        {.x = 0.2, .y = 0.1} 
    }},
    // T 
    [19] = {.control_points = {{.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.1}, {.x = 0.5, .y = 0.1}, {.x = 0.5, .y = 0.9}, 
        {.x = 0.5, .y = 0.1}, {.x = 0.5, .y = 0.9}, {.x = 0.5, .y = 0.5}, {.x = 0.5, .y = 0.1}, 
        {.x = 0.8, .y = 0.1} 
    }},
    // U 
    [20] = {.control_points = {{.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.8}, {.x = 0.5, .y = 0.9}, {.x = 0.8, .y = 0.8}, 
        {.x = 0.8, .y = 0.1}, {.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.8}, {.x = 0.5, .y = 0.9}, 
        {.x = 0.2, .y = 0.8} 
    }},
    // V 
    [21] = {.control_points = {{.x = 0.2, .y = 0.1}, {.x = 0.5, .y = 0.9}, {.x = 0.8, .y = 0.1}, {.x = 0.2, .y = 0.1}, 
        {.x = 0.8, .y = 0.1}, {.x = 0.5, .y = 0.9}, {.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.1}, 
        {.x = 0.5, .y = 0.9} 
    }},
    // W 
    [22] = {.control_points = {{.x = 0.1, .y = 0.1}, {.x = 0.3, .y = 0.9}, {.x = 0.5, .y = 0.5}, {.x = 0.7, .y = 0.9}, 
        {.x = 0.9, .y = 0.1}, {.x = 0.1, .y = 0.1}, {.x = 0.9, .y = 0.1}, {.x = 0.5, .y = 0.5}, 
        {.x = 0.3, .y = 0.9} 
    }},
    // X 
    [23] = {.control_points = {{.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.9}, {.x = 0.8, .y = 0.1}, {.x = 0.2, .y = 0.9}, 
        {.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.9}, {.x = 0.8, .y = 0.1}, {.x = 0.2, .y = 0.9}, 
        {.x = 0.5, .y = 0.5} 
    }},
    // Y 
    [24] = {.control_points = {{.x = 0.2, .y = 0.1}, {.x = 0.5, .y = 0.5}, {.x = 0.8, .y = 0.1}, {.x = 0.5, .y = 0.5}, 
        {.x = 0.5, .y = 0.9}, {.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.1}, {.x = 0.5, .y = 0.9}, 
        {.x = 0.5, .y = 0.5} 
    }},
    // Z 
    [25] = {.control_points = {{.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.1}, {.x = 0.2, .y = 0.9}, {.x = 0.8, .y = 0.9}, 
        {.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.9}, {.x = 0.2, .y = 0.5}, {.x = 0.8, .y = 0.5}, 
        {.x = 0.2, .y = 0.1} 
    }},

    // Lowercase Letters (a-z) - Adjusted for better curve definition
    // a
    [26] = {.control_points = {{.x = 0.7, .y = 0.5}, {.x = 0.5, .y = 0.3}, {.x = 0.3, .y = 0.5}, {.x = 0.3, .y = 0.7}, 
        {.x = 0.5, .y = 0.9}, {.x = 0.7, .y = 0.7}, {.x = 0.7, .y = 0.3}, {.x = 0.7, .y = 0.9}, 
        {.x = 0.3, .y = 0.9} 
    }},
    // b
    [27] = {.control_points = {{.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.9}, {.x = 0.7, .y = 0.8}, {.x = 0.7, .y = 0.6}, 
        {.x = 0.2, .y = 0.6}, {.x = 0.2, .y = 0.9}, {.x = 0.7, .y = 0.6}, {.x = 0.2, .y = 0.1}, 
        {.x = 0.7, .y = 0.8} 
    }},
    // c
    [28] = {.control_points = {{.x = 0.7, .y = 0.4}, {.x = 0.5, .y = 0.2}, {.x = 0.3, .y = 0.4}, {.x = 0.3, .y = 0.8}, 
        {.x = 0.5, .y = 0.9}, {.x = 0.7, .y = 0.8}, {.x = 0.3, .y = 0.4}, {.x = 0.3, .y = 0.8}, 
        {.x = 0.7, .y = 0.4} 
    }},
    // d
    [29] = {.control_points = {{.x = 0.8, .y = 0.1}, {.x = 0.8, .y = 0.9}, {.x = 0.3, .y = 0.8}, {.x = 0.3, .y = 0.6}, 
        {.x = 0.8, .y = 0.6}, {.x = 0.8, .y = 0.9}, {.x = 0.3, .y = 0.6}, {.x = 0.8, .y = 0.1}, 
        {.x = 0.3, .y = 0.8} 
    }},
    // e
    [30] = {.control_points = {{.x = 0.2, .y = 0.5}, {.x = 0.8, .y = 0.5}, {.x = 0.2, .y = 0.3}, {.x = 0.5, .y = 0.1}, 
        {.x = 0.8, .y = 0.3}, {.x = 0.8, .y = 0.7}, {.x = 0.5, .y = 0.9}, {.x = 0.2, .y = 0.7}, 
        {.x = 0.2, .y = 0.5} 
    }},
    // f
    [31] = {.control_points = {{.x = 0.5, .y = 0.1}, {.x = 0.5, .y = 0.9}, {.x = 0.3, .y = 0.1}, {.x = 0.7, .y = 0.1}, 
        {.x = 0.2, .y = 0.5}, {.x = 0.6, .y = 0.5}, {.x = 0.5, .y = 0.1}, {.x = 0.5, .y = 0.9}, 
        {.x = 0.2, .y = 0.5} 
    }},
    // g
    [32] = {.control_points = {{.x = 0.7, .y = 0.7}, {.x = 0.5, .y = 0.3}, {.x = 0.3, .y = 0.5}, {.x = 0.3, .y = 0.7}, 
        {.x = 0.5, .y = 0.9}, {.x = 0.7, .y = 0.7}, {.x = 0.7, .y = 0.9}, {.x = 0.5, .y = 1.1}, 
        {.x = 0.3, .y = 1.1} 
    }},
    // h
    [33] = {.control_points = {{.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.9}, {.x = 0.2, .y = 0.6}, {.x = 0.7, .y = 0.5}, 
        {.x = 0.7, .y = 0.9}, {.x = 0.2, .y = 0.9}, {.x = 0.2, .y = 0.1}, {.x = 0.7, .y = 0.9}, 
        {.x = 0.2, .y = 0.6} 
    }},
    // i
    [34] = {.control_points = {{.x = 0.5, .y = 0.5}, {.x = 0.5, .y = 0.9}, {.x = 0.5, .y = 0.2}, {.x = 0.5, .y = 0.2}, 
        {.x = 0.5, .y = 0.2}, {.x = 0.5, .y = 0.9}, {.x = 0.5, .y = 0.5}, {.x = 0.5, .y = 0.5}, 
        {.x = 0.5, .y = 0.1} 
    }},
    // j
    [35] = {.control_points = {{.x = 0.6, .y = 0.5}, {.x = 0.6, .y = 0.9}, {.x = 0.4, .y = 1.1}, {.x = 0.2, .y = 1.0}, 
        {.x = 0.6, .y = 0.2}, {.x = 0.6, .y = 0.1}, {.x = 0.6, .y = 0.5}, {.x = 0.6, .y = 0.9}, 
        {.x = 0.6, .y = 0.1} 
    }},
    // k
    [36] = {.control_points = {{.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.9}, {.x = 0.2, .y = 0.6}, {.x = 0.8, .y = 0.3}, 
        {.x = 0.2, .y = 0.6}, {.x = 0.8, .y = 0.9}, {.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.9}, 
        {.x = 0.8, .y = 0.3} 
    }},
    // l
    [37] = {.control_points = {{.x = 0.5, .y = 0.1}, {.x = 0.5, .y = 0.9}, {.x = 0.5, .y = 0.1}, {.x = 0.5, .y = 0.9}, 
        {.x = 0.5, .y = 0.5}, {.x = 0.5, .y = 0.5}, {.x = 0.5, .y = 0.1}, {.x = 0.5, .y = 0.9}, 
        {.x = 0.5, .y = 0.1} 
    }},
    // m
    [38] = {.control_points = {{.x = 0.1, .y = 0.9}, {.x = 0.1, .y = 0.5}, {.x = 0.3, .y = 0.3}, {.x = 0.5, .y = 0.5}, 
        {.x = 0.5, .y = 0.9}, {.x = 0.7, .y = 0.3}, {.x = 0.9, .y = 0.5}, {.x = 0.9, .y = 0.9}, 
        {.x = 0.1, .y = 0.5} 
    }},
    // n
    [39] = {.control_points = {{.x = 0.2, .y = 0.9}, {.x = 0.2, .y = 0.5}, {.x = 0.4, .y = 0.3}, {.x = 0.7, .y = 0.5}, 
        {.x = 0.7, .y = 0.9}, {.x = 0.2, .y = 0.5}, {.x = 0.7, .y = 0.9}, {.x = 0.2, .y = 0.9}, 
        {.x = 0.4, .y = 0.3} 
    }},
    // o
    [40] = {.control_points = {{.x = 0.5, .y = 0.3}, {.x = 0.7, .y = 0.4}, {.x = 0.7, .y = 0.8}, 
        {.x = 0.5, .y = 0.9}, {.x = 0.3, .y = 0.8}, {.x = 0.3, .y = 0.4}, 
        {.x = 0.5, .y = 0.3}, {.x = 0.5, .y = 0.3}, {.x = 0.5, .y = 0.3}  
    }},
    // p
    [41] = {.control_points = {{.x = 0.2, .y = 1.1}, {.x = 0.2, .y = 0.3}, {.x = 0.7, .y = 0.4}, {.x = 0.7, .y = 0.6}, 
        {.x = 0.2, .y = 0.6}, {.x = 0.2, .y = 0.3}, {.x = 0.7, .y = 0.6}, {.x = 0.2, .y = 1.1}, 
        {.x = 0.2, .y = 0.6} 
    }},
    // q
    [42] = {.control_points = {{.x = 0.7, .y = 1.1}, {.x = 0.7, .y = 0.3}, {.x = 0.2, .y = 0.4}, {.x = 0.2, .y = 0.6}, 
        {.x = 0.7, .y = 0.6}, {.x = 0.7, .y = 0.3}, {.x = 0.2, .y = 0.6}, {.x = 0.7, .y = 1.1}, 
        {.x = 0.7, .y = 0.6} 
    }},
    // r
    [43] = {.control_points = {{.x = 0.2, .y = 0.9}, {.x = 0.2, .y = 0.5}, {.x = 0.4, .y = 0.3}, {.x = 0.7, .y = 0.4}, 
        {.x = 0.7, .y = 0.5}, {.x = 0.2, .y = 0.5}, {.x = 0.7, .y = 0.5}, {.x = 0.2, .y = 0.9}, 
        {.x = 0.4, .y = 0.3} 
    }},
    // s
    [44] = {.control_points = {{.x = 0.7, .y = 0.3}, {.x = 0.3, .y = 0.3}, {.x = 0.3, .y = 0.6}, {.x = 0.7, .y = 0.6}, 
        {.x = 0.7, .y = 0.9}, {.x = 0.3, .y = 0.9}, {.x = 0.7, .y = 0.3}, {.x = 0.3, .y = 0.9}, 
        {.x = 0.3, .y = 0.3} 
    }},
    // t
    [45] = {.control_points = {{.x = 0.5, .y = 0.1}, {.x = 0.5, .y = 0.9}, {.x = 0.5, .y = 0.9}, {.x = 0.3, .y = 0.4}, 
        {.x = 0.7, .y = 0.4}, {.x = 0.5, .y = 0.1}, {.x = 0.5, .y = 0.9}, {.x = 0.3, .y = 0.4}, 
        {.x = 0.7, .y = 0.4} 
    }},
    // u
    [46] = {.control_points = {{.x = 0.2, .y = 0.5}, {.x = 0.2, .y = 0.8}, {.x = 0.5, .y = 0.9}, {.x = 0.8, .y = 0.8}, 
        {.x = 0.8, .y = 0.5}, {.x = 0.8, .y = 0.9}, {.x = 0.2, .y = 0.5}, {.x = 0.8, .y = 0.9}, 
        {.x = 0.5, .y = 0.9} 
    }},
    // v
    [47] = {.control_points = {{.x = 0.2, .y = 0.5}, {.x = 0.5, .y = 0.9}, {.x = 0.8, .y = 0.5}, {.x = 0.2, .y = 0.5}, 
        {.x = 0.8, .y = 0.5}, {.x = 0.5, .y = 0.9}, {.x = 0.2, .y = 0.5}, {.x = 0.8, .y = 0.5}, 
        {.x = 0.5, .y = 0.9} 
    }},
    // w
    [48] = {.control_points = {{.x = 0.1, .y = 0.5}, {.x = 0.3, .y = 0.9}, {.x = 0.5, .y = 0.7}, {.x = 0.7, .y = 0.9}, 
        {.x = 0.9, .y = 0.5}, {.x = 0.1, .y = 0.5}, {.x = 0.9, .y = 0.5}, {.x = 0.5, .y = 0.7}, 
        {.x = 0.3, .y = 0.9} 
    }},
    // x
    [49] = {.control_points = {{.x = 0.2, .y = 0.5}, {.x = 0.8, .y = 0.9}, {.x = 0.8, .y = 0.5}, {.x = 0.2, .y = 0.9}, 
        {.x = 0.2, .y = 0.5}, {.x = 0.8, .y = 0.9}, {.x = 0.8, .y = 0.5}, {.x = 0.2, .y = 0.9}, 
        {.x = 0.5, .y = 0.7} 
    }},
    // y
    [50] = {.control_points = {{.x = 0.2, .y = 0.5}, {.x = 0.5, .y = 0.9}, {.x = 0.8, .y = 0.5}, {.x = 0.5, .y = 0.9}, 
        {.x = 0.5, .y = 1.1}, {.x = 0.2, .y = 0.5}, {.x = 0.8, .y = 0.5}, {.x = 0.5, .y = 1.1}, 
        {.x = 0.5, .y = 0.9} 
    }},
    // z
    [51] = {.control_points = {{.x = 0.2, .y = 0.5}, {.x = 0.8, .y = 0.5}, {.x = 0.2, .y = 0.9}, {.x = 0.8, .y = 0.9}, 
        {.x = 0.2, .y = 0.5}, {.x = 0.8, .y = 0.9}, {.x = 0.2, .y = 0.7}, {.x = 0.8, .y = 0.7}, 
        {.x = 0.2, .y = 0.5} 
    }},

    // Digits (0-9) - Adjusted
    // 0
    [52] = {.control_points = {{.x = 0.5, .y = 0.1}, {.x = 0.8, .y = 0.3}, {.x = 0.8, .y = 0.7}, 
        {.x = 0.5, .y = 0.9}, {.x = 0.2, .y = 0.7}, {.x = 0.2, .y = 0.3}, 
        {.x = 0.5, .y = 0.1}, {.x = 0.5, .y = 0.1}, {.x = 0.5, .y = 0.1}  
    }},
    // 1
    [53] = {.control_points = {{.x = 0.5, .y = 0.1}, {.x = 0.5, .y = 0.9}, {.x = 0.3, .y = 0.2}, 
        {.x = 0.7, .y = 0.9}, {.x = 0.3, .y = 0.9}, {.x = 0.5, .y = 0.1}, 
        {.x = 0.5, .y = 0.9}, {.x = 0.5, .y = 0.1}, {.x = 0.3, .y = 0.2} 
    }},
    // 2
    [54] = {.control_points = {{.x = 0.2, .y = 0.2}, {.x = 0.8, .y = 0.1}, {.x = 0.8, .y = 0.4}, 
        {.x = 0.2, .y = 0.6}, {.x = 0.8, .y = 0.9}, {.x = 0.2, .y = 0.9}, 
        {.x = 0.8, .y = 0.1}, {.x = 0.2, .y = 0.6}, {.x = 0.8, .y = 0.9} 
    }},
    // 3
    [55] = {.control_points = {{.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.2}, {.x = 0.5, .y = 0.5}, 
        {.x = 0.8, .y = 0.8}, {.x = 0.2, .y = 0.9}, {.x = 0.8, .y = 0.2}, 
        {.x = 0.5, .y = 0.5}, {.x = 0.8, .y = 0.8}, {.x = 0.2, .y = 0.1} 
    }},
    // 4
    [56] = {.control_points = {{.x = 0.6, .y = 0.1}, {.x = 0.2, .y = 0.5}, {.x = 0.8, .y = 0.5}, 
        {.x = 0.6, .y = 0.5}, {.x = 0.6, .y = 0.9}, {.x = 0.2, .y = 0.5}, 
        {.x = 0.6, .y = 0.1}, {.x = 0.6, .y = 0.9}, {.x = 0.8, .y = 0.5} 
    }},
    // 5
    [57] = {.control_points = {{.x = 0.8, .y = 0.1}, {.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.5}, 
        {.x = 0.8, .y = 0.5}, {.x = 0.8, .y = 0.9}, {.x = 0.2, .y = 0.9}, 
        {.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.9}, {.x = 0.2, .y = 0.5} 
    }},
    // 6
    [58] = {.control_points = {{.x = 0.7, .y = 0.1}, {.x = 0.2, .y = 0.3}, {.x = 0.2, .y = 0.7}, 
        {.x = 0.5, .y = 0.9}, {.x = 0.8, .y = 0.7}, {.x = 0.5, .y = 0.5}, 
        {.x = 0.2, .y = 0.7}, {.x = 0.7, .y = 0.1}, {.x = 0.5, .y = 0.9} 
    }},
    // 7
    [59] = {.control_points = {{.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.1}, {.x = 0.3, .y = 0.9}, 
        {.x = 0.7, .y = 0.9}, {.x = 0.8, .y = 0.1}, {.x = 0.3, .y = 0.9}, 
        {.x = 0.5, .y = 0.5}, {.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.1} 
    }},
    // 8
    [60] = {.control_points = {{.x = 0.5, .y = 0.1}, {.x = 0.8, .y = 0.3}, {.x = 0.5, .y = 0.5}, {.x = 0.2, .y = 0.3}, 
        {.x = 0.5, .y = 0.1}, {.x = 0.8, .y = 0.7}, {.x = 0.5, .y = 0.9}, {.x = 0.2, .y = 0.7}, 
        {.x = 0.5, .y = 0.5} 
    }},
    // 9
    [61] = {.control_points = {{.x = 0.5, .y = 0.1}, {.x = 0.8, .y = 0.2}, {.x = 0.5, .y = 0.4}, {.x = 0.2, .y = 0.2}, 
        {.x = 0.5, .y = 0.1}, {.x = 0.8, .y = 0.4}, {.x = 0.8, .y = 0.9}, {.x = 0.5, .y = 0.7}, 
        {.x = 0.8, .y = 0.9} 
    }}
};


// --- Memory Wrapper Functions ---

void* safe_malloc(size_t size) {
    void *ptr = malloc(size);
    if (ptr) {
        total_allocated_bytes += size;
    }
    return ptr;
}

void safe_free(void *ptr, size_t size) {
    if (ptr) {
        free(ptr);
        total_freed_bytes += size;
    }
}


// --- Core Recognition Functions ---

void apply_deformation(Point *point, const double alpha[NUM_DEFORMATIONS]) {
    point->x = point->x + alpha[0] * (point->y - 0.5);
    point->x = point->x + alpha[1] * sin(M_PI * point->y);
}

Point get_deformed_point(const double t, const Ideal_Curve_Params *const params, const double alpha[NUM_DEFORMATIONS]) {
    Point p = {.x = 0.0, .y = 0.0};
    const int N = NUM_CONTROL_POINTS; 
    const double segment_length_t = 1.0 / (N - 1); 

    int segment_index = (int)floor(t / segment_length_t);
    if (segment_index >= N - 1) segment_index = N - 2; 

    const Point P_start = params->control_points[segment_index];
    const Point P_end = params->control_points[segment_index + 1];

    const double segment_t = (t - segment_index * segment_length_t) / segment_length_t;

    p.x = P_start.x + (P_end.x - P_start.x) * segment_t;
    p.y = P_start.y + (P_end.y - P_start.y) * segment_t;

    apply_deformation(&p, alpha);

    p.x = fmax(0.0, fmin(GRID_SIZE - 1.0, p.x * GRID_SIZE));
    p.y = fmax(0.0, fmin(GRID_SIZE - 1.0, p.y * GRID_SIZE));

    return p;
}

// Function to calculate Gaussian value (used for smoothing/stroke width)
static double gaussian(double x, double y, double sigma) {
    return exp(-(x*x + y*y) / (2.0 * sigma*sigma));
}

// UPDATED draw_curve to include stroke width simulation via Gaussian smoothing
void draw_curve(const double alpha[NUM_DEFORMATIONS], Generated_Image img, const Ideal_Curve_Params *const ideal_params, int stroke_width) {
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            img[i][j] = 0.0;
        }
    }

    Generated_Image temp_img;
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            temp_img[i][j] = 0.0;
        }
    }

    // 1. Draw single-pixel curve to a temporary image
    for (int i = 0; i <= NUM_POINTS; i++) {
        const double t = (double)i / NUM_POINTS;
        const Point current_p = get_deformed_point(t, ideal_params, alpha);

        const int px = (int)round(current_p.x);
        const int py = (int)round(current_p.y);

        if (px >= 0 && px < GRID_SIZE && py >= 0 && py < GRID_SIZE) {
            temp_img[py][px] = 1.0; 
        }
    }

    // 2. Apply Gaussian blur to simulate stroke width
    if (stroke_width <= 1) {
        // No blurring for minimal stroke width
        memcpy(img, temp_img, sizeof(Generated_Image));
        return;
    }

    const double sigma = (double)stroke_width / 4.0; 
    const int radius = stroke_width / 2 + 1; // Slight increase in radius for better smoothing

    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            if (temp_img[i][j] > 0) { // If it's a curve pixel
                for (int ky = -radius; ky <= radius; ky++) {
                    for (int kx = -radius; kx <= radius; kx++) {
                        int ni = i + ky;
                        int nj = j + kx;

                        if (ni >= 0 && ni < GRID_SIZE && nj >= 0 && nj < GRID_SIZE) {
                            // Add Gaussian intensity to the neighborhood
                            double g = gaussian((double)kx, (double)ky, sigma);
                            img[ni][nj] = fmin(1.0, img[ni][nj] + g * temp_img[i][j]);
                        }
                    }
                }
            }
        }
    }
    
    // Normalize to 1.0
    double max_val = 0.0;
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            if (img[i][j] > max_val) max_val = img[i][j];
        }
    }
    if (max_val > 0.0) {
        for (int i = 0; i < GRID_SIZE; i++) {
            for (int j = 0; j < GRID_SIZE; j++) {
                img[i][j] /= max_val;
            }
        }
    }
}

void extract_geometric_features(const Generated_Image img, Feature_Vector features_out) {
    double vectors[NUM_VECTORS][2];
    for (int k = 0; k < NUM_VECTORS; k++) { 
        const double angle = 2.0 * M_PI * k / NUM_VECTORS; 
        vectors[k][0] = cos(angle);
        vectors[k][1] = sin(angle);
    }
    
    const double center = (GRID_SIZE - 1.0) / 2.0; 
    const double MAX_PROJECTION_MAGNITUDE = sqrt(center * center + center * center); 

    for (int k = 0; k < NUM_FEATURES; k++) {
        features_out[k] = 0.0;
    }

    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            const double intensity = img[i][j];
            if (intensity < 0.1) continue; 

            const double vx = (double)j - center;
            const double vy = (double)i - center;
            
            for (int k = 0; k < NUM_VECTORS; k++) {
                const double projection = (vx * vectors[k][0] + vy * vectors[k][1]);
                const double normalized_projection = projection / MAX_PROJECTION_MAGNITUDE; 
                
                int bin_index = (int)floor((normalized_projection + 1.0) * (NUM_BINS / 2.0));
                
                if (bin_index < 0) bin_index = 0;
                if (bin_index >= NUM_BINS) bin_index = NUM_BINS - 1;
                
                const int feature_index = k * NUM_BINS + bin_index;
                features_out[feature_index] += intensity;
            }
        }
    }
}

double calculate_feature_loss_L2(const Feature_Vector generated, const Feature_Vector observed) {
    double loss = 0.0;
    for (int k = 0; k < NUM_FEATURES; k++) {
        const double error = observed[k] - generated[k];
        loss += error * error;
    }
    return loss;
}

double calculate_pixel_loss_L2(const Generated_Image generated, const Generated_Image observed) {
    double loss = 0.0;
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            const double error = observed[i][j] - generated[i][j];
            loss += error * error;
        }
    }
    return loss;
}

double calculate_combined_loss(const Generated_Image generated_img, const Feature_Vector generated_features,
                               const Generated_Image observed_img, const Feature_Vector observed_features) {
    
    const double feature_loss = calculate_feature_loss_L2(generated_features, observed_features);
    const double pixel_loss = calculate_pixel_loss_L2(generated_img, observed_img);
    
    return feature_loss + PIXEL_LOSS_WEIGHT * pixel_loss;
}

void calculate_gradient(const Generated_Image observed_img, const Feature_Vector observed_features, 
                        const Deformation_Coefficients *const alpha, const double loss_base, 
                        double grad_out[NUM_DEFORMATIONS], const Ideal_Curve_Params *const ideal_params, int stroke_width) {
    
    const double epsilon = GRADIENT_EPSILON; 
    Generated_Image generated_img_perturbed; 
    Feature_Vector generated_features_perturbed; 

    for (int k = 0; k < NUM_DEFORMATIONS; k++) {
        Deformation_Coefficients alpha_perturbed = *alpha;
        alpha_perturbed.alpha[k] += epsilon; 

        draw_curve(alpha_perturbed.alpha, generated_img_perturbed, ideal_params, stroke_width);
        extract_geometric_features(generated_img_perturbed, generated_features_perturbed);
        
        const double loss_perturbed = calculate_combined_loss(generated_img_perturbed, generated_features_perturbed, observed_img, observed_features);

        grad_out[k] = (loss_perturbed - loss_base) / epsilon;
    }
}


void run_optimization(const Generated_Image observed_image, const Feature_Vector observed_features, 
                      int ideal_char_index, EstimationResult *result, int stroke_width) {
    
    const Ideal_Curve_Params *ideal_params = &IDEAL_TEMPLATES[ideal_char_index];
    
    Deformation_Coefficients alpha_hat; 
    // Increased base learning rate
    double learning_rate = 0.0000003; 
    const double min_learning_rate = 0.00000000005;
    double gradient[NUM_DEFORMATIONS];
    Generated_Image generated_image;
    Feature_Vector generated_features;
    double combined_loss;
    double prev_combined_loss = HUGE_VAL; 
    double current_feature_loss_only = HUGE_VAL;
    
    // **Improvement 2: Random Initialization**
    // Introduce a small random jitter to avoid starting exactly at a local minimum for (0,0)
    alpha_hat.alpha[0] = ((double)rand() / RAND_MAX - 0.5) * 0.2; 
    alpha_hat.alpha[1] = ((double)rand() / RAND_MAX - 0.5) * 0.2;

    for (int t = 0; t <= ITERATIONS; t++) {
        draw_curve(alpha_hat.alpha, generated_image, ideal_params, stroke_width);
        extract_geometric_features(generated_image, generated_features);
        
        current_feature_loss_only = calculate_feature_loss_L2(generated_features, observed_features);
        combined_loss = current_feature_loss_only + PIXEL_LOSS_WEIGHT * calculate_pixel_loss_L2(generated_image, observed_image);

        // More aggressive learning rate decay
        if (combined_loss > prev_combined_loss * 1.0005 && learning_rate > min_learning_rate) { 
            learning_rate *= 0.75; // More aggressive decay (0.75 vs 0.5)
        }

        prev_combined_loss = combined_loss;

        if (t < ITERATIONS) {
            calculate_gradient(observed_image, observed_features, &alpha_hat, combined_loss, gradient, ideal_params, stroke_width);
            
            double step_rate = (learning_rate > min_learning_rate) ? learning_rate : min_learning_rate;
            
            alpha_hat.alpha[0] -= step_rate * gradient[0];
            alpha_hat.alpha[1] -= step_rate * gradient[1];
        }
    }

    result->final_loss = current_feature_loss_only;
    result->stroke_width = stroke_width;
    memcpy(result->estimated_alpha, alpha_hat.alpha, sizeof(double) * NUM_DEFORMATIONS);
}


void recognize_segment(SegmentResult *segment) {
    Feature_Vector observed_features;
    extract_geometric_features(segment->resized_img, observed_features);
    
    double min_feature_loss = HUGE_VAL;
    int best_match_index = -1;
    int best_stroke_width = -1;

    EstimationResult current_result = {0}; 
    EstimationResult best_result = {0}; 

    // Iterate through all character templates (62) AND all stroke widths (4)
    for (int s = 0; s < NUM_STROKE_WIDTHS; s++) {
        int sw = STROKE_WIDTHS[s];
        for (int i = 0; i < NUM_IDEAL_CHARS; i++) {
            run_optimization(segment->resized_img, observed_features, i, &current_result, sw); 

            // Only compare based on feature loss, which is more robust to stroke width differences.
            if (current_result.final_loss < min_feature_loss) {
                min_feature_loss = current_result.final_loss;
                best_match_index = i;
                best_stroke_width = sw;
                best_result = current_result;
            }
        }
    }
    
    segment->best_match_index = best_match_index;
    segment->final_loss = min_feature_loss;
    segment->best_stroke_width = best_stroke_width;
    memcpy(segment->estimated_alpha, best_result.estimated_alpha, sizeof(double) * NUM_DEFORMATIONS);
}


// --- PNG Rendering Functions ---

void set_pixel(unsigned char *buffer, int x, int y, int width, int height, unsigned char r, unsigned char g, unsigned char b) {
    if (x >= 0 && x < width && y >= 0 && y < height) {
        long index = (long)y * width * NUM_CHANNELS + x * NUM_CHANNELS;
        buffer[index] = r;
        buffer[index + 1] = g;
        buffer[index + 2] = b;
    }
}

void get_pixel_color(double intensity, int is_error_map, unsigned char *r, unsigned char *g, unsigned char *b) {
    double clamped_intensity = fmax(0.0, fmin(1.0, intensity));

    if (is_error_map) {
        if (clamped_intensity > 0.3) {
            *r = 255; *g = 50; *b = 50; 
        } else if (clamped_intensity > 0.1) {
            *r = 255; *g = 150; *b = 0; 
        } else {
            *r = 0; *g = 0; *b = 0; 
        }
    } else {
        if (clamped_intensity > 0.6) {
            *r = 255; *g = 255; *b = 100; 
        } else if (clamped_intensity > 0.3) {
            *r = 100; *g = 100; *b = 100; 
        } else {
            *r = 0; *g = 0; *b = 0; 
        }
    }
}

void draw_text_placeholder_box(unsigned char *buffer, int buf_width, int buf_height, int x, int y, int width, int height, unsigned char r, unsigned char g, unsigned char b) {
    for(int py = 0; py < height; py++) {
        for(int px = 0; px < width; px++) {
            set_pixel(buffer, x + px, y + py, buf_width, buf_height, r / 3, g / 3, b / 3);
            if (py == 0 || py == height - 1 || px == 0 || px == width - 1) {
                 set_pixel(buffer, x + px, y + py, buf_width, buf_height, r, g, b);
            }
        }
    }
}

void render_single_image_to_png(unsigned char *buffer, int buf_width, int buf_height, const Generated_Image img, int x_offset, int y_offset, int is_error_map) {
    unsigned char r, g, b;
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            get_pixel_color(img[i][j], is_error_map, &r, &g, &b);
            
            for (int py = 0; py < PIXEL_SIZE; py++) {
                for (int px = 0; px < PIXEL_SIZE; px++) {
                    set_pixel(buffer, 
                              x_offset + j * PIXEL_SIZE + px, 
                              y_offset + i * PIXEL_SIZE + py, 
                              buf_width, buf_height, r, g, b);
                }
            }
        }
    }
}


// --- Image Loading and Preprocessing ---

int load_image_stb(const char *filename, double **data_out, int *width_out, int *height_out) {
    int channels = 0;
    
    unsigned char *img_data = stbi_load(filename, width_out, height_out, &channels, 1); 

    if (img_data == NULL) {
        fprintf(stderr, "Error: Failed to load image file '%s'. Ensure the file exists and is readable.\n", filename);
        // Try the other uploaded image name if the first one fails
        const char *fallback_filename = "1000000809.jpg"; 
        img_data = stbi_load(fallback_filename, width_out, height_out, &channels, 1); 
        if (img_data == NULL) {
             fprintf(stderr, "Error: Failed to load image file '%s'. Image processing aborted.\n", fallback_filename);
             return 0;
        }
        printf("Using fallback image '%s'.\n", fallback_filename);
    }
    
    size_t total_pixels = (size_t)(*width_out) * (*height_out);
    *data_out = (double *)safe_malloc(sizeof(double) * total_pixels);
    
    if (*data_out == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for image data.\n");
        stbi_image_free(img_data);
        return 0;
    }

    for (size_t i = 0; i < total_pixels; i++) {
        // Invert intensity: White (255) -> 0.0, Black (0) -> 1.0. 
        (*data_out)[i] = 1.0 - (img_data[i] / 255.0); 
    }
    
    stbi_image_free(img_data);
    printf("Successfully loaded image (%dx%d, inverted intensity).\n", *width_out, *height_out);
    return 1;
}

void resize_segment(const double *full_data, int full_width, int full_height, 
                    int x_start, int x_end, int y_start, int y_end, 
                    Generated_Image segment_out) {
    
    int segment_w = x_end - x_start + 1;
    int segment_h = y_end - y_start + 1;

    for (int i = 0; i < GRID_SIZE; i++) { 
        for (int j = 0; j < GRID_SIZE; j++) { 
            
            int src_y = y_start + (int)round((double)i / (GRID_SIZE - 1) * (segment_h - 1));
            int src_x = x_start + (int)round((double)j / (GRID_SIZE - 1) * (segment_w - 1));

            src_y = fmax(0, fmin(full_height - 1, src_y));
            src_x = fmax(0, fmin(full_width - 1, src_x));

            segment_out[i][j] = full_data[src_y * full_width + src_x];
        }
    }
}


// --- Segmentation Logic: Boolean Array Interval Method ---

int find_content_intervals_bool(const unsigned char *bool_array, int size, int min_length, Boundary *boundaries_out) {
    int final_count = 0;
    int in_content = 0;
    int content_start = 0;

    for (int i = 0; i < size; i++) {
        if (bool_array[i]) { 
            if (!in_content) {
                content_start = i;
                in_content = 1;
            }
        } else { 
            if (in_content) {
                int content_length = i - content_start;
                if (content_length >= min_length) {
                    boundaries_out[final_count].start = content_start;
                    boundaries_out[final_count].end = i - 1;
                    if (final_count < MAX_SEGMENTS) final_count++;
                }
                in_content = 0;
            }
        }
    }
    
    if (in_content) {
        if (size - content_start >= min_length) {
            boundaries_out[final_count].start = content_start;
            boundaries_out[final_count].end = size - 1;
            if (final_count < MAX_SEGMENTS) final_count++;
        }
    }

    return final_count;
}

int segment_image_naive(const double *full_data, int full_width, int full_height, SegmentResult *segments_out) {
    int total_segments = 0;
    const double threshold = SEGMENTATION_THRESHOLD;
    
    // 1. Line Segmentation (Y-axis intervals)
    unsigned char *y_bool_array = (unsigned char *)safe_malloc(sizeof(unsigned char) * full_height);
    Boundary line_boundaries[MAX_SEGMENTS];

    for (int i = 0; i < full_height; i++) {
        y_bool_array[i] = 0; 
        for (int j = 0; j < full_width; j++) {
            if (full_data[i * full_width + j] > threshold) {
                y_bool_array[i] = 1; 
                break;
            }
        }
    }
    
    int min_line_height = full_height / 15;
    if (min_line_height < 10) min_line_height = 10;
    
    int num_lines = find_content_intervals_bool(y_bool_array, full_height, min_line_height, line_boundaries);
    safe_free(y_bool_array, sizeof(unsigned char) * full_height);
    
    printf("Segmentation: Found %d line(s).\n", num_lines);

    // 2. Letter Segmentation (X-axis intervals) within each line
    for (int l = 0; l < num_lines; l++) {
        int line_y_start = line_boundaries[l].start;
        int line_y_end = line_boundaries[l].end;
        int line_height = line_y_end - line_y_start + 1;
        
        unsigned char *x_bool_array = (unsigned char *)safe_malloc(sizeof(unsigned char) * full_width);

        for (int j = 0; j < full_width; j++) {
            x_bool_array[j] = 0; 
            for (int i = line_y_start; i <= line_y_end; i++) {
                if (full_data[i * full_width + j] > threshold) {
                    x_bool_array[j] = 1; 
                    break;
                }
            }
        }
        
        Boundary letter_boundaries[MAX_SEGMENTS];
        
        int min_char_width = line_height / 5;
        if (min_char_width < 5) min_char_width = 5; 
        
        int num_letters = find_content_intervals_bool(x_bool_array, full_width, min_char_width, letter_boundaries);
        safe_free(x_bool_array, sizeof(unsigned char) * full_width);
        
        printf("  Line %d (Y:[%d,%d]): Found %d letter(s).\n", l + 1, line_y_start, line_y_end, num_letters);

        // 3. Process each letter segment
        for (int c = 0; c < num_letters; c++) {
            if (total_segments >= MAX_SEGMENTS) {
                printf("  WARNING: Reached maximum segment limit (%d).\n", MAX_SEGMENTS);
                return total_segments;
            }
            
            SegmentResult *seg = &segments_out[total_segments];
            seg->y_start = line_y_start; 
            seg->y_end = line_y_end;
            seg->x_start = letter_boundaries[c].start;
            seg->x_end = letter_boundaries[c].end;
            
            resize_segment(full_data, full_width, full_height, 
                           seg->x_start, seg->x_end, seg->y_start, seg->y_end, 
                           seg->resized_img);
                           
            total_segments++;
        }
    }
    
    return total_segments;
}


// --- PNG Rendering for Segmentation Output ---

#define SEG_ROW_HEIGHT (IMG_SIZE + TEXT_HEIGHT + SET_SPACING) 
// The actual PNG width is calculated in generate_segment_png dynamically
// #define SEG_PNG_WIDTH (IMG_SIZE * 2 + IMG_SPACING * 3 + SET_SPACING * 2) 

void draw_segment_info_box(unsigned char *buffer, int buf_width, int buf_height, int x, int y, int width, int height, const SegmentResult *seg) {
    char info[100];
    const char* char_name = (seg->best_match_index != -1) ? CHAR_NAMES[seg->best_match_index] : "N/A";
    
    // Include stroke width in the info box
    sprintf(info, "Match: '%s' | Loss: %.2f | SW:%d | a1:%.2f", char_name, seg->final_loss, seg->best_stroke_width, seg->estimated_alpha[0]);
    draw_text_placeholder_box(buffer, buf_width, buf_height, x, y + 2, width, TEXT_HEIGHT - 4, 200, 200, 255);
}

void render_segment_to_png(unsigned char *buffer, int buf_width, int buf_height, const SegmentResult *seg, int x_set, int y_set) {
    int current_x = x_set;
    
    draw_segment_info_box(buffer, buf_width, buf_height, x_set, y_set, IMG_SIZE * 2 + IMG_SPACING, TEXT_HEIGHT, seg);
    
    // 1. Draw Observed Segment
    render_single_image_to_png(buffer, buf_width, buf_height, seg->resized_img, current_x, y_set + TEXT_HEIGHT, 0); 
    current_x += IMG_SIZE + IMG_SPACING;
    
    // 2. Draw Estimated Curve (using the best determined stroke width)
    Generated_Image estimated_img;
    if (seg->best_match_index != -1) {
        draw_curve(seg->estimated_alpha, estimated_img, &IDEAL_TEMPLATES[seg->best_match_index], seg->best_stroke_width);
    } else {
        for(int i=0; i<GRID_SIZE; i++) for(int j=0; j<GRID_SIZE; j++) estimated_img[i][j] = 0.0;
    }
    render_single_image_to_png(buffer, buf_width, buf_height, estimated_img, current_x, y_set + TEXT_HEIGHT, 0);
}

void generate_segment_png(const SegmentResult *segments, int num_segments, const double *full_data, int full_width, int full_height) {
    if (num_segments == 0) {
        printf("\nWARNING: No segments found. Skipping segment PNG generation.\n");
        return;
    }
    
    // **Improvement 1: Dynamic Width Calculation**
    const int SEGMENT_SET_WIDTH = (IMG_SIZE * 2 + IMG_SPACING); // Width of a single observed + generated pair
    const int MAX_SEGMENTS_PER_ROW = 10; // Use the updated max per row
    const int SEGMENTS_PER_ROW = fmin(MAX_SEGMENTS_PER_ROW, num_segments);
    
    int full_img_row_height = (full_height * PIXEL_SIZE) + TEXT_HEIGHT + SET_SPACING;
    int seg_row_height = SEG_ROW_HEIGHT;
    int num_seg_rows = (num_segments + SEGMENTS_PER_ROW - 1) / SEGMENTS_PER_ROW; 
    
    // Calculate required width based on MAX_SEGMENTS_PER_ROW
    int max_seg_row_width = SET_SPACING * 2 + SEGMENTS_PER_ROW * SEGMENT_SET_WIDTH + (SEGMENTS_PER_ROW - 1) * IMG_SPACING;
    int full_img_width = full_width * PIXEL_SIZE + SET_SPACING * 2;
    
    int png_width = fmax(max_seg_row_width, full_img_width); // Ensure width is enough for the widest row/full image
    int png_height = full_img_row_height + num_seg_rows * seg_row_height + SET_SPACING;
    
    long buffer_size = (long)png_width * png_height * NUM_CHANNELS;
    unsigned char *buffer = (unsigned char *)safe_malloc(buffer_size);
    if (buffer == NULL) {
        printf("\nERROR: Failed to allocate buffer for segment PNG.\n");
        return;
    }
    memset(buffer, 0, buffer_size);

    int x_set = SET_SPACING;
    int y_set = SET_SPACING;

    // 1. Draw Full Image
    draw_text_placeholder_box(buffer, png_width, png_height, x_set, y_set + 2, full_width * PIXEL_SIZE, TEXT_HEIGHT - 4, 150, 150, 255);
    
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
    
    // 2. Draw Segment Boundaries on the Full Image (Blue Outline)
    int y_offset = y_set + TEXT_HEIGHT; 
    for (int k = 0; k < num_segments; k++) {
        const SegmentResult *seg = &segments[k];
        
        // Vertical lines
        for (int y = seg->y_start * PIXEL_SIZE; y <= seg->y_end * PIXEL_SIZE + PIXEL_SIZE; y++) {
            set_pixel(buffer, x_set + seg->x_start * PIXEL_SIZE, y_offset + y, png_width, png_height, 0, 0, 255);
            set_pixel(buffer, x_set + seg->x_end * PIXEL_SIZE, y_offset + y, png_width, png_height, 0, 0, 255);
        }
        // Horizontal lines
        for (int x = seg->x_start * PIXEL_SIZE; x <= seg->x_end * PIXEL_SIZE + PIXEL_SIZE; x++) {
            set_pixel(buffer, x_set + x, y_offset + seg->y_start * PIXEL_SIZE, png_width, png_height, 0, 0, 255);
            set_pixel(buffer, x_set + x, y_offset + seg->y_end * PIXEL_SIZE, png_width, png_height, 0, 0, 255);
        }
    }
    
    y_set += full_img_row_height;

    // 3. Draw Letter Segments
    for (int k = 0; k < num_segments; k++) {
        int row_index = k / MAX_SEGMENTS_PER_ROW;
        int col_index = k % MAX_SEGMENTS_PER_ROW;
        
        int seg_x = SET_SPACING + col_index * (SEGMENT_SET_WIDTH + IMG_SPACING);
        int seg_y = y_set + row_index * seg_row_height;
        
        render_segment_to_png(buffer, png_width, png_height, &segments[k], seg_x, seg_y);
    }

    int success = stbi_write_png("segmentation_output.png", png_width, png_height, NUM_CHANNELS, buffer, png_width * NUM_CHANNELS);

    safe_free(buffer, buffer_size);

    if (success) {
        printf("\nSegmentation Output Complete: segmentation_output.png created.\n");
    } else {
        printf("\nERROR: Failed to write segmentation_output.png.\n");
    }
}


// --- Main Execution ---

int main(void) {
    srand(42); 

    // Using the image from the context
    // NOTE: The name 'test1.jpg' is used for the previous placeholder. 
    // The name '1000000825.png' from the upload is used here directly.
    const char *input_filename = "test1.png"; 
    double *full_image_data = NULL;
    int full_width = 0;
    int full_height = 0;
    
    // Start timing
    clock_t start_time = clock();

    if (!load_image_stb(input_filename, &full_image_data, &full_width, &full_height) || full_image_data == NULL) {
        fprintf(stderr, "Fatal Error: Image processing aborted.\n");
        return 1;
    }
    
    size_t data_size = (size_t)full_width * full_height * sizeof(double);
    size_t segments_size = sizeof(SegmentResult) * MAX_SEGMENTS;
    SegmentResult *segments = (SegmentResult *)safe_malloc(segments_size);
    if (segments == NULL) {
        fprintf(stderr, "Fatal Error: Failed to allocate memory for segments.\n");
        safe_free(full_image_data, data_size);
        return 1;
    }
    
    // 1. Segment Image using the Boolean Array method
    int num_segments = segment_image_naive(full_image_data, full_width, full_height, segments);

    // 2. Recognize Each Segment
    printf("\nStarting recognition for %d segments (testing stroke widths %d, %d, %d, %d) with %d iterations per match...\n", 
           num_segments, STROKE_WIDTHS[0], STROKE_WIDTHS[1], STROKE_WIDTHS[2], STROKE_WIDTHS[3], ITERATIONS);
           
    for (int i = 0; i < num_segments; i++) {
        recognize_segment(&segments[i]);
        const char* char_name = (segments[i].best_match_index != -1) ? CHAR_NAMES[segments[i].best_match_index] : "N/A";
        printf("  Segment %d: Match='%s' (Loss: %.4f, SW: %d, a1: %.2f)\n", 
               i + 1, char_name, segments[i].final_loss, segments[i].best_stroke_width, segments[i].estimated_alpha[0]);
    }
    
    // 3. Generate PNG Output
    generate_segment_png(segments, num_segments, full_image_data, full_width, full_height);

    // Stop timing
    clock_t end_time = clock();
    double time_spent = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    // Free resources
    safe_free(segments, segments_size);
    safe_free(full_image_data, data_size);

    printf("\n--- Performance Summary ---\n");
    printf("Total Time Spent: %.2f seconds (Target: 180 seconds)\n", time_spent);
    printf("Final Memory Check: Allocated: %zu bytes | Freed: %zu bytes | Net: %zu bytes\n", 
           total_allocated_bytes, total_freed_bytes, total_allocated_bytes - total_freed_bytes);

    return 0;
}
