#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h> 

// --- STB Image Write Configuration ---
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Define M_PI explicitly
#define M_PI 3.14159265358979323846

// --- Global Configuration ---
#define GRID_SIZE 32        // **UPDATED: Increased to 32x32 resolution**
#define NUM_DEFORMATIONS 2  
#define NUM_VECTORS 16      
#define NUM_BINS 32         
#define NUM_FEATURES (NUM_VECTORS * NUM_BINS) 
#define PIXEL_LOSS_WEIGHT 2.5   // **UPDATED: Reduced from 5.0 to 2.5 to compensate for 4x pixel count**
#define NUM_POINTS 200
#define ITERATIONS 500      // Set for speed
#define GRADIENT_EPSILON 0.01 
#define NUM_IDEAL_CHARS 36  
#define TESTS_PER_CHAR 8    
#define NUM_TESTS (NUM_IDEAL_CHARS * TESTS_PER_CHAR) // 36 * 8 = 288 total tests
#define NUM_CONTROL_POINTS 9 
#define MAX_PIXEL_ERROR (GRID_SIZE * GRID_SIZE) // **UPDATED: 32*32 = 1024**
#define TIME_LIMIT_SECONDS 240.0 // 4 minutes limit

// Loss history configuration
#define LOSS_HISTORY_STEP 5 
#define LOSS_HISTORY_SIZE (ITERATIONS / LOSS_HISTORY_STEP + 1) 

// --- Data Structures ---

typedef struct { double x; double y; } Point;
typedef struct { const Point control_points[NUM_CONTROL_POINTS]; } Ideal_Curve_Params; 
typedef struct { double alpha[NUM_DEFORMATIONS]; } Deformation_Coefficients;
typedef double Generated_Image[GRID_SIZE][GRID_SIZE]; 
typedef double Feature_Vector[NUM_FEATURES]; 
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

TestResult all_results[NUM_TESTS]; 
int tests_completed_before_limit = 0; 

// --- Fixed Ideal Curves (A-Z, 0-9) ---
const char *CHAR_NAMES[NUM_IDEAL_CHARS] = {
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", 
    "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", 
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
};

// Templates are included here for completeness but are very long.
// (The full template array from the previous version remains here)
const Ideal_Curve_Params IDEAL_TEMPLATES[NUM_IDEAL_CHARS] = {
    [0] = {.control_points = {{.x = 0.5, .y = 0.1}, {.x = 0.3, .y = 0.3}, {.x = 0.2, .y = 0.5}, {.x = 0.3, .y = 0.6}, 
        {.x = 0.7, .y = 0.6}, {.x = 0.8, .y = 0.5}, {.x = 0.7, .y = 0.3}, {.x = 0.5, .y = 0.1}, 
        {.x = 0.7, .y = 0.9} 
    }},
    [1] = {.control_points = {{.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.9}, {.x = 0.2, .y = 0.1}, 
        {.x = 0.8, .y = 0.1}, {.x = 0.8, .y = 0.3}, {.x = 0.2, .y = 0.5}, 
        {.x = 0.8, .y = 0.5}, {.x = 0.8, .y = 0.7}, {.x = 0.2, .y = 0.9} 
    }},
    [2] = {.control_points = {{.x = 0.8, .y = 0.1}, {.x = 0.4, .y = 0.1}, {.x = 0.1, .y = 0.3}, {.x = 0.1, .y = 0.7},
        {.x = 0.4, .y = 0.9}, {.x = 0.8, .y = 0.9}, {.x = 0.8, .y = 0.7}, {.x = 0.8, .y = 0.3}, 
        {.x = 0.1, .y = 0.5} 
    }},
    [3] = {.control_points = {{.x = 0.2, .y = 0.1}, {.x = 0.7, .y = 0.1}, {.x = 0.8, .y = 0.3}, {.x = 0.8, .y = 0.5}, 
        {.x = 0.8, .y = 0.7}, {.x = 0.7, .y = 0.9}, {.x = 0.2, .y = 0.9}, {.x = 0.2, .y = 0.5}, 
        {.x = 0.2, .y = 0.1} 
    }},
    [4] = {.control_points = {{.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.1}, {.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.5}, 
        {.x = 0.75, .y = 0.5}, {.x = 0.2, .y = 0.5}, {.x = 0.2, .y = 0.9}, {.x = 0.8, .y = 0.9}, 
        {.x = 0.2, .y = 0.9} 
    }},
    [5] = {.control_points = {{.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.1}, {.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.5}, 
        {.x = 0.6, .y = 0.5}, {.x = 0.2, .y = 0.5}, {.x = 0.2, .y = 0.9}, {.x = 0.2, .y = 0.9}, 
        {.x = 0.2, .y = 0.1} 
    }},
    [6] = {.control_points = {{.x = 0.8, .y = 0.1}, {.x = 0.2, .y = 0.1}, {.x = 0.1, .y = 0.5}, {.x = 0.2, .y = 0.9}, 
        {.x = 0.8, .y = 0.9}, {.x = 0.8, .y = 0.6}, {.x = 0.4, .y = 0.6}, {.x = 0.4, .y = 0.9}, 
        {.x = 0.8, .y = 0.9} 
    }},
    [7] = {.control_points = {{.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.9}, {.x = 0.2, .y = 0.5}, {.x = 0.8, .y = 0.5}, 
        {.x = 0.8, .y = 0.1}, {.x = 0.8, .y = 0.9}, {.x = 0.2, .y = 0.5}, {.x = 0.8, .y = 0.5}, 
        {.x = 0.8, .y = 0.5} 
    }},
    [8] = {.control_points = {{.x = 0.3, .y = 0.1}, {.x = 0.7, .y = 0.1}, {.x = 0.5, .y = 0.1}, {.x = 0.5, .y = 0.9}, 
        {.x = 0.3, .y = 0.9}, {.x = 0.7, .y = 0.9}, {.x = 0.5, .y = 0.5}, {.x = 0.5, .y = 0.1}, 
        {.x = 0.5, .y = 0.9} 
    }},
    [9] = {.control_points = {{.x = 0.6, .y = 0.1}, {.x = 0.6, .y = 0.2}, {.x = 0.6, .y = 0.4}, {.x = 0.6, .y = 0.5}, 
        {.x = 0.5, .y = 0.6}, {.x = 0.4, .y = 0.75}, {.x = 0.3, .y = 0.9}, {.x = 0.4, .y = 0.85}, 
        {.x = 0.5, .y = 0.8}  
    }},
    [10] = {.control_points = {{.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.9}, {.x = 0.2, .y = 0.5}, {.x = 0.8, .y = 0.1}, 
        {.x = 0.2, .y = 0.5}, {.x = 0.8, .y = 0.9}, {.x = 0.2, .y = 0.5}, {.x = 0.2, .y = 0.5}, 
        {.x = 0.2, .y = 0.5} 
    }},
    [11] = {.control_points = {{.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.9}, {.x = 0.8, .y = 0.9}, {.x = 0.2, .y = 0.9}, 
        {.x = 0.2, .y = 0.6}, {.x = 0.2, .y = 0.3}, {.x = 0.2, .y = 0.1}, {.x = 0.5, .y = 0.5}, 
        {.x = 0.2, .y = 0.9} 
    }},
    [12] = {.control_points = {{.x = 0.1, .y = 0.9}, {.x = 0.1, .y = 0.1}, {.x = 0.5, .y = 0.8}, 
        {.x = 0.9, .y = 0.1}, {.x = 0.9, .y = 0.9}, {.x = 0.5, .y = 0.8}, {.x = 0.1, .y = 0.1}, 
        {.x = 0.9, .y = 0.9}, {.x = 0.5, .y = 0.3} 
    }},
    [13] = {.control_points = {{.x = 0.2, .y = 0.9}, {.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.9}, 
        {.x = 0.8, .y = 0.9}, {.x = 0.8, .y = 0.1}, {.x = 0.5, .y = 0.5}, {.x = 0.2, .y = 0.9}, 
        {.x = 0.8, .y = 0.1} 
    }},
    [14] = {.control_points = {{.x = 0.5, .y = 0.1}, {.x = 0.85, .y = 0.3}, {.x = 0.85, .y = 0.7}, 
        {.x = 0.5, .y = 0.9}, {.x = 0.15, .y = 0.7}, {.x = 0.15, .y = 0.3}, 
        {.x = 0.5, .y = 0.1}, {.x = 0.5, .y = 0.5}, {.x = 0.5, .y = 0.5} 
    }},
    [15] = {.control_points = {{.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.9}, {.x = 0.2, .y = 0.1}, 
        {.x = 0.7, .y = 0.1}, {.x = 0.8, .y = 0.2}, {.x = 0.8, .y = 0.3}, 
        {.x = 0.7, .y = 0.4}, {.x = 0.2, .y = 0.4}, {.x = 0.2, .y = 0.4} 
    }},
    [16] = {.control_points = {{.x = 0.5, .y = 0.1}, {.x = 0.8, .y = 0.3}, {.x = 0.8, .y = 0.7}, {.x = 0.5, .y = 0.9}, 
        {.x = 0.2, .y = 0.7}, {.x = 0.2, .y = 0.3}, {.x = 0.5, .y = 0.1}, {.x = 0.6, .y = 0.7}, 
        {.x = 0.8, .y = 0.9} 
    }},
    [17] = {.control_points = {{.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.9}, {.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.1}, 
        {.x = 0.8, .y = 0.4}, {.x = 0.2, .y = 0.5}, {.x = 0.2, .y = 0.5}, {.x = 0.9, .y = 0.9}, 
        {.x = 0.2, .y = 0.1} 
    }},
    [18] = {.control_points = {{.x = 0.8, .y = 0.1}, {.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.3}, {.x = 0.8, .y = 0.5}, 
        {.x = 0.8, .y = 0.7}, {.x = 0.2, .y = 0.9}, {.x = 0.8, .y = 0.9}, {.x = 0.5, .y = 0.5}, 
        {.x = 0.8, .y = 0.1} 
    }},
    [19] = {.control_points = {{.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.1}, {.x = 0.5, .y = 0.1}, {.x = 0.5, .y = 0.9}, 
        {.x = 0.5, .y = 0.5}, {.x = 0.8, .y = 0.1}, {.x = 0.2, .y = 0.1}, {.x = 0.5, .y = 0.9}, 
        {.x = 0.5, .y = 0.1} 
    }},
    [20] = {.control_points = {{.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.7}, {.x = 0.5, .y = 0.9}, {.x = 0.8, .y = 0.7}, 
        {.x = 0.8, .y = 0.1}, {.x = 0.5, .y = 0.9}, {.x = 0.2, .y = 0.7}, {.x = 0.8, .y = 0.7}, 
        {.x = 0.2, .y = 0.1} 
    }},
    [21] = {.control_points = {{.x = 0.2, .y = 0.1}, {.x = 0.5, .y = 0.9}, {.x = 0.8, .y = 0.1}, {.x = 0.5, .y = 0.9}, 
        {.x = 0.2, .y = 0.5}, {.x = 0.8, .y = 0.5}, {.x = 0.5, .y = 0.9}, {.x = 0.2, .y = 0.1}, 
        {.x = 0.8, .y = 0.1} 
    }},
    [22] = {.control_points = {{.x = 0.1, .y = 0.1}, {.x = 0.3, .y = 0.9}, {.x = 0.5, .y = 0.2}, 
        {.x = 0.7, .y = 0.9}, {.x = 0.9, .y = 0.1}, {.x = 0.5, .y = 0.5}, {.x = 0.5, .y = 0.5}, 
        {.x = 0.5, .y = 0.5}, {.x = 0.5, .y = 0.5}
    }},
    [23] = {.control_points = {{.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.9}, {.x = 0.5, .y = 0.5}, {.x = 0.8, .y = 0.1}, 
        {.x = 0.2, .y = 0.9}, {.x = 0.5, .y = 0.5}, {.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.9}, 
        {.x = 0.5, .y = 0.5} 
    }},
    [24] = {.control_points = {{.x = 0.2, .y = 0.1}, {.x = 0.5, .y = 0.5}, {.x = 0.8, .y = 0.1}, {.x = 0.5, .y = 0.5}, 
        {.x = 0.5, .y = 0.9}, {.x = 0.5, .y = 0.7}, {.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.1}, 
        {.x = 0.5, .y = 0.9} 
    }},
    [25] = {.control_points = {{.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.1}, {.x = 0.2, .y = 0.9}, {.x = 0.8, .y = 0.9}, 
        {.x = 0.5, .y = 0.5}, {.x = 0.8, .y = 0.1}, {.x = 0.2, .y = 0.9}, {.x = 0.5, .y = 0.5}, 
        {.x = 0.2, .y = 0.1} 
    }},
    [26] = {.control_points = {{.x = 0.5, .y = 0.1}, {.x = 0.75, .y = 0.3}, {.x = 0.75, .y = 0.7}, 
        {.x = 0.5, .y = 0.9}, {.x = 0.25, .y = 0.7}, {.x = 0.25, .y = 0.3}, 
        {.x = 0.5, .y = 0.1}, {.x = 0.5, .y = 0.5}, {.x = 0.5, .y = 0.1} 
    }},
    [27] = {.control_points = {{.x = 0.4, .y = 0.1}, {.x = 0.5, .y = 0.1}, {.x = 0.5, .y = 0.2}, {.x = 0.5, .y = 0.35}, 
        {.x = 0.5, .y = 0.5}, {.x = 0.5, .y = 0.65}, {.x = 0.5, .y = 0.8}, {.x = 0.5, .y = 0.9}, 
        {.x = 0.5, .y = 0.9} 
    }},
    [28] = {.control_points = {{.x = 0.2, .y = 0.1}, {.x = 0.7, .y = 0.1}, {.x = 0.8, .y = 0.25}, {.x = 0.7, .y = 0.4}, 
        {.x = 0.3, .y = 0.55}, {.x = 0.2, .y = 0.7}, {.x = 0.3, .y = 0.8}, {.x = 0.5, .y = 0.9}, 
        {.x = 0.8, .y = 0.9} 
    }}, 
    [29] = {.control_points = {{.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.1}, {.x = 0.8, .y = 0.3}, {.x = 0.4, .y = 0.5}, 
        {.x = 0.8, .y = 0.5}, {.x = 0.8, .y = 0.7}, {.x = 0.4, .y = 0.9}, {.x = 0.8, .y = 0.9}, 
        {.x = 0.2, .y = 0.9} 
    }}, 
    [30] = {.control_points = {{.x = 0.2, .y = 0.1}, {.x = 0.3, .y = 0.2}, {.x = 0.4, .y = 0.3}, {.x = 0.5, .y = 0.4}, 
        {.x = 0.2, .y = 0.55}, {.x = 0.8, .y = 0.55}, {.x = 0.8, .y = 0.7}, {.x = 0.8, .y = 0.85}, 
        {.x = 0.8, .y = 0.9} 
    }}, 
    [31] = {.control_points = {{.x = 0.8, .y = 0.1}, {.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.4}, {.x = 0.6, .y = 0.4}, 
        {.x = 0.8, .y = 0.6}, {.x = 0.7, .y = 0.8}, {.x = 0.2, .y = 0.9}, {.x = 0.8, .y = 0.9}, 
        {.x = 0.8, .y = 0.1} 
    }},
    [32] = {.control_points = {{.x = 0.7, .y = 0.1}, {.x = 0.2, .y = 0.3}, {.x = 0.2, .y = 0.5}, {.x = 0.5, .y = 0.7}, 
        {.x = 0.8, .y = 0.6}, {.x = 0.5, .y = 0.5}, {.x = 0.2, .y = 0.5}, {.x = 0.2, .y = 0.9}, 
        {.x = 0.7, .y = 0.1} 
    }},
    [33] = {.control_points = {{.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.1}, {.x = 0.8, .y = 0.1}, {.x = 0.3, .y = 0.9}, 
        {.x = 0.5, .y = 0.5}, {.x = 0.6, .y = 0.7}, {.x = 0.8, .y = 0.1}, {.x = 0.2, .y = 0.9}, 
        {.x = 0.8, .y = 0.1} 
    }},
    [34] = {.control_points = {{.x = 0.5, .y = 0.15}, {.x = 0.7, .y = 0.2}, {.x = 0.7, .y = 0.35}, 
        {.x = 0.5, .y = 0.45}, {.x = 0.3, .y = 0.35}, {.x = 0.3, .y = 0.2}, 
        {.x = 0.7, .y = 0.65}, {.x = 0.5, .y = 0.85}, {.x = 0.3, .y = 0.65} 
    }},
    [35] = {.control_points = {{.x = 0.5, .y = 0.1}, {.x = 0.8, .y = 0.2}, {.x = 0.5, .y = 0.4}, {.x = 0.2, .y = 0.2}, 
        {.x = 0.5, .y = 0.1}, {.x = 0.8, .y = 0.4}, {.x = 0.8, .y = 0.9}, {.x = 0.5, .y = 0.7}, 
        {.x = 0.8, .y = 0.9} 
    }}
};


// --- Function Definitions (Optimization Logic) ---

void apply_deformation(Point *point, const double alpha[NUM_DEFORMATIONS]) {
    point->x = point->x + alpha[0] * (point->y - 0.5);
    point->x = point->x + alpha[1] * sin(M_PI * point->y);
}

Point get_deformed_point(const double t, const Ideal_Curve_Params *const params, const double alpha[NUM_DEFORMATIONS]) {
    Point p = {.x = 0.0, .y = 0.0};
    const int N = NUM_CONTROL_POINTS; 
    const double segment_length_t = 1.0 / (N - 1); 

    int segment_index = (int)floor(t / segment_length_t);
    if (segment_index >= N - 1) {
        segment_index = N - 2; 
    }

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

void draw_curve(const double alpha[NUM_DEFORMATIONS], Generated_Image img, const Ideal_Curve_Params *const ideal_params) {
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            img[i][j] = 0.0;
        }
    }

    for (int i = 0; i <= NUM_POINTS; i++) {
        const double t = (double)i / NUM_POINTS;
        const Point current_p = get_deformed_point(t, ideal_params, alpha);

        const int px = (int)round(current_p.x);
        const int py = (int)round(current_p.y);

        if (px >= 0 && px < GRID_SIZE && py >= 0 && py < GRID_SIZE) {
            img[py][px] = fmin(1.0, img[py][px] + 0.5); 
        }

        if (py + 1 < GRID_SIZE) img[py + 1][px] = fmin(1.0, img[py + 1][px] + 0.1);
        if (py - 1 >= 0) img[py - 1][px] = fmin(1.0, img[py - 1][px] + 0.1);
        if (px + 1 < GRID_SIZE) img[py][px + 1] = fmin(1.0, img[py][px + 1] + 0.1);
        if (px - 1 >= 0) img[py][px - 1] = fmin(1.0, img[py][px - 1] + 0.1);
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
    
    // Pixel loss weight adjusted for 32x32 resolution
    return feature_loss + PIXEL_LOSS_WEIGHT * pixel_loss;
}

void calculate_gradient(const Generated_Image observed_img, const Feature_Vector observed_features, 
                        const Deformation_Coefficients *const alpha, const double loss_base, 
                        double grad_out[NUM_DEFORMATIONS], const Ideal_Curve_Params *const ideal_params) {
    
    const double epsilon = GRADIENT_EPSILON; 
    Generated_Image generated_img_perturbed; 
    Feature_Vector generated_features_perturbed; 

    for (int k = 0; k < NUM_DEFORMATIONS; k++) {
        Deformation_Coefficients alpha_perturbed = *alpha;
        alpha_perturbed.alpha[k] += epsilon; 

        draw_curve(alpha_perturbed.alpha, generated_img_perturbed, ideal_params);
        extract_geometric_features(generated_img_perturbed, generated_features_perturbed);
        
        const double loss_perturbed = calculate_combined_loss(generated_img_perturbed, generated_features_perturbed, observed_img, observed_features);

        grad_out[k] = (loss_perturbed - loss_base) / epsilon;
    }
}

void generate_target_image(Generated_Image image_out, const double true_alpha[NUM_DEFORMATIONS], const Ideal_Curve_Params *const ideal_params, int add_noise) {
    draw_curve(true_alpha, image_out, ideal_params); 

    if (add_noise) {
        for (int i = 0; i < GRID_SIZE; i++) {
            for (int j = 0; j < GRID_SIZE; j++) {
                double noise = ((double)rand() / RAND_MAX - 0.5) * 0.3; 
                image_out[i][j] = fmax(0.0, fmin(1.0, image_out[i][j] + noise));
            }
        }
    }
}

double calculate_pixel_error_sum(const Generated_Image obs, const Generated_Image est) {
    double total_error = 0.0;
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            total_error += fabs(obs[i][j] - est[i][j]);
        }
    }
    return total_error;
}

void run_optimization(const Generated_Image observed_image, const Feature_Vector observed_features, 
                      int ideal_char_index, EstimationResult *result) {
    
    for (int i = 0; i < LOSS_HISTORY_SIZE; i++) {
        result->loss_history[i] = HUGE_VAL;
    }

    const Ideal_Curve_Params *ideal_params = &IDEAL_TEMPLATES[ideal_char_index];
    
    Deformation_Coefficients alpha_hat = { .alpha = {0.0, 0.0} };
    double learning_rate = 0.0000001; 
    const double min_learning_rate = 0.00000000005;
    double gradient[NUM_DEFORMATIONS];
    Generated_Image generated_image;
    Feature_Vector generated_features;
    double combined_loss;
    double prev_combined_loss = HUGE_VAL; 
    double current_feature_loss_only = HUGE_VAL;

    for (int t = 0; t <= ITERATIONS; t++) {
        draw_curve(alpha_hat.alpha, generated_image, ideal_params);
        extract_geometric_features(generated_image, generated_features);
        
        current_feature_loss_only = calculate_feature_loss_L2(generated_features, observed_features);
        combined_loss = current_feature_loss_only + PIXEL_LOSS_WEIGHT * calculate_pixel_loss_L2(generated_image, observed_image);

        if (t % LOSS_HISTORY_STEP == 0 && (t / LOSS_HISTORY_STEP) < LOSS_HISTORY_SIZE) {
            result->loss_history[t / LOSS_HISTORY_STEP] = current_feature_loss_only;
        }

        if (combined_loss > prev_combined_loss * 1.001 && learning_rate > min_learning_rate) { 
            learning_rate *= 0.5;
        }

        prev_combined_loss = combined_loss;

        if (t < ITERATIONS) {
            calculate_gradient(observed_image, observed_features, &alpha_hat, combined_loss, gradient, ideal_params);
            
            double step_rate = (learning_rate > min_learning_rate) ? learning_rate : min_learning_rate;
            
            alpha_hat.alpha[0] -= step_rate * gradient[0];
            alpha_hat.alpha[1] -= step_rate * gradient[1];
        }
    }

    result->final_loss = current_feature_loss_only;
    memcpy(result->estimated_alpha, alpha_hat.alpha, sizeof(double) * NUM_DEFORMATIONS);
}

void calculate_difference_image(const Generated_Image obs, const Generated_Image est, Generated_Image diff) {
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            diff[i][j] = fabs(obs[i][j] - est[i][j]);
        }
    }
}

void run_classification_test(int test_id, int true_char_index, const double true_alpha[NUM_DEFORMATIONS], TestResult *result) {
    result->id = test_id;
    result->true_char_index = true_char_index;
    memcpy(result->true_alpha, true_alpha, sizeof(double) * NUM_DEFORMATIONS);
    const Ideal_Curve_Params *true_params = &IDEAL_TEMPLATES[true_char_index];
    
    // 1. Generate clean true image
    generate_target_image(result->true_image, true_alpha, true_params, 0); 
    
    // 2. Generate noisy observed image
    Generated_Image observed_image; 
    generate_target_image(observed_image, true_alpha, true_params, 1);
    memcpy(result->observed_image, observed_image, sizeof(Generated_Image));
    
    // 3. Extract features from observed image
    Feature_Vector observed_features;
    extract_geometric_features(observed_image, observed_features);
    
    double min_feature_loss = HUGE_VAL;
    int best_match_index = -1;
    
    // 4. Run optimization against ALL ideal character templates
    for (int i = 0; i < NUM_IDEAL_CHARS; i++) {
        run_optimization(observed_image, observed_features, i, &result->classification_results[i]);

        if (result->classification_results[i].final_loss < min_feature_loss) {
            min_feature_loss = result->classification_results[i].final_loss;
            best_match_index = i;
        }
    }

    result->best_match_index = best_match_index;

    // 5. Generate the best-fit image and difference image
    EstimationResult *best_fit = &result->classification_results[best_match_index];
    
    draw_curve(best_fit->estimated_alpha, result->best_estimated_image, &IDEAL_TEMPLATES[best_match_index]);
    calculate_difference_image(result->observed_image, result->best_estimated_image, result->best_diff_image);
}


void summarize_results_console(double total_elapsed_time) {
    printf("\n\n=================================================================================================\n");
    printf("                  CLASSIFICATION SUMMARY (%d CHARS * %d TESTS = %d TOTAL TESTS)                    \n", NUM_IDEAL_CHARS, TESTS_PER_CHAR, NUM_TESTS);
    printf("                                  Tests Completed: %d/%d in %.2f seconds\n", tests_completed_before_limit, NUM_TESTS, total_elapsed_time);
    if (tests_completed_before_limit < NUM_TESTS) {
        printf("                                    !!! TIME LIMIT OF %.0f SECONDS REACHED !!!\n", TIME_LIMIT_SECONDS);
    }
    printf("=================================================================================================\n");
    
    printf("  ID | TRUE | PRED | Feat Loss (Best Fit) | TRUE $a_1$ | TRUE $a_2$ | EST $a_1$ | EST $a_2$ | PIXEL ERROR %% | Result \n");
    printf("-----|------|------|----------------------|-----------|-----------|-----------|-----------|---------------|--------\n");

    int correct_classifications = 0;
    for (int k = 0; k < tests_completed_before_limit; k++) {
        TestResult *r = &all_results[k];
        const EstimationResult *best_fit = &r->classification_results[r->best_match_index];
        
        double pixel_error_sum = calculate_pixel_error_sum(r->observed_image, r->best_estimated_image);
        double pixel_error_percent = (pixel_error_sum / MAX_PIXEL_ERROR) * 100.0;

        int is_correct = (r->true_char_index == r->best_match_index);
        if (is_correct) correct_classifications++;

        printf("%4d | %4s | %4s | %20.4f | %9.4f | %9.4f | %9.4f | %9.4f | %13.2f | %6s\n", 
               r->id, CHAR_NAMES[r->true_char_index], CHAR_NAMES[r->best_match_index],
               best_fit->final_loss, 
               r->true_alpha[0], r->true_alpha[1], 
               best_fit->estimated_alpha[0], best_fit->estimated_alpha[1], 
               pixel_error_percent, is_correct ? "CORRECT" : "WRONG");
    }
    printf("-----|------|------|----------------------|-----------|-----------|-----------|-----------|---------------|--------\n");
    printf("Overall Accuracy (Completed Tests): %d/%d (%.2f%%)\n", correct_classifications, tests_completed_before_limit, 
           (double)correct_classifications / tests_completed_before_limit * 100.0);
    printf("=================================================================================================\n");
}


// --- PNG Rendering Constants (Vertical Layout) ---
#define PIXEL_SIZE 2    // **UPDATED: Reduced for manageable 32x32 image size**
#define IMG_SIZE (GRID_SIZE * PIXEL_SIZE) 
#define IMG_SPACING 5   
#define TEXT_HEIGHT 15  
#define SET_SPACING 25  
#define GRAPH_WIDTH 100 
#define GRAPH_HEIGHT IMG_SIZE 

// PNG Dimensions (HEIGHT dynamically calculated based on completed tests)
#define PNG_WIDTH (IMG_SIZE * 4 + IMG_SPACING * 3 + GRAPH_WIDTH + SET_SPACING * 2) 
#define NUM_CHANNELS 3 

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

void draw_text_placeholder(unsigned char *buffer, int buf_width, int buf_height, int x, int y, const char* text, unsigned char r, unsigned char g, unsigned char b) {
    // This function provides a minimal visual representation for the text label
    // by drawing a colored line segment.
    int line_len = 20; 
    for(int i = 0; i < line_len; i++) {
        set_pixel(buffer, x + i, y, buf_width, buf_height, r, g, b);
    }
}

void draw_loss_graph(unsigned char *buffer, int buf_width, int buf_height, int x_offset, int y_offset, const EstimationResult *result) {
    
    double max_loss = 0.0;
    for (int i = 0; i < LOSS_HISTORY_SIZE; i++) {
        if (result->loss_history[i] != HUGE_VAL && result->loss_history[i] > max_loss) {
            max_loss = result->loss_history[i];
        }
    }
    if (max_loss < 1.0) max_loss = 1.0; 

    // Draw background
    for(int py = 0; py < GRAPH_HEIGHT; py++) {
        for(int px = 0; px < GRAPH_WIDTH; px++) {
            set_pixel(buffer, x_offset + px, y_offset + py, buf_width, buf_height, 20, 20, 20); 
        }
    }
    // Draw axes
    for(int i = 0; i < GRAPH_WIDTH; i++) set_pixel(buffer, x_offset + i, y_offset + GRAPH_HEIGHT - 1, buf_width, buf_height, 255, 255, 255);
    for(int i = 0; i < GRAPH_HEIGHT; i++) set_pixel(buffer, x_offset, y_offset + i, buf_width, buf_height, 255, 255, 255);

    int prev_y = -1;
    for (int i = 0; i < LOSS_HISTORY_SIZE; i++) {
        if (result->loss_history[i] == HUGE_VAL) continue;

        int current_x = x_offset + (int)round((double)i / (LOSS_HISTORY_SIZE - 1) * (GRAPH_WIDTH - 1));

        double normalized_loss = 1.0 - (result->loss_history[i] / max_loss);
        int current_y = y_offset + (int)round(normalized_loss * (GRAPH_HEIGHT - 1));
        current_y = fmax(y_offset, fmin(y_offset + GRAPH_HEIGHT - 1, current_y)); 

        set_pixel(buffer, current_x, current_y, buf_width, buf_height, 50, 255, 50); 

        if (prev_y != -1) {
            int prev_x = x_offset + (int)round((double)(i-1) / (LOSS_HISTORY_SIZE - 1) * (GRAPH_WIDTH - 1));

            int dx = abs(current_x - prev_x);
            int dy = abs(current_y - prev_y);
            int sx = prev_x < current_x ? 1 : -1;
            int sy = prev_y < current_y ? 1 : -1;
            int err = dx - dy;
            
            int px = prev_x;
            int py = prev_y;

            // Simple line drawing (Bresenham's)
            while(1) {
                set_pixel(buffer, px, py, buf_width, buf_height, 50, 255, 50); 

                if (px == current_x && py == current_y) break;
                int e2 = 2 * err;
                if (e2 > -dy) { err -= dy; px += sx; }
                if (e2 < dx) { err += dx; py += sy; }
            }
        }
        prev_y = current_y;
    }
}

void render_test_to_png(unsigned char *buffer, int buf_width, int buf_height, const TestResult *r, int x_set, int y_set) {
    const EstimationResult *true_char_fit = &r->classification_results[r->true_char_index]; 
    const EstimationResult *best_fit = &r->classification_results[r->best_match_index]; 
    char label[180];
    double pixel_error_sum = calculate_pixel_error_sum(r->observed_image, r->best_estimated_image);
    double pixel_error_percent = (pixel_error_sum / MAX_PIXEL_ERROR) * 100.0;
    
    // Label showing True Char, Predicted Char, True Alpha, Best Fit Alpha, and Pixel Error
    sprintf(label, "ID %03d: T:'%s'(%.2f,%.2f) | P:'%s'($a_1$=%.2f,$a_2$=%.2f) | Err:%.2f%% (%s)", 
            r->id, CHAR_NAMES[r->true_char_index], r->true_alpha[0], r->true_alpha[1], 
            CHAR_NAMES[r->best_match_index], best_fit->estimated_alpha[0], best_fit->estimated_alpha[1],
            pixel_error_percent, 
            (r->true_char_index == r->best_match_index) ? "CORRECT" : "WRONG");
    
    int img_step = IMG_SIZE + IMG_SPACING;
    int current_x = x_set;
    
    // 1. TRUE CLEAN
    render_single_image_to_png(buffer, buf_width, buf_height, r->true_image, current_x, y_set + TEXT_HEIGHT, 0); 
    current_x += img_step;
    
    // 2. OBSERVED NOISY
    render_single_image_to_png(buffer, buf_width, buf_height, r->observed_image, current_x, y_set + TEXT_HEIGHT, 0);
    current_x += img_step;
    
    // 3. BEST ESTIMATED
    render_single_image_to_png(buffer, buf_width, buf_height, r->best_estimated_image, current_x, y_set + TEXT_HEIGHT, 0);
    current_x += img_step;
    
    // 4. ERROR DIFF
    render_single_image_to_png(buffer, buf_width, buf_height, r->best_diff_image, current_x, y_set + TEXT_HEIGHT, 1);
    current_x += img_step;

    // 5. Loss Graph (for the TRUE character optimization)
    draw_loss_graph(buffer, buf_width, buf_height, current_x + IMG_SPACING, y_set + TEXT_HEIGHT, true_char_fit);

    // 6. Draw Text Label placeholder
    draw_text_placeholder(buffer, buf_width, buf_height, x_set, y_set + 5, label, 255, 255, 255);
}

void generate_png_file() {
    // Only generate the PNG for the tests that were completed.
    const int PNG_DYNAMIC_HEIGHT = TEXT_HEIGHT + (IMG_SIZE + TEXT_HEIGHT + SET_SPACING) * tests_completed_before_limit;
    if (tests_completed_before_limit == 0) {
        printf("\nWARNING: No tests completed before time limit. Skipping PNG generation.\n");
        return;
    }

    long buffer_size = (long)PNG_WIDTH * PNG_DYNAMIC_HEIGHT * NUM_CHANNELS;
    unsigned char *buffer = (unsigned char *)calloc(buffer_size, 1);
    if (buffer == NULL) {
        printf("\nERROR: Failed to allocate buffer for PNG.\n");
        return;
    }

    // Draw main column titles (placeholder text only)
    int title_y = 5;
    int current_x = SET_SPACING + IMG_SIZE / 2;
    int img_step = IMG_SIZE + IMG_SPACING;
    
    draw_text_placeholder(buffer, PNG_WIDTH, PNG_DYNAMIC_HEIGHT, current_x, title_y, "TRUE CLEAN", 255, 255, 255);
    draw_text_placeholder(buffer, PNG_WIDTH, PNG_DYNAMIC_HEIGHT, current_x + img_step, title_y, "OBSERVED NOISY", 255, 255, 255);
    draw_text_placeholder(buffer, PNG_WIDTH, PNG_DYNAMIC_HEIGHT, current_x + 2 * img_step, title_y, "BEST ESTIMATED", 255, 255, 255);
    draw_text_placeholder(buffer, PNG_WIDTH, PNG_DYNAMIC_HEIGHT, current_x + 3 * img_step, title_y, "ERROR DIFF", 255, 255, 255);
    draw_text_placeholder(buffer, PNG_WIDTH, PNG_DYNAMIC_HEIGHT, current_x + 4 * img_step + 30, title_y, "FEATURE LOSS (vs ITER)", 255, 255, 255);
    
    int x_set = SET_SPACING; 
    
    // Render only the completed test sets vertically
    for (int k = 0; k < tests_completed_before_limit; k++) {
        int y_set = TEXT_HEIGHT + k * (IMG_SIZE + TEXT_HEIGHT + SET_SPACING);
        render_test_to_png(buffer, PNG_WIDTH, PNG_DYNAMIC_HEIGHT, &all_results[k], x_set, y_set);
    }

    int success = stbi_write_png("network_full_vertical.png", PNG_WIDTH, PNG_DYNAMIC_HEIGHT, NUM_CHANNELS, buffer, PNG_WIDTH * NUM_CHANNELS);

    free(buffer);

    if (success) {
        printf("\n\n======================================================\n");
        printf("PNG Output Complete: network_full_vertical.png created.\n");
        printf("Size: %d x %d pixels (Total Rows: %d).\n", PNG_WIDTH, PNG_DYNAMIC_HEIGHT, tests_completed_before_limit);
        printf("======================================================\n");
    } else {
        printf("\nERROR: Failed to write network_full_vertical.png using stb_image_write.\n");
    }
}


// --- Main Execution ---

int main(void) {
    srand(42); 

    const double MIN_ALPHA = -0.15;
    const double MAX_ALPHA = 0.15;
    
    // START TIMER
    clock_t start_time = clock();

    printf("Starting %d classification tests (%d chars * %d trials) with %d iterations each...\n", NUM_TESTS, NUM_IDEAL_CHARS, TESTS_PER_CHAR, ITERATIONS);
    printf("Image Resolution: %dx%d. Time limit set to %.0f seconds.\n", GRID_SIZE, GRID_SIZE, TIME_LIMIT_SECONDS);


    int test_counter = 0;
    
    for (int char_index = 0; char_index < NUM_IDEAL_CHARS; char_index++) {
        for (int test_run = 0; test_run < TESTS_PER_CHAR; test_run++) {
            
            // TIMER CHECK: Check time outside of the inner (iteration) loops
            double elapsed_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
            if (elapsed_time > TIME_LIMIT_SECONDS) {
                printf("\nTIME LIMIT REACHED: Stopping execution after %.2f seconds.\n", elapsed_time);
                goto end_testing; // Jump out of all loops
            }
            
            double true_alpha[NUM_DEFORMATIONS];
            
            true_alpha[0] = MIN_ALPHA + ((double)rand() / RAND_MAX) * (MAX_ALPHA - MIN_ALPHA);
            true_alpha[1] = MIN_ALPHA + ((double)rand() / RAND_MAX) * (MAX_ALPHA - MIN_ALPHA);
            
            run_classification_test(test_counter + 1, char_index, true_alpha, &all_results[test_counter]);
            
            test_counter++;
            tests_completed_before_limit = test_counter; // Update global counter
            
            if (test_counter % (NUM_IDEAL_CHARS * 2) == 0) {
                 printf("Processed %d/%d tests (Current Char: %s)...\n", test_counter, NUM_TESTS, CHAR_NAMES[char_index]);
            }
        }
    }
    
end_testing:
    // END TIMER
    double final_elapsed_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;

    summarize_results_console(final_elapsed_time);

    generate_png_file();

    return 0;
}
