#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// --- STB Image Write Configuration ---
// Define the implementation only once in one C file
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Define M_PI explicitly
#define M_PI 3.14159265358979323846

// --- Global Configuration ---
#define GRID_SIZE 16
#define NUM_DEFORMATIONS 2  // alpha_1 (Slant), alpha_2 (Curvature)
#define NUM_VECTORS 16      // Number of directional unit vectors (16 rotational features)
#define NUM_BINS 32         // Number of histogram bins per vector
#define NUM_FEATURES (NUM_VECTORS * NUM_BINS) // 16 * 32 = 512 total features
#define PIXEL_LOSS_WEIGHT 5.0 // Weighting factor lambda (Combined Loss Term)
#define NUM_POINTS 200
#define ITERATIONS 1000     // ITERATIONS SET TO 1000
#define GRADIENT_EPSILON 0.01 
#define NUM_IDEAL_CHARS 36  // A-Z and 0-9 (36 total)
#define NUM_TESTS 36        
#define NUM_CONTROL_POINTS 9 
#define MAX_PIXEL_ERROR (GRID_SIZE * GRID_SIZE) 

// --- Data Structures ---

typedef struct { double x; double y; } Point;
typedef struct { const Point control_points[NUM_CONTROL_POINTS]; } Ideal_Curve_Params; 
typedef struct { double alpha[NUM_DEFORMATIONS]; } Deformation_Coefficients;
typedef double Generated_Image[GRID_SIZE][GRID_SIZE]; 
typedef double Feature_Vector[NUM_FEATURES]; 
typedef struct { double estimated_alpha[NUM_DEFORMATIONS]; double final_loss; } EstimationResult;

// Structure to hold results for one full test case (classification results)
typedef struct {
    int id;
    int true_char_index; 
    int best_match_index; 
    double true_alpha[NUM_DEFORMATIONS];
    Generated_Image true_image; 
    Generated_Image observed_image;
    EstimationResult classification_results[NUM_IDEAL_CHARS]; 
    Generated_Image best_estimated_image;
    Generated_Image best_diff_image;
} TestResult;

// Global storage for all test results
TestResult all_results[NUM_TESTS];

// --- Fixed Ideal Curves (A-Z, 0-9) ---

// Lookup table for character names (A-Z, 0-9)
const char *CHAR_NAMES[NUM_IDEAL_CHARS] = {
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", 
    "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", 
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
};

// Ideal Templates using 9 control points (8 segments)
const Ideal_Curve_Params IDEAL_TEMPLATES[NUM_IDEAL_CHARS] = {
    // 0: 'A'
    [0] = {.control_points = {
        {.x = 0.5, .y = 0.1}, {.x = 0.3, .y = 0.3}, {.x = 0.2, .y = 0.5}, {.x = 0.3, .y = 0.6}, 
        {.x = 0.7, .y = 0.6}, {.x = 0.8, .y = 0.5}, {.x = 0.7, .y = 0.3}, {.x = 0.5, .y = 0.1}, 
        {.x = 0.7, .y = 0.9} 
    }},
    // 1: 'B'
    [1] = {.control_points = {
        {.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.9}, 
        {.x = 0.2, .y = 0.1}, 
        {.x = 0.8, .y = 0.1}, {.x = 0.8, .y = 0.3}, {.x = 0.2, .y = 0.5}, 
        {.x = 0.8, .y = 0.5}, {.x = 0.8, .y = 0.7}, {.x = 0.2, .y = 0.9} 
    }},
    // 2: 'C'
    [2] = {.control_points = {
        {.x = 0.8, .y = 0.1}, {.x = 0.4, .y = 0.1}, {.x = 0.1, .y = 0.3}, {.x = 0.1, .y = 0.7},
        {.x = 0.4, .y = 0.9}, {.x = 0.8, .y = 0.9}, {.x = 0.8, .y = 0.7}, {.x = 0.8, .y = 0.3}, 
        {.x = 0.1, .y = 0.5} 
    }},
    // 3: 'D'
    [3] = {.control_points = {
        {.x = 0.2, .y = 0.1}, {.x = 0.7, .y = 0.1}, {.x = 0.8, .y = 0.3}, {.x = 0.8, .y = 0.5}, 
        {.x = 0.8, .y = 0.7}, {.x = 0.7, .y = 0.9}, {.x = 0.2, .y = 0.9}, {.x = 0.2, .y = 0.5}, 
        {.x = 0.2, .y = 0.1} 
    }},
    // 4: 'E'
    [4] = {.control_points = {
        {.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.1}, 
        {.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.5}, 
        {.x = 0.75, .y = 0.5}, 
        {.x = 0.2, .y = 0.5}, {.x = 0.2, .y = 0.9}, 
        {.x = 0.8, .y = 0.9}, 
        {.x = 0.2, .y = 0.9} 
    }},
    // 5: 'F'
    [5] = {.control_points = {
        {.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.1}, {.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.5}, 
        {.x = 0.6, .y = 0.5}, {.x = 0.2, .y = 0.5}, {.x = 0.2, .y = 0.9}, {.x = 0.2, .y = 0.9}, 
        {.x = 0.2, .y = 0.1} 
    }},
    // 6: 'G'
    [6] = {.control_points = {
        {.x = 0.8, .y = 0.1}, {.x = 0.2, .y = 0.1}, {.x = 0.1, .y = 0.5}, {.x = 0.2, .y = 0.9}, 
        {.x = 0.8, .y = 0.9}, {.x = 0.8, .y = 0.6}, {.x = 0.4, .y = 0.6}, {.x = 0.4, .y = 0.9}, 
        {.x = 0.8, .y = 0.9} 
    }},
    // 7: 'H'
    [7] = {.control_points = {
        {.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.9}, 
        {.x = 0.2, .y = 0.5}, {.x = 0.8, .y = 0.5}, 
        {.x = 0.8, .y = 0.1}, {.x = 0.8, .y = 0.9}, 
        {.x = 0.2, .y = 0.5}, {.x = 0.8, .y = 0.5}, 
        {.x = 0.8, .y = 0.5} 
    }},
    // 8: 'I'
    [8] = {.control_points = {
        {.x = 0.3, .y = 0.1}, {.x = 0.7, .y = 0.1}, // Top bar
        {.x = 0.5, .y = 0.1}, {.x = 0.5, .y = 0.9}, // Vertical stem
        {.x = 0.3, .y = 0.9}, {.x = 0.7, .y = 0.9}, // Bottom bar
        {.x = 0.5, .y = 0.5}, {.x = 0.5, .y = 0.1}, 
        {.x = 0.5, .y = 0.9} 
    }},
    // 9: 'J'
    [9] = {.control_points = {
        {.x = 0.6, .y = 0.1}, {.x = 0.6, .y = 0.2}, {.x = 0.6, .y = 0.4}, {.x = 0.6, .y = 0.5}, 
        {.x = 0.5, .y = 0.6}, {.x = 0.4, .y = 0.75}, {.x = 0.3, .y = 0.9}, {.x = 0.4, .y = 0.85}, 
        {.x = 0.5, .y = 0.8}  
    }},
    // 10: 'K'
    [10] = {.control_points = {
        {.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.9}, // P0-P1: Vertical stem
        {.x = 0.2, .y = 0.5}, // P2: Central joint
        {.x = 0.8, .y = 0.1}, // P3: Top right leg
        {.x = 0.2, .y = 0.5}, // P4: Return to joint
        {.x = 0.8, .y = 0.9}, // P5: Bottom right leg
        {.x = 0.2, .y = 0.5}, // P6: Return to joint
        {.x = 0.2, .y = 0.5}, {.x = 0.2, .y = 0.5} 
    }},
    // 11: 'L'
    [11] = {.control_points = {
        {.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.9}, {.x = 0.8, .y = 0.9}, {.x = 0.2, .y = 0.9}, 
        {.x = 0.2, .y = 0.6}, {.x = 0.2, .y = 0.3}, {.x = 0.2, .y = 0.1}, {.x = 0.5, .y = 0.5}, 
        {.x = 0.2, .y = 0.9} 
    }},
    // 12: 'M'
    [12] = {.control_points = {
        {.x = 0.1, .y = 0.9}, {.x = 0.1, .y = 0.1}, // Left stem (Full height)
        {.x = 0.5, .y = 0.8}, // Deep center V point (y=0.8 is even deeper)
        {.x = 0.9, .y = 0.1}, // Right peak (Full height)
        {.x = 0.9, .y = 0.9}, // Right stem (Full height)
        {.x = 0.5, .y = 0.8}, {.x = 0.1, .y = 0.1}, // Back to center
        {.x = 0.9, .y = 0.9}, {.x = 0.5, .y = 0.3} 
    }},
    // 13: 'N'
    [13] = {.control_points = {
        {.x = 0.2, .y = 0.9}, {.x = 0.2, .y = 0.1}, // Left vertical
        {.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.9}, // Clear, long diagonal
        {.x = 0.8, .y = 0.9}, {.x = 0.8, .y = 0.1}, // Right vertical
        {.x = 0.5, .y = 0.5}, {.x = 0.2, .y = 0.9}, 
        {.x = 0.8, .y = 0.1} 
    }},
    // 14: 'O'
    [14] = {.control_points = {
        {.x = 0.5, .y = 0.1}, {.x = 0.85, .y = 0.3}, {.x = 0.85, .y = 0.7}, 
        {.x = 0.5, .y = 0.9}, {.x = 0.15, .y = 0.7}, {.x = 0.15, .y = 0.3}, 
        {.x = 0.5, .y = 0.1}, {.x = 0.5, .y = 0.5}, {.x = 0.5, .y = 0.5} 
    }},
    // 15: 'P'
    [15] = {.control_points = {
        {.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.9}, 
        {.x = 0.2, .y = 0.1}, 
        {.x = 0.7, .y = 0.1}, {.x = 0.8, .y = 0.2}, {.x = 0.8, .y = 0.3}, 
        {.x = 0.7, .y = 0.4}, {.x = 0.2, .y = 0.4}, 
        {.x = 0.2, .y = 0.4} 
    }},
    // 16: 'Q'
    [16] = {.control_points = {
        {.x = 0.5, .y = 0.1}, {.x = 0.8, .y = 0.3}, {.x = 0.8, .y = 0.7}, {.x = 0.5, .y = 0.9}, 
        {.x = 0.2, .y = 0.7}, {.x = 0.2, .y = 0.3}, {.x = 0.5, .y = 0.1}, {.x = 0.6, .y = 0.7}, 
        {.x = 0.8, .y = 0.9} 
    }},
    // 17: 'R'
    [17] = {.control_points = {
        {.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.9}, // Vertical stem
        {.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.1}, 
        {.x = 0.8, .y = 0.4}, {.x = 0.2, .y = 0.5}, // Top loop closure point
        {.x = 0.2, .y = 0.5}, {.x = 0.9, .y = 0.9}, // Diagonal leg starts from middle (y=0.5), ends far right (x=0.9)
        {.x = 0.2, .y = 0.1} 
    }},
    // 18: 'S'
    [18] = {.control_points = {
        {.x = 0.8, .y = 0.1}, {.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.3}, {.x = 0.8, .y = 0.5}, 
        {.x = 0.8, .y = 0.7}, {.x = 0.2, .y = 0.9}, {.x = 0.8, .y = 0.9}, {.x = 0.5, .y = 0.5}, 
        {.x = 0.8, .y = 0.1} 
    }},
    // 19: 'T'
    [19] = {.control_points = {
        {.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.1}, {.x = 0.5, .y = 0.1}, {.x = 0.5, .y = 0.9}, 
        {.x = 0.5, .y = 0.5}, {.x = 0.8, .y = 0.1}, {.x = 0.2, .y = 0.1}, {.x = 0.5, .y = 0.9}, 
        {.x = 0.5, .y = 0.1} 
    }},
    // 20: 'U'
    [20] = {.control_points = {
        {.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.7}, {.x = 0.5, .y = 0.9}, {.x = 0.8, .y = 0.7}, 
        {.x = 0.8, .y = 0.1}, {.x = 0.5, .y = 0.9}, {.x = 0.2, .y = 0.7}, {.x = 0.8, .y = 0.7}, 
        {.x = 0.2, .y = 0.1} 
    }},
    // 21: 'V'
    [21] = {.control_points = {
        {.x = 0.2, .y = 0.1}, {.x = 0.5, .y = 0.9}, {.x = 0.8, .y = 0.1}, {.x = 0.5, .y = 0.9}, 
        {.x = 0.2, .y = 0.5}, {.x = 0.8, .y = 0.5}, {.x = 0.5, .y = 0.9}, {.x = 0.2, .y = 0.1}, 
        {.x = 0.8, .y = 0.1} 
    }},
    // 22: 'W'
    [22] = {.control_points = {
        {.x = 0.1, .y = 0.1}, {.x = 0.3, .y = 0.9}, 
        {.x = 0.5, .y = 0.2}, 
        {.x = 0.7, .y = 0.9}, 
        {.x = 0.9, .y = 0.1}, 
        {.x = 0.5, .y = 0.5}, {.x = 0.5, .y = 0.5}, 
        {.x = 0.5, .y = 0.5}, {.x = 0.5, .y = 0.5}
    }},
    // 23: 'X'
    [23] = {.control_points = {
        {.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.9}, {.x = 0.5, .y = 0.5}, {.x = 0.8, .y = 0.1}, 
        {.x = 0.2, .y = 0.9}, {.x = 0.5, .y = 0.5}, {.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.9}, 
        {.x = 0.5, .y = 0.5} 
    }},
    // 24: 'Y'
    [24] = {.control_points = {
        {.x = 0.2, .y = 0.1}, {.x = 0.5, .y = 0.5}, {.x = 0.8, .y = 0.1}, {.x = 0.5, .y = 0.5}, 
        {.x = 0.5, .y = 0.9}, {.x = 0.5, .y = 0.7}, {.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.1}, 
        {.x = 0.5, .y = 0.9} 
    }},
    // 25: 'Z'
    [25] = {.control_points = {
        {.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.1}, {.x = 0.2, .y = 0.9}, {.x = 0.8, .y = 0.9}, 
        {.x = 0.5, .y = 0.5}, {.x = 0.8, .y = 0.1}, {.x = 0.2, .y = 0.9}, {.x = 0.5, .y = 0.5}, 
        {.x = 0.2, .y = 0.1} 
    }},
    
    // --- Digits (0-9) ---
    // 26: '0'
    [26] = {.control_points = {
        {.x = 0.5, .y = 0.1}, {.x = 0.75, .y = 0.3}, {.x = 0.75, .y = 0.7}, 
        {.x = 0.5, .y = 0.9}, {.x = 0.25, .y = 0.7}, {.x = 0.25, .y = 0.3}, 
        {.x = 0.5, .y = 0.1}, {.x = 0.5, .y = 0.5}, {.x = 0.5, .y = 0.1} 
    }},
    // 27: '1'
    [27] = {.control_points = {
        {.x = 0.4, .y = 0.1}, {.x = 0.5, .y = 0.1}, {.x = 0.5, .y = 0.2}, {.x = 0.5, .y = 0.35}, 
        {.x = 0.5, .y = 0.5}, {.x = 0.5, .y = 0.65}, {.x = 0.5, .y = 0.8}, {.x = 0.5, .y = 0.9}, 
        {.x = 0.5, .y = 0.9} 
    }},
    // 28: '2'
    [28] = {.control_points = {
        {.x = 0.2, .y = 0.1}, {.x = 0.7, .y = 0.1}, {.x = 0.8, .y = 0.25}, {.x = 0.7, .y = 0.4}, 
        {.x = 0.3, .y = 0.55}, {.x = 0.2, .y = 0.7}, {.x = 0.3, .y = 0.8}, {.x = 0.5, .y = 0.9}, 
        {.x = 0.8, .y = 0.9} 
    }}, 
    // 29: '3'
    [29] = {.control_points = {
        {.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.1}, {.x = 0.8, .y = 0.3}, {.x = 0.4, .y = 0.5}, 
        {.x = 0.8, .y = 0.5}, {.x = 0.8, .y = 0.7}, {.x = 0.4, .y = 0.9}, {.x = 0.8, .y = 0.9}, 
        {.x = 0.2, .y = 0.9} 
    }}, 
    // 30: '4'
    [30] = {.control_points = {
        {.x = 0.2, .y = 0.1}, {.x = 0.3, .y = 0.2}, {.x = 0.4, .y = 0.3}, {.x = 0.5, .y = 0.4}, 
        {.x = 0.2, .y = 0.55}, {.x = 0.8, .y = 0.55}, {.x = 0.8, .y = 0.7}, {.x = 0.8, .y = 0.85}, 
        {.x = 0.8, .y = 0.9} 
    }}, 
    // 31: '5'
    [31] = {.control_points = {
        {.x = 0.8, .y = 0.1}, {.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.4}, {.x = 0.6, .y = 0.4}, 
        {.x = 0.8, .y = 0.6}, {.x = 0.7, .y = 0.8}, {.x = 0.2, .y = 0.9}, {.x = 0.8, .y = 0.9}, 
        {.x = 0.8, .y = 0.1} 
    }},
    // 32: '6'
    [32] = {.control_points = {
        {.x = 0.7, .y = 0.1}, {.x = 0.2, .y = 0.3}, {.x = 0.2, .y = 0.5}, {.x = 0.5, .y = 0.7}, 
        {.x = 0.8, .y = 0.6}, {.x = 0.5, .y = 0.5}, {.x = 0.2, .y = 0.5}, {.x = 0.2, .y = 0.9}, 
        {.x = 0.7, .y = 0.1} 
    }},
    // 33: '7'
    [33] = {.control_points = {
        {.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.1}, {.x = 0.8, .y = 0.1}, {.x = 0.3, .y = 0.9}, 
        {.x = 0.5, .y = 0.5}, {.x = 0.6, .y = 0.7}, {.x = 0.8, .y = 0.1}, {.x = 0.2, .y = 0.9}, 
        {.x = 0.8, .y = 0.1} 
    }},
    // 34: '8'
    [34] = {.control_points = {
        {.x = 0.5, .y = 0.15}, {.x = 0.7, .y = 0.2}, {.x = 0.7, .y = 0.35}, 
        {.x = 0.5, .y = 0.45}, {.x = 0.3, .y = 0.35}, {.x = 0.3, .y = 0.2}, 
        {.x = 0.7, .y = 0.65}, {.x = 0.5, .y = 0.85}, {.x = 0.3, .y = 0.65} 
    }},
    // 35: '9'
    [35] = {.control_points = {
        {.x = 0.5, .y = 0.1}, {.x = 0.8, .y = 0.2}, {.x = 0.5, .y = 0.4}, {.x = 0.2, .y = 0.2}, 
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
    // Clear the canvas
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            img[i][j] = 0.0;
        }
    }

    // Sample and draw points
    for (int i = 0; i <= NUM_POINTS; i++) {
        const double t = (double)i / NUM_POINTS;
        const Point current_p = get_deformed_point(t, ideal_params, alpha);

        const int px = (int)round(current_p.x);
        const int py = (int)round(current_p.y);

        if (px >= 0 && px < GRID_SIZE && py >= 0 && py < GRID_SIZE) {
            img[py][px] = fmin(1.0, img[py][px] + 0.5); 
        }

        // Neighbor smoothing (basic anti-aliasing)
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
                      int ideal_char_index, EstimationResult *result, int print_trace) {
    
    const Ideal_Curve_Params *ideal_params = &IDEAL_TEMPLATES[ideal_char_index];
    
    Deformation_Coefficients alpha_hat = { .alpha = {0.0, 0.0} };
    double learning_rate = 0.00000005; 
    const double min_learning_rate = 0.00000000005;
    double gradient[NUM_DEFORMATIONS];
    Generated_Image generated_image;
    Feature_Vector generated_features;
    double combined_loss;
    double prev_combined_loss = HUGE_VAL; 
    double current_feature_loss_only = HUGE_VAL;

    if (print_trace) {
        printf("\n    --- Optimizing against '%s' ---\n", CHAR_NAMES[ideal_char_index]);
        printf("    It | Feat Loss| L Rate  | a_1 (Slant) | a_2 (Curve)\n");
        printf("    ------------------------------------------------------\n");
    }

    for (int t = 0; t <= ITERATIONS; t++) {
        draw_curve(alpha_hat.alpha, generated_image, ideal_params);
        extract_geometric_features(generated_image, generated_features);
        
        current_feature_loss_only = calculate_feature_loss_L2(generated_features, observed_features);
        combined_loss = current_feature_loss_only + PIXEL_LOSS_WEIGHT * calculate_pixel_loss_L2(generated_image, observed_image);

        if (combined_loss > prev_combined_loss * 1.001 && learning_rate > min_learning_rate) { 
            learning_rate *= 0.5;
        }

        prev_combined_loss = combined_loss;

        calculate_gradient(observed_image, observed_features, &alpha_hat, combined_loss, gradient, ideal_params);
        
        if (print_trace && (t % 500 == 0 || t == ITERATIONS)) {
            printf("    %04d | %8.5f | %7.8f | %8.4f | %8.4f\n", t, current_feature_loss_only, learning_rate, alpha_hat.alpha[0], alpha_hat.alpha[1]);
        }

        if (t < ITERATIONS) {
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
    
    generate_target_image(result->true_image, true_alpha, true_params, 0); 
    
    Generated_Image observed_image; 
    generate_target_image(observed_image, true_alpha, true_params, 1);
    memcpy(result->observed_image, observed_image, sizeof(Generated_Image));
    
    Feature_Vector observed_features;
    extract_geometric_features(observed_image, observed_features);
    
    printf("\n======================================================\n");
    printf("TEST %02d/%02d (TRUE: '%s'): Slant (a_1)=%.4f, Curve (a_2)=%.4f\n", 
           test_id, NUM_TESTS, CHAR_NAMES[true_char_index], true_alpha[0], true_alpha[1]);

    double min_feature_loss = HUGE_VAL;
    int best_match_index = -1;
    
    for (int i = 0; i < NUM_IDEAL_CHARS; i++) {
        int print_trace = (i == true_char_index && i < 5); 
        run_optimization(observed_image, observed_features, i, &result->classification_results[i], print_trace);

        if (result->classification_results[i].final_loss < min_feature_loss) {
            min_feature_loss = result->classification_results[i].final_loss;
            best_match_index = i;
        }
    }

    result->best_match_index = best_match_index;

    EstimationResult *best_fit = &result->classification_results[best_match_index];
    
    draw_curve(best_fit->estimated_alpha, result->best_estimated_image, &IDEAL_TEMPLATES[best_match_index]);
    calculate_difference_image(result->observed_image, result->best_estimated_image, result->best_diff_image);
}

#define DETAILED_SUMMARY_LIMIT 5 

void summarize_results_console() {
    printf("\n\n================================================================================\n");
    printf("               CLASSIFICATION SUMMARY (%d TESTS: A-Z, 0-9)                   \n", NUM_TESTS);
    printf("     (Features: %d Vectors * %d Bins = %d. Loss: L_Feature + %.1f * L_Pixel)     \n", 
           NUM_VECTORS, NUM_BINS, NUM_FEATURES, PIXEL_LOSS_WEIGHT);
    printf("================================================================================\n");
    
    printf("\n--- DETAILED LOSS MATRIX (Showing ALL %d Template Losses for first %d Tests) ---\n", 
           NUM_IDEAL_CHARS, DETAILED_SUMMARY_LIMIT);
    
    printf("  ID | TRUE | PRED | Best Feat Loss | Loss against: ");
    for (int i = 0; i < NUM_IDEAL_CHARS; i++) {
        printf(" %s |", CHAR_NAMES[i]);
    }
    printf("\n-----|------|------|----------------|");
    for (int i = 0; i < NUM_IDEAL_CHARS; i++) {
        printf("----|"); 
    }
    printf("\n");
    
    for (int k = 0; k < DETAILED_SUMMARY_LIMIT; k++) {
        TestResult *r = &all_results[k];
        printf("%4d | %4s | %4s | %14.4f |", 
               r->id, CHAR_NAMES[r->true_char_index], CHAR_NAMES[r->best_match_index],
               r->classification_results[r->best_match_index].final_loss);
        
        for (int i = 0; i < NUM_IDEAL_CHARS; i++) {
            printf(" %.1f |", r->classification_results[i].final_loss);
        }
        printf("\n");
    }
    printf("--------------------------------------------------------------------------------\n");

    printf("\n--- CLASSIFICATION PERFORMANCE SUMMARY (All %d Tests) ---\n", NUM_TESTS);
    printf("  ID | TRUE | PRED | Best Feat Loss | a_1 (Slant) | a_2 (Curve) | **PIXEL ERROR %%** | Correct?\n");
    printf("-----|------|------|----------------|-------------|-------------|-------------------|----------\n");

    int correct_classifications = 0;
    for (int k = 0; k < NUM_TESTS; k++) {
        TestResult *r = &all_results[k];
        const EstimationResult *best_fit = &r->classification_results[r->best_match_index];
        
        double pixel_error_sum = calculate_pixel_error_sum(r->observed_image, r->best_estimated_image);
        double pixel_error_percent = (pixel_error_sum / MAX_PIXEL_ERROR) * 100.0;

        int is_correct = (r->true_char_index == r->best_match_index);
        if (is_correct) correct_classifications++;

        printf("%4d | %4s | %4s | %14.4f | %11.4f | %11.4f | %15.2f | %8s\n", 
               r->id, CHAR_NAMES[r->true_char_index], CHAR_NAMES[r->best_match_index],
               best_fit->final_loss, best_fit->estimated_alpha[0], best_fit->estimated_alpha[1], 
               pixel_error_percent, is_correct ? "YES" : "NO");
    }
    printf("--------------------------------------------------------------------------------\n");
    printf("Overall Accuracy: %d/%d (%.2f%%)\n", correct_classifications, NUM_TESTS, 
           (double)correct_classifications / NUM_TESTS * 100.0);
    printf("================================================================================\n");
}


// --- PNG Rendering Constants ---
#define PIXEL_SIZE 5    
#define IMG_SIZE (GRID_SIZE * PIXEL_SIZE) // 80
#define IMG_SPACING 5   
#define TEXT_HEIGHT 15  
#define SET_SPACING 40  
// Width for 4 images, 3 spacings, and the large text label space
#define SET_WIDTH (4 * IMG_SIZE + 3 * IMG_SPACING + SET_SPACING) 
#define PNG_RENDER_LIMIT 10 // Number of test sets per row
#define PNG_WIDTH (PNG_RENDER_LIMIT * SET_WIDTH + SET_SPACING) 
#define PNG_HEIGHT (IMG_SIZE + 2 * TEXT_HEIGHT + SET_SPACING) // Image height + Title + Label + Spacing
#define NUM_CHANNELS 3 // RGB (24-bit color)

// --- PNG Rendering Functions ---

void set_pixel(unsigned char *buffer, int x, int y, int width, unsigned char r, unsigned char g, unsigned char b) {
    if (x >= 0 && x < width && y >= 0 && y < PNG_HEIGHT) {
        long index = (long)y * width * NUM_CHANNELS + x * NUM_CHANNELS;
        buffer[index] = r;
        buffer[index + 1] = g;
        buffer[index + 2] = b;
    }
}

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

void draw_text_placeholder(unsigned char *buffer, int buf_width, int x, int y, const char* text, unsigned char r, unsigned char g, unsigned char b) {
    // This is a placeholder for text start position.
    // In a real application, a font rendering library would be used here.
    set_pixel(buffer, x, y, buf_width, r, g, b);

    // Draw a small horizontal line to indicate text length/area
    for(int i = 1; i < 50; i++) {
        set_pixel(buffer, x + i, y, buf_width, r, g, b);
    }
}

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
    
    // TRUE CLEAN (starts at y_set + TEXT_HEIGHT to leave room for column titles)
    render_single_image_to_png(buffer, buf_width, r->true_image, x_set, y_set + TEXT_HEIGHT, 0); 
    
    // OBSERVED NOISY
    render_single_image_to_png(buffer, buf_width, r->observed_image, x_set + img_step, y_set + TEXT_HEIGHT, 0);
    
    // BEST ESTIMATED
    render_single_image_to_png(buffer, buf_width, r->best_estimated_image, x_set + 2 * img_step, y_set + TEXT_HEIGHT, 0);
    
    // ERROR DIFF (is_error_map = 1)
    render_single_image_to_png(buffer, buf_width, r->best_diff_image, x_set + 3 * img_step, y_set + TEXT_HEIGHT, 1);
    
    // 2. Draw Text Placeholder (at y_set + TEXT_HEIGHT + IMG_SIZE + 5)
    draw_text_placeholder(buffer, buf_width, x_set, y_set + TEXT_HEIGHT + IMG_SIZE + 5, label, 255, 255, 255);
}

void generate_png_file() {
    long buffer_size = (long)PNG_WIDTH * PNG_HEIGHT * NUM_CHANNELS;
    unsigned char *buffer = (unsigned char *)calloc(buffer_size, 1);
    if (buffer == NULL) {
        printf("\nERROR: Failed to allocate buffer for PNG.\n");
        return;
    }

    // Draw column titles (simple white pixel marker)
    int initial_x = SET_SPACING + IMG_SIZE / 2;
    int initial_y = TEXT_HEIGHT / 2;
    int img_step = IMG_SIZE + IMG_SPACING;
    
    // Using a placeholder function that also acts as a visual marker for the text area
    draw_text_placeholder(buffer, PNG_WIDTH, initial_x, initial_y, "TRUE CLEAN", 255, 255, 255);
    draw_text_placeholder(buffer, PNG_WIDTH, initial_x + img_step, initial_y, "OBSERVED NOISY", 255, 255, 255);
    draw_text_placeholder(buffer, PNG_WIDTH, initial_x + 2 * img_step, initial_y, "BEST ESTIMATED", 255, 255, 255);
    draw_text_placeholder(buffer, PNG_WIDTH, initial_x + 3 * img_step, initial_y, "ERROR DIFF", 255, 255, 255);
    
    // Render the test sets in a single row
    int actual_render_limit = (NUM_TESTS < PNG_RENDER_LIMIT) ? NUM_TESTS : PNG_RENDER_LIMIT;
    int y_set = TEXT_HEIGHT; 
    
    for (int k = 0; k < actual_render_limit; k++) {
        int x_set = SET_SPACING + k * SET_WIDTH;
        
        // **FIXED: Removed the extra '0' argument from render_test_to_png**
        render_test_to_png(buffer, PNG_WIDTH, &all_results[k], x_set, y_set);
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


// --- Main Execution ---

int main(void) {
    srand(42); // Seed for reproducible results

    const double MIN_ALPHA = -0.15;
    const double MAX_ALPHA = 0.15;

    for (int i = 0; i < NUM_TESTS; i++) {
        double true_alpha[NUM_DEFORMATIONS];
        // Generate random deformation in [-0.15, 0.15]
        true_alpha[0] = MIN_ALPHA + ((double)rand() / RAND_MAX) * (MAX_ALPHA - MIN_ALPHA);
        true_alpha[1] = MIN_ALPHA + ((double)rand() / RAND_MAX) * (MAX_ALPHA - MIN_ALPHA);
        
        run_classification_test(i + 1, i, true_alpha, &all_results[i]);
    }
    
    summarize_results_console();

    generate_png_file();

    return 0;
}
