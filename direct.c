#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

// The number of tests is set equal to the number of ideal characters
#define NUM_TESTS 36        

// 8 segments require 9 control points (P0 to P8)
#define NUM_CONTROL_POINTS 9 

// Maximum possible total absolute pixel difference (16*16 = 256.0)
#define MAX_PIXEL_ERROR (GRID_SIZE * GRID_SIZE) 

// --- Data Structures ---

typedef struct {
    double x; 
    double y; 
} Point;

typedef struct {
    // Array of 9 control points defining the 8 segments of the curve
    const Point control_points[NUM_CONTROL_POINTS];
} Ideal_Curve_Params; 

typedef struct {
    double alpha[NUM_DEFORMATIONS]; 
} Deformation_Coefficients;

typedef double Generated_Image[GRID_SIZE][GRID_SIZE]; 
typedef double Feature_Vector[NUM_FEATURES]; 

// Structure to hold one potential estimation result (Template vs. Observed)
typedef struct {
    double estimated_alpha[NUM_DEFORMATIONS];
    double final_loss; // Stores FEATURE LOSS only for reporting
} EstimationResult;

// Structure to hold results for one full test case (classification results)
typedef struct {
    int id;
    int true_char_index; // Index of the true character (0='A', 1='B', ...)
    int best_match_index; // Index of the character with the minimum loss
    double true_alpha[NUM_DEFORMATIONS];
    double true_image[GRID_SIZE][GRID_SIZE];
    double observed_image[GRID_SIZE][GRID_SIZE];
    
    // Stores the results of running the observation against all 36 ideal templates
    EstimationResult classification_results[NUM_IDEAL_CHARS]; 
    
    // The final estimated image and diff image based on the BEST MATCH
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
    // 0: 'A' (Diagonal V with a crossbar)
    [0] = {.control_points = {
        {.x = 0.5, .y = 0.1}, {.x = 0.3, .y = 0.3}, {.x = 0.2, .y = 0.5}, {.x = 0.3, .y = 0.6}, 
        {.x = 0.7, .y = 0.6}, {.x = 0.8, .y = 0.5}, {.x = 0.7, .y = 0.3}, {.x = 0.5, .y = 0.1}, 
        {.x = 0.7, .y = 0.9} 
    }},
    // 1: 'B' (Vertical stem, two right curves) 
    [1] = {.control_points = {
        {.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.9}, 
        {.x = 0.2, .y = 0.1}, 
        {.x = 0.8, .y = 0.1}, {.x = 0.8, .y = 0.3}, {.x = 0.2, .y = 0.5}, 
        {.x = 0.8, .y = 0.5}, {.x = 0.8, .y = 0.7}, {.x = 0.2, .y = 0.9} 
    }},
    // 2: 'C' (Open curve)
    [2] = {.control_points = {
        {.x = 0.8, .y = 0.1}, {.x = 0.4, .y = 0.1}, {.x = 0.1, .y = 0.3}, {.x = 0.1, .y = 0.7},
        {.x = 0.4, .y = 0.9}, {.x = 0.8, .y = 0.9}, {.x = 0.8, .y = 0.7}, {.x = 0.8, .y = 0.3}, 
        {.x = 0.1, .y = 0.5} 
    }},
    // 3: 'D' (Vertical stem, large right curve) 
    [3] = {.control_points = {
        {.x = 0.2, .y = 0.1}, {.x = 0.7, .y = 0.1}, {.x = 0.8, .y = 0.3}, {.x = 0.8, .y = 0.5}, 
        {.x = 0.8, .y = 0.7}, {.x = 0.7, .y = 0.9}, {.x = 0.2, .y = 0.9}, {.x = 0.2, .y = 0.5}, 
        {.x = 0.2, .y = 0.1} 
    }},
    // 4: 'E' (Vertical stem, three horizontal bars)
    [4] = {.control_points = {
        {.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.1}, 
        {.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.5}, 
        {.x = 0.75, .y = 0.5}, 
        {.x = 0.2, .y = 0.5}, {.x = 0.2, .y = 0.9}, 
        {.x = 0.8, .y = 0.9}, 
        {.x = 0.2, .y = 0.9} 
    }},
    // 5: 'F' (Vertical stem, two horizontal bars)
    [5] = {.control_points = {
        {.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.1}, {.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.5}, 
        {.x = 0.6, .y = 0.5}, {.x = 0.2, .y = 0.5}, {.x = 0.2, .y = 0.9}, {.x = 0.2, .y = 0.9}, 
        {.x = 0.2, .y = 0.1} 
    }},
    // 6: 'G' (Open curve, bar near bottom)
    [6] = {.control_points = {
        {.x = 0.8, .y = 0.1}, {.x = 0.2, .y = 0.1}, {.x = 0.1, .y = 0.5}, {.x = 0.2, .y = 0.9}, 
        {.x = 0.8, .y = 0.9}, {.x = 0.8, .y = 0.6}, {.x = 0.4, .y = 0.6}, {.x = 0.4, .y = 0.9}, 
        {.x = 0.8, .y = 0.9} 
    }},
    // 7: 'H' (Two vertical stems, one crossbar)
    [7] = {.control_points = {
        {.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.9}, 
        {.x = 0.2, .y = 0.5}, {.x = 0.8, .y = 0.5}, 
        {.x = 0.8, .y = 0.1}, {.x = 0.8, .y = 0.9}, 
        {.x = 0.2, .y = 0.5}, {.x = 0.8, .y = 0.5}, 
        {.x = 0.8, .y = 0.5} 
    }},
    // 8: 'I' (Vertical stem, with top and bottom serifs/bars) - UPDATED
    [8] = {.control_points = {
        {.x = 0.3, .y = 0.1}, {.x = 0.7, .y = 0.1}, // Top bar
        {.x = 0.5, .y = 0.1}, {.x = 0.5, .y = 0.9}, // Vertical stem
        {.x = 0.3, .y = 0.9}, {.x = 0.7, .y = 0.9}, // Bottom bar
        {.x = 0.5, .y = 0.5}, {.x = 0.5, .y = 0.1}, 
        {.x = 0.5, .y = 0.9} 
    }},
    // 9: 'J' (Vertical stroke down, wide hook left at bottom)
    [9] = {.control_points = {
        {.x = 0.6, .y = 0.1}, {.x = 0.6, .y = 0.2}, {.x = 0.6, .y = 0.4}, {.x = 0.6, .y = 0.5}, 
        {.x = 0.5, .y = 0.6}, {.x = 0.4, .y = 0.75}, {.x = 0.3, .y = 0.9}, {.x = 0.4, .y = 0.85}, 
        {.x = 0.5, .y = 0.8}  
    }},
    // 10: 'K' (Vertical stem, two diagonal legs)
    [10] = {.control_points = {
        {.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.9}, // P0-P1: Vertical stem
        {.x = 0.2, .y = 0.5}, // P2: Central joint
        {.x = 0.8, .y = 0.1}, // P3: Top right leg
        {.x = 0.2, .y = 0.5}, // P4: Return to joint
        {.x = 0.8, .y = 0.9}, // P5: Bottom right leg
        {.x = 0.2, .y = 0.5}, // P6: Return to joint
        {.x = 0.2, .y = 0.5}, {.x = 0.2, .y = 0.5} 
    }},
    // 11: 'L' (Vertical stem, horizontal base)
    [11] = {.control_points = {
        {.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.9}, {.x = 0.8, .y = 0.9}, {.x = 0.2, .y = 0.9}, 
        {.x = 0.2, .y = 0.6}, {.x = 0.2, .y = 0.3}, {.x = 0.2, .y = 0.1}, {.x = 0.5, .y = 0.5}, 
        {.x = 0.2, .y = 0.9} 
    }},
    // 12: 'M' (W shape upside down) - REFINED
    [12] = {.control_points = {
        {.x = 0.1, .y = 0.9}, {.x = 0.1, .y = 0.1}, // Left stem (Full height)
        {.x = 0.5, .y = 0.8}, // Deep center V point (y=0.8 is even deeper)
        {.x = 0.9, .y = 0.1}, // Right peak (Full height)
        {.x = 0.9, .y = 0.9}, // Right stem (Full height)
        {.x = 0.5, .y = 0.8}, {.x = 0.1, .y = 0.1}, // Back to center
        {.x = 0.9, .y = 0.9}, {.x = 0.5, .y = 0.3} 
    }},
    // 13: 'N' (Two verticals, one diagonal) - REFINED
    [13] = {.control_points = {
        {.x = 0.2, .y = 0.9}, {.x = 0.2, .y = 0.1}, // Left vertical
        {.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.9}, // Clear, long diagonal
        {.x = 0.8, .y = 0.9}, {.x = 0.8, .y = 0.1}, // Right vertical
        {.x = 0.5, .y = 0.5}, {.x = 0.2, .y = 0.9}, 
        {.x = 0.8, .y = 0.1} 
    }},
    // 14: 'O' (Circle)
    [14] = {.control_points = {
        {.x = 0.5, .y = 0.1}, {.x = 0.85, .y = 0.3}, {.x = 0.85, .y = 0.7}, 
        {.x = 0.5, .y = 0.9}, {.x = 0.15, .y = 0.7}, {.x = 0.15, .y = 0.3}, 
        {.x = 0.5, .y = 0.1}, {.x = 0.5, .y = 0.5}, {.x = 0.5, .y = 0.5} 
    }},
    // 15: 'P' (Vertical stem, top right curve)
    [15] = {.control_points = {
        {.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.9}, 
        {.x = 0.2, .y = 0.1}, 
        {.x = 0.7, .y = 0.1}, {.x = 0.8, .y = 0.2}, {.x = 0.8, .y = 0.3}, 
        {.x = 0.7, .y = 0.4}, {.x = 0.2, .y = 0.4}, 
        {.x = 0.2, .y = 0.4} 
    }},
    // 16: 'Q' (Circle with a tail)
    [16] = {.control_points = {
        {.x = 0.5, .y = 0.1}, {.x = 0.8, .y = 0.3}, {.x = 0.8, .y = 0.7}, {.x = 0.5, .y = 0.9}, 
        {.x = 0.2, .y = 0.7}, {.x = 0.2, .y = 0.3}, {.x = 0.5, .y = 0.1}, {.x = 0.6, .y = 0.7}, 
        {.x = 0.8, .y = 0.9} 
    }},
    // 17: 'R' (Vertical stem, top right curve, diagonal leg) - REFINED for steeper leg
    [17] = {.control_points = {
        {.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.9}, // Vertical stem
        {.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.1}, 
        {.x = 0.8, .y = 0.4}, {.x = 0.2, .y = 0.5}, // Top loop closure point
        {.x = 0.2, .y = 0.5}, {.x = 0.9, .y = 0.9}, // Diagonal leg starts from middle (y=0.5), ends far right (x=0.9)
        {.x = 0.2, .y = 0.1} 
    }},
    // 18: 'S' (Continuous S-curve)
    [18] = {.control_points = {
        {.x = 0.8, .y = 0.1}, {.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.3}, {.x = 0.8, .y = 0.5}, 
        {.x = 0.8, .y = 0.7}, {.x = 0.2, .y = 0.9}, {.x = 0.8, .y = 0.9}, {.x = 0.5, .y = 0.5}, 
        {.x = 0.8, .y = 0.1} 
    }},
    // 19: 'T' (Horizontal top bar, vertical stem)
    [19] = {.control_points = {
        {.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.1}, {.x = 0.5, .y = 0.1}, {.x = 0.5, .y = 0.9}, 
        {.x = 0.5, .y = 0.5}, {.x = 0.8, .y = 0.1}, {.x = 0.2, .y = 0.1}, {.x = 0.5, .y = 0.9}, 
        {.x = 0.5, .y = 0.1} 
    }},
    // 20: 'U' (Two vertical stems, bottom curve)
    [20] = {.control_points = {
        {.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.7}, {.x = 0.5, .y = 0.9}, {.x = 0.8, .y = 0.7}, 
        {.x = 0.8, .y = 0.1}, {.x = 0.5, .y = 0.9}, {.x = 0.2, .y = 0.7}, {.x = 0.8, .y = 0.7}, 
        {.x = 0.2, .y = 0.1} 
    }},
    // 21: 'V' (Two diagonals meeting at bottom)
    [21] = {.control_points = {
        {.x = 0.2, .y = 0.1}, {.x = 0.5, .y = 0.9}, {.x = 0.8, .y = 0.1}, {.x = 0.5, .y = 0.9}, 
        {.x = 0.2, .y = 0.5}, {.x = 0.8, .y = 0.5}, {.x = 0.5, .y = 0.9}, {.x = 0.2, .y = 0.1}, 
        {.x = 0.8, .y = 0.1} 
    }},
    // 22: 'W' (Two V-shapes joined)
    [22] = {.control_points = {
        {.x = 0.1, .y = 0.1}, {.x = 0.3, .y = 0.9}, 
        {.x = 0.5, .y = 0.2}, 
        {.x = 0.7, .y = 0.9}, 
        {.x = 0.9, .y = 0.1}, 
        {.x = 0.5, .y = 0.5}, {.x = 0.5, .y = 0.5}, 
        {.x = 0.5, .y = 0.5}, {.x = 0.5, .y = 0.5}
    }},
    // 23: 'X' (Two crossing diagonals)
    [23] = {.control_points = {
        {.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.9}, {.x = 0.5, .y = 0.5}, {.x = 0.8, .y = 0.1}, 
        {.x = 0.2, .y = 0.9}, {.x = 0.5, .y = 0.5}, {.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.9}, 
        {.x = 0.5, .y = 0.5} 
    }},
    // 24: 'Y' (Top V-fork, vertical stem)
    [24] = {.control_points = {
        {.x = 0.2, .y = 0.1}, {.x = 0.5, .y = 0.5}, {.x = 0.8, .y = 0.1}, {.x = 0.5, .y = 0.5}, 
        {.x = 0.5, .y = 0.9}, {.x = 0.5, .y = 0.7}, {.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.1}, 
        {.x = 0.5, .y = 0.9} 
    }},
    // 25: 'Z' (Horizontal top, diagonal, horizontal bottom)
    [25] = {.control_points = {
        {.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.1}, {.x = 0.2, .y = 0.9}, {.x = 0.8, .y = 0.9}, 
        {.x = 0.5, .y = 0.5}, {.x = 0.8, .y = 0.1}, {.x = 0.2, .y = 0.9}, {.x = 0.5, .y = 0.5}, 
        {.x = 0.2, .y = 0.1} 
    }},
    
    // --- Digits (0-9) ---
    // 26: '0' (Oval shape)
    [26] = {.control_points = {
        {.x = 0.5, .y = 0.1}, {.x = 0.75, .y = 0.3}, {.x = 0.75, .y = 0.7}, 
        {.x = 0.5, .y = 0.9}, {.x = 0.25, .y = 0.7}, {.x = 0.25, .y = 0.3}, 
        {.x = 0.5, .y = 0.1}, {.x = 0.5, .y = 0.5}, {.x = 0.5, .y = 0.1} 
    }},
    // 27: '1' (Mostly vertical line)
    [27] = {.control_points = {
        {.x = 0.4, .y = 0.1}, {.x = 0.5, .y = 0.1}, {.x = 0.5, .y = 0.2}, {.x = 0.5, .y = 0.35}, 
        {.x = 0.5, .y = 0.5}, {.x = 0.5, .y = 0.65}, {.x = 0.5, .y = 0.8}, {.x = 0.5, .y = 0.9}, 
        {.x = 0.5, .y = 0.9} 
    }},
    // 28: '2' (S-curve: Top arc, down-left diagonal, horizontal base)
    [28] = {.control_points = {
        {.x = 0.2, .y = 0.1}, {.x = 0.7, .y = 0.1}, {.x = 0.8, .y = 0.25}, {.x = 0.7, .y = 0.4}, 
        {.x = 0.3, .y = 0.55}, {.x = 0.2, .y = 0.7}, {.x = 0.3, .y = 0.8}, {.x = 0.5, .y = 0.9}, 
        {.x = 0.8, .y = 0.9} 
    }}, 
    // 29: '3' (Two right-facing curves)
    [29] = {.control_points = {
        {.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.1}, {.x = 0.8, .y = 0.3}, {.x = 0.4, .y = 0.5}, 
        {.x = 0.8, .y = 0.5}, {.x = 0.8, .y = 0.7}, {.x = 0.4, .y = 0.9}, {.x = 0.8, .y = 0.9}, 
        {.x = 0.2, .y = 0.9} 
    }}, 
    // 30: '4' (Down-left, then across, then vertical stem)
    [30] = {.control_points = {
        {.x = 0.2, .y = 0.1}, {.x = 0.3, .y = 0.2}, {.x = 0.4, .y = 0.3}, {.x = 0.5, .y = 0.4}, 
        {.x = 0.2, .y = 0.55}, {.x = 0.8, .y = 0.55}, {.x = 0.8, .y = 0.7}, {.x = 0.8, .y = 0.85}, 
        {.x = 0.8, .y = 0.9} 
    }}, 
    // 31: '5' (Horizontal top, vertical down, right curve)
    [31] = {.control_points = {
        {.x = 0.8, .y = 0.1}, {.x = 0.2, .y = 0.1}, {.x = 0.2, .y = 0.4}, {.x = 0.6, .y = 0.4}, 
        {.x = 0.8, .y = 0.6}, {.x = 0.7, .y = 0.8}, {.x = 0.2, .y = 0.9}, {.x = 0.8, .y = 0.9}, 
        {.x = 0.8, .y = 0.1} 
    }},
    // 32: '6' (Top curve, closed bottom loop)
    [32] = {.control_points = {
        {.x = 0.7, .y = 0.1}, {.x = 0.2, .y = 0.3}, {.x = 0.2, .y = 0.5}, {.x = 0.5, .y = 0.7}, 
        {.x = 0.8, .y = 0.6}, {.x = 0.5, .y = 0.5}, {.x = 0.2, .y = 0.5}, {.x = 0.2, .y = 0.9}, 
        {.x = 0.7, .y = 0.1} 
    }},
    // 33: '7' (Horizontal top, steep diagonal)
    [33] = {.control_points = {
        {.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.1}, {.x = 0.8, .y = 0.1}, {.x = 0.3, .y = 0.9}, 
        {.x = 0.5, .y = 0.5}, {.x = 0.6, .y = 0.7}, {.x = 0.8, .y = 0.1}, {.x = 0.2, .y = 0.9}, 
        {.x = 0.8, .y = 0.1} 
    }},
    // 34: '8' (Two stacked circles)
    [34] = {.control_points = {
        {.x = 0.5, .y = 0.15}, {.x = 0.7, .y = 0.2}, {.x = 0.7, .y = 0.35}, 
        {.x = 0.5, .y = 0.45}, {.x = 0.3, .y = 0.35}, {.x = 0.3, .y = 0.2}, 
        {.x = 0.7, .y = 0.65}, {.x = 0.5, .y = 0.85}, {.x = 0.3, .y = 0.65} 
    }},
    // 35: '9' (Closed top loop, vertical stem)
    [35] = {.control_points = {
        {.x = 0.5, .y = 0.1}, {.x = 0.8, .y = 0.2}, {.x = 0.5, .y = 0.4}, {.x = 0.2, .y = 0.2}, 
        {.x = 0.5, .y = 0.1}, {.x = 0.8, .y = 0.4}, {.x = 0.8, .y = 0.9}, {.x = 0.5, .y = 0.7}, 
        {.x = 0.8, .y = 0.9} 
    }}
};

/**
 * @brief Applies Slant and Curvature deformation to a point.
 */
void apply_deformation(Point *point, const double alpha[NUM_DEFORMATIONS]) {
    // Deformation 1: Slant (Shear Transform)
    point->x = point->x + alpha[0] * (point->y - 0.5);

    // Deformation 2: Curvature/Width
    point->x = point->x + alpha[1] * sin(M_PI * point->y);
}

/**
 * @brief Generates a point on the curve using the ideal form and deformations (8-segment interpolation).
 */
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

    // Linear interpolation
    p.x = P_start.x + (P_end.x - P_start.x) * segment_t;
    p.y = P_start.y + (P_end.y - P_start.y) * segment_t;

    apply_deformation(&p, alpha);

    // Scale to pixel grid and clamp
    p.x = fmax(0.0, fmin(GRID_SIZE - 1.0, p.x * GRID_SIZE));
    p.y = fmax(0.0, fmin(GRID_SIZE - 1.0, p.y * GRID_SIZE));

    return p;
}

/**
 * @brief Rasterizes the deformed curve onto the image grid (Forward Model G).
 */
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

        // Main pixel
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

// --- Feature Extraction and Loss (Using 16 Rotational Features) ---

/**
 * @brief Extracts 512 directional features (16 vectors * 32 bins)
 */
void extract_geometric_features(const Generated_Image img, Feature_Vector features_out) {
    // Generate 16 normalized unit vectors
    double vectors[NUM_VECTORS][2];
    for (int k = 0; k < NUM_VECTORS; k++) { 
        // Angle step is 2*PI / 16 (22.5 degrees)
        const double angle = 2.0 * M_PI * k / NUM_VECTORS; 
        vectors[k][0] = cos(angle); // x component
        vectors[k][1] = sin(angle); // y component
    }
    
    // Center point for coordinate calculation
    const double center = (GRID_SIZE - 1.0) / 2.0; 
    const double MAX_PROJECTION_MAGNITUDE = sqrt(center * center + center * center); 

    // Initialize feature vector (all 512 bins)
    for (int k = 0; k < NUM_FEATURES; k++) {
        features_out[k] = 0.0;
    }

    // Iterate over all pixels
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            const double intensity = img[i][j];
            if (intensity < 0.1) continue; 

            const double vx = (double)j - center;
            const double vy = (double)i - center;
            
            // Project the mass vector onto all 16 basis vectors
            for (int k = 0; k < NUM_VECTORS; k++) {
                const double projection = (vx * vectors[k][0] + vy * vectors[k][1]);
                
                const double normalized_projection = projection / MAX_PROJECTION_MAGNITUDE;
                
                // Map the normalized projection [-1.0, 1.0] to a bin index [0, NUM_BINS-1]
                int bin_index = (int)floor((normalized_projection + 1.0) * (NUM_BINS / 2.0));
                
                // Clamp index 
                if (bin_index < 0) bin_index = 0;
                if (bin_index >= NUM_BINS) bin_index = NUM_BINS - 1;
                
                // Add the intensity contribution to the correct feature bin
                const int feature_index = k * NUM_BINS + bin_index;
                features_out[feature_index] += intensity;
            }
        }
    }
}

/**
 * @brief Calculates the L2 Loss (Squared Error) between 512-dimensional feature vectors.
 */
double calculate_feature_loss_L2(const Feature_Vector generated, const Feature_Vector observed) {
    double loss = 0.0;
    for (int k = 0; k < NUM_FEATURES; k++) {
        const double error = observed[k] - generated[k];
        loss += error * error;
    }
    return loss;
}

/**
 * @brief Calculates the L2 Loss (Squared Error) between 16x16 images.
 */
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

/**
 * @brief Calculates the Combined Loss: L_Feature + Lambda * L_Pixel.
 */
double calculate_combined_loss(const Generated_Image generated_img, const Feature_Vector generated_features,
                               const Generated_Image observed_img, const Feature_Vector observed_features) {
    
    const double feature_loss = calculate_feature_loss_L2(generated_features, observed_features);
    const double pixel_loss = calculate_pixel_loss_L2(generated_img, observed_img);
    
    // Combined Loss = Feature Loss + lambda * Pixel Loss
    return feature_loss + PIXEL_LOSS_WEIGHT * pixel_loss;
}

/**
 * @brief Simulates the Gradient calculation using Finite Differences based on COMBINED LOSS.
 */
void calculate_gradient(const Generated_Image observed_img, const Feature_Vector observed_features, 
                        const Deformation_Coefficients *const alpha, const double loss_base, 
                        double grad_out[NUM_DEFORMATIONS], const Ideal_Curve_Params *const ideal_params) {
    
    const double epsilon = GRADIENT_EPSILON; 
    Generated_Image generated_img_perturbed; 
    Feature_Vector generated_features_perturbed; 

    for (int k = 0; k < NUM_DEFORMATIONS; k++) {
        // Perturb alpha_k
        Deformation_Coefficients alpha_perturbed = *alpha;
        alpha_perturbed.alpha[k] += epsilon; 

        // Forward Pass: Draw curve -> Extract Features
        draw_curve(alpha_perturbed.alpha, generated_img_perturbed, ideal_params);
        extract_geometric_features(generated_img_perturbed, generated_features_perturbed);
        
        // Calculate Loss_perturbed (Combined Loss)
        const double loss_perturbed = calculate_combined_loss(generated_img_perturbed, generated_features_perturbed, observed_img, observed_features);

        // Compute Gradient (Finite Difference)
        grad_out[k] = (loss_perturbed - loss_base) / epsilon;
    }
}

/**
 * @brief Creates a synthetic image based on true parameters and noise.
 */
void generate_target_image(Generated_Image image_out, const double true_alpha[NUM_DEFORMATIONS], const Ideal_Curve_Params *const ideal_params, int add_noise) {
    // 1. Rasterize the TRUE deformed curve (Signal)
    draw_curve(true_alpha, image_out, ideal_params); 

    // 2. Add random noise if requested
    if (add_noise) {
        for (int i = 0; i < GRID_SIZE; i++) {
            for (int j = 0; j < GRID_SIZE; j++) {
                // White noise [-0.15, 0.15]
                double noise = ((double)rand() / RAND_MAX - 0.5) * 0.3; 
                image_out[i][j] = fmax(0.0, fmin(1.0, image_out[i][j] + noise));
            }
        }
    }
}

/**
 * @brief Calculates the L1 sum of absolute pixel differences (Image Space Error).
 */
double calculate_pixel_error_sum(const Generated_Image obs, const Generated_Image est) {
    double total_error = 0.0;
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            // Sum of absolute differences (L1 norm)
            total_error += fabs(obs[i][j] - est[i][j]);
        }
    }
    return total_error;
}

/**
 * @brief Runs a single optimization of a test image against one ideal template.
 */
void run_optimization(const Generated_Image observed_image, const Feature_Vector observed_features, 
                      int ideal_char_index, EstimationResult *result, int print_trace) {
    
    const Ideal_Curve_Params *ideal_params = &IDEAL_TEMPLATES[ideal_char_index];
    
    // Initialization (MUTABLE data)
    Deformation_Coefficients alpha_hat = {
        .alpha = {0.0, 0.0} // Starting guess: Ideal (zero deformation)
    };
    
    // Dynamic learning rate initialization and floor
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
        // Note: The trace prints FEATURE LOSS only, for comparison.
        printf("    It | Feat Loss| L Rate  | a_1 (Slant) | a_2 (Curve)\n");
        printf("    ------------------------------------------------------\n");
    }

    // Training/Estimation Loop 
    for (int t = 0; t <= ITERATIONS; t++) {
        // Forward Pass: Draw curve -> Extract Features
        draw_curve(alpha_hat.alpha, generated_image, ideal_params);
        extract_geometric_features(generated_image, generated_features);
        
        // Calculate Losses
        current_feature_loss_only = calculate_feature_loss_L2(generated_features, observed_features);
        combined_loss = current_feature_loss_only + PIXEL_LOSS_WEIGHT * calculate_pixel_loss_L2(generated_image, observed_image);

        // Check for bouncing/overshooting and decay learning rate
        if (combined_loss > prev_combined_loss * 1.001 && learning_rate > min_learning_rate) { 
            learning_rate *= 0.5; // Halve the learning rate
        }

        prev_combined_loss = combined_loss;

        // Calculate Gradient (uses COMBINED LOSS)
        calculate_gradient(observed_image, observed_features, &alpha_hat, combined_loss, gradient, ideal_params);
        
        // Print progress only every 500 iterations, and at start/end
        if (print_trace && (t % 500 == 0 || t == ITERATIONS)) {
            printf("    %04d | %8.5f | %7.8f | %8.4f | %8.4f\n", t, current_feature_loss_only, learning_rate, alpha_hat.alpha[0], alpha_hat.alpha[1]);
        }

        // Gradient Descent Update
        if (t < ITERATIONS) {
            double step_rate = (learning_rate > min_learning_rate) ? learning_rate : min_learning_rate;
            
            const double delta_a1 = step_rate * gradient[0];
            const double delta_a2 = step_rate * gradient[1];
            
            alpha_hat.alpha[0] -= delta_a1;
            alpha_hat.alpha[1] -= delta_a2;
        }
    }

    // Store Final Results (Storing only the Feature Loss component)
    result->final_loss = current_feature_loss_only;
    memcpy(result->estimated_alpha, alpha_hat.alpha, sizeof(double) * NUM_DEFORMATIONS);
}

/**
 * @brief Calculates the absolute pixel-wise difference between two images.
 */
void calculate_difference_image(const Generated_Image obs, const Generated_Image est, Generated_Image diff) {
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            // Absolute difference of intensity
            diff[i][j] = fabs(obs[i][j] - est[i][j]);
        }
    }
}

/**
 * @brief Runs one full classification test against all ideal characters.
 */
void run_classification_test(int test_id, int true_char_index, const double true_alpha[NUM_DEFORMATIONS], TestResult *result) {
    result->id = test_id;
    result->true_char_index = true_char_index;
    memcpy(result->true_alpha, true_alpha, sizeof(double) * NUM_DEFORMATIONS);
    const Ideal_Curve_Params *true_params = &IDEAL_TEMPLATES[true_char_index];
    
    // 1. Generate the CLEAN True Image and store it
    generate_target_image(result->true_image, true_alpha, true_params, 0); 
    
    // 2. Generate the NOISY Observed Image and store it
    Generated_Image observed_image; 
    generate_target_image(observed_image, true_alpha, true_params, 1);
    memcpy(result->observed_image, observed_image, sizeof(Generated_Image));
    
    // 3. Extract the TARGET features once
    Feature_Vector observed_features;
    extract_geometric_features(observed_image, observed_features);
    
    printf("\n======================================================\n");
    printf("TEST %02d/%02d (TRUE: '%s'): Slant (a_1)=%.4f, Curve (a_2)=%.4f\n", 
           test_id, NUM_TESTS, CHAR_NAMES[true_char_index], true_alpha[0], true_alpha[1]);

    // 4. Run Optimization against ALL ideal templates
    double min_feature_loss = HUGE_VAL;
    int best_match_index = -1;
    
    for (int i = 0; i < NUM_IDEAL_CHARS; i++) {
        // Run optimization (passing both observed image and features)
        int print_trace = (i == true_char_index && i < 5); 
        run_optimization(observed_image, observed_features, i, &result->classification_results[i], print_trace);

        // Classification is based on the final FEATURE LOSS component
        if (result->classification_results[i].final_loss < min_feature_loss) {
            min_feature_loss = result->classification_results[i].final_loss;
            best_match_index = i;
        }
    }

    result->best_match_index = best_match_index;

    // 5. Finalize the Best Match Images (for summary and SVG)
    EstimationResult *best_fit = &result->classification_results[best_match_index];
    
    // Draw the best estimated image
    draw_curve(best_fit->estimated_alpha, result->best_estimated_image, &IDEAL_TEMPLATES[best_match_index]);
    
    // Calculate difference image
    calculate_difference_image(result->observed_image, result->best_estimated_image, result->best_diff_image);
}

/**
 * @brief Calculates the absolute pixel-wise difference between two images.
 */
void calculate_difference_image_summary(const Generated_Image obs, const Generated_Image est, Generated_Image diff) {
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            // Absolute difference of intensity
            diff[i][j] = fabs(obs[i][j] - est[i][j]);
        }
    }
}

// --- Console Summary Function ---

// The number of tests to show the full loss matrix for (e.g., A, B, C, D, E)
#define DETAILED_SUMMARY_LIMIT 5 

void summarize_results_console() {
    printf("\n\n================================================================================\n");
    printf("               CLASSIFICATION SUMMARY (%d TESTS: A-Z, 0-9)                   \n", NUM_TESTS);
    printf("     (Features: %d Vectors * %d Bins = %d. Loss: L_Feature + %.1f * L_Pixel)     \n", 
           NUM_VECTORS, NUM_BINS, NUM_FEATURES, PIXEL_LOSS_WEIGHT);
    printf("================================================================================\n");
    
    printf("\n--- DETAILED LOSS MATRIX (Showing ALL %d Template Losses for first %d Tests) ---\n", 
           NUM_IDEAL_CHARS, DETAILED_SUMMARY_LIMIT);
    
    // Print header: A | B | C | D | ... | 9
    printf("  ID | TRUE | PRED | Best Feat Loss | Loss against: ");
    for (int i = 0; i < NUM_IDEAL_CHARS; i++) {
        printf(" %s |", CHAR_NAMES[i]);
    }
    printf("\n-----|------|------|----------------|");
    for (int i = 0; i < NUM_IDEAL_CHARS; i++) {
        printf("----|"); 
    }
    printf("\n");
    
    // Print data for the first DETAILED_SUMMARY_LIMIT tests
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

    // Print Concise Summary for all 36 tests
    printf("\n--- CLASSIFICATION PERFORMANCE SUMMARY (All %d Tests) ---\n", NUM_TESTS);
    printf("  ID | TRUE | PRED | Best Feat Loss | a_1 (Slant) | a_2 (Curve) | **PIXEL ERROR %%** | Correct?\n");
    printf("-----|------|------|----------------|-------------|-------------|-------------------|----------\n");

    int correct_classifications = 0;
    for (int k = 0; k < NUM_TESTS; k++) {
        TestResult *r = &all_results[k];
        const EstimationResult *best_fit = &r->classification_results[r->best_match_index];
        
        // Calculate the Pixel Error Percentage (L1-norm)
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


// --- SVG Generation Functions ---

// Constants for SVG rendering
#define PIXEL_SIZE 5    
#define IMG_SIZE (GRID_SIZE * PIXEL_SIZE) // 80
#define IMG_SPACING 5   
// *** ADJUSTED CONSTANTS TO PREVENT TEXT OVERLAP ***
#define SET_SPACING 40  // Increased spacing between sets
#define SET_WIDTH (400) // Explicitly set width to allow for long text label (335 required for images + some buffer)
// *************************************************

// Dimensions calculated for 10 sets in a single row
#define SVG_RENDER_LIMIT 10
#define SVG_WIDTH (SVG_RENDER_LIMIT * SET_WIDTH + (SVG_RENDER_LIMIT) * SET_SPACING) 
#define SVG_HEIGHT (IMG_SIZE + 15 + 2 * SET_SPACING) // Height remains constant

/**
 * @brief Maps an intensity [0.0, 1.0] to an RGB grayscale color string.
 */
void get_grayscale_color(double intensity, char *color_out) {
    double clamped_intensity = fmax(0.0, fmin(1.0, intensity));
    if (clamped_intensity > 0.6) {
        sprintf(color_out, "rgb(255, 255, 100)"); 
    } else if (clamped_intensity > 0.3) {
        sprintf(color_out, "rgb(100, 100, 100)"); 
    } else {
        sprintf(color_out, "rgb(0, 0, 0)"); 
    }
}

/**
 * @brief Renders a single image into the SVG file at a specific offset.
 */
void render_single_image_to_svg(FILE *fp, const Generated_Image img, double x_offset, double y_offset) {
    char color[20];
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            if (img[i][j] > 0.0) {
                get_grayscale_color(img[i][j], color);
                fprintf(fp, "<rect x=\"%.1f\" y=\"%.1f\" width=\"%d\" height=\"%d\" fill=\"%s\"/>\n",
                        x_offset + j * PIXEL_SIZE, 
                        y_offset + i * PIXEL_SIZE, 
                        PIXEL_SIZE, PIXEL_SIZE, color);
            } else {
                 fprintf(fp, "<rect x=\"%.1f\" y=\"%.1f\" width=\"%d\" height=\"%d\" fill=\"black\"/>\n",
                        x_offset + j * PIXEL_SIZE, 
                        y_offset + i * PIXEL_SIZE, 
                        PIXEL_SIZE, PIXEL_SIZE);
            }
        }
    }
}

/**
 * @brief Renders the 4-image comparison set for one test case into the SVG file.
 */
void render_test_to_svg(FILE *fp, const TestResult *r, double x_set, double y_set) {
    // 1. True Image (Clean)
    render_single_image_to_svg(fp, r->true_image, x_set, y_set);
    
    // 2. Target Image (Noisy)
    render_single_image_to_svg(fp, r->observed_image, x_set + IMG_SIZE + IMG_SPACING, y_set);

    const EstimationResult *best_fit = &r->classification_results[r->best_match_index];
    
    // 3. Best Estimated Image (Clean Fit from Best Match)
    render_single_image_to_svg(fp, r->best_estimated_image, x_set + 2 * (IMG_SIZE + IMG_SPACING), y_set);

    // 4. Difference Image (Error Magnitude)
    char error_color[20];
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            double error = r->best_diff_image[i][j];
            if (error > 0.3) {
                sprintf(error_color, "rgb(255, 50, 50)"); // High Error: Red
            } else if (error > 0.1) {
                sprintf(error_color, "rgb(255, 150, 0)"); // Medium Error: Orange
            } else {
                sprintf(error_color, "black");
            }
            fprintf(fp, "<rect x=\"%.1f\" y=\"%.1f\" width=\"%d\" height=\"%d\" fill=\"%s\"/>\n",
                    x_set + 3 * (IMG_SIZE + IMG_SPACING) + j * PIXEL_SIZE, 
                    y_set + i * PIXEL_SIZE, 
                    PIXEL_SIZE, PIXEL_SIZE, error_color);
        }
    }
    
    // Add text label for the test ID and classification result
    char label[150];
    char correct_str[5];
    if (r->true_char_index == r->best_match_index) {
        strcpy(correct_str, "YES");
    } else {
        strcpy(correct_str, "NO");
    }
    
    // Recalculate pixel error for SVG label
    double pixel_error_sum = calculate_pixel_error_sum(r->observed_image, r->best_estimated_image);
    double pixel_error_percent = (pixel_error_sum / MAX_PIXEL_ERROR) * 100.0;

    sprintf(label, "T:'%s' (%.2f,%.2f) | P:'%s' | Error:%.2f%% (%s)", 
            CHAR_NAMES[r->true_char_index], r->true_alpha[0], r->true_alpha[1], 
            CHAR_NAMES[r->best_match_index], pixel_error_percent, correct_str);
    
    fprintf(fp, "<text x=\"%.1f\" y=\"%.1f\" font-family=\"sans-serif\" font-size=\"10\" fill=\"white\">%s</text>\n",
            x_set, y_set + IMG_SIZE + 10, label);
}

/**
 * @brief Generates the final SVG file with the first SVG_RENDER_LIMIT test results.
 */
void generate_svg_file() {
    FILE *fp = fopen("network.svg", "w");
    if (fp == NULL) {
        printf("\nERROR: Could not open network.svg for writing.\n");
        return;
    }

    // SVG Header
    fprintf(fp, "<svg width=\"%d\" height=\"%d\" viewBox=\"0 0 %d %d\" xmlns=\"http://www.w3.org/2000/svg\">\n",
            SVG_WIDTH, SVG_HEIGHT, SVG_WIDTH, SVG_HEIGHT);
    fprintf(fp, "<rect width=\"100%%\" height=\"100%%\" fill=\"black\"/>\n");

    // Add general titles for the 4 image columns
    double image_set_total_width = 4.0 * IMG_SIZE + 3.0 * IMG_SPACING;
    double initial_x = SET_SPACING + IMG_SIZE / 2.0;
    double initial_y = SET_SPACING / 2.0;
    fprintf(fp, "<text x=\"%.1f\" y=\"%.1f\" font-family=\"sans-serif\" font-size=\"12\" fill=\"white\" text-anchor=\"middle\">TRUE CLEAN</text>\n", initial_x, initial_y);
    fprintf(fp, "<text x=\"%.1f\" y=\"%.1f\" font-family=\"sans-serif\" font-size=\"12\" fill=\"white\" text-anchor=\"middle\">OBSERVED NOISY</text>\n", initial_x + IMG_SIZE + IMG_SPACING, initial_y);
    fprintf(fp, "<text x=\"%.1f\" y=\"%.1f\" font-family=\"sans-serif\" font-size=\"12\" fill=\"white\" text-anchor=\"middle\">BEST ESTIMATED</text>\n", initial_x + 2 * (IMG_SIZE + IMG_SPACING), initial_y);
    fprintf(fp, "<text x=\"%.1f\" y=\"%.1f\" font-family=\"sans-serif\" font-size=\"12\" fill=\"white\" text-anchor=\"middle\">ERROR DIFF</text>\n", initial_x + 3 * (IMG_SIZE + IMG_SPACING), initial_y);


    // Render the first 10 test sets in a single row
    int actual_render_limit = (NUM_TESTS < SVG_RENDER_LIMIT) ? NUM_TESTS : SVG_RENDER_LIMIT;
    for (int k = 0; k < actual_render_limit; k++) {
        int col = k; 
        
        // Calculate top-left corner position for the current 4-image set
        // Start X is SET_SPACING + column index * (SET_WIDTH)
        double x_set = SET_SPACING + col * (SET_WIDTH);
        // Offset y to leave room for the title text at the top
        double y_set = SET_SPACING + 15; 
        
        render_test_to_svg(fp, &all_results[k], x_set, y_set);
    }

    // SVG Footer
    fprintf(fp, "</svg>\n");
    
    fclose(fp);
    printf("\n\n======================================================\n");
    printf("SVG Output Complete: network.svg created (First %d tests).\n", actual_render_limit);
    printf("======================================================\n");
}


// --- Main Execution ---

int main(void) {
    srand(42); // Seed for reproducible results

    // Test data: We run one test case for each of the 36 ideal characters, 
    // each with a random deformation.
    const double MIN_ALPHA = -0.15;
    const double MAX_ALPHA = 0.15;

    for (int i = 0; i < NUM_TESTS; i++) {
        double true_alpha[NUM_DEFORMATIONS];
        // Generate random deformation in [-0.15, 0.15]
        true_alpha[0] = MIN_ALPHA + ((double)rand() / RAND_MAX) * (MAX_ALPHA - MIN_ALPHA);
        true_alpha[1] = MIN_ALPHA + ((double)rand() / RAND_MAX) * (MAX_ALPHA - MIN_ALPHA);
        
        // true_char_index is simply i, as the tests are run in the order of the CHAR_NAMES array
        run_classification_test(i + 1, i, true_alpha, &all_results[i]);
    }
    
    // 1. Print the console-based summary
    summarize_results_console();

    // 2. Generate the SVG file (only first 10 tests)
    generate_svg_file();

    return 0;
}
