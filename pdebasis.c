#define _XOPEN_SOURCE // Define this to ensure math constants like M_PI are available

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <string.h> 

// --- Configuration ---
#define N_SAMPLES_MAX 12000 // Increased samples for 3 classes
#define GRID_SIZE 32       
#define D_SIZE (GRID_SIZE * GRID_SIZE) 
#define N_INPUT D_SIZE     
#define N_OUTPUT 14        // [Cls, x, y, w, h, rotation, P1x, P1y, P2x, P2y, P3x, P3y, P4x, P4y]
#define N_HIDDEN 256       
#define N_TEST_SAMPLES 600 // Increased test samples
#define N_REGRESSION_TESTS 30 // 10 samples per class for regression test

// Neural Network Parameters
#define LEARNING_RATE 0.0005 
#define N_EPOCHS_TRAIN 100000 
#define TARGET_LINE_SET 0.0
#define TARGET_RECTANGLE 1.0
#define TARGET_SPLINE 2.0 // New target for 4-point spline
#define CLASSIFICATION_WEIGHT 1.0 
#define REGRESSION_WEIGHT 5.0     
#define MAX_ROTATION_DEGREE 180.0 
// ---------------------

// --- Dynamic Globals ---
int N_SAMPLES = 12000; 
int N_EPOCHS = N_EPOCHS_TRAIN; 
 
// Global Data & Matrices 
double dataset[N_SAMPLES_MAX][D_SIZE]; 
double targets[N_SAMPLES_MAX][N_OUTPUT]; 

// Neural Network Weights and Biases 
double w_ih[N_INPUT][N_HIDDEN]; double b_h[N_HIDDEN]; 
double w_ho[N_HIDDEN][N_OUTPUT]; double b_o[N_OUTPUT];

// Test Data 
double test_data[N_TEST_SAMPLES][D_SIZE];
double test_targets_cls[N_TEST_SAMPLES]; 


// --- Helper Macros ---
#define CLAMP(val, min, max) ((val) < (min) ? (min) : ((val) > (max) ? (max) : (val)))
#define NORMALIZE_COORD(coord) ((double)(coord) / GRID_SIZE)


// --- Function Prototypes ---
void generate_rectangle(double image[D_SIZE], double target_data[N_OUTPUT]);
void generate_random_lines(double image[D_SIZE], double target_data[N_OUTPUT]);
void generate_4_point_spline(double image[D_SIZE], double target_data[N_OUTPUT]);
void load_data_balanced(int n_samples);
void load_balanced_dataset();
void generate_test_set();

void initialize_nn();
void train_nn(const double input_set[N_SAMPLES_MAX][N_INPUT]);
double test_on_set_cls(int n_set_size, const double input_set[][N_INPUT]);
double sigmoid(double x);
void forward_pass(const double input[N_INPUT], double hidden_out[N_HIDDEN], double output[N_OUTPUT]);
void backward_pass_and_update(const double input[N_INPUT], const double hidden_out[N_HIDDEN], const double output[N_OUTPUT], const double target[N_OUTPUT]);

void test_regression();


// -----------------------------------------------------------------
// --- DATA GENERATION FUNCTIONS (UPDATED for SPLINE) ---
// -----------------------------------------------------------------

// Helper function to draw a line for the spline rendering
void draw_line(double image[D_SIZE], int x1, int y1, int x2, int y2, double val) {
    int dx = abs(x2 - x1);
    int dy = abs(y2 - y1);
    int sx = x1 < x2 ? 1 : -1;
    int sy = y1 < y2 ? 1 : -1;
    int err = dx - dy;

    while (1) {
        if (x1 >= 0 && x1 < GRID_SIZE && y1 >= 0 && y1 < GRID_SIZE) {
            image[GRID_SIZE * y1 + x1] = val;
        }
        if (x1 == x2 && y1 == y2) break;
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x1 += sx; }
        if (e2 < dx) { err += dx; y1 += sy; }
    }
}

// Function to generate a 4-point Cubic Bézier Spline
void generate_4_point_spline(double image[D_SIZE], double target_data[N_OUTPUT]) {
    // 1. Clear image
    for (int i = 0; i < D_SIZE; i++) { image[i] = 0.0; } 
    
    // 2. Define 4 random control points (P1 to P4)
    double points[4][2];
    for(int i = 0; i < 4; i++) {
        // Points are restricted to middle 3/4 of grid to ensure visibility
        points[i][0] = 4.0 + (double)(rand() % (GRID_SIZE - 8));
        points[i][1] = 4.0 + (double)(rand() % (GRID_SIZE - 8));
    }
    
    // 3. Draw Spline (Bézier curve approximation)
    double value = 200.0 + (double)(rand() % 50);
    int prev_x = (int)points[0][0];
    int prev_y = (int)points[0][1];
    
    // Use 100 segments for smooth curve
    for (int i = 1; i <= 100; i++) {
        double t = (double)i / 100.0;
        double t2 = t * t;
        double t3 = t2 * t;
        double one_minus_t = 1.0 - t;
        double one_minus_t2 = one_minus_t * one_minus_t;
        double one_minus_t3 = one_minus_t2 * one_minus_t;
        
        // Cubic Bézier formula: B(t) = P1*(1-t)^3 + P2*3t(1-t)^2 + P3*3t^2(1-t) + P4*t^3
        double x = points[0][0] * one_minus_t3 + 
                   points[1][0] * 3 * t * one_minus_t2 + 
                   points[2][0] * 3 * t2 * one_minus_t + 
                   points[3][0] * t3;
        double y = points[0][1] * one_minus_t3 + 
                   points[1][1] * 3 * t * one_minus_t2 + 
                   points[2][1] * 3 * t2 * one_minus_t + 
                   points[3][1] * t3;
                   
        int curr_x = (int)(x + 0.5);
        int curr_y = (int)(y + 0.5);
        
        // Draw segment between previous point and current point
        draw_line(image, prev_x, prev_y, curr_x, curr_y, value);
        prev_x = curr_x;
        prev_y = curr_y;
    }

    // 4. Set Targets (14 outputs)
    target_data[0] = TARGET_SPLINE; // Classification

    // Bounding Box and Rotation are irrelevant/zero for lines/splines
    target_data[1] = target_data[2] = target_data[3] = target_data[4] = 0.0; 
    target_data[5] = 0.0; // Rotation

    // Spline Points (P1x, P1y, ..., P4y)
    for(int i = 0; i < 4; i++) {
        target_data[6 + 2*i + 0] = NORMALIZE_COORD(points[i][0]); // Px
        target_data[6 + 2*i + 1] = NORMALIZE_COORD(points[i][1]); // Py
    }
}

void generate_rectangle(double image[D_SIZE], double target_data[N_OUTPUT]) {
    // 1. Initial Rectangle Parameters
    int rect_w = 8 + (rand() % (GRID_SIZE - 12)); 
    int rect_h = 8 + (rand() % (GRID_SIZE - 12)); 
    int center_x = GRID_SIZE / 2;
    int center_y = GRID_SIZE / 2;
    double rotation_deg = (double)(rand() % 180);
    double rotation_rad = rotation_deg * M_PI / 180.0; 
    double cos_r = cos(rotation_rad);
    double sin_r = sin(rotation_rad);
    
    // 2. Clear Image and Initialize Bounding Box extremes
    for (int i = 0; i < D_SIZE; i++) { image[i] = 0.0; } 
    double min_x = DBL_MAX, min_y = DBL_MAX;
    double max_x = DBL_MIN, max_y = DBL_MIN;
    
    // 3. Draw Rotated Rectangle and Calculate Minimal Bounding Box
    double value = 200.0 + (double)(rand() % 50);

    for (int dy = 0; dy < rect_h; ++dy) {
        for (int dx = 0; dx < rect_w; ++dx) {
            double x_rel = dx - (rect_w / 2.0);
            double y_rel = dy - (rect_h / 2.0);

            double x_rot = x_rel * cos_r - y_rel * sin_r;
            double y_rot = x_rel * sin_r + y_rel * cos_r;

            int x_grid = (int)(center_x + x_rot + 0.5);
            int y_grid = (int)(center_y + y_rot + 0.5);

            if (x_grid >= 0 && x_grid < GRID_SIZE && y_grid >= 0 && y_grid < GRID_SIZE) {
                int index = GRID_SIZE * y_grid + x_grid;
                image[index] = value;
                
                if (x_grid < min_x) min_x = x_grid;
                if (y_grid < min_y) min_y = y_grid;
                if (x_grid > max_x) max_x = x_grid;
                if (y_grid > max_y) max_y = y_grid;
            }
        }
    }
    
    // 4. Calculate Final Bounding Box Dimensions
    int final_start_x = (int)min_x;
    int final_start_y = (int)min_y;
    int final_w = (int)(max_x - min_x + 1);
    int final_h = (int)(max_y - min_y + 1);

    if (final_start_x < 0 || final_start_x >= GRID_SIZE) final_start_x = 0;
    if (final_start_y < 0 || final_start_y >= GRID_SIZE) final_start_y = 0;
    final_w = CLAMP(final_w, 1, GRID_SIZE - final_start_x);
    final_h = CLAMP(final_h, 1, GRID_SIZE - final_start_y);

    // 5. Set Targets (14 outputs)
    target_data[0] = TARGET_RECTANGLE; // Classification
    // Bounding Box and Rotation
    target_data[1] = NORMALIZE_COORD(final_start_x); 
    target_data[2] = NORMALIZE_COORD(final_start_y); 
    target_data[3] = NORMALIZE_COORD(final_w);       
    target_data[4] = NORMALIZE_COORD(final_h);       
    target_data[5] = rotation_deg / MAX_ROTATION_DEGREE; 
    // Spline Points (irrelevant/zero)
    for(int i = 6; i < N_OUTPUT; i++) { target_data[i] = 0.0; }
}

void generate_random_lines(double image[D_SIZE], double target_data[N_OUTPUT]) {
    for (int i = 0; i < D_SIZE; i++) { image[i] = 0.0; } 
    int num_lines = 1 + (rand() % 6); 
    for (int l = 0; l < num_lines; l++) {
        int length_options[] = {4, 8, 16};
        int length = length_options[rand() % 3];
        int x_start = rand() % GRID_SIZE;
        int y_start = rand() % GRID_SIZE;
        int orientation = rand() % 2; 
        double value = 200.0 + (double)(rand() % 50);

        for (int i = 0; i < length; i++) {
            int x = x_start, y = y_start;
            if (orientation == 0) { x = (x_start + i) % GRID_SIZE; } 
            else { y = (y_start + i) % GRID_SIZE; }
            int index = GRID_SIZE * y + x;
            if (index >= 0 && index < D_SIZE) { image[index] = value; }
        }
    }
    
    // Set Targets for Line (14 placeholder values)
    target_data[0] = TARGET_LINE_SET; // Classification
    for(int i = 1; i < N_OUTPUT; i++) { target_data[i] = 0.0; }
}


void load_data_balanced(int n_samples) {
    for (int k = 0; k < n_samples; ++k) {
        if (k % 3 == 0) { 
            generate_random_lines(dataset[k], targets[k]);
        } else if (k % 3 == 1) { 
            generate_rectangle(dataset[k], targets[k]);
        } else {
            generate_4_point_spline(dataset[k], targets[k]);
        }
    }
}
void load_balanced_dataset() {
    printf("Generating BALANCED dataset (%d images): 33.3%% Lines, 33.3%% Rectangles, 33.3%% Splines.\n", N_SAMPLES);
    load_data_balanced(N_SAMPLES);
}
void generate_test_set() {
    printf("Generating CLASSIFICATION TEST dataset (%d images): 1/3 mix.\n", N_TEST_SAMPLES);
    for (int k = 0; k < N_TEST_SAMPLES; ++k) {
        double temp_target[N_OUTPUT];
        if (k % 3 == 0) { 
            generate_random_lines(test_data[k], temp_target);
        } else if (k % 3 == 1) { 
            generate_rectangle(test_data[k], temp_target);
        } else {
            generate_4_point_spline(test_data[k], temp_target);
        }
        test_targets_cls[k] = temp_target[0]; 
    }
}


// -----------------------------------------------------------------
// --- NN CORE FUNCTIONS (UPDATED for N_OUTPUT=14) ---
// -----------------------------------------------------------------

void initialize_nn() {
    for (int i = 0; i < N_INPUT; i++) {
        for (int j = 0; j < N_HIDDEN; j++) {
            w_ih[i][j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * 0.1;
        }
    }
    for (int j = 0; j < N_HIDDEN; j++) {
        b_h[j] = 0.0;
        for (int k = 0; k < N_OUTPUT; k++) {
            w_ho[j][k] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * 0.1;
        }
    }
    for (int k = 0; k < N_OUTPUT; k++) {
        b_o[k] = 0.0;
    }
}

void train_nn(const double input_set[N_SAMPLES_MAX][N_INPUT]) {
    printf("Training on raw %d-dimensional image pixels with 14-output regression...\n", N_INPUT);
    for (int epoch = 0; epoch < N_EPOCHS; epoch++) {
        int sample_index = rand() % N_SAMPLES;
        
        double hidden_out[N_HIDDEN];
        double output[N_OUTPUT];
        
        forward_pass(input_set[sample_index], hidden_out, output);
        backward_pass_and_update(input_set[sample_index], hidden_out, output, targets[sample_index]);

        if (N_EPOCHS > 1000 && (epoch % (N_EPOCHS / 10) == 0) && epoch != 0) {
            printf("  Epoch %d/%d completed.\n", epoch, N_EPOCHS);
        }
    }
}

double sigmoid(double x) { 
    return 1.0 / (1.0 + exp(-x)); 
}

void forward_pass(const double input[N_INPUT], double hidden_out[N_HIDDEN], double output[N_OUTPUT]) {
    // Input to Hidden Layer (Sigmoid activation)
    for (int j = 0; j < N_HIDDEN; j++) {
        double h_net = b_h[j];
        for (int i = 0; i < N_INPUT; i++) {
            h_net += input[i] * w_ih[i][j];
        }
        hidden_out[j] = sigmoid(h_net);
    }
    
    // Hidden to Output Layer
    for (int k = 0; k < N_OUTPUT; k++) {
        double o_net = b_o[k]; 
        for (int j = 0; j < N_HIDDEN; j++) { 
            o_net += hidden_out[j] * w_ho[j][k]; 
        } 
        
        // Output[0] (Classification) uses linear output for multi-target regression.
        // The interpretation of this score determines the class.
        // Regression outputs [1..13] use identity (linear)
        output[k] = o_net; 
    }
}

void backward_pass_and_update(const double input[N_INPUT], const double hidden_out[N_HIDDEN], const double output[N_OUTPUT], const double target[N_OUTPUT]) {
    double delta_o[N_OUTPUT];
    double delta_h[N_HIDDEN]; 
    double error_h[N_HIDDEN] = {0.0};
    
    // 1. Output Layer Deltas 
    for (int k = 0; k < N_OUTPUT; k++) {
        double error = output[k] - target[k];
        double weight = REGRESSION_WEIGHT; // Default weight for all regression outputs

        if (k == 0) { 
            // Classification output: Use Classification Weight
            weight = CLASSIFICATION_WEIGHT;
        } else {
            // Regression outputs (1..13): Apply loss only if the shape is RELEVANT
            double target_cls = target[0];
            
            // Outputs [1..5] (Rect: x, y, w, h, rotation) relevant only for TARGET_RECTANGLE (1.0)
            if (k >= 1 && k <= 5) {
                if (fabs(target_cls - TARGET_RECTANGLE) > DBL_EPSILON) {
                    weight = 0.0;
                }
            }
            // Outputs [6..13] (Spline: Px, Py...) relevant only for TARGET_SPLINE (2.0)
            else if (k >= 6 && k <= 13) {
                if (fabs(target_cls - TARGET_SPLINE) > DBL_EPSILON) {
                    weight = 0.0;
                }
            }
            // Lines (0.0) have no active regression outputs, weight remains 0.0 after check
        }
        
        // Output layer uses linear activation, so derivative is 1.0
        delta_o[k] = error * weight; 
    }
    
    // 2. Hidden Layer Deltas 
    for (int j = 0; j < N_HIDDEN; j++) { 
        for (int k = 0; k < N_OUTPUT; k++) {
            error_h[j] += delta_o[k] * w_ho[j][k];
        }
        // Sigmoid derivative for hidden layer
        delta_h[j] = error_h[j] * hidden_out[j] * (1.0 - hidden_out[j]);
    }
    
    // 3. Update Weights and Biases (Hidden-to-Output)
    for (int k = 0; k < N_OUTPUT; k++) { 
        for (int j = 0; j < N_HIDDEN; j++) { 
            w_ho[j][k] -= LEARNING_RATE * delta_o[k] * hidden_out[j]; 
        } 
        b_o[k] -= LEARNING_RATE * delta_o[k];
    } 
    
    // 4. Update Weights and Biases (Input-to-Hidden)
    for (int i = 0; i < N_INPUT; i++) { 
        for (int j = 0; j < N_HIDDEN; j++) { 
            w_ih[i][j] -= LEARNING_RATE * delta_h[j] * input[i]; 
        } 
    }
    for (int j = 0; j < N_HIDDEN; j++) { 
        b_h[j] -= LEARNING_RATE * delta_h[j]; 
    }
}

// Multi-class classification test (nearest target)
double test_on_set_cls(int n_set_size, const double input_set[][N_INPUT]) {
    int correct_predictions = 0; 
    double hidden_out[N_HIDDEN]; 
    double output[N_OUTPUT];
    
    for (int i = 0; i < n_set_size; i++) {
        forward_pass(input_set[i], hidden_out, output);
        
        double cls_score = output[0];
        double actual = test_targets_cls[i];
        
        // Find nearest target class (0.0, 1.0, or 2.0)
        double prediction = round(cls_score);
        prediction = CLAMP(prediction, TARGET_LINE_SET, TARGET_SPLINE);

        if (fabs(prediction - actual) < DBL_EPSILON) { 
            correct_predictions++; 
        }
    }
    return (double)correct_predictions / n_set_size;
}

// -----------------------------------------------------------------
// --- REGRESSION TESTING (UPDATED for Splines) ---
// -----------------------------------------------------------------

// Helper function to print a known class name based on its target value
const char* get_class_name(double target_cls) {
    if (fabs(target_cls - TARGET_RECTANGLE) < DBL_EPSILON) return "RECTANGLE";
    if (fabs(target_cls - TARGET_SPLINE) < DBL_EPSILON) return "SPLINE";
    return "LINE";
}

void test_regression() {
    printf("\n--- STEP 3: REGRESSION TEST (%d Samples: Rects/Splines) ---\n", N_REGRESSION_TESTS);
    printf("Image dimensions: %dx%d pixels. Cls Targets: 0.0=Line, 1.0=Rect, 2.0=Spline.\n", GRID_SIZE, GRID_SIZE);
    printf("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    printf("| # | Cls | Est. Cls | Est. X, Y, W, H (Px) | Known X, Y, W, H | Est. Rot (Deg) | Known Rot (Deg) | Estimated Spline Points (Px) | Known Spline Points (Px) |\n");
    printf("|---|-----|----------|----------------------|------------------|----------------|-----------------|------------------------------|--------------------------|\n");

    double hidden_out[N_HIDDEN];
    double output[N_OUTPUT];
    
    // Test 10 lines, 10 rectangles, 10 splines
    for (int i = 0; i < N_REGRESSION_TESTS; i++) {
        double test_image[D_SIZE];
        double known_target[N_OUTPUT];
        
        // Generate test data for 3 classes (0=Line, 1=Rect, 2=Spline)
        if (i < N_REGRESSION_TESTS/3) {
             generate_random_lines(test_image, known_target); // Lines
        } else if (i < 2 * N_REGRESSION_TESTS/3) {
            generate_rectangle(test_image, known_target); // Rectangles
        } else {
            generate_4_point_spline(test_image, known_target); // Splines
        }
        
        forward_pass(test_image, hidden_out, output);
        
        double target_cls = known_target[0];

        // Clamp outputs for display
        int est_x = (int)(CLAMP(output[1], 0.0, 1.0) * GRID_SIZE + 0.5);
        int est_y = (int)(CLAMP(output[2], 0.0, 1.0) * GRID_SIZE + 0.5);
        int est_w = (int)(CLAMP(output[3], 0.0, 1.0) * GRID_SIZE + 0.5);
        int est_h = (int)(CLAMP(output[4], 0.0, 1.0) * GRID_SIZE + 0.5);
        
        double est_rot_norm = CLAMP(output[5], 0.0, 1.0);
        double est_rot_deg = est_rot_norm * MAX_ROTATION_DEGREE;

        // Known values (denormalized)
        int known_x = (int)(known_target[1] * GRID_SIZE + 0.5);
        int known_y = (int)(known_target[2] * GRID_SIZE + 0.5);
        int known_w = (int)(known_target[3] * GRID_SIZE + 0.5);
        int known_h = (int)(known_target[4] * GRID_SIZE + 0.5);
        double known_rot_deg = known_target[5] * MAX_ROTATION_DEGREE;
        
        // Spline Points
        char est_spline_str[64];
        char known_spline_str[64];
        
        if (fabs(target_cls - TARGET_SPLINE) < DBL_EPSILON) {
            // Only print if it's actually a spline
            snprintf(est_spline_str, 64, "%2d,%2d %2d,%2d %2d,%2d %2d,%2d",
                     (int)(CLAMP(output[6], 0.0, 1.0) * GRID_SIZE + 0.5), (int)(CLAMP(output[7], 0.0, 1.0) * GRID_SIZE + 0.5),
                     (int)(CLAMP(output[8], 0.0, 1.0) * GRID_SIZE + 0.5), (int)(CLAMP(output[9], 0.0, 1.0) * GRID_SIZE + 0.5),
                     (int)(CLAMP(output[10], 0.0, 1.0) * GRID_SIZE + 0.5), (int)(CLAMP(output[11], 0.0, 1.0) * GRID_SIZE + 0.5),
                     (int)(CLAMP(output[12], 0.0, 1.0) * GRID_SIZE + 0.5), (int)(CLAMP(output[13], 0.0, 1.0) * GRID_SIZE + 0.5));
            snprintf(known_spline_str, 64, "%2d,%2d %2d,%2d %2d,%2d %2d,%2d",
                     (int)(known_target[6] * GRID_SIZE + 0.5), (int)(known_target[7] * GRID_SIZE + 0.5),
                     (int)(known_target[8] * GRID_SIZE + 0.5), (int)(known_target[9] * GRID_SIZE + 0.5),
                     (int)(known_target[10] * GRID_SIZE + 0.5), (int)(known_target[11] * GRID_SIZE + 0.5),
                     (int)(known_target[12] * GRID_SIZE + 0.5), (int)(known_target[13] * GRID_SIZE + 0.5));
        } else {
            // Print N/A for lines and rectangles
            snprintf(est_spline_str, 64, "N/A");
            snprintf(known_spline_str, 64, "N/A");
        }
        
        printf("| %1d | %s | %8.4f | %2d,%2d,%2d,%2d | %2d,%2d,%2d,%2d | %14.1f | %15.1f | %28s | %24s |\n",
               i + 1,
               get_class_name(target_cls),
               output[0],
               est_x, est_y, est_w, est_h,
               known_x, known_y, known_w, known_h,
               est_rot_deg, known_rot_deg,
               est_spline_str, known_spline_str);
    }
    printf("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
}

// -----------------------------------------------------------------
// --- MAIN EXECUTION ---
// -----------------------------------------------------------------
int main() {
    srand(time(NULL));
    
    clock_t start_total, end_total;
    start_total = clock();

    load_balanced_dataset(); 
    generate_test_set();
    estimate_nn_epochs();

    printf("\n--- GLOBAL CONFIGURATION ---\n");
    printf("Model: Classification + 13-Output Regression (Rect Bounding Box + Spline Points)\n");
    printf("Train Samples: %d | Input Dim: %d | Hidden Dim: %d | Output Dim: %d\n", N_SAMPLES, N_INPUT, N_HIDDEN, N_OUTPUT);
    printf("Learning Rate: %.4f | Classification Weight: %.1f | Regression Weight: %.1f\n", LEARNING_RATE, CLASSIFICATION_WEIGHT, REGRESSION_WEIGHT);

    // --- STEP 1: NN Training ---
    printf("\n--- STEP 1: NN Training ---\n");
    clock_t start_nn = clock();
    initialize_nn();
    train_nn(dataset);
    clock_t end_nn = clock();
    printf("NN Training time: %.4f seconds.\n", (double)(end_nn - start_nn) / CLOCKS_PER_SEC);

    // --- STEP 2: Standard Classification Testing ---
    printf("\n--- STEP 2: Standard Classification Testing Results ---\n");
    
    double acc_test = test_on_set_cls(N_TEST_SAMPLES, test_data);
    printf("NN Testing Accuracy (Line/Rect/Spline): %.2f%%\n", acc_test * 100.0);
    
    // --- STEP 3: Regression Testing ---
    test_regression();
    
    end_total = clock();
    printf("\nTotal execution time: %.4f seconds.\n", (double)(end_total - start_total) / CLOCKS_PER_SEC);

    return 0;
}
