#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <string.h> 

// --- Configuration ---
#define N_SAMPLES_MAX 50000 // Increased training size
#define D_SIZE 256         // 16x16 image size (RAW INPUT DIMENSION)
#define GRID_SIZE 16       // Image grid size (16x16)
#define N_INPUT D_SIZE     // NN Input Dimension
#define N_OUTPUT 5         // NN Output: [Classification, x, y, w, h]
#define N_HIDDEN 64        // Increased hidden layer size
#define N_TEST_SAMPLES 500 // Standard test set size
#define N_REGRESSION_TESTS 50 // New regression test size

// Time limit in seconds
#define MAX_TIME_NN_SEC 120.0

// Neural Network Parameters
#define LEARNING_RATE 0.005 // Further reduced learning rate for complex task
#define N_EPOCHS_MAX 10000 
#define TARGET_RECTANGLE 1.0
#define TARGET_LINE_SET 0.0
#define CLASSIFICATION_WEIGHT 1.0 // Weight for classification loss
#define REGRESSION_WEIGHT 0.5     // Weight for bounding box loss
// ---------------------

// --- Dynamic Globals ---
int N_SAMPLES = 50000; 
int N_EPOCHS;  

// Global Data & Matrices
double dataset[N_SAMPLES_MAX][D_SIZE];  // Raw Image Data (NN input)
// targets structure: [Classification, x_norm, y_norm, w_norm, h_norm]
double targets[N_SAMPLES_MAX][N_OUTPUT]; 

// Neural Network Weights and Biases 
double w_ih[N_INPUT][N_HIDDEN]; double b_h[N_HIDDEN]; 
double w_ho[N_HIDDEN][N_OUTPUT]; double b_o[N_OUTPUT];

// Test Data (Standard Classification Test Set)
double test_data[N_TEST_SAMPLES][D_SIZE];
double test_targets_cls[N_TEST_SAMPLES]; // Only used for classification test

// -----------------------------------------------------------------
// --- FUNCTION PROTOTYPES ---
// -----------------------------------------------------------------

// Data Generation
void generate_rectangle(double image[D_SIZE], double target_data[N_OUTPUT]);
void generate_random_lines(double image[D_SIZE], double target_data[N_OUTPUT]);
void load_data_balanced(int n_samples);
void load_subset_for_profiling(int n_subset);
void load_balanced_dataset();
void generate_test_set();

// Profiling
void estimate_nn_epochs();

// NN Core Functions
void initialize_nn();
void train_nn(const double input_set[N_SAMPLES_MAX][N_INPUT]);
double test_on_set_cls(int n_set_size, const double input_set[][N_INPUT]);
double sigmoid(double x);
void forward_pass(const double input[N_INPUT], double hidden_out[N_HIDDEN], double output[N_OUTPUT]);
void backward_pass_and_update(const double input[N_INPUT], const double hidden_out[N_HIDDEN], const double output[N_OUTPUT], const double target[N_OUTPUT]);

// New Test Functions
void test_regression();
void generate_test_rectangle(double image[D_SIZE], double target_data[N_OUTPUT]);

// -----------------------------------------------------------------
// --- DATA GENERATION FUNCTIONS (Updated for Regression Targets) ---
// -----------------------------------------------------------------

void generate_rectangle(double image[D_SIZE], double target_data[N_OUTPUT]) {
    // Determine random rectangle dimensions and position (in pixel counts, 1 to 16)
    int rect_w = 4 + (rand() % 8);
    int rect_h = 4 + (rand() % 8);
    int start_x = rand() % (GRID_SIZE - rect_w);
    int start_y = rand() % (GRID_SIZE - rect_h);
    
    // Initialize image to black
    for (int i = 0; i < D_SIZE; i++) { image[i] = 0.0; } 
    
    // Draw rectangle
    for (int y = start_y; y < start_y + rect_h; ++y) {
        for (int x = start_x; x < start_x + rect_w; ++x) {
            image[GRID_SIZE * y + x] = 200.0 + (double)(rand() % 50);
        }
    }

    // Set Targets (Normalized to 0.0 - 1.0)
    target_data[0] = TARGET_RECTANGLE; // Classification
    target_data[1] = (double)start_x / GRID_SIZE; // x_norm
    target_data[2] = (double)start_y / GRID_SIZE; // y_norm
    target_data[3] = (double)rect_w / GRID_SIZE;  // w_norm
    target_data[4] = (double)rect_h / GRID_SIZE;  // h_norm
}

void generate_random_lines(double image[D_SIZE], double target_data[N_OUTPUT]) {
    // Draw lines (similar to previous version)
    for (int i = 0; i < D_SIZE; i++) { image[i] = 0.0; } 
    int num_lines = 1 + (rand() % 4); 
    for (int l = 0; l < num_lines; l++) {
        int length_options[] = {2, 4, 8};
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
    
    // Set Targets for Non-Rectangle (Classification + placeholder regression values)
    target_data[0] = TARGET_LINE_SET; // Classification
    target_data[1] = target_data[2] = target_data[3] = target_data[4] = 0.0; 
}

void load_data_balanced(int n_samples) {
    for (int k = 0; k < n_samples; ++k) {
        if (k % 2 == 0) { 
            generate_rectangle(dataset[k], targets[k]);
        } else { 
            generate_random_lines(dataset[k], targets[k]);
        }
    }
}
void load_subset_for_profiling(int n_subset) {
    // Profiling uses only a small subset of the training set
    for (int k = 0; k < n_subset; ++k) {
        if (k % 2 == 0) { 
            generate_rectangle(dataset[k], targets[k]);
        } else { 
            generate_random_lines(dataset[k], targets[k]);
        }
    }
}
void load_balanced_dataset() {
    printf("Generating BALANCED dataset (%d images): 50%% Rectangles, 50%% Random Lines.\n", N_SAMPLES);
    load_data_balanced(N_SAMPLES);
}
void generate_test_set() {
    printf("Generating CLASSIFICATION TEST dataset (%d images): 50/50 mix.\n", N_TEST_SAMPLES);
    for (int k = 0; k < N_TEST_SAMPLES; ++k) {
        double temp_target[N_OUTPUT];
        if (k % 2 == 0) { 
            generate_rectangle(test_data[k], temp_target);
        } else { 
            generate_random_lines(test_data[k], temp_target);
        }
        test_targets_cls[k] = temp_target[0]; // Only store classification target
    }
}
void generate_test_rectangle(double image[D_SIZE], double target_data[N_OUTPUT]) {
    // A wrapper to generate a rectangle for the regression test
    generate_rectangle(image, target_data);
}

// -----------------------------------------------------------------
// --- PROFILING FUNCTIONS ---
// -----------------------------------------------------------------

void estimate_nn_epochs() {
    clock_t start, end;
    #define N_EPOCHS_PROFILE 100
    
    // Only profile training time, not the heavy data loading
    for (int k = 0; k < N_SAMPLES; k++) {
        if (k < 50) load_subset_for_profiling(50); 
    }
    
    initialize_nn(); 

    start = clock();
    for (int epoch = 0; epoch < N_EPOCHS_PROFILE; epoch++) {
        int sample_index = rand() % 50; 
        double hidden_out[N_HIDDEN]; 
        double output[N_OUTPUT];

        forward_pass(dataset[sample_index], hidden_out, output);
        backward_pass_and_update(dataset[sample_index], hidden_out, output, targets[sample_index]);
    }
    end = clock();
    double time_spent_profile = (double)(end - start) / CLOCKS_PER_SEC;

    if (time_spent_profile < 1e-6) time_spent_profile = 1e-6;

    double epoch_scale_factor = MAX_TIME_NN_SEC / time_spent_profile;
    N_EPOCHS = (int)(N_EPOCHS_PROFILE * epoch_scale_factor);
    
    if (N_EPOCHS > N_EPOCHS_MAX) N_EPOCHS = N_EPOCHS_MAX;

    printf("\n--- NN EPOCHS TIME PROFILING (Input %d, Hidden %d, Output %d) ---\n", N_INPUT, N_HIDDEN, N_OUTPUT);
    printf("Profile (%d epochs): %.4f sec\n", N_EPOCHS_PROFILE, time_spent_profile);
    printf("Estimated Epochs for %.1f sec limit: %d (Using N_EPOCHS=%d)\n", MAX_TIME_NN_SEC, (int)(N_EPOCHS_PROFILE * epoch_scale_factor), N_EPOCHS);
}

// -----------------------------------------------------------------
// --- NN CORE FUNCTIONS (Updated for 5-Output Regression) ---
// -----------------------------------------------------------------

void initialize_nn() {
    // Input-to-Hidden
    for (int i = 0; i < N_INPUT; i++) {
        for (int j = 0; j < N_HIDDEN; j++) {
            w_ih[i][j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * 0.1;
        }
    }
    for (int j = 0; j < N_HIDDEN; j++) {
        b_h[j] = 0.0;
        // Hidden-to-Output
        for (int k = 0; k < N_OUTPUT; k++) {
            w_ho[j][k] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * 0.1;
        }
    }
    for (int k = 0; k < N_OUTPUT; k++) {
        b_o[k] = 0.0;
    }
}

void train_nn(const double input_set[N_SAMPLES_MAX][N_INPUT]) {
    printf("Training on raw %d-dimensional image pixels with 5-output regression...\n", N_INPUT);
    for (int epoch = 0; epoch < N_EPOCHS; epoch++) {
        int sample_index = rand() % N_SAMPLES;
        
        double hidden_out[N_HIDDEN];
        double output[N_OUTPUT];
        
        forward_pass(input_set[sample_index], hidden_out, output);
        backward_pass_and_update(input_set[sample_index], hidden_out, output, targets[sample_index]);
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
        
        // Output[0] (Classification) uses Sigmoid
        if (k == 0) {
            output[k] = sigmoid(o_net);
        } else {
            // Outputs [1..4] (Regression) use identity (no activation)
            output[k] = o_net;
        }
    }
}

void backward_pass_and_update(const double input[N_INPUT], const double hidden_out[N_HIDDEN], const double output[N_OUTPUT], const double target[N_OUTPUT]) {
    double delta_o[N_OUTPUT];
    double error_h[N_HIDDEN] = {0.0};
    
    // 1. Output Layer Deltas
    for (int k = 0; k < N_OUTPUT; k++) {
        double error = output[k] - target[k];
        double weight = CLASSIFICATION_WEIGHT;
        
        if (k == 0) { 
            // Classification loss (Cross-Entropy/Sigmoid approximation)
            delta_o[k] = error * output[k] * (1.0 - output[k]) * weight;
        } else {
            // Regression loss (MSE)
            // Apply regression loss ONLY if the image is a rectangle (target[0] is 1.0)
            if (fabs(target[0] - TARGET_RECTANGLE) < DBL_EPSILON) {
                weight = REGRESSION_WEIGHT;
            } else {
                weight = 0.0; // Skip regression loss for line images
            }
            delta_o[k] = error * weight; // For identity activation, d(L)/d(net) is just error * weight
        }
    }
    
    // 2. Hidden Layer Deltas
    for (int j = 0; j < N_HIDDEN; j++) { 
        for (int k = 0; k < N_OUTPUT; k++) {
            error_h[j] += delta_o[k] * w_ho[j][k];
        }
        delta_o[j] = error_h[j] * hidden_out[j] * (1.0 - hidden_out[j]); // Sigmoid derivative
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
            w_ih[i][j] -= LEARNING_RATE * delta_o[j] * input[i]; 
        } 
    }
    for (int j = 0; j < N_HIDDEN; j++) { 
        b_h[j] -= LEARNING_RATE * delta_o[j]; 
    }
}

double test_on_set_cls(int n_set_size, const double input_set[][N_INPUT]) {
    int correct_predictions = 0; 
    double hidden_out[N_HIDDEN]; 
    double output[N_OUTPUT];
    
    for (int i = 0; i < n_set_size; i++) {
        forward_pass(input_set[i], hidden_out, output);
        
        // Only check classification (output[0]) against the stored classification target
        double prediction = (output[0] >= 0.5) ? TARGET_RECTANGLE : TARGET_LINE_SET;
        double actual = test_targets_cls[i];

        if (fabs(prediction - actual) < DBL_EPSILON) { 
            correct_predictions++; 
        }
    }
    return (double)correct_predictions / n_set_size;
}

// -----------------------------------------------------------------
// --- REGRESSION TESTING ---
// -----------------------------------------------------------------

void test_regression() {
    printf("\n--- STEP 3: REGRESSION TEST (50 Random Rectangles) ---\n");
    printf("Image dimensions are 16x16 pixels.\n");
    printf("--------------------------------------------------------------------------------------\n");
    printf("| # | Cls Score | Est. X, Y, W, H (Norm) | Est. X, Y, W, H (Pixels) | Known X, Y, W, H |\n");
    printf("|---|-----------|------------------------|--------------------------|------------------|\n");

    double hidden_out[N_HIDDEN];
    double output[N_OUTPUT];
    
    for (int i = 0; i < N_REGRESSION_TESTS; i++) {
        double test_image[D_SIZE];
        double known_target[N_OUTPUT];
        
        // Generate a new rectangle image and get its true normalized dimensions
        generate_test_rectangle(test_image, known_target);
        
        forward_pass(test_image, hidden_out, output);
        
        // Denormalize known and estimated dimensions
        int known_x = (int)(known_target[1] * GRID_SIZE + 0.5);
        int known_y = (int)(known_target[2] * GRID_SIZE + 0.5);
        int known_w = (int)(known_target[3] * GRID_SIZE + 0.5);
        int known_h = (int)(known_target[4] * GRID_SIZE + 0.5);

        int est_x = (int)(output[1] * GRID_SIZE + 0.5);
        int est_y = (int)(output[2] * GRID_SIZE + 0.5);
        int est_w = (int)(output[3] * GRID_SIZE + 0.5);
        int est_h = (int)(output[4] * GRID_SIZE + 0.5);
        
        printf("| %1d | %9.4f | %0.2f, %0.2f, %0.2f, %0.2f | %2d, %2d, %2d, %2d | %2d, %2d, %2d, %2d |\n",
               i + 1,
               output[0],
               output[1], output[2], output[3], output[4],
               est_x, est_y, est_w, est_h,
               known_x, known_y, known_w, known_h);
    }
    printf("--------------------------------------------------------------------------------------\n");
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
    printf("Model: Classification + Regression NN\n");
    printf("Train Samples: %d | Input Dim: %d | Hidden Dim: %d | Output Dim: %d\n", N_SAMPLES, N_INPUT, N_HIDDEN, N_OUTPUT);

    // --- STEP 1: NN Training ---
    printf("\n--- STEP 1: NN Training ---\n");
    clock_t start_nn = clock();
    initialize_nn();
    train_nn(dataset);
    clock_t end_nn = clock();
    printf("NN Training time: %.4f seconds.\n", (double)(end_nn - start_nn) / CLOCKS_PER_SEC);

    // --- STEP 2: Standard Classification Testing ---
    printf("\n--- STEP 2: Standard Classification Testing Results ---\n");
    
    // Test on Unseen Classification Test Set
    double acc_test = test_on_set_cls(N_TEST_SAMPLES, test_data);
    printf("NN Testing Accuracy (Rect/Line): %.2f%%\n", acc_test * 100.0);
    
    // --- STEP 3: Regression Testing ---
    test_regression();
    
    end_total = clock();
    printf("\nTotal execution time (including profiling): %.4f seconds.\n", (double)(end_total - start_total) / CLOCKS_PER_SEC);

    return 0;
}
