#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <string.h> 

// --- Configuration ---
#define N_SAMPLES_MAX 150000 // Training size
#define D_SIZE 256         // 16x16 image size (RAW INPUT DIMENSION)
#define GRID_SIZE 16       // Image grid size (16x16)
#define N_INPUT D_SIZE     // NN Input Dimension
#define N_OUTPUT 5         // NN Output: [Classification, x, y, w, h]
#define N_HIDDEN 64        // Hidden layer size
#define N_TEST_SAMPLES 45000 // Standard test set size
#define N_REGRESSION_TESTS 500 // Regression test size

// Neural Network Parameters
#define LEARNING_RATE 0.002 
#define N_EPOCHS_TRAIN 250000 // FIXED: Increased training epochs significantly
#define TARGET_RECTANGLE 1.0
#define TARGET_LINE_SET 0.0
#define CLASSIFICATION_WEIGHT 1.0 
#define REGRESSION_WEIGHT 2.0     // INCREASED: Prioritizing bounding box loss
// ---------------------

// --- Dynamic Globals ---
int N_SAMPLES = 50000; 
int N_EPOCHS = N_EPOCHS_TRAIN; // Use the fixed large value
 
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


// --- Function Prototypes ---
void generate_rectangle(double image[D_SIZE], double target_data[N_OUTPUT]);
void generate_random_lines(double image[D_SIZE], double target_data[N_OUTPUT]);
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
void generate_test_rectangle(double image[D_SIZE], double target_data[N_OUTPUT]);

// -----------------------------------------------------------------
// --- DATA GENERATION FUNCTIONS (Unchanged logic) ---
// -----------------------------------------------------------------

void generate_rectangle(double image[D_SIZE], double target_data[N_OUTPUT]) {
    int rect_w = 4 + (rand() % 8);
    int rect_h = 4 + (rand() % 8);
    int start_x = rand() % (GRID_SIZE - rect_w);
    int start_y = rand() % (GRID_SIZE - rect_h);
    
    for (int i = 0; i < D_SIZE; i++) { image[i] = 0.0; } 
    
    for (int y = start_y; y < start_y + rect_h; ++y) {
        for (int x = start_x; x < start_x + rect_w; ++x) {
            image[GRID_SIZE * y + x] = 200.0 + (double)(rand() % 50);
        }
    }

    target_data[0] = TARGET_RECTANGLE; 
    target_data[1] = (double)start_x / GRID_SIZE; 
    target_data[2] = (double)start_y / GRID_SIZE; 
    target_data[3] = (double)rect_w / GRID_SIZE;  
    target_data[4] = (double)rect_h / GRID_SIZE;  
}

void generate_random_lines(double image[D_SIZE], double target_data[N_OUTPUT]) {
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
    
    target_data[0] = TARGET_LINE_SET; 
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
        test_targets_cls[k] = temp_target[0]; 
    }
}
void generate_test_rectangle(double image[D_SIZE], double target_data[N_OUTPUT]) {
    generate_rectangle(image, target_data);
}

// -----------------------------------------------------------------
// --- PROFILING FUNCTIONS (Simplified/Removed Time Limit) ---
// -----------------------------------------------------------------

void estimate_nn_epochs() {
    printf("\n--- NN EPOCHS CONFIGURATION ---\n");
    printf("Setting fixed epochs to N_EPOCHS=%d for convergence of complex regression task.\n", N_EPOCHS);
}

// -----------------------------------------------------------------
// --- NN CORE FUNCTIONS (Bug fix maintained, Logic updated) ---
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

        // Simple progress indicator for long training
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
        
        // Output[0] (Classification) uses Sigmoid
        if (k == 0) {
            output[k] = sigmoid(o_net);
        } else {
            // Outputs [1..4] (Regression) use identity (linear)
            output[k] = o_net;
        }
    }
}

void backward_pass_and_update(const double input[N_INPUT], const double hidden_out[N_HIDDEN], const double output[N_OUTPUT], const double target[N_OUTPUT]) {
    double delta_o[N_OUTPUT];
    double delta_h[N_HIDDEN]; 
    double error_h[N_HIDDEN] = {0.0};
    
    // 1. Output Layer Deltas (delta_o calculated)
    for (int k = 0; k < N_OUTPUT; k++) {
        double error = output[k] - target[k];
        double weight = CLASSIFICATION_WEIGHT;
        
        if (k == 0) { 
            delta_o[k] = error * output[k] * (1.0 - output[k]) * weight;
        } else {
            // Apply regression loss ONLY if the image is a rectangle (target[0] is 1.0)
            if (fabs(target[0] - TARGET_RECTANGLE) < DBL_EPSILON) {
                weight = REGRESSION_WEIGHT; // Use increased regression weight
            } else {
                weight = 0.0; 
            }
            delta_o[k] = error * weight; 
        }
    }
    
    // 2. Hidden Layer Deltas (delta_h calculated)
    for (int j = 0; j < N_HIDDEN; j++) { 
        for (int k = 0; k < N_OUTPUT; k++) {
            error_h[j] += delta_o[k] * w_ho[j][k];
        }
        delta_h[j] = error_h[j] * hidden_out[j] * (1.0 - hidden_out[j]); // Sigmoid derivative
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

double test_on_set_cls(int n_set_size, const double input_set[][N_INPUT]) {
    int correct_predictions = 0; 
    double hidden_out[N_HIDDEN]; 
    double output[N_OUTPUT];
    
    for (int i = 0; i < n_set_size; i++) {
        forward_pass(input_set[i], hidden_out, output);
        
        double prediction = (output[0] >= 0.5) ? TARGET_RECTANGLE : TARGET_LINE_SET;
        double actual = test_targets_cls[i];

        if (fabs(prediction - actual) < DBL_EPSILON) { 
            correct_predictions++; 
        }
    }
    return (double)correct_predictions / n_set_size;
}

// -----------------------------------------------------------------
// --- REGRESSION TESTING (Updated to clamp output) ---
// -----------------------------------------------------------------

void test_regression() {
    printf("\n--- STEP 3: REGRESSION TEST (%d Random Rectangles) ---\n", N_REGRESSION_TESTS);
    printf("Image dimensions are %dx%d pixels.\n", GRID_SIZE, GRID_SIZE);
    printf("--------------------------------------------------------------------------------------\n");
    printf("| # | Cls Score | Est. X, Y, W, H (Norm) | Est. X, Y, W, H (Pixels) | Known X, Y, W, H |\n");
    printf("|---|-----------|------------------------|--------------------------|------------------|\n");

    double hidden_out[N_HIDDEN];
    double output[N_OUTPUT];
    
    for (int i = 0; i < N_REGRESSION_TESTS; i++) {
        double test_image[D_SIZE];
        double known_target[N_OUTPUT];
        
        generate_test_rectangle(test_image, known_target);
        
        forward_pass(test_image, hidden_out, output);
        
        // --- Clamping estimated normalized outputs to [0.0, 1.0] for validity ---
        double est_x_norm = CLAMP(output[1], 0.0, 1.0);
        double est_y_norm = CLAMP(output[2], 0.0, 1.0);
        double est_w_norm = CLAMP(output[3], 0.0, 1.0);
        double est_h_norm = CLAMP(output[4], 0.0, 1.0);
        
        // Denormalize known and estimated dimensions (round to nearest integer)
        int known_x = (int)(known_target[1] * GRID_SIZE + 0.5);
        int known_y = (int)(known_target[2] * GRID_SIZE + 0.5);
        int known_w = (int)(known_target[3] * GRID_SIZE + 0.5);
        int known_h = (int)(known_target[4] * GRID_SIZE + 0.5);

        int est_x = (int)(est_x_norm * GRID_SIZE + 0.5);
        int est_y = (int)(est_y_norm * GRID_SIZE + 0.5);
        int est_w = (int)(est_w_norm * GRID_SIZE + 0.5);
        int est_h = (int)(est_h_norm * GRID_SIZE + 0.5);
        
        printf("| %1d | %9.4f | %0.2f, %0.2f, %0.2f, %0.2f | %2d, %2d, %2d, %2d | %2d, %2d, %2d, %2d |\n",
               i + 1,
               output[0],
               est_x_norm, est_y_norm, est_w_norm, est_h_norm, // Print clamped normalized values
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
    estimate_nn_epochs(); // Simply prints the new epoch config

    printf("\n--- GLOBAL CONFIGURATION ---\n");
    printf("Model: Classification + Regression NN\n");
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
    printf("NN Testing Accuracy (Rect/Line): %.2f%%\n", acc_test * 100.0);
    
    // --- STEP 3: Regression Testing ---
    test_regression();
    
    end_total = clock();
    printf("\nTotal execution time: %.4f seconds.\n", (double)(end_total - start_total) / CLOCKS_PER_SEC);

    return 0;
}
