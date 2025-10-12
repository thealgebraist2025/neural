#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <string.h> 

// --- Configuration ---
#define N_SAMPLES_MAX 1000 // Maximum training size
#define D_SIZE 256         // 16x16 image size (RAW INPUT DIMENSION)
#define N_INPUT D_SIZE     // NN Input Dimension is now the raw image size
#define N_HIDDEN 12        // Hidden layer size
#define N_TEST_SAMPLES 500 // Standard test set size

// Time limit in seconds
#define MAX_TIME_NN_SEC 120.0

// Neural Network Parameters
#define LEARNING_RATE 0.01 // Reduced learning rate for stability with large input
#define N_EPOCHS_MAX 10000 
#define TARGET_RECTANGLE 1.0
#define TARGET_LINE_SET 0.0
// ---------------------

// --- Dynamic Globals ---
int N_SAMPLES = 1000; 
int N_EPOCHS;  

// Global Data & Matrices (Sized by MAX N)
double dataset[N_SAMPLES_MAX][D_SIZE];  // Raw Image Data (NN input)
double targets[N_SAMPLES_MAX];

// Neural Network Weights and Biases (Only one set needed)
double w_ih[N_INPUT][N_HIDDEN]; double b_h[N_HIDDEN]; 
double w_ho[N_HIDDEN][1]; double b_o[1];

// Test Data (fixed size)
double test_data[N_TEST_SAMPLES][D_SIZE];
double test_targets[N_TEST_SAMPLES];


// --- Function Prototypes ---
// Data Generation
void generate_rectangle(double image[D_SIZE]);
void generate_random_lines(double image[D_SIZE]);
void load_data_balanced(int n_samples, int start_index);
void load_subset_for_profiling(int n_subset);
void load_balanced_dataset();
void generate_test_set();

// Profiling
void estimate_nn_epochs();

// NN Core Functions
void initialize_nn();
void train_nn(const double input_set[N_SAMPLES_MAX][N_INPUT]);
double test_on_set(int n_set_size, const double input_set[][N_INPUT], const double target_set[]);
double sigmoid(double x);
double forward_pass(const double input[N_INPUT], double hidden_out[N_HIDDEN], double* output);
void backward_pass_and_update(const double input[N_INPUT], const double hidden_out[N_HIDDEN], double output, double target);

// New Robustness Test
void test_noise_robustness();

// -----------------------------------------------------------------
// --- DATA GENERATION FUNCTIONS ---
// -----------------------------------------------------------------

void generate_rectangle(double image[D_SIZE]) {
    int rect_w = 4 + (rand() % 8);
    int rect_h = 4 + (rand() % 8);
    int start_x = rand() % (16 - rect_w);
    int start_y = rand() % (16 - rect_h);
    
    for (int i = 0; i < D_SIZE; i++) { image[i] = 0.0; } 
    
    for (int y = start_y; y < start_y + rect_h; ++y) {
        for (int x = start_x; x < start_x + rect_w; ++x) {
            image[16 * y + x] = 200.0 + (double)(rand() % 50);
        }
    }
}
void generate_random_lines(double image[D_SIZE]) {
    for (int i = 0; i < D_SIZE; i++) { image[i] = 0.0; } 
    int num_lines = 1 + (rand() % 4); 
    for (int l = 0; l < num_lines; l++) {
        int length_options[] = {2, 4, 8};
        int length = length_options[rand() % 3];
        int x_start = rand() % 16;
        int y_start = rand() % 16;
        int orientation = rand() % 2; 
        double value = 200.0 + (double)(rand() % 50);

        for (int i = 0; i < length; i++) {
            int x = x_start, y = y_start;
            if (orientation == 0) { x = (x_start + i) % 16; } 
            else { y = (y_start + i) % 16; }
            int index = 16 * y + x;
            if (index >= 0 && index < D_SIZE) { image[index] = value; }
        }
    }
}
void load_data_balanced(int n_samples, int start_index) {
    for (int k = 0; k < n_samples; ++k) {
        int current_idx = start_index + k;
        if (k % 2 == 0) { 
            generate_rectangle(dataset[current_idx]);
            targets[current_idx] = TARGET_RECTANGLE;
        } else { 
            generate_random_lines(dataset[current_idx]);
            targets[current_idx] = TARGET_LINE_SET;
        }
    }
}
void load_subset_for_profiling(int n_subset) {
    for (int k = 0; k < n_subset; ++k) {
        if (k % 2 == 0) { 
            generate_rectangle(dataset[k]);
            targets[k] = TARGET_RECTANGLE;
        } else { 
            generate_random_lines(dataset[k]);
            targets[k] = TARGET_LINE_SET;
        }
    }
}
void load_balanced_dataset() {
    printf("Generating BALANCED dataset (%d images): 50%% Rectangles, 50%% Random Lines.\n", N_SAMPLES);
    load_data_balanced(N_SAMPLES, 0);
}
void generate_test_set() {
    printf("Generating TEST dataset (%d images): 50/50 mix of Rectangles/Random Lines.\n", N_TEST_SAMPLES);
    for (int k = 0; k < N_TEST_SAMPLES; ++k) {
        if (k % 2 == 0) { 
            generate_rectangle(test_data[k]);
            test_targets[k] = TARGET_RECTANGLE;
        } else { 
            generate_random_lines(test_data[k]);
            test_targets[k] = TARGET_LINE_SET;
        }
    }
}

// -----------------------------------------------------------------
// --- PROFILING FUNCTIONS ---
// -----------------------------------------------------------------

void estimate_nn_epochs() {
    clock_t start, end;
    #define N_EPOCHS_PROFILE 100
    
    initialize_nn(); 

    start = clock();
    for (int epoch = 0; epoch < N_EPOCHS_PROFILE; epoch++) {
        int sample_index = rand() % 50; 
        double hidden_out[N_HIDDEN]; double output;
        forward_pass(dataset[sample_index], hidden_out, &output);
        backward_pass_and_update(dataset[sample_index], hidden_out, output, targets[sample_index]);
    }
    end = clock();
    double time_spent_profile = (double)(end - start) / CLOCKS_PER_SEC;

    if (time_spent_profile < 1e-6) time_spent_profile = 1e-6;

    double epoch_scale_factor = MAX_TIME_NN_SEC / time_spent_profile;
    N_EPOCHS = (int)(N_EPOCHS_PROFILE * epoch_scale_factor);
    
    if (N_EPOCHS > N_EPOCHS_MAX) N_EPOCHS = N_EPOCHS_MAX;
    if (N_EPOCHS < N_EPOCHS_PROFILE) N_EPOCHS = N_EPOCHS_PROFILE;

    printf("\n--- NN EPOCHS TIME PROFILING ---\n");
    printf("Profile (%d epochs): %.4f sec\n", N_EPOCHS_PROFILE, time_spent_profile);
    printf("Estimated Epochs for %.1f sec limit: %d (Using N_EPOCHS=%d)\n", MAX_TIME_NN_SEC, (int)(N_EPOCHS_PROFILE * epoch_scale_factor), N_EPOCHS);
}

// -----------------------------------------------------------------
// --- NN CORE FUNCTIONS ---
// -----------------------------------------------------------------

void initialize_nn() {
    for (int i = 0; i < N_INPUT; i++) {
        for (int j = 0; j < N_HIDDEN; j++) {
            w_ih[i][j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * 0.1;
        }
    }
    for (int j = 0; j < N_HIDDEN; j++) {
        b_h[j] = 0.0;
        w_ho[j][0] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * 0.1;
    }
    b_o[0] = 0.0;
}

void train_nn(const double input_set[N_SAMPLES_MAX][N_INPUT]) {
    printf("Training on raw %d-dimensional image pixels...\n", N_INPUT);
    for (int epoch = 0; epoch < N_EPOCHS; epoch++) {
        int sample_index = rand() % N_SAMPLES;
        double hidden_out[N_HIDDEN];
        double output;
        forward_pass(input_set[sample_index], hidden_out, &output);
        backward_pass_and_update(input_set[sample_index], hidden_out, output, targets[sample_index]);
    }
}

double sigmoid(double x) { 
    return 1.0 / (1.0 + exp(-x)); 
}

double forward_pass(const double input[N_INPUT], double hidden_out[N_HIDDEN], double* output) {
    for (int j = 0; j < N_HIDDEN; j++) {
        double h_net = b_h[j];
        for (int i = 0; i < N_INPUT; i++) {
            h_net += input[i] * w_ih[i][j];
        }
        hidden_out[j] = sigmoid(h_net);
    }
    double o_net = b_o[0]; 
    for (int j = 0; j < N_HIDDEN; j++) { 
        o_net += hidden_out[j] * w_ho[j][0]; 
    } 
    *output = sigmoid(o_net);
    return 0.5 * pow(*output - TARGET_RECTANGLE, 2); 
}

void backward_pass_and_update(const double input[N_INPUT], const double hidden_out[N_HIDDEN], double output, double target) {
    double error_o = (output - target); 
    double delta_o = error_o * output * (1.0 - output); 
    
    double error_h[N_HIDDEN]; 
    for (int j = 0; j < N_HIDDEN; j++) { 
        error_h[j] = delta_o * w_ho[j][0]; 
    }
    double delta_h[N_HIDDEN]; 
    for (int j = 0; j < N_HIDDEN; j++) { 
        delta_h[j] = error_h[j] * hidden_out[j] * (1.0 - hidden_out[j]); 
    }
    
    for (int j = 0; j < N_HIDDEN; j++) { 
        w_ho[j][0] -= LEARNING_RATE * delta_o * hidden_out[j]; 
    } 
    b_o[0] -= LEARNING_RATE * delta_o;
    
    for (int i = 0; i < N_INPUT; i++) { 
        for (int j = 0; j < N_HIDDEN; j++) { 
            w_ih[i][j] -= LEARNING_RATE * delta_h[j] * input[i]; 
        } 
    }
    for (int j = 0; j < N_HIDDEN; j++) { 
        b_h[j] -= LEARNING_RATE * delta_h[j]; 
    }
}

double test_on_set(int n_set_size, const double input_set[][N_INPUT], const double target_set[]) {
    int correct_predictions = 0; 
    double hidden_out[N_HIDDEN]; 
    double output;
    for (int i = 0; i < n_set_size; i++) {
        forward_pass(input_set[i], hidden_out, &output);
        double prediction = (output >= 0.5) ? TARGET_RECTANGLE : TARGET_LINE_SET;
        double actual = target_set[i];
        if (fabs(prediction - actual) < DBL_EPSILON) { 
            correct_predictions++; 
        }
    }
    return (double)correct_predictions / n_set_size;
}

// -----------------------------------------------------------------
// --- NOISE ROBUSTNESS TEST ---
// -----------------------------------------------------------------

void test_noise_robustness() {
    double original_image[D_SIZE];
    double noisy_image[D_SIZE];
    
    // 1. Generate a single random rectangle image
    generate_rectangle(original_image);

    printf("\n--- NOISE ROBUSTNESS TEST ---\n");
    printf("Testing trained NN on a single random rectangle image with increasing noise:\n");
    printf("-------------------------------------------------------------------------\n");
    printf("| Noise %% | Output Score | Prediction | Correct |\n");
    printf("|---------|--------------|------------|---------|\n");

    double hidden_out[N_HIDDEN];
    double output;
    int correct_count = 0;
    int total_tests = 0;

    for (int noise_percent = 10; noise_percent <= 100; noise_percent += 10) {
        total_tests++;
        
        // 2. Clone the original and apply noise
        memcpy(noisy_image, original_image, D_SIZE * sizeof(double));
        
        int pixels_to_randomize = (int)(D_SIZE * (noise_percent / 100.0));
        
        for (int i = 0; i < pixels_to_randomize; i++) {
            int idx = rand() % D_SIZE;
            // Randomize pixel value (0.0 to 250.0)
            noisy_image[idx] = (double)(rand() % 251); 
        }

        // 3. Run forward pass
        forward_pass(noisy_image, hidden_out, &output);

        // 4. Calculate result
        double prediction = (output >= 0.5) ? TARGET_RECTANGLE : TARGET_LINE_SET;
        int is_correct = (fabs(prediction - TARGET_RECTANGLE) < DBL_EPSILON);
        
        if (is_correct) {
            correct_count++;
        }

        printf("| %7d | %12.4f | %10.0f | %7s |\n", 
               noise_percent, 
               output, 
               prediction, 
               is_correct ? "YES" : "NO");
    }

    printf("-------------------------------------------------------------------------\n");
    printf("Summary: The NN correctly classified the noisy rectangle %d out of %d times.\n", 
           correct_count, total_tests);
}

// -----------------------------------------------------------------
// --- MAIN EXECUTION ---
// -----------------------------------------------------------------
int main() {
    srand(time(NULL));
    clock_t start_total, end_total;
    start_total = clock();

    estimate_nn_epochs();

    load_balanced_dataset(); 
    generate_test_set();

    printf("\n--- GLOBAL CONFIGURATION ---\n");
    printf("Model: Simple NN (2 layers) | Input Dim: %d | Hidden Dim: %d\n", N_INPUT, N_HIDDEN);

    // --- STEP 1: NN Training on Raw Pixels ---
    printf("\n--- STEP 1: NN Training on Raw Pixels ---\n");
    clock_t start_nn = clock();
    initialize_nn();
    train_nn(dataset); // Training on the full raw dataset
    clock_t end_nn = clock();
    printf("NN Training time: %.4f seconds.\n", (double)(end_nn - start_nn) / CLOCKS_PER_SEC);

    // --- STEP 2: Standard Testing ---
    printf("\n--- STEP 2: Standard Testing Results ---\n");
    
    // Test on Training Set
    double acc_train = test_on_set(N_SAMPLES, dataset, targets);
    printf("NN Training Accuracy: %.2f%%\n", acc_train * 100.0);

    // Test on Unseen Test Set
    double acc_test = test_on_set(N_TEST_SAMPLES, test_data, test_targets);
    printf("NN Testing Accuracy: %.2f%%\n", acc_test * 100.0);
    
    // --- STEP 3: Noise Robustness Testing ---
    test_noise_robustness();
    
    end_total = clock();
    printf("\nTotal execution time (including profiling): %.4f seconds.\n", (double)(end_total - start_total) / CLOCKS_PER_SEC);

    return 0;
}
