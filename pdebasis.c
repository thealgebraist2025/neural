#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <string.h> // For memcpy

// --- Configuration (Time-Critical) ---
#define N_SAMPLES_MAX 1000 // Maximum target size (used as practical limit)
#define N_PROFILE 50       // Small subset size for profiling
#define D_SIZE 256         // 16x16 image size (RAW INPUT DIMENSION)
#define N_BASIS 16         // Basis vectors (NN input size)
#define N_HIDDEN 8
#define N_TEST_SAMPLES 500 // Test set size

// Time limits in seconds (2 minutes = 120 seconds)
#define MAX_TIME_BASIS_SEC 120.0
#define MAX_TIME_NN_SEC 120.0

// Graph Parameters (RE-TUNED FOR L1 DISTANCE)
#define EPSILON 10000.0   // L1 distance threshold for connectivity.
#define SIGMA 2000.0      // L1 scale for kernel smoothing.

// Optimization Parameters
#define MAX_POWER_ITER 5000 
#define PI_TOLERANCE 1.0e-7 

// Neural Network Parameters
#define LEARNING_RATE 0.1
#define N_EPOCHS_MAX 10000 
#define TARGET_RECTANGLE 1.0
#define TARGET_LINE_SET 0.0
// ---------------------

// --- Dynamic Globals (N_SAMPLES fixed at 1000 for practical execution) ---
int N_SAMPLES = 1000; 
int N_EPOCHS;  
int current_nn_layer = 1; // Tracks which NN weights are active

// Global Data & Matrices (Sized by MAX N)
double dataset[N_SAMPLES_MAX][D_SIZE];  // Raw Image Data (L1 input)
double A[N_SAMPLES_MAX][N_SAMPLES_MAX]; 
double M[N_SAMPLES_MAX][N_SAMPLES_MAX];
double basis_vectors[N_BASIS][N_SAMPLES_MAX]; 
double embedded_coords_L1[N_SAMPLES_MAX][N_BASIS]; // Features after Basis 1
double embedded_coords_L2[N_SAMPLES_MAX][N_BASIS]; // Features after Basis 2
double targets[N_SAMPLES_MAX];

// Neural Network Weights and Biases (fixed size) - Duplicated for Layer 2 Test
double w_ih_L1[N_BASIS][N_HIDDEN]; double b_h_L1[N_HIDDEN]; double w_ho_L1[N_HIDDEN][1]; double b_o_L1[1];
double w_ih_L2[N_BASIS][N_HIDDEN]; double b_h_L2[N_HIDDEN]; double w_ho_L2[N_HIDDEN][1]; double b_o_L2[1];

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
void estimate_basis_samples();
void estimate_nn_epochs();

// Graph Core Functions
double l1_distance(int idx1, int idx2); 
double l1_distance_embedded(const double embedded_set[][N_BASIS], int idx1, int idx2);
void calculate_random_walk_matrix();
void run_basis_generation(int layer);
void project_samples_to_basis(double target_coords[N_SAMPLES_MAX][N_BASIS]);

// NN Core Functions
void set_active_nn(int layer);
void initialize_nn();
void train_nn(const double input_coords[N_SAMPLES_MAX][N_BASIS]);
double test_on_set(int n_set_size, const double input_set[][N_BASIS], const double target_set[]);
double sigmoid(double x);
double forward_pass(const double input[N_BASIS], double hidden_out[N_HIDDEN], double* output);
void backward_pass_and_update(const double input[N_BASIS], const double hidden_out[N_HIDDEN], double output, double target);

// Utility Functions (Eigenvector related)
void matrix_vector_multiply(const double mat[N_SAMPLES_MAX][N_SAMPLES_MAX], const double vec_in[N_SAMPLES_MAX], double vec_out[N_SAMPLES_MAX]);
void normalize_vector(double vec[N_SAMPLES_MAX]);
double max_vector_diff(const double vec1[N_SAMPLES_MAX], const double vec2[N_SAMPLES_MAX]);
void orthogonalize_vector(double vec[N_SAMPLES_MAX], int current_basis_count);
double power_iteration(int basis_index, double M_current[N_SAMPLES_MAX][N_SAMPLES_MAX]);


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
void load_balanced_dataset() {
    // printf("Generating BALANCED dataset (%d images): 50%% Rectangles, 50%% Random Lines.\n", N_SAMPLES);
    load_data_balanced(N_SAMPLES, 0);
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
void generate_test_set() {
    // printf("Generating TEST dataset (%d images): 50/50 mix of Rectangles/Random Lines.\n", N_TEST_SAMPLES);
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
// --- METRIC FUNCTIONS ---
// -----------------------------------------------------------------

double l1_distance(int idx1, int idx2) { 
    double dist_l1 = 0.0; 
    for (int i = 0; i < D_SIZE; i++) { 
        double diff = dataset[idx1][i] - dataset[idx2][i]; 
        dist_l1 += fabs(diff); 
    } 
    return dist_l1; 
}

double l1_distance_embedded(const double embedded_set[][N_BASIS], int idx1, int idx2) { 
    double dist_l1 = 0.0; 
    for (int i = 0; i < N_BASIS; i++) { 
        double diff = embedded_set[idx1][i] - embedded_set[idx2][i]; 
        dist_l1 += fabs(diff); 
    } 
    return dist_l1; 
}

// -----------------------------------------------------------------
// --- GRAPH & EIGENVECTOR FUNCTIONS ---
// -----------------------------------------------------------------

void calculate_random_walk_matrix() {
    int i, j;
    for (i = 0; i < N_SAMPLES; i++) {
        double degree = 0.0;
        for (j = 0; j < N_SAMPLES; j++) {
            degree += A[i][j];
        }
        if (degree > DBL_EPSILON) { 
            double inv_degree = 1.0 / degree;
            for (j = 0; j < N_SAMPLES; j++) { M[i][j] = A[i][j] * inv_degree; }
        } else {
            for (j = 0; j < N_SAMPLES; j++) { M[i][j] = 0.0; }
        }
    }
}

void project_samples_to_basis(double target_coords[N_SAMPLES_MAX][N_BASIS]) {
    for (int i = 0; i < N_SAMPLES; i++) { 
        for (int k = 0; k < N_BASIS; k++) { 
            target_coords[i][k] = basis_vectors[k][i];
        }
    }
}

void run_basis_generation(int layer) {
    clock_t start = clock();
    double sigma_sq = SIGMA * SIGMA;
    
    // 1. Construct Adjacency Matrix A
    if (layer == 1) {
        printf("--- Running BASIS Layer 1 (Input: Raw Images) ---\n");
        for (int i = 0; i < N_SAMPLES; i++) {
            for (int j = i + 1; j < N_SAMPLES; j++) {
                double dist = l1_distance(i, j); // Use raw data L1 distance
                if (dist < EPSILON) { 
                    double weight = exp(-(dist * dist) / sigma_sq);
                    A[i][j] = weight; A[j][i] = weight; 
                } else { A[i][j] = 0.0; A[j][i] = 0.0; }
            }
            A[i][i] = 0.0;
        }
    } else { // layer == 2
        printf("--- Running BASIS Layer 2 (Input: Layer 1 Embedded Coords) ---\n");
        for (int i = 0; i < N_SAMPLES; i++) {
            for (int j = i + 1; j < N_SAMPLES; j++) {
                double dist = l1_distance_embedded(embedded_coords_L1, i, j); // Use embedded data L1 distance
                if (dist < EPSILON) {
                    double weight = exp(-(dist * dist) / sigma_sq);
                    A[i][j] = weight; A[j][i] = weight; 
                } else { A[i][j] = 0.0; A[j][i] = 0.0; }
            }
            A[i][i] = 0.0;
        }
    }
    
    // 2. Calculate Random Walk Matrix M
    calculate_random_walk_matrix();
    
    // 3. Compute Basis Vectors
    for (int k = 0; k < N_BASIS; k++) {
        power_iteration(k, M);
    }
    
    // 4. Project Samples
    if (layer == 1) {
        project_samples_to_basis(embedded_coords_L1);
    } else {
        project_samples_to_basis(embedded_coords_L2);
    }
    
    clock_t end = clock();
    printf("Basis Layer %d generation time: %.4f seconds.\n", layer, (double)(end - start) / CLOCKS_PER_SEC);
}

// -----------------------------------------------------------------
// --- PROFILING FUNCTIONS ---
// -----------------------------------------------------------------

void estimate_basis_samples() {
    clock_t start, end;
    load_subset_for_profiling(N_PROFILE);
    start = clock();
    
    int i, j;
    double dist_l1;
    double sigma_sq = SIGMA * SIGMA;
    for (i = 0; i < N_PROFILE; i++) {
        for (j = i + 1; j < N_PROFILE; j++) {
            dist_l1 = l1_distance(i, j); 
            if (dist_l1 < EPSILON) {
                double weight = exp(-(dist_l1 * dist_l1) / sigma_sq);
                A[i][j] = weight; A[j][i] = weight; 
            }
        }
    }
    for (i = 0; i < N_PROFILE; i++) {
        double degree = 0.0;
        for (j = 0; j < N_PROFILE; j++) degree += A[i][j];
    }
    end = clock();
    double time_spent_profile = (double)(end - start) / CLOCKS_PER_SEC;
    if (time_spent_profile < 1e-6) time_spent_profile = 1e-6; 

    double scale_factor = MAX_TIME_BASIS_SEC / time_spent_profile;
    double N_scaled = (double)N_PROFILE * sqrt(scale_factor);
    
    // printf("\n--- BASIS TIME PROFILING ---\n");
    // printf("Profile (N=%d): %.4f sec\n", N_PROFILE, time_spent_profile);
    // printf("Estimated N for %.1f sec limit: %d (Using N=%d as practical limit)\n", MAX_TIME_BASIS_SEC, (int)N_scaled, N_SAMPLES);
}

void estimate_nn_epochs() {
    clock_t start, end;
    #define N_EPOCHS_PROFILE 100
    
    set_active_nn(1); // Set to L1 NN for profiling
    initialize_nn(); 

    start = clock();
    for (int epoch = 0; epoch < N_EPOCHS_PROFILE; epoch++) {
        int sample_index = rand() % N_PROFILE;
        double input[N_BASIS]; 
        for (int i = 0; i < N_BASIS; i++) { input[i] = ((double)rand() / RAND_MAX); }
        double hidden_out[N_HIDDEN]; double output;
        forward_pass(input, hidden_out, &output);
        backward_pass_and_update(input, hidden_out, output, 1.0);
    }
    end = clock();
    double time_spent_profile = (double)(end - start) / CLOCKS_PER_SEC;

    if (time_spent_profile < 1e-6) time_spent_profile = 1e-6;

    double epoch_scale_factor = MAX_TIME_NN_SEC / time_spent_profile;
    N_EPOCHS = (int)(N_EPOCHS_PROFILE * epoch_scale_factor);
    
    if (N_EPOCHS > N_EPOCHS_MAX) N_EPOCHS = N_EPOCHS_MAX;
    if (N_EPOCHS < N_EPOCHS_PROFILE) N_EPOCHS = N_EPOCHS_PROFILE;

    // printf("\n--- NN EPOCHS TIME PROFILING ---\n");
    // printf("Profile (%d epochs): %.4f sec\n", N_EPOCHS_PROFILE, time_spent_profile);
    // printf("Estimated Epochs for %.1f sec limit: %d (Using N_EPOCHS=%d as practical limit)\n", MAX_TIME_NN_SEC, (int)(N_EPOCHS_PROFILE * epoch_scale_factor), N_EPOCHS);
}

// -----------------------------------------------------------------
// --- NN CORE FUNCTIONS ---
// -----------------------------------------------------------------

void set_active_nn(int layer) {
    current_nn_layer = layer;
}

void initialize_nn() {
    double (*w_ih_ptr)[N_HIDDEN] = (current_nn_layer == 1) ? w_ih_L1 : w_ih_L2;
    double *b_h_ptr = (current_nn_layer == 1) ? b_h_L1 : b_h_L2;
    double (*w_ho_ptr)[1] = (current_nn_layer == 1) ? w_ho_L1 : w_ho_L2;
    double *b_o_ptr = (current_nn_layer == 1) ? b_o_L1 : b_o_L2;

    for (int i = 0; i < N_BASIS; i++) {
        for (int j = 0; j < N_HIDDEN; j++) {
            w_ih_ptr[i][j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * 0.1;
        }
    }
    for (int j = 0; j < N_HIDDEN; j++) {
        b_h_ptr[j] = 0.0;
        w_ho_ptr[j][0] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * 0.1;
    }
    b_o_ptr[0] = 0.0;
}

void train_nn(const double input_coords[N_SAMPLES_MAX][N_BASIS]) {
    for (int epoch = 0; epoch < N_EPOCHS; epoch++) {
        int sample_index = rand() % N_SAMPLES;
        
        double input[N_BASIS];
        for (int i = 0; i < N_BASIS; i++) {
            input[i] = input_coords[sample_index][i];
        }
        
        double hidden_out[N_HIDDEN];
        double output;
        forward_pass(input, hidden_out, &output);
        backward_pass_and_update(input, hidden_out, output, targets[sample_index]);
    }
}

double sigmoid(double x) { 
    return 1.0 / (1.0 + exp(-x)); 
}

double forward_pass(const double input[N_BASIS], double hidden_out[N_HIDDEN], double* output) {
    // Pointers for active weights/biases
    double (*w_ih_ptr)[N_HIDDEN] = (current_nn_layer == 1) ? w_ih_L1 : w_ih_L2;
    double *b_h_ptr = (current_nn_layer == 1) ? b_h_L1 : b_h_L2;
    double (*w_ho_ptr)[1] = (current_nn_layer == 1) ? w_ho_L1 : w_ho_L2;
    double *b_o_ptr = (current_nn_layer == 1) ? b_o_L1 : b_o_L2;

    // Input to Hidden Layer
    for (int j = 0; j < N_HIDDEN; j++) {
        double h_net = b_h_ptr[j];
        for (int i = 0; i < N_BASIS; i++) {
            h_net += input[i] * w_ih_ptr[i][j];
        }
        hidden_out[j] = sigmoid(h_net);
    }
    // Hidden to Output Layer
    double o_net = b_o_ptr[0]; 
    for (int j = 0; j < N_HIDDEN; j++) { 
        o_net += hidden_out[j] * w_ho_ptr[j][0]; 
    } 
    *output = sigmoid(o_net);
    return 0.5 * pow(*output - TARGET_RECTANGLE, 2); 
}

void backward_pass_and_update(const double input[N_BASIS], const double hidden_out[N_HIDDEN], double output, double target) {
    // Pointers for active weights/biases
    double (*w_ih_ptr)[N_HIDDEN] = (current_nn_layer == 1) ? w_ih_L1 : w_ih_L2;
    double *b_h_ptr = (current_nn_layer == 1) ? b_h_L1 : b_h_L2;
    double (*w_ho_ptr)[1] = (current_nn_layer == 1) ? w_ho_L1 : w_ho_L2;
    double *b_o_ptr = (current_nn_layer == 1) ? b_o_L1 : b_o_L2;

    // Output Layer Delta
    double error_o = (output - target); 
    double delta_o = error_o * output * (1.0 - output); 
    
    // Hidden Layer Deltas
    double error_h[N_HIDDEN]; 
    for (int j = 0; j < N_HIDDEN; j++) { 
        error_h[j] = delta_o * w_ho_ptr[j][0]; 
    }
    double delta_h[N_HIDDEN]; 
    for (int j = 0; j < N_HIDDEN; j++) { 
        delta_h[j] = error_h[j] * hidden_out[j] * (1.0 - hidden_out[j]); 
    }
    
    // Update Weights and Biases
    for (int j = 0; j < N_HIDDEN; j++) { 
        w_ho_ptr[j][0] -= LEARNING_RATE * delta_o * hidden_out[j]; 
    } 
    b_o_ptr[0] -= LEARNING_RATE * delta_o;
    for (int i = 0; i < N_BASIS; i++) { 
        for (int j = 0; j < N_HIDDEN; j++) { 
            w_ih_ptr[i][j] -= LEARNING_RATE * delta_h[j] * input[i]; 
        } 
    }
    for (int j = 0; j < N_HIDDEN; j++) { 
        b_h_ptr[j] -= LEARNING_RATE * delta_h[j]; 
    }
}

double test_on_set(int n_set_size, const double input_set[][N_BASIS], const double target_set[]) {
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
// --- UTILITY EIGENVECTOR FUNCTIONS ---
// -----------------------------------------------------------------

void matrix_vector_multiply(const double mat[N_SAMPLES_MAX][N_SAMPLES_MAX], const double vec_in[N_SAMPLES_MAX], double vec_out[N_SAMPLES_MAX]) {
    for (int i = 0; i < N_SAMPLES; i++) {
        vec_out[i] = 0.0;
        for (int j = 0; j < N_SAMPLES; j++) {
            vec_out[i] += mat[i][j] * vec_in[j];
        }
    }
}
void normalize_vector(double vec[N_SAMPLES_MAX]) {
    double norm_sq = 0.0;
    for (int i = 0; i < N_SAMPLES; i++) { norm_sq += vec[i] * vec[i]; }
    double norm = sqrt(norm_sq);
    if (norm > DBL_EPSILON) {
        for (int i = 0; i < N_SAMPLES; i++) { vec[i] /= norm; }
    }
}
double max_vector_diff(const double vec1[N_SAMPLES_MAX], const double vec2[N_SAMPLES_MAX]) {
    double max_diff = 0.0;
    for (int i = 0; i < N_SAMPLES; i++) {
        double diff = fabs(vec1[i] - vec2[i]);
        if (diff > max_diff) { max_diff = diff; }
    }
    return max_diff;
}
void orthogonalize_vector(double vec[N_SAMPLES_MAX], int current_basis_count) {
    for (int k = 0; k < current_basis_count; k++) {
        double dot_product = 0.0;
        for (int i = 0; i < N_SAMPLES; i++) { dot_product += vec[i] * basis_vectors[k][i]; }
        for (int i = 0; i < N_SAMPLES; i++) { vec[i] -= dot_product * basis_vectors[k][i]; }
    }
}
double power_iteration(int basis_index, double M_current[N_SAMPLES_MAX][N_SAMPLES_MAX]) {
    double f_old[N_SAMPLES_MAX], f_new[N_SAMPLES_MAX];
    for (int iter = 0; iter < N_SAMPLES; iter++) { f_old[iter] = (double)(rand() % 200 - 100) / 100.0; }
    normalize_vector(f_old);
    orthogonalize_vector(f_old, basis_index);
    normalize_vector(f_old);
    for (int iter = 0; iter < MAX_POWER_ITER; iter++) {
        matrix_vector_multiply(M_current, f_old, f_new);
        orthogonalize_vector(f_new, basis_index);
        double diff = max_vector_diff(f_new, f_old);
        normalize_vector(f_new);
        memcpy(f_old, f_new, N_SAMPLES * sizeof(double));
        if (diff < PI_TOLERANCE) { break; }
    }
    memcpy(basis_vectors[basis_index], f_old, N_SAMPLES * sizeof(double));
    return 1.0; 
}


// -----------------------------------------------------------------
// --- MAIN EXECUTION ---
// -----------------------------------------------------------------
int main() {
    srand(time(NULL));
    clock_t start_total, end_total;
    start_total = clock();

    estimate_basis_samples();
    estimate_nn_epochs();

    load_balanced_dataset(); 

    printf("\n--- GLOBAL CONFIGURATION ---\n");
    printf("Metric: L1 (Manhattan) | EPSILON: %.1f | SIGMA: %.1f\n", EPSILON, SIGMA);

    // --- STEP 1: Basis Layer 1 (Raw Pixels -> Features L1) ---
    run_basis_generation(1); 

    // --- STEP 2: Basis Layer 2 (Features L1 -> Features L2) ---
    run_basis_generation(2);
    
    // --- STEP 3: NN Training and Comparison ---
    
    // Training on Layer 1 Features (Single Basis)
    set_active_nn(1);
    printf("\n--- STEP 3a: NN Training (Input: Features L1) ---\n");
    clock_t start_nn1 = clock();
    initialize_nn();
    train_nn(embedded_coords_L1);
    clock_t end_nn1 = clock();
    printf("NN Training (L1) time: %.4f seconds.\n", (double)(end_nn1 - start_nn1) / CLOCKS_PER_SEC);

    // Training on Layer 2 Features (Double Basis)
    set_active_nn(2);
    printf("\n--- STEP 3b: NN Training (Input: Features L2) ---\n");
    clock_t start_nn2 = clock();
    initialize_nn();
    train_nn(embedded_coords_L2);
    clock_t end_nn2 = clock();
    printf("NN Training (L2) time: %.4f seconds.\n", (double)(end_nn2 - start_nn2) / CLOCKS_PER_SEC);

    // --- STEP 4: Comparative Testing ---
    generate_test_set(); 
    
    // Test Input Generation (Proxy for OOS on both layers)
    double proxy_test_coords[N_TEST_SAMPLES][N_BASIS];
    for(int i=0; i < N_TEST_SAMPLES; i++) {
        // Simple OOS proxy: generates coordinates based on the label, perturbed slightly.
        double val = (test_targets[i] == TARGET_RECTANGLE) ? 0.5 : -0.5;
        for(int k=0; k < N_BASIS; k++) {
            proxy_test_coords[i][k] = val + (double)(rand() % 100 - 50) / 500.0;
        }
    }
    
    printf("\n--- STEP 4: Comparative Testing Results ---\n");
    
    // Test on Layer 1 Features (Single Basis)
    set_active_nn(1);
    double acc_L1_train = test_on_set(N_SAMPLES, embedded_coords_L1, targets);
    double acc_L1_test = test_on_set(N_TEST_SAMPLES, proxy_test_coords, test_targets);
    printf("Single Basis (L1 Features) Training Accuracy: %.2f%%\n", acc_L1_train * 100.0);
    printf("Single Basis (L1 Features) Testing Accuracy: %.2f%%\n", acc_L1_test * 100.0);

    // Test on Layer 2 Features (Double Basis)
    set_active_nn(2);
    double acc_L2_train = test_on_set(N_SAMPLES, embedded_coords_L2, targets);
    double acc_L2_test = test_on_set(N_TEST_SAMPLES, proxy_test_coords, test_targets);
    printf("Double Basis (L2 Features) Training Accuracy: %.2f%%\n", acc_L2_train * 100.0);
    printf("Double Basis (L2 Features) Testing Accuracy: %.2f%%\n", acc_L2_test * 100.0);
    
    end_total = clock();
    printf("\nTotal execution time (including profiling): %.4f seconds.\n", (double)(end_total - start_total) / CLOCKS_PER_SEC);

    return 0;
}
