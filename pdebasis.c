#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>

// --- Configuration (Time-Critical) ---
#define N_SAMPLES_MAX 1000 // Maximum target size (used as practical limit)
#define N_PROFILE 50       // Small subset size for profiling
#define D_SIZE 256         // 16x16 image size
#define N_BASIS 16         // Basis vectors (NN input size)
#define N_HIDDEN 8
#define N_TEST_SAMPLES 500 // Test set size

// Time limits in seconds (2 minutes = 120 seconds)
#define MAX_TIME_BASIS_SEC 120.0
#define MAX_TIME_NN_SEC 120.0

// Graph Parameters (RE-TUNED FOR L1 DISTANCE)
#define EPSILON 10000.0   // Increased to reflect typical L1 distances.
#define SIGMA 2000.0      // Adjusted for new L1 scale.

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

// Global Data & Matrices (Sized by MAX N)
double dataset[N_SAMPLES_MAX][D_SIZE];  
double A[N_SAMPLES_MAX][N_SAMPLES_MAX]; 
double M[N_SAMPLES_MAX][N_SAMPLES_MAX];
double basis_vectors[N_BASIS][N_SAMPLES_MAX]; 
double embedded_coords[N_SAMPLES_MAX][N_BASIS]; 
double targets[N_SAMPLES_MAX];

// Neural Network Weights and Biases (fixed size)
double w_ih[N_BASIS][N_HIDDEN];   
double b_h[N_HIDDEN];             
double w_ho[N_HIDDEN][1];         
double b_o[1];                    

// Test Data (fixed size)
double test_data[N_TEST_SAMPLES][D_SIZE];
double test_targets[N_TEST_SAMPLES];

// --- Function Prototypes ---
void generate_rectangle(double image[D_SIZE]);
void generate_random_lines(double image[D_SIZE]);
void load_data_balanced(int n_samples, int start_index);
void load_subset_for_profiling(int n_subset);
void estimate_basis_samples();
void estimate_nn_epochs();
void generate_test_set();
double l1_distance(int idx1, int idx2); 
void construct_adjacency_matrix();
void calculate_random_walk_matrix();
void matrix_vector_multiply(const double mat[N_SAMPLES_MAX][N_SAMPLES_MAX], const double vec_in[N_SAMPLES_MAX], double vec_out[N_SAMPLES_MAX]);
void normalize_vector(double vec[N_SAMPLES_MAX]);
double max_vector_diff(const double vec1[N_SAMPLES_MAX], const double vec2[N_SAMPLES_MAX]);
void orthogonalize_vector(double vec[N_SAMPLES_MAX], int current_basis_count);
double power_iteration(int basis_index, double M_current[N_SAMPLES_MAX][N_SAMPLES_MAX]);
void project_samples_to_basis();
double sigmoid(double x);
void initialize_nn();
double forward_pass(const double input[N_BASIS], double hidden_out[N_HIDDEN], double* output);
void backward_pass_and_update(const double input[N_BASIS], const double hidden_out[N_HIDDEN], double output, double target);
double test_on_set(int n_set_size, const double input_set[][N_BASIS], const double target_set[]);


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
            int x = x_start;
            int y = y_start;

            if (orientation == 0) { 
                x = (x_start + i) % 16;
            } else { 
                y = (y_start + i) % 16;
            }

            int index = 16 * y + x;
            if (index >= 0 && index < D_SIZE) {
                image[index] = value;
            }
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
    printf("Generating BALANCED dataset (%d images): 50%% Rectangles, 50%% Random Lines.\n", N_SAMPLES);
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
// --- METRIC: L1 DISTANCE ---
// -----------------------------------------------------------------

double l1_distance(int idx1, int idx2) { 
    double dist_l1 = 0.0; 
    for (int i = 0; i < D_SIZE; i++) { 
        double diff = dataset[idx1][i] - dataset[idx2][i]; 
        dist_l1 += fabs(diff); // Sum of absolute differences
    } 
    return dist_l1; 
}


// -----------------------------------------------------------------
// --- GRAPH & EIGENVECTOR FUNCTIONS ---
// -----------------------------------------------------------------

void construct_adjacency_matrix() {
    int i, j;
    double dist_l1;
    double sigma_sq = SIGMA * SIGMA;

    printf("Constructing Adjacency Matrix A (N=%d)...\n", N_SAMPLES);
    for (i = 0; i < N_SAMPLES; i++) {
        for (j = i + 1; j < N_SAMPLES; j++) {
            dist_l1 = l1_distance(i, j); 
            if (dist_l1 < EPSILON) { 
                double weight = exp(-(dist_l1 * dist_l1) / sigma_sq);
                A[i][j] = weight;
                A[j][i] = weight; 
            } else {
                A[i][j] = 0.0;
                A[j][i] = 0.0;
            }
        }
        A[i][i] = 0.0;
    }
}

void calculate_random_walk_matrix() {
    int i, j;
    printf("Calculating Random Walk Matrix M (N=%d)...\n", N_SAMPLES);

    for (i = 0; i < N_SAMPLES; i++) {
        double degree = 0.0;
        for (j = 0; j < N_SAMPLES; j++) {
            degree += A[i][j];
        }

        if (degree > DBL_EPSILON) { 
            double inv_degree = 1.0 / degree;
            for (j = 0; j < N_SAMPLES; j++) {
                M[i][j] = A[i][j] * inv_degree;
            }
        } else {
            for (j = 0; j < N_SAMPLES; j++) {
                M[i][j] = 0.0;
            }
        }
    }
}

void matrix_vector_multiply(const double mat[N_SAMPLES_MAX][N_SAMPLES_MAX], const double vec_in[N_SAMPLES_MAX], double vec_out[N_SAMPLES_MAX]) {
    int i, j;
    for (i = 0; i < N_SAMPLES; i++) {
        vec_out[i] = 0.0;
        for (j = 0; j < N_SAMPLES; j++) {
            vec_out[i] += mat[i][j] * vec_in[j];
        }
    }
}

void normalize_vector(double vec[N_SAMPLES_MAX]) {
    int i;
    double norm_sq = 0.0;
    for (i = 0; i < N_SAMPLES; i++) {
        norm_sq += vec[i] * vec[i];
    }
    double norm = sqrt(norm_sq);
    if (norm > DBL_EPSILON) {
        for (i = 0; i < N_SAMPLES; i++) {
            vec[i] /= norm;
        }
    }
}

double max_vector_diff(const double vec1[N_SAMPLES_MAX], const double vec2[N_SAMPLES_MAX]) {
    double max_diff = 0.0;
    for (int i = 0; i < N_SAMPLES; i++) {
        double diff = fabs(vec1[i] - vec2[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

void orthogonalize_vector(double vec[N_SAMPLES_MAX], int current_basis_count) {
    for (int k = 0; k < current_basis_count; k++) {
        double dot_product = 0.0;
        for (int i = 0; i < N_SAMPLES; i++) {
            dot_product += vec[i] * basis_vectors[k][i];
        }
        for (int i = 0; i < N_SAMPLES; i++) {
            vec[i] -= dot_product * basis_vectors[k][i];
        }
    }
}

double power_iteration(int basis_index, double M_current[N_SAMPLES_MAX][N_SAMPLES_MAX]) {
    int iter;
    double f_old[N_SAMPLES_MAX];
    double f_new[N_SAMPLES_MAX];
    
    for (iter = 0; iter < N_SAMPLES; iter++) {
        f_old[iter] = (double)(rand() % 200 - 100) / 100.0; 
    }
    normalize_vector(f_old);
    orthogonalize_vector(f_old, basis_index);
    normalize_vector(f_old);

    for (iter = 0; iter < MAX_POWER_ITER; iter++) {
        matrix_vector_multiply(M_current, f_old, f_new);
        orthogonalize_vector(f_new, basis_index);
        double diff = max_vector_diff(f_new, f_old);
        normalize_vector(f_new);
        
        for(int i=0; i < N_SAMPLES; i++) {
            f_old[i] = f_new[i];
        }

        if (diff < PI_TOLERANCE) {
            break;
        }
    }
    
    for (int i = 0; i < N_SAMPLES; i++) {
        basis_vectors[basis_index][i] = f_old[i];
    }
    return 1.0; 
}

void project_samples_to_basis() {
    for (int i = 0; i < N_SAMPLES; i++) { 
        for (int k = 0; k < N_BASIS; k++) { 
            embedded_coords[i][k] = basis_vectors[k][i];
        }
    }
}

// -----------------------------------------------------------------
// --- PROFILING FUNCTIONS ---
// -----------------------------------------------------------------

void estimate_basis_samples() {
    clock_t start, end;
    double time_spent_profile;
    
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
    time_spent_profile = (double)(end - start) / CLOCKS_PER_SEC;
    if (time_spent_profile < 1e-6) time_spent_profile = 1e-6; 

    double scale_factor = MAX_TIME_BASIS_SEC / time_spent_profile;
    double N_scaled = (double)N_PROFILE * sqrt(scale_factor);
    
    printf("\n--- BASIS TIME PROFILING ---\n");
    printf("Profile (N=%d): %.4f sec\n", N_PROFILE, time_spent_profile);
    printf("Estimated N for %.1f sec limit: %d (Using N=%d as practical limit)\n", MAX_TIME_BASIS_SEC, (int)N_scaled, N_SAMPLES);
}

void estimate_nn_epochs() {
    clock_t start, end;
    double time_spent_profile;
    
    #define N_EPOCHS_PROFILE 100
    
    initialize_nn(); 

    start = clock();
    for (int epoch = 0; epoch < N_EPOCHS_PROFILE; epoch++) {
        int sample_index = rand() % N_PROFILE;
        double input[N_BASIS]; 
        for (int i = 0; i < N_BASIS; i++) { 
            input[i] = ((double)rand() / RAND_MAX); 
        }
        double hidden_out[N_HIDDEN]; double output;
        forward_pass(input, hidden_out, &output);
        backward_pass_and_update(input, hidden_out, output, 1.0);
    }
    end = clock();
    time_spent_profile = (double)(end - start) / CLOCKS_PER_SEC;

    if (time_spent_profile < 1e-6) time_spent_profile = 1e-6;

    double epoch_scale_factor = MAX_TIME_NN_SEC / time_spent_profile;
    N_EPOCHS = (int)(N_EPOCHS_PROFILE * epoch_scale_factor);
    
    if (N_EPOCHS > N_EPOCHS_MAX) N_EPOCHS = N_EPOCHS_MAX;
    if (N_EPOCHS < N_EPOCHS_PROFILE) N_EPOCHS = N_EPOCHS_PROFILE;

    printf("\n--- NN EPOCHS TIME PROFILING ---\n");
    printf("Profile (%d epochs): %.4f sec\n", N_EPOCHS_PROFILE, time_spent_profile);
    printf("Estimated Epochs for %.1f sec limit: %d (Using N_EPOCHS=%d as practical limit)\n", MAX_TIME_NN_SEC, (int)(N_EPOCHS_PROFILE * epoch_scale_factor), N_EPOCHS);
}

// -----------------------------------------------------------------
//
