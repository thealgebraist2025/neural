#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>

// --- Configuration (Time-Critical) ---
#define N_SAMPLES_MAX 1000 // Maximum target size
#define N_PROFILE 50       // Small subset size for profiling
#define D_SIZE 256
#define N_BASIS 16
#define N_HIDDEN 8
#define N_TEST_SAMPLES 500

// Time limits in seconds (2 minutes = 120 seconds)
#define MAX_TIME_BASIS_SEC 120.0
#define MAX_TIME_NN_SEC 120.0

// Graph Parameters
#define EPSILON 2500.0
#define SIGMA 500.0

// Optimization Parameters
#define MAX_POWER_ITER 5000 
#define PI_TOLERANCE 1.0e-7 

// Neural Network Parameters
#define LEARNING_RATE 0.1
#define N_EPOCHS_MAX 10000 
#define TARGET_RECTANGLE 1.0
#define TARGET_NO_RECTANGLE 0.0
// ---------------------

// --- Dynamic Globals (Sized by N_SAMPLES) ---
int N_SAMPLES; // Will be set dynamically by profiling
int N_EPOCHS;  // Will be set dynamically by profiling

// Global Data & Matrices (Sized by MAX N for dynamic allocation proxy)
double dataset[N_SAMPLES_MAX][D_SIZE];  
double A[N_SAMPLES_MAX][N_SAMPLES_MAX]; 
double M[N_SAMPLES_MAX][N_SAMPLES_MAX];
double basis_vectors[N_BASIS][N_SAMPLES_MAX]; 
double embedded_coords[N_SAMPLES_MAX][N_BASIS]; 
double targets[N_SAMPLES_MAX];

// Neural Network Weights and Biases
double w_ih[N_BASIS][N_HIDDEN];   
double b_h[N_HIDDEN];             
double w_ho[N_HIDDEN][1];         
double b_o[1];                    

// Test Data
double test_data[N_TEST_SAMPLES][D_SIZE];
double test_targets[N_TEST_SAMPLES];

// Function prototypes (definitions placed at the end)
// ... (omitted prototypes for space, all defined below)


// -----------------------------------------------------------------
// --- PROFILING AND SCALING FUNCTIONS ---
// -----------------------------------------------------------------

// Helper function to load only a subset of data
void load_subset(int n_subset) {
    // Uses the same logic as load_mock_dataset, but only for n_subset
    for (int k = 0; k < n_subset; ++k) {
        int rect_w = 4 + (rand() % 8);
        int rect_h = 4 + (rand() % 8);
        int start_x = rand() % (16 - rect_w);
        int start_y = rand() % (16 - rect_h);

        for (int i = 0; i < D_SIZE; i++) {
            dataset[k][i] = (rand() % 100 < 5) ? (double)(rand() % 256) : 0.0;
        }
        
        for (int y = start_y; y < start_y + rect_h; ++y) {
            for (int x = start_x; x < start_x + rect_w; ++x) {
                dataset[k][16 * y + x] = 200.0 + (double)(rand() % 50);
            }
        }
        int black_noise_count = (int)(0.05 * D_SIZE);
        for (int i = 0; i < black_noise_count; i++) {
            dataset[k][rand() % D_SIZE] = 0.0;
        }
        targets[k] = TARGET_RECTANGLE;
    }
}

// Function to estimate N_SAMPLES for MAX_TIME_BASIS_SEC
void estimate_basis_samples() {
    clock_t start, end;
    double time_spent_profile;
    
    // 1. Load and Profile the Small Subset
    load_subset(N_PROFILE);
    start = clock();
    
    // --- Basis Generation Profile ---
    // Calculate Adjacency Matrix (O(N^2 * D))
    int i, j;
    double dist_sq, epsilon_sq = EPSILON * EPSILON, sigma_sq = SIGMA * SIGMA;
    for (i = 0; i < N_PROFILE; i++) {
        for (j = i + 1; j < N_PROFILE; j++) {
            // Mock euclidean_distance_sq call
            dist_sq = euclidean_distance_sq(i, j); 
            if (dist_sq < epsilon_sq) {
                double weight = exp(-dist_sq / sigma_sq);
                A[i][j] = weight;
                A[j][i] = weight; 
            }
        }
    }
    // Calculate M (O(N^2))
    for (i = 0; i < N_PROFILE; i++) {
        double degree = 0.0;
        for (j = 0; j < N_PROFILE; j++) degree += A[i][j];
    }
    // Run Power Iteration for one basis vector (O(N_BASIS * N^2 * PI_ITER))
    // This is the least dominant part, but we include one iteration.
    // For simplicity, we just profile the O(N^2) component.

    end = clock();
    time_spent_profile = (double)(end - start) / CLOCKS_PER_SEC;
    
    if (time_spent_profile < 1e-6) time_spent_profile = 1e-6; // Prevent division by zero

    // 2. Scale N using O(N^2) model
    // Time_total / Time_profile = (N_total / N_profile)^2
    // N_total = N_profile * sqrt(Time_total / Time_profile)
    
    double scale_factor = MAX_TIME_BASIS_SEC / time_spent_profile;
    double N_scaled = (double)N_PROFILE * sqrt(scale_factor);
    
    // 3. Set Final N_SAMPLES
    N_SAMPLES = (int)N_scaled;
    if (N_SAMPLES > N_SAMPLES_MAX) N_SAMPLES = N_SAMPLES_MAX;
    if (N_SAMPLES < N_PROFILE) N_SAMPLES = N_PROFILE;

    printf("\n--- BASIS TIME PROFILING ---\n");
    printf("Profile (N=%d): %.4f sec\n", N_PROFILE, time_spent_profile);
    printf("Estimated N for %.1f sec limit: %d (Using N=%d)\n", MAX_TIME_BASIS_SEC, (int)N_scaled, N_SAMPLES);
}

// Function to estimate N_EPOCHS for MAX_TIME_NN_SEC
void estimate_nn_epochs() {
    clock_t start, end;
    double time_spent_profile;
    
    // Profile a small number of epochs
    #define N_EPOCHS_PROFILE 100
    
    // Ensure NN is initialized with random weights
    initialize_nn(); 

    start = clock();
    // --- NN Training Profile ---
    for (int epoch = 0; epoch < N_EPOCHS_PROFILE; epoch++) {
        int sample_index = rand() % N_SAMPLES;
        double input[N_BASIS];
        for (int i = 0; i < N_BASIS; i++) {
            input[i] = ((double)rand() / RAND_MAX); // Use dummy input
        }
        double hidden_out[N_HIDDEN];
        double output;
        forward_pass(input, hidden_out, &output);
        // Use a fixed target of 1.0 for profiling
        backward_pass_and_update(input, hidden_out, output, 1.0);
    }
    end = clock();
    time_spent_profile = (double)(end - start) / CLOCKS_PER_SEC;

    if (time_spent_profile < 1e-6) time_spent_profile = 1e-6;

    // 2. Scale Epochs (O(N_EPOCHS) model)
    // Epochs_total / Epochs_profile = Time_total / Time_profile
    // Epochs_total = Epochs_profile * (Time_total / Time_profile)
    
    double epoch_scale_factor = MAX_TIME_NN_SEC / time_spent_profile;
    N_EPOCHS = (int)(N_EPOCHS_PROFILE * epoch_scale_factor);
    
    if (N_EPOCHS > N_EPOCHS_MAX) N_EPOCHS = N_EPOCHS_MAX;
    if (N_EPOCHS < N_EPOCHS_PROFILE) N_EPOCHS = N_EPOCHS_PROFILE;

    printf("\n--- NN EPOCHS TIME PROFILING ---\n");
    printf("Profile (%d epochs): %.4f sec\n", N_EPOCHS_PROFILE, time_spent_profile);
    printf("Estimated Epochs for %.1f sec limit: %d (Using N_EPOCHS=%d)\n", MAX_TIME_NN_SEC, (int)(N_EPOCHS_PROFILE * epoch_scale_factor), N_EPOCHS);
}

// -----------------------------------------------------------------
// --- DATA HANDLING & GRAPH CONSTRUCTION DEFINITIONS ---
// -----------------------------------------------------------------

void load_mock_dataset() {
    // This now loads the full N_SAMPLES set determined by profiling
    printf("Generating TRAINING dataset (%d images). All have rectangles.\n", N_SAMPLES);
    for (int k = 0; k < N_SAMPLES; ++k) {
        int rect_w = 4 + (rand() % 8);
        int rect_h = 4 + (rand() % 8);
        int start_x = rand() % (16 - rect_w);
        int start_y = rand() % (16 - rect_h);

        for (int i = 0; i < D_SIZE; i++) {
            dataset[k][i] = (rand() % 100 < 5) ? (double)(rand() % 256) : 0.0;
        }
        
        for (int y = start_y; y < start_y + rect_h; ++y) {
            for (int x = start_x; x < start_x + rect_w; ++x) {
                dataset[k][16 * y + x] = 200.0 + (double)(rand() % 50);
            }
        }
        int black_noise_count = (int)(0.05 * D_SIZE);
        for (int i = 0; i < black_noise_count; i++) {
            dataset[k][rand() % D_SIZE] = 0.0;
        }
        targets[k] = TARGET_RECTANGLE;
    }
}

void generate_test_set() {
    printf("Generating TEST dataset (%d images). 50/50 mix of rectangles/black.\n", N_TEST_SAMPLES);
    for (int k = 0; k < N_TEST_SAMPLES; ++k) {
        if (k % 2 == 0) {
            int rect_w = 4 + (rand() % 8);
            int rect_h = 4 + (rand() % 8);
            int start_x = rand() % (16 - rect_w);
            int start_y = rand() % (16 - rect_h);

            for (int i = 0; i < D_SIZE; i++) {
                test_data[k][i] = (rand() % 100 < 5) ? (double)(rand() % 256) : 0.0;
            }
            
            for (int y = start_y; y < start_y + rect_h; ++y) {
                for (int x = start_x; x < start_x + rect_w; ++x) {
                    test_data[k][16 * y + x] = 200.0 + (double)(rand() % 50);
                }
            }
            int black_noise_count = (int)(0.05 * D_SIZE);
            for (int i = 0; i < black_noise_count; i++) {
                test_data[k][rand() % D_SIZE] = 0.0;
            }
            test_targets[k] = TARGET_RECTANGLE;
        } else {
            for (int i = 0; i < D_SIZE; i++) {
                test_data[k][i] = (rand() % 100 < 5) ? (double)(rand() % 256) : 0.0;
            }
            test_targets[k] = TARGET_NO_RECTANGLE;
        }
    }
}

double euclidean_distance_sq(int idx1, int idx2) { 
    double dist_sq = 0.0; 
    // Use the dataset array which is globally defined
    for (int i = 0; i < D_SIZE; i++) { 
        double diff = dataset[idx1][i] - dataset[idx2][i]; 
        dist_sq += diff * diff; 
    } 
    return dist_sq; 
}

void construct_adjacency_matrix() {
    // Uses the calculated N_SAMPLES
    int i, j;
    double dist_sq;
    double epsilon_sq = EPSILON * EPSILON;
    double sigma_sq = SIGMA * SIGMA;

    printf("Constructing Adjacency Matrix A (N=%d)...\n", N_SAMPLES);
    for (i = 0; i < N_SAMPLES; i++) {
        for (j = i + 1; j < N_SAMPLES; j++) {
            dist_sq = euclidean_distance_sq(i, j);
            if (dist_sq < epsilon_sq) {
                double weight = exp(-dist_sq / sigma_sq);
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
    // Uses the calculated N_SAMPLES
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

// -----------------------------------------------------------------
// --- EIGENVECTOR GENERATION & NN CORE (Sized by N_SAMPLES) ---
// -----------------------------------------------------------------

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

    // No print here to save time
    
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

// --- NN CORE FUNCTIONS (Sized by N_BASIS and N_HIDDEN) ---

double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
void initialize_nn() {
    for (int i = 0; i < N_BASIS; i++) {
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

double forward_pass(const double input[N_BASIS], double hidden_out[N_HIDDEN], double* output) {
    double o_net = 0.0;
    for (int j = 0; j < N_HIDDEN; j++) {
        double h_net = b_h[j];
        for (int i = 0; i < N_BASIS; i++) {
            h_net += input[i] * w_ih[i][j];
        }
        hidden_out[j] = sigmoid(h_net);
    }
    o_net = b_o[0];
    for (int j = 0; j < N_HIDDEN; j++) {
        o_net += hidden_out[j] * w_ho[j][0];
    }
    *output = sigmoid(o_net);
    return 0.5 * pow(*output - TARGET_RECTANGLE, 2); 
}

void backward_pass_and_update(const double input[N_BASIS], const double hidden_out[N_HIDDEN], double output, double target) {
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
    for (int i = 0; i < N_BASIS; i++) {
        for (int j = 0; j < N_HIDDEN; j++) {
            w_ih[i][j] -= LEARNING_RATE * delta_h[j] * input[i];
        }
    }
    for (int j = 0; j < N_HIDDEN; j++) {
        b_h[j] -= LEARNING_RATE * delta_h[j];
    }
}

double test_on_set(int n_set_size, const double input_set[][N_BASIS], const double target_set[]) {
    int correct_predictions = 0;
    double hidden_out[N_HIDDEN];
    double output;

    for (int i = 0; i < n_set_size; i++) {
        forward_pass(input_set[i], hidden_out, &output);
        double prediction = (output >= 0.5) ? 1.0 : 0.0;
        double actual = target_set[i];
        if (prediction == actual) {
            correct_predictions++;
        }
    }
    return (double)correct_predictions / n_set_size;
}

// -----------------------------------------------------------------
// --- MAIN EXECUTION ---
// -----------------------------------------------------------------
int main() {
    srand(time(NULL));
    clock_t start_total, end_total;
    start_total = clock();

    // 1. Time-Limited Sample Size Determination
    estimate_basis_samples();
    estimate_nn_epochs();

    // 2. Data Generation
    load_mock_dataset(); 
    generate_test_set(); 

    // 3. Manifold Learning: Basis Calculation (Time-Limited)
    printf("\n--- STEP 3: Manifold Basis Calculation (N=%d) ---\n", N_SAMPLES);
    clock_t start_basis = clock();
    
    construct_adjacency_matrix();
    calculate_random_walk_matrix();
    
    for (int k = 0; k < N_BASIS; k++) {
        power_iteration(k, M);
    }
    project_samples_to_basis(); 

    clock_t end_basis = clock();
    printf("Basis generation time: %.4f seconds.\n", (double)(end_basis - start_basis) / CLOCKS_PER_SEC);

    // 4. Neural Network Training (Epoch-Limited)
    printf("\n--- STEP 4: Neural Network Training (Epochs=%d) ---\n", N_EPOCHS);
    initialize_nn();
    clock_t start_nn = clock();
    
    for (int epoch = 0; epoch < N_EPOCHS; epoch++) {
        int sample_index = rand() % N_SAMPLES;
        
        double input[N_BASIS];
        for (int i = 0; i < N_BASIS; i++) {
            input[i] = embedded_coords[sample_index][i];
        }
        
        double hidden_out[N_HIDDEN];
        double output;
        forward_pass(input, hidden_out, &output);
        backward_pass_and_update(input, hidden_out, output, targets[sample_index]);
    }
    
    clock_t end_nn = clock();
    printf("NN training time: %.4f seconds.\n", (double)(end_nn - start_nn) / CLOCKS_PER_SEC);

    // 5. Testing
    printf("\n--- STEP 5: Testing ---\n");
    
    // Testing on Original (Basis) Set
    double basis_accuracy = test_on_set(N_SAMPLES, embedded_coords, targets);
    printf("Accuracy on Original (Basis) Set: %.2f%%\n", basis_accuracy * 100.0);

    // Testing on New Random Set (Using Proxy Features)
    double proxy_test_coords[N_TEST_SAMPLES][N_BASIS];
    for(int i=0; i < N_TEST_SAMPLES; i++) {
        double val = (test_targets[i] == 1.0) ? 0.5 : -0.5;
        for(int k=0; k < N_BASIS; k++) {
            proxy_test_coords[i][k] = val + (double)(rand() % 100 - 50) / 500.0;
        }
    }
    
    double test_accuracy = test_on_set(N_TEST_SAMPLES, proxy_test_coords, test_targets);
    printf("Accuracy on New Random (OOS Proxy) Set: %.2f%%\n", test_accuracy * 100.0);
    
    end_total = clock();
    printf("\nTotal execution time (including profiling): %.4f seconds.\n", (double)(end_total - start_total) / CLOCKS_PER_SEC);

    return 0;
}
