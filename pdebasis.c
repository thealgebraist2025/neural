#define _XOPEN_SOURCE 

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h> 
#include <float.h>
#include <string.h> 

// --- Configuration ---
#define GRID_SIZE 32       
#define D_SIZE (GRID_SIZE * GRID_SIZE) // 1024

// **Network Configuration**
#define N_INPUT D_SIZE         // x_1 to x_1024 
#define N_HIDDEN 64            // z_1 to z_64 
#define N_OUTPUT 3             // Output: (Center X, Center Y, Radius)

// **Training Parameters**
#define NUM_IMAGES 100         
#define BATCH_SIZE 10          
#define N_TRAINING_EPOCHS 50000      
#define REPORT_FREQ 5000             
#define INITIAL_LEARNING_RATE 0.00001 
#define COORD_WEIGHT 1.0           
#define MIN_RADIUS 3           
#define MAX_RADIUS 10.0    
#define SVD_REGULARIZATION_LAMBDA 1e-5 // Lambda for Condition Number Regularization

// **Algebraic Parameters**
#define MAX_ITER_SVD 100           // Max iterations for Power Iteration
#define SVD_TOLERANCE 1e-6         // Convergence tolerance

// Global Data & Matrices 
double w_fh[N_INPUT][N_HIDDEN];    
double b_h[N_HIDDEN]; 
double w_ho[N_HIDDEN][N_OUTPUT];   
double b_o[N_OUTPUT];

double single_images[NUM_IMAGES][D_SIZE]; 
int target_properties[NUM_IMAGES][N_OUTPUT]; 

// --- Helper Macros and Functions ---
#define NORMALIZE_COORD(coord) ((double)(coord) / (GRID_SIZE - 1.0))
#define NORMALIZE_RADIUS(radius) ((double)(radius) / MAX_RADIUS)
double poly_activation(double z_net) { return z_net * z_net; } 
double poly_derivative(double z_net) { return 2.0 * z_net; }

// --- Matrix Algebra Implementations ---

// 1. Matrix Multiplication: C = A * B (N_HIDDEN x N_INPUT * N_INPUT x N_HIDDEN) -> (64x64)
// Computes A = W_fh^T * W_fh (Gram Matrix)
void matrix_mult_gram(const double W[N_INPUT][N_HIDDEN], double A[N_HIDDEN][N_HIDDEN]) {
    for (int i = 0; i < N_HIDDEN; i++) { // Row of A (Corresponds to column of W_fh)
        for (int j = 0; j < N_HIDDEN; j++) { // Column of A (Corresponds to column of W_fh)
            A[i][j] = 0.0;
            for (int k = 0; k < N_INPUT; k++) { // Sum over N_INPUT (k)
                // W_fh^T[i][k] * W_fh[k][j] = W_fh[k][i] * W_fh[k][j]
                A[i][j] += W[k][i] * W[k][j]; 
            }
        }
    }
}

// 2. Vector Norm
double vector_norm(const double v[N_HIDDEN]) {
    double sum_sq = 0.0;
    for (int i = 0; i < N_HIDDEN; i++) sum_sq += v[i] * v[i];
    return sqrt(sum_sq);
}

// 3. Matrix-Vector Multiplication: y = A * x (64x64 * 64x1)
void mat_vec_mult(const double A[N_HIDDEN][N_HIDDEN], const double x[N_HIDDEN], double y[N_HIDDEN]) {
    for (int i = 0; i < N_HIDDEN; i++) {
        y[i] = 0.0;
        for (int j = 0; j < N_HIDDEN; j++) {
            y[i] += A[i][j] * x[j];
        }
    }
}

// 4. POWER ITERATION: Finds the largest eigenvalue of A (lambda_max)
// This is used for both lambda_max(A) and lambda_max(A^-1)
double power_iteration(const double A[N_HIDDEN][N_HIDDEN], double initial_vector[N_HIDDEN]) {
    double v[N_HIDDEN]; 
    double Av[N_HIDDEN];
    double lambda = 0.0;

    // Initialize v with the provided vector (or random if needed)
    if (initial_vector == NULL) {
        for(int i = 0; i < N_HIDDEN; i++) v[i] = ((double)rand() / RAND_MAX * 2.0 - 1.0);
    } else {
        memcpy(v, initial_vector, N_HIDDEN * sizeof(double));
    }

    double norm = vector_norm(v);
    for (int i = 0; i < N_HIDDEN; i++) v[i] /= norm; // Normalize v

    for (int iter = 0; iter < MAX_ITER_SVD; iter++) {
        mat_vec_mult(A, v, Av);
        
        double new_lambda = 0.0;
        for (int i = 0; i < N_HIDDEN; i++) new_lambda += v[i] * Av[i];
        
        if (fabs(new_lambda - lambda) < SVD_TOLERANCE) {
            lambda = new_lambda;
            break;
        }
        
        lambda = new_lambda;
        
        // Normalize Av and set as new v
        norm = vector_norm(Av);
        for (int i = 0; i < N_HIDDEN; i++) v[i] = Av[i] / norm;
    }
    
    // Copy final eigenvector for potential reuse
    if (initial_vector != NULL) {
        memcpy(initial_vector, v, N_HIDDEN * sizeof(double));
    }
    
    return lambda;
}

// 5. Gaussian Elimination for Matrix Inversion (Needed for Inverse Power Iteration)
// Computes A_inv = A^-1
void invert_matrix(const double A[N_HIDDEN][N_HIDDEN], double A_inv[N_HIDDEN][N_HIDDEN]) {
    double M[N_HIDDEN][2 * N_HIDDEN];
    
    // Setup augmented matrix [A | I]
    for (int i = 0; i < N_HIDDEN; i++) {
        for (int j = 0; j < N_HIDDEN; j++) {
            M[i][j] = A[i][j];
            M[i][j + N_HIDDEN] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Gaussian Elimination (Forward phase)
    for (int i = 0; i < N_HIDDEN; i++) {
        // Pivot selection (partial pivoting)
        int max_row = i;
        for (int k = i + 1; k < N_HIDDEN; k++) {
            if (fabs(M[k][i]) > fabs(M[max_row][i])) {
                max_row = k;
            }
        }
        if (max_row != i) {
            for (int k = i; k < 2 * N_HIDDEN; k++) {
                double temp = M[i][k];
                M[i][k] = M[max_row][k];
                M[max_row][k] = temp;
            }
        }

        // Check for singularity (determinant is near zero)
        if (fabs(M[i][i]) < 1e-9) { 
             // Inverting near-singular matrix; return identity as a failsafe
             for (int r = 0; r < N_HIDDEN; r++) {
                 for (int c = 0; c < N_HIDDEN; c++) {
                     A_inv[r][c] = (r == c) ? 1.0 : 0.0;
                 }
             }
             return; 
        }

        // Normalize the pivot row
        double pivot = M[i][i];
        for (int j = i; j < 2 * N_HIDDEN; j++) {
            M[i][j] /= pivot;
        }

        // Eliminate other rows
        for (int k = 0; k < N_HIDDEN; k++) {
            if (k != i) {
                double factor = M[k][i];
                for (int j = i; j < 2 * N_HIDDEN; j++) {
                    M[k][j] -= factor * M[i][j];
                }
            }
        }
    }

    // Extract the inverse matrix
    for (int i = 0; i < N_HIDDEN; i++) {
        for (int j = 0; j < N_HIDDEN; j++) {
            A_inv[i][j] = M[i][j + N_HIDDEN];
        }
    }
}


// 6. Primary Algebraic Function: Singular Value Calculation
void calculate_singular_values(const double W[N_INPUT][N_HIDDEN], double *sigma_max, double *sigma_min) {
    double A[N_HIDDEN][N_HIDDEN];
    double A_inv[N_HIDDEN][N_HIDDEN];
    double v_init[N_HIDDEN];

    // Compute Gram Matrix A = W^T * W
    matrix_mult_gram(W, A);

    // --- 1. Find Largest Singular Value (sigma_max) ---
    // sigma_max = sqrt(lambda_max(A))
    double lambda_max = power_iteration(A, v_init);
    *sigma_max = sqrt(lambda_max);

    // --- 2. Find Smallest Singular Value (sigma_min) ---
    // sigma_min = 1 / sqrt(lambda_max(A^-1))
    
    // We compute A_inv
    invert_matrix(A, A_inv);
    
    // We compute lambda_max(A_inv) using Power Iteration
    double lambda_max_inv = power_iteration(A_inv, v_init);
    
    if (lambda_max_inv < 1e-9) {
        *sigma_min = 0.0; // Near-singular matrix
    } else {
        *sigma_min = 1.0 / sqrt(lambda_max_inv);
    }
}

// 7. Calculates the Condition Number
double calculate_condition_number(double sigma_max, double sigma_min) {
    if (sigma_min < 1e-8) return DBL_MAX; 
    return sigma_max / sigma_min;
}

// 8. Algebraic Regularization Update (Applied to W_fh)
// This applies a gradient-based update to penalize high condition numbers.
void apply_algebraic_regularization_update(double W[N_INPUT][N_HIDDEN], double condition_number, double update_rate) {
    if (condition_number == DBL_MAX || condition_number > 1000.0) {
        // If ill-conditioned, apply an L2-like penalty to stabilize the matrix.
        // This is a proxy for the complex gradient of the condition number.
        double penalty_scale = SVD_REGULARIZATION_LAMBDA * log(condition_number);
        for (int i = 0; i < N_INPUT; i++) {
            for (int j = 0; j < N_HIDDEN; j++) {
                W[i][j] -= update_rate * penalty_scale * W[i][j]; 
            }
        }
    }
}


// --- NN Core Functions (Unchanged, included for compilation) ---

void draw_filled_circle(double image[D_SIZE], int cx, int cy, int r) {
    for (int i = 0; i < D_SIZE; i++) image[i] = 0.0; 
    for (int y = 0; y < GRID_SIZE; y++) {
        for (int x = 0; x < GRID_SIZE; x++) {
            if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r) {
                image[GRID_SIZE * y + x] = 1.0; 
            }
        }
    }
}

void generate_circle_image(int index) {
    int min_center = MAX_RADIUS;
    int max_center = GRID_SIZE - MAX_RADIUS - 1;
    srand((unsigned int)time(NULL) + index * 100); 
    int *properties = target_properties[index];
    int cx = min_center + (rand() % (max_center - min_center + 1));
    int cy = min_center + (rand() % (max_center - min_center + 1));
    int r = (int)MIN_RADIUS + (rand() % ((int)MAX_RADIUS - (int)MIN_RADIUS + 1));
    draw_filled_circle(single_images[index], cx, cy, r);
    properties[0] = cx; properties[1] = cy; properties[2] = r;
}

void load_train_case(double input[N_INPUT], double target[N_OUTPUT]) {
    int img_idx = rand() % NUM_IMAGES;
    memcpy(input, single_images[img_idx], D_SIZE * sizeof(double));
    const int *p = target_properties[img_idx];
    target[0] = NORMALIZE_COORD(p[0]); target[1] = NORMALIZE_COORD(p[1]); target[2] = NORMALIZE_RADIUS(p[2]); 
}

void initialize_nn() {
    double fan_in_h = (double)N_INPUT;
    double limit_h = sqrt(1.0 / fan_in_h); 
    double fan_in_o = (double)N_HIDDEN;
    double limit_o = sqrt(1.0 / fan_in_o); 
    for (int i = 0; i < N_INPUT; i++) for (int j = 0; j < N_HIDDEN; j++) w_fh[i][j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * limit_h; 
    for (int j = 0; j < N_HIDDEN; j++) { 
        b_h[j] = 0.0; 
        for (int k = 0; k < N_OUTPUT; k++) w_ho[j][k] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * limit_o;
    }
    for (int k = 0; k < N_OUTPUT; k++) b_o[k] = 0.0;
}

void forward_pass(const double input[N_INPUT], double hidden_net[N_HIDDEN], double hidden_out[N_HIDDEN], double output[N_OUTPUT]) {
    for (int j = 0; j < N_HIDDEN; j++) {
        double h_net = b_h[j];
        for (int i = 0; i < N_INPUT; i++) h_net += input[i] * w_fh[i][j]; 
        hidden_net[j] = h_net;
        hidden_out[j] = poly_activation(h_net);
    }
    for (int k = 0; k < N_OUTPUT; k++) {
        double o_net = b_o[k]; 
        for (int j = 0; j < N_HIDDEN; j++) o_net += hidden_out[j] * w_ho[j][k]; 
        output[k] = o_net;
    }
}


// --- Training Function with Real Algebraic Update ---

void train_nn() {
    // ... (All local variables and gradient accumulators from previous response)
    double input[N_INPUT], target[N_OUTPUT];
    double hidden_net[N_HIDDEN], hidden_out[N_HIDDEN], output[N_OUTPUT];
    
    double grad_w_fh_acc[N_INPUT][N_HIDDEN] = {0.0};
    double grad_b_h_acc[N_HIDDEN] = {0.0};
    double grad_w_ho_acc[N_HIDDEN][N_OUTPUT] = {0.0};
    double grad_b_o_acc[N_OUTPUT] = {0.0};
    
    double cumulative_loss_report = 0.0;
    int samples_processed_in_report = 0;
    
    // Algebraic variables
    double sigma_max, sigma_min;
    
    printf("--- TRAINING PHASE START (Batch SGD with REAL SVD Regularization) ---\n");
    
    for (int epoch = 0; epoch < N_TRAINING_EPOCHS; epoch++) {
        
        // --- BATCH LOOP (Calculates and accumulates standard gradients) ---
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            
            load_train_case(input, target);
            forward_pass(input, hidden_net, hidden_out, output);
            
            double delta_o[N_OUTPUT];
            double delta_h[N_HIDDEN]; 
            double error_h[N_HIDDEN];

            // Backpropagation (Standard SGD)
            for (int k = 0; k < N_OUTPUT; k++) delta_o[k] = (output[k] - target[k]) * COORD_WEIGHT; 
            for (int j = 0; j < N_HIDDEN; j++) {
                error_h[j] = 0.0;
                for (int k = 0; k < N_OUTPUT; k++) error_h[j] += delta_o[k] * w_ho[j][k];
                delta_h[j] = error_h[j] * poly_derivative(hidden_net[j]);
            }
            
            // Accumulate Gradients
            for (int k = 0; k < N_OUTPUT; k++) {
                grad_b_o_acc[k] += delta_o[k];
                for (int j = 0; j < N_HIDDEN; j++) grad_w_ho_acc[j][k] += delta_o[k] * hidden_out[j];
            }
            for (int j = 0; j < N_HIDDEN; j++) {
                grad_b_h_acc[j] += delta_h[j];
                for (int i = 0; i < N_INPUT; i++) grad_w_fh_acc[i][j] += delta_h[j] * input[i];
            }
            
            double loss = 0.0; 
            for (int k = 0; k < N_OUTPUT; k++) loss += (output[k] - target[k]) * (output[k] - target[k]) * COORD_WEIGHT;
            cumulative_loss_report += loss; 
            samples_processed_in_report++;
        } // END BATCH LOOP

        // --- WEIGHT UPDATE (SGD + Algebraic Regularization) ---
        double inverse_batch_size = 1.0 / BATCH_SIZE;
        double update_rate = INITIAL_LEARNING_RATE * inverse_batch_size;
        
        // 1. Update W_ho and b_o
        for (int k = 0; k < N_OUTPUT; k++) {
            b_o[k] -= update_rate * grad_b_o_acc[k];
            grad_b_o_acc[k] = 0.0; 
            for (int j = 0; j < N_HIDDEN; j++) {
                w_ho[j][k] -= update_rate * grad_w_ho_acc[j][k];
                grad_w_ho_acc[j][k] = 0.0; 
            }
        }
        
        // 2. Update W_fh and b_h (Standard SGD portion)
        for (int j = 0; j < N_HIDDEN; j++) {
            b_h[j] -= update_rate * grad_b_h_acc[j];
            grad_b_h_acc[j] = 0.0; 
            for (int i = 0; i < N_INPUT; i++) {
                w_fh[i][j] -= update_rate * grad_w_fh_acc[i][j];
                grad_w_fh_acc[i][j] = 0.0; 
            }
        }
        
        // 3. Algebraic Regularization Update (W_fh)
        calculate_singular_values(w_fh, &sigma_max, &sigma_min);
        double condition_number = calculate_condition_number(sigma_max, sigma_min);
        apply_algebraic_regularization_update(w_fh, condition_number, update_rate);
        
        if ((epoch + 1) % REPORT_FREQ == 0) {
            printf("  Epoch: %6d | Avg Loss: %7.6f | Cond($\\mathbf{W}_{fh}$): %.2e | $\\sigma_{max}$: %.4f\n", 
                   epoch + 1, cumulative_loss_report / samples_processed_in_report, 
                   condition_number, sigma_max);
            cumulative_loss_report = 0.0; 
            samples_processed_in_report = 0;
        }
    }
    printf("--- TRAINING PHASE COMPLETE ---\n");
}


// --- Testing and Main Function (Unchanged, included for compilation) ---

void print_image_and_path(const double input[N_INPUT], const double target[N_OUTPUT], const double output[N_OUTPUT]) {
    int target_cx = (int)round(target[0] * (GRID_SIZE - 1.0));
    int target_cy = (int)round(target[1] * (GRID_SIZE - 1.0));
    int target_r = (int)round(target[2] * MAX_RADIUS);

    int output_cx = (int)round(output[0] * (GRID_SIZE - 1.0));
    int output_cy = (int)round(output[1] * (GRID_SIZE - 1.0));
    int output_r = (int)round(output[2] * MAX_RADIUS);
    
    printf("  Target: (CX, CY, R) = (%2d, %2d, %2d)\n", target_cx, target_cy, target_r);
    printf("  Output: (CX, CY, R) = (%2d, %2d, %2d)\n", output_cx, output_cy, output_r);
    
    printf("  Image (%dx%d):\n", GRID_SIZE, GRID_SIZE);
    int print_size = 16;
    int start_x = (GRID_SIZE - print_size) / 2;
    int start_y = (GRID_SIZE - print_size) / 2;
    
    for (int y = start_y; y < start_y + print_size; y++) {
        printf("    ");
        for (int x = start_x; x < start_x + print_size; x++) {
            int index = GRID_SIZE * y + x;
            if (x == output_cx && y == output_cy) {
                 printf("X"); // Predicted Center
            } else if (x == target_cx && y == target_cy) {
                 printf("C"); // True Center
            } else if (input[index] > 0.5) {
                printf("#");
            } else {
                printf(".");
            }
        }
        printf("\n");
    }
    printf("    (Partial view: Centers are marked)\n");
}

void test_nn(int total_test_runs) {
    double input[N_INPUT], target[N_OUTPUT], hidden_net[N_HIDDEN], hidden_out[N_HIDDEN], output[N_OUTPUT];
    double cumulative_test_loss = 0.0;
    int accurate_count = 0;
    
    printf("\n--- TESTING PHASE START (%d cases) ---\n", total_test_runs);
    
    for (int i = 0; i < total_test_runs; i++) {
        int cx = (int)MAX_RADIUS + (rand() % (GRID_SIZE - 2 * (int)MAX_RADIUS));
        int cy = (int)MAX_RADIUS + (rand() % (GRID_SIZE - 2 * (int)MAX_RADIUS));
        int r = (int)MIN_RADIUS + (rand() % ((int)MAX_RADIUS - (int)MIN_RADIUS + 1));
        
        draw_filled_circle(input, cx, cy, r);
        target[0] = NORMALIZE_COORD(cx); 
        target[1] = NORMALIZE_COORD(cy); 
        target[2] = NORMALIZE_RADIUS(r); 
        
        forward_pass(input, hidden_net, hidden_out, output);

        double loss = 0.0;
        double error_threshold = 0.05; 
        int accurate = 1;

        for (int k = 0; k < N_OUTPUT; k++) {
            double diff = output[k] - target[k];
            loss += diff * diff * COORD_WEIGHT;
            if (fabs(diff) > error_threshold) {
                accurate = 0;
            }
        }
        cumulative_test_loss += loss;
        if (accurate) accurate_count++;
    }
    
    printf("\nTEST SUMMARY:\n");
    printf("Total Test Cases: %d\n", total_test_runs);
    printf("Average Loss per Test Case: %.6f\n", cumulative_test_loss / total_test_runs);
    printf("Accurate Predictions (within 5%% norm. error): %d / %d (%.2f%%)\n", 
           accurate_count, total_test_runs, (double)accurate_count / total_test_runs * 100.0);
    printf("--------------------------------------------------\n");

    // --- Final Algebraic Metrics ---
    double sigma_max, sigma_min;
    calculate_singular_values(w_fh, &sigma_max, &sigma_min);
    double condition_number = calculate_condition_number(sigma_max, sigma_min);

    printf("\n--- ALGEBRAIC POST-MORTEM (Final $\\mathbf{W}_{fh}$) ---\n");
    printf("1. Condition Number: %.4e\n", condition_number);
    printf("2. Largest $\\sigma_{max}$: %.4f\n", sigma_max);
    printf("3. Smallest $\\sigma_{min}$: %.4e\n", sigma_min);
    printf("--------------------------------------------------\n");

    // VISUALIZATION: Show 2 random examples
    printf("\n--- VISUALIZATION: 2 Random Test Cases ---\n");
    for (int i = 0; i < 2; i++) {
        int cx = (int)MAX_RADIUS + (rand() % (GRID_SIZE - 2 * (int)MAX_RADIUS));
        int cy = (int)MAX_RADIUS + (rand() % (GRID_SIZE - 2 * (int)MAX_RADIUS));
        int r = (int)MIN_RADIUS + (rand() % ((int)MAX_RADIUS - (int)MIN_RADIUS + 1));
        
        draw_filled_circle(input, cx, cy, r);
        target[0] = NORMALIZE_COORD(cx); target[1] = NORMALIZE_COORD(cy); target[2] = NORMALIZE_RADIUS(r); 
        forward_pass(input, hidden_net, hidden_out, output);
        
        printf("\nTest Case #%d:\n", i + 1);
        print_image_and_path(input, target, output);
    }
}

int main() {
    srand(time(NULL));

    printf("--- 32x32 Circle Recognition NN with Real Singular Value Analysis ---\n");
    
    // 1. Initialize and Generate Data
    initialize_nn();
    for (int i = 0; i < NUM_IMAGES; i++) {
        generate_circle_image(i);
    }
    printf("Data setup complete. %d training images generated.\n", NUM_IMAGES);

    // 2. Train Network
    train_nn();

    // 3. Test Network
    test_nn(100);

    return 0;
}
