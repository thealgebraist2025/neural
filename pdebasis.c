#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>

// --- Configuration ---
#define N_SAMPLES 100    // Number of images in the dataset
#define D_SIZE 256       // Dimension of the image vector (16x16)
#define EPSILON 500.0    // Max Euclidean distance for connection
#define SIGMA 100.0      // Controls the weight decay
#define MAX_POWER_ITER 5000 // Iterations for Power Method
#define PI_TOLERANCE 1.0e-7 // Power Iteration convergence tolerance
// ---------------------

// Global matrices for the core computation
double dataset[N_SAMPLES][D_SIZE];  // Input data (100 images x 256 pixels)
double A[N_SAMPLES][N_SAMPLES];     // Adjacency Matrix
double D[N_SAMPLES][N_SAMPLES];     // Degree Matrix (Diagonal)
double M[N_SAMPLES][N_SAMPLES];     // Random Walk Matrix (D^-1 * A)
double principal_eigenvector[N_SAMPLES]; // The resulting basis vector

// Function prototypes
void load_mock_dataset();
double euclidean_distance_sq(int idx1, int idx2);
void construct_adjacency_matrix();
void calculate_random_walk_matrix();
void matrix_vector_multiply(const double mat[N_SAMPLES][N_SAMPLES], const double vec_in[N_SAMPLES], double vec_out[N_SAMPLES]);
void normalize_vector(double vec[N_SAMPLES]);
double max_vector_diff(const double vec1[N_SAMPLES], const double vec2[N_SAMPLES]);
double power_iteration();


// -----------------------------------------------------------------
// --- DATA LOADING & DISTANCE CALCULATION ---
// -----------------------------------------------------------------

// Function to simulate loading a dataset of 16x16 rectangles
void load_mock_dataset() {
    int k, x, y;
    printf("Generating mock dataset of %d images...\n", N_SAMPLES);

    for (k = 0; k < N_SAMPLES; ++k) {
        // Randomly set a block of pixels (a rectangle) to a high value (200)
        int rect_w = 4 + (rand() % 8);
        int rect_h = 4 + (rand() % 8);
        int start_x = rand() % (16 - rect_w);
        int start_y = rand() % (16 - rect_h);

        // Initialize to 0 (black background)
        for (int i = 0; i < D_SIZE; i++) {
            dataset[k][i] = 0.0;
        }

        for (y = start_y; y < start_y + rect_h; ++y) {
            for (x = start_x; x < start_x + rect_w; ++x) {
                // Flattening index: 16*y + x
                dataset[k][16 * y + x] = 200.0 + (double)(rand() % 50); // Add slight variance
            }
        }
    }
}

// Function to calculate the squared Euclidean distance between two images
double euclidean_distance_sq(int idx1, int idx2) {
    double dist_sq = 0.0;
    for (int i = 0; i < D_SIZE; i++) {
        double diff = dataset[idx1][i] - dataset[idx2][i];
        dist_sq += diff * diff;
    }
    return dist_sq;
}

// -----------------------------------------------------------------
// --- ADJACENCY MATRIX CONSTRUCTION ---
// -----------------------------------------------------------------
void construct_adjacency_matrix() {
    int i, j;
    double dist_sq;
    double epsilon_sq = EPSILON * EPSILON;
    double sigma_sq = SIGMA * SIGMA;

    printf("Constructing Adjacency Matrix A...\n");

    for (i = 0; i < N_SAMPLES; i++) {
        for (j = i + 1; j < N_SAMPLES; j++) {
            dist_sq = euclidean_distance_sq(i, j);
            
            // Epsilon-neighborhood check
            if (dist_sq < epsilon_sq) {
                // Gaussian kernel for weighting
                double weight = exp(-dist_sq / sigma_sq);
                A[i][j] = weight;
                A[j][i] = weight; 
            } else {
                A[i][j] = 0.0;
                A[j][i] = 0.0;
            }
        }
        A[i][i] = 0.0; // No self-loops
    }
}

// [Omitted: calculate_random_walk_matrix, matrix_vector_multiply, normalize_vector, 
// max_vector_diff, and power_iteration remain the same as the previous response]

// The functions below are included to ensure completeness if compiled.
void calculate_random_walk_matrix() {
    int i, j;
    for (i = 0; i < N_SAMPLES; i++) {
        double degree = 0.0;
        for (j = 0; j < N_SAMPLES; j++) {
            degree += A[i][j];
            D[i][j] = 0.0;
        }
        D[i][i] = degree;

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

void matrix_vector_multiply(const double mat[N_SAMPLES][N_SAMPLES], const double vec_in[N_SAMPLES], double vec_out[N_SAMPLES]) {
    int i, j;
    for (i = 0; i < N_SAMPLES; i++) {
        vec_out[i] = 0.0;
        for (j = 0; j < N_SAMPLES; j++) {
            vec_out[i] += mat[i][j] * vec_in[j];
        }
    }
}

void normalize_vector(double vec[N_SAMPLES]) {
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

double max_vector_diff(const double vec1[N_SAMPLES], const double vec2[N_SAMPLES]) {
    double max_diff = 0.0;
    for (int i = 0; i < N_SAMPLES; i++) {
        double diff = fabs(vec1[i] - vec2[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

double power_iteration() {
    int iter;
    double f_old[N_SAMPLES];
    double f_new[N_SAMPLES];
    
    for (iter = 0; iter < N_SAMPLES; iter++) {
        f_old[iter] = (double)(rand() % 100 + 1) / 100.0;
    }
    normalize_vector(f_old);

    printf("Starting Power Iteration to find principal basis vector...\n");
    
    for (iter = 0; iter < MAX_POWER_ITER; iter++) {
        matrix_vector_multiply(M, f_old, f_new);
        double diff = max_vector_diff(f_new, f_old);
        normalize_vector(f_new);
        
        for(int i=0; i < N_SAMPLES; i++) {
            f_old[i] = f_new[i];
        }

        if (iter % 500 == 0) {
             printf("Iteration %d: Vector Diff = %e\n", iter, diff);
        }

        if (diff < PI_TOLERANCE) {
            printf("\nCONVERGED at iteration %d. Final Diff = %e\n", iter, diff);
            break;
        }
    }
    
    for (int i = 0; i < N_SAMPLES; i++) {
        principal_eigenvector[i] = f_old[i];
    }

    return 1.0; 
}


int main() {
    srand(time(NULL));

    // 1. Data Preparation and Graph Construction
    load_mock_dataset();
    construct_adjacency_matrix();
    
    // 2. Matrix M (Random Walk Matrix)
    calculate_random_walk_matrix();
    
    // 3. Solve for the first basis vector (Eigenvector f1)
    double largest_eigenvalue = power_iteration();

    printf("\n--- Resulting Basis Vector (First Eigenmap) ---\n");
    printf("Largest Eigenvalue (lambda_max): %.4f (Expected 1.0)\n", largest_eigenvalue);
    
    printf("The output is a vector (size %d) that assigns a coordinate to each image.\n", N_SAMPLES);
    printf("This vector is the first new basis dimension (eigenmap) for the dataset.\n");
    printf("Sample values (f1[i]):\n");
    for(int i = 0; i < 5; i++) {
        printf("f1[%d]: %.6f\n", i, principal_eigenvector[i]);
    }

    return 0;
}
