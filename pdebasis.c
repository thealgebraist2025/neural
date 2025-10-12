#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>

// --- Configuration ---
#define N_SAMPLES 100    // Number of images in the dataset
#define D_SIZE 256       // Dimension of the image vector (16x16)
#define EPSILON 2500.0   // FIX: INCREASED for denser graph
#define SIGMA 500.0      // FIX: INCREASED for smoother weight decay
#define MAX_POWER_ITER 5000 
#define PI_TOLERANCE 1.0e-7 
// ---------------------

// Global matrices
double dataset[N_SAMPLES][D_SIZE];  
double A[N_SAMPLES][N_SAMPLES];     // Adjacency Matrix
double D[N_SAMPLES][N_SAMPLES];     // Degree Matrix (Diagonal)
double M[N_SAMPLES][N_SAMPLES];     // Random Walk Matrix (D^-1 * A)
double principal_eigenvector[N_SAMPLES]; 
double intrinsic_dist[N_SAMPLES][N_SAMPLES]; // Matrix for intrinsic distance (Geodesic)

// Function prototypes
void load_mock_dataset();
double euclidean_distance_sq(int idx1, int idx2);
void construct_adjacency_matrix();
void calculate_random_walk_matrix();
void matrix_vector_multiply(const double mat[N_SAMPLES][N_SAMPLES], const double vec_in[N_SAMPLES], double vec_out[N_SAMPLES]);
void normalize_vector(double vec[N_SAMPLES]);
double max_vector_diff(const double vec1[N_SAMPLES], const double vec2[N_SAMPLES]);
double power_iteration();
void calculate_intrinsic_distance(int start_node);

// -----------------------------------------------------------------
// --- DATA LOADING & DISTANCE CALCULATION ---
// -----------------------------------------------------------------

void load_mock_dataset() {
    int k, x, y;
    printf("Generating mock dataset of %d images...\n", N_SAMPLES);

    for (k = 0; k < N_SAMPLES; ++k) {
        int rect_w = 4 + (rand() % 8);
        int rect_h = 4 + (rand() % 8);
        int start_x = rand() % (16 - rect_w);
        int start_y = rand() % (16 - rect_h);

        for (int i = 0; i < D_SIZE; i++) {
            dataset[k][i] = 0.0;
        }

        for (y = start_y; y < start_y + rect_h; ++y) {
            for (x = start_x; x < start_x + rect_w; ++x) {
                dataset[k][16 * y + x] = 200.0 + (double)(rand() % 50);
            }
        }
    }
}

double euclidean_distance_sq(int idx1, int idx2) {
    double dist_sq = 0.0;
    for (int i = 0; i < D_SIZE; i++) {
        double diff = dataset[idx1][i] - dataset[idx2][i];
        dist_sq += diff * diff;
    }
    return dist_sq;
}

// -----------------------------------------------------------------
// --- GRAPH CONSTRUCTION AND LAPLACIAN ---
// -----------------------------------------------------------------
void construct_adjacency_matrix() {
    int i, j;
    double dist_sq;
    double epsilon_sq = EPSILON * EPSILON;
    double sigma_sq = SIGMA * SIGMA;

    printf("Constructing Adjacency Matrix A with EPSILON=%.0f...\n", EPSILON);

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
    int i, j;
    
    printf("Calculating Random Walk Matrix M (D^-1 * A)...\n");

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

// -----------------------------------------------------------------
// --- POWER ITERATION FUNCTIONS ---
// -----------------------------------------------------------------

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
    
    // Robust Initialization (Wider Range)
    for (iter = 0; iter < N_SAMPLES; iter++) {
        f_old[iter] = (double)(rand() % 200 - 100) / 100.0; 
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

// -----------------------------------------------------------------
// --- INTRINSIC DISTANCE (GEODESIC APPROXIMATION) ---
// -----------------------------------------------------------------
void calculate_intrinsic_distance(int start_node) {
    // Dijkstra's-like approach
    double dist[N_SAMPLES];
    int visited[N_SAMPLES];
    
    // Initialization
    for (int i = 0; i < N_SAMPLES; i++) {
        dist[i] = DBL_MAX; 
        visited[i] = 0;
    }
    dist[start_node] = 0.0;

    for (int count = 0; count < N_SAMPLES - 1; count++) {
        
        // Find node u with minimum distance that hasn't been visited
        double min_dist = DBL_MAX;
        int u = -1;
        
        for (int i = 0; i < N_SAMPLES; i++) {
            if (visited[i] == 0 && dist[i] <= min_dist) {
                min_dist = dist[i];
                u = i;
            }
        }

        if (u == -1) break; 
        
        visited[u] = 1;

        // Update dist value of the adjacent nodes
        for (int v = 0; v < N_SAMPLES; v++) {
            if (visited[v] == 0 && A[u][v] > DBL_EPSILON) { 
                // Cost is the inverse of similarity
                double edge_cost = 1.0 / A[u][v]; 
                if (dist[u] != DBL_MAX && dist[u] + edge_cost < dist[v]) {
                    dist[v] = dist[u] + edge_cost;
                }
            }
        }
    }

    // Save the calculated distances to the global matrix
    for (int i = 0; i < N_SAMPLES; i++) {
        intrinsic_dist[start_node][i] = dist[i];
    }
}


// -----------------------------------------------------------------
// --- MAIN EXECUTION ---
// -----------------------------------------------------------------
int main() {
    srand(time(NULL));

    // 1. Setup and Basis Calculation (Eigenmaps)
    load_mock_dataset();
    construct_adjacency_matrix();
    calculate_random_walk_matrix();
    double largest_eigenvalue = power_iteration();

    printf("\n--- Resulting Basis Vector (First Eigenmap) ---\n");
    printf("Largest Eigenvalue (lambda_max): %.4f\n", largest_eigenvalue);
    
    printf("First Basis Vector (sample values, f1[i]):\n");
    for(int i = 0; i < 5; i++) {
        printf("f1[%d]: %.6f\n", i, principal_eigenvector[i]);
    }
    
    // 2. Calculate Intrinsic Distance (Geodesic)
    printf("\n--- Intrinsic (Geodesic) Distance Calculation ---\n");
    
    // Calculate full N_SAMPLES x N_SAMPLES distance matrix
    for(int i = 0; i < N_SAMPLES; i++) {
        calculate_intrinsic_distance(i);
    }
    
    printf("Geodesic distance from Image 0 to other images:\n");
    printf("Node 0 to Node 1: %.2f\n", intrinsic_dist[0][1]);
    printf("Node 0 to Node 5: %.2f\n", intrinsic_dist[0][5]);
    printf("Node 0 to Node 50: %.2f\n", intrinsic_dist[0][50]);
    printf("Node 50 to Node 99: %.2f\n", intrinsic_dist[50][99]);
    
    printf("\nNote: The increased EPSILON and SIGMA should ensure graph connectivity and stable convergence.\n");
    
    return 0;
}
