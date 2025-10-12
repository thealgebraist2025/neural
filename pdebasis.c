#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>

// --- Configuration ---
#define N_SAMPLES 100    // Number of images in the dataset
#define D_SIZE 256       // Dimension of the image vector (16x16 pixels)
#define N_BASIS 8        // Number of basis vectors to generate
#define EPSILON 2500.0   // Max Euclidean distance for connection (Tuned for Mock Data)
#define SIGMA 500.0      // Controls the weight decay (Tuned for Mock Data)
#define MAX_POWER_ITER 5000 
#define PI_TOLERANCE 1.0e-7 
// ---------------------

// Global matrices
double dataset[N_SAMPLES][D_SIZE];  
double A[N_SAMPLES][N_SAMPLES];     // Adjacency Matrix
double D[N_SAMPLES][N_SAMPLES];     // Degree Matrix (Diagonal)
double M[N_SAMPLES][N_SAMPLES];     // Random Walk Matrix (D^-1 * A)
double basis_vectors[N_BASIS][N_SAMPLES]; // Storage for 8 basis vectors (Eigenmaps)
double intrinsic_dist[N_SAMPLES][N_SAMPLES]; // Matrix for intrinsic distance (Geodesic)
double embedded_coords[N_SAMPLES][N_BASIS]; // The 8-D coordinates for each image

// Function prototypes
void load_mock_dataset();
double euclidean_distance_sq(int idx1, int idx2);
void construct_adjacency_matrix();
void calculate_random_walk_matrix();
void matrix_vector_multiply(const double mat[N_SAMPLES][N_SAMPLES], const double vec_in[N_SAMPLES], double vec_out[N_SAMPLES]);
void normalize_vector(double vec[N_SAMPLES]);
double max_vector_diff(const double vec1[N_SAMPLES], const double vec2[N_SAMPLES]);
void orthogonalize_vector(double vec[N_SAMPLES], int current_basis_count);
double power_iteration(int basis_index, double M_current[N_SAMPLES][N_SAMPLES]);
void calculate_intrinsic_distance(int start_node);
void project_samples_to_basis();
double original_euclidean_distance(int idx1, int idx2);
double embedded_distance(int idx1, int idx2);


// -----------------------------------------------------------------
// --- DATA HANDLING & GRAPH CONSTRUCTION ---
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
            // D[i][j] = 0.0; // D is not strictly needed after M calc, but let's clear it
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
// --- POWER ITERATION & EIGENVECTOR GENERATION ---
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

void orthogonalize_vector(double vec[N_SAMPLES], int current_basis_count) {
    // Gram-Schmidt process against previously found basis vectors
    for (int k = 0; k < current_basis_count; k++) {
        double dot_product = 0.0;
        for (int i = 0; i < N_SAMPLES; i++) {
            dot_product += vec[i] * basis_vectors[k][i];
        }
        // vec = vec - (vec . basis_k) * basis_k
        for (int i = 0; i < N_SAMPLES; i++) {
            vec[i] -= dot_product * basis_vectors[k][i];
        }
    }
}

double power_iteration(int basis_index, double M_current[N_SAMPLES][N_SAMPLES]) {
    int iter;
    double f_old[N_SAMPLES];
    double f_new[N_SAMPLES];
    
    // Robust Initialization
    for (iter = 0; iter < N_SAMPLES; iter++) {
        f_old[iter] = (double)(rand() % 200 - 100) / 100.0; 
    }
    normalize_vector(f_old);
    
    // Orthogonalize initial guess
    orthogonalize_vector(f_old, basis_index);
    normalize_vector(f_old);

    printf("\nStarting Power Iteration for Basis #%d...\n", basis_index + 1);
    
    for (iter = 0; iter < MAX_POWER_ITER; iter++) {
        matrix_vector_multiply(M_current, f_old, f_new);
        
        // Orthogonalize the new vector
        orthogonalize_vector(f_new, basis_index);
        
        double diff = max_vector_diff(f_new, f_old);
        normalize_vector(f_new);
        
        for(int i=0; i < N_SAMPLES; i++) {
            f_old[i] = f_new[i];
        }

        if (diff < PI_TOLERANCE) {
            printf("CONVERGED at iteration %d. Final Diff = %e\n", iter, diff);
            break;
        }
    }
    
    for (int i = 0; i < N_SAMPLES; i++) {
        basis_vectors[basis_index][i] = f_old[i];
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
    
    for (int i = 0; i < N_SAMPLES; i++) {
        dist[i] = DBL_MAX; 
        visited[i] = 0;
    }
    dist[start_node] = 0.0;

    for (int count = 0; count < N_SAMPLES - 1; count++) {
        
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

        for (int v = 0; v < N_SAMPLES; v++) {
            if (visited[v] == 0 && A[u][v] > DBL_EPSILON) { 
                double edge_cost = 1.0 / A[u][v]; 
                if (dist[u] != DBL_MAX && dist[u] + edge_cost < dist[v]) {
                    dist[v] = dist[u] + edge_cost;
                }
            }
        }
    }

    for (int i = 0; i < N_SAMPLES; i++) {
        intrinsic_dist[start_node][i] = dist[i];
    }
}

// -----------------------------------------------------------------
// --- PROJECTION & DISTANCE COMPARISON ---
// -----------------------------------------------------------------

void project_samples_to_basis() {
    printf("\nProjecting samples onto the %d Eigenmaps...\n", N_BASIS);
    
    // For Laplacian Eigenmaps, the k-th Eigenmap (basis_vectors[k]) directly gives 
    // the k-th coordinate for all N_SAMPLES images.
    for (int i = 0; i < N_SAMPLES; i++) { // Loop over each image (sample)
        for (int k = 0; k < N_BASIS; k++) { // Loop over each basis vector (new dimension)
            embedded_coords[i][k] = basis_vectors[k][i];
        }
    }
}

double original_euclidean_distance(int idx1, int idx2) {
    // Calculates the Euclidean distance in the original 256D pixel space.
    return sqrt(euclidean_distance_sq(idx1, idx2));
}

double embedded_distance(int idx1, int idx2) {
    // Calculates the Euclidean distance in the new 8D coordinate space.
    double dist_sq = 0.0;
    for (int k = 0; k < N_BASIS; k++) { // Loop over the 8 new dimensions
        double diff = embedded_coords[idx1][k] - embedded_coords[idx2][k];
        dist_sq += diff * diff;
    }
    return sqrt(dist_sq);
}


// -----------------------------------------------------------------
// --- MAIN EXECUTION ---
// -----------------------------------------------------------------
int main() {
    srand(time(NULL));

    // 1. Setup and Graph Construction
    load_mock_dataset();
    construct_adjacency_matrix();
    calculate_random_walk_matrix();
    
    // 2. Generating N_BASIS (8) eigenvectors
    for (int k = 0; k < N_BASIS; k++) {
        power_iteration(k, M);
    }

    // 3. Display Basis Results
    printf("\n--- Resulting Basis Vectors (8 Eigenmaps) ---\n");
    for(int k = 0; k < N_BASIS; k++) {
        printf("Basis Vector f%d (Sample values): ", k + 1);
        for(int i = 0; i < 5; i++) {
            printf("%.6f ", basis_vectors[k][i]);
        }
        printf("\n");
    }
    
    // 4. Calculate and Print Intrinsic Distance
    printf("\n--- Intrinsic (Geodesic) Distance Calculation ---\n");
    
    for(int i = 0; i < N_SAMPLES; i++) {
        calculate_intrinsic_distance(i);
    }
    
    printf("Geodesic distance from Image 0 to other images:\n");
    printf("Node 0 to Node 1: %.2f\n", intrinsic_dist[0][1]);
    printf("Node 0 to Node 5: %.2f\n", intrinsic_dist[0][5]);
    
    // 5. Projection and Distance Comparison
    project_samples_to_basis();
    
    printf("\n--- Comparison of Original vs. Embedded Distance ---\n");
    
    int img_A = 10;
    int img_B = 12;
    int img_C = 50;
    int img_D = 90;

    printf("Image Pair A (%d) and B (%d) - Likely Similar:\n", img_A, img_B);
    printf("  Original 256D Euclidean Distance: %.2f\n", original_euclidean_distance(img_A, img_B));
    printf("  Embedded 8D Euclidean Distance:   %.2f\n", embedded_distance(img_A, img_B));
    printf("  Intrinsic (Geodesic) Distance:    %.2f\n", intrinsic_dist[img_A][img_B]);

    printf("\nImage Pair C (%d) and D (%d) - Likely Dissimilar:\n", img_C, img_D);
    printf("  Original 256D Euclidean Distance: %.2f\n", original_euclidean_distance(img_C, img_D));
    printf("  Embedded 8D Euclidean Distance:   %.2f\n", embedded_distance(img_C, img_D));
    printf("  Intrinsic (Geodesic) Distance:    %.2f\n", intrinsic_dist[img_C][img_D]);

    printf("\nNOTE: In successful manifold learning, the Embedded 8D Distance should ideally correlate closely with the Intrinsic Distance.\n");

    return 0;
}
