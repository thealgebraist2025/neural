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

// Global matrices
double dataset[N_SAMPLES][D_SIZE];  
double A[N_SAMPLES][N_SAMPLES];     // Adjacency Matrix
double M[N_SAMPLES][N_SAMPLES];     // Random Walk Matrix (D^-1 * A)
double principal_eigenvector[N_SAMPLES]; 
double intrinsic_dist[N_SAMPLES][N_SAMPLES]; // New matrix for intrinsic distance

// Function prototypes (many omitted for brevity)
void load_mock_dataset();
void construct_adjacency_matrix();
void calculate_random_walk_matrix();
double power_iteration();
void calculate_intrinsic_distance(int start_node);

// [load_mock_dataset, construct_adjacency_matrix, calculate_random_walk_matrix, matrix_vector_multiply, normalize_vector, max_vector_diff are assumed to be implemented correctly from the last response]

// *************************************************************
// FIX: Power Iteration for improved stability
// *************************************************************
double power_iteration() {
    int iter;
    double f_old[N_SAMPLES];
    double f_new[N_SAMPLES];
    
    // 1. Initialize f_old with a slightly less uniform random vector
    for (iter = 0; iter < N_SAMPLES; iter++) {
        // Init with a wider range to break symmetry
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


// *************************************************************
// NEW: Calculate Intrinsic Distance (Geodesic Approx.)
// *************************************************************
void calculate_intrinsic_distance(int start_node) {
    // Naive Dijkstra's-like approach for fixed-source shortest path
    double dist[N_SAMPLES];
    int visited[N_SAMPLES];
    
    // Initialization
    for (int i = 0; i < N_SAMPLES; i++) {
        dist[i] = DBL_MAX; // Use DBL_MAX for infinity
        visited[i] = 0;
    }
    dist[start_node] = 0.0;

    // Iterative search
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

        if (u == -1) break; // All remaining nodes unreachable
        
        visited[u] = 1;

        // Update dist value of the adjacent nodes
        for (int v = 0; v < N_SAMPLES; v++) {
            // A[u][v] > 0 means there's an edge. The intrinsic distance is the inverse of the weight.
            if (visited[v] == 0 && A[u][v] > DBL_EPSILON) { 
                double edge_cost = 1.0 / A[u][v]; // Cost is the inverse of similarity
                if (dist[u] + edge_cost < dist[v]) {
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


// *************************************************************
// Main function and output logic
// *************************************************************
int main() {
    srand(time(NULL));

    // 1. Setup and Basis Calculation (Eigenmaps)
    load_mock_dataset();
    construct_adjacency_matrix();
    calculate_random_walk_matrix();
    double largest_eigenvalue = power_iteration();

    printf("\n--- Resulting Basis Vector (First Eigenmap) ---\n");
    printf("Largest Eigenvalue (lambda_max): %.4f\n", largest_eigenvalue);
    
    printf("Sample values (f1[i]):\n");
    for(int i = 0; i < 5; i++) {
        printf("f1[%d]: %.6f\n", i, principal_eigenvector[i]);
    }
    
    // 2. Calculate Intrinsic Distance (Geodesic)
    printf("\n--- Intrinsic (Geodesic) Distance Calculation ---\n");
    int source_node = 0;
    
    // Calculate full N_SAMPLES x N_SAMPLES distance matrix
    for(int i = 0; i < N_SAMPLES; i++) {
        calculate_intrinsic_distance(i);
    }
    
    printf("Intrinsic distance from Image 0 to other images:\n");
    printf("Node 0 to Node 1: %.2f\n", intrinsic_dist[0][1]);
    printf("Node 0 to Node 5: %.2f\n", intrinsic_dist[0][5]);
    printf("Node 0 to Node 50: %.2f\n", intrinsic_dist[0][50]);
    
    // Test the intrinsic distance between two distant nodes (Image 50 and 99)
    printf("Node 50 to Node 99: %.2f\n", intrinsic_dist[50][99]);
    
    printf("\nNote: DBL_MAX (%e) indicates unreachable nodes in the graph.\n", DBL_MAX);
    
    return 0;
}
