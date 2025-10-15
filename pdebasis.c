#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// --- SIMULATION PARAMETERS ---
#define N 20                // Grid size (N x N)
#define MAX_ITER 100        // Number of time steps to run
#define ALPHA 0.1           // Thermal diffusivity coefficient (alpha)
#define DX 1.0              // Spatial step size (Delta x and Delta y)
#define H_SQUARE (DX * DX)  // h^2 for convenience

// Time step calculation based on stability criteria: Delta t <= h^2 / (4 * alpha)
// We choose a slightly smaller value for stability margin.
#define DT (0.95 * H_SQUARE / (4.0 * ALPHA))

// Diffusion constant 'r' for the explicit scheme: r = alpha * Delta t / h^2
#define R_CONSTANT (ALPHA * DT / H_SQUARE)

// --- FUNCTION PROTOTYPES ---
void initialize_grid(double grid[N][N]);
void run_simulation(double current_grid[N][N], double next_grid[N][N]);
void print_grid(const double grid[N][N]);

int main() {
    // Check stability criterion
    if (R_CONSTANT > 0.25) {
        fprintf(stderr, "Error: Stability criterion R <= 0.25 is violated (R=%.4f).\n", R_CONSTANT);
        fprintf(stderr, "Adjust DT, ALPHA, or DX.\n");
        return 1;
    }

    printf("2D Heat Equation Solver (Explicit FDM)\n");
    printf("Parameters: N=%d, Iterations=%d, R=%.4f (Stable)\n\n", N, MAX_ITER, R_CONSTANT);

    // Grid storage: need two grids to calculate the next state from the current state
    double grid_current[N][N];
    double grid_next[N][N];

    // Initialize the grid with boundary conditions and initial hotspot
    initialize_grid(grid_current);

    // Run the main simulation loop
    run_simulation(grid_current, grid_next);

    return 0;
}

// Initializes the grid: 0 on boundaries, a high-temp hotspot in the center
void initialize_grid(double grid[N][N]) {
    // 1. Set all points to 0.0 (low boundary temperature)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            grid[i][j] = 0.0;
        }
    }

    // 2. Set the initial condition (a central heat source, or "hotspot")
    // We create a small 4x4 hotspot in the center of the grid (from index N/2-2 to N/2+1)
    int center_start = N / 2 - 2;
    int center_end = N / 2 + 1;

    for (int i = center_start; i <= center_end; i++) {
        for (int j = center_start; j <= center_end; j++) {
            // Check if the point is within the domain (not on the boundary, though this is unlikely for N>=4)
            if (i > 0 && i < N - 1 && j > 0 && j < N - 1) {
                grid[i][j] = 100.0; // Initial high temperature
            }
        }
    }

    printf("--- Initial Grid (t=0) ---\n");
    print_grid(grid);
}

// Runs the FDM simulation for MAX_ITER time steps
void run_simulation(double current_grid[N][N], double next_grid[N][N]) {
    double (*u_current)[N] = current_grid;
    double (*u_next)[N] = next_grid;

    // Time marching loop
    for (int k = 0; k < MAX_ITER; k++) {
        // 1. Apply the FDM update rule to the interior points
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                // Laplacian approximation (finite difference)
                double laplacian = u_current[i + 1][j] + u_current[i - 1][j] +
                                   u_current[i][j + 1] + u_current[i][j - 1] -
                                   4.0 * u_current[i][j];

                // Explicit update equation: u_next = u_current + r * (Laplacian)
                u_next[i][j] = u_current[i][j] + R_CONSTANT * laplacian;
            }
        }

        // 2. Apply Boundary Conditions (Dirichlet: fixed temperature)
        // Since we initialized all boundaries to 0.0 and never update them,
        // we must explicitly copy those values from the current step to the next step.
        // Copy top/bottom rows
        for (int j = 0; j < N; j++) {
            u_next[0][j] = u_current[0][j];
            u_next[N - 1][j] = u_current[N - 1][j];
        }
        // Copy left/right columns (excluding corners already copied)
        for (int i = 1; i < N - 1; i++) {
            u_next[i][0] = u_current[i][0];
            u_next[i][N - 1] = u_current[i][N - 1];
        }


        // 3. Swap the grids: next_grid becomes the current_grid for the next iteration
        double (*temp)[N] = u_current;
        u_current = u_next;
        u_next = temp;

        // Print intermediate or final state
        if ((k + 1) % (MAX_ITER / 4) == 0 || k == MAX_ITER - 1) {
            printf("--- Grid after Time Step t=%d ---\n", k + 1);
            print_grid(u_current);
        }
    }

    printf("Simulation completed after %d iterations.\n", MAX_ITER);
}

// Prints the grid to the console with minimal formatting
void print_grid(const double grid[N][N]) {
    // Only print a subset of the grid for visualization in the console
    int step = N / 10;
    if (step == 0) step = 1;

    for (int i = 0; i < N; i += step) {
        for (int j = 0; j < N; j += step) {
            // Print temperature rounded to the nearest integer
            printf("%4.0f ", grid[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}