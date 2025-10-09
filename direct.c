#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Define M_PI explicitly as it's not guaranteed by the C standard
#define M_PI 3.14159265358979323846

// --- Configuration ---
#define GRID_SIZE 16
#define NUM_DEFORMATIONS 2 // alpha_1 (Slant), alpha_2 (Curvature)
#define NUM_POINTS 200     // Number of points to sample the curve
#define LEARNING_RATE 0.05 // ADJUSTED: Increased learning rate for faster convergence
#define ITERATIONS 10
#define GRADIENT_EPSILON 1e-3 // ADJUSTED: Increased epsilon for detectable change in discrete image

// --- OCaml-like Immutability & Const Correctness ---

// Mutable/Immutable Type: Represents a coordinate.
typedef struct {
    double x; // MUTABLE during calculation
    double y; // MUTABLE during calculation
} Point;

// Immutable Type: Fixed Definition of the Ideal Letter 'J' (control points)
typedef struct {
    const Point stroke_1_start;
    const Point stroke_1_mid;
    const Point stroke_1_end;
} Ideal_Curve_Params;

// Mutable Type: The Learnable Deformation Coefficients (alpha_k)
typedef struct {
    double alpha[NUM_DEFORMATIONS]; // MUTABLE
} Deformation_Coefficients;

// Immutable Type: The observed input image
typedef const double Observed_Image[GRID_SIZE][GRID_SIZE];

// Mutable Type: The image buffer used for rasterization
typedef double Generated_Image[GRID_SIZE][GRID_SIZE]; // MUTABLE

// --- Fixed Ideal Curve and Basis Functions (IMMUTABLE) ---

// Define the Ideal 'J' form in normalized coordinates [0, 1]
const Ideal_Curve_Params IDEAL_J = {
    .stroke_1_start = {.x = 0.5, .y = 0.1}, // Top Bar
    .stroke_1_mid = {.x = 0.5, .y = 0.7},   // Vertical stem
    .stroke_1_end = {.x = 0.2, .y = 0.9}    // Bottom Hook
};

// Deformation Basis Function Phi_k(t)
void apply_deformation(Point *point, const double alpha[NUM_DEFORMATIONS]) {
    // Deformation 1: Slant (Shear Transform)
    point->x = point->x + alpha[0] * (point->y - 0.5);

    // Deformation 2: Curvature/Width
    point->x = point->x + alpha[1] * sin(M_PI * point->y);
}

// --- Forward Model: Rasterization (Curve to Image) ---

/**
 * @brief Generates a point on the 'J' curve using the ideal form and deformations.
 */
Point get_deformed_point(const double t, const Ideal_Curve_Params *const params, const double alpha[NUM_DEFORMATIONS]) {
    Point p = {.x = 0.0, .y = 0.0};
    
    // Simplified J curve as a piecewise linear approximation for t in [0, 1]
    if (t < 0.3) {
        const double segment_t = t / 0.3;
        p.x = params->stroke_1_start.x;
        p.y = params->stroke_1_start.y + (params->stroke_1_mid.y - params->stroke_1_start.y) * segment_t;
    } else if (t < 0.8) {
        const double segment_t = (t - 0.3) / 0.5;
        p.x = params->stroke_1_mid.x;
        p.y = params->stroke_1_mid.y + (params->stroke_1_mid.y - params->stroke_1_end.y) * segment_t;
    } else {
        const double segment_t = (t - 0.8) / 0.2;
        p.x = params->stroke_1_mid.x + (params->stroke_1_end.x - params->stroke_1_mid.x) * segment_t;
        p.y = params->stroke_1_end.y;
    }

    apply_deformation(&p, alpha);

    // Scale to pixel grid and clamp (MUTABLE operations on p)
    p.x = fmax(0.0, fmin(GRID_SIZE - 1.0, p.x * GRID_SIZE));
    p.y = fmax(0.0, fmin(GRID_SIZE - 1.0, p.y * GRID_SIZE));

    return p;
}

/**
 * @brief Rasterizes the deformed curve onto the image grid (Forward Model G).
 */
void draw_curve(const double alpha[NUM_DEFORMATIONS], Generated_Image img) {
    // 1. Clear the canvas (MUTABLE operation on img)
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            img[i][j] = 0.0;
        }
    }

    // 2. Sample and draw points (Simple line drawing)
    for (int i = 0; i <= NUM_POINTS; i++) {
        const double t = (double)i / NUM_POINTS;
        const Point current_p = get_deformed_point(t, &IDEAL_J, alpha);

        // Simple pixel darkening
        const int px = (int)round(current_p.x);
        const int py = (int)round(current_p.y);

        if (px >= 0 && px < GRID_SIZE && py >= 0 && py < GRID_SIZE) {
            // MUTABLE operation on img
            img[py][px] = fmin(1.0, img[py][px] + 0.5);
        }

        // Simple neighbor smoothing (MUTABLE operations on img)
        if (py + 1 < GRID_SIZE) img[py + 1][px] = fmin(1.0, img[py + 1][px] + 0.1);
        if (py - 1 >= 0) img[py - 1][px] = fmin(1.0, img[py - 1][px] + 0.1);
        if (px + 1 < GRID_SIZE) img[py][px + 1] = fmin(1.0, img[py][px + 1] + 0.1);
        if (px - 1 >= 0) img[py][px - 1] = fmin(1.0, img[py][px - 1] + 0.1);
    }
}

// --- Inverse Problem: Loss and Gradient ---

/**
 * @brief Calculates the L2 Loss (Squared Error) between generated and observed image.
 */
double calculate_loss(const Generated_Image generated, const Observed_Image observed) {
    double loss = 0.0;
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            const double error = observed[i][j] - generated[i][j];
            loss += error * error;
        }
    }
    return loss;
}

/**
 * @brief Simulates the Gradient calculation using Finite Differences.
 */
void calculate_gradient(const Observed_Image observed, const Deformation_Coefficients *const alpha, const double loss_base, double grad_out[NUM_DEFORMATIONS]) {
    const double epsilon = GRADIENT_EPSILON; 
    Generated_Image generated_perturbed; // MUTABLE buffer for perturbed image

    for (int k = 0; k < NUM_DEFORMATIONS; k++) {
        // 1. Perturb alpha_k (MUTABLE temporary copy)
        Deformation_Coefficients alpha_perturbed = *alpha;
        alpha_perturbed.alpha[k] += epsilon; // MUTABLE temporary change

        // 2. Calculate Loss_perturbed (Forward Model G)
        draw_curve(alpha_perturbed.alpha, generated_perturbed);
        const double loss_perturbed = calculate_loss(generated_perturbed, observed);

        // 3. Compute Gradient (Finite Difference)
        grad_out[k] = (loss_perturbed - loss_base) / epsilon;
    }
}

// --- Display Functions ---

void print_image(const char *const title, const double image[GRID_SIZE][GRID_SIZE]) {
    printf("\n%s (16x16):\n", title);
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            // Simplified visualization: only use space or hash ('#')
            if (image[i][j] < 0.3) printf(" "); // Background
            else printf("#"); // Foreground
        }
        printf("\n");
    }
}

// --- Simulation Setup (Generating a Target/Observed Image) ---

/**
 * @brief Creates a synthetic observed image with known deformation and noise.
 * @param observed_out The mutable image array to fill.
 */
void generate_observed_target(Generated_Image observed_out) {
    // 1. Define the TRUE, underlying deformation for the target (IMMUTABLE)
    const double true_alpha[NUM_DEFORMATIONS] = {-0.1, 0.05}; // Slanted left, slightly curved

    // 2. Rasterize the TRUE deformed curve (Signal)
    draw_curve(true_alpha, observed_out);

    // 3. Add random noise (Stochastic Process/Error) (MUTABLE operation on observed_out)
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            double noise = ((double)rand() / RAND_MAX - 0.5) * 0.3; // White noise [-0.15, 0.15]
            observed_out[i][j] = fmax(0.0, fmin(1.0, observed_out[i][j] + noise));
        }
    }
}


// --- Main Execution ---

int main(void) {
    // 1. Setup (IMMUTABLE data)
    srand(42); // Seed for reproducible noise
    Generated_Image observed_image; // Observed image (mutable buffer)
    generate_observed_target(observed_image);
    print_image("Observed Noisy Target Image (J)", observed_image); 

    // 2. Initialization (MUTABLE data)
    // Initial guess for the deformation coefficients (starting near zero/ideal)
    Deformation_Coefficients alpha_hat = {
        .alpha = {0.0, 0.0} // Starting guess: Ideal 'J'
    };
    
    // Gradient and generated image buffers (MUTABLE)
    double gradient[NUM_DEFORMATIONS];
    Generated_Image generated_image;
    double loss;

    printf("\n--- Optimization (Gradient Descent) ---\n");
    printf("It | Loss     | a_1 (Slant) | a_2 (Curve) | da_1 | da_2\n");
    printf("----------------------------------------------------------\n");

    // 3. Training/Estimation Loop (MUTABLE iterations)
    for (int t = 0; t <= ITERATIONS; t++) {
        // Forward Pass: Generate image based on current alpha_hat (MUTABLE)
        draw_curve(alpha_hat.alpha, generated_image);
        
        // Calculate Loss (IMMUTABLE operation)
        loss = calculate_loss(generated_image, observed_image);
        
        // Calculate Gradient (IMMUTABLE operation)
        calculate_gradient(observed_image, &alpha_hat, loss, gradient);
        
        // Print status
        printf("%02d | %8.5f | %8.4f | %8.4f |", t, loss, alpha_hat.alpha[0], alpha_hat.alpha[1]);

        // Gradient Descent Update (MUTABLE operation on alpha_hat)
        if (t < ITERATIONS) {
            const double delta_a1 = LEARNING_RATE * gradient[0];
            const double delta_a2 = LEARNING_RATE * gradient[1];
            
            alpha_hat.alpha[0] -= delta_a1;
            alpha_hat.alpha[1] -= delta_a2;

            printf(" %+.4f | %+.4f\n", delta_a1, delta_a2);
        } else {
            printf(" --- Final ---\n");
        }
    }

    // 4. Final Result (IMMUTABLE visualization)
    draw_curve(alpha_hat.alpha, generated_image);
    print_image("Final Denoised/Estimated Image", generated_image);

    printf("\n--- Conclusion ---\n");
    printf("Estimated Slant (a_1): %.4f (True was -0.10)\n", alpha_hat.alpha[0]);
    printf("Estimated Curvature (a_2): %.4f (True was 0.05)\n", alpha_hat.alpha[1]);

    return 0;
}