#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// --- Configuration ---
#define GRID_SIZE 16
#define NUM_DEFORMATIONS 2 // alpha_1 (Slant), alpha_2 (Curvature)
#define NUM_POINTS 200     // Number of points to sample the curve
#define LEARNING_RATE 0.001
#define ITERATIONS 10

// --- OCaml-like Immutability & Const Correctness ---

// Immutable Type: Represents a coordinate (fixed definition)
typedef struct {
    const double x;
    const double y;
} Point;

// Immutable Type: Fixed Definition of the Ideal Letter 'J' (control points)
typedef struct {
    const Point stroke_1_start;
    const Point stroke_1_mid;
    const Point stroke_1_end;
} Ideal_Curve_Params;

// Mutable Type: The Learnable Deformation Coefficients (alpha_k)
// These are the parameters we are trying to estimate.
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
// These define the *way* the curve is allowed to deform.
// For simplicity, we define Phi_k(t) as a function operating on a point (x, y)
// Phi_1 (Slant): Applies a horizontal shift proportional to the vertical position.
// Phi_2 (Curvature): Applies an x-shift based on a parabolic function.
void apply_deformation(Point *point, const double alpha[NUM_DEFORMATIONS]) {
    // Deformation 1: Slant (Shear Transform)
    // Horizontal shift is proportional to alpha_1 and the y-coordinate.
    point->x = point->x + alpha[0] * (point->y - 0.5);

    // Deformation 2: Curvature/Width
    // Horizontal shift is proportional to alpha_2 and a parabolic function
    // (max shift at y=0.5, zero shift at y=0 and y=1).
    point->x = point->x + alpha[1] * sin(M_PI * point->y);
}

// --- Forward Model: Rasterization (Curve to Image) ---

/**
 * @brief Generates a point on the 'J' curve using the ideal form and deformations.
 * @param t The path parameter, t in [0, 1].
 * @param params The immutable ideal curve definition.
 * @param alpha The mutable deformation coefficients.
 * @return A point structure representing the deformed curve coordinate.
 */
Point get_deformed_point(const double t, const Ideal_Curve_Params *const params, const double alpha[NUM_DEFORMATIONS]) {
    Point p = {.x = 0.0, .y = 0.0};
    
    // Simplified J curve as a piecewise linear approximation for t in [0, 1]
    if (t < 0.3) {
        // Top line/start of stem (0.0 to 0.3)
        double segment_t = t / 0.3;
        p.x = params->stroke_1_start.x;
        p.y = params->stroke_1_start.y + (params->stroke_1_mid.y - params->stroke_1_start.y) * segment_t;
    } else if (t < 0.8) {
        // Vertical stem (0.3 to 0.8)
        double segment_t = (t - 0.3) / 0.5;
        p.x = params->stroke_1_mid.x;
        p.y = params->stroke_1_mid.y + (params->stroke_1_mid.y - params->stroke_1_end.y) * segment_t;
    } else {
        // Hook (0.8 to 1.0)
        double segment_t = (t - 0.8) / 0.2;
        p.x = params->stroke_1_mid.x + (params->stroke_1_end.x - params->stroke_1_mid.x) * segment_t;
        p.y = params->stroke_1_end.y;
    }

    // Apply the deformation (Phi_k * alpha_k)
    apply_deformation(&p, alpha);

    // Scale to pixel grid and clamp
    p.x = fmax(0.0, fmin(GRID_SIZE - 1.0, p.x * GRID_SIZE));
    p.y = fmax(0.0, fmin(GRID_SIZE - 1.0, p.y * GRID_SIZE));

    return p;
}

/**
 * @brief Rasterizes the deformed curve onto the image grid (Forward Model G).
 * @param alpha The mutable deformation coefficients.
 * @param img The mutable image buffer to draw onto.
 */
void draw_curve(const double alpha[NUM_DEFORMATIONS], Generated_Image img) {
    // 1. Clear the canvas (MUTABLE operation on img)
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            img[i][j] = 0.0;
        }
    }

    // 2. Sample and draw points (Simple line drawing)
    Point prev_p = {.x = -1, .y = -1};
    for (int i = 0; i <= NUM_POINTS; i++) {
        const double t = (double)i / NUM_POINTS;
        Point current_p = get_deformed_point(t, &IDEAL_J, alpha);

        // Simple pixel darkening (no sophisticated line algorithm needed for simulation)
        int px = (int)round(current_p.x);
        int py = (int)round(current_p.y);

        if (px >= 0 && px < GRID_SIZE && py >= 0 && py < GRID_SIZE) {
            // MUTABLE operation on img
            img[py][px] = fmin(1.0, img[py][px] + 0.5);
        }

        // Simple neighbor smoothing
        if (py + 1 < GRID_SIZE) img[py + 1][px] = fmin(1.0, img[py + 1][px] + 0.1);
        if (py - 1 >= 0) img[py - 1][px] = fmin(1.0, img[py - 1][px] + 0.1);
        if (px + 1 < GRID_SIZE) img[py][px + 1] = fmin(1.0, img[py][px + 1] + 0.1);
        if (px - 1 >= 0) img[py][px - 1] = fmin(1.0, img[py][px - 1] + 0.1);
    }
}

// --- Inverse Problem: Loss and Gradient ---

/**
 * @brief Calculates the L2 Loss (Squared Error) between generated and observed image.
 * @param generated The immutable generated image.
 * @param observed The immutable observed input image.
 * @return The calculated loss value.
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
 * @param observed The immutable observed input image.
 * @param alpha The mutable coefficients to calculate gradient around.
 * @param loss_base The calculated loss at the current alpha.
 * @param grad_out The mutable array to store the calculated gradients.
 */
void calculate_gradient(const Observed_Image observed, const Deformation_Coefficients *const alpha, const double loss_base, double grad_out[NUM_DEFORMATIONS]) {
    const double epsilon = 1e-4; // Step size for finite difference
    Generated_Image generated_perturbed;

    for (int k = 0; k < NUM_DEFORMATIONS; k++) {
        // 1. Perturb alpha_k (MUTABLE temporary change)
        Deformation_Coefficients alpha_perturbed = *alpha;
        alpha_perturbed.alpha[k] += epsilon;

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
            // Use simple char map to show grayscale
            if (image[i][j] < 0.1) printf(" ");
            else if (image[i][j] < 0.3) printf(".");
            else if (image[i][j] < 0.6) printf("-");
            else printf("#");
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
    printf("It | Loss     | a_1 (Slant) | a_2 (Curve) | Est. Change\n");
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
        printf("%02d | %8.5f | %8.4f | %8.4f | ", t, loss, alpha_hat.alpha[0], alpha_hat.alpha[1]);

        // Gradient Descent Update (MUTABLE operation on alpha_hat)
        if (t < ITERATIONS) {
            double delta_a1 = LEARNING_RATE * gradient[0];
            double delta_a2 = LEARNING_RATE * gradient[1];
            
            alpha_hat.alpha[0] -= delta_a1;
            alpha_hat.alpha[1] -= delta_a2;

            printf("da1: %+.4f", delta_a1);
        } else {
            printf("--- Final ---");
        }
        printf("\n");
    }

    // 4. Final Result (IMMUTABLE visualization)
    draw_curve(alpha_hat.alpha, generated_image);
    print_image("Final Denoised/Estimated Image", generated_image);

    printf("\n--- Conclusion ---\n");
    printf("Estimated Slant (a_1): %.4f (True was -0.10)\n", alpha_hat.alpha[0]);
    printf("Estimated Curvature (a_2): %.4f (True was 0.05)\n", alpha_hat.alpha[1]);

    // The residual error between the observed and the estimated image is the noise estimate.
    
    return 0;
}