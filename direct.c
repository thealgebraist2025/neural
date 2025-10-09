#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Define M_PI explicitly as it's not guaranteed by the C standard
#define M_PI 3.14159265358979323846

// --- Global Configuration ---
#define GRID_SIZE 16
#define NUM_DEFORMATIONS 2 // alpha_1 (Slant), alpha_2 (Curvature)
#define NUM_FEATURES 8     // ADDED: Using 8 directional projections for loss calculation
#define NUM_POINTS 200
#define LEARNING_RATE 0.001 // Stable step size
#define ITERATIONS 100
#define GRADIENT_EPSILON 0.01 

// --- OCaml-like Immutability & Const Correctness ---

typedef struct {
    double x; // MUTABLE during calculation
    double y; // MUTABLE during calculation
} Point;

typedef struct {
    const Point stroke_1_start;
    const Point stroke_1_mid;
    const Point stroke_1_end;
} Ideal_Curve_Params; // IMMUTABLE

typedef struct {
    double alpha[NUM_DEFORMATIONS]; // MUTABLE
} Deformation_Coefficients;

typedef const double Observed_Image[GRID_SIZE][GRID_SIZE]; // IMMUTABLE
typedef double Generated_Image[GRID_SIZE][GRID_SIZE]; // MUTABLE
typedef double Feature_Vector[NUM_FEATURES]; // MUTABLE

// --- Fixed Ideal Curve and Basis Functions (IMMUTABLE) ---

// Define the Ideal 'J' form in normalized coordinates [0, 1]
const Ideal_Curve_Params IDEAL_J = {
    .stroke_1_start = {.x = 0.5, .y = 0.1}, 
    .stroke_1_mid = {.x = 0.5, .y = 0.7},   
    .stroke_1_end = {.x = 0.2, .y = 0.9}    
};

// Deformation Basis Function Phi_k(t)
void apply_deformation(Point *point, const double alpha[NUM_DEFORMATIONS]) {
    // Deformation 1: Slant (Shear Transform)
    point->x = point->x + alpha[0] * (point->y - 0.5);

    // Deformation 2: Curvature/Width
    point->x = point->x + alpha[1] * sin(M_PI * point->y);
}

/**
 * @brief Generates a point on the 'J' curve using the ideal form and deformations.
 */
Point get_deformed_point(const double t, const Ideal_Curve_Params *const params, const double alpha[NUM_DEFORMATIONS]) {
    Point p = {.x = 0.0, .y = 0.0};
    
    // Piecewise Linear Interpolation
    if (t < 0.5) {
        const double segment_t = t / 0.5;
        p.x = params->stroke_1_start.x + (params->stroke_1_mid.x - params->stroke_1_start.x) * segment_t;
        p.y = params->stroke_1_start.y + (params->stroke_1_mid.y - params->stroke_1_start.y) * segment_t;
    } else {
        const double segment_t = (t - 0.5) / 0.5;
        p.x = params->stroke_1_mid.x + (params->stroke_1_end.x - params->stroke_1_mid.x) * segment_t;
        p.y = params->stroke_1_mid.y + (params->stroke_1_end.y - params->stroke_1_mid.y) * segment_t;
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

    // 2. Sample and draw points
    for (int i = 0; i <= NUM_POINTS; i++) {
        const double t = (double)i / NUM_POINTS;
        const Point current_p = get_deformed_point(t, &IDEAL_J, alpha);

        // Main pixel
        const int px = (int)round(current_p.x);
        const int py = (int)round(current_p.y);

        if (px >= 0 && px < GRID_SIZE && py >= 0 && py < GRID_SIZE) {
            img[py][px] = fmin(1.0, img[py][px] + 0.5); // MUTABLE
        }

        // Neighbor smoothing (basic anti-aliasing)
        if (py + 1 < GRID_SIZE) img[py + 1][px] = fmin(1.0, img[py + 1][px] + 0.1); // MUTABLE
        if (py - 1 >= 0) img[py - 1][px] = fmin(1.0, img[py - 1][px] + 0.1); // MUTABLE
        if (px + 1 < GRID_SIZE) img[py][px + 1] = fmin(1.0, img[py][px + 1] + 0.1); // MUTABLE
        if (px - 1 >= 0) img[py][px - 1] = fmin(1.0, img[py][px - 1] + 0.1); // MUTABLE
    }
}

// --- Feature Extraction and Loss (The new Stable Core) ---

/**
 * @brief Extracts 8 geometric projection features from the image (Directional Moments).
 * Projections are onto (1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1).
 */
void extract_geometric_features(const Generated_Image img, Feature_Vector features_out) {
    // 8 normalized basis vectors (x, y)
    const double vectors[NUM_FEATURES][2] = {
        {1.0, 0.0}, {M_SQRT1_2, M_SQRT1_2}, {0.0, 1.0}, {-M_SQRT1_2, M_SQRT1_2},
        {-1.0, 0.0}, {-M_SQRT1_2, -M_SQRT1_2}, {0.0, -1.0}, {M_SQRT1_2, -M_SQRT1_2}
    };
    
    // Center point for coordinate calculation
    const double center = (GRID_SIZE - 1.0) / 2.0;

    // Initialize feature vector (MUTABLE)
    for (int k = 0; k < NUM_FEATURES; k++) {
        features_out[k] = 0.0;
    }

    // Iterate over all pixels
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            const double intensity = img[i][j];
            if (intensity < 0.1) continue; // Skip near-background pixels

            // Vector from center to pixel (j is x-axis, i is y-axis)
            const double vx = (double)j - center;
            const double vy = (double)i - center;
            
            // Project the mass vector onto all 8 basis vectors
            for (int k = 0; k < NUM_FEATURES; k++) {
                // Dot product: projection = (vx * basis_x + vy * basis_y) * intensity
                const double projection = (vx * vectors[k][0] + vy * vectors[k][1]) * intensity;
                features_out[k] += projection; // MUTABLE
            }
        }
    }
}

/**
 * @brief Calculates the L2 Loss (Squared Error) between feature vectors.
 */
double calculate_feature_loss(const Feature_Vector generated, const Feature_Vector observed) {
    double loss = 0.0;
    for (int k = 0; k < NUM_FEATURES; k++) {
        const double error = observed[k] - generated[k];
        loss += error * error;
    }
    return loss;
}

/**
 * @brief Simulates the Gradient calculation using Finite Differences.
 */
void calculate_gradient(const Feature_Vector observed_features, const Deformation_Coefficients *const alpha, const double loss_base, double grad_out[NUM_DEFORMATIONS]) {
    const double epsilon = GRADIENT_EPSILON; 
    Generated_Image generated_img_perturbed; // MUTABLE image buffer
    Feature_Vector generated_features_perturbed; // MUTABLE feature buffer

    for (int k = 0; k < NUM_DEFORMATIONS; k++) {
        // 1. Perturb alpha_k (MUTABLE temporary copy)
        Deformation_Coefficients alpha_perturbed = *alpha;
        alpha_perturbed.alpha[k] += epsilon; // MUTABLE temporary change

        // 2. Forward Pass: Draw curve -> Extract Features
        draw_curve(alpha_perturbed.alpha, generated_img_perturbed);
        extract_geometric_features(generated_img_perturbed, generated_features_perturbed);
        
        // 3. Calculate Loss_perturbed (Feature Loss)
        const double loss_perturbed = calculate_feature_loss(generated_features_perturbed, observed_features);

        // 4. Compute Gradient (Finite Difference)
        grad_out[k] = (loss_perturbed - loss_base) / epsilon;
    }
}

// --- Display Functions ---

void print_image(const char *const title, const double image[GRID_SIZE][GRID_SIZE]) {
    printf("\n%s (16x16):\n", title);
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            if (image[i][j] < 0.3) printf(" ");
            else printf("#");
        }
        printf("\n");
    }
}

// --- Simulation Setup (Generating a Target/Observed Image) ---

/**
 * @brief Creates a synthetic observed image with known deformation and noise.
 */
void generate_observed_target(Generated_Image observed_out, const double true_alpha[NUM_DEFORMATIONS]) {
    // 1. Rasterize the TRUE deformed curve (Signal)
    draw_curve(true_alpha, observed_out);

    // 2. Add random noise (MUTABLE operation on observed_out)
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            // White noise [-0.15, 0.15]
            double noise = ((double)rand() / RAND_MAX - 0.5) * 0.3; 
            observed_out[i][j] = fmax(0.0, fmin(1.0, observed_out[i][j] + noise));
        }
    }
}

// --- Test Runner ---

void run_test(int test_id, const double true_alpha[NUM_DEFORMATIONS]) {
    // Setup (IMMUTABLE data)
    Generated_Image observed_image; // Observed image (mutable buffer)
    generate_observed_target(observed_image, true_alpha);
    
    // Extract the target features once (IMMUTABLE feature vector)
    Feature_Vector observed_features;
    extract_geometric_features(observed_image, observed_features);
    
    printf("\n======================================================\n");
    printf("TEST %d: Target Slant (a_1)=%.4f, Curve (a_2)=%.4f\n", test_id, true_alpha[0], true_alpha[1]);
    printf("======================================================\n");
    print_image("Observed Noisy Target Image (J)", observed_image); 

    // Initialization (MUTABLE data)
    Deformation_Coefficients alpha_hat = {
        .alpha = {0.0, 0.0} // Starting guess: Ideal 'J'
    };
    
    double gradient[NUM_DEFORMATIONS];
    Generated_Image generated_image;
    Feature_Vector generated_features;
    double loss;

    printf("\n--- Optimization Trace ---\n");
    printf("It | Loss     | a_1 (Slant) | a_2 (Curve) | da_1 | da_2\n");
    printf("----------------------------------------------------------\n");

    // Training/Estimation Loop (MUTABLE iterations)
    for (int t = 0; t <= ITERATIONS; t++) {
        // Forward Pass: Draw curve -> Extract Features
        draw_curve(alpha_hat.alpha, generated_image);
        extract_geometric_features(generated_image, generated_features);
        
        // Calculate Loss (IMMUTABLE operation)
        loss = calculate_feature_loss(generated_features, observed_features);
        
        // Calculate Gradient (IMMUTABLE operation)
        calculate_gradient(observed_features, &alpha_hat, loss, gradient);
        
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

    // Final Result (IMMUTABLE visualization)
    draw_curve(alpha_hat.alpha, generated_image);
    print_image("Final Denoised/Estimated Image", generated_image);

    printf("\n--- Conclusion ---\n");
    printf("TRUE Slant (a_1): %.4f, Estimated: %.4f\n", true_alpha[0], alpha_hat.alpha[0]);
    printf("TRUE Curvature (a_2): %.4f, Estimated: %.4f\n", true_alpha[1], alpha_hat.alpha[1]);
}


// --- Main Execution ---

int main(void) {
    // Array of 10 test cases (IMMUTABLE data)
    // {Slant (a1), Curvature (a2)}
    const double test_cases[10][NUM_DEFORMATIONS] = {
        {-0.10, 0.05}, // 1. Original Target (Slanted Left, Curved Out)
        { 0.10, -0.05}, // 2. Slanted Right, Curved In
        { 0.00, 0.15},  // 3. Very Curved Out (Vertical)
        {-0.20, 0.00},  // 4. Very Slanted Left (Straight)
        { 0.05, 0.00},  // 5. Slightly Slanted Right (Straight)
        {-0.05, -0.05}, // 6. Slightly Slanted Left, Curved In
        { 0.15, 0.10},  // 7. Slanted Right, Curved Out (Exaggerated)
        { 0.00, 0.00},  // 8. Ideal J (No Deformation)
        { 0.20, 0.05},  // 9. Highly Slanted Right
        {-0.10, -0.10}  // 10. Slanted Left, Highly Curved In
    };
    
    srand(42); // Seed for reproducible results

    for (int i = 0; i < 10; i++) {
        run_test(i + 1, test_cases[i]);
    }

    return 0;
}