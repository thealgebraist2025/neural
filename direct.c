#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Define M_PI explicitly
#define M_PI 3.14159265358979323846

// --- Global Configuration ---
#define GRID_SIZE 16
#define NUM_DEFORMATIONS 2  // alpha_1 (Slant), alpha_2 (Curvature)
#define NUM_FEATURES 32     // 32 directional projection features
#define NUM_POINTS 200
#define ITERATIONS 5000     // 5000 iterations for stable convergence
#define GRADIENT_EPSILON 0.01 
#define NUM_IDEAL_CHARS 5   // J, 1, 2, 3, 4
#define NUM_TESTS 5         // One test case for each character

// NEW: 8 segments require 9 control points (P0 to P8)
#define NUM_CONTROL_POINTS 9 

// --- Data Structures ---

typedef struct {
    double x; 
    double y; 
} Point;

typedef struct {
    // Array of 9 control points defining the 8 segments of the curve
    const Point control_points[NUM_CONTROL_POINTS];
} Ideal_Curve_Params; 

typedef struct {
    double alpha[NUM_DEFORMATIONS]; 
} Deformation_Coefficients;

typedef double Generated_Image[GRID_SIZE][GRID_SIZE]; 
typedef double Feature_Vector[NUM_FEATURES]; 

// Structure to hold one potential estimation result (Template vs. Observed)
typedef struct {
    double estimated_alpha[NUM_DEFORMATIONS];
    double final_loss;
} EstimationResult;

// Structure to hold results for one full test case (classification results)
typedef struct {
    int id;
    int true_char_index; // Index of the true character (0='J', 1='1', ...)
    int best_match_index; // Index of the character with the minimum loss
    double true_alpha[NUM_DEFORMATIONS];
    double true_image[GRID_SIZE][GRID_SIZE];
    double observed_image[GRID_SIZE][GRID_SIZE];
    
    // Stores the results of running the observation against all 5 ideal templates
    EstimationResult classification_results[NUM_IDEAL_CHARS]; 
    
    // The final estimated image and diff image based on the BEST MATCH
    Generated_Image best_estimated_image;
    Generated_Image best_diff_image;
} TestResult;

// Global storage for all test results
TestResult all_results[NUM_TESTS];

// --- Fixed Ideal Curves ('J', '1', '2', '3', '4') ---

// Lookup table for character names
const char *CHAR_NAMES[NUM_IDEAL_CHARS] = {"J", "1", "2", "3", "4"};

// Updated Ideal Templates using 9 control points (8 segments)
const Ideal_Curve_Params IDEAL_TEMPLATES[NUM_IDEAL_CHARS] = {
    // 0: 'J' (Curved stroke down, hook left at bottom) - CORRECTED
    [0] = {.control_points = {
        {.x = 0.6, .y = 0.1}, {.x = 0.6, .y = 0.2}, {.x = 0.6, .y = 0.3}, {.x = 0.55, .y = 0.4}, 
        {.x = 0.5, .y = 0.5}, {.x = 0.45, .y = 0.65}, // Vertical stem ends here
        {.x = 0.4, .y = 0.8}, // Start of tight curve
        {.x = 0.3, .y = 0.9}, // Bottom center of hook
        {.x = 0.2, .y = 0.85} // Curve up to finish the hook
    }},
    
    // 1: '1' (Mostly vertical line)
    [1] = {.control_points = {
        {.x = 0.4, .y = 0.1}, {.x = 0.5, .y = 0.1}, {.x = 0.5, .y = 0.2}, {.x = 0.5, .y = 0.35}, 
        {.x = 0.5, .y = 0.5}, {.x = 0.5, .y = 0.65}, {.x = 0.5, .y = 0.8}, {.x = 0.5, .y = 0.9}, 
        {.x = 0.5, .y = 0.9} 
    }},
    
    // 2: '2' (S-curve: Top arc, down-left diagonal, horizontal base)
    [2] = {.control_points = {
        {.x = 0.2, .y = 0.1}, {.x = 0.7, .y = 0.1}, {.x = 0.8, .y = 0.25}, {.x = 0.7, .y = 0.4}, 
        {.x = 0.3, .y = 0.55}, {.x = 0.2, .y = 0.7}, {.x = 0.3, .y = 0.8}, {.x = 0.5, .y = 0.9}, 
        {.x = 0.8, .y = 0.9} 
    }}, 
    
    // 3: '3' (Two right-facing curves)
    [3] = {.control_points = {
        {.x = 0.2, .y = 0.1}, {.x = 0.8, .y = 0.1}, {.x = 0.8, .y = 0.3}, {.x = 0.4, .y = 0.5}, 
        {.x = 0.8, .y = 0.5}, {.x = 0.8, .y = 0.7}, {.x = 0.4, .y = 0.9}, {.x = 0.8, .y = 0.9}, 
        {.x = 0.2, .y = 0.9} 
    }}, 
    
    // 4: '4' (Down-left, then across, then vertical stem)
    [4] = {.control_points = {
        {.x = 0.2, .y = 0.1}, {.x = 0.3, .y = 0.2}, {.x = 0.4, .y = 0.3}, {.x = 0.5, .y = 0.4}, 
        {.x = 0.2, .y = 0.55}, {.x = 0.8, .y = 0.55}, {.x = 0.8, .y = 0.7}, {.x = 0.8, .y = 0.85}, 
        {.x = 0.8, .y = 0.9} 
    }} 
};

/**
 * @brief Applies Slant and Curvature deformation to a point.
 */
void apply_deformation(Point *point, const double alpha[NUM_DEFORMATIONS]) {
    // Deformation 1: Slant (Shear Transform)
    point->x = point->x + alpha[0] * (point->y - 0.5);

    // Deformation 2: Curvature/Width
    point->x = point->x + alpha[1] * sin(M_PI * point->y);
}

/**
 * @brief Generates a point on the curve using the ideal form and deformations (8-segment interpolation).
 */
Point get_deformed_point(const double t, const Ideal_Curve_Params *const params, const double alpha[NUM_DEFORMATIONS]) {
    Point p = {.x = 0.0, .y = 0.0};
    
    const int N = NUM_CONTROL_POINTS; // 9
    const double segment_length_t = 1.0 / (N - 1); // 1/8 = 0.125

    // Find the segment index (0 to 7)
    int segment_index = (int)floor(t / segment_length_t);
    if (segment_index >= N - 1) {
        segment_index = N - 2; // Clamp to the last segment index (7)
    }

    // Get the starting and ending control points for the segment
    const Point P_start = params->control_points[segment_index];
    const Point P_end = params->control_points[segment_index + 1];

    // Normalize t within the current segment [0, 1]
    const double segment_t = (t - segment_index * segment_length_t) / segment_length_t;

    // Linear interpolation
    p.x = P_start.x + (P_end.x - P_start.x) * segment_t;
    p.y = P_start.y + (P_end.y - P_start.y) * segment_t;

    apply_deformation(&p, alpha);

    // Scale to pixel grid and clamp
    p.x = fmax(0.0, fmin(GRID_SIZE - 1.0, p.x * GRID_SIZE));
    p.y = fmax(0.0, fmin(GRID_SIZE - 1.0, p.y * GRID_SIZE));

    return p;
}

/**
 * @brief Rasterizes the deformed curve onto the image grid (Forward Model G).
 */
void draw_curve(const double alpha[NUM_DEFORMATIONS], Generated_Image img, const Ideal_Curve_Params *const ideal_params) {
    // Clear the canvas
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            img[i][j] = 0.0;
        }
    }

    // Sample and draw points
    for (int i = 0; i <= NUM_POINTS; i++) {
        const double t = (double)i / NUM_POINTS;
        const Point current_p = get_deformed_point(t, ideal_params, alpha);

        // Main pixel
        const int px = (int)round(current_p.x);
        const int py = (int)round(current_p.y);

        if (px >= 0 && px < GRID_SIZE && py >= 0 && py < GRID_SIZE) {
            img[py][px] = fmin(1.0, img[py][px] + 0.5); 
        }

        // Neighbor smoothing (basic anti-aliasing)
        if (py + 1 < GRID_SIZE) img[py + 1][px] = fmin(1.0, img[py + 1][px] + 0.1);
        if (py - 1 >= 0) img[py - 1][px] = fmin(1.0, img[py - 1][px] + 0.1);
        if (px + 1 < GRID_SIZE) img[py][px + 1] = fmin(1.0, img[py][px + 1] + 0.1);
        if (px - 1 >= 0) img[py][px - 1] = fmin(1.0, img[py][px - 1] + 0.1);
    }
}

// --- Feature Extraction and Loss ---

/**
 * @brief Extracts 32 geometric projection features from the image (Directional Moments).
 */
void extract_geometric_features(const Generated_Image img, Feature_Vector features_out) {
    // Generate 32 normalized unit vectors (length 1)
    double vectors[NUM_FEATURES][2];
    for (int k = 0; k < NUM_FEATURES; k++) {
        const double angle = 2.0 * M_PI * k / NUM_FEATURES;
        vectors[k][0] = cos(angle); // x component
        vectors[k][1] = sin(angle); // y component
    }
    
    // Center point for coordinate calculation
    const double center = (GRID_SIZE - 1.0) / 2.0;

    // Initialize feature vector
    for (int k = 0; k < NUM_FEATURES; k++) {
        features_out[k] = 0.0;
    }

    // Iterate over all pixels
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            const double intensity = img[i][j];
            if (intensity < 0.1) continue; 

            // Vector from center to pixel (j is x-axis, i is y-axis)
            const double vx = (double)j - center;
            const double vy = (double)i - center;
            
            // Project the mass vector onto all 32 basis vectors
            for (int k = 0; k < NUM_FEATURES; k++) {
                // Dot product: projection = (vx * basis_x + vy * basis_y) * intensity
                const double projection = (vx * vectors[k][0] + vy * vectors[k][1]) * intensity;
                features_out[k] += projection; 
            }
        }
    }
}

/**
 * @brief Calculates the L2 Loss (Squared Error) between 32-dimensional feature vectors.
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
void calculate_gradient(const Feature_Vector observed_features, const Deformation_Coefficients *const alpha, const double loss_base, double grad_out[NUM_DEFORMATIONS], const Ideal_Curve_Params *const ideal_params) {
    const double epsilon = GRADIENT_EPSILON; 
    Generated_Image generated_img_perturbed; 
    Feature_Vector generated_features_perturbed; 

    for (int k = 0; k < NUM_DEFORMATIONS; k++) {
        // Perturb alpha_k
        Deformation_Coefficients alpha_perturbed = *alpha;
        alpha_perturbed.alpha[k] += epsilon; 

        // Forward Pass: Draw curve -> Extract Features
        draw_curve(alpha_perturbed.alpha, generated_img_perturbed, ideal_params);
        extract_geometric_features(generated_img_perturbed, generated_features_perturbed);
        
        // Calculate Loss_perturbed (Feature Loss)
        const double loss_perturbed = calculate_feature_loss(generated_features_perturbed, observed_features);

        // Compute Gradient (Finite Difference)
        grad_out[k] = (loss_perturbed - loss_base) / epsilon;
    }
}

/**
 * @brief Creates a synthetic image based on true parameters and noise.
 */
void generate_target_image(Generated_Image image_out, const double true_alpha[NUM_DEFORMATIONS], const Ideal_Curve_Params *const ideal_params, int add_noise) {
    // 1. Rasterize the TRUE deformed curve (Signal)
    draw_curve(true_alpha, image_out, ideal_params);

    // 2. Add random noise if requested
    if (add_noise) {
        for (int i = 0; i < GRID_SIZE; i++) {
            for (int j = 0; j < GRID_SIZE; j++) {
                // White noise [-0.15, 0.15]
                double noise = ((double)rand() / RAND_MAX - 0.5) * 0.3; 
                image_out[i][j] = fmax(0.0, fmin(1.0, image_out[i][j] + noise));
            }
        }
    }
}

/**
 * @brief Runs a single optimization of a test image against one ideal template.
 */
void run_optimization(const Feature_Vector observed_features, int ideal_char_index, EstimationResult *result, int print_trace) {
    const Ideal_Curve_Params *ideal_params = &IDEAL_TEMPLATES[ideal_char_index];
    
    // Initialization (MUTABLE data)
    Deformation_Coefficients alpha_hat = {
        .alpha = {0.0, 0.0} // Starting guess: Ideal (zero deformation)
    };
    
    // Dynamic learning rate initialization and floor
    double learning_rate = 0.0000001; 
    const double min_learning_rate = 0.0000000001;
    double gradient[NUM_DEFORMATIONS];
    Generated_Image generated_image;
    Feature_Vector generated_features;
    double loss;
    double prev_loss = HUGE_VAL; 

    if (print_trace) {
        printf("\n    --- Optimizing against '%s' ---\n", CHAR_NAMES[ideal_char_index]);
        printf("    It | Loss     | L Rate  | a_1 (Slant) | a_2 (Curve)\n");
        printf("    ------------------------------------------------------\n");
    }

    // Training/Estimation Loop 
    for (int t = 0; t <= ITERATIONS; t++) {
        // Forward Pass: Draw curve -> Extract Features
        draw_curve(alpha_hat.alpha, generated_image, ideal_params);
        extract_geometric_features(generated_image, generated_features);
        
        // Calculate Loss 
        loss = calculate_feature_loss(generated_features, observed_features);
        
        // Check for bouncing/overshooting and decay learning rate
        if (loss > prev_loss * 1.001 && learning_rate > min_learning_rate) { 
            learning_rate *= 0.5; // Halve the learning rate
        }

        prev_loss = loss;

        // Calculate Gradient 
        calculate_gradient(observed_features, &alpha_hat, loss, gradient, ideal_params);
        
        // Print progress only every 200 iterations, and at start/end
        if (print_trace && (t % 200 == 0 || t == ITERATIONS)) {
            printf("    %04d | %8.5f | %7.8f | %8.4f | %8.4f\n", t, loss, learning_rate, alpha_hat.alpha[0], alpha_hat.alpha[1]);
        }

        // Gradient Descent Update
        if (t < ITERATIONS) {
            double step_rate = (learning_rate > min_learning_rate) ? learning_rate : min_learning_rate;
            
            const double delta_a1 = step_rate * gradient[0];
            const double delta_a2 = step_rate * gradient[1];
            
            alpha_hat.alpha[0] -= delta_a1;
            alpha_hat.alpha[1] -= delta_a2;
        }
    }

    // Store Final Results
    result->final_loss = loss;
    memcpy(result->estimated_alpha, alpha_hat.alpha, sizeof(double) * NUM_DEFORMATIONS);
}

/**
 * @brief Calculates the absolute pixel-wise difference between two images.
 */
void calculate_difference_image(const Generated_Image obs, const Generated_Image est, Generated_Image diff) {
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            // Absolute difference of intensity
            diff[i][j] = fabs(obs[i][j] - est[i][j]);
        }
    }
}

/**
 * @brief Runs one full classification test against all ideal characters.
 */
void run_classification_test(int test_id, int true_char_index, const double true_alpha[NUM_DEFORMATIONS], TestResult *result) {
    result->id = test_id;
    result->true_char_index = true_char_index;
    memcpy(result->true_alpha, true_alpha, sizeof(double) * NUM_DEFORMATIONS);
    const Ideal_Curve_Params *true_params = &IDEAL_TEMPLATES[true_char_index];
    
    // 1. Generate the CLEAN True Image and store it
    generate_target_image(result->true_image, true_alpha, true_params, 0); 
    
    // 2. Generate the NOISY Observed Image and store it
    Generated_Image observed_image; 
    generate_target_image(observed_image, true_alpha, true_params, 1);
    memcpy(result->observed_image, observed_image, sizeof(Generated_Image));
    
    // 3. Extract the TARGET features once
    Feature_Vector observed_features;
    extract_geometric_features(observed_image, observed_features);
    
    printf("\n======================================================\n");
    printf("TEST %02d/%02d (TRUE: '%s'): Slant (a_1)=%.4f, Curve (a_2)=%.4f\n", 
           test_id, NUM_TESTS, CHAR_NAMES[true_char_index], true_alpha[0], true_alpha[1]);

    // 4. Run Optimization against ALL ideal templates
    double min_loss = HUGE_VAL;
    int best_match_index = -1;
    
    for (int i = 0; i < NUM_IDEAL_CHARS; i++) {
        // Print trace only for the true character match for brevity
        int print_trace = (i == true_char_index);
        
        run_optimization(observed_features, i, &result->classification_results[i], print_trace);

        if (result->classification_results[i].final_loss < min_loss) {
            min_loss = result->classification_results[i].final_loss;
            best_match_index = i;
        }
    }

    result->best_match_index = best_match_index;

    // 5. Finalize the Best Match Images (for summary and SVG)
    EstimationResult *best_fit = &result->classification_results[best_match_index];
    
    draw_curve(best_fit->estimated_alpha, result->best_estimated_image, &IDEAL_TEMPLATES[best_match_index]);
    calculate_difference_image(observed_image, result->best_estimated_image, result->best_diff_image);
}

// --- Console Summary Function ---
void summarize_results_console() {
    printf("\n\n==================================================================\n");
    printf("                  CLASSIFICATION SUMMARY (5 Tests)                  \n");
    printf("==================================================================\n");
    printf("  ID | TRUE | PRED | Classification Losses (L2 Feature Error) \n");
    printf("     | Char | Char |   'J'    |   '1'    |   '2'    |   '3'    |   '4'    \n");
    printf("-----|------|------|----------|----------|----------|----------|----------\n");
    
    for (int k = 0; k < NUM_TESTS; k++) {
        TestResult *r = &all_results[k];
        printf("%4d | %4s | %4s |", 
               r->id, CHAR_NAMES[r->true_char_index], CHAR_NAMES[r->best_match_index]);
        
        for (int i = 0; i < NUM_IDEAL_CHARS; i++) {
            printf(" %8.4f |", r->classification_results[i].final_loss);
        }
        printf("\n");
    }
    printf("------------------------------------------------------------------\n");


    printf("\n\n======================================================\n");
    printf("          TEST VISUAL SUMMARY (TRUE vs. BEST FIT)     \n");
    printf("======================================================\n");
    
    for (int k = 0; k < NUM_TESTS; k++) {
        TestResult *r = &all_results[k];
        EstimationResult *best_fit = &r->classification_results[r->best_match_index];
        
        printf("\n--- TEST %02d (TRUE: '%s' | PRED: '%s') ---\n", 
               r->id, CHAR_NAMES[r->true_char_index], CHAR_NAMES[r->best_match_index]);
        printf("    TRUE Params (%.4f, %.4f) | BEST FIT Params (%.4f, %.4f) | Loss: %.4f\n", 
               r->true_alpha[0], r->true_alpha[1], 
               best_fit->estimated_alpha[0], best_fit->estimated_alpha[1], 
               best_fit->final_loss);
        
        printf("| Observed Noisy Target | Best Estimated Fit | Difference (Error) |\n");
        printf("|-----------------------|--------------------|--------------------|\n");

        for (int i = 0; i < GRID_SIZE; i++) {
            printf("| ");
            // Observed Image (Noisy Target)
            for (int j = 0; j < GRID_SIZE; j++) {
                if (r->observed_image[i][j] < 0.3) printf(" ");
                else if (r->observed_image[i][j] < 0.6) printf(":");
                else printf("#");
            }
            printf(" | ");
            // Best Estimated Image
            for (int j = 0; j < GRID_SIZE; j++) {
                if (r->best_estimated_image[i][j] < 0.3) printf(" ");
                else if (r->best_estimated_image[i][j] < 0.6) printf(":");
                else printf("#");
            }
            printf(" | ");
            // Error visualization (threshold > 0.2 to filter out background noise)
            for (int j = 0; j < GRID_SIZE; j++) {
                if (r->best_diff_image[i][j] > 0.3) printf("*"); 
                else if (r->best_diff_image[i][j] > 0.2) printf("+"); 
                else printf(" "); 
            }
            printf(" |\n");
        }
    }
    printf("======================================================\n");
}


// --- SVG Generation Functions ---

// Constants for SVG rendering
#define PIXEL_SIZE 5    
#define IMG_SIZE (GRID_SIZE * PIXEL_SIZE) // 80
#define IMG_SPACING 5   
#define SET_SPACING 25  
#define SET_WIDTH (4 * IMG_SIZE + 3 * IMG_SPACING) // 335
// Dimensions calculated for 5 sets in a single row
#define SVG_RENDER_LIMIT 5
#define SVG_WIDTH (SVG_RENDER_LIMIT * SET_WIDTH + (SVG_RENDER_LIMIT - 1) * SET_SPACING + 2 * SET_SPACING) // 1805
#define SVG_HEIGHT (IMG_SIZE + 15 + 2 * SET_SPACING) // 130

/**
 * @brief Maps an intensity [0.0, 1.0] to an RGB grayscale color string.
 */
void get_grayscale_color(double intensity, char *color_out) {
    double clamped_intensity = fmax(0.0, fmin(1.0, intensity));
    // Line is white/yellow, background is black.
    if (clamped_intensity > 0.6) {
        sprintf(color_out, "rgb(255, 255, 100)"); // Yellowish white for high intensity
    } else if (clamped_intensity > 0.3) {
        sprintf(color_out, "rgb(100, 100, 100)"); // Gray for medium intensity
    } else {
        sprintf(color_out, "rgb(0, 0, 0)"); // Black/Invisible for low intensity
    }
}

/**
 * @brief Renders a single image into the SVG file at a specific offset.
 */
void render_single_image_to_svg(FILE *fp, const Generated_Image img, double x_offset, double y_offset) {
    char color[20];
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            // Check intensity to decide on background vs line drawing
            if (img[i][j] > 0.0) {
                // Use bright color for the line itself
                get_grayscale_color(img[i][j], color);
                fprintf(fp, "<rect x=\"%.1f\" y=\"%.1f\" width=\"%d\" height=\"%d\" fill=\"%s\"/>\n",
                        x_offset + j * PIXEL_SIZE, 
                        y_offset + i * PIXEL_SIZE, 
                        PIXEL_SIZE, PIXEL_SIZE, color);
            } else {
                // Draw black for the empty background grid cells
                 fprintf(fp, "<rect x=\"%.1f\" y=\"%.1f\" width=\"%d\" height=\"%d\" fill=\"black\"/>\n",
                        x_offset + j * PIXEL_SIZE, 
                        y_offset + i * PIXEL_SIZE, 
                        PIXEL_SIZE, PIXEL_SIZE);
            }
        }
    }
}

/**
 * @brief Renders the 4-image comparison set for one test case into the SVG file.
 */
void render_test_to_svg(FILE *fp, const TestResult *r, double x_set, double y_set) {
    // 1. True Image (Clean)
    render_single_image_to_svg(fp, r->true_image, x_set, y_set);
    
    // 2. Target Image (Noisy)
    render_single_image_to_svg(fp, r->observed_image, x_set + IMG_SIZE + IMG_SPACING, y_set);

    // 3. Best Estimated Image (Clean Fit from Best Match)
    render_single_image_to_svg(fp, r->best_estimated_image, x_set + 2 * (IMG_SIZE + IMG_SPACING), y_set);

    // 4. Difference Image (Error Magnitude) - using a different coloring for error map
    char error_color[20];
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            double error = r->best_diff_image[i][j];
            if (error > 0.3) {
                sprintf(error_color, "rgb(255, 50, 50)"); // High Error: Red
            } else if (error > 0.1) {
                sprintf(error_color, "rgb(255, 150, 0)"); // Medium Error: Orange
            } else {
                sprintf(error_color, "black");
            }
            fprintf(fp, "<rect x=\"%.1f\" y=\"%.1f\" width=\"%d\" height=\"%d\" fill=\"%s\"/>\n",
                    x_set + 3 * (IMG_SIZE + IMG_SPACING) + j * PIXEL_SIZE, 
                    y_set + i * PIXEL_SIZE, 
                    PIXEL_SIZE, PIXEL_SIZE, error_color);
        }
    }
    
    // Add text label for the test ID and classification result
    char label[150];
    EstimationResult *best_fit = &r->classification_results[r->best_match_index];
    sprintf(label, "T:'%s' (%.2f,%.2f) | P:'%s' L:%.2f", 
            CHAR_NAMES[r->true_char_index], r->true_alpha[0], r->true_alpha[1], 
            CHAR_NAMES[r->best_match_index], best_fit->final_loss);
    
    fprintf(fp, "<text x=\"%.1f\" y=\"%.1f\" font-family=\"sans-serif\" font-size=\"10\" fill=\"white\">%s</text>\n",
            x_set, y_set + IMG_SIZE + 10, label);
}

/**
 * @brief Generates the final SVG file with the 5 test results.
 */
void generate_svg_file() {
    FILE *fp = fopen("network.svg", "w");
    if (fp == NULL) {
        printf("\nERROR: Could not open network.svg for writing.\n");
        return;
    }

    // SVG Header
    fprintf(fp, "<svg width=\"%d\" height=\"%d\" viewBox=\"0 0 %d %d\" xmlns=\"http://www.w3.org/2000/svg\">\n",
            SVG_WIDTH, SVG_HEIGHT, SVG_WIDTH, SVG_HEIGHT);
    fprintf(fp, "<rect width=\"100%%\" height=\"100%%\" fill=\"black\"/>\n");

    // Add general titles for the 4 image columns
    double initial_x = SET_SPACING + IMG_SIZE / 2.0;
    double initial_y = SET_SPACING / 2.0;
    fprintf(fp, "<text x=\"%.1f\" y=\"%.1f\" font-family=\"sans-serif\" font-size=\"12\" fill=\"white\" text-anchor=\"middle\">TRUE CLEAN</text>\n", initial_x, initial_y);
    fprintf(fp, "<text x=\"%.1f\" y=\"%.1f\" font-family=\"sans-serif\" font-size=\"12\" fill=\"white\" text-anchor=\"middle\">OBSERVED NOISY</text>\n", initial_x + IMG_SIZE + IMG_SPACING, initial_y);
    fprintf(fp, "<text x=\"%.1f\" y=\"%.1f\" font-family=\"sans-serif\" font-size=\"12\" fill=\"white\" text-anchor=\"middle\">BEST ESTIMATED</text>\n", initial_x + 2 * (IMG_SIZE + IMG_SPACING), initial_y);
    fprintf(fp, "<text x=\"%.1f\" y=\"%.1f\" font-family=\"sans-serif\" font-size=\"12\" fill=\"white\" text-anchor=\"middle\">ERROR DIFF</text>\n", initial_x + 3 * (IMG_SIZE + IMG_SPACING), initial_y);


    // Render the 5 test sets in a single row
    for (int k = 0; k < NUM_TESTS; k++) {
        int col = k; 
        
        // Calculate top-left corner position for the current 4-image set
        double x_set = SET_SPACING + col * (SET_WIDTH + SET_SPACING);
        // Offset y to leave room for the title text at the top
        double y_set = SET_SPACING + 15; 
        
        render_test_to_svg(fp, &all_results[k], x_set, y_set);
    }

    // SVG Footer
    fprintf(fp, "</svg>\n");
    
    fclose(fp);
    printf("\n\n======================================================\n");
    printf("SVG Output Complete: network.svg created (First %d tests).\n", SVG_RENDER_LIMIT);
    printf("======================================================\n");
}


// --- Main Execution ---

int main(void) {
    srand(42); // Seed for reproducible results

    // Test data: We use one test case for each of the 5 ideal characters, 
    // each with a random deformation.
    const double MIN_ALPHA = -0.15;
    const double MAX_ALPHA = 0.15;

    for (int i = 0; i < NUM_TESTS; i++) {
        double true_alpha[NUM_DEFORMATIONS];
        // Generate random deformation in [-0.15, 0.15]
        true_alpha[0] = MIN_ALPHA + ((double)rand() / RAND_MAX) * (MAX_ALPHA - MIN_ALPHA);
        true_alpha[1] = MIN_ALPHA + ((double)rand() / RAND_MAX) * (MAX_ALPHA - MIN_ALPHA);
        
        run_classification_test(i + 1, i, true_alpha, &all_results[i]);
    }
    
    // 1. Print the console-based summary
    summarize_results_console();

    // 2. Generate the SVG file (only first 5 tests)
    generate_svg_file();

    return 0;
}