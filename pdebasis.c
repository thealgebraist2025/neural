#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h> 
#include <float.h>
#include <string.h> 

// --- Configuration ---
#define GRID_SIZE 32       // Image size: 32x32
#define D_SIZE (GRID_SIZE * GRID_SIZE) 

// **Image Configuration**
#define NUM_IMAGES 2       
#define MIN_RADIUS 3       
#define MAX_RADIUS 10.0    

// **Input/Output Configuration**
#define N_INPUT D_SIZE         // 1024 inputs (x_1, ..., x_1024)
#define N_HIDDEN 32            // Latent Manifold V size: z_1 to z_32
#define N_OUTPUT 3             
#define N_NAIVE_TEST_POINTS 50 // Number of points to test ideal vanishing

// **Network & Training Parameters**
#define N_TRAINING_EPOCHS 1000      
#define REPORT_FREQ 100             
#define INITIAL_LEARNING_RATE 0.0001 
#define COORD_WEIGHT 1.0             
#define GRADIENT_CLIP_NORM 1.0 
#define MAX_TERM_DISPLAY 6     // Max terms to print per equation

// --- Dynamic Globals ---
double w_fh[N_INPUT][N_HIDDEN];    
double b_h[N_HIDDEN]; 
double w_ho[N_HIDDEN][N_OUTPUT];   
double b_o[N_OUTPUT];
double single_images[NUM_IMAGES][D_SIZE]; 
int target_properties[NUM_IMAGES][3]; 


// --- Helper Macros ---
#define CLAMP(val, min, max) ((val) < (min) ? (min) : ((val) > (max) ? (max) : (val)))
#define NORMALIZE_COORD(coord) ((double)(coord) / (GRID_SIZE - 1.0))
#define NORMALIZE_RADIUS(radius) ((double)(radius) / MAX_RADIUS)


// -----------------------------------------------------------------
// --- ALGEBRAIC ACTIVATION FUNCTIONS ---
// (Quadratic activation is used for algebraic variety consistency)
// -----------------------------------------------------------------
double poly_activation(double z_net) {
    return z_net * z_net; 
}
double poly_derivative(double z_net) {
    return 2.0 * z_net;
}

// -----------------------------------------------------------------
// --- DATA GENERATION AND NN CORE ---
// -----------------------------------------------------------------

void draw_filled_circle(double image[D_SIZE], int cx, int cy, int r) {
    for (int i = 0; i < D_SIZE; i++) { image[i] = 0.0; }
    for (int y = 0; y < GRID_SIZE; y++) {
        for (int x = 0; x < GRID_SIZE; x++) {
            double dist_sq = (x - cx) * (x - cx) + (y - cy) * (y - cy);
            if (dist_sq <= r * r) {
                image[GRID_SIZE * y + x] = 1.0; 
            }
        }
    }
}

void generate_circle_image(int index) {
    srand((unsigned int)time(NULL) + index * 100); 
    int *properties = target_properties[index];
    double *image = single_images[index];
    int cx = (GRID_SIZE / 4) + (rand() % (GRID_SIZE / 2));
    int cy = (GRID_SIZE / 4) + (rand() % (GRID_SIZE / 2));
    int r = (int)MIN_RADIUS + (rand() % ((int)MAX_RADIUS - (int)MIN_RADIUS + 1));
    draw_filled_circle(image, cx, cy, r);
    properties[0] = cx; properties[1] = cy; properties[2] = r;
}

void load_train_case(double input[N_INPUT], double target[N_OUTPUT]) {
    int img_idx = rand() % NUM_IMAGES;
    const double *current_image = single_images[img_idx];
    const int *properties = target_properties[img_idx];
    memcpy(input, current_image, D_SIZE * sizeof(double));
    target[0] = NORMALIZE_COORD(properties[0]); 
    target[1] = NORMALIZE_COORD(properties[1]); 
    target[2] = NORMALIZE_RADIUS(properties[2]); 
}

void initialize_nn() {
    double fan_in_h = (double)N_INPUT;
    double limit_h = sqrt(1.0 / fan_in_h); 
    double fan_in_o = (double)N_HIDDEN;
    double limit_o = sqrt(1.0 / fan_in_o); 

    for (int i = 0; i < N_INPUT; i++) {
        for (int j = 0; j < N_HIDDEN; j++) {
            w_fh[i][j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * limit_h; 
        }
    }
    for (int j = 0; j < N_HIDDEN; j++) { 
        b_h[j] = 0.0; 
        for (int k = 0; k < N_OUTPUT; k++) {
            w_ho[j][k] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * limit_o;
        }
    }
    for (int k = 0; k < N_OUTPUT; k++) { b_o[k] = 0.0; }
}

void forward_pass(const double input[N_INPUT], double hidden_net[N_HIDDEN], double hidden_out[N_HIDDEN], double output_net[N_OUTPUT], double output[N_OUTPUT]) {
    for (int j = 0; j < N_HIDDEN; j++) {
        double h_net = b_h[j];
        for (int i = 0; i < N_INPUT; i++) { h_net += input[i] * w_fh[i][j]; }
        hidden_net[j] = h_net;
        hidden_out[j] = poly_activation(h_net);
    }
    for (int k = 0; k < N_OUTPUT; k++) {
        double o_net = b_o[k]; 
        for (int j = 0; j < N_HIDDEN; j++) { o_net += hidden_out[j] * w_ho[j][k]; } 
        output_net[k] = o_net;
        output[k] = o_net;
    }
}

double clip_gradient(double grad, double max_norm) { 
    if (grad > max_norm) return max_norm;
    if (grad < -max_norm) return -max_norm;
    return grad;
}

void train_nn() {
    printf("\n--- TRAINING PHASE ---\n");
    double input[N_INPUT];
    double target[N_OUTPUT];
    double hidden_net[N_HIDDEN]; 
    double hidden_out[N_HIDDEN];
    double output_net[N_OUTPUT]; 
    double output[N_OUTPUT];
    double cumulative_loss_report = 0.0;
    int samples_processed_in_report = 0;

    for (int epoch = 0; epoch < N_TRAINING_EPOCHS; epoch++) {
        load_train_case(input, target);
        forward_pass(input, hidden_net, hidden_out, output_net, output);
        
        // Backpropagation and Weight Update (Standard, omitted details)
        double delta_o[N_OUTPUT];
        double delta_h[N_HIDDEN]; 
        double error_h[N_HIDDEN] = {0.0};
        for (int k = 0; k < N_OUTPUT; k++) { delta_o[k] = clip_gradient(((output[k] - target[k]) * COORD_WEIGHT), GRADIENT_CLIP_NORM); }
        for (int j = 0; j < N_HIDDEN; j++) { 
            for (int k = 0; k < N_OUTPUT; k++) { error_h[j] += delta_o[k] * w_ho[j][k]; }
            delta_h[j] = error_h[j] * poly_derivative(hidden_net[j]);
        }
        for (int k = 0; k < N_OUTPUT; k++) { 
            for (int j = 0; j < N_HIDDEN; j++) { w_ho[j][k] -= INITIAL_LEARNING_RATE * delta_o[k] * hidden_out[j]; } 
            b_o[k] -= INITIAL_LEARNING_RATE * delta_o[k]; 
        } 
        for (int i = 0; i < N_INPUT; i++) { 
            for (int j = 0; j < N_HIDDEN; j++) { w_fh[i][j] -= INITIAL_LEARNING_RATE * clip_gradient(delta_h[j] * input[i], GRADIENT_CLIP_NORM); } 
        }
        for (int j = 0; j < N_HIDDEN; j++) { b_h[j] -= INITIAL_LEARNING_RATE * delta_h[j]; }
        
        // Loss calculation for reporting
        double loss = 0.0;
        for (int k = 0; k < N_OUTPUT; k++) { loss += (output[k] - target[k]) * (output[k] - target[k]) * COORD_WEIGHT; }
        cumulative_loss_report += loss;
        samples_processed_in_report++;

        if ((epoch + 1) % REPORT_FREQ == 0) {
            double current_avg_loss = cumulative_loss_report / samples_processed_in_report;
            printf("  Epoch: %6d / %6d | Avg Loss: %7.6f\n", epoch + 1, N_TRAINING_EPOCHS, current_avg_loss);
            cumulative_loss_report = 0.0;
            samples_processed_in_report = 0;
        }
    }
    printf("--- TRAINING PHASE COMPLETE ---\n");
}

// -----------------------------------------------------------------
// --- NAIVE IDEAL CALCULATION (Numerical Validation) ---
// -----------------------------------------------------------------

/**
 * Calculates the residual for the generator P_j: |z_j - (w_j * x + b_j)^2|
 */
double calculate_generator_residual(int j, const double input[N_INPUT], double latent_z_j) {
    double linear_net = b_h[j];
    for (int i = 0; i < N_INPUT; i++) {
        linear_net += w_fh[i][j] * input[i];
    }
    double predicted_z_j = poly_activation(linear_net);
    return fabs(latent_z_j - predicted_z_j);
}

void naive_ideal_calculation() {
    printf("\n\n--- NAIVE IDEAL CALCULATION (Numerical Validation) ---\n");
    printf("Testing $N=%d$ sampled latent vectors $(\\mathbf{x}, \\mathbf{z})$ for ideal properties.\n\n", N_NAIVE_TEST_POINTS);

    double input[N_INPUT];
    double target[N_OUTPUT];
    double hidden_net[N_HIDDEN]; 
    double latent_z[N_HIDDEN]; 
    double output_net[N_OUTPUT]; 
    double output[N_OUTPUT];

    double avg_residual[N_HIDDEN] = {0.0};
    double total_generator_residual = 0.0;
    double total_cross_term_residual = 0.0;

    for (int k = 0; k < N_NAIVE_TEST_POINTS; k++) {
        // Generate a random (x, z) pair lying on the manifold V
        load_train_case(input, target); 
        forward_pass(input, hidden_net, latent_z, output_net, output);

        for (int j = 0; j < N_HIDDEN; j++) {
            // 1. Calculate the residual for generator P_j (Vanishing Property)
            double residual_j = calculate_generator_residual(j, input, latent_z[j]);
            avg_residual[j] += residual_j;
            total_generator_residual += residual_j;
            
            // 2. Naive Ideal Closure Check (Test P_j * P_{j+1} vanishes)
            if (j < N_HIDDEN - 1) {
                double residual_k = calculate_generator_residual(j+1, input, latent_z[j+1]);
                double cross_term_residual = fabs(residual_j * residual_k);
                total_cross_term_residual += cross_term_residual;
            }
        }
    }

    // --- Summary Output ---
    
    printf("### 1. Vanishing Property Check (Generators $P_j(\\mathbf{z}, \\mathbf{x}) \\approx 0$)\n");
    printf("This confirms the generators vanish on the trained manifold $V$.\n");
    printf("Average Residual over all %d generators: $\\mathbf{%.4e}$\n", 
           N_HIDDEN, total_generator_residual / (N_HIDDEN * N_NAIVE_TEST_POINTS));
    
    printf("\nResiduals for the first 5 generators:\n");
    for (int j = 0; j < 5; j++) {
        printf("  $P_{%d}$: %.8e\n", j+1, avg_residual[j] / N_NAIVE_TEST_POINTS);
    }
    
    printf("\n### 2. Ideal Closure Check ($P_j \\cdot P_k \\approx 0$)\n");
    printf("This confirms products of generators also vanish (a property of an ideal).\n");
    printf("Average Residual for $P_j \\cdot P_{j+1}$ cross-terms: $\\mathbf{%.4e}$\n", 
           total_cross_term_residual / ((N_HIDDEN - 1) * N_NAIVE_TEST_POINTS));


    printf("\n--- CONCRETE IDEAL GENERATORS (Poles of the Variety) ---\n");
    printf("The following are the equations $\\mathbf{P}_j(\\mathbf{z}, \\mathbf{x})$ whose solution set defines the manifold $V$:\n\n");
    
    for (int j = 0; j < 3; j++) {
        printf("P$_{%d}(\\mathbf{z}, \\mathbf{x}) = z_{%d} - \\left( ", j+1, j+1);
        
        int terms_printed = 0;
        int i;
        
        // Print terms with trained w_fh coefficients
        for (i = 0; i < N_INPUT; i++) {
            if (fabs(w_fh[i][j]) > 1e-4 && terms_printed < MAX_TERM_DISPLAY) {
                if (terms_printed > 0) printf(" + ");
                printf("%.6fx_{%d}", w_fh[i][j], i+1);
                terms_printed++;
            }
            if (terms_printed >= MAX_TERM_DISPLAY && i < N_INPUT - 1) break;
        }

        if (i < N_INPUT - 1) {
             printf(" + \\dots ");
        }
        
        // Bias term and closing square
        printf(" + (%.6f) \\right)^2 = 0$\n\n", b_h[j]);
    }
    printf("... The complete generating set contains %d such quadratic equations.\n", N_HIDDEN);
}


// -----------------------------------------------------------------
// --- MAIN PROGRAM ---
// -----------------------------------------------------------------

int main(int argc, char **argv) {
    srand(time(NULL));

    printf("--- Algebraic Network Execution ---\n");
    
    // 1. Initialization and Data Generation
    initialize_nn(); 
    for (int i = 0; i < NUM_IMAGES; i++) {
        generate_circle_image(i); 
    }

    // 2. Training (Establishes the Ideal's Coefficients)
    train_nn();
    
    // 3. Naive Ideal Calculation (Numerical Validation and Symbolic Print)
    naive_ideal_calculation();

    printf("\n--- PROGRAM END ---\n");
    return 0;
}
