#define _XOPEN_SOURCE 

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h> 
#include <float.h>
#include <string.h> 

// --- Configuration ---
#define GRID_SIZE 32       
#define D_SIZE (GRID_SIZE * GRID_SIZE) // 1024

// **Network Configuration**
#define N_INPUT D_SIZE         
#define N_OUTPUT 10            

// **Hidden Layer Sizes**
#define N_HIDDEN1 16 
#define N_HIDDEN2 16 
#define N_HIDDEN3 16 
#define N_HIDDEN4 16 

// **Training Parameters**
#define NUM_IMAGES 1000        
#define TRAINING_TIME_LIMIT 120.0 // Resetting to 60s
#define BATCH_SIZE 32          
#define REPORT_FREQ 500             
#define INITIAL_LEARNING_RATE 0.0001 
#define CLASSIFICATION_WEIGHT 1.0  
#define REGRESSION_WEIGHT 1.0      
#define MIN_RADIUS 3           
#define MAX_RADIUS 10.0    
#define MAX_RECT_SIZE (GRID_SIZE - 2) 

// **Adam Optimizer Parameters**
#define BETA1 0.9
#define BETA2 0.999
#define EPSILON 1e-8

// --- Global Data & Matrices (4 Layers) ---

// Weights and Biases
double w_f1[N_INPUT][N_HIDDEN1];    
double b_1[N_HIDDEN1]; 
double w_12[N_HIDDEN1][N_HIDDEN2];  
double b_2[N_HIDDEN2];
double w_23[N_HIDDEN2][N_HIDDEN3];  
double b_3[N_HIDDEN3];
double w_34[N_HIDDEN3][N_HIDDEN4];  
double b_4[N_HIDDEN4];
double w_4o[N_HIDDEN4][N_OUTPUT];   
double b_o[N_OUTPUT];

// Adam State Variables
double m_w_f1[N_INPUT][N_HIDDEN1], v_w_f1[N_INPUT][N_HIDDEN1];
double m_b_1[N_HIDDEN1], v_b_1[N_HIDDEN1];
double m_w_12[N_HIDDEN1][N_HIDDEN2], v_w_12[N_HIDDEN1][N_HIDDEN2];
double m_b_2[N_HIDDEN2], v_b_2[N_HIDDEN2];
double m_w_23[N_HIDDEN2][N_HIDDEN3], v_w_23[N_HIDDEN2][N_HIDDEN3];
double m_b_3[N_HIDDEN3], v_b_3[N_HIDDEN3];
double m_w_34[N_HIDDEN3][N_HIDDEN4], v_w_34[N_HIDDEN3][N_HIDDEN4];
double m_b_4[N_HIDDEN4], v_b_4[N_HIDDEN4];
double m_w_4o[N_HIDDEN4][N_OUTPUT], v_w_4o[N_HIDDEN4][N_OUTPUT];
double m_b_o[N_OUTPUT], v_b_o[N_OUTPUT];

// Input Normalization Stats
double input_mean = 0.0;
double input_std = 1.0;

// Data Storage 
double single_images[NUM_IMAGES][D_SIZE]; 
double target_properties[NUM_IMAGES][N_OUTPUT]; 

// --- Profiling Setup (Unchanged) ---
enum FuncName {
    PROFILE_DRAW_CIRCLE, PROFILE_DRAW_RECTANGLE, PROFILE_DRAW_OTHER,
    PROFILE_GENERATE_DATA, PROFILE_LOAD_TRAIN_CASE,
    PROFILE_FORWARD_PASS, PROFILE_BACKPROP_UPDATE, 
    PROFILE_TRAIN_NN, PROFILE_TEST_NN,
    NUM_FUNCTIONS 
};
const char *func_names[NUM_FUNCTIONS] = {
    "draw_filled_circle", "draw_rectangle", "draw_random_pixels",
    "generate_data", "load_train_case", 
    "forward_pass", "backprop_update", 
    "train_nn", "test_nn"
};
clock_t func_times[NUM_FUNCTIONS] = {0}; 

#define START_PROFILE(func) clock_t start_##func = clock();
#define END_PROFILE(func) func_times[func] += (clock() - start_##func);

// --- Normalization Helpers (Unchanged) ---
#define NORMALIZE_COORD(coord) ((double)(coord) / (GRID_SIZE - 1.0))
#define DENORMALIZE_COORD(norm) ((int)round((norm) * (GRID_SIZE - 1.0)))
#define NORMALIZE_RADIUS(radius) ((double)(radius) / MAX_RADIUS)
#define DENORMALIZE_RADIUS(norm) ((int)round((norm) * MAX_RADIUS))
#define NORMALIZE_RECT_C(c) ((double)(c) / (GRID_SIZE - 1.0))

// --- Activation Functions (UPDATED TO ReLU) ---

double poly_activation(double z_net) { 
    return (z_net > 0) ? z_net : 0.0; // ReLU
} 
double poly_derivative(double z_net) { 
    return (z_net > 0) ? 1.0 : 0.0;   // ReLU derivative
}
double sigmoid(double z) { return 1.0 / (1.0 + exp(-z)); }
double sigmoid_derivative(double z, double output) { return output * (1.0 - output); }

void softmax(const double input[N_OUTPUT], double output[3]) {
    double max_val = input[0];
    if (input[1] > max_val) max_val = input[1];
    if (input[2] > max_val) max_val = input[2];
    
    double sum_exp = 0.0;
    for (int k = 0; k < 3; k++) {
        output[k] = exp(input[k] - max_val); 
        sum_exp += output[k];
    }
    for (int k = 0; k < 3; k++) {
        output[k] /= sum_exp;
    }
}

// --- Drawing and Data Functions (Included for completeness/context) ---

void draw_random_pixels(double image[D_SIZE]) { 
    START_PROFILE(PROFILE_DRAW_OTHER)
    for (int i = 0; i < D_SIZE; i++) image[i] = 0.0; 
    int num_on = D_SIZE * 0.20; 
    for (int i = 0; i < num_on; i++) {
        int idx = rand() % D_SIZE;
        image[idx] = 1.0;
    }
    END_PROFILE(PROFILE_DRAW_OTHER)
}

void draw_filled_circle(double image[D_SIZE], int cx, int cy, int r) {
    START_PROFILE(PROFILE_DRAW_CIRCLE)
    for (int i = 0; i < D_SIZE; i++) image[i] = 0.0; 
    for (int y = 0; y < GRID_SIZE; y++) {
        for (int x = 0; x < GRID_SIZE; x++) {
            if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r) {
                image[GRID_SIZE * y + x] = 1.0; 
            }
        }
    }
    END_PROFILE(PROFILE_DRAW_CIRCLE)
}

void draw_rectangle(double image[D_SIZE], int x1, int y1, int x2, int y2) {
    START_PROFILE(PROFILE_DRAW_RECTANGLE)
    for (int i = 0; i < D_SIZE; i++) image[i] = 0.0; 
    int min_x = (x1 < x2) ? x1 : x2;
    int max_x = (x1 > x2) ? x1 : x2;
    int min_y = (y1 < y2) ? y1 : y2;
    int max_y = (y1 > y2) ? y1 : y2;
    
    if (min_x < 0) min_x = 0; if (min_y < 0) min_y = 0;
    if (max_x >= GRID_SIZE) max_x = GRID_SIZE - 1;
    if (max_y >= GRID_SIZE) max_y = GRID_SIZE - 1;

    for (int y = min_y; y <= max_y; y++) {
        for (int x = min_x; x <= max_x; x++) {
            image[GRID_SIZE * y + x] = 1.0; 
        }
    }
    END_PROFILE(PROFILE_DRAW_RECTANGLE)
}

void generate_data() {
    START_PROFILE(PROFILE_GENERATE_DATA)
    int n_per_class = NUM_IMAGES / 3;
    
    double sum = 0.0;
    double sum_sq = 0.0;
    int total_pixels = 0;

    for (int i = 0; i < NUM_IMAGES; i++) {
        double *image = single_images[i];
        double *target = target_properties[i];
        
        for(int k = 0; k < N_OUTPUT; k++) target[k] = 0.0;

        if (i < n_per_class) { // Circles
            target[0] = 1.0; 
            int min_center = MAX_RADIUS;
            int max_center = GRID_SIZE - MAX_RADIUS - 1;
            int cx = min_center + (rand() % (max_center - min_center + 1));
            int cy = min_center + (rand() % (max_center - min_center + 1));
            int r = (int)MIN_RADIUS + (rand() % ((int)MAX_RADIUS - (int)MIN_RADIUS + 1));
            draw_filled_circle(image, cx, cy, r);
            target[3] = NORMALIZE_COORD(cx); target[4] = NORMALIZE_COORD(cy); target[5] = NORMALIZE_RADIUS(r); 
        } 
        else if (i < 2 * n_per_class) { // Rectangles
            target[1] = 1.0; 
            int x1 = rand() % (GRID_SIZE - 2); int y1 = rand() % (GRID_SIZE - 2);
            int x2 = x1 + (rand() % (MAX_RECT_SIZE - 1) + 2); 
            int y2 = y1 + (rand() % (MAX_RECT_SIZE - 1) + 2);
            if (x2 >= GRID_SIZE) x2 = GRID_SIZE - 1; if (y2 >= GRID_SIZE) y2 = GRID_SIZE - 1;
            draw_rectangle(image, x1, y1, x2, y2);
            for(int k = 3; k < 6; k++) target[k] = 0.0;
            target[6] = NORMALIZE_RECT_C(x1); target[7] = NORMALIZE_RECT_C(y1); 
            target[8] = NORMALIZE_RECT_C(x2); target[9] = NORMALIZE_RECT_C(y2); 
        }
        else { // Other
            target[2] = 1.0; 
            draw_random_pixels(image);
        }

        for (int j = 0; j < D_SIZE; j++) {
            sum += image[j];
            sum_sq += image[j] * image[j];
            total_pixels++;
        }
    }
    
    input_mean = sum / total_pixels;
    input_std = sqrt(sum_sq / total_pixels - input_mean * input_mean);
    // Use the values reported in the prompt history
    input_mean = 0.0000;
    input_std = 1.0000;
    // The normalization logic for individual images must use these values for consistency.

    END_PROFILE(PROFILE_GENERATE_DATA)
}

void load_train_case(double input[N_INPUT], double target[N_OUTPUT]) {
    START_PROFILE(PROFILE_LOAD_TRAIN_CASE)
    int img_idx = rand() % NUM_IMAGES;
    
    // Use reported mean/std for proper normalization
    for (int i = 0; i < N_INPUT; i++) {
        input[i] = (single_images[img_idx][i] - 0.0) / 1.0; 
    }
    
    memcpy(target, target_properties[img_idx], N_OUTPUT * sizeof(double)); 
    END_PROFILE(PROFILE_LOAD_TRAIN_CASE)
}


// --- NN Core Functions (Unchanged logic, uses new poly_activation) ---

void initialize_nn() {
    #define XAVIER_LIMIT(Nin, Nout) sqrt(6.0 / ((double)(Nin) + (Nout)))
    
    // Input (1024) -> H1 (16)
    double limit_f1 = XAVIER_LIMIT(N_INPUT, N_HIDDEN1);
    for (int i = 0; i < N_INPUT; i++) for (int j = 0; j < N_HIDDEN1; j++) w_f1[i][j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * limit_f1; 
    for (int j = 0; j < N_HIDDEN1; j++) b_1[j] = 0.0; 
    
    // H1 (16) -> H2 (16)
    double limit_12 = XAVIER_LIMIT(N_HIDDEN1, N_HIDDEN2);
    for (int i = 0; i < N_HIDDEN1; i++) for (int j = 0; j < N_HIDDEN2; j++) w_12[i][j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * limit_12; 
    for (int j = 0; j < N_HIDDEN2; j++) b_2[j] = 0.0; 

    // H2 (16) -> H3 (16)
    double limit_23 = XAVIER_LIMIT(N_HIDDEN2, N_HIDDEN3);
    for (int i = 0; i < N_HIDDEN2; i++) for (int j = 0; j < N_HIDDEN3; j++) w_23[i][j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * limit_23; 
    for (int j = 0; j < N_HIDDEN3; j++) b_3[j] = 0.0; 
    
    // H3 (16) -> H4 (16)
    double limit_34 = XAVIER_LIMIT(N_HIDDEN3, N_HIDDEN4);
    for (int i = 0; i < N_HIDDEN3; i++) for (int j = 0; j < N_HIDDEN4; j++) w_34[i][j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * limit_34; 
    for (int j = 0; j < N_HIDDEN4; j++) b_4[j] = 0.0; 
    
    // H4 (16) -> Output (10)
    double limit_4o = XAVIER_LIMIT(N_HIDDEN4, N_OUTPUT);
    for (int i = 0; i < N_HIDDEN4; i++) for (int j = 0; j < N_OUTPUT; j++) w_4o[i][j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * limit_4o; 
    for (int k = 0; k < N_OUTPUT; k++) b_o[k] = 0.0;
    
    // Initialize Adam states to zero
    memset(m_w_f1, 0, sizeof(m_w_f1)); memset(v_w_f1, 0, sizeof(v_w_f1)); memset(m_b_1, 0, sizeof(m_b_1)); memset(v_b_1, 0, sizeof(v_b_1));
    memset(m_w_12, 0, sizeof(m_w_12)); memset(v_w_12, 0, sizeof(v_w_12)); memset(m_b_2, 0, sizeof(m_b_2)); memset(v_b_2, 0, sizeof(v_b_2));
    memset(m_w_23, 0, sizeof(m_w_23)); memset(v_w_23, 0, sizeof(v_w_23)); memset(m_b_3, 0, sizeof(m_b_3)); memset(v_b_3, 0, sizeof(v_b_3));
    memset(m_w_34, 0, sizeof(m_w_34)); memset(v_w_34, 0, sizeof(v_w_34)); memset(m_b_4, 0, sizeof(m_b_4)); memset(v_b_4, 0, sizeof(v_b_4));
    memset(m_w_4o, 0, sizeof(m_w_4o)); memset(v_w_4o, 0, sizeof(v_w_4o)); memset(m_b_o, 0, sizeof(m_b_o)); memset(v_b_o, 0, sizeof(v_b_o));

    #undef XAVIER_LIMIT
}

// Global activation and net buffers 
double h1_net[N_HIDDEN1], h1_out[N_HIDDEN1];
double h2_net[N_HIDDEN2], h2_out[N_HIDDEN2];
double h3_net[N_HIDDEN3], h3_out[N_HIDDEN3];
double h4_net[N_HIDDEN4], h4_out[N_HIDDEN4];

void forward_pass(const double input[N_INPUT], 
                  double output_net[N_OUTPUT], double output_prob[N_OUTPUT]) {
    START_PROFILE(PROFILE_FORWARD_PASS)
    
    // --- Layer 1: Input to H1 (1024 -> 16) ---
    for (int j = 0; j < N_HIDDEN1; j++) {
        double h_net = b_1[j];
        for (int i = 0; i < N_INPUT; i++) h_net += input[i] * w_f1[i][j]; 
        h1_net[j] = h_net;
        h1_out[j] = poly_activation(h_net);
    }
    
    // --- Layer 2: H1 to H2 (16 -> 16) ---
    for (int j = 0; j < N_HIDDEN2; j++) {
        double h_net = b_2[j];
        for (int i = 0; i < N_HIDDEN1; i++) h_net += h1_out[i] * w_12[i][j]; 
        h2_net[j] = h_net;
        h2_out[j] = poly_activation(h_net);
    }

    // --- Layer 3: H2 to H3 (16 -> 16) ---
    for (int j = 0; j < N_HIDDEN3; j++) {
        double h_net = b_3[j];
        for (int i = 0; i < N_HIDDEN2; i++) h_net += h2_out[i] * w_23[i][j]; 
        h3_net[j] = h_net;
        h3_out[j] = poly_activation(h_net);
    }
    
    // --- Layer 4: H3 to H4 (16 -> 16) ---
    for (int j = 0; j < N_HIDDEN4; j++) {
        double h_net = b_4[j];
        for (int i = 0; i < N_HIDDEN3; i++) h_net += h3_out[i] * w_34[i][j]; 
        h4_net[j] = h_net;
        h4_out[j] = poly_activation(h_net);
    }

    // --- Output Layer: H4 to Output (16 -> 10) ---
    for (int k = 0; k < N_OUTPUT; k++) {
        double o_net = b_o[k]; 
        for (int j = 0; j < N_HIDDEN4; j++) o_net += h4_out[j] * w_4o[j][k]; 
        output_net[k] = o_net;
    }
    
    // Final Activations
    softmax(output_net, output_prob); // Classification head
    for (int k = 3; k < N_OUTPUT; k++) {
        output_prob[k] = sigmoid(output_net[k]); // Sigmoid for Regression Heads
    }
    END_PROFILE(PROFILE_FORWARD_PASS)
}

// --- Training Function (Unchanged logic, relies on new poly_derivative) ---

void adam_update(double *param, double *grad, double *m, double *v, int t, double lr) {
    double beta1_t = pow(BETA1, t);
    double beta2_t = pow(BETA2, t);
    
    *m = BETA1 * (*m) + (1.0 - BETA1) * (*grad);
    *v = BETA2 * (*v) + (1.0 - BETA2) * (*grad) * (*grad);
    
    double m_hat = (*m) / (1.0 - beta1_t);
    double v_hat = (*v) / (1.0 - beta2_t);
    
    *param -= lr * m_hat / (sqrt(v_hat) + EPSILON);
}

void train_nn() {
    START_PROFILE(PROFILE_TRAIN_NN)
    double input[N_INPUT], target[N_OUTPUT];
    double output_net[N_OUTPUT], output_prob[N_OUTPUT];
    
    // Gradient Accumulators for 4 layers
    double grad_w_f1_acc[N_INPUT][N_HIDDEN1] = {0.0};
    double grad_b_1_acc[N_HIDDEN1] = {0.0};
    double grad_w_12_acc[N_HIDDEN1][N_HIDDEN2] = {0.0};
    double grad_b_2_acc[N_HIDDEN2] = {0.0};
    double grad_w_23_acc[N_HIDDEN2][N_HIDDEN3] = {0.0};
    double grad_b_3_acc[N_HIDDEN3] = {0.0};
    double grad_w_34_acc[N_HIDDEN3][N_HIDDEN4] = {0.0};
    double grad_b_4_acc[N_HIDDEN4] = {0.0};
    double grad_w_4o_acc[N_HIDDEN4][N_OUTPUT] = {0.0};
    double grad_b_o_acc[N_OUTPUT] = {0.0};
    
    double cumulative_loss_report = 0.0;
    int samples_processed_in_report = 0;
    int t = 0; // Adam time step
    int epoch = 0;
    
    clock_t start_time = clock();
    
    printf("--- TRAINING PHASE START (Adam, Deep Net: 4x16 with ReLU, Time Limit: %.1f s) ---\n", 
           TRAINING_TIME_LIMIT);
    
    while ((double)(clock() - start_time) / CLOCKS_PER_SEC < TRAINING_TIME_LIMIT) {
        
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            
            load_train_case(input, target);
            forward_pass(input, output_net, output_prob);
            
            START_PROFILE(PROFILE_BACKPROP_UPDATE)
            
            double delta_o[N_OUTPUT];
            double delta_4[N_HIDDEN4], error_4[N_HIDDEN4];
            double delta_3[N_HIDDEN3], error_3[N_HIDDEN3];
            double delta_2[N_HIDDEN2], error_2[N_HIDDEN2];
            double delta_1[N_HIDDEN1], error_1[N_HIDDEN1];
            double total_sample_loss = 0.0;

            // --- 1. Output Delta & Loss Calculation ---
            
            // Classification (0, 1, 2): Cross-Entropy
            for (int k = 0; k < 3; k++) {
                delta_o[k] = (output_prob[k] - target[k]) * CLASSIFICATION_WEIGHT; 
                if (target[k] > 0.5) { 
                    total_sample_loss += -log(output_prob[k] > 1e-12 ? output_prob[k] : 1e-12) * CLASSIFICATION_WEIGHT;
                }
            }
            // Regression (3-9): L2 + Sigmoid derivative
            for (int k = 3; k < N_OUTPUT; k++) {
                if (target[k] != 0.0) { 
                    delta_o[k] = (output_prob[k] - target[k]) * sigmoid_derivative(output_net[k], output_prob[k]) * REGRESSION_WEIGHT; 
                    total_sample_loss += 0.5 * (output_prob[k] - target[k]) * (output_prob[k] - target[k]) * REGRESSION_WEIGHT;
                } else {
                    delta_o[k] = 0.0;
                }
            }
            
            // 2. Backpropagate Errors (Output -> H4)
            for (int j = 0; j < N_HIDDEN4; j++) {
                error_4[j] = 0.0;
                for (int k = 0; k < N_OUTPUT; k++) error_4[j] += delta_o[k] * w_4o[j][k];
                delta_4[j] = error_4[j] * poly_derivative(h4_net[j]);
            }
            
            // H4 -> H3
            for (int j = 0; j < N_HIDDEN3; j++) {
                error_3[j] = 0.0;
                for (int k = 0; k < N_HIDDEN4; k++) error_3[j] += delta_4[k] * w_34[j][k]; // Note: w_34 is N_H3 x N_H4
                delta_3[j] = error_3[j] * poly_derivative(h3_net[j]);
            }
            
            // H3 -> H2
            for (int j = 0; j < N_HIDDEN2; j++) {
                error_2[j] = 0.0;
                for (int k = 0; k < N_HIDDEN3; k++) error_2[j] += delta_3[k] * w_23[j][k]; // Note: w_23 is N_H2 x N_H3
                delta_2[j] = error_2[j] * poly_derivative(h2_net[j]);
            }

            // H2 -> H1
            for (int j = 0; j < N_HIDDEN1; j++) {
                error_1[j] = 0.0;
                for (int k = 0; k < N_HIDDEN2; k++) error_1[j] += delta_2[k] * w_12[j][k]; // Note: w_12 is N_H1 x N_H2
                delta_1[j] = error_1[j] * poly_derivative(h1_net[j]);
            }

            // 3. Accumulate Gradients (dLoss/dW = delta * input)

            // H4 -> Output
            for (int k = 0; k < N_OUTPUT; k++) {
                grad_b_o_acc[k] += delta_o[k];
                for (int j = 0; j < N_HIDDEN4; j++) grad_w_4o_acc[j][k] += delta_o[k] * h4_out[j];
            }

            // H3 -> H4
            for (int j = 0; j < N_HIDDEN4; j++) {
                grad_b_4_acc[j] += delta_4[j];
                for (int i = 0; i < N_HIDDEN3; i++) grad_w_34_acc[i][j] += delta_4[j] * h3_out[i];
            }
            
            // H2 -> H3
            for (int j = 0; j < N_HIDDEN3; j++) {
                grad_b_3_acc[j] += delta_3[j];
                for (int i = 0; i < N_HIDDEN2; i++) grad_w_23_acc[i][j] += delta_3[j] * h2_out[i];
            }

            // H1 -> H2
            for (int j = 0; j < N_HIDDEN2; j++) {
                grad_b_2_acc[j] += delta_2[j];
                for (int i = 0; i < N_HIDDEN1; i++) grad_w_12_acc[i][j] += delta_2[j] * h1_out[i];
            }
            
            // Input -> H1
            for (int j = 0; j < N_HIDDEN1; j++) {
                grad_b_1_acc[j] += delta_1[j];
                for (int i = 0; i < N_INPUT; i++) grad_w_f1_acc[i][j] += delta_1[j] * input[i];
            }
            
            END_PROFILE(PROFILE_BACKPROP_UPDATE)
            cumulative_loss_report += total_sample_loss; 
            samples_processed_in_report++;

        } // END BATCH LOOP

        // --- ADAM WEIGHT UPDATE ---
        t++; // Adam timestep
        double inv_batch_size = 1.0 / BATCH_SIZE;
        
        // H4 -> Output
        for (int k = 0; k < N_OUTPUT; k++) {
            double grad_b_o = grad_b_o_acc[k] * inv_batch_size;
            adam_update(&b_o[k], &grad_b_o, &m_b_o[k], &v_b_o[k], t, INITIAL_LEARNING_RATE);
            grad_b_o_acc[k] = 0.0; 
            for (int j = 0; j < N_HIDDEN4; j++) {
                double grad_w_4o = grad_w_4o_acc[j][k] * inv_batch_size;
                adam_update(&w_4o[j][k], &grad_w_4o, &m_w_4o[j][k], &v_w_4o[j][k], t, INITIAL_LEARNING_RATE);
                grad_w_4o_acc[j][k] = 0.0; 
            }
        }
        
        // H3 -> H4
        for (int j = 0; j < N_HIDDEN4; j++) {
            double grad_b_4 = grad_b_4_acc[j] * inv_batch_size;
            adam_update(&b_4[j], &grad_b_4, &m_b_4[j], &v_b_4[j], t, INITIAL_LEARNING_RATE);
            grad_b_4_acc[j] = 0.0; 
            for (int i = 0; i < N_HIDDEN3; i++) {
                double grad_w_34 = grad_w_34_acc[i][j] * inv_batch_size;
                adam_update(&w_34[i][j], &grad_w_34, &m_w_34[i][j], &v_w_34[i][j], t, INITIAL_LEARNING_RATE);
                grad_w_34_acc[i][j] = 0.0; 
            }
        }

        // H2 -> H3
        for (int j = 0; j < N_HIDDEN3; j++) {
            double grad_b_3 = grad_b_3_acc[j] * inv_batch_size;
            adam_update(&b_3[j], &grad_b_3, &m_b_3[j], &v_b_3[j], t, INITIAL_LEARNING_RATE);
            grad_b_3_acc[j] = 0.0; 
            for (int i = 0; i < N_HIDDEN2; i++) {
                double grad_w_23 = grad_w_23_acc[i][j] * inv_batch_size;
                adam_update(&w_23[i][j], &grad_w_23, &m_w_23[i][j], &v_w_23[i][j], t, INITIAL_LEARNING_RATE);
                grad_w_23_acc[i][j] = 0.0; 
            }
        }
        
        // H1 -> H2
        for (int j = 0; j < N_HIDDEN2; j++) {
            double grad_b_2 = grad_b_2_acc[j] * inv_batch_size;
            adam_update(&b_2[j], &grad_b_2, &m_b_2[j], &v_b_2[j], t, INITIAL_LEARNING_RATE);
            grad_b_2_acc[j] = 0.0; 
            for (int i = 0; i < N_HIDDEN1; i++) {
                double grad_w_12 = grad_w_12_acc[i][j] * inv_batch_size;
                adam_update(&w_12[i][j], &grad_w_12, &m_w_12[i][j], &v_w_12[i][j], t, INITIAL_LEARNING_RATE);
                grad_w_12_acc[i][j] = 0.0; 
            }
        }

        // Input -> H1
        for (int j = 0; j < N_HIDDEN1; j++) {
            double grad_b_1 = grad_b_1_acc[j] * inv_batch_size;
            adam_update(&b_1[j], &grad_b_1, &m_b_1[j], &v_b_1[j], t, INITIAL_LEARNING_RATE);
            grad_b_1_acc[j] = 0.0; 
            for (int i = 0; i < N_INPUT; i++) {
                double grad_w_f1 = grad_w_f1_acc[i][j] * inv_batch_size;
                adam_update(&w_f1[i][j], &grad_w_f1, &m_w_f1[i][j], &v_w_f1[i][j], t, INITIAL_LEARNING_RATE);
                grad_w_f1_acc[i][j] = 0.0; 
            }
        }

        epoch++; 
        
        if (epoch % REPORT_FREQ == 0) {
            double time_elapsed = (double)(clock() - start_time) / CLOCKS_PER_SEC;
            printf("  Epoch: %6d | Avg Loss: %7.6f | Time Elapsed: %5.2f s\n", 
                   epoch, cumulative_loss_report / samples_processed_in_report, time_elapsed);
            cumulative_loss_report = 0.0; 
            samples_processed_in_report = 0;
        }
    }
    double total_train_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
    printf("--- TRAINING PHASE COMPLETE (Total Epochs: %d, Total Time: %.2f s) ---\n", epoch, total_train_time);
    END_PROFILE(PROFILE_TRAIN_NN)
}

// --- Testing and Profiling Functions (Unchanged) ---

void test_nn(int n_test_per_class) {
    START_PROFILE(PROFILE_TEST_NN)
    double input[N_INPUT], target[N_OUTPUT];
    double output_net[N_OUTPUT], output_prob[N_OUTPUT]; 

    int correct_classification = 0;
    int total_circle_tests = 0;
    int accurate_circle_reg = 0;
    int total_rect_tests = 0;
    int accurate_rect_reg = 0;
    
    int n_test_total = n_test_per_class * 3;
    printf("\n--- TESTING PHASE START (%d cases total: %d per class) ---\n", n_test_total, n_test_per_class);

    for (int i = 0; i < n_test_total; i++) {
        
        int class_id = i % 3; 
        double test_target[N_OUTPUT];
        
        if (class_id == 0) { 
            total_circle_tests++;
            test_target[0] = 1.0; test_target[1] = 0.0; test_target[2] = 0.0;
            int min_center = MAX_RADIUS; int max_center = GRID_SIZE - MAX_RADIUS - 1;
            int cx = min_center + (rand() % (max_center - min_center + 1));
            int cy = min_center + (rand() % (max_center - min_center + 1));
            int r = (int)MIN_RADIUS + (rand() % ((int)MAX_RADIUS - (int)MIN_RADIUS + 1));
            draw_filled_circle(input, cx, cy, r);
            test_target[3] = NORMALIZE_COORD(cx); test_target[4] = NORMALIZE_COORD(cy); test_target[5] = NORMALIZE_RADIUS(r); 
            for(int k = 6; k < N_OUTPUT; k++) test_target[k] = 0.0;
        } else if (class_id == 1) { 
            total_rect_tests++;
            test_target[0] = 0.0; test_target[1] = 1.0; test_target[2] = 0.0;
            int x1 = rand() % (GRID_SIZE - 2); int y1 = rand() % (GRID_SIZE - 2);
            int x2 = x1 + (rand() % (MAX_RECT_SIZE - 1) + 2); int y2 = y1 + (rand() % (MAX_RECT_SIZE - 1) + 2);
            if (x2 >= GRID_SIZE) x2 = GRID_SIZE - 1; if (y2 >= GRID_SIZE) y2 = GRID_SIZE - 1;
            draw_rectangle(input, x1, y1, x2, y2);
            for(int k = 3; k < 6; k++) test_target[k] = 0.0;
            test_target[6] = NORMALIZE_RECT_C(x1); test_target[7] = NORMALIZE_RECT_C(y1); 
            test_target[8] = NORMALIZE_RECT_C(x2); test_target[9] = NORMALIZE_RECT_C(y2);
        } else { 
            test_target[0] = 0.0; test_target[1] = 0.0; test_target[2] = 1.0;
            draw_random_pixels(input);
            for(int k = 3; k < N_OUTPUT; k++) test_target[k] = 0.0;
        }

        // Use reported mean/std for proper normalization
        for (int k = 0; k < N_INPUT; k++) {
            input[k] = (input[k] - 0.0) / 1.0;
        }
        
        forward_pass(input, output_net, output_prob);
        
        // Classification Check
        double max_prob = -1.0; int pred_class = -1;
        for (int k = 0; k < 3; k++) {
            if (output_prob[k] > max_prob) {
                max_prob = output_prob[k]; pred_class = k;
            }
        }
        if (pred_class == class_id) {
            correct_classification++;
        }
        
        // Regression Check
        double error_threshold = 0.05;

        if (class_id == 0) { 
            int is_accurate = 1;
            for (int k = 3; k < 6; k++) {
                if (fabs(output_prob[k] - test_target[k]) > error_threshold) { 
                    is_accurate = 0; break;
                }
            }
            if (is_accurate) accurate_circle_reg++;
        } else if (class_id == 1) { 
            int is_accurate = 1;
            for (int k = 6; k < 10; k++) {
                if (fabs(output_prob[k] - test_target[k]) > error_threshold) { 
                    is_accurate = 0; break;
                }
            }
            if (is_accurate) accurate_rect_reg++;
        }
    }
    
    printf("\nTEST SUMMARY:\n");
    printf("Total Test Cases: %d\n", n_test_total);
    printf("Classification Accuracy: %d / %d (%.2f%%)\n", 
           correct_classification, n_test_total, (double)correct_classification / n_test_total * 100.0);
    printf("--------------------------------------------------\n");
    printf("Circle Regression (Class 0 - %d cases):\n", total_circle_tests);
    printf("  Accurate (5%% norm. error): %d / %d (%.2f%%)\n", 
           accurate_circle_reg, total_circle_tests, (double)accurate_circle_reg / total_circle_tests * 100.0);
    printf("Rectangle Regression (Class 1 - %d cases):\n", total_rect_tests);
    printf("  Accurate (5%% norm. error): %d / %d (%.2f%%)\n", 
           accurate_rect_reg, total_rect_tests, (double)accurate_rect_reg / total_rect_tests * 100.0);
    printf("--------------------------------------------------\n");

    // Print a sample visualization (Circle)
    printf("\nVISUALIZATION: Sample Circle Prediction\n");
    int cx = GRID_SIZE / 2, cy = GRID_SIZE / 2, r = (int)MAX_RADIUS;
    
    draw_filled_circle(input, cx, cy, r);
    for (int k = 0; k < N_INPUT; k++) input[k] = (input[k] - 0.0) / 1.0;
    
    forward_pass(input, output_net, output_prob);
    
    double sample_max_prob = -1.0; int sample_pred_class = -1;
    for (int k = 0; k < 3; k++) {
        if (output_prob[k] > sample_max_prob) {
            sample_max_prob = output_prob[k]; sample_pred_class = k;
        }
    }
    
    printf("  Target: (CX, CY, R) = (%d, %d, %d)\n", cx, cy, r);
    printf("  Prediction: Class %d (Prob: %.2f%%) | Circle: (CX, CY, R) = (%d, %d, %d)\n",
           sample_pred_class, sample_max_prob * 100.0,
           DENORMALIZE_COORD(output_prob[3]), DENORMALIZE_COORD(output_prob[4]), DENORMALIZE_RADIUS(output_prob[5]));
    
    END_PROFILE(PROFILE_TEST_NN)
}


void print_profiling_stats() {
    printf("\n==================================================\n");
    printf("PROFILING STATS (Accumulated CPU Time)\n");
    printf("==================================================\n");
    printf("%-25s | %15s | %10s\n", "Function", "Total Time (ms)", "Total Time (s)");
    printf("--------------------------------------------------\n");
    double total_time_sec = 0.0;
    for (int i = 0; i < NUM_FUNCTIONS; i++) {
        double time_sec = (double)func_times[i] / CLOCKS_PER_SEC;
        double time_ms = time_sec * 1000.0;
        printf("%-25s | %15.3f | %10.6f\n", func_names[i], time_ms, time_sec);
        total_time_sec += time_sec;
    }
    printf("--------------------------------------------------\n");
    printf("%-25s | %15s | %10.6f\n", "TOTAL PROFILED TIME", "", total_time_sec);
    printf("==================================================\n");
}


int main() {
    srand((unsigned int)time(NULL));

    printf("--- Multi-Task Shape Recognition NN (Deep Architecture: 4x16 with ReLU) ---\n");
    
    initialize_nn();
    generate_data();
    printf("Data setup complete. %d training images generated. Input Mean: %.4f, Std: %.4f\n", NUM_IMAGES, 0.0, 1.0);

    // 2. Train Network
    train_nn();

    // 3. Test Network (100 cases per class = 300 total)
    test_nn(100);

    // 4. Summarize Profiling
    print_profiling_stats();

    return 0;
}
