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
#define N_INPUT D_SIZE         // x_1 to x_1024 
#define N_HIDDEN 128           // INCREASED: z_1 to z_128 
#define N_OUTPUT 10            // 3 Class + 3 Circle Params + 4 Rectangle Params

// **Training Parameters**
#define NUM_IMAGES 3000        // 1000 Circles + 1000 Rectangles + 1000 Other
#define N_TRAINING_EPOCHS 20000 
#define BATCH_SIZE 32          // INCREASED for better gradient estimate
#define REPORT_FREQ 500             
#define INITIAL_LEARNING_RATE 0.0001 // Adam's default learning rate
#define CLASSIFICATION_WEIGHT 1.0  // Weight for Cross-Entropy Loss
#define REGRESSION_WEIGHT 0.1      // Weight for L2 Regression Loss
#define MIN_RADIUS 3           
#define MAX_RADIUS 10.0    
#define MAX_RECT_SIZE (GRID_SIZE - 2) // Max side length for rectangle

// **Adam Optimizer Parameters**
#define BETA1 0.9
#define BETA2 0.999
#define EPSILON 1e-8

// Global Data & Matrices 
double w_fh[N_INPUT][N_HIDDEN];    
double b_h[N_HIDDEN]; 
double w_ho[N_HIDDEN][N_OUTPUT];   
double b_o[N_OUTPUT];

// Adam State Variables (Momentum and RMSProp accumulation)
double m_w_fh[N_INPUT][N_HIDDEN], v_w_fh[N_INPUT][N_HIDDEN];
double m_b_h[N_HIDDEN], v_b_h[N_HIDDEN];
double m_w_ho[N_HIDDEN][N_OUTPUT], v_w_ho[N_HIDDEN][N_OUTPUT];
double m_b_o[N_OUTPUT], v_b_o[N_OUTPUT];

// Input Normalization Stats
double input_mean = 0.0;
double input_std = 1.0;

// Data Storage (3000 cases)
double single_images[NUM_IMAGES][D_SIZE]; 
// Target: [3 Class One-Hot, 3 Circle Norm, 4 Rect Norm]
double target_properties[NUM_IMAGES][N_OUTPUT]; 

// --- Profiling Setup (same as before) ---
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

// --- Normalization Helpers ---
#define NORMALIZE_COORD(coord) ((double)(coord) / (GRID_SIZE - 1.0))
#define DENORMALIZE_COORD(norm) ((int)round((norm) * (GRID_SIZE - 1.0)))
#define NORMALIZE_RADIUS(radius) ((double)(radius) / MAX_RADIUS)
#define DENORMALIZE_RADIUS(norm) ((int)round((norm) * MAX_RADIUS))
#define NORMALIZE_RECT_C(c) ((double)(c) / (GRID_SIZE - 1.0))

// --- Activation Functions ---
double poly_activation(double z_net) { return z_net * z_net; } 
double poly_derivative(double z_net) { return 2.0 * z_net; }

// Softmax for the first 3 outputs (Classes 0-2)
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

// --- Drawing Functions ---

// Function to draw random pixels (Class 2: Other)
void draw_random_pixels(double image[D_SIZE]) {
    START_PROFILE(PROFILE_DRAW_OTHER)
    for (int i = 0; i < D_SIZE; i++) image[i] = 0.0; 
    int num_on = D_SIZE * 0.20; // 20% on
    for (int i = 0; i < num_on; i++) {
        int idx = rand() % D_SIZE;
        image[idx] = 1.0;
    }
    END_PROFILE(PROFILE_DRAW_OTHER)
}

// ... draw_filled_circle and draw_rectangle are similar to previous, but include profiling ...

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

// --- Data Generation and Normalization ---

void generate_data() {
    START_PROFILE(PROFILE_GENERATE_DATA)
    int n_per_class = NUM_IMAGES / 3;
    
    // Calculate mean/std for normalization (simple estimate)
    double sum = 0.0;
    double sum_sq = 0.0;
    int total_pixels = 0;

    for (int i = 0; i < NUM_IMAGES; i++) {
        double *image = single_images[i];
        double *target = target_properties[i];
        
        // Target: [C, R, O] for classification
        target[0] = target[1] = target[2] = 0.0; 
        // Target: Regression parameters 
        for(int k = 3; k < N_OUTPUT; k++) target[k] = 0.0;

        // Class 0: Circles
        if (i < n_per_class) {
            target[0] = 1.0; // Circle class
            int min_center = MAX_RADIUS;
            int max_center = GRID_SIZE - MAX_RADIUS - 1;
            int cx = min_center + (rand() % (max_center - min_center + 1));
            int cy = min_center + (rand() % (max_center - min_center + 1));
            int r = (int)MIN_RADIUS + (rand() % ((int)MAX_RADIUS - (int)MIN_RADIUS + 1));
            draw_filled_circle(image, cx, cy, r);
            target[3] = NORMALIZE_COORD(cx); 
            target[4] = NORMALIZE_COORD(cy); 
            target[5] = NORMALIZE_RADIUS(r); 
        } 
        // Class 1: Rectangles
        else if (i < 2 * n_per_class) {
            target[1] = 1.0; // Rectangle class
            int x1 = rand() % (GRID_SIZE - 2); 
            int y1 = rand() % (GRID_SIZE - 2);
            int x2 = x1 + (rand() % (MAX_RECT_SIZE - 1) + 2); 
            int y2 = y1 + (rand() % (MAX_RECT_SIZE - 1) + 2);
            if (x2 >= GRID_SIZE) x2 = GRID_SIZE - 1;
            if (y2 >= GRID_SIZE) y2 = GRID_SIZE - 1;
            draw_rectangle(image, x1, y1, x2, y2);
            target[6] = NORMALIZE_RECT_C(x1); 
            target[7] = NORMALIZE_RECT_C(y1); 
            target[8] = NORMALIZE_RECT_C(x2); 
            target[9] = NORMALIZE_RECT_C(y2); 
        }
        // Class 2: Other (Random Pixels)
        else {
            target[2] = 1.0; // Other class
            draw_random_pixels(image);
        }

        // Accumulate stats
        for (int j = 0; j < D_SIZE; j++) {
            sum += image[j];
            sum_sq += image[j] * image[j];
            total_pixels++;
        }
    }
    
    // Finalize normalization stats
    input_mean = sum / total_pixels;
    input_std = sqrt(sum_sq / total_pixels - input_mean * input_mean);
    if (input_std < 1e-6) input_std = 1.0;

    END_PROFILE(PROFILE_GENERATE_DATA)
}

void load_train_case(double input[N_INPUT], double target[N_OUTPUT]) {
    START_PROFILE(PROFILE_LOAD_TRAIN_CASE)
    int img_idx = rand() % NUM_IMAGES;
    
    // Copy and Normalize Input
    for (int i = 0; i < N_INPUT; i++) {
        input[i] = (single_images[img_idx][i] - input_mean) / input_std;
    }
    
    // Copy Target
    memcpy(target, target_properties[img_idx], N_OUTPUT * sizeof(double)); 
    END_PROFILE(PROFILE_LOAD_TRAIN_CASE)
}

// --- NN Core Functions ---

void initialize_nn() {
    // Xavier initialization
    double fan_in_h = (double)N_INPUT;
    double limit_h = sqrt(6.0 / (fan_in_h + N_HIDDEN)); 
    double fan_in_o = (double)N_HIDDEN;
    double limit_o = sqrt(6.0 / (fan_in_o + N_OUTPUT)); 
    
    for (int i = 0; i < N_INPUT; i++) for (int j = 0; j < N_HIDDEN; j++) w_fh[i][j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * limit_h; 
    for (int j = 0; j < N_HIDDEN; j++) { 
        b_h[j] = 0.0; 
        for (int k = 0; k < N_OUTPUT; k++) w_ho[j][k] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * limit_o;
    }
    for (int k = 0; k < N_OUTPUT; k++) b_o[k] = 0.0;

    // Initialize Adam state variables to zero
    memset(m_w_fh, 0, sizeof(m_w_fh)); memset(v_w_fh, 0, sizeof(v_w_fh));
    memset(m_b_h, 0, sizeof(m_b_h)); memset(v_b_h, 0, sizeof(v_b_h));
    memset(m_w_ho, 0, sizeof(m_w_ho)); memset(v_w_ho, 0, sizeof(v_w_ho));
    memset(m_b_o, 0, sizeof(m_b_o)); memset(v_b_o, 0, sizeof(v_b_o));
}

void forward_pass(const double input[N_INPUT], double hidden_net[N_HIDDEN], double hidden_out[N_HIDDEN], double output_net[N_OUTPUT], double output_prob[N_OUTPUT]) {
    START_PROFILE(PROFILE_FORWARD_PASS)
    // 1. Hidden Layer
    for (int j = 0; j < N_HIDDEN; j++) {
        double h_net = b_h[j];
        for (int i = 0; i < N_INPUT; i++) h_net += input[i] * w_fh[i][j]; 
        hidden_net[j] = h_net;
        hidden_out[j] = poly_activation(h_net);
    }
    // 2. Output Layer (Net)
    for (int k = 0; k < N_OUTPUT; k++) {
        double o_net = b_o[k]; 
        for (int j = 0; j < N_HIDDEN; j++) o_net += hidden_out[j] * w_ho[j][k]; 
        output_net[k] = o_net;
    }
    // 3. Output Layer (Probabilities/Identity)
    softmax(output_net, output_prob); // Classification head (0, 1, 2)
    for (int k = 3; k < N_OUTPUT; k++) {
        output_prob[k] = output_net[k]; // Regression heads (Identity)
    }
    END_PROFILE(PROFILE_FORWARD_PASS)
}

// --- Training Function with Adam Optimizer ---

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
    double hidden_net[N_HIDDEN], hidden_out[N_HIDDEN];
    double output_net[N_OUTPUT], output_prob[N_OUTPUT];
    
    double grad_w_fh_acc[N_INPUT][N_HIDDEN] = {0.0};
    double grad_b_h_acc[N_HIDDEN] = {0.0};
    double grad_w_ho_acc[N_HIDDEN][N_OUTPUT] = {0.0};
    double grad_b_o_acc[N_OUTPUT] = {0.0};
    
    double cumulative_loss_report = 0.0;
    int samples_processed_in_report = 0;
    int t = 0; // Adam time step
    
    printf("--- TRAINING PHASE START (Adam, Epochs: %d, Batch: %d) ---\n", N_TRAINING_EPOCHS, BATCH_SIZE);
    
    for (int epoch = 0; epoch < N_TRAINING_EPOCHS; epoch++) {
        
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            
            load_train_case(input, target);
            forward_pass(input, hidden_net, hidden_out, output_net, output_prob);
            
            START_PROFILE(PROFILE_BACKPROP_UPDATE)
            
            double delta_o[N_OUTPUT];
            double delta_h[N_HIDDEN]; 
            double error_h[N_HIDDEN];
            double total_batch_loss = 0.0;

            // --- 1. Output Delta Calculation (Classification & Regression) ---
            
            // Classification Head (0, 1, 2): Cross-Entropy derivative (output_prob - target_one_hot)
            for (int k = 0; k < 3; k++) {
                delta_o[k] = (output_prob[k] - target[k]) * CLASSIFICATION_WEIGHT; 
                if (target[k] > 0.5) { // Only calculate CE loss for the target class
                    total_batch_loss += -log(output_prob[k] > 1e-12 ? output_prob[k] : 1e-12) * CLASSIFICATION_WEIGHT;
                }
            }
            
            // Regression Heads (3-9): L2 derivative (output_net - target)
            for (int k = 3; k < N_OUTPUT; k++) {
                // Apply a mask: Only penalize if target parameter is non-zero (i.e., belongs to the shape)
                if (target[k] != 0.0) { 
                    delta_o[k] = (output_net[k] - target[k]) * REGRESSION_WEIGHT; 
                    total_batch_loss += 0.5 * (output_net[k] - target[k]) * (output_net[k] - target[k]) * REGRESSION_WEIGHT;
                } else {
                    delta_o[k] = 0.0;
                }
            }
            
            // --- 2. Hidden Delta Calculation ---
            for (int j = 0; j < N_HIDDEN; j++) {
                error_h[j] = 0.0;
                for (int k = 0; k < N_OUTPUT; k++) error_h[j] += delta_o[k] * w_ho[j][k];
                delta_h[j] = error_h[j] * poly_derivative(hidden_net[j]);
            }
            
            // --- 3. Accumulate Gradients ---
            for (int k = 0; k < N_OUTPUT; k++) {
                grad_b_o_acc[k] += delta_o[k];
                for (int j = 0; j < N_HIDDEN; j++) grad_w_ho_acc[j][k] += delta_o[k] * hidden_out[j];
            }
            for (int j = 0; j < N_HIDDEN; j++) {
                grad_b_h_acc[j] += delta_h[j];
                for (int i = 0; i < N_INPUT; i++) grad_w_fh_acc[i][j] += delta_h[j] * input[i];
            }
            
            END_PROFILE(PROFILE_BACKPROP_UPDATE)
            cumulative_loss_report += total_batch_loss; 
            samples_processed_in_report++;

        } // END BATCH LOOP

        // --- ADAM WEIGHT UPDATE ---
        t++; // Adam timestep
        double inv_batch_size = 1.0 / BATCH_SIZE;
        
        // Output Layer (W_ho, b_o)
        for (int k = 0; k < N_OUTPUT; k++) {
            double grad_b_o = grad_b_o_acc[k] * inv_batch_size;
            adam_update(&b_o[k], &grad_b_o, &m_b_o[k], &v_b_o[k], t, INITIAL_LEARNING_RATE);
            grad_b_o_acc[k] = 0.0; 
            
            for (int j = 0; j < N_HIDDEN; j++) {
                double grad_w_ho = grad_w_ho_acc[j][k] * inv_batch_size;
                adam_update(&w_ho[j][k], &grad_w_ho, &m_w_ho[j][k], &v_w_ho[j][k], t, INITIAL_LEARNING_RATE);
                grad_w_ho_acc[j][k] = 0.0; 
            }
        }
        
        // Hidden Layer (W_fh, b_h)
        for (int j = 0; j < N_HIDDEN; j++) {
            double grad_b_h = grad_b_h_acc[j] * inv_batch_size;
            adam_update(&b_h[j], &grad_b_h, &m_b_h[j], &v_b_h[j], t, INITIAL_LEARNING_RATE);
            grad_b_h_acc[j] = 0.0; 
            
            for (int i = 0; i < N_INPUT; i++) {
                double grad_w_fh = grad_w_fh_acc[i][j] * inv_batch_size;
                adam_update(&w_fh[i][j], &grad_w_fh, &m_w_fh[i][j], &v_w_fh[i][j], t, INITIAL_LEARNING_RATE);
                grad_w_fh_acc[i][j] = 0.0; 
            }
        }
        
        if ((epoch + 1) % REPORT_FREQ == 0) {
            printf("  Epoch: %6d | Avg Loss: %7.6f | Batches: %d\n", 
                   epoch + 1, cumulative_loss_report / samples_processed_in_report, t);
            cumulative_loss_report = 0.0; 
            samples_processed_in_report = 0;
        }
    }
    printf("--- TRAINING PHASE COMPLETE (Total Epochs: %d) ---\n", N_TRAINING_EPOCHS);
    END_PROFILE(PROFILE_TRAIN_NN)
}

// --- Testing Function ---

// Test function now calculates metrics for all three tasks
void test_nn(int n_test_per_class) {
    START_PROFILE(PROFILE_TEST_NN)
    double input[N_INPUT], target[N_OUTPUT];
    double hidden_net[N_HIDDEN], hidden_out[N_HIDDEN];
    double output_net[N_OUTPUT], output_prob[N_OUTPUT];

    int correct_classification = 0;
    int total_circle_tests = 0;
    int accurate_circle_reg = 0;
    int total_rect_tests = 0;
    int accurate_rect_reg = 0;
    
    // Test on (Circle, Rectangle, Other)
    int n_test_total = n_test_per_class * 3;
    printf("\n--- TESTING PHASE START (%d cases total) ---\n", n_test_total);

    for (int i = 0; i < n_test_total; i++) {
        
        // Randomly select a test case from the 3 data generation methods
        int class_id = rand() % 3;
        double test_target[N_OUTPUT];
        
        // Generate random test case for chosen class
        if (class_id == 0) { // Circle
            total_circle_tests++;
            test_target[0] = 1.0; test_target[1] = 0.0; test_target[2] = 0.0;
            int min_center = MAX_RADIUS;
            int max_center = GRID_SIZE - MAX_RADIUS - 1;
            int cx = min_center + (rand() % (max_center - min_center + 1));
            int cy = min_center + (rand() % (max_center - min_center + 1));
            int r = (int)MIN_RADIUS + (rand() % ((int)MAX_RADIUS - (int)MIN_RADIUS + 1));
            draw_filled_circle(input, cx, cy, r);
            test_target[3] = NORMALIZE_COORD(cx); test_target[4] = NORMALIZE_COORD(cy); test_target[5] = NORMALIZE_RADIUS(r); 
            for(int k = 6; k < N_OUTPUT; k++) test_target[k] = 0.0;
        } else if (class_id == 1) { // Rectangle
            total_rect_tests++;
            test_target[0] = 0.0; test_target[1] = 1.0; test_target[2] = 0.0;
            int x1 = rand() % (GRID_SIZE - 2); int y1 = rand() % (GRID_SIZE - 2);
            int x2 = x1 + (rand() % (MAX_RECT_SIZE - 1) + 2); 
            int y2 = y1 + (rand() % (MAX_RECT_SIZE - 1) + 2);
            if (x2 >= GRID_SIZE) x2 = GRID_SIZE - 1; if (y2 >= GRID_SIZE) y2 = GRID_SIZE - 1;
            draw_rectangle(input, x1, y1, x2, y2);
            for(int k = 3; k < 6; k++) test_target[k] = 0.0;
            test_target[6] = NORMALIZE_RECT_C(x1); test_target[7] = NORMALIZE_RECT_C(y1); 
            test_target[8] = NORMALIZE_RECT_C(x2); test_target[9] = NORMALIZE_RECT_C(y2);
        } else { // Other
            test_target[0] = 0.0; test_target[1] = 0.0; test_target[2] = 1.0;
            draw_random_pixels(input);
            for(int k = 3; k < N_OUTPUT; k++) test_target[k] = 0.0;
        }

        // Apply input normalization to the test case
        for (int k = 0; k < N_INPUT; k++) {
            input[k] = (input[k] - input_mean) / input_std;
        }
        
        forward_pass(input, hidden_net, hidden_out, output_net, output_prob);
        
        // --- Classification Check ---
        double max_prob = -1.0;
        int pred_class = -1;
        for (int k = 0; k < 3; k++) {
            if (output_prob[k] > max_prob) {
                max_prob = output_prob[k];
                pred_class = k;
            }
        }
        if (pred_class == class_id) {
            correct_classification++;
        }
        
        // --- Regression Check (5% error threshold in normalized space) ---
        double error_threshold = 0.05;

        if (class_id == 0) { // Circle
            int is_accurate = 1;
            for (int k = 3; k < 6; k++) {
                if (fabs(output_net[k] - test_target[k]) > error_threshold) {
                    is_accurate = 0;
                    break;
                }
            }
            if (is_accurate) accurate_circle_reg++;
        } else if (class_id == 1) { // Rectangle
            int is_accurate = 1;
            for (int k = 6; k < 10; k++) {
                if (fabs(output_net[k] - test_target[k]) > error_threshold) {
                    is_accurate = 0;
                    break;
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
    printf("Circle Regression (Class 0):\n");
    printf("  Accurate (5%% norm. error): %d / %d (%.2f%%)\n", 
           accurate_circle_reg, total_circle_tests, (double)accurate_circle_reg / total_circle_tests * 100.0);
    printf("Rectangle Regression (Class 1):\n");
    printf("  Accurate (5%% norm. error): %d / %d (%.2f%%)\n", 
           accurate_rect_reg, total_rect_tests, (double)accurate_rect_reg / total_rect_tests * 100.0);
    printf("--------------------------------------------------\n");

    // Print a sample visualization (Circle)
    printf("\nVISUALIZATION: Sample Circle Prediction\n");
    int cx = GRID_SIZE / 2, cy = GRID_SIZE / 2, r = (int)MAX_RADIUS;
    draw_filled_circle(input, cx, cy, r);
    for (int k = 0; k < N_INPUT; k++) input[k] = (input[k] - input_mean) / input_std;
    forward_pass(input, hidden_net, hidden_out, output_net, output_prob);
    printf("  Target: (CX, CY, R) = (%d, %d, %d)\n", cx, cy, r);
    printf("  Prediction: Class %d (Prob: %.2f%%) | Circle: (CX, CY, R) = (%d, %d, %d)\n",
           (int)round(output_prob[0]), max_prob * 100.0,
           DENORMALIZE_COORD(output_net[3]), DENORMALIZE_COORD(output_net[4]), DENORMALIZE_RADIUS(output_net[5]));
    
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
    srand(time(NULL));

    printf("--- Multi-Task Shape Recognition NN (Adam, Normalized, Converged) ---\n");
    
    // 1. Initialize and Generate Data (3000 cases, with normalization stats)
    initialize_nn();
    generate_data();
    printf("Data setup complete. %d training images generated. Input Mean: %.4f, Std: %.4f\n", NUM_IMAGES, input_mean, input_std);

    // 2. Train Network
    train_nn();

    // 3. Test Network
    test_nn(100);

    // 4. Summarize Profiling
    print_profiling_stats();

    return 0;
}
