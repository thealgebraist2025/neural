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
#define N_HIDDEN 64            // z_1 to z_64 
#define N_OUTPUT 3             // Output: (Center X, Center Y, Radius)

// **Training Parameters**
#define NUM_IMAGES 100         
#define BATCH_SIZE 10          
#define TRAIN_TIME_SECONDS 30
#define REPORT_FREQ 5000             
#define INITIAL_LEARNING_RATE 0.00001 
#define COORD_WEIGHT 1.0           
#define MIN_RADIUS 3           
#define MAX_RADIUS 10.0    

// **Testing Parameters**
#define N_TEST_CIRCLES 1000
#define N_TEST_RECTANGLES 1000

// Global Data & Matrices 
double w_fh[N_INPUT][N_HIDDEN];    
double b_h[N_HIDDEN]; 
double w_ho[N_HIDDEN][N_OUTPUT];   
double b_o[N_OUTPUT];

double single_images[NUM_IMAGES][D_SIZE]; 
int target_properties[NUM_IMAGES][N_OUTPUT]; 

// --- Profiling Setup ---
enum FuncName {
    PROFILE_DRAW_CIRCLE,
    PROFILE_DRAW_RECTANGLE, // New function for testing
    PROFILE_GENERATE_CIRCLE,
    PROFILE_LOAD_TRAIN_CASE,
    PROFILE_FORWARD_PASS,
    PROFILE_BACKPROP_UPDATE, // Training-related update step
    PROFILE_TRAIN_NN,
    PROFILE_TEST_NN_CIRCLES,
    PROFILE_TEST_NN_RECTS,
    NUM_FUNCTIONS 
};
const char *func_names[NUM_FUNCTIONS] = {
    "draw_filled_circle", "draw_rectangle", "generate_circle_image", "load_train_case", 
    "forward_pass", "backprop_update", "train_nn", "test_nn_circles", "test_nn_rects"
};
clock_t func_times[NUM_FUNCTIONS] = {0}; 

#define START_PROFILE(func) clock_t start_##func = clock();
#define END_PROFILE(func) func_times[func] += (clock() - start_##func);

// --- Helper Macros and Functions ---
#define NORMALIZE_COORD(coord) ((double)(coord) / (GRID_SIZE - 1.0))
#define NORMALIZE_RADIUS(radius) ((double)(radius) / MAX_RADIUS)
double poly_activation(double z_net) { return z_net * z_net; } 
double poly_derivative(double z_net) { return 2.0 * z_net; }

// --- Drawing Functions ---

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

// NEW: Draw filled rectangle
void draw_rectangle(double image[D_SIZE], int x1, int y1, int x2, int y2) {
    START_PROFILE(PROFILE_DRAW_RECTANGLE)
    for (int i = 0; i < D_SIZE; i++) image[i] = 0.0; 
    int min_x = (x1 < x2) ? x1 : x2;
    int max_x = (x1 > x2) ? x1 : x2;
    int min_y = (y1 < y2) ? y1 : y2;
    int max_y = (y1 > y2) ? y1 : y2;
    
    // Clamp to grid
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

void generate_circle_image(int index) {
    START_PROFILE(PROFILE_GENERATE_CIRCLE)
    int min_center = MAX_RADIUS;
    int max_center = GRID_SIZE - MAX_RADIUS - 1;
    srand((unsigned int)time(NULL) + index * 100); 
    int *properties = target_properties[index];
    int cx = min_center + (rand() % (max_center - min_center + 1));
    int cy = min_center + (rand() % (max_center - min_center + 1));
    int r = (int)MIN_RADIUS + (rand() % ((int)MAX_RADIUS - (int)MIN_RADIUS + 1));
    draw_filled_circle(single_images[index], cx, cy, r);
    properties[0] = cx; properties[1] = cy; properties[2] = r;
    END_PROFILE(PROFILE_GENERATE_CIRCLE)
}

void load_train_case(double input[N_INPUT], double target[N_OUTPUT]) {
    START_PROFILE(PROFILE_LOAD_TRAIN_CASE)
    int img_idx = rand() % NUM_IMAGES;
    memcpy(input, single_images[img_idx], D_SIZE * sizeof(double));
    const int *p = target_properties[img_idx];
    target[0] = NORMALIZE_COORD(p[0]); target[1] = NORMALIZE_COORD(p[1]); target[2] = NORMALIZE_RADIUS(p[2]); 
    END_PROFILE(PROFILE_LOAD_TRAIN_CASE)
}

void initialize_nn() {
    double fan_in_h = (double)N_INPUT;
    double limit_h = sqrt(1.0 / fan_in_h); 
    double fan_in_o = (double)N_HIDDEN;
    double limit_o = sqrt(1.0 / fan_in_o); 
    for (int i = 0; i < N_INPUT; i++) for (int j = 0; j < N_HIDDEN; j++) w_fh[i][j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * limit_h; 
    for (int j = 0; j < N_HIDDEN; j++) { 
        b_h[j] = 0.0; 
        for (int k = 0; k < N_OUTPUT; k++) w_ho[j][k] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * limit_o;
    }
    for (int k = 0; k < N_OUTPUT; k++) b_o[k] = 0.0;
}

void forward_pass(const double input[N_INPUT], double hidden_net[N_HIDDEN], double hidden_out[N_HIDDEN], double output[N_OUTPUT]) {
    START_PROFILE(PROFILE_FORWARD_PASS)
    for (int j = 0; j < N_HIDDEN; j++) {
        double h_net = b_h[j];
        for (int i = 0; i < N_INPUT; i++) h_net += input[i] * w_fh[i][j]; 
        hidden_net[j] = h_net;
        hidden_out[j] = poly_activation(h_net);
    }
    for (int k = 0; k < N_OUTPUT; k++) {
        double o_net = b_o[k]; 
        for (int j = 0; j < N_HIDDEN; j++) o_net += hidden_out[j] * w_ho[j][k]; 
        output[k] = o_net;
    }
    END_PROFILE(PROFILE_FORWARD_PASS)
}

// --- Training Function with Time Limit ---

void train_nn() {
    START_PROFILE(PROFILE_TRAIN_NN)
    double input[N_INPUT], target[N_OUTPUT];
    double hidden_net[N_HIDDEN], hidden_out[N_HIDDEN], output[N_OUTPUT];
    
    double grad_w_fh_acc[N_INPUT][N_HIDDEN] = {0.0};
    double grad_b_h_acc[N_HIDDEN] = {0.0};
    double grad_w_ho_acc[N_HIDDEN][N_OUTPUT] = {0.0};
    double grad_b_o_acc[N_OUTPUT] = {0.0};
    
    double cumulative_loss_report = 0.0;
    int samples_processed_in_report = 0;
    int epoch = 0;
    
    clock_t start_time = clock();
    double time_limit = (double)TRAIN_TIME_SECONDS * CLOCKS_PER_SEC;

    printf("--- TRAINING PHASE START (Batch SGD for %d seconds) ---\n", TRAIN_TIME_SECONDS);
    
    while ((double)(clock() - start_time) < time_limit) {
        
        // --- BATCH LOOP ---
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            
            load_train_case(input, target);
            forward_pass(input, hidden_net, hidden_out, output);
            
            // Backpropagation (Standard SGD)
            START_PROFILE(PROFILE_BACKPROP_UPDATE)
            double delta_o[N_OUTPUT];
            double delta_h[N_HIDDEN]; 
            double error_h[N_HIDDEN];

            for (int k = 0; k < N_OUTPUT; k++) delta_o[k] = (output[k] - target[k]) * COORD_WEIGHT; 
            for (int j = 0; j < N_HIDDEN; j++) {
                error_h[j] = 0.0;
                for (int k = 0; k < N_OUTPUT; k++) error_h[j] += delta_o[k] * w_ho[j][k];
                delta_h[j] = error_h[j] * poly_derivative(hidden_net[j]);
            }
            
            // Accumulate Gradients
            for (int k = 0; k < N_OUTPUT; k++) {
                grad_b_o_acc[k] += delta_o[k];
                for (int j = 0; j < N_HIDDEN; j++) grad_w_ho_acc[j][k] += delta_o[k] * hidden_out[j];
            }
            for (int j = 0; j < N_HIDDEN; j++) {
                grad_b_h_acc[j] += delta_h[j];
                for (int i = 0; i < N_INPUT; i++) grad_w_fh_acc[i][j] += delta_h[j] * input[i];
            }
            END_PROFILE(PROFILE_BACKPROP_UPDATE)
            
            double loss = 0.0; 
            for (int k = 0; k < N_OUTPUT; k++) loss += (output[k] - target[k]) * (output[k] - target[k]) * COORD_WEIGHT;
            cumulative_loss_report += loss; 
            samples_processed_in_report++;
        } // END BATCH LOOP

        // --- WEIGHT UPDATE (Standard SGD) ---
        double inverse_batch_size = 1.0 / BATCH_SIZE;
        double update_rate = INITIAL_LEARNING_RATE * inverse_batch_size;
        
        // 1. Update W_ho and b_o
        for (int k = 0; k < N_OUTPUT; k++) {
            b_o[k] -= update_rate * grad_b_o_acc[k];
            grad_b_o_acc[k] = 0.0; 
            for (int j = 0; j < N_HIDDEN; j++) {
                w_ho[j][k] -= update_rate * grad_w_ho_acc[j][k];
                grad_w_ho_acc[j][k] = 0.0; 
            }
        }
        
        // 2. Update W_fh and b_h
        for (int j = 0; j < N_HIDDEN; j++) {
            b_h[j] -= update_rate * grad_b_h_acc[j];
            grad_b_h_acc[j] = 0.0; 
            for (int i = 0; i < N_INPUT; i++) {
                w_fh[i][j] -= update_rate * grad_w_fh_acc[i][j];
                grad_w_fh_acc[i][j] = 0.0; 
            }
        }
        
        epoch++;
        if (epoch % REPORT_FREQ == 0) {
            printf("  Epoch: %6d | Avg Loss: %7.6f | Time Elapsed: %.2f s\n", 
                   epoch, cumulative_loss_report / samples_processed_in_report, 
                   (double)(clock() - start_time) / CLOCKS_PER_SEC);
            cumulative_loss_report = 0.0; 
            samples_processed_in_report = 0;
        }
    }
    printf("--- TRAINING PHASE COMPLETE (Total Epochs: %d) ---\n", epoch);
    END_PROFILE(PROFILE_TRAIN_NN)
}

// --- Testing Functions ---

// Test on circles (same as before)
void test_nn_circles(int total_test_runs) {
    START_PROFILE(PROFILE_TEST_NN_CIRCLES)
    double input[N_INPUT], target[N_OUTPUT], hidden_net[N_HIDDEN], hidden_out[N_HIDDEN], output[N_OUTPUT];
    double cumulative_test_loss = 0.0;
    int accurate_count = 0;
    
    printf("\n--- TESTING PHASE: CIRCLES (%d cases) ---\n", total_test_runs);
    
    for (int i = 0; i < total_test_runs; i++) {
        // Generate random circle
        int cx = (int)MAX_RADIUS + (rand() % (GRID_SIZE - 2 * (int)MAX_RADIUS));
        int cy = (int)MAX_RADIUS + (rand() % (GRID_SIZE - 2 * (int)MAX_RADIUS));
        int r = (int)MIN_RADIUS + (rand() % ((int)MAX_RADIUS - (int)MIN_RADIUS + 1));
        
        draw_filled_circle(input, cx, cy, r); // Profiling handled inside
        target[0] = NORMALIZE_COORD(cx); 
        target[1] = NORMALIZE_COORD(cy); 
        target[2] = NORMALIZE_RADIUS(r); 
        
        forward_pass(input, hidden_net, hidden_out, output); // Profiling handled inside

        double loss = 0.0;
        double error_threshold = 0.05; 
        int accurate = 1;

        for (int k = 0; k < N_OUTPUT; k++) {
            double diff = output[k] - target[k];
            loss += diff * diff * COORD_WEIGHT;
            if (fabs(diff) > error_threshold) {
                accurate = 0;
            }
        }
        cumulative_test_loss += loss;
        if (accurate) accurate_count++;
    }
    
    printf("CIRCLE TEST SUMMARY:\n");
    printf("Total Test Cases: %d\n", total_test_runs);
    printf("Average Loss per Test Case: %.6f\n", cumulative_test_loss / total_test_runs);
    printf("Accurate Predictions (within 5%% norm. error): %d / %d (%.2f%%)\n", 
           accurate_count, total_test_runs, (double)accurate_count / total_test_runs * 100.0);
    printf("--------------------------------------------------\n");
    END_PROFILE(PROFILE_TEST_NN_CIRCLES)
}

// NEW: Test on rectangles
void test_nn_rectangles(int total_test_runs) {
    START_PROFILE(PROFILE_TEST_NN_RECTS)
    double input[N_INPUT], target[N_OUTPUT], hidden_net[N_HIDDEN], hidden_out[N_HIDDEN], output[N_OUTPUT];
    double cumulative_test_loss = 0.0;
    
    printf("\n--- TESTING PHASE: RECTANGLES (%d cases) ---\n", total_test_runs);

    // Target properties are meaningless for a rectangle test, but we use the output variables
    // to see what the network predicts for Center X, Center Y, and Radius.
    
    for (int i = 0; i < total_test_runs; i++) {
        // Generate random rectangle corners
        int x1 = rand() % GRID_SIZE;
        int y1 = rand() % GRID_SIZE;
        int x2 = rand() % GRID_SIZE;
        int y2 = rand() % GRID_SIZE;
        
        draw_rectangle(input, x1, y1, x2, y2); // Profiling handled inside
        
        // Target is arbitrary (or all zero) since it's an out-of-distribution test.
        // We'll set target to zero just for loss calculation, but the primary metric is inspection.
        target[0] = 0.0; target[1] = 0.0; target[2] = 0.0;
        
        forward_pass(input, hidden_net, hidden_out, output); // Profiling handled inside

        double loss = 0.0;
        for (int k = 0; k < N_OUTPUT; k++) {
            double diff = output[k] - target[k];
            loss += diff * diff * COORD_WEIGHT;
        }
        cumulative_test_loss += loss;
    }
    
    // We cannot compute "accuracy" since there is no target, only loss relative to zero target.
    printf("RECTANGLE TEST SUMMARY:\n");
    printf("Total Test Cases: %d\n", total_test_runs);
    printf("Average (Arbitrary) Loss per Test Case (against zero target): %.6f\n", cumulative_test_loss / total_test_runs);
    printf("NOTE: Network was trained on CIRCLES. This tests its generalization/failure on RECTANGLES.\n");

    // Print a sample prediction on a rectangle
    if (total_test_runs > 0) {
        printf("\nSample Prediction on a Random Rectangle:\n");
        int x1 = GRID_SIZE / 4, y1 = GRID_SIZE / 4;
        int x2 = 3 * GRID_SIZE / 4, y2 = 3 * GRID_SIZE / 4;
        draw_rectangle(input, x1, y1, x2, y2);
        forward_pass(input, hidden_net, hidden_out, output);

        int pred_cx = (int)round(output[0] * (GRID_SIZE - 1.0));
        int pred_cy = (int)round(output[1] * (GRID_SIZE - 1.0));
        int pred_r = (int)round(output[2] * MAX_RADIUS);
        
        printf("  Input (Rectangle): Corner 1 (%d, %d), Corner 2 (%d, %d)\n", x1, y1, x2, y2);
        printf("  Network Prediction (Circle-like interpretation): (CX, CY, R) = (%2d, %2d, %2d)\n", pred_cx, pred_cy, pred_r);
    }
    printf("--------------------------------------------------\n");
    END_PROFILE(PROFILE_TEST_NN_RECTS)
}

// --- Main and Profiling Print Functions ---

void print_profiling_stats() {
    printf("\n==================================================\n");
    printf("PROFILING STATS (Accumulated CPU Time)\n");
    printf("==================================================\n");
    printf("%-20s | %15s | %10s\n", "Function", "Total Time (ms)", "Total Time (s)");
    printf("--------------------------------------------------\n");
    double total_time_sec = 0.0;
    for (int i = 0; i < NUM_FUNCTIONS; i++) {
        double time_sec = (double)func_times[i] / CLOCKS_PER_SEC;
        double time_ms = time_sec * 1000.0;
        printf("%-20s | %15.3f | %10.6f\n", func_names[i], time_ms, time_sec);
        total_time_sec += time_sec;
    }
    printf("--------------------------------------------------\n");
    printf("%-20s | %15s | %10.6f\n", "TOTAL PROFILED TIME", "", total_time_sec);
    printf("==================================================\n");
}

int main() {
    srand(time(NULL));

    printf("--- 32x32 Circle Recognition NN (SVD Removed, Profiling Added) ---\n");
    
    // 1. Initialize and Generate Data
    initialize_nn();
    for (int i = 0; i < NUM_IMAGES; i++) {
        generate_circle_image(i);
    }
    printf("Data setup complete. %d training images generated.\n", NUM_IMAGES);

    // 2. Train Network
    train_nn();

    // 3. Test Network
    test_nn_circles(N_TEST_CIRCLES);
    test_nn_rectangles(N_TEST_RECTANGLES);

    // 4. Summarize Profiling
    print_profiling_stats();

    return 0;
}
