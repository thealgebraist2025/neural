#define _XOPEN_SOURCE 

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <string.h> 

// --- Configuration ---
#define GRID_SIZE 64       // ⬅️ UPDATED: 64x64 grid
#define D_SIZE (GRID_SIZE * GRID_SIZE) 

// **Input Configuration**
#define NUM_LONGEST_PATHS 8
#define PATH_FEATURE_SIZE (4 + 1) 
#define N_LABYRINTH_PIXELS D_SIZE
// ⬅️ UPDATED: Input size reflects 64x64 pixels + 8 path features
#define N_INPUT (D_SIZE + (NUM_LONGEST_PATHS * PATH_FEATURE_SIZE)) 

#define NUM_SEGMENTS 7
#define N_DIRECTION_CLASSES 4 
// Output = (Start X, Start Y) + (7 * Direction Class) + (7 * Steps) + (Exit X, Exit Y)
#define N_OUTPUT (2 + (NUM_SEGMENTS * N_DIRECTION_CLASSES) + (NUM_SEGMENTS * 1) + 2) 

// **Network & Training Parameters**
#define N_HIDDEN 64       
#define N_SAMPLES_TOTAL 10000 // ⬅️ UPDATED: Number of unique training paths
#define N_TEST_CASES_PER_LABYRINTH 10 
#define INITIAL_LEARNING_RATE 0.00001 // ⬅️ UPDATED: Slightly higher LR for faster convergence
#define N_EPOCHS_TRAIN 1000000 // Set high, but constrained by time
#define COORD_WEIGHT 1.0                 
#define CLASSIFICATION_WEIGHT 1.0 
#define MAX_STEPS 16.0 // Increased to reflect larger map size
#define MAX_TRAINING_SECONDS 120.0 // ⬅️ UPDATED: Max 2 minutes (120 seconds)
#define SOLVED_ERROR_THRESHOLD 0.1 

// **Gradient Clipping Parameter**
#define GRADIENT_CLIP_NORM 1.0 

// Direction Encoding
#define DIR_UP_IDX 0
#define DIR_DOWN_IDX 1
#define DIR_LEFT_IDX 2
#define DIR_RIGHT_IDX 3


// --- Dynamic Globals ---
double current_learning_rate = INITIAL_LEARNING_RATE; 
double last_avg_loss = DBL_MAX;                       
 
// Global Data & Matrices 
double single_labyrinth[D_SIZE]; // ⬅️ Single fixed labyrinth
int fixed_exit_coord[2]; // Exit coord for the single labyrinth

// Neural Network Weights and Biases (must be recompiled for new N_INPUT size)
double w_fh[N_INPUT][N_HIDDEN];    
double b_h[N_HIDDEN]; 
double w_ho[N_HIDDEN][N_OUTPUT];   
double b_o[N_OUTPUT];


// --- Helper Macros ---
#define CLAMP(val, min, max) ((val) < (min) ? (min) : ((val) > (max) ? (max) : (val)))
#define NORMALIZE_COORD(coord) ((double)(coord) / (GRID_SIZE - 1.0))
#define DENORMALIZE_COORD(coord) ((int)(round((coord) * (GRID_SIZE - 1.0))))
#define NORMALIZE_STEPS(steps) ((double)(steps) / MAX_STEPS)
#define DENORMALIZE_STEPS(steps) ((int)(CLAMP(round((steps) * MAX_STEPS), 0, GRID_SIZE)))
#define GET_DIR_OUTPUT_START_IDX(segment) (2 + (segment) * N_DIRECTION_CLASSES) 
#define GET_STEPS_OUTPUT_IDX(segment) (2 + (NUM_SEGMENTS * N_DIRECTION_CLASSES) + (segment))


// --- Function Prototypes ---
void draw_line(double image[D_SIZE], int x1, int y1, int x2, int y2, double val);
void generate_path_and_target(const double labyrinth[D_SIZE], int start_x, int start_y, int exit_x, int exit_y, double target_data[N_OUTPUT]);
void generate_single_labyrinth(); // ⬅️ Renamed and modified
void extract_longest_paths(const double labyrinth[D_SIZE], double feature_output[NUM_LONGEST_PATHS * PATH_FEATURE_SIZE]);
void load_train_case(double input[N_INPUT], double target[N_OUTPUT]); // ⬅️ Simplified to not take index

double calculate_loss(const double output[N_OUTPUT], const double target[N_OUTPUT]);

// Tanh Activation
double tanh_activation(double x) { return tanh(x); }
double tanh_derivative(double tanh_out) { return 1.0 - (tanh_out * tanh_out); }
// Sigmoid Activation
double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
double sigmoid_derivative(double sigmoid_out) { return sigmoid_out * (1.0 - sigmoid_out); }

void softmax(double vector[N_DIRECTION_CLASSES]);
void forward_pass(const double input[N_INPUT], double hidden_net[N_HIDDEN], double hidden_out[N_HIDDEN], double output_net[N_OUTPUT], double output[N_OUTPUT]);
void train_nn();
void test_nn_and_summarize();
int is_path_legal(const double labyrinth[D_SIZE], int start_x, int start_y, const double output_vec[N_OUTPUT]);
void print_labyrinth_and_path(const double input_vec[N_INPUT], const double target_output[N_OUTPUT], const double estimated_output[N_OUTPUT]);


// -----------------------------------------------------------------
// --- DATA GENERATION FUNCTIONS (UPDATED) ---
// -----------------------------------------------------------------

void draw_line(double image[D_SIZE], int x1, int y1, int x2, int y2, double val) {
    // Bresenham's line algorithm (path drawing logic remains the same)
    int dx = abs(x2 - x1);
    int dy = abs(y2 - y1);
    int sx = x1 < x2 ? 1 : -1;
    int sy = y1 < y2 ? 1 : -1;
    int err = dx - dy;

    while (1) {
        if (x1 >= 0 && x1 < GRID_SIZE && y1 >= 0 && y1 < GRID_SIZE) {
            image[GRID_SIZE * y1 + x1] = val; 
        }
        if (x1 == x2 && y1 == y2) break;
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x1 += sx; }
        if (e2 < dx) { err += dy; y1 += sy; }
    }
}

void generate_single_labyrinth() {
    printf("Generating a single fixed %dx%d labyrinth structure.\n", GRID_SIZE, GRID_SIZE);

    for (int i = 0; i < D_SIZE; i++) { single_labyrinth[i] = 0.0; } 
    
    int num_connection_points = GRID_SIZE / 4; 
    int points[num_connection_points + 1][2]; 

    // Generate internal points
    for(int i = 0; i < num_connection_points; i++) {
        points[i][0] = 3 + (rand() % (GRID_SIZE - 6));
        points[i][1] = 3 + (rand() % (GRID_SIZE - 6));
    }
    
    // Generate exit point on a random border
    int side = rand() % 4; 
    if (side == 0) { points[num_connection_points][0] = 1 + (rand() % (GRID_SIZE - 2)); points[num_connection_points][1] = 0; }
    else if (side == 1) { points[num_connection_points][0] = 1 + (rand() % (GRID_SIZE - 2)); points[num_connection_points][1] = GRID_SIZE - 1; }
    else if (side == 2) { points[num_connection_points][0] = 0; points[num_connection_points][1] = 1 + (rand() % (GRID_SIZE - 2)); }
    else { points[num_connection_points][0] = GRID_SIZE - 1; points[num_connection_points][1] = 1 + (rand() % (GRID_SIZE - 2)); }

    fixed_exit_coord[0] = points[num_connection_points][0];
    fixed_exit_coord[1] = points[num_connection_points][1];

    int current_x = points[0][0];
    int current_y = points[0][1];
    
    for (int i = 0; i < num_connection_points; i++) {
        int next_x = points[i+1][0];
        int next_y = points[i+1][1];
        
        // Connect points with right angles (Corridor creation)
        if (rand() % 2) { 
            draw_line(single_labyrinth, current_x, current_y, next_x, current_y, 1.0);
            current_x = next_x;
            draw_line(single_labyrinth, current_x, current_y, current_x, next_y, 1.0);
            current_y = next_y;
        } else {
            draw_line(single_labyrinth, current_x, current_y, current_x, next_y, 1.0);
            current_y = next_y;
            draw_line(single_labyrinth, current_x, current_y, next_x, current_y, 1.0);
            current_x = next_x;
        }
    }
    printf("Labyrinth generated successfully (Exit: %d, %d).\n", fixed_exit_coord[0], fixed_exit_coord[1]);
}


// Pathfinding, Feature Extraction, NN Core functions (generate_path_and_target, extract_longest_paths, initialize_nn, forward_pass, calculate_loss, etc.) 
// remain as implemented in the previous stable revision, with N_INPUT and D_SIZE adjusted globally.


void load_train_case(double input[N_INPUT], double target[N_OUTPUT]) {
    
    // Part 1: Copy Labyrinth Pixels (always the same single labyrinth)
    memcpy(input, single_labyrinth, D_SIZE * sizeof(double));

    // Part 2: Path Instructions (Target) generation
    int start_x, start_y;
    int exit_x = fixed_exit_coord[0];
    int exit_y = fixed_exit_coord[1];

    int attempts = 0;
    // Randomly find a starting point on a path (white pixel)
    do {
        start_x = 1 + (rand() % (GRID_SIZE - 2)); 
        start_y = 1 + (rand() % (GRID_SIZE - 2));
        attempts++;
    } while (input[GRID_SIZE * start_y + start_x] < 0.5 && attempts < 100); 
    
    if (attempts >= 100) {
        start_x = GRID_SIZE/2;
        start_y = GRID_SIZE/2;
    }
    
    generate_path_and_target(input, start_x, start_y, exit_x, exit_y, target);

    // Part 3: Extract Longest Path Features (always the same for this single labyrinth)
    double* feature_start = input + N_LABYRINTH_PIXELS; 
    extract_longest_paths(single_labyrinth, feature_start);
}


void train_nn() {
    printf("Training Vanilla NN with %d inputs and %d hidden neurons (Samples: %d, Initial LR: %.6e, Coords Weight: %.1f, Clip: %.1f, Hidden: Tanh, Reg Output: Sigmoid).\n", 
           N_INPUT, N_HIDDEN, N_SAMPLES_TOTAL, INITIAL_LEARNING_RATE, COORD_WEIGHT, GRADIENT_CLIP_NORM);
    
    clock_t start_time = clock();
    double time_elapsed;
    // Report every 500 epochs to reduce reporting overhead and focus on convergence
    int report_interval = 500; 

    double input[N_INPUT];
    double target[N_OUTPUT];
    double hidden_net[N_HIDDEN]; 
    double hidden_out[N_HIDDEN];
    double output_net[N_OUTPUT]; 
    double output[N_OUTPUT];
    
    double cumulative_loss_report = 0.0;
    int samples_processed_in_report = 0;


    for (int epoch = 0; epoch < N_EPOCHS_TRAIN; epoch++) {
        
        time_elapsed = (double)(clock() - start_time) / CLOCKS_PER_SEC;
        if (time_elapsed >= MAX_TRAINING_SECONDS) {
            printf("\n--- Training stopped: Maximum time limit of %.0f seconds reached after %d epochs. ---\n", MAX_TRAINING_SECONDS, epoch);
            break;
        }

        // Use MODULO to cycle through the conceptual 10000 samples
        // (In reality, the random start point in load_train_case creates a new sample)
        load_train_case(input, target);

        forward_pass(input, hidden_net, hidden_out, output_net, output);
        cumulative_loss_report += calculate_loss(output, target);
        samples_processed_in_report++;
        
        // **Backpropagation and Update** (Same as previous revision)
        // ... [BP and update logic remains here] ...
        
        double delta_o[N_OUTPUT];
        double delta_h[N_HIDDEN]; 
        double error_h[N_HIDDEN] = {0.0};
        
        // 1. Calculate Output Delta 
        for (int k = 0; k < N_OUTPUT; k++) {
            if (k >= 2 && k < (2 + NUM_SEGMENTS * N_DIRECTION_CLASSES)) {
                delta_o[k] = (output[k] - target[k]) * CLASSIFICATION_WEIGHT; 
            } else { 
                double error = output[k] - target[k];
                double sig_deriv = sigmoid_derivative(output[k]); 
                delta_o[k] = error * COORD_WEIGHT * sig_deriv; 
            }
            delta_o[k] = clip_gradient(delta_o[k], GRADIENT_CLIP_NORM);
        }
        
        // 2. Calculate Hidden Delta (Tanh)
        for (int j = 0; j < N_HIDDEN; j++) { 
            for (int k = 0; k < N_OUTPUT; k++) {
                error_h[j] += delta_o[k] * w_ho[j][k];
            }
            double tanh_deriv = tanh_derivative(hidden_out[j]);
            delta_h[j] = error_h[j] * tanh_deriv;
        }
        
        // 3. Update Hidden-to-Output Weights and Biases
        for (int k = 0; k < N_OUTPUT; k++) { 
            for (int j = 0; j < N_HIDDEN; j++) { 
                w_ho[j][k] -= current_learning_rate * delta_o[k] * hidden_out[j]; 
            } 
            b_o[k] -= current_learning_rate * delta_o[k];
        } 
        
        // 4. Update Input-to-Hidden Weights 
        for (int i = 0; i < N_INPUT; i++) { 
            for (int j = 0; j < N_HIDDEN; j++) { 
                double gradient = delta_h[j] * input[i];
                w_fh[i][j] -= current_learning_rate * clip_gradient(gradient, GRADIENT_CLIP_NORM);
            } 
        }
        // 5. Update Hidden Biases
        for (int j = 0; j < N_HIDDEN; j++) { 
            b_h[j] -= current_learning_rate * delta_h[j]; 
        }
        
        // ERROR RATE REPORTING AND LR SCHEDULING
        if ((epoch % report_interval == 0) && epoch != 0) {
            double current_avg_loss = cumulative_loss_report / samples_processed_in_report;
            update_learning_rate(current_avg_loss); 
            
            printf("  Epoch %d/%d completed. Time elapsed: %.2f s. LR: %.6e. Avg Loss (per sample): %.4f\n", 
                   epoch, N_EPOCHS_TRAIN, time_elapsed, current_learning_rate, current_avg_loss);
            
            cumulative_loss_report = 0.0;
            samples_processed_in_report = 0;
        }
    }
}

// -----------------------------------------------------------------
// --- TESTING AND VISUALIZATION FUNCTIONS (Minimal changes) ---
// -----------------------------------------------------------------

void test_nn_and_summarize() {
    
    // Testing is simplified to 1 single labyrinth
    int total_fixed_labyrinths = 1;
    int total_test_runs = N_TEST_CASES_PER_LABYRINTH * 10; // Test 100 paths total (10 runs * 10 checks)

    printf("\n--- STEP 3: LABYRINTH PATH PREDICTION TEST SUMMARY ---\n");
    printf("Testing on the single fixed labyrinth with %d random start points (Total %d tests).\n", 
           total_test_runs, total_test_runs);
    
    double cumulative_test_loss = 0.0;
    int solved_count = 0;
    
    double input[N_INPUT];
    double target[N_OUTPUT];
    double hidden_net[N_HIDDEN];
    double hidden_out[N_HIDDEN]; 
    double output_net[N_OUTPUT];
    double output[N_OUTPUT];

    srand(12345); // Seed for reproducible testing

    for (int test_run = 0; test_run < total_test_runs; test_run++) {
        
        load_train_case(input, target); 
        
        forward_pass(input, hidden_net, hidden_out, output_net, output);
        cumulative_test_loss += calculate_loss(output, target);
        
        int true_start_x = DENORMALIZE_COORD(target[0]);
        int true_start_y = DENORMALIZE_COORD(target[1]);

        int legal_path = is_path_legal(input, true_start_x, true_start_y, output);
        
        // ... [Accuracy check logic remains here] ...
        
        int coords_accurate = 1;
        
        for (int k = 0; k < 2; k++) { 
            if (fabs(output[k] - target[k]) > SOLVED_ERROR_THRESHOLD) { coords_accurate = 0; break; }
        }
        if (coords_accurate) {
            for (int k = N_OUTPUT - 2; k < N_OUTPUT; k++) { 
                if (fabs(output[k] - target[k]) > SOLVED_ERROR_THRESHOLD) { coords_accurate = 0; break; }
            }
        }
        if (coords_accurate) {
            for (int s = 0; s < NUM_SEGMENTS; s++) {
                int steps_idx = GET_STEPS_OUTPUT_IDX(s);
                 if (fabs(output[steps_idx] - target[steps_idx]) > SOLVED_ERROR_THRESHOLD) { coords_accurate = 0; break; }
            }
        }


        if (legal_path && coords_accurate) {
             solved_count++;
        }
    }
    
    printf("\nTEST SUMMARY:\n");
    printf("Total Test Cases: %d\n", total_test_runs);
    printf("Average Loss per Test Case: %.4f\n", cumulative_test_loss / total_test_runs);
    printf("Labyrinths Solved (Accurate Coords + Legal Path): %d / %d (%.2f%%)\n", 
           solved_count, total_test_runs, (double)solved_count / total_test_runs * 100.0);
    printf("--------------------------------------------------\n");


    // VISUALIZATION: Show 10 random examples
    printf("\n--- VISUALIZATION: 10 Random Test Cases ---\n");
    
    srand(time(NULL)); 

    for (int i = 0; i < 10; i++) {
        
        load_train_case(input, target);
        forward_pass(input, hidden_net, hidden_out, output_net, output);

        printf("Labyrinth #%d (Single Fixed ID):\n", i + 1);
        print_labyrinth_and_path(input, target, output);
    }
}


// -----------------------------------------------------------------
// --- MAIN PROGRAM ---
// -----------------------------------------------------------------

int main(int argc, char **argv) {
    srand(time(NULL));

    // 1. Initialize, Generate Single Labyrinth
    initialize_nn();
    generate_single_labyrinth(); // ⬅️ Single labyrinth generation

    // 2. Training (Time limited to 2 minutes)
    train_nn();
    
    // 3. Testing and Visualization
    test_nn_and_summarize();

    return 0;
}
