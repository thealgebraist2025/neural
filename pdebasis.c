#define _XOPEN_SOURCE 

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <string.h> 

// --- Configuration ---
#define N_SAMPLES_MAX 52000 
#define GRID_SIZE 32       
#define D_SIZE (GRID_SIZE * GRID_SIZE) // 1024
#define FEATURE_SIZE (GRID_SIZE / 4) * (GRID_SIZE / 4) // 64 (Simplified CNN output)
#define N_INPUT FEATURE_SIZE // 64
#define NUM_SEGMENTS 7

// Output Structure: [Start X/Y] + 7 * [Direction(4) + Steps(1)] + [Exit X/Y]
// 2 + 7 * (4 + 1) + 2 = 39
#define N_DIRECTION_CLASSES 4 // UP, DOWN, LEFT, RIGHT
#define N_OUTPUT (2 + (NUM_SEGMENTS * N_DIRECTION_CLASSES) + (NUM_SEGMENTS * 1) + 2) // 2 + 28 + 7 + 2 = 39 

#define N_HIDDEN 128       
#define N_SAMPLES_TRAIN 5000 
#define N_SAMPLES_TEST 100 

// Neural Network Parameters
#define LEARNING_RATE 0.0002 
#define N_EPOCHS_TRAIN 800000 
#define COORD_WEIGHT 10.0      // Weight for coordinates (Start/Exit) and Steps
#define CLASSIFICATION_WEIGHT 1.0 // Weight for direction classification
#define MAX_STEPS 10.0 // Max steps in one instruction segment for normalization
#define MAX_TRAINING_SECONDS 120.0 // 2 minutes

// Direction Encoding (Used for target index in classification, not regression value)
#define DIR_UP_IDX 0
#define DIR_DOWN_IDX 1
#define DIR_LEFT_IDX 2
#define DIR_RIGHT_IDX 3

// --- Dynamic Globals ---
int N_SAMPLES = N_SAMPLES_TRAIN; 
int N_EPOCHS = N_EPOCHS_TRAIN; 
 
// Global Data & Matrices 
double dataset[N_SAMPLES_MAX][D_SIZE]; // Store raw image data
double targets[N_SAMPLES_MAX][N_OUTPUT]; // Store complex target data

// Neural Network Weights and Biases 
double w_if[D_SIZE][FEATURE_SIZE]; // Input (1024) to Feature (64) - Simplified
double w_fh[N_INPUT][N_HIDDEN];    // Feature (64) to Hidden (128)
double b_h[N_HIDDEN]; 
double w_ho[N_HIDDEN][N_OUTPUT];   // Hidden (128) to Output (39)
double b_o[N_OUTPUT];

// Test Data 
double test_data[N_SAMPLES_TEST][D_SIZE];
double test_targets[N_SAMPLES_TEST][N_OUTPUT];


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
void generate_labyrinth(double image[D_SIZE], double target_data[N_OUTPUT]);
void load_data(int n_samples, double set[][D_SIZE], double target_set[][N_OUTPUT]); 
void load_train_set();
void load_test_set();

void initialize_nn();
void extract_features(const double input[D_SIZE], double feature_out[N_INPUT]);
void train_nn(const double input_set[N_SAMPLES_MAX][D_SIZE]);
double relu(double x);
void softmax(double vector[N_DIRECTION_CLASSES]);
void forward_pass(const double input[D_SIZE], double feature_out[N_INPUT], double hidden_out[N_HIDDEN], double output[N_OUTPUT]);
void backward_pass_and_update(const double input[D_SIZE], const double feature_out[N_INPUT], const double hidden_out[N_HIDDEN], const double output[N_OUTPUT], const double target[N_OUTPUT]);

void test_labyrinth_path(int n_set_size, const double input_set[][D_SIZE], const double target_set[][N_OUTPUT]);


// -----------------------------------------------------------------
// --- DATA GENERATION FUNCTIONS ---
// -----------------------------------------------------------------

void draw_line(double image[D_SIZE], int x1, int y1, int x2, int y2, double val) {
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
        if (e2 < dx) { err += dx; y1 += sy; }
    }
}

// Generates a random, non-intersecting, mostly horizontal/vertical path
// and sets the target data.
void generate_labyrinth(double image[D_SIZE], double target_data[N_OUTPUT]) {
    for (int i = 0; i < D_SIZE; i++) { image[i] = 0.0; } 
    
    // Path value is 1.0 for easier normalization (0.0 wall, 1.0 path)
    double path_val = 1.0; 

    // 1. Define 7 random intermediate points inside the image
    int target_points[NUM_SEGMENTS + 1][2]; // [x, y]

    // Start point (x>2, x<GRID_SIZE-3, same for y)
    // Ensures start is well inside the grid
    target_points[0][0] = 3 + (rand() % (GRID_SIZE - 6));
    target_points[0][1] = 3 + (rand() % (GRID_SIZE - 6));
    
    // Intermediate points (must be internal)
    for(int i = 1; i < NUM_SEGMENTS; i++) {
        // Points are confined well inside the grid (e.g., 3 to 28)
        target_points[i][0] = 3 + (rand() % (GRID_SIZE - 6));
        target_points[i][1] = 3 + (rand() % (GRID_SIZE - 6));
    }
    
    // Last point: Exit on boundary (x=0, x=31, y=0, or y=31)
    int side = rand() % 4; // 0:Top, 1:Bottom, 2:Left, 3:Right
    if (side == 0) { // Top (y=0)
        target_points[NUM_SEGMENTS][0] = 1 + (rand() % (GRID_SIZE - 2));
        target_points[NUM_SEGMENTS][1] = 0;
    } else if (side == 1) { // Bottom (y=31)
        target_points[NUM_SEGMENTS][0] = 1 + (rand() % (GRID_SIZE - 2));
        target_points[NUM_SEGMENTS][1] = GRID_SIZE - 1;
    } else if (side == 2) { // Left (x=0)
        target_points[NUM_SEGMENTS][0] = 0;
        target_points[NUM_SEGMENTS][1] = 1 + (rand() % (GRID_SIZE - 2));
    } else { // Right (x=31)
        target_points[NUM_SEGMENTS][0] = GRID_SIZE - 1;
        target_points[NUM_SEGMENTS][1] = 1 + (rand() % (GRID_SIZE - 2));
    }

    // 2. Generate non-intersecting path segments (Horizontal/Vertical moves only)
    int current_x = target_points[0][0];
    int current_y = target_points[0][1];
    
    // Instructions (Classification: 4 neurons for Direction, Regression: 1 for Steps)
    int current_segment = 0;
    
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        int next_x = target_points[i+1][0];
        int next_y = target_points[i+1][1];
        
        // Randomly choose to move H or V first
        int move_order[2] = {0, 1}; 
        if (rand() % 2) { move_order[0] = 1; move_order[1] = 0; }
        
        for (int m = 0; m < 2; m++) {
            
            if (current_segment >= NUM_SEGMENTS) {
                goto end_instruction_generation;
            }

            int dir_idx = -1; // -1 means no instruction this move
            int steps = 0;

            if (move_order[m] == 0) { // Horizontal move
                int dx = next_x - current_x;
                if (dx != 0) {
                    // Draw line from current to next X coordinate
                    draw_line(image, current_x, current_y, next_x, current_y, path_val);
                    dir_idx = (dx > 0) ? DIR_RIGHT_IDX : DIR_LEFT_IDX;
                    steps = abs(dx);
                    current_x = next_x; // Update current position
                }
            } else { // Vertical move
                int dy = next_y - current_y;
                if (dy != 0) {
                    // Draw line from current to next Y coordinate
                    draw_line(image, current_x, current_y, current_x, next_y, path_val);
                    dir_idx = (dy > 0) ? DIR_DOWN_IDX : DIR_UP_IDX;
                    steps = abs(dy);
                    current_y = next_y; // Update current position
                }
            }
            
            // Store valid instruction
            if (dir_idx != -1) {
                // Direction (Classification)
                int dir_start_idx = GET_DIR_OUTPUT_START_IDX(current_segment);
                for(int k = 0; k < N_DIRECTION_CLASSES; k++) {
                    target_data[dir_start_idx + k] = 0.0;
                }
                target_data[dir_start_idx + dir_idx] = 1.0; // One-hot encoding
                
                // Steps (Regression)
                int steps_idx = GET_STEPS_OUTPUT_IDX(current_segment);
                target_data[steps_idx] = NORMALIZE_STEPS(steps);
                
                current_segment++;
            }
        }
    }

end_instruction_generation: 

    // 3. Set Target Data
    
    // Start X/Y
    target_data[0] = NORMALIZE_COORD(target_points[0][0]);
    target_data[1] = NORMALIZE_COORD(target_points[0][1]);
    
    // Instructions: Pad remaining segments with 'no-op' (UP/0 steps)
    while (current_segment < NUM_SEGMENTS) {
        // Direction: UP (first index in one-hot)
        int dir_start_idx = GET_DIR_OUTPUT_START_IDX(current_segment);
        for(int k = 0; k < N_DIRECTION_CLASSES; k++) {
            target_data[dir_start_idx + k] = 0.0;
        }
        target_data[dir_start_idx + DIR_UP_IDX] = 1.0; 

        // Steps: 0.0
        int steps_idx = GET_STEPS_OUTPUT_IDX(current_segment);
        target_data[steps_idx] = 0.0;
        
        current_segment++;
    }

    // Exit X/Y
    target_data[N_OUTPUT-2] = NORMALIZE_COORD(target_points[NUM_SEGMENTS][0]);
    target_data[N_OUTPUT-1] = NORMALIZE_COORD(target_points[NUM_SEGMENTS][1]);
}


void load_data(int n_samples, double set[][D_SIZE], double target_set[][N_OUTPUT]) {
    for (int k = 0; k < n_samples; ++k) {
        generate_labyrinth(set[k], target_set[k]);
    }
}
void load_train_set() {
    printf("Generating TRAINING dataset (%d labyrinths). N_OUTPUT=%d (Classification + Regression).\n", N_SAMPLES_TRAIN, N_OUTPUT);
    load_data(N_SAMPLES_TRAIN, dataset, targets);
    N_SAMPLES = N_SAMPLES_TRAIN;
}
void load_test_set() {
    printf("Generating TEST dataset (%d labyrinths).\n", N_SAMPLES_TEST);
    load_data(N_SAMPLES_TEST, test_data, test_targets);
}


// -----------------------------------------------------------------
// --- NN CORE FUNCTIONS ---
// -----------------------------------------------------------------

void initialize_nn() {
    // Initialize Input-to-Feature (Skipped, using Feature Extraction)
    // Initialize Feature-to-Hidden
    for (int i = 0; i < N_INPUT; i++) {
        for (int j = 0; j < N_HIDDEN; j++) {
            w_fh[i][j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * sqrt(2.0 / N_INPUT); // Kaiming/He initialization for ReLU
        }
    }
    // Initialize Hidden-to-Output
    for (int j = 0; j < N_HIDDEN; j++) {
        b_h[j] = 0.0;
        for (int k = 0; k < N_OUTPUT; k++) {
            w_ho[j][k] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * sqrt(2.0 / N_HIDDEN);
        }
    }
    for (int k = 0; k < N_OUTPUT; k++) {
        b_o[k] = 0.0;
    }
}

// Simplified Feature Extraction (32x32 -> 8x8 by 4x4 Mean Pooling)
void extract_features(const double input[D_SIZE], double feature_out[N_INPUT]) {
    int pool_size = 4;
    int feature_idx = 0;
    for (int y = 0; y < GRID_SIZE; y += pool_size) {
        for (int x = 0; x < GRID_SIZE; x += pool_size) {
            double sum = 0.0;
            for (int py = 0; py < pool_size; py++) {
                for (int px = 0; px < pool_size; px++) {
                    sum += input[GRID_SIZE * (y + py) + (x + px)];
                }
            }
            // Mean pooling for feature output
            feature_out[feature_idx++] = sum / (pool_size * pool_size);
        }
    }
}

void train_nn(const double input_set[N_SAMPLES_MAX][D_SIZE]) {
    printf("Training on %d features (simplified CNN) with %d-output (Cls + Reg).\n", N_INPUT, N_OUTPUT);
    
    clock_t start_time = clock();
    double time_elapsed;

    for (int epoch = 0; epoch < N_EPOCHS; epoch++) {
        
        time_elapsed = (double)(clock() - start_time) / CLOCKS_PER_SEC;
        if (time_elapsed >= MAX_TRAINING_SECONDS) {
            printf("\n--- Training stopped: Maximum time limit of %.0f seconds reached after %d epochs. ---\n", MAX_TRAINING_SECONDS, epoch);
            break;
        }

        int sample_index = rand() % N_SAMPLES;
        
        double feature_out[N_INPUT];
        double hidden_out[N_HIDDEN];
        double output[N_OUTPUT];
        
        forward_pass(input_set[sample_index], feature_out, hidden_out, output);
        backward_pass_and_update(input_set[sample_index], feature_out, hidden_out, output, targets[sample_index]);

        if (N_EPOCHS > 1000 && (epoch % (N_EPOCHS / 10) == 0) && epoch != 0) {
            printf("  Epoch %d/%d completed. Time elapsed: %.2f s\n", epoch, N_EPOCHS, time_elapsed);
        }
    }
}

double relu(double x) { 
    return (x > 0.0) ? x : 0.0; 
}

// Apply Softmax to a 4-element vector (used for Direction Classification)
void softmax(double vector[N_DIRECTION_CLASSES]) {
    double max_val = vector[0];
    for (int k = 1; k < N_DIRECTION_CLASSES; k++) {
        if (vector[k] > max_val) max_val = vector[k];
    }
    
    double sum = 0.0;
    for (int k = 0; k < N_DIRECTION_CLASSES; k++) {
        vector[k] = exp(vector[k] - max_val); // Numerical stability trick
        sum += vector[k];
    }
    
    for (int k = 0; k < N_DIRECTION_CLASSES; k++) {
        vector[k] /= sum;
    }
}

void forward_pass(const double input[D_SIZE], double feature_out[N_INPUT], double hidden_out[N_HIDDEN], double output[N_OUTPUT]) {
    
    // 1. Feature Extraction (Simplified CNN)
    extract_features(input, feature_out);

    // 2. Feature to Hidden (ReLU)
    for (int j = 0; j < N_HIDDEN; j++) {
        double h_net = b_h[j];
        for (int i = 0; i < N_INPUT; i++) {
            h_net += feature_out[i] * w_fh[i][j]; 
        }
        hidden_out[j] = relu(h_net); // Using ReLU
    }
    
    // 3. Hidden to Output
    for (int k = 0; k < N_OUTPUT; k++) {
        double o_net = b_o[k]; 
        for (int j = 0; j < N_HIDDEN; j++) { 
            o_net += hidden_out[j] * w_ho[j][k]; 
        } 
        output[k] = o_net; 
    }
    
    // 4. Softmax on Direction segments (applied after all calculations)
    for (int s = 0; s < NUM_SEGMENTS; s++) {
        int dir_start_idx = GET_DIR_OUTPUT_START_IDX(s);
        softmax(&output[dir_start_idx]);
    }
}

void backward_pass_and_update(const double input[D_SIZE], const double feature_out[N_INPUT], const double hidden_out[N_HIDDEN], const double output[N_OUTPUT], const double target[N_OUTPUT]) {
    double delta_o[N_OUTPUT];
    double delta_h[N_HIDDEN]; 
    double error_h[N_HIDDEN] = {0.0};
    
    // 1. Calculate Output Delta (Classification and Regression)
    for (int k = 0; k < N_OUTPUT; k++) {
        
        // Handle Direction Classification (Cross-Entropy Loss)
        // Indices 2 to 29 (4 * 7 segments)
        if (k >= 2 && k < (2 + NUM_SEGMENTS * N_DIRECTION_CLASSES)) {
            // Softmax Cross-Entropy derivative: output[k] - target[k]
            delta_o[k] = (output[k] - target[k]) * CLASSIFICATION_WEIGHT; 
        } 
        // Handle Regression (Coordinates and Steps - L2 Loss)
        else { 
            double error = output[k] - target[k];
            // L2 derivative: error * 1 (error itself)
            delta_o[k] = error * COORD_WEIGHT; 
        }
    }
    
    // 2. Calculate Hidden Delta (ReLU)
    for (int j = 0; j < N_HIDDEN; j++) { 
        for (int k = 0; k < N_OUTPUT; k++) {
            error_h[j] += delta_o[k] * w_ho[j][k];
        }
        // ReLU derivative is 1 or 0 (implicitly using hidden_out[j] > 0)
        delta_h[j] = error_h[j] * ((hidden_out[j] > 0.0) ? 1.0 : 0.0);
    }
    
    // 3. Update Hidden-to-Output Weights and Biases
    for (int k = 0; k < N_OUTPUT; k++) { 
        for (int j = 0; j < N_HIDDEN; j++) { 
            w_ho[j][k] -= LEARNING_RATE * delta_o[k] * hidden_out[j]; 
        } 
        b_o[k] -= LEARNING_RATE * delta_o[k];
    } 
    
    // 4. Update Feature-to-Hidden Weights
    for (int i = 0; i < N_INPUT; i++) { 
        for (int j = 0; j < N_HIDDEN; j++) { 
            w_fh[i][j] -= LEARNING_RATE * delta_h[j] * feature_out[i]; 
        } 
    }
    // 5. Update Hidden Biases
    for (int j = 0; j < N_HIDDEN; j++) { 
        b_h[j] -= LEARNING_RATE * delta_h[j]; 
    }
}


// -----------------------------------------------------------------
// --- TESTING AND VISUALIZATION ---
// -----------------------------------------------------------------

// Helper to decode a single instruction based on classification output
void decode_instruction_output(const double output_vec[N_OUTPUT], int segment, char *dir_char, int *steps) {
    
    // 1. Steps (Regression)
    double steps_norm = output_vec[GET_STEPS_OUTPUT_IDX(segment)];
    *steps = DENORMALIZE_STEPS(steps_norm);

    // 2. Direction (Classification) - Find Max Index
    int dir_start_idx = GET_DIR_OUTPUT_START_IDX(segment);
    int max_idx = 0;
    double max_val = output_vec[dir_start_idx];

    for (int k = 1; k < N_DIRECTION_CLASSES; k++) {
        if (output_vec[dir_start_idx + k] > max_val) {
            max_val = output_vec[dir_start_idx + k];
            max_idx = k;
        }
    }
    
    // Map index back to character
    if (max_idx == DIR_UP_IDX) *dir_char = 'U';
    else if (max_idx == DIR_DOWN_IDX) *dir_char = 'D';
    else if (max_idx == DIR_LEFT_IDX) *dir_char = 'L';
    else if (max_idx == DIR_RIGHT_IDX) *dir_char = 'R';
    else *dir_char = '?'; 
}

// Helper to decode target instruction
void decode_instruction_target(const double target_vec[N_OUTPUT], int segment, char *dir_char, int *steps) {
    
    // 1. Steps (Regression)
    *steps = DENORMALIZE_STEPS(target_vec[GET_STEPS_OUTPUT_IDX(segment)]);

    // 2. Direction (Classification) - Find One-Hot Index
    int dir_start_idx = GET_DIR_OUTPUT_START_IDX(segment);
    int one_hot_idx = -1;

    for (int k = 0; k < N_DIRECTION_CLASSES; k++) {
        if (target_vec[dir_start_idx + k] > 0.5) { // Find the 1.0 in the one-hot
            one_hot_idx = k;
            break;
        }
    }
    
    if (one_hot_idx == DIR_UP_IDX) *dir_char = 'U';
    else if (one_hot_idx == DIR_DOWN_IDX) *dir_char = 'D';
    else if (one_hot_idx == DIR_LEFT_IDX) *dir_char = 'L';
    else if (one_hot_idx == DIR_RIGHT_IDX) *dir_char = 'R';
    else *dir_char = '?';
}

void draw_path(char map[GRID_SIZE][GRID_SIZE], int start_x, int start_y, int exit_x, int exit_y, const double output_vec[N_OUTPUT], int is_target) {
    int current_x = start_x;
    int current_y = start_y;
    
    if (current_x >= 0 && current_x < GRID_SIZE && current_y >= 0 && current_y < GRID_SIZE) {
        map[current_y][current_x] = '0';
    }
    if (exit_x >= 0 && exit_x < GRID_SIZE && exit_y >= 0 && exit_y < GRID_SIZE) {
        map[exit_y][exit_x] = 'E';
    }
    
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        char dir_char;
        int steps;
        
        if (is_target) {
            decode_instruction_target(output_vec, i, &dir_char, &steps);
        } else {
            decode_instruction_output(output_vec, i, &dir_char, &steps);
        }
        
        if (steps == 0) continue; 

        for (int s = 1; s <= steps; s++) {
            if (dir_char == 'U') current_y--;
            else if (dir_char == 'D') current_y++;
            else if (dir_char == 'L') current_x--;
            else if (dir_char == 'R') current_x++;
            
            if (current_x >= 0 && current_x < GRID_SIZE && current_y >= 0 && current_y < GRID_SIZE) {
                if (map[current_y][current_x] == ' ') {
                    map[current_y][current_x] = '*';
                }
            }
        }
        
        if (i < NUM_SEGMENTS - 1) {
            if (current_x >= 0 && current_x < GRID_SIZE && current_y >= 0 && current_y < GRID_SIZE) {
                if (map[current_y][current_x] != 'E') {
                    map[current_y][current_x] = '1' + i; 
                }
            }
        }
    }
}


void print_labyrinth_and_path(const double input_image[D_SIZE], const double target_output[N_OUTPUT], const double estimated_output[N_OUTPUT]) {
    
    char true_path_map[GRID_SIZE][GRID_SIZE];
    char est_path_map[GRID_SIZE][GRID_SIZE];
    
    for (int y = 0; y < GRID_SIZE; y++) {
        for (int x = 0; x < GRID_SIZE; x++) {
            // Input image is already scaled [0, 1]
            if (input_image[GRID_SIZE * y + x] < 0.5) { // Wall
                true_path_map[y][x] = '#';
                est_path_map[y][x] = '#';
            } else { // Open Path
                true_path_map[y][x] = ' ';
                est_path_map[y][x] = ' ';
            }
        }
    }
    
    int true_start_x = DENORMALIZE_COORD(target_output[0]);
    int true_start_y = DENORMALIZE_COORD(target_output[1]);
    int true_exit_x = DENORMALIZE_COORD(target_output[N_OUTPUT-2]);
    int true_exit_y = DENORMALIZE_COORD(target_output[N_OUTPUT-1]);

    int est_start_x = DENORMALIZE_COORD(estimated_output[0]);
    int est_start_y = DENORMALIZE_COORD(estimated_output[1]);
    int est_exit_x = DENORMALIZE_COORD(estimated_output[N_OUTPUT-2]);
    int est_exit_y = DENORMALIZE_COORD(estimated_output[N_OUTPUT-1]);
    
    draw_path(true_path_map, true_start_x, true_start_y, true_exit_x, true_exit_y, target_output, 1);
    draw_path(est_path_map, est_start_x, est_start_y, est_exit_x, est_exit_y, estimated_output, 0);
    
    printf("\n--- True Path (Target) | Predicted Path (Output) ---\n");
    printf("TRUE Start: (%d, %d), Exit: (%d, %d)\n", true_start_x, true_start_y, true_exit_x, true_exit_y);
    printf("EST Start:  (%d, %d), Exit: (%d, %d)\n", est_start_x, est_start_y, est_exit_x, est_exit_y);
    printf("--------------------------------------------------------------------------------------------------\n");
    
    for (int y = 0; y < GRID_SIZE; y++) {
        for (int x = 0; x < GRID_SIZE; x++) {
            printf("%c", true_path_map[y][x]);
        }
        printf(" | ");
        for (int x = 0; x < GRID_SIZE; x++) {
            printf("%c", est_path_map[y][x]);
        }
        printf("\n");
    }
    printf("--------------------------------------------------------------------------------------------------\n");
}


void test_labyrinth_path(int n_set_size, const double input_set[][D_SIZE], const double target_set[][N_OUTPUT]) {
    double feature_out[N_INPUT];
    double hidden_out[N_HIDDEN]; 
    double output[N_OUTPUT];
    
    printf("\n--- STEP 3: LABYRINTH PATH PREDICTION TEST (%d Samples) ---\n", n_set_size);
    
    for (int i = 0; i < n_set_size; i++) {
        if (i < 5) { 
            forward_pass(input_set[i], feature_out, hidden_out, output);
            print_labyrinth_and_path(input_set[i], target_set[i], output);
        }
    }
    
    // Print example instructions for the first visualized sample
    printf("--- Example Instructions (Sample 1) ---\n");
    forward_pass(input_set[0], feature_out, hidden_out, output);
    
    printf("True Instructions:\n");
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        char dir_char;
        int steps;
        decode_instruction_target(target_set[0], i, &dir_char, &steps);
        int steps_idx = GET_STEPS_OUTPUT_IDX(i);
        printf("  Segment %d: (%c, %d) -> Normalized Steps: %.2f\n", i+1, dir_char, steps, target_set[0][steps_idx]);
    }
    
    printf("\nPredicted Instructions:\n");
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        char dir_char;
        int steps;
        decode_instruction_output(output, i, &dir_char, &steps);
        int dir_start_idx = GET_DIR_OUTPUT_START_IDX(i);
        int steps_idx = GET_STEPS_OUTPUT_IDX(i);
        printf("  Segment %d: (%c, %d) -> Max Dir: %.2f, Steps: %.2f\n", i+1, dir_char, steps, output[dir_start_idx + (dir_char == 'U' ? 0 : dir_char == 'D' ? 1 : dir_char == 'L' ? 2 : 3)], output[steps_idx]);
    }
}


// -----------------------------------------------------------------
// --- MAIN PROGRAM ---
// -----------------------------------------------------------------

int main(int argc, char **argv) {
    srand(time(NULL));

    // 1. Initialize and Load Data
    initialize_nn();
    load_train_set();
    load_test_set();

    // 2. Training
    train_nn(dataset);
    
    // 3. Testing and Visualization
    test_labyrinth_path(N_SAMPLES_TEST, test_data, test_targets);

    return 0;
}
