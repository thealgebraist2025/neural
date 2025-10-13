#define _XOPEN_SOURCE 

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <string.h> 

// --- Configuration ---
#define GRID_SIZE 16       
#define D_SIZE (GRID_SIZE * GRID_SIZE) // 256
#define N_INPUT D_SIZE 
#define NUM_SEGMENTS 7

// Output Structure: [Start X/Y] + 7 * [Direction(4) + Steps(1)] + [Exit X/Y]
#define N_DIRECTION_CLASSES 4 
#define N_OUTPUT (2 + (NUM_SEGMENTS * N_DIRECTION_CLASSES) + (NUM_SEGMENTS * 1) + 2) // 39

// **CHANGE 1: Hidden Size Halved**
#define N_HIDDEN 64       
// **CHANGE 2: Fixed Labyrinth Count**
#define N_SAMPLES_FIXED 50 
#define N_SAMPLES_TRAIN N_SAMPLES_FIXED
#define N_SAMPLES_TEST N_SAMPLES_FIXED
// **CHANGE 3: Reduced Test Cases per Labyrinth**
#define N_TEST_CASES_PER_LABYRINTH 10 

// Neural Network Parameters
// **CHANGE 4: Increased Learning Rate for convergence attempt**
#define LEARNING_RATE 0.001 
#define N_EPOCHS_TRAIN 800000 
#define COORD_WEIGHT 10.0      
#define CLASSIFICATION_WEIGHT 1.0 
#define MAX_STEPS 8.0 
// **CHANGE 5: Training Time Halved**
#define MAX_TRAINING_SECONDS 180.0 
#define SOLVED_ERROR_THRESHOLD 0.1 // Max normalized distance error to be considered "solved"

// Direction Encoding
#define DIR_UP_IDX 0
#define DIR_DOWN_IDX 1
#define DIR_LEFT_IDX 2
#define DIR_RIGHT_IDX 3

// --- Dynamic Globals ---
int N_SAMPLES = N_SAMPLES_TRAIN; 
int N_EPOCHS = N_EPOCHS_TRAIN; 
 
// Global Data & Matrices 
double fixed_labyrinths[N_SAMPLES_FIXED][D_SIZE]; 
int fixed_exit_coords[N_SAMPLES_FIXED][2]; // [x, y] of the exit

// Neural Network Weights and Biases 
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
void generate_path_and_target(const double labyrinth[D_SIZE], int start_x, int start_y, int exit_x, int exit_y, double target_data[N_OUTPUT], int draw_flag);
void generate_fixed_labyrinths(); 
void load_train_case(int idx, double input[D_SIZE], double target[N_OUTPUT]);

double calculate_loss(const double output[N_OUTPUT], const double target[N_OUTPUT]);
double relu(double x);
void softmax(double vector[N_DIRECTION_CLASSES]);
void forward_pass(const double input[D_SIZE], double hidden_out[N_HIDDEN], double output[N_OUTPUT]);
void backward_pass_and_update(const double input[D_SIZE], const double hidden_out[N_HIDDEN], const double output[N_OUTPUT], const double target[N_OUTPUT]);
void test_nn_and_summarize();
int is_path_legal(const double labyrinth[D_SIZE], int start_x, int start_y, const double output_vec[N_OUTPUT]);


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


void generate_path_and_target(const double labyrinth[D_SIZE], int start_x, int start_y, int exit_x, int exit_y, double target_data[N_OUTPUT], int draw_flag) {
    
    for (int i = 0; i < N_OUTPUT; i++) { target_data[i] = 0.0; } 

    int current_x = start_x;
    int current_y = start_y;
    int target_points[NUM_SEGMENTS + 1][2]; 
    
    target_points[0][0] = start_x;
    target_points[0][1] = start_y;
    target_points[NUM_SEGMENTS][0] = exit_x;
    target_points[NUM_SEGMENTS][1] = exit_y;

    // Generate intermediate points on open path
    for(int i = 1; i < NUM_SEGMENTS; i++) {
        int attempt = 0;
        do {
            target_points[i][0] = 1 + (rand() % (GRID_SIZE - 2));
            target_points[i][1] = 1 + (rand() % (GRID_SIZE - 2));
            attempt++;
        } while (labyrinth[GRID_SIZE * target_points[i][1] + target_points[i][0]] < 0.5 && attempt < 10);
        if (attempt >= 10) { 
             target_points[i][0] = GRID_SIZE / 2;
             target_points[i][1] = GRID_SIZE / 2;
        }
    }
    
    int current_segment = 0;
    
    // Generate instructions
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        int next_x = target_points[i+1][0];
        int next_y = target_points[i+1][1];
        
        int move_order[2] = {0, 1}; 
        if (rand() % 2) { move_order[0] = 1; move_order[1] = 0; }
        
        for (int m = 0; m < 2; m++) {
            if (current_segment >= NUM_SEGMENTS) goto end_instruction_generation;

            int dir_idx = -1; 
            int steps = 0;
            int dx = 0, dy = 0;

            if (move_order[m] == 0) { // Horizontal move
                dx = next_x - current_x;
                if (dx != 0) {
                    dir_idx = (dx > 0) ? DIR_RIGHT_IDX : DIR_LEFT_IDX;
                    steps = abs(dx);
                    current_x = next_x; 
                }
            } else { // Vertical move
                dy = next_y - current_y;
                if (dy != 0) {
                    dir_idx = (dy > 0) ? DIR_DOWN_IDX : DIR_UP_IDX;
                    steps = abs(dy);
                    current_y = next_y; 
                }
            }
            
            if (dir_idx != -1) {
                // Direction (Classification)
                int dir_start_idx = GET_DIR_OUTPUT_START_IDX(current_segment);
                target_data[dir_start_idx + dir_idx] = 1.0; 
                
                // Steps (Regression)
                int steps_idx = GET_STEPS_OUTPUT_IDX(current_segment);
                target_data[steps_idx] = NORMALIZE_STEPS(steps);
                
                current_segment++;
            }
        }
    }

end_instruction_generation: 
    // Final Target Data assignment (for coordinates)
    target_data[0] = NORMALIZE_COORD(start_x);
    target_data[1] = NORMALIZE_COORD(start_y);
    target_data[N_OUTPUT-2] = NORMALIZE_COORD(exit_x);
    target_data[N_OUTPUT-1] = NORMALIZE_COORD(exit_y);
}


void generate_fixed_labyrinths() {
    printf("Generating %d fixed labyrinth structures (16x16).\n", N_SAMPLES_FIXED);

    for (int k = 0; k < N_SAMPLES_FIXED; ++k) {
        // Initialize with walls
        for (int i = 0; i < D_SIZE; i++) { fixed_labyrinths[k][i] = 0.0; } 
        
        int points[NUM_SEGMENTS + 1][2]; // [x, y]

        // Internal path points
        for(int i = 0; i < NUM_SEGMENTS; i++) {
            points[i][0] = 3 + (rand() % (GRID_SIZE - 6));
            points[i][1] = 3 + (rand() % (GRID_SIZE - 6));
        }
        
        // Last point: Exit on boundary (fixed for this structure)
        int side = rand() % 4; 
        if (side == 0) { points[NUM_SEGMENTS][0] = 1 + (rand() % (GRID_SIZE - 2)); points[NUM_SEGMENTS][1] = 0; }
        else if (side == 1) { points[NUM_SEGMENTS][0] = 1 + (rand() % (GRID_SIZE - 2)); points[NUM_SEGMENTS][1] = GRID_SIZE - 1; }
        else if (side == 2) { points[NUM_SEGMENTS][0] = 0; points[NUM_SEGMENTS][1] = 1 + (rand() % (GRID_SIZE - 2)); }
        else { points[NUM_SEGMENTS][0] = GRID_SIZE - 1; points[NUM_SEGMENTS][1] = 1 + (rand() % (GRID_SIZE - 2)); }

        fixed_exit_coords[k][0] = points[NUM_SEGMENTS][0];
        fixed_exit_coords[k][1] = points[NUM_SEGMENTS][1];

        // Draw the path segments
        int current_x = points[0][0];
        int current_y = points[0][1];
        
        for (int i = 0; i < NUM_SEGMENTS; i++) {
            int next_x = points[i+1][0];
            int next_y = points[i+1][1];
            
            if (rand() % 2) { 
                draw_line(fixed_labyrinths[k], current_x, current_y, next_x, current_y, 1.0);
                current_x = next_x;
                draw_line(fixed_labyrinths[k], current_x, current_y, current_x, next_y, 1.0);
                current_y = next_y;
            } else {
                draw_line(fixed_labyrinths[k], current_x, current_y, current_x, next_y, 1.0);
                current_y = next_y;
                draw_line(fixed_labyrinths[k], current_x, current_y, next_x, current_y, 1.0);
                current_x = next_x;
            }
        }
    }
}


void load_train_case(int idx, double input[D_SIZE], double target[N_OUTPUT]) {
    
    memcpy(input, fixed_labyrinths[idx], D_SIZE * sizeof(double));

    int start_x, start_y;
    int exit_x = fixed_exit_coords[idx][0];
    int exit_y = fixed_exit_coords[idx][1];

    // Find a random, legal start position (on an open path cell)
    int attempts = 0;
    do {
        start_x = 1 + (rand() % (GRID_SIZE - 2)); 
        start_y = 1 + (rand() % (GRID_SIZE - 2));
        attempts++;
    } while (input[GRID_SIZE * start_y + start_x] < 0.5 && attempts < 100); 
    
    // If no path is found, use a fixed open spot (failsafe)
    if (attempts >= 100) {
        start_x = GRID_SIZE/2;
        start_y = GRID_SIZE/2;
    }

    // Generate the path segments and target data for this specific start/exit pair
    generate_path_and_target(input, start_x, start_y, exit_x, exit_y, target, 0);
}


// -----------------------------------------------------------------
// --- NN CORE FUNCTIONS ---
// -----------------------------------------------------------------

void initialize_nn() {
    for (int i = 0; i < N_INPUT; i++) {
        for (int j = 0; j < N_HIDDEN; j++) {
            w_fh[i][j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * sqrt(2.0 / N_INPUT); 
        }
    }
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

double calculate_loss(const double output[N_OUTPUT], const double target[N_OUTPUT]) {
    double total_loss = 0.0;
    
    for (int k = 0; k < N_OUTPUT; k++) {
        if (k >= 2 && k < (2 + NUM_SEGMENTS * N_DIRECTION_CLASSES)) {
            if (target[k] > 0.5) { 
                total_loss += -log(output[k] + 1e-9) * CLASSIFICATION_WEIGHT;
            }
        } else { 
            double error = output[k] - target[k];
            total_loss += error * error * COORD_WEIGHT; 
        }
    }
    return total_loss; 
}


void train_nn() {
    printf("Training Vanilla NN with %d inputs and %d hidden neurons.\n", N_INPUT, N_HIDDEN);
    
    clock_t start_time = clock();
    double time_elapsed;
    int report_interval = N_EPOCHS_TRAIN / 10;
    if (report_interval == 0) report_interval = 1;

    double input[D_SIZE];
    double target[N_OUTPUT];

    for (int epoch = 0; epoch < N_EPOCHS; epoch++) {
        
        time_elapsed = (double)(clock() - start_time) / CLOCKS_PER_SEC;
        if (time_elapsed >= MAX_TRAINING_SECONDS) {
            printf("\n--- Training stopped: Maximum time limit of %.0f seconds reached after %d epochs. ---\n", MAX_TRAINING_SECONDS, epoch);
            break;
        }

        int lab_index = rand() % N_SAMPLES_FIXED;
        
        load_train_case(lab_index, input, target);

        double hidden_out[N_HIDDEN];
        double output[N_OUTPUT];
        
        forward_pass(input, hidden_out, output);
        backward_pass_and_update(input, hidden_out, output, target);

        // ERROR RATE REPORTING
        if ((epoch % report_interval == 0) && epoch != 0) {
            double cumulative_loss = 0.0;
            double temp_hidden[N_HIDDEN];
            double temp_output[N_OUTPUT];
            
            for (int i = 0; i < 100; i++) {
                int idx = rand() % N_SAMPLES_FIXED;
                load_train_case(idx, input, target);
                
                forward_pass(input, temp_hidden, temp_output);
                cumulative_loss += calculate_loss(temp_output, target);
            }

            printf("  Epoch %d/%d completed. Time elapsed: %.2f s. Avg Loss (per sample): %.4f\n", 
                   epoch, N_EPOCHS, time_elapsed, cumulative_loss / 100.0);
        }
    }
}

double relu(double x) { 
    return (x > 0.0) ? x : 0.0; 
}

void softmax(double vector[N_DIRECTION_CLASSES]) {
    double max_val = vector[0];
    for (int k = 1; k < N_DIRECTION_CLASSES; k++) {
        if (vector[k] > max_val) max_val = vector[k];
    }
    
    double sum = 0.0;
    for (int k = 0; k < N_DIRECTION_CLASSES; k++) {
        vector[k] = exp(vector[k] - max_val); 
        sum += vector[k];
    }
    
    for (int k = 0; k < N_DIRECTION_CLASSES; k++) {
        vector[k] /= sum;
    }
}

void forward_pass(const double input[D_SIZE], double hidden_out[N_HIDDEN], double output[N_OUTPUT]) {
    
    // Input to Hidden (ReLU)
    for (int j = 0; j < N_HIDDEN; j++) {
        double h_net = b_h[j];
        for (int i = 0; i < N_INPUT; i++) {
            h_net += input[i] * w_fh[i][j]; 
        }
        hidden_out[j] = relu(h_net); 
    }
    
    // Hidden to Output
    for (int k = 0; k < N_OUTPUT; k++) {
        double o_net = b_o[k]; 
        for (int j = 0; j < N_HIDDEN; j++) { 
            o_net += hidden_out[j] * w_ho[j][k]; 
        } 
        output[k] = o_net; 
    }
    
    // Softmax on Direction segments
    for (int s = 0; s < NUM_SEGMENTS; s++) {
        int dir_start_idx = GET_DIR_OUTPUT_START_IDX(s);
        softmax(&output[dir_start_idx]);
    }
}

void backward_pass_and_update(const double input[D_SIZE], const double hidden_out[N_HIDDEN], const double output[N_OUTPUT], const double target[N_OUTPUT]) {
    double delta_o[N_OUTPUT];
    double delta_h[N_HIDDEN]; 
    double error_h[N_HIDDEN] = {0.0};
    
    // 1. Calculate Output Delta 
    for (int k = 0; k < N_OUTPUT; k++) {
        
        if (k >= 2 && k < (2 + NUM_SEGMENTS * N_DIRECTION_CLASSES)) {
            delta_o[k] = (output[k] - target[k]) * CLASSIFICATION_WEIGHT; 
        } 
        else { 
            double error = output[k] - target[k];
            delta_o[k] = error * COORD_WEIGHT; 
        }
    }
    
    // 2. Calculate Hidden Delta (ReLU)
    for (int j = 0; j < N_HIDDEN; j++) { 
        for (int k = 0; k < N_OUTPUT; k++) {
            error_h[j] += delta_o[k] * w_ho[j][k];
        }
        delta_h[j] = error_h[j] * ((hidden_out[j] > 0.0) ? 1.0 : 0.0);
    }
    
    // 3. Update Hidden-to-Output Weights and Biases
    for (int k = 0; k < N_OUTPUT; k++) { 
        for (int j = 0; j < N_HIDDEN; j++) { 
            w_ho[j][k] -= LEARNING_RATE * delta_o[k] * hidden_out[j]; 
        } 
        b_o[k] -= LEARNING_RATE * delta_o[k];
    } 
    
    // 4. Update Input-to-Hidden Weights
    for (int i = 0; i < N_INPUT; i++) { 
        for (int j = 0; j < N_HIDDEN; j++) { 
            w_fh[i][j] -= LEARNING_RATE * delta_h[j] * input[i]; 
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

void decode_instruction_output(const double output_vec[N_OUTPUT], int segment, char *dir_char, int *steps) {
    double steps_norm = output_vec[GET_STEPS_OUTPUT_IDX(segment)];
    *steps = DENORMALIZE_STEPS(steps_norm);

    int dir_start_idx = GET_DIR_OUTPUT_START_IDX(segment);
    int max_idx = 0;
    double max_val = output_vec[dir_start_idx];

    for (int k = 1; k < N_DIRECTION_CLASSES; k++) {
        if (output_vec[dir_start_idx + k] > max_val) {
            max_val = output_vec[dir_start_idx + k];
            max_idx = k;
        }
    }
    
    if (max_idx == DIR_UP_IDX) *dir_char = 'U';
    else if (max_idx == DIR_DOWN_IDX) *dir_char = 'D';
    else if (max_idx == DIR_LEFT_IDX) *dir_char = 'L';
    else if (max_idx == DIR_RIGHT_IDX) *dir_char = 'R';
    else *dir_char = '?'; 
}

void decode_instruction_target(const double target_vec[N_OUTPUT], int segment, char *dir_char, int *steps) {
    *steps = DENORMALIZE_STEPS(target_vec[GET_STEPS_OUTPUT_IDX(segment)]);

    int dir_start_idx = GET_DIR_OUTPUT_START_IDX(segment);
    int one_hot_idx = -1;

    for (int k = 0; k < N_DIRECTION_CLASSES; k++) {
        if (target_vec[dir_start_idx + k] > 0.5) { 
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

// **NEW FUNCTION:** Checks if the predicted path is physically valid (doesn't cross walls)
int is_path_legal(const double labyrinth[D_SIZE], int start_x, int start_y, const double output_vec[N_OUTPUT]) {
    int current_x = DENORMALIZE_COORD(output_vec[0]);
    int current_y = DENORMALIZE_COORD(output_vec[1]);
    int exit_x = DENORMALIZE_COORD(output_vec[N_OUTPUT-2]);
    int exit_y = DENORMALIZE_COORD(output_vec[N_OUTPUT-1]);

    // Check Start and Exit are on path/open space
    if (labyrinth[GRID_SIZE * current_y + current_x] < 0.5) return 0;
    
    // Path tracing logic
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        char dir_char;
        int steps;
        
        decode_instruction_output(output_vec, i, &dir_char, &steps);
        if (steps == 0) continue; 

        for (int s = 1; s <= steps; s++) {
            int next_x = current_x;
            int next_y = current_y;

            if (dir_char == 'U') next_y--;
            else if (dir_char == 'D') next_y++;
            else if (dir_char == 'L') next_x--;
            else if (dir_char == 'R') next_x++;
            
            // Boundary Check
            if (next_x < 0 || next_x >= GRID_SIZE || next_y < 0 || next_y >= GRID_SIZE) return 0;

            // Wall Check (Must be on a path cell, which is >= 0.5 in the input)
            if (labyrinth[GRID_SIZE * next_y + next_x] < 0.5) return 0;
            
            current_x = next_x;
            current_y = next_y;
        }
    }

    // Check if the final position is close to the predicted exit
    return (abs(current_x - exit_x) + abs(current_y - exit_y) < 2); // Simple Manhattan distance check
}


void print_labyrinth_and_path(const double input_image[D_SIZE], const double target_output[N_OUTPUT], const double estimated_output[N_OUTPUT]) {
    
    char true_path_map[GRID_SIZE][GRID_SIZE];
    char est_path_map[GRID_SIZE][GRID_SIZE];
    
    for (int y = 0; y < GRID_SIZE; y++) {
        for (int x = 0; x < GRID_SIZE; x++) {
            if (input_image[GRID_SIZE * y + x] < 0.5) { 
                true_path_map[y][x] = '#';
                est_path_map[y][x] = '#';
            } else { 
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
    printf("--------------------------------------------------\n");
    
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
    printf("--------------------------------------------------\n");
}


void test_nn_and_summarize() {
    
    int total_fixed_labyrinths = N_SAMPLES_FIXED;
    int total_test_runs = N_SAMPLES_FIXED * N_TEST_CASES_PER_LABYRINTH;

    printf("\n--- STEP 3: LABYRINTH PATH PREDICTION TEST SUMMARY ---\n");
    printf("Testing %d fixed labyrinths with %d random start points each (Total %d tests).\n", 
           total_fixed_labyrinths, N_TEST_CASES_PER_LABYRINTH, total_test_runs);
    
    double cumulative_test_loss = 0.0;
    int solved_count = 0;
    
    double input[D_SIZE];
    double target[N_OUTPUT];
    double hidden_out[N_HIDDEN]; 
    double output[N_OUTPUT];

    // Loop through each of the 50 fixed labyrinths
    for (int lab_idx = 0; lab_idx < total_fixed_labyrinths; lab_idx++) {
        // Run 10 tests for each labyrinth
        for (int test_run = 0; test_run < N_TEST_CASES_PER_LABYRINTH; test_run++) {
            
            load_train_case(lab_idx, input, target); 
            
            forward_pass(input, hidden_out, output);
            cumulative_test_loss += calculate_loss(output, target);
            
            // Check if the predicted path is "Solved"
            int legal_path = is_path_legal(input, DENORMALIZE_COORD(output[0]), DENORMALIZE_COORD(output[1]), output);
            
            // Check coordinate accuracy for start, exit, and segment ends
            int coords_accurate = 1;
            
            // 1. Check Start/Exit points (4 values)
            for (int k = 0; k < 2; k++) { // Start X/Y
                if (fabs(output[k] - target[k]) > SOLVED_ERROR_THRESHOLD) { coords_accurate = 0; break; }
            }
            if (!coords_accurate) goto end_coord_check;

            for (int k = N_OUTPUT - 2; k < N_OUTPUT; k++) { // Exit X/Y
                if (fabs(output[k] - target[k]) > SOLVED_ERROR_THRESHOLD) { coords_accurate = 0; break; }
            }
            
            // 2. Check Steps (7 values) - indirectly checks segment end-points
            for (int s = 0; s < NUM_SEGMENTS; s++) {
                int steps_idx = GET_STEPS_OUTPUT_IDX(s);
                 if (fabs(output[steps_idx] - target[steps_idx]) > SOLVED_ERROR_THRESHOLD) { coords_accurate = 0; break; }
            }

            end_coord_check:;

            if (legal_path && coords_accurate) {
                 solved_count++;
            }
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
    
    for (int i = 0; i < 10; i++) {
        int lab_idx = rand() % N_SAMPLES_FIXED;
        
        load_train_case(lab_idx, input, target);
        forward_pass(input, hidden_out, output);

        printf("Labyrinth #%d (Fixed ID: %d):\n", i + 1, lab_idx);
        print_labyrinth_and_path(input, target, output);
    }
}


// -----------------------------------------------------------------
// --- MAIN PROGRAM ---
// -----------------------------------------------------------------

int main(int argc, char **argv) {
    srand(time(NULL));

    // 1. Initialize, Generate Fixed Data
    initialize_nn();
    generate_fixed_labyrinths();

    // 2. Training
    train_nn();
    
    // 3. Testing and Visualization
    test_nn_and_summarize();

    return 0;
}
