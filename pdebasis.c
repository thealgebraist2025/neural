#define _XOPEN_SOURCE // Define this to ensure math constants like M_PI are available

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <string.h> 

// --- Configuration ---
#define N_SAMPLES_MAX 52000 
#define GRID_SIZE 32       
#define D_SIZE (GRID_SIZE * GRID_SIZE) 
#define N_INPUT D_SIZE     
#define NUM_SEGMENTS 7
// [Start X, Start Y] + 7 * [Direction, Steps] + [Exit X, Exit Y] = 2 + 14 + 2 = 18
#define N_OUTPUT (2 + (NUM_SEGMENTS * 2) + 2) 
#define N_HIDDEN 128       
#define N_SAMPLES_TRAIN 5000 
#define N_SAMPLES_TEST 100 

// Neural Network Parameters
#define LEARNING_RATE 0.0002 
#define N_EPOCHS_TRAIN 800000 
#define PATH_WEIGHT 5.0     
#define MAX_STEPS 10.0 // Max steps in one instruction segment for normalization

// Direction Encoding (Used as regression targets, rounded in output)
#define DIR_UP 0.0
#define DIR_DOWN 1.0
#define DIR_LEFT 2.0
#define DIR_RIGHT 3.0

// --- Dynamic Globals ---
int N_SAMPLES = N_SAMPLES_TRAIN; 
int N_EPOCHS = N_EPOCHS_TRAIN; 
 
// Global Data & Matrices 
double dataset[N_SAMPLES_MAX][D_SIZE]; 
double targets[N_SAMPLES_MAX][N_OUTPUT]; 

// Neural Network Weights and Biases 
double w_ih[N_INPUT][N_HIDDEN]; double b_h[N_HIDDEN]; 
double w_ho[N_HIDDEN][N_OUTPUT]; double b_o[N_OUTPUT];

// Test Data - Targets are generated alongside the image for the test set
double test_data[N_SAMPLES_TEST][D_SIZE];
double test_targets[N_SAMPLES_TEST][N_OUTPUT];


// --- Helper Macros ---
#define CLAMP(val, min, max) ((val) < (min) ? (min) : ((val) > (max) ? (max) : (val)))
#define NORMALIZE_COORD(coord) ((double)(coord) / (GRID_SIZE - 1.0))
#define DENORMALIZE_COORD(coord) ((int)(round((coord) * (GRID_SIZE - 1.0))))
#define NORMALIZE_STEPS(steps) ((double)(steps) / MAX_STEPS)
#define DENORMALIZE_STEPS(steps) ((int)(CLAMP(round((steps) * MAX_STEPS), 0, GRID_SIZE)))


// --- Function Prototypes ---
void draw_line(double image[D_SIZE], int x1, int y1, int x2, int y2, double val);
void generate_labyrinth(double image[D_SIZE], double target_data[N_OUTPUT]);
void load_data(int n_samples, double set[][D_SIZE], double target_set[][N_OUTPUT]); // Corrected signature
void load_train_set();
void load_test_set();

void initialize_nn();
void train_nn(const double input_set[N_SAMPLES_MAX][N_INPUT]);
double sigmoid(double x);
void forward_pass(const double input[N_INPUT], double hidden_out[N_HIDDEN], double output[N_OUTPUT]);
void backward_pass_and_update(const double input[N_INPUT], const double hidden_out[N_HIDDEN], const double output[N_OUTPUT], const double target[N_OUTPUT]);

void decode_instruction(double dir_norm, double steps_norm, char *dir_char, int *steps);
void draw_path(char map[GRID_SIZE][GRID_SIZE], int start_x, int start_y, int exit_x, int exit_y, const double output_vec[N_OUTPUT]); 
void test_labyrinth_path(int n_set_size, const double input_set[][N_INPUT], const double target_set[][N_OUTPUT]);
void print_labyrinth_and_path(const double input_image[D_SIZE], const double target_output[N_OUTPUT], const double estimated_output[N_OUTPUT]);


// -----------------------------------------------------------------
// --- DATA GENERATION FUNCTIONS (Labyrinth Specific) ---
// -----------------------------------------------------------------

// Helper function to draw a line for rendering
void draw_line(double image[D_SIZE], int x1, int y1, int x2, int y2, double val) {
    int dx = abs(x2 - x1);
    int dy = abs(y2 - y1);
    int sx = x1 < x2 ? 1 : -1;
    int sy = y1 < y2 ? 1 : -1;
    int err = dx - dy;

    while (1) {
        if (x1 >= 0 && x1 < GRID_SIZE && y1 >= 0 && y1 < GRID_SIZE) {
            image[GRID_SIZE * y1 + x1] = val; // White pixel for path
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
    for (int i = 0; i < D_SIZE; i++) { image[i] = 0.0; } // Black image (walls)
    
    // Path value
    double path_val = 250.0; 

    // 1. Define 7 random intermediate points inside the image
    int target_points[NUM_SEGMENTS + 1][2]; // [x, y]

    // Start point (x>2, x<30, same for y)
    target_points[0][0] = 3 + (rand() % (GRID_SIZE - 6));
    target_points[0][1] = 3 + (rand() % (GRID_SIZE - 6));
    
    // Intermediate points
    for(int i = 1; i < NUM_SEGMENTS; i++) {
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
    
    // Store instructions for the target array
    int instruction_idx = 2; // Start after Start X/Y
    int max_instruction_idx = 2 + (NUM_SEGMENTS * 2); // 16 (index 15 is the last instruction value)
    
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        int next_x = target_points[i+1][0];
        int next_y = target_points[i+1][1];
        
        // Randomly choose to move H or V first
        int move_order[2] = {0, 1}; // 0: Horizontal, 1: Vertical
        if (rand() % 2) { move_order[0] = 1; move_order[1] = 0; }
        
        int start_segment_x = current_x;
        int start_segment_y = current_y;

        for (int m = 0; m < 2; m++) {
            
            // 🛑 CRITICAL FIX: Ensure we do not write past the allocated instruction space
            if (instruction_idx >= max_instruction_idx) {
                goto end_instruction_generation;
            }

            if (move_order[m] == 0) { // Horizontal move
                int dx = next_x - current_x;
                if (dx != 0) {
                    draw_line(image, current_x, current_y, next_x, current_y, path_val);
                    
                    // Store instruction
                    target_data[instruction_idx] = (dx > 0) ? DIR_RIGHT : DIR_LEFT;
                    target_data[instruction_idx+1] = NORMALIZE_STEPS(abs(dx));
                    instruction_idx += 2;

                    current_x = next_x;
                }
            } else { // Vertical move
                int dy = next_y - current_y;
                if (dy != 0) {
                    draw_line(image, current_x, current_y, current_x, next_y, path_val);
                    
                    // Store instruction
                    target_data[instruction_idx] = (dy > 0) ? DIR_DOWN : DIR_UP;
                    target_data[instruction_idx+1] = NORMALIZE_STEPS(abs(dy));
                    instruction_idx += 2;

                    current_y = next_y;
                }
            }
        }
    }

end_instruction_generation: // Label for the goto

    // 3. Set Target Data
    
    // Start X/Y
    target_data[0] = NORMALIZE_COORD(target_points[0][0]);
    target_data[1] = NORMALIZE_COORD(target_points[0][1]);
    
    // Instructions (already set above)
    // Pad the remaining instruction slots with a 'no-op' instruction (UP, 0 steps).
    while (instruction_idx < max_instruction_idx) {
        target_data[instruction_idx] = DIR_UP; 
        target_data[instruction_idx+1] = 0.0; // 0 steps
        instruction_idx += 2;
    }

    // Exit X/Y
    target_data[N_OUTPUT-2] = NORMALIZE_COORD(target_points[NUM_SEGMENTS][0]);
    target_data[N_OUTPUT-1] = NORMALIZE_COORD(target_points[NUM_SEGMENTS][1]);

    // 4. Post-processing: Add walls/noise around the path (optional, current 0.0 serves as wall)
}


// Corrected function signature to avoid array size mismatch errors
void load_data(int n_samples, double set[][D_SIZE], double target_set[][N_OUTPUT]) {
    for (int k = 0; k < n_samples; ++k) {
        generate_labyrinth(set[k], target_set[k]);
    }
}
void load_train_set() {
    printf("Generating TRAINING dataset (%d labyrinths). N_OUTPUT=%d.\n", N_SAMPLES_TRAIN, N_OUTPUT);
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
    for (int i = 0; i < N_INPUT; i++) {
        for (int j = 0; j < N_HIDDEN; j++) {
            w_ih[i][j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * 0.1;
        }
    }
    for (int j = 0; j < N_HIDDEN; j++) {
        b_h[j] = 0.0;
        for (int k = 0; k < N_OUTPUT; k++) {
            w_ho[j][k] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * 0.1;
        }
    }
    for (int k = 0; k < N_OUTPUT; k++) {
        b_o[k] = 0.0;
    }
}

void train_nn(const double input_set[N_SAMPLES_MAX][N_INPUT]) {
    printf("Training on raw %d-dimensional image pixels with %d-output path prediction...\n", N_INPUT, N_OUTPUT);
    for (int epoch = 0; epoch < N_EPOCHS; epoch++) {
        int sample_index = rand() % N_SAMPLES;
        
        double hidden_out[N_HIDDEN];
        double output[N_OUTPUT];
        
        forward_pass(input_set[sample_index], hidden_out, output);
        backward_pass_and_update(input_set[sample_index], hidden_out, output, targets[sample_index]);

        if (N_EPOCHS > 1000 && (epoch % (N_EPOCHS / 10) == 0) && epoch != 0) {
            printf("  Epoch %d/%d completed.\n", epoch, N_EPOCHS);
        }
    }
}

double sigmoid(double x) { 
    return 1.0 / (1.0 + exp(-x)); 
}

void forward_pass(const double input[N_INPUT], double hidden_out[N_HIDDEN], double output[N_OUTPUT]) {
    for (int j = 0; j < N_HIDDEN; j++) {
        double h_net = b_h[j];
        for (int i = 0; i < N_INPUT; i++) {
            // Note: Normalization is done implicitly by the fact that path_val is around 250.0
            // For stability, input should ideally be scaled to 0..1 or -1..1
            h_net += input[i] * w_ih[i][j]; 
        }
        hidden_out[j] = sigmoid(h_net);
    }
    
    for (int k = 0; k < N_OUTPUT; k++) {
        double o_net = b_o[k]; 
        for (int j = 0; j < N_HIDDEN; j++) { 
            o_net += hidden_out[j] * w_ho[j][k]; 
        } 
        output[k] = o_net; 
    }
}

void backward_pass_and_update(const double input[N_INPUT], const double hidden_out[N_HIDDEN], const double output[N_OUTPUT], const double target[N_OUTPUT]) {
    double delta_o[N_OUTPUT];
    double delta_h[N_HIDDEN]; 
    double error_h[N_HIDDEN] = {0.0};
    
    for (int k = 0; k < N_OUTPUT; k++) {
        double error = output[k] - target[k];
        double weight = PATH_WEIGHT; // Use higher weight for path data
        
        // No Classification to check, all outputs are weighted for regression
        
        delta_o[k] = error * weight; 
    }
    
    for (int j = 0; j < N_HIDDEN; j++) { 
        for (int k = 0; k < N_OUTPUT; k++) {
            error_h[j] += delta_o[k] * w_ho[j][k];
        }
        delta_h[j] = error_h[j] * hidden_out[j] * (1.0 - hidden_out[j]);
    }
    
    for (int k = 0; k < N_OUTPUT; k++) { 
        for (int j = 0; j < N_HIDDEN; j++) { 
            w_ho[j][k] -= LEARNING_RATE * delta_o[k] * hidden_out[j]; 
        } 
        b_o[k] -= LEARNING_RATE * delta_o[k];
    } 
    
    for (int i = 0; i < N_INPUT; i++) { 
        for (int j = 0; j < N_HIDDEN; j++) { 
            w_ih[i][j] -= LEARNING_RATE * delta_h[j] * input[i]; 
        } 
    }
    for (int j = 0; j < N_HIDDEN; j++) { 
        b_h[j] -= LEARNING_RATE * delta_h[j]; 
    }
}


// -----------------------------------------------------------------
// --- TESTING AND VISUALIZATION (Labyrinth Specific) ---
// -----------------------------------------------------------------

// Helper to decode a single instruction
void decode_instruction(double dir_norm, double steps_norm, char *dir_char, int *steps) {
    int dir_rounded = (int)round(dir_norm);
    
    if (dir_rounded == (int)DIR_UP) *dir_char = 'U';
    else if (dir_rounded == (int)DIR_DOWN) *dir_char = 'D';
    else if (dir_rounded == (int)DIR_LEFT) *dir_char = 'L';
    else if (dir_rounded == (int)DIR_RIGHT) *dir_char = 'R';
    else *dir_char = '?';
    
    *steps = DENORMALIZE_STEPS(steps_norm);
}

// Helper to draw the path on the ASCII map
void draw_path(char map[GRID_SIZE][GRID_SIZE], int start_x, int start_y, int exit_x, int exit_y, const double output_vec[N_OUTPUT]) {
    int current_x = start_x;
    int current_y = start_y;
    
    // Mark Start and Exit
    if (current_x >= 0 && current_x < GRID_SIZE && current_y >= 0 && current_y < GRID_SIZE) {
        map[current_y][current_x] = '0';
    }
    // Only mark 'E' for exit if it's the final target location
    if (exit_x >= 0 && exit_x < GRID_SIZE && exit_y >= 0 && exit_y < GRID_SIZE) {
        map[exit_y][exit_x] = 'E';
    }
    
    // Draw Segments
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        double dir_norm = output_vec[2 + 2*i];
        double steps_norm = output_vec[3 + 2*i];
        char dir_char;
        int steps;
        decode_instruction(dir_norm, steps_norm, &dir_char, &steps); 
        
        if (steps == 0) continue; // Skip no-op instruction

        for (int s = 1; s <= steps; s++) {
            if (dir_char == 'U') current_y--;
            else if (dir_char == 'D') current_y++;
            else if (dir_char == 'L') current_x--;
            else if (dir_char == 'R') current_x++;
            
            // Draw '*' for path segment
            if (current_x >= 0 && current_x < GRID_SIZE && current_y >= 0 && current_y < GRID_SIZE) {
                if (map[current_y][current_x] == ' ') {
                    map[current_y][current_x] = '*';
                }
            }
        }
        
        // Mark segment end point
        if (i < NUM_SEGMENTS - 1) {
            if (current_x >= 0 && current_x < GRID_SIZE && current_y >= 0 && current_y < GRID_SIZE) {
                // If it's not the exit, mark intermediate point 1-6
                if (map[current_y][current_x] != 'E') {
                    map[current_y][current_x] = '1' + i; 
                }
            }
        }
    }
}


/**
 * Renders the labyrinth image and plots the true and estimated path.
 */
void print_labyrinth_and_path(const double input_image[D_SIZE], const double target_output[N_OUTPUT], const double estimated_output[N_OUTPUT]) {
    
    char true_path_map[GRID_SIZE][GRID_SIZE];
    char est_path_map[GRID_SIZE][GRID_SIZE];
    
    // Initialize maps: # for wall (0.0 input), ' ' for open path (250.0 input)
    for (int y = 0; y < GRID_SIZE; y++) {
        for (int x = 0; x < GRID_SIZE; x++) {
            if (input_image[GRID_SIZE * y + x] < 1.0) { // Black pixel = Wall
                true_path_map[y][x] = '#';
                est_path_map[y][x] = '#';
            } else { // White pixel = Open Path
                true_path_map[y][x] = ' ';
                est_path_map[y][x] = ' ';
            }
        }
    }
    
    // --- Decode Targets and Predictions ---
    
    int true_start_x = DENORMALIZE_COORD(target_output[0]);
    int true_start_y = DENORMALIZE_COORD(target_output[1]);
    int true_exit_x = DENORMALIZE_COORD(target_output[N_OUTPUT-2]);
    int true_exit_y = DENORMALIZE_COORD(target_output[N_OUTPUT-1]);

    int est_start_x = DENORMALIZE_COORD(estimated_output[0]);
    int est_start_y = DENORMALIZE_COORD(estimated_output[1]);
    int est_exit_x = DENORMALIZE_COORD(estimated_output[N_OUTPUT-2]);
    int est_exit_y = DENORMALIZE_COORD(estimated_output[N_OUTPUT-1]);
    
    // --- Draw Paths ---
    draw_path(true_path_map, true_start_x, true_start_y, true_exit_x, true_exit_y, target_output);
    draw_path(est_path_map, est_start_x, est_start_y, est_exit_x, est_exit_y, estimated_output);
    
    // --- Print Side-by-Side ---
    printf("\n--- True Path (Target) | Predicted Path (Output) ---\n");
    printf("TRUE Start: (%d, %d), Exit: (%d, %d)\n", true_start_x, true_start_y, true_exit_x, true_exit_y);
    printf("EST Start:  (%d, %d), Exit: (%d, %d)\n", est_start_x, est_start_y, est_exit_x, est_exit_y);
    printf("--------------------------------------------------------------------------------------------------\n");
    
    for (int y = 0; y < GRID_SIZE; y++) {
        // Print True Path Map
        for (int x = 0; x < GRID_SIZE; x++) {
            printf("%c", true_path_map[y][x]);
        }
        
        // Separator
        printf(" | ");
        
        // Print Estimated Path Map
        for (int x = 0; x < GRID_SIZE; x++) {
            printf("%c", est_path_map[y][x]);
        }
        printf("\n");
    }
    printf("--------------------------------------------------------------------------------------------------\n");
}


// --- Test Function ---

void test_labyrinth_path(int n_set_size, const double input_set[][N_INPUT], const double target_set[][N_OUTPUT]) {
    double hidden_out[N_HIDDEN]; 
    double output[N_OUTPUT];
    
    printf("\n--- STEP 3: LABYRINTH PATH PREDICTION TEST (%d Samples) ---\n", n_set_size);
    
    // Test and visualize a few samples
    for (int i = 0; i < n_set_size; i++) {
        if (i < 5) { // Show 5 visual tests
            forward_pass(input_set[i], hidden_out, output);
            print_labyrinth_and_path(input_set[i], target_set[i], output);
        }
    }
    
    // Print example instructions for the first visualized sample
    printf("--- Example Instructions (Sample 1) ---\n");
    forward_pass(input_set[0], hidden_out, output);
    
    printf("True Instructions:\n");
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        char dir_char;
        int steps;
        decode_instruction(target_set[0][2 + 2*i], target_set[0][3 + 2*i], &dir_char, &steps);
        printf("  Segment %d: (%c, %d) -> Normalized Dir: %.2f, Steps: %.2f\n", i+1, dir_char, steps, target_set[0][2 + 2*i], target_set[0][3 + 2*i]);
    }
    
    printf("\nPredicted Instructions:\n");
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        char dir_char;
        int steps;
        decode_instruction(output[2 + 2*i], output[3 + 2*i], &dir_char, &steps);
        printf("  Segment %d: (%c, %d) -> Raw Output Dir: %.2f, Steps: %.2f\n", i+1, dir_char, steps, output[2 + 2*i], output[3 + 2*i]);
    }
}


// -----------------------------------------------------------------
// --- MAIN PROGRAM (Updated to use new Labyrinth functions) ---
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
