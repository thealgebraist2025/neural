#define _XOPEN_SOURCE 

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h> 
#include <float.h>
#include <string.h> 

// --- Configuration ---
#define GRID_SIZE 64       
#define D_SIZE (GRID_SIZE * GRID_SIZE) 

// **Labyrinth Configuration**
#define NUM_LABYRINTHS 2
#define NUM_LONGEST_PATHS 8
#define PATH_FEATURE_SIZE (4 + 1) 
#define N_LABYRINTH_PIXELS D_SIZE
#define N_INPUT (D_SIZE + (NUM_LONGEST_PATHS * PATH_FEATURE_SIZE)) 
#define NUM_SEGMENTS 7
#define N_DIRECTION_CLASSES 4 
#define N_OUTPUT (2 + (NUM_SEGMENTS * N_DIRECTION_CLASSES) + (NUM_SEGMENTS * 1) + 2) 

// **Network & Training Parameters**
#define N_HIDDEN 128       
#define N_TEST_CASES_PER_LABYRINTH 10 
#define INITIAL_LEARNING_RATE 0.00001 
#define N_EPOCHS_MAX 1000000 
#define COORD_WEIGHT 1.0                 
#define CLASSIFICATION_WEIGHT 1.0 
#define MAX_STEPS 16.0 
#define MAX_TRAINING_SECONDS 60
#define WARMUP_EPOCHS 10
#define REPORT_SECONDS 1
#define SOLVED_ERROR_THRESHOLD 0.1 
#define GRADIENT_CLIP_NORM 1.0 

// Direction Encoding
#define DIR_UP_IDX 0
#define DIR_DOWN_IDX 1
#define DIR_LEFT_IDX 2
#define DIR_RIGHT_IDX 3

// --- Dynamic Globals ---
double current_learning_rate = INITIAL_LEARNING_RATE; 
double last_avg_loss = DBL_MAX;                       
double single_labyrinths[NUM_LABYRINTHS][D_SIZE];
int fixed_exit_coords[NUM_LABYRINTHS][2];
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
void generate_labyrinth(int index); 
void extract_longest_paths(const double labyrinth[D_SIZE], double feature_output[NUM_LONGEST_PATHS * PATH_FEATURE_SIZE]);
void load_train_case(double input[N_INPUT], double target[N_OUTPUT]);
double calculate_loss(const double output[N_OUTPUT], const double target[N_OUTPUT]);
double tanh_activation(double x);
double tanh_derivative(double x);
double sigmoid(double x);
double sigmoid_derivative(double x);
void softmax(double vector[N_DIRECTION_CLASSES]);
void initialize_nn(); 
void update_learning_rate(double current_avg_loss); 
double clip_gradient(double grad, double max_norm); 
void forward_pass(const double input[N_INPUT], double hidden_net[N_HIDDEN], double hidden_out[N_HIDDEN], double output_net[N_OUTPUT], double output[N_OUTPUT]);
int estimate_epochs_per_second(double input[N_INPUT], double target[N_OUTPUT]);
void train_nn();
void test_nn_and_summarize();
int is_path_legal(const double labyrinth[D_SIZE], int start_x, int start_y, const double output_vec[N_OUTPUT]);
void print_labyrinth_and_path(const double input_vec[N_INPUT], const double target_output[N_OUTPUT], const double estimated_output[N_OUTPUT]);
void decode_instruction_output(const double output_vec[N_OUTPUT], int segment, char *dir_char, int *steps);
void decode_instruction_target(const double target_vec[N_OUTPUT], int segment, char *dir_char, int *steps);

// --- NN Core Activation Functions ---
double tanh_activation(double x) { return tanh(x); }
double tanh_derivative(double tanh_out) { return 1.0 - (tanh_out * tanh_out); }
double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
double sigmoid_derivative(double sigmoid_out) { return sigmoid_out * (1.0 - sigmoid_out); }
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

// --- Data Generation Functions ---
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

void generate_path_and_target(const double labyrinth[D_SIZE], int start_x, int start_y, int exit_x, int exit_y, double target_data[N_OUTPUT]) {
    for (int i = 0; i < N_OUTPUT; i++) { target_data[i] = 0.0; } 
    typedef struct { int x, y, prev_idx; } BFSNode;
    BFSNode queue[D_SIZE];
    int visited[GRID_SIZE][GRID_SIZE];
    memset(visited, 0, sizeof(visited));
    int head = 0, tail = 0;

    queue[tail++] = (BFSNode){start_x, start_y, -1};
    visited[start_y][start_x] = 1;

    int path_end_idx = -1;
    int dx[] = {0, 0, -1, 1}; 
    int dy[] = {-1, 1, 0, 0};

    while (head < tail) {
        BFSNode current = queue[head++];
        if (current.x == exit_x && current.y == exit_y) { path_end_idx = head - 1; break; }
        for (int i = 0; i < 4; i++) {
            int next_x = current.x + dx[i];
            int next_y = current.y + dy[i];
            if (next_x >= 0 && next_x < GRID_SIZE && next_y >= 0 && next_y < GRID_SIZE && 
                !visited[next_y][next_x] && 
                labyrinth[GRID_SIZE * next_y + next_x] > 0.5) 
            {
                visited[next_y][next_x] = 1;
                queue[tail] = (BFSNode){next_x, next_y, head - 1};
                tail++;
            }
        }
    }

    if (path_end_idx == -1) { 
        target_data[0] = NORMALIZE_COORD(start_x);
        target_data[1] = NORMALIZE_COORD(start_y);
        target_data[N_OUTPUT-2] = NORMALIZE_COORD(exit_x);
        target_data[N_OUTPUT-1] = NORMALIZE_COORD(exit_y);
        return;
    }

    int path_indices[D_SIZE];
    int path_length = 0;
    int current_idx = path_end_idx;
    while (current_idx != -1) {
        path_indices[path_length++] = current_idx;
        current_idx = queue[current_idx].prev_idx;
    }

    int current_segment = 0;
    int current_path_idx = path_length - 1;
    int last_dir = -1;
    int steps = 0;
    
    while (current_path_idx > 0 && current_segment < NUM_SEGMENTS) {
        int next_path_idx = current_path_idx - 1;
        int prev_x = queue[path_indices[current_path_idx]].x;
        int prev_y = queue[path_indices[current_path_idx]].y;
        int next_x = queue[path_indices[next_path_idx]].x;
        int next_y = queue[path_indices[next_path_idx]].y;

        int dir = -1;
        if (next_y < prev_y) dir = DIR_UP_IDX;
        else if (next_y > prev_y) dir = DIR_DOWN_IDX;
        else if (next_x < prev_x) dir = DIR_LEFT_IDX;
        else if (next_x > prev_x) dir = DIR_RIGHT_IDX;
        
        if (last_dir == -1) {
            last_dir = dir;
            steps = 1;
        } else if (dir == last_dir) {
            steps++;
            if (steps >= MAX_STEPS) { goto finalize_segment; }
        } else {
            goto finalize_segment;
        }

        current_path_idx = next_path_idx;
        continue;
        
    finalize_segment:
        target_data[GET_DIR_OUTPUT_START_IDX(current_segment) + last_dir] = 1.0; 
        target_data[GET_STEPS_OUTPUT_IDX(current_segment)] = NORMALIZE_STEPS(steps);
        current_segment++;
        last_dir = dir;
        steps = 1;
        current_path_idx = next_path_idx;
    }
    
    if (steps > 0 && current_segment < NUM_SEGMENTS) {
        target_data[GET_DIR_OUTPUT_START_IDX(current_segment) + last_dir] = 1.0; 
        target_data[GET_STEPS_OUTPUT_IDX(current_segment)] = NORMALIZE_STEPS(steps);
    }

    target_data[0] = NORMALIZE_COORD(start_x);
    target_data[1] = NORMALIZE_COORD(start_y);
    target_data[N_OUTPUT-2] = NORMALIZE_COORD(exit_x);
    target_data[N_OUTPUT-1] = NORMALIZE_COORD(exit_y);
}

void generate_labyrinth(int index) {
    printf("[DEBUG] Generating Labyrinth %d...\n", index);
    srand((unsigned int)time(NULL) + index * 100); 
    double *labyrinth = single_labyrinths[index];
    int *exit_coord = fixed_exit_coords[index];

    for (int i = 0; i < D_SIZE; i++) { labyrinth[i] = 0.0; } 
    int num_connection_points = GRID_SIZE / 4; 
    int points[num_connection_points + 1][2]; 

    for(int i = 0; i < num_connection_points; i++) {
        points[i][0] = 3 + (rand() % (GRID_SIZE - 6));
        points[i][1] = 3 + (rand() % (GRID_SIZE - 6));
    }
    
    int side = rand() % 4; 
    if (side == 0) { points[num_connection_points][0] = 1 + (rand() % (GRID_SIZE - 2)); points[num_connection_points][1] = 0; }
    else if (side == 1) { points[num_connection_points][0] = 1 + (rand() % (GRID_SIZE - 2)); points[num_connection_points][1] = GRID_SIZE - 1; }
    else if (side == 2) { points[num_connection_points][0] = 0; points[num_connection_points][1] = 1 + (rand() % (GRID_SIZE - 2)); }
    else { points[num_connection_points][0] = GRID_SIZE - 1; points[num_connection_points][1] = 1 + (rand() % (GRID_SIZE - 2)); }

    exit_coord[0] = points[num_connection_points][0];
    exit_coord[1] = points[num_connection_points][1];

    int current_x = points[0][0];
    int current_y = points[0][1];
    
    for (int i = 0; i < num_connection_points; i++) {
        int next_x = points[i+1][0];
        int next_y = points[i+1][1];
        if (rand() % 2) { 
            draw_line(labyrinth, current_x, current_y, next_x, current_y, 1.0);
            current_x = next_x;
            draw_line(labyrinth, current_x, current_y, current_x, next_y, 1.0);
            current_y = next_y;
        } else {
            draw_line(labyrinth, current_x, current_y, current_x, next_y, 1.0);
            current_y = next_y;
            draw_line(labyrinth, current_x, current_y, next_x, current_y, 1.0);
            current_x = next_x;
        }
    }
    printf("[DEBUG] Labyrinth %d generated (Exit: %d, %d).\n", index, exit_coord[0], exit_coord[1]);
}

void extract_longest_paths(const double labyrinth[D_SIZE], double feature_output[NUM_LONGEST_PATHS * PATH_FEATURE_SIZE]) {
    typedef struct { int length, direction; } PathFeature;
    PathFeature all_paths[GRID_SIZE * GRID_SIZE]; 
    int path_count = 0;

    for (int y = 0; y < GRID_SIZE; y++) {
        int current_length = 0;
        for (int x = 0; x < GRID_SIZE; x++) {
            if (labyrinth[y * GRID_SIZE + x] > 0.5) { current_length++; } 
            else { 
                if (current_length >= 2) { 
                    all_paths[path_count++] = (PathFeature){current_length, DIR_RIGHT_IDX};
                }
                current_length = 0;
            }
        }
        if (current_length >= 2) {
            all_paths[path_count++] = (PathFeature){current_length, DIR_RIGHT_IDX};
        }
    }

    for (int x = 0; x < GRID_SIZE; x++) {
        int current_length = 0;
        for (int y = 0; y < GRID_SIZE; y++) {
            if (labyrinth[y * GRID_SIZE + x] > 0.5) { current_length++; } 
            else { 
                if (current_length >= 2) { 
                    all_paths[path_count++] = (PathFeature){current_length, DIR_DOWN_IDX};
                }
                current_length = 0;
            }
        }
        if (current_length >= 2) {
            all_paths[path_count++] = (PathFeature){current_length, DIR_DOWN_IDX};
        }
    }

    for (int i = 1; i < path_count; i++) {
        PathFeature key = all_paths[i];
        int j = i - 1;
        while (j >= 0 && all_paths[j].length < key.length) {
            all_paths[j + 1] = all_paths[j];
            j = j - 1;
        }
        all_paths[j + 1] = key;
    }

    for (int i = 0; i < NUM_LONGEST_PATHS; i++) {
        int start_idx = i * PATH_FEATURE_SIZE;
        if (i < path_count) {
            int dir_idx = all_paths[i].direction;
            for(int k = 0; k < 4; k++) {
                feature_output[start_idx + k] = (k == dir_idx) ? 1.0 : 0.0;
            }
            feature_output[start_idx + 4] = NORMALIZE_STEPS(all_paths[i].length);
        } else {
            for(int k = 0; k < PATH_FEATURE_SIZE; k++) {
                feature_output[start_idx + k] = 0.0;
            }
            feature_output[start_idx + DIR_UP_IDX] = 1.0; 
        }
    }
}

void load_train_case(double input[N_INPUT], double target[N_OUTPUT]) {
    int lab_idx = rand() % NUM_LABYRINTHS;
    const double *current_labyrinth = single_labyrinths[lab_idx];
    const int *current_exit_coord = fixed_exit_coords[lab_idx];
    memcpy(input, current_labyrinth, D_SIZE * sizeof(double));
    int start_x, start_y;
    int exit_x = current_exit_coord[0];
    int exit_y = current_exit_coord[1];

    int attempts = 0;
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
    double* feature_start = input + N_LABYRINTH_PIXELS; 
    extract_longest_paths(current_labyrinth, feature_start);
}

// --- NN Core Functions ---
void initialize_nn() {
    double fan_in_h = (double)N_INPUT;
    double fan_in_o = (double)N_HIDDEN;
    double limit_h = sqrt(1.0 / fan_in_h); 
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
    for (int k = 0; k < N_OUTPUT; k++) {
        b_o[k] = 0.0;
    }
}

double clip_gradient(double grad, double max_norm) { 
    if (grad > max_norm) return max_norm;
    if (grad < -max_norm) return -max_norm;
    return grad;
}

void forward_pass(const double input[N_INPUT], double hidden_net[N_HIDDEN], double hidden_out[N_HIDDEN], double output_net[N_OUTPUT], double output[N_OUTPUT]) {
    for (int j = 0; j < N_HIDDEN; j++) {
        double h_net = b_h[j];
        for (int i = 0; i < N_INPUT; i++) {
            h_net += input[i] * w_fh[i][j]; 
        }
        hidden_net[j] = h_net;
        hidden_out[j] = tanh_activation(h_net); 
    }
    
    for (int k = 0; k < N_OUTPUT; k++) {
        double o_net = b_o[k]; 
        for (int j = 0; j < N_HIDDEN; j++) { 
            o_net += hidden_out[j] * w_ho[j][k]; 
        } 
        output_net[k] = o_net;
    }
    
    for (int k = 0; k < N_OUTPUT; k++) {
        if (k < 2 || k >= (2 + NUM_SEGMENTS * N_DIRECTION_CLASSES)) {
             output[k] = sigmoid(output_net[k]); 
        } else {
             output[k] = output_net[k];
        }
    }
    
    for (int s = 0; s < NUM_SEGMENTS; s++) {
        int dir_start_idx = GET_DIR_OUTPUT_START_IDX(s);
        softmax(&output[dir_start_idx]);
    }
}

void update_learning_rate(double current_avg_loss) { 
    if (isnan(current_avg_loss) || current_avg_loss > last_avg_loss * 1.5 || current_avg_loss > 100000.0) {
        current_learning_rate *= 0.5;
        if (current_learning_rate < 1e-10) current_learning_rate = 1e-10; 
        printf("\n!!! Learning Rate DECAYED aggressively to %.6e (Loss explosion/major increase).\n", current_learning_rate);
    } else if (current_avg_loss > last_avg_loss * 1.001) {
        current_learning_rate *= 0.9;
    } 
    if (!isnan(current_avg_loss) && current_avg_loss < DBL_MAX) {
        if (current_avg_loss < last_avg_loss) { 
            last_avg_loss = current_avg_loss;
        }
    }
}

int estimate_epochs_per_second(double input[N_INPUT], double target[N_OUTPUT]) {
    printf("[DEBUG] --- Timing Warmup: Running %d epochs to estimate performance. ---\n", WARMUP_EPOCHS);
    
    double hidden_net[N_HIDDEN], hidden_out[N_HIDDEN], output_net[N_OUTPUT], output[N_OUTPUT];
    
    // Store copies of all weights and biases to restore after warmup
    double w_fh_copy[N_INPUT][N_HIDDEN];
    double b_h_copy[N_HIDDEN];
    double w_ho_copy[N_HIDDEN][N_OUTPUT];
    double b_o_copy[N_OUTPUT];
    memcpy(w_fh_copy, w_fh, sizeof(w_fh));
    memcpy(b_h_copy, b_h, sizeof(b_h));
    memcpy(w_ho_copy, w_ho, sizeof(w_ho));
    memcpy(b_o_copy, b_o, sizeof(b_o));
    
    time_t start_time = time(NULL);
    
    for (int i = 0; i < WARMUP_EPOCHS; i++) {
        load_train_case(input, target);
        forward_pass(input, hidden_net, hidden_out, output_net, output);
        
        double error = calculate_loss(output, target);
        
        // Full backpropagation step, matching train_nn()
        double delta_o[N_OUTPUT] = {0.0};
        double delta_h[N_HIDDEN] = {0.0};
        double error_h[N_HIDDEN] = {0.0};
        
        // Compute output layer deltas
        for (int k = 0; k < N_OUTPUT; k++) {
            if (k >= 2 && k < (2 + NUM_SEGMENTS * N_DIRECTION_CLASSES)) {
                delta_o[k] = (output[k] - target[k]);
            } else {
                double error = output[k] - target[k];
                double sig_deriv = sigmoid_derivative(output[k]);
                delta_o[k] = error * COORD_WEIGHT * sig_deriv;
            }
            delta_o[k] = clip_gradient(delta_o[k], GRADIENT_CLIP_NORM);
        }
        
        // Compute hidden layer deltas
        for (int j = 0; j < N_HIDDEN; j++) {
            for (int k = 0; k < N_OUTPUT; k++) {
                error_h[j] += delta_o[k] * w_ho[j][k];
            }
            double tanh_deriv = tanh_derivative(hidden_out[j]);
            delta_h[j] = error_h[j] * tanh_deriv;
        }
        
        // Update output weights and biases
        for (int k = 0; k < N_OUTPUT; k++) {
            for (int j = 0; j < N_HIDDEN; j++) {
                w_ho[j][k] -= current_learning_rate * delta_o[k] * hidden_out[j];
            }
            b_o[k] -= current_learning_rate * delta_o[k];
        }
        
        // Update input-to-hidden weights and biases
        for (int i = 0; i < N_INPUT; i++) {
            for (int j = 0; j < N_HIDDEN; j++) {
                double gradient = delta_h[j] * input[i];
                w_fh[i][j] -= current_learning_rate * clip_gradient(gradient, GRADIENT_CLIP_NORM);
            }
        }
        for (int j = 0; j < N_HIDDEN; j++) {
            b_h[j] -= current_learning_rate * delta_h[j];
        }
    }
    
    time_t end_time = time(NULL);
    double elapsed_time = difftime(end_time, start_time);
    
    // Restore original weights and biases
    memcpy(w_fh, w_fh_copy, sizeof(w_fh));
    memcpy(b_h, b_h_copy, sizeof(b_h));
    memcpy(w_ho, w_ho_copy, sizeof(w_ho));
    memcpy(b_o, b_o_copy, sizeof(b_o));

    if (elapsed_time < 1.0) {
        printf("[DEBUG] Warmup too fast (%.2f s). Defaulting to 1000 epochs/second guess.\n", elapsed_time);
        return 1000;
    }

    int epochs_per_second = (int)round((double)WARMUP_EPOCHS / elapsed_time);
    if (epochs_per_second < 1) {
        printf("[DEBUG] Estimated epochs/second too low (%d). Setting to 1 to avoid zero division.\n", epochs_per_second);
        epochs_per_second = 1;
    }
    
    printf("[DEBUG] --- Warmup finished: %.2f seconds for %d epochs. Estimated %d epochs per second. ---\n", 
           elapsed_time, WARMUP_EPOCHS, epochs_per_second);
           
    return epochs_per_second;
}

void train_nn() {
    printf("\n--- STEP 2: STARTING TRAINING PHASE ---\n");
    double input[N_INPUT];
    double target[N_OUTPUT];
    
    int epochs_per_sec = estimate_epochs_per_second(input, target);
    int max_epochs_to_run = (int)((double)epochs_per_sec * MAX_TRAINING_SECONDS * 0.98); 
    if (max_epochs_to_run > N_EPOCHS_MAX) max_epochs_to_run = N_EPOCHS_MAX;
    
    printf("Training Vanilla NN. Time Limit: %d seconds. Epoch Limit (Estimated): %d.\n", 
           MAX_TRAINING_SECONDS, max_epochs_to_run);
    
    time_t start_time = time(NULL);
    time_t next_report_time = start_time + REPORT_SECONDS;

    double hidden_net[N_HIDDEN]; 
    double hidden_out[N_HIDDEN];
    double output_net[N_OUTPUT]; 
    double output[N_OUTPUT];
    
    double cumulative_loss_report = 0.0;
    int samples_processed_in_report = 0;
    int total_samples_processed = 0;

    for (int epoch = 0; epoch < max_epochs_to_run; epoch++) {
        if (difftime(time(NULL), start_time) >= MAX_TRAINING_SECONDS) {
            printf("\n--- Training stopped: Time limit (%d s) reached at epoch %d. ---\n", 
                   MAX_TRAINING_SECONDS, epoch + 1);
            break;
        }

        load_train_case(input, target);
        forward_pass(input, hidden_net, hidden_out, output_net, output);
        cumulative_loss_report += calculate_loss(output, target);
        samples_processed_in_report++;
        total_samples_processed++;
        
        double delta_o[N_OUTPUT] = {0.0};
        double delta_h[N_HIDDEN]; 
        double error_h[N_HIDDEN] = {0.0};
        
        for (int k = 0; k < N_OUTPUT; k++) {
            if (k >= 2 && k < (2 + NUM_SEGMENTS * N_DIRECTION_CLASSES)) {
                delta_o[k] = (output[k] - target[k]); 
            } else { 
                double error = output[k] - target[k];
                double sig_deriv = sigmoid_derivative(output[k]); 
                delta_o[k] = error * COORD_WEIGHT * sig_deriv; 
            }
            delta_o[k] = clip_gradient(delta_o[k], GRADIENT_CLIP_NORM); 
        }
        
        for (int j = 0; j < N_HIDDEN; j++) { 
            for (int k = 0; k < N_OUTPUT; k++) {
                error_h[j] += delta_o[k] * w_ho[j][k];
            }
            double tanh_deriv = tanh_derivative(hidden_out[j]);
            delta_h[j] = error_h[j] * tanh_deriv;
        }
        
        for (int k = 0; k < N_OUTPUT; k++) { 
            for (int j = 0; j < N_HIDDEN; j++) { 
                w_ho[j][k] -= current_learning_rate * delta_o[k] * hidden_out[j]; 
            } 
            b_o[k] -= current_learning_rate * delta_o[k];
        } 
        
        for (int i = 0; i < N_INPUT; i++) { 
            for (int j = 0; j < N_HIDDEN; j++) { 
                double gradient = delta_h[j] * input[i];
                w_fh[i][j] -= current_learning_rate * clip_gradient(gradient, GRADIENT_CLIP_NORM); 
            } 
        }
        for (int j = 0; j < N_HIDDEN; j++) { 
            b_h[j] -= current_learning_rate * delta_h[j]; 
        }
        
        if (time(NULL) >= next_report_time) {
            double current_avg_loss = cumulative_loss_report / samples_processed_in_report;
            update_learning_rate(current_avg_loss); 
            
            double time_elapsed = difftime(time(NULL), start_time);
            int epochs_remaining = max_epochs_to_run - (epoch + 1);
            double estimated_end_time = time_elapsed + (double)epochs_remaining / epochs_per_sec;

            printf("  Time: %3.0f s | Epoch: %6d / %6d | Avg Loss: %7.4f | LR: %.2e | Est. End: %4.0f s\n", 
                   time_elapsed, epoch + 1, max_epochs_to_run, current_avg_loss, current_learning_rate, estimated_end_time);
            
            cumulative_loss_report = 0.0;
            samples_processed_in_report = 0;
            next_report_time = time(NULL) + REPORT_SECONDS;
        }
    }
    printf("--- TRAINING PHASE COMPLETE ---\n");
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

// --- Testing and Visualization Functions ---
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
    
    if (*steps > 0 && max_val < 0.25) { 
         *steps = 0;
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
        if (map[exit_y][exit_x] != '0') {
             map[exit_y][exit_x] = 'E';
        }
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
                if (map[current_y][current_x] == '*') { 
                    map[current_y][current_x] = '1' + i; 
                }
            }
        }
    }
}

int is_path_legal(const double labyrinth[D_SIZE], int start_x, int start_y, const double output_vec[N_OUTPUT]) {
    int current_x = DENORMALIZE_COORD(output_vec[0]);
    int current_y = DENORMALIZE_COORD(output_vec[1]);
    int exit_x = DENORMALIZE_COORD(output_vec[N_OUTPUT-2]);
    int exit_y = DENORMALIZE_COORD(output_vec[N_OUTPUT-1]);

    if (current_x < 0 || current_x >= GRID_SIZE || current_y < 0 || current_y >= GRID_SIZE) return 0;
    if (labyrinth[GRID_SIZE * current_y + current_x] < 0.5) return 0;
    
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
            
            if (next_x < 0 || next_x >= GRID_SIZE || next_y < 0 || next_y >= GRID_SIZE) return 0;
            if (labyrinth[GRID_SIZE * next_y + next_x] < 0.5) return 0;
            
            current_x = next_x;
            current_y = next_y;
        }
    }

    return (abs(current_x - exit_x) + abs(current_y - exit_y) < 2); 
}

void print_labyrinth_and_path(const double input_vec[N_INPUT], const double target_output[N_OUTPUT], const double estimated_output[N_OUTPUT]) {
    int VIS_SIZE = (GRID_SIZE > 16) ? 16 : GRID_SIZE;
    const double* input_image = input_vec;
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
    
    int center_x = (true_start_x + true_exit_x) / 2;
    int center_y = (true_start_y + true_exit_y) / 2;
    int start_vis_x = CLAMP(center_x - VIS_SIZE/2, 0, GRID_SIZE - VIS_SIZE);
    int start_vis_y = CLAMP(center_y - VIS_SIZE/2, 0, GRID_SIZE - VIS_SIZE);

    printf("\n--- True Path (Target) | Predicted Path (Output) ---\n");
    printf("TRUE Start: (%d, %d), Exit: (%d, %d)\n", true_start_x, true_start_y, true_exit_x, true_exit_y);
    printf("EST Start:  (%d, %d), Exit: (%d, %d)\n", est_start_x, est_start_y, est_exit_x, est_exit_y);
    
    if (GRID_SIZE > VIS_SIZE) {
        printf("Visualizing %dx%d area around path center (%d, %d)\n", VIS_SIZE, VIS_SIZE, center_x, center_y);
    }
    printf("--------------------------------------------------\n");
    
    for (int y = 0; y < VIS_SIZE; y++) {
        for (int x = 0; x < VIS_SIZE; x++) {
            printf("%c", true_path_map[start_vis_y + y][start_vis_x + x]);
        }
        printf(" | ");
        for (int x = 0; x < VIS_SIZE; x++) {
            printf("%c", est_path_map[start_vis_y + y][start_vis_x + x]);
        }
        printf("\n");
    }
    printf("--------------------------------------------------\n");
}

void test_nn_and_summarize() {
    int total_test_runs = N_TEST_CASES_PER_LABYRINTH * NUM_LABYRINTHS * 5; 
    printf("\n--- STEP 3: STARTING TEST AND SUMMARY PHASE ---\n");
    printf("Testing on %d fixed labyrinths with %d random start points (Total %d tests).\n", 
           NUM_LABYRINTHS, total_test_runs / NUM_LABYRINTHS, total_test_runs);
    
    double cumulative_test_loss = 0.0;
    int solved_count = 0;
    
    double input[N_INPUT];
    double target[N_OUTPUT];
    double hidden_net[N_HIDDEN];
    double hidden_out[N_HIDDEN]; 
    double output_net[N_OUTPUT];
    double output[N_OUTPUT];

    srand(12345); 

    for (int test_run = 0; test_run < total_test_runs; test_run++) {
        load_train_case(input, target); 
        forward_pass(input, hidden_net, hidden_out, output_net, output);
        cumulative_test_loss += calculate_loss(output, target);
        
        int true_start_x = DENORMALIZE_COORD(target[0]);
        int true_start_y = DENORMALIZE_COORD(target[1]);
        int legal_path = is_path_legal(input, true_start_x, true_start_y, output);
        
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

    printf("\n--- VISUALIZATION: 10 Random Test Cases (from Labyrinth 0 or 1) ---\n");
    srand(time(NULL)); 

    for (int i = 0; i < 10; i++) {
        load_train_case(input, target); 
        forward_pass(input, hidden_net, hidden_out, output_net, output);
        printf("Test Case #%d (Labyrinth ID determined in load_train_case):\n", i + 1);
        print_labyrinth_and_path(input, target, output);
    }
    printf("--- TEST PHASE COMPLETE ---\n");
}

// --- Main Program ---
int main(int argc, char **argv) {
    printf("--- STEP 0: PROGRAM START ---\n");
    srand(time(NULL));
    printf("--- STEP 1: INITIALIZATION AND LABYRINTH GENERATION ---\n");
    initialize_nn(); 
    for (int i = 0; i < NUM_LABYRINTHS; i++) {
        generate_labyrinth(i); 
    }
    train_nn();
    test_nn_and_summarize();
    printf("--- STEP 4: PROGRAM END ---\n");
    return 0;
}