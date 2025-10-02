#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <string.h>

// --- Fix for M_PI undeclared error ---
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- Global Constants ---
#define CANVAS_WIDTH 800.0
#define CANVAS_HEIGHT 600.0
#define BORDER_WIDTH 10.0 // Wall thickness for collision check

// Game/Physics Constants
#define MOVE_STEP_SIZE 15.0 
#define MAX_EPISODE_STEPS 150 // INCREASED from 50 to allow longer episodes
#define NUM_OBSTACLES 5
#define NUM_DIAMONDS 10 

// Pathfinding Constants (Simplified BFS Grid)
#define GRID_CELL_SIZE 20.0 
#define GRID_COLS (int)(CANVAS_WIDTH / GRID_CELL_SIZE)
#define GRID_ROWS (int)(CANVAS_HEIGHT / GRID_CELL_SIZE)
#define MAX_PATH_NODES 2000

// NN & RL Constants
#define NN_INPUT_SIZE 9 
#define NN_HIDDEN_SIZE 16 
#define NN_OUTPUT_SIZE 4 
#define NN_LEARNING_RATE 0.0025 
#define GAMMA 0.95 

// Reward Goals and Values (Full Simulation)
#define REWARD_PER_STEP -1.0 
#define REWARD_CRASH -500.0
#define REWARD_SUCCESS 1000.0
#define REWARD_COLLECT_DIAMOND 100.0
#define REWARD_PROGRESS_SCALE 0.01 

// Action History Buffer
#define ACTION_HISTORY_SIZE 10 
const char* action_names[NN_OUTPUT_SIZE] = {"UP", "DOWN", "LEFT", "RIGHT"};

// --- Unittest Constants (UPDATED) ---
#define UNITTEST_EPISODES 50
#define UNITTEST_SUCCESS_THRESHOLD 0.5 // Threshold for average reward per step
#define UNITTEST_MAX_STEPS 20 // Max steps for the minimal test
#define UNITTEST_PROGRESS_REWARD 1.0 // Strong reward for moving closer (replaces REWARD_PROGRESS_SCALE)
#define UNITTEST_STEP_PENALTY -0.1 // Minimal penalty for time (replaces REWARD_PER_STEP)

// --- Data Structures ---
typedef struct { double x, y; double w, h; } Obstacle;
typedef struct { double x, y; double size; bool collected; } Diamond;
typedef struct { double x, y; double w, h; } TargetArea;
typedef struct { double x, y; double size; bool is_alive; bool has_reached_target; } Robot;
typedef struct { int score; int total_diamonds; Robot robot; Obstacle obstacles[NUM_OBSTACLES]; Diamond diamonds[NUM_DIAMONDS]; TargetArea target; } GameState;

typedef struct { int rows; int cols; double** data; } Matrix;
typedef struct { double input[NN_INPUT_SIZE]; int action_index; double reward; } EpisodeStep;
typedef struct { EpisodeStep steps[MAX_EPISODE_STEPS]; int count; double total_score; } Episode;
typedef struct { Matrix weights_ih; Matrix weights_ho; double* bias_h; double* bias_o; double lr; } NeuralNetwork;

// BFS Node for Pathfinding
typedef struct { int r, c; int parent_r, parent_c; } PathNode;

// --- Global State ---
GameState state;
NeuralNetwork nn;
Episode episode_buffer;
int current_episode = 0;
int action_history[ACTION_HISTORY_SIZE];
int action_history_idx = 0; 
int step_count = 0;
time_t last_print_time = 0; 

// --- C99 Utility Functions ---
void check_nan_and_stop(double value, const char* var_name, const char* context) {
    if (isnan(value) || isinf(value)) { 
        fprintf(stderr, "\n\nCRITICAL NAN/INF ERROR: %s in %s is %.1f. Stopping execution.\n", var_name, context, value); 
        exit(EXIT_FAILURE); 
    }
}
double check_double(double value, const char* var_name, const char* context) {
    check_nan_and_stop(value, var_name, context);
    return value;
}
double sigmoid(double x) { return check_double(1.0 / (1.0 + exp(-x)), "sigmoid_output", "sigmoid"); }
double sigmoid_derivative(double y) { double result = y * (1.0 - y); return check_double(result, "sigmoid_deriv_output", "sigmoid_derivative"); }
void softmax(const double* input, double* output, int size) {
    double max_val = input[0];
    for (int i = 1; i < size; i++) { if (input[i] > max_val) max_val = input[i]; }
    double sum_exp = 0.0;
    for (int i = 0; i < size; i++) {
        output[i] = exp(input[i] - max_val);
        check_nan_and_stop(output[i], "exp_val", "softmax");
        sum_exp += output[i];
    }
    if (sum_exp < 1e-6) {
        fprintf(stderr, "WARNING: Sum of exp in softmax is near zero (%.10f). Using uniform probability.\n", sum_exp);
        for (int i = 0; i < size; i++) output[i] = 1.0 / size;
    } else {
        for (int i = 0; i < size; i++) { output[i] = check_double(output[i] / sum_exp, "softmax_output", "softmax"); }
    }
}
double distance_2d(double x1, double y1, double x2, double y2) {
    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}

// --- Matrix Functions ---
Matrix matrix_create(int rows, int cols) {
    Matrix m; m.rows = rows; m.cols = cols;
    m.data = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        m.data[i] = (double*)malloc(cols * sizeof(double));
        for (int j = 0; j < cols; j++) {
            m.data[i][j] = check_double((((double)rand() / RAND_MAX) * 2.0 - 1.0) * sqrt(2.0 / (rows + cols)), "rand_val", "matrix_create");
        }
    }
    return m;
}

void matrix_free(Matrix m) {
    for (int i = 0; i < m.rows; i++) free(m.data[i]);
    free(m.data);
}

Matrix array_to_matrix(const double* arr, int size) {
    Matrix m = matrix_create(size, 1);
    for (int i = 0; i < size; i++) { m.data[i][0] = arr[i]; }
    return m;
}

Matrix matrix_dot(Matrix A, Matrix B) {
    if (A.cols != B.rows) { fprintf(stderr, "Dot product dimension mismatch.\n"); exit(EXIT_FAILURE); }
    Matrix result = matrix_create(A.rows, B.cols);
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < B.cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < A.cols; k++) { 
                double term = check_double(A.data[i][k], "A[i][k]", "matrix_dot") * check_double(B.data[k][j], "B[k][j]", "matrix_dot");
                sum += term; 
            }
            result.data[i][j] = check_double(sum, "dot_sum", "matrix_dot");
        }
    }
    return result;
}

Matrix matrix_transpose(Matrix m) {
    Matrix result = matrix_create(m.cols, m.rows);
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) { result.data[j][i] = m.data[i][j]; }
    }
    return result;
}

Matrix matrix_add_subtract(Matrix A, Matrix B, bool is_add) {
    if (A.rows != B.rows || A.cols != B.cols) { fprintf(stderr, "Add/Subtract dimension mismatch.\n"); exit(EXIT_FAILURE); }
    Matrix result = matrix_create(A.rows, A.cols);
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            if (is_add) { result.data[i][j] = check_double(A.data[i][j], "A[i][j]", "matrix_add") + check_double(B.data[i][j], "B[i][j]", "matrix_add"); } 
            else { result.data[i][j] = check_double(A.data[i][j], "A[i][j]", "matrix_subtract") - check_double(B.data[i][j], "B[i][j]", "matrix_subtract"); }
        }
    }
    return result;
}

Matrix matrix_multiply_scalar(Matrix A, double scalar) {
    Matrix result = matrix_create(A.rows, A.cols);
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) { result.data[i][j] = check_double(A.data[i][j], "A[i][j]", "matrix_multiply_scalar") * scalar; }
    }
    return result;
}

Matrix matrix_multiply_elem(Matrix A, Matrix B) {
    if (A.rows != B.rows || A.cols != B.cols) { fprintf(stderr, "Multiply (element-wise) dimension mismatch.\n"); exit(EXIT_FAILURE); }
    Matrix result = matrix_create(A.rows, A.cols);
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) { result.data[i][j] = check_double(A.data[i][j], "A[i][j]", "matrix_multiply_elem") * check_double(B.data[i][j], "B[i][j]", "matrix_multiply_elem"); }
    }
    return result;
}

Matrix matrix_map(Matrix m, double (*func)(double)) {
    Matrix result = matrix_create(m.rows, m.cols);
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) { result.data[i][j] = func(m.data[i][j]); }
    }
    return result;
}

// --- Neural Network Core Functions ---

void nn_init(NeuralNetwork* nn) {
    nn->lr = check_double(NN_LEARNING_RATE, "NN_LEARNING_RATE", "nn_init");
    nn->weights_ih = matrix_create(NN_HIDDEN_SIZE, NN_INPUT_SIZE);
    nn->weights_ho = matrix_create(NN_OUTPUT_SIZE, NN_HIDDEN_SIZE);
    nn->bias_h = (double*)malloc(NN_HIDDEN_SIZE * sizeof(double));
    nn->bias_o = (double*)malloc(NN_OUTPUT_SIZE * sizeof(double));

    for (int i = 0; i < NN_HIDDEN_SIZE; i++) nn->bias_h[i] = check_double((((double)rand() / RAND_MAX) * 2.0 - 1.0) * 0.01, "bias_h_val", "nn_init");
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) nn->bias_o[i] = check_double((((double)rand() / RAND_MAX) * 2.0 - 1.0) * 0.01, "bias_o_val", "nn_init");
}

void nn_policy_forward(NeuralNetwork* nn, const double* input_array, double* output_probabilities, double* logit_output) {
    Matrix inputs = array_to_matrix(input_array, NN_INPUT_SIZE);
    Matrix hidden = matrix_dot(nn->weights_ih, inputs);
    for (int i = 0; i < NN_HIDDEN_SIZE; i++) hidden.data[i][0] += nn->bias_h[i];
    Matrix hidden_output = matrix_map(hidden, sigmoid);
    Matrix output_logits_m = matrix_dot(nn->weights_ho, hidden_output);
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
        output_logits_m.data[i][0] += nn->bias_o[i];
        logit_output[i] = output_logits_m.data[i][0];
    }
    softmax(logit_output, output_probabilities, NN_OUTPUT_SIZE);
    matrix_free(inputs); matrix_free(hidden); matrix_free(hidden_output); matrix_free(output_logits_m);
}

void nn_reinforce_train(NeuralNetwork* nn, const double* input_array, int action_index, double discounted_return) {
    check_nan_and_stop(discounted_return, "discounted_return", "nn_reinforce_train");
    
    Matrix inputs = array_to_matrix(input_array, NN_INPUT_SIZE);
    
    // 1. Feedforward 
    Matrix hidden = matrix_dot(nn->weights_ih, inputs);
    for (int i = 0; i < NN_HIDDEN_SIZE; i++) hidden.data[i][0] += nn->bias_h[i];
    Matrix hidden_output = matrix_map(hidden, sigmoid);

    Matrix output_logits_m = matrix_dot(nn->weights_ho, hidden_output);
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) output_logits_m.data[i][0] += nn->bias_o[i];
    
    double logits[NN_OUTPUT_SIZE];
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) logits[i] = output_logits_m.data[i][0];
    double probs[NN_OUTPUT_SIZE];
    softmax(logits, probs, NN_OUTPUT_SIZE);
    
    // Sanity Check: Ensure probabilities are valid
    if (probs[action_index] < 1e-6) {
        fprintf(stderr, "WARNING: Probability of taken action %d is near zero (%.10f). Skipping update.\n", action_index, probs[action_index]);
        // Cleanup and return without update
        matrix_free(inputs); matrix_free(hidden); matrix_free(hidden_output); matrix_free(output_logits_m);
        return;
    }

    // 2. Calculate Output Gradient (dLoss/dLogits)
    Matrix output_gradients = matrix_create(NN_OUTPUT_SIZE, 1);
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
        double target = (i == action_index) ? 1.0 : 0.0;
        double grad_base = check_double(probs[i] - target, "output_grad_base", "nn_reinforce_train");
        output_gradients.data[i][0] = grad_base * discounted_return;
        check_nan_and_stop(output_gradients.data[i][0], "output_grad", "nn_reinforce_train");
    }

    // 3. Update Weights HO and Bias O
    Matrix delta_weights_ho = matrix_multiply_scalar(matrix_dot(output_gradients, matrix_transpose(hidden_output)), -nn->lr);
    Matrix new_weights_ho = matrix_add_subtract(nn->weights_ho, delta_weights_ho, true);
    matrix_free(nn->weights_ho); nn->weights_ho = new_weights_ho;

    for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
        nn->bias_o[i] = check_double(nn->bias_o[i], "bias_o", "nn_train_upd") - check_double(output_gradients.data[i][0], "grad_o", "nn_train_upd") * nn->lr;
    }

    // 4. Calculate Hidden Errors and Update Weights IH and Bias H (Backprop to Hidden Layer)
    Matrix weights_ho_T = matrix_transpose(nn->weights_ho);
    Matrix hidden_errors = matrix_dot(weights_ho_T, output_gradients);

    Matrix hidden_gradients = matrix_map(hidden_output, sigmoid_derivative);
    Matrix hidden_gradients_mul = matrix_multiply_elem(hidden_gradients, hidden_errors);
    
    Matrix delta_weights_ih = matrix_multiply_scalar(matrix_dot(hidden_gradients_mul, matrix_transpose(inputs)), -nn->lr);
    Matrix new_weights_ih = matrix_add_subtract(nn->weights_ih, delta_weights_ih, true);
    matrix_free(nn->weights_ih); nn->weights_ih = new_weights_ih;
    
    for (int i = 0; i < NN_HIDDEN_SIZE; i++) {
        nn->bias_h[i] = check_double(nn->bias_h[i], "bias_h", "nn_train_upd") - check_double(hidden_gradients_mul.data[i][0], "grad_h", "nn_train_upd") * nn->lr;
    }
    
    // 5. Cleanup
    matrix_free(inputs); matrix_free(hidden); matrix_free(hidden_output);
    matrix_free(output_logits_m); matrix_free(output_gradients); matrix_free(weights_ho_T);
    matrix_free(hidden_errors); matrix_free(hidden_gradients); matrix_free(hidden_gradients_mul);
    matrix_free(delta_weights_ho); matrix_free(delta_weights_ih);
}


// --- Game Logic Functions ---

void init_minimal_state() {
    step_count = 0;
    state.score = 0;
    state.total_diamonds = 0;

    // Robot setup (Start position)
    state.robot.x = 50.0;
    state.robot.y = 50.0; 
    state.robot.size = 10.0;
    state.robot.is_alive = true;
    state.robot.has_reached_target = false;
    
    // Reset episode buffer
    episode_buffer.count = 0;
    episode_buffer.total_score = 0;

    // Minimal Target Area: (60, 60) to (70, 70) 
    state.target.x = 60.0;
    state.target.y = 60.0;
    state.target.w = 10.0;
    state.target.h = 10.0;
    
    // Clear all obstacles and diamonds
    for(int i = 0; i < NUM_OBSTACLES; i++) state.obstacles[i].w = 0.0;
    for(int i = 0; i < NUM_DIAMONDS; i++) state.diamonds[i].collected = true;
}

void init_game_state() {
    step_count = 0;
    state.score = 0;
    state.total_diamonds = 0;

    // Robot setup (Start position)
    state.robot.x = 50.0;
    state.robot.y = 50.0; 
    state.robot.size = 10.0;
    state.robot.is_alive = true;
    state.robot.has_reached_target = false;
    
    // Reset episode buffer
    episode_buffer.count = 0;
    episode_buffer.total_score = 0;

    // --- FIXED LEVEL CONFIGURATION ---
    
    // Fixed Obstacles (x, y, w, h) - 5 Rectangular Obstacles
    double obs_configs[NUM_OBSTACLES][4] = {
        {150.0, 150.0, 50.0, 250.0},  // 1. Vertical Left
        {350.0, 150.0, 50.0, 250.0},  // 2. Vertical Center
        {550.0, 50.0, 50.0, 200.0},   // 3. Vertical Top Right
        {550.0, 450.0, 50.0, 100.0},  // 4. Vertical Bottom Right
        {250.0, 400.0, 200.0, 30.0}   // 5. Horizontal Center Bottom
    };
    for (int i = 0; i < NUM_OBSTACLES; i++) {
        state.obstacles[i].x = obs_configs[i][0];
        state.obstacles[i].y = obs_configs[i][1];
        state.obstacles[i].w = obs_configs[i][2];
        state.obstacles[i].h = obs_configs[i][3];
    }

    // Fixed Diamonds (x, y) - 10 legal diamonds
    double diamond_pos[NUM_DIAMONDS][2] = {
        {100.0, 100.0}, // D0
        {300.0, 100.0}, // D1
        {700.0, 100.0}, // D2
        {100.0, 450.0}, // D3
        {250.0, 500.0}, // D4
        {500.0, 500.0}, // D5
        {700.0, 500.0}, // D6
        {50.0, 300.0},  // D7
        {500.0, 350.0}, // D8
        {700.0, 300.0}  // D9
    };
    for (int i = 0; i < NUM_DIAMONDS; i++) {
        state.diamonds[i].x = diamond_pos[i][0];
        state.diamonds[i].y = diamond_pos[i][1];
        state.diamonds[i].size = 8.0;
        state.diamonds[i].collected = false;
    }
    
    // Fixed Target Area (End Goal)
    state.target.x = CANVAS_WIDTH - 100.0;
    state.target.y = CANVAS_HEIGHT - 100.0;
    state.target.w = 50.0;
    state.target.h = 50.0;
}

// Selects an action stochastically based on probabilities
int select_action(const double* probabilities) {
    double r = check_double((double)rand() / RAND_MAX, "rand_action", "select_action");
    double cumulative_prob = 0.0;
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
        cumulative_prob += probabilities[i];
        if (r < cumulative_prob) {
            return i;
        }
    }
    return NN_OUTPUT_SIZE - 1; 
}

void get_state_features(double* input, double* min_dist_to_goal_ptr) {
    Robot* robot = &state.robot;
    
    // --- 1. Robot State (2) ---
    input[0] = robot->x / CANVAS_WIDTH;  
    input[1] = robot->y / CANVAS_HEIGHT; 
    
    // --- 2. Target State (2) ---
    input[2] = (state.target.x + state.target.w/2.0) / CANVAS_WIDTH;
    input[3] = (state.target.y + state.target.h/2.0) / CANVAS_HEIGHT;
    
    // Determine the GOAL location 
    double goal_x, goal_y;
    
    // Find Nearest Diamond or Target
    Diamond* nearest_diamond = NULL;
    double min_diamond_dist = INFINITY;
    for (int i = 0; i < NUM_DIAMONDS; i++) {
        Diamond* d = &state.diamonds[i];
        if (!d->collected) {
            double dist = distance_2d(robot->x, robot->y, d->x, d->y);
            if (dist < min_diamond_dist) { min_diamond_dist = dist; nearest_diamond = d; }
        }
    }
    
    if (nearest_diamond) {
        goal_x = nearest_diamond->x; 
        goal_y = nearest_diamond->y;
    } else {
        goal_x = state.target.x + state.target.w / 2.0;
        goal_y = state.target.y + state.target.h / 2.0;
    }
    
    // --- 3. Goal Distance (2) ---
    double goal_dx = goal_x - robot->x;
    double goal_dy = goal_y - robot->y;
    
    input[4] = check_double(goal_dx / CANVAS_WIDTH, "norm_goal_dx", "get_state_features"); 
    input[5] = check_double(goal_dy / CANVAS_HEIGHT, "norm_goal_dy", "get_state_features"); 
    
    *min_dist_to_goal_ptr = distance_2d(robot->x, robot->y, goal_x, goal_y);

    // --- 4. Nearest Obstacle Distance (2) ---
    Obstacle* nearest_obs = NULL;
    double min_obs_dist = INFINITY;
    for (int i = 0; i < NUM_OBSTACLES; i++) {
        Obstacle* obs = &state.obstacles[i];
        // Skip zero-width obstacles (used in minimal test)
        if (obs->w <= 0.0) continue; 
        
        double obs_center_x = obs->x + obs->w / 2.0;
        double obs_center_y = obs->y + obs->h / 2.0;
        
        double dist = distance_2d(robot->x, robot->y, obs_center_x, obs_center_y);
        if (dist < min_obs_dist) { min_obs_dist = dist; nearest_obs = obs; }
    }
    
    double obs_dx = nearest_obs ? (nearest_obs->x + nearest_obs->w / 2.0) - robot->x : 0.0;
    double obs_dy = nearest_obs ? (nearest_obs->y + nearest_obs->h / 2.0) - robot->y : 0.0;
    
    // Normalise based on min_obs_dist instead of CANVAS dimensions for more responsive feature
    input[6] = check_double(obs_dx / (min_obs_dist > 1.0 ? min_obs_dist : CANVAS_WIDTH), "norm_obs_dx", "get_state_features");
    input[7] = check_double(obs_dy / (min_obs_dist > 1.0 ? min_obs_dist : CANVAS_HEIGHT), "norm_obs_dy", "get_state_features"); 

    // --- 5. Collected Ratio (1) ---
    input[8] = (double)state.total_diamonds / (NUM_DIAMONDS > 0 ? NUM_DIAMONDS : 1.0);

    for (int i = 0; i < NN_INPUT_SIZE; i++) {
        if (input[i] > 1.0) input[i] = 1.0;
        if (input[i] < -1.0) input[i] = -1.0;
        check_nan_and_stop(input[i], "input_feature", "get_state_features");
    }
}

// Function signature updated to include is_unittest
double calculate_reward(double old_min_dist_to_goal, int diamonds_collected_this_step, bool expert_run, bool is_unittest) {
    double reward;
    Robot* robot = &state.robot;
    double progress_scale;
    double step_penalty;
    
    if (is_unittest) {
        // Use high-signal rewards for the minimal test
        reward = 0.0; // Start neutral
        progress_scale = UNITTEST_PROGRESS_REWARD;
        step_penalty = UNITTEST_STEP_PENALTY;
    } else {
        // Use normal, complex rewards for the full simulation/expert
        reward = expert_run ? 5.0 : REWARD_PER_STEP;
        progress_scale = REWARD_PROGRESS_SCALE;
        step_penalty = REWARD_PER_STEP;
    }
    
    if (!robot->is_alive) return REWARD_CRASH;
    
    if (diamonds_collected_this_step > 0) {
        reward += REWARD_COLLECT_DIAMOND * diamonds_collected_this_step;
    }

    // Progress Reward
    double min_dist_to_goal;
    double dummy_input[NN_INPUT_SIZE];
    get_state_features(dummy_input, &min_dist_to_goal); 
    
    double distance_change = old_min_dist_to_goal - min_dist_to_goal;
    
    if (distance_change > 0) {
        // Moving closer
        reward += progress_scale * distance_change;
    } else {
        // Moving further or stuck
        if (!robot->has_reached_target && robot->is_alive) {
            reward += progress_scale * distance_change; 
        }
    }
    
    // Apply standard step penalty (if not already covered by REWARD_PER_STEP in base reward)
    if (is_unittest || expert_run) {
        reward += step_penalty;
    }
    
    // Terminal Rewards (Overwrite progress/step rewards for the final step)
    if (robot->has_reached_target) {
        reward = REWARD_SUCCESS;
        // In full sim, penalize if diamonds are missed
        if (!is_unittest && state.total_diamonds < NUM_DIAMONDS) {
             if (NUM_DIAMONDS > 0) reward -= (NUM_DIAMONDS - state.total_diamonds) * 50.0; 
        }
    } 
    // Note: Crash reward is handled at the start of the function
    
    return check_double(reward, "final_reward", "calc_reward");
}


// Function signature updated to include is_unittest
void update_game(bool is_training_run, bool expert_run, bool is_unittest) {
    Robot* robot = &state.robot;
    int max_steps = is_unittest ? UNITTEST_MAX_STEPS : MAX_EPISODE_STEPS;
    
    if (!robot->is_alive || robot->has_reached_target || step_count >= max_steps) return; 

    double input[NN_INPUT_SIZE];
    double old_min_dist_to_goal;
    get_state_features(input, &old_min_dist_to_goal);

    double logit_output[NN_OUTPUT_SIZE];
    double probabilities[NN_OUTPUT_SIZE];
    nn_policy_forward(&nn, input, probabilities, logit_output);

    int action_index = select_action(probabilities);
    
    // Store old distance for progress reward calculation
    double old_dist_copy = old_min_dist_to_goal; 

    apply_action(robot, action_index);

    int diamonds_collected = check_collision(robot);
    
    // Calculate reward, passing is_unittest
    double final_reward = calculate_reward(old_dist_copy, diamonds_collected, expert_run, is_unittest);
    
    if (is_training_run && episode_buffer.count < max_steps) {
        EpisodeStep step;
        memcpy(step.input, input, NN_INPUT_SIZE * sizeof(double));
        step.action_index = action_index;
        step.reward = check_double(final_reward, "final_reward", "update_game");
        
        episode_buffer.steps[episode_buffer.count] = step;
        episode_buffer.count++;
        episode_buffer.total_score += final_reward;
    }
    
    step_count++;
}

void run_reinforce_training() {
    if (episode_buffer.count == 0) return;

    // 1. Calculate Discounted Returns (G_t)
    double returns[episode_buffer.count];
    double cumulative_return = 0.0;
    
    for (int i = episode_buffer.count - 1; i >= 0; i--) {
        cumulative_return = episode_buffer.steps[i].reward + GAMMA * cumulative_return;
        check_nan_and_stop(cumulative_return, "cumulative_return", "run_reinforce_training");
        returns[i] = cumulative_return;
    }
    
    // 2. Normalize Returns (Baseline)
    double sum_returns = 0.0;
    double sum_sq_returns = 0.0;
    for (int i = 0; i < episode_buffer.count; i++) {
        sum_returns += returns[i];
        sum_sq_returns += returns[i] * returns[i];
    }
    double mean_return = sum_returns / episode_buffer.count;
    double variance = (sum_sq_returns / episode_buffer.count) - (mean_return * mean_return);
    double std_dev = sqrt(variance > 1e-6 ? variance : 1.0); 

    // 3. Train the Network using Backpropagation (REINFORCE)
    for (int i = 0; i < episode_buffer.count; i++) {
        // Subtract baseline and normalize (Advantage function)
        double Gt = (returns[i] - mean_return) / std_dev; 
        
        // Pass -Gt because we are optimizing likelihood, and the reward should be maximized (Gradient Ascent).
        // The implementation uses grad_base * discounted_return, so a large positive reward needs a positive gradient signal.
        // Since we are using an implicit loss based on log-likelihood of action taken, we pass -Gt to mimic minimization of loss.
        // We ensure that the final result in the update function aligns with standard policy gradient (log(pi(a|s)) * Gt).
        nn_reinforce_train(&nn, 
                           episode_buffer.steps[i].input, 
                           episode_buffer.steps[i].action_index, 
                           -Gt); 
    }
}

void print_episode_stats(double train_time_ms, bool is_expert) {
    Robot* robot = &state.robot;
    
    printf("====================================================\n");
    printf("%sEPISODE %d SUMMARY (Steps: %d/%d)\n", is_expert ? "EXPERT " : "", current_episode, step_count, MAX_EPISODE_STEPS);
    printf("----------------------------------------------------\n");
    
    const char* status = "TIMEOUT (Max Steps)";
    if (!robot->is_alive) {
        status = "CRASHED (Wall/Obstacle)";
    } else if (robot->has_reached_target) {
        status = (state.total_diamonds == NUM_DIAMONDS) ? "SUCCESS (ALL COLLECTED)" : "SUCCESS (PARTIAL)";
    }
    
    printf("Termination Status: %s\n", status);
    printf("Total Diamonds Collected: %d/%d\n", state.total_diamonds, NUM_DIAMONDS);
    printf("Final Policy Reward (Score): %.2f\n", episode_buffer.total_score);
    
    printf("Reinforcement Learning Training Time: %.3f ms\n", train_time_ms);
    
    if (!is_expert) {
        printf("Last %d Actions by AI (Newest to Oldest):\n", ACTION_HISTORY_SIZE);
        for (int i = 1; i <= ACTION_HISTORY_SIZE; i++) {
            int index = (action_history_idx - i + ACTION_HISTORY_SIZE) % ACTION_HISTORY_SIZE;
            printf("%s%s", action_names[action_history[index]], (i < ACTION_HISTORY_SIZE) ? ", " : "");
        }
        printf("\n");
    }
    printf("====================================================\n\n");
}


// --- Pathfinding and Expert Training Functions (Minimal changes to signatures) ---

double col_to_x(int c) { return c * GRID_CELL_SIZE + GRID_CELL_SIZE / 2.0; }
double row_to_y(int r) { return r * GRID_CELL_SIZE + GRID_CELL_SIZE / 2.0; }
int x_to_col(double x) { return (int)(x / GRID_CELL_SIZE); }
int y_to_row(double y) { return (int)(y / GRID_CELL_SIZE); }
bool is_point_in_rect(double px, double py, double rx, double ry, double rw, double rh) { return px >= rx && px <= rx + rw && py >= ry && py <= ry + rh; }

bool is_point_legal(double x, double y) {
    double r = state.robot.size; 
    if (x - r < BORDER_WIDTH || x + r > CANVAS_WIDTH - BORDER_WIDTH || y - r < BORDER_WIDTH || y + r > CANVAS_HEIGHT - BORDER_WIDTH) { return false; }
    for (int i = 0; i < NUM_OBSTACLES; i++) {
        Obstacle* obs = &state.obstacles[i];
        if (obs->w <= 0.0) continue; 
        double closest_x = fmax(obs->x, fmin(x, obs->x + obs->w));
        double closest_y = fmax(obs->y, fmin(y, obs->y + obs->h));
        double dx = x - closest_x;
        double dy = y - closest_y;
        if (dx * dx + dy * dy < r * r) { return false; }
    }
    return true;
}

int find_path_segment_bfs(double start_x, double start_y, double end_x, double end_y, PathNode* path_out) {
    int start_r = y_to_row(start_y); int start_c = x_to_col(start_x);
    int end_r = y_to_row(end_y); int end_c = x_to_col(end_x);
    if (!is_point_legal(start_x, start_y) || !is_point_legal(end_x, end_y)) { return 0; }
    PathNode queue[GRID_ROWS * GRID_COLS]; int head = 0, tail = 0;
    int visited[GRID_ROWS][GRID_COLS]; memset(visited, 0, sizeof(visited));
    queue[tail++] = (PathNode){start_r, start_c, -1, -1}; visited[start_r][start_c] = 1;
    int dr[] = {-1, 1, 0, 0}; int dc[] = {0, 0, -1, 1};
    PathNode* final_node = NULL;
    while (head < tail) {
        PathNode current = queue[head++];
        if (current.r == end_r && current.c == end_c) { final_node = &queue[head - 1]; break; }
        for (int i = 0; i < 4; i++) {
            int next_r = current.r + dr[i]; int next_c = current.c + dc[i];
            if (next_r >= 0 && next_r < GRID_ROWS && next_c >= 0 && next_c < GRID_COLS && !visited[next_r][next_c]) {
                double next_x = col_to_x(next_c); double next_y = row_to_y(next_r);
                if (is_point_legal(next_x, next_y)) {
                    visited[next_r][next_c] = 1;
                    queue[tail++] = (PathNode){next_r, next_c, current.r, current.c};
                }
            }
        }
    }
    if (!final_node) return 0;
    int path_len = 0; PathNode* node = final_node;
    while (node->parent_r != -1) { 
        if (path_len < MAX_PATH_NODES) { path_out[path_len++] = *node; }
        int parent_index = -1;
        for (int i = 0; i < tail; i++) {
            if (queue[i].r == node->parent_r && queue[i].c == node->parent_c) { parent_index = i; break; }
        }
        if (parent_index == -1) break; 
        node = &queue[parent_index];
    }
    for (int i = 0; i < path_len / 2; i++) {
        PathNode temp = path_out[i];
        path_out[i] = path_out[path_len - 1 - i];
        path_out[path_len - 1 - i] = temp;
    }
    return path_len;
}

void generate_expert_path_training_data() {
    double waypoints_x[NUM_DIAMONDS + 2]; 
    double waypoints_y[NUM_DIAMONDS + 2];
    int num_waypoints = NUM_DIAMONDS + 2;

    waypoints_x[0] = state.robot.x;
    waypoints_y[0] = state.robot.y;

    double rnd_x, rnd_y;
    do {
        rnd_x = 600.0 + (double)rand() / RAND_MAX * 150.0; 
        rnd_y = 50.0 + (double)rand() / RAND_MAX * 500.0;
    } while (!is_point_legal(rnd_x, rnd_y));

    waypoints_x[1] = rnd_x;
    waypoints_y[1] = rnd_y;

    for (int i = 0; i < NUM_DIAMONDS; i++) {
        waypoints_x[i + 2] = state.diamonds[i].x;
        waypoints_y[i + 2] = state.diamonds[i].y;
    }

    waypoints_x[num_waypoints - 1] = state.target.x + state.target.w / 2.0;
    waypoints_y[num_waypoints - 1] = state.target.y + state.target.h / 2.0;

    printf("\n--- EXPERT PATH WAYPOINTS ---\n");
    for (int i = 0; i < num_waypoints; i++) {
        printf("WP %d: (%.1f, %.1f)\n", i, waypoints_x[i], waypoints_y[i]);
    }

    PathNode full_path[MAX_PATH_NODES];
    int full_path_len = 0;

    for (int i = 0; i < num_waypoints - 1; i++) {
        PathNode segment[MAX_PATH_NODES];
        int segment_len = find_path_segment_bfs(
            waypoints_x[i], waypoints_y[i], 
            waypoints_x[i+1], waypoints_y[i+1], 
            segment);

        if (segment_len == 0) {
            fprintf(stderr, "CRITICAL ERROR: Could not find legal path from WP %d (%.1f, %.1f) to WP %d (%.1f, %.1f). Skipping segment.\n", 
                i, waypoints_x[i], waypoints_y[i], i+1, waypoints_x[i+1], waypoints_y[i+1]);
            continue; 
        }

        for (int j = 0; j < segment_len; j++) {
            if (full_path_len < MAX_PATH_NODES - 1) {
                full_path[full_path_len++] = segment[j];
            }
        }
    }
    
    printf("--- GENERATED EXPERT PATH (%d Grid Steps) ---\n", full_path_len);

    init_game_state(); 
    Robot* robot = &state.robot;
    episode_buffer.count = 0;
    episode_buffer.total_score = 0;
    
    for (int i = 0; i < full_path_len && episode_buffer.count < MAX_EPISODE_STEPS; i++) {
        PathNode current_node = full_path[i];
        double target_x = col_to_x(current_node.c);
        double target_y = row_to_y(current_node.r);
        
        double min_dist_after_move = INFINITY;
        int best_action = -1;
        
        for (int a = 0; a < NN_OUTPUT_SIZE; a++) {
            double test_x = robot->x;
            double test_y = robot->y;
            
            switch (a) {
                case 0: test_y -= MOVE_STEP_SIZE; break; 
                case 1: test_y += MOVE_STEP_SIZE; break; 
                case 2: test_x -= MOVE_STEP_SIZE; break; 
                case 3: test_x += MOVE_STEP_SIZE; break; 
            }
            
            double dist = distance_2d(test_x, test_y, target_x, target_y);
            if (dist < min_dist_after_move) {
                min_dist_after_move = dist;
                best_action = a;
            }
        }
        
        if (best_action != -1) {
            double old_min_dist_to_goal;
            double input[NN_INPUT_SIZE];
            get_state_features(input, &old_min_dist_to_goal);

            apply_action(robot, best_action);

            int diamonds_collected = check_collision(robot);
            
            // Pass false for is_unittest
            double reward = calculate_reward(old_min_dist_to_goal, diamonds_collected, true, false); 

            EpisodeStep step;
            memcpy(step.input, input, NN_INPUT_SIZE * sizeof(double));
            step.action_index = best_action;
            step.reward = reward;
            
            episode_buffer.steps[episode_buffer.count] = step;
            episode_buffer.count++;
            episode_buffer.total_score += reward;
            
            if (robot->is_alive == false || robot->has_reached_target == true) break;
        }
    }
    
    if (robot->has_reached_target) {
        episode_buffer.total_score += (state.total_diamonds == NUM_DIAMONDS) ? REWARD_SUCCESS : REWARD_SUCCESS / 2.0;
    } else if (!robot->is_alive) {
        episode_buffer.total_score += REWARD_CRASH;
    }

    printf("\nEXPERT PATH COORDINATES:\n");
    printf("(%.1f, %.1f)", waypoints_x[0], waypoints_y[0]);
    for (int i = 0; i < full_path_len; i++) {
        printf(" -> (%.1f, %.1f)", col_to_x(full_path[i].c), row_to_y(full_path[i].r));
    }
    printf("\n");
}


void pre_train_with_shortest_path() {
    clock_t start = clock();
    generate_expert_path_training_data(); 
    
    if (episode_buffer.count > 0) {
        run_reinforce_training();
    }
    
    clock_t end = clock();
    double train_time_ms = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;
    
    current_episode = 0; 
    
    print_episode_stats(train_time_ms, true);
}


// --- New ASCII Rendering Function ---

void print_ascii_map() {
    printf("\n\n--- FIXED LEVEL ASCII MAP ---\n");
    printf("Legend: #=Wall, O=Obstacle, D=Diamond, S=Start (Robot), T=Target (Goal), .=Free\n");
    
    for (int r = 0; r < GRID_ROWS; r++) {
        for (int c = 0; c < GRID_COLS; c++) {
            
            char symbol = '.';
            double cell_x = col_to_x(c);
            double cell_y = row_to_y(r);

            if (r == 0 || r == GRID_ROWS - 1 || c == 0 || c == GRID_COLS - 1) {
                symbol = '#';
            }
            
            for (int i = 0; i < NUM_OBSTACLES; i++) {
                Obstacle* obs = &state.obstacles[i];
                if (is_point_in_rect(cell_x, cell_y, obs->x, obs->y, obs->w, obs->h)) {
                    symbol = 'O';
                    break;
                }
            }
            
            if (symbol == '.') {
                for (int i = 0; i < NUM_DIAMONDS; i++) {
                    Diamond* d = &state.diamonds[i];
                    if (!d->collected && x_to_col(d->x) == c && y_to_row(d->y) == r) {
                        symbol = 'D';
                        break;
                    }
                }
            }

            if (x_to_col(state.robot.x) == c && y_to_row(state.robot.y) == r) {
                symbol = 'S';
            }
            
            TargetArea* target = &state.target;
            if (is_point_in_rect(cell_x, cell_y, target->x, target->y, target->w, target->h)) {
                if (symbol != 'S') {
                    symbol = 'T';
                }
            }

            printf("%c", symbol);
        }
        printf("\n");
    }
    printf("-----------------------------------------\n\n");
}

// --- UNITTEST Function (Updated logic and constant usage) ---

bool run_rl_unittest() {
    printf("\n\n*** RUNNING RL UNITTEST (Minimal Environment) ***\n");
    printf("Goal: Learn to move from (50, 50) to Target (60, 60).\n");
    
    double total_final_score = 0.0;
    int success_count = 0;
    
    for (int i = 1; i <= UNITTEST_EPISODES; i++) {
        init_minimal_state();
        current_episode = i;

        // Play episode, using UNITTEST_MAX_STEPS
        while (state.robot.is_alive && !state.robot.has_reached_target && step_count < UNITTEST_MAX_STEPS) {
            update_game(true, false, true); // Pass true for is_unittest
        }

        // Train 
        run_reinforce_training(); 
        
        if (state.robot.has_reached_target) success_count++;
        total_final_score += episode_buffer.total_score;
    }
    
    double avg_score_per_episode = total_final_score / UNITTEST_EPISODES;
    double avg_score_per_step = (total_final_score / UNITTEST_EPISODES) / UNITTEST_MAX_STEPS;

    printf("\n--- UNITTEST RESULTS ---\n");
    printf("Total Episodes: %d\n", UNITTEST_EPISODES);
    printf("Success Count (Reached Target): %d\n", success_count);
    printf("Average Score Per Episode: %.2f\n", avg_score_per_episode);
    printf("Average Score Per Step: %.4f (Threshold: > %.4f)\n", avg_score_per_step, UNITTEST_SUCCESS_THRESHOLD);
    printf("--------------------------\n");
    
    if (avg_score_per_step > UNITTEST_SUCCESS_THRESHOLD) {
        printf("*** UNITTEST PASSED! Continuing to main simulation. ***\n\n");
        return true;
    } else {
        fprintf(stderr, "*** UNITTEST FAILED. AI cannot solve minimal task. Halting simulation. ***\n");
        return false;
    }
}


// --- Main Simulation Loop (Updated to correctly pass is_unittest) ---

int main() {
    srand((unsigned int)time(NULL)); 
    nn_init(&nn);
    
    // 1. Run RL Unittest
    if (!run_rl_unittest()) {
        matrix_free(nn.weights_ih); matrix_free(nn.weights_ho);
        free(nn.bias_h); free(nn.bias_o);
        return 1; // Exit on failure
    }

    // 2. Initialize Full Game State
    init_game_state();
    
    // 3. Print the ASCII Level Map
    print_ascii_map();

    for(int i = 0; i < ACTION_HISTORY_SIZE; i++) {
        action_history[i] = 3; 
    }

    printf("--- RL 2D Robot Collector Simulation (EXPERT PRE-TRAIN) ---\n");
    printf("Input Size: %d, Hidden Size: %d, Output Size: %d\n", NN_INPUT_SIZE, NN_HIDDEN_SIZE, NN_OUTPUT_SIZE);
    printf("Training will run for 3 minutes (180 seconds). Stats printed every 10s.\n");

    // --- EXPERT PRE-TRAINING PHASE ---
    pre_train_with_shortest_path();
    printf("Expert pre-training complete. Starting RL exploration.\n\n");
    
    // --- RL EXPLORATION PHASE ---
    time_t start_time = time(NULL);
    const int TIME_LIMIT_SECONDS = 180; 
    last_print_time = start_time;
    
    while (time(NULL) - start_time < TIME_LIMIT_SECONDS) {
        
        current_episode++;
        init_game_state();

        // Play episode
        while (state.robot.is_alive && !state.robot.has_reached_target && step_count < MAX_EPISODE_STEPS) {
            update_game(true, false, false); // Pass false for is_unittest
        }

        // Train and Time it
        clock_t train_start = clock();
        run_reinforce_training(); 
        clock_t train_end = clock();
        
        double train_time_ms = (double)(train_end - train_start) * 1000.0 / CLOCKS_PER_SEC;

        // Print stats every 10 seconds
        time_t current_time = time(NULL);
        if (current_time - last_print_time >= 10) {
            print_episode_stats(train_time_ms, false);
            last_print_time = current_time;
        }
    }
    
    printf("\n--- TIME LIMIT REACHED. TRAINING HALTED. Total Episodes: %d ---\n", current_episode);

    // --- Cleanup ---
    matrix_free(nn.weights_ih);
    matrix_free(nn.weights_ho);
    free(nn.bias_h);
    free(nn.bias_o);
    printf("Simulation finished and memory cleaned up.\n");
    return 0;
}
