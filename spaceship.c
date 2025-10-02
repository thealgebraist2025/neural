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
#define BORDER_WIDTH 10.0 

// Game/Physics Constants
#define MOVE_STEP_SIZE 15.0 
#define MAX_EPISODE_STEPS 150 
#define NUM_OBSTACLES 5
#define NUM_DIAMONDS 10 

// Pathfinding Constants (Simplified BFS Grid)
#define GRID_CELL_SIZE 20.0 
#define GRID_COLS (int)(CANVAS_WIDTH / GRID_CELL_SIZE)
#define GRID_ROWS (int)(CANVAS_HEIGHT / GRID_CELL_SIZE)
#define MAX_PATH_NODES 2000

// NN & RL Constants
#define NN_INPUT_SIZE 9 
#define NN_HIDDEN_SIZE 128 // Increased network capacity
#define NN_OUTPUT_SIZE 4 
#define NN_LEARNING_RATE 0.01 
#define GAMMA 0.99 // Increased discount factor

// Exploration Constants (Epsilon-Greedy)
#define EPSILON_START 1.0 
#define EPSILON_END 0.01 
#define EPSILON_DECAY_EPISODES 5000 

// Reward Goals and Values
#define REWARD_PER_STEP -1.0 
#define REWARD_CRASH -500.0
#define REWARD_SUCCESS 1000.0
#define REWARD_COLLECT_DIAMOND 100.0
#define REWARD_PROGRESS_SCALE 0.01 

// Action History Buffer
#define ACTION_HISTORY_SIZE 10 

// Q-Learning Constants
#define Q_LEARNING_EPISODES 1000
#define Q_LEARNING_ALPHA 0.1 // Learning rate for Q-table
#define Q_X_BINS 4
#define Q_Y_BINS 4
#define Q_DIAMOND_STATUS_BINS 2 // Closest diamond status (0: uncollected, 1: collected)
#define Q_DIAMOND_COUNT_BINS 4 // Total diamonds collected (0-2, 3-5, 6-8, 9-10)
#define Q_STATE_SIZE (Q_X_BINS * Q_Y_BINS * Q_DIAMOND_STATUS_BINS * Q_DIAMOND_COUNT_BINS) 

// Use enum for type safety and clarity
typedef enum {
    ACTION_UP = 0,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_INVALID 
} Action;

const char* action_names[NN_OUTPUT_SIZE] = {"UP", "DOWN", "LEFT", "RIGHT"};

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

// --- Global State ---
GameState state;
NeuralNetwork nn;
Episode episode_buffer;
int current_episode = 0;
int action_history[ACTION_HISTORY_SIZE];
int action_history_idx = 0; 
int step_count = 0;

// Q-Learning Global State
double Q_table[Q_STATE_SIZE][NN_OUTPUT_SIZE]; 
typedef enum { MODE_QL_TEST, MODE_NN_REINFORCE } RL_Mode;
RL_Mode current_rl_mode = MODE_QL_TEST; 

// --- Utility Functions (Matrix functions, etc. - unchanged) ---

void check_nan_and_stop(double value, const char* var_name, const char* context) {
    if (isnan(value)) { fprintf(stderr, "\n\nCRITICAL NAN ERROR: %s in %s is NaN. Stopping execution.\n", var_name, context); exit(EXIT_FAILURE); }
}
double check_double(double value, const char* var_name, const char* context) {
    check_nan_and_stop(value, var_name, context);
    return value;
}

// ReLU Activation Function
double relu(double x) {
    return check_double(x > 0 ? x : 0.0, "relu_output", "relu");
}

// ReLU Derivative
double relu_derivative(double y) {
    return check_double(y > 0 ? 1.0 : 0.0, "relu_deriv_output", "relu_derivative");
}

void softmax(const double* input, double* output, int size) {
    double max_val = input[0];
    for (int i = 1; i < size; i++) { if (input[i] > max_val) max_val = input[i]; }
    double sum_exp = 0.0;
    const double epsilon_safety = 1e-12;
    
    for (int i = 0; i < size; i++) {
        output[i] = exp(input[i] - max_val);
        sum_exp += output[i];
    }
    
    if (sum_exp < epsilon_safety) sum_exp = epsilon_safety; 

    for (int i = 0; i < size; i++) { output[i] = check_double(output[i] / sum_exp, "softmax_output", "softmax"); }
}
double distance_2d(double x1, double y1, double x2, double y2) {
    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}

// --- Matrix Functions (Same as previous, omitted for brevity) ---

Matrix matrix_create(int rows, int cols) {
    Matrix m; m.rows = rows; m.cols = cols;
    m.data = (double**)calloc(rows, sizeof(double*));
    if (m.data == NULL) { fprintf(stderr, "Allocation failed for matrix rows.\n"); exit(EXIT_FAILURE); }
    for (int i = 0; i < rows; i++) {
        m.data[i] = (double*)calloc(cols, sizeof(double));
        if (m.data[i] == NULL) { fprintf(stderr, "Allocation failed for matrix column %d.\n", i); exit(EXIT_FAILURE); }
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
            for (int k = 0; k < A.cols; k++) { sum += check_double(A.data[i][k], "A[i][k]", "matrix_dot") * check_double(B.data[k][j], "B[k][j]", "matrix_dot"); }
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

// --- Neural Network Core Functions (Kept for later phase) ---

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
    Matrix hidden_output = matrix_map(hidden, relu); 
    Matrix output_logits_m = matrix_dot(nn->weights_ho, hidden_output);
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
        output_logits_m.data[i][0] += nn->bias_o[i];
        logit_output[i] = output_logits_m.data[i][0];
    }
    softmax(logit_output, output_probabilities, NN_OUTPUT_SIZE);
    matrix_free(inputs); matrix_free(hidden); matrix_free(hidden_output); matrix_free(output_logits_m);
}

void nn_reinforce_train(NeuralNetwork* nn, const double* input_array, int action_index, double discounted_return) {
    Matrix inputs = array_to_matrix(input_array, NN_INPUT_SIZE);
    
    // 1. Feedforward 
    Matrix hidden = matrix_dot(nn->weights_ih, inputs);
    for (int i = 0; i < NN_HIDDEN_SIZE; i++) hidden.data[i][0] += nn->bias_h[i];
    Matrix hidden_output = matrix_map(hidden, relu); 

    Matrix output_logits_m = matrix_dot(nn->weights_ho, hidden_output);
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) output_logits_m.data[i][0] += nn->bias_o[i];
    
    double logits[NN_OUTPUT_SIZE];
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) logits[i] = output_logits_m.data[i][0];
    double probs[NN_OUTPUT_SIZE];
    softmax(logits, probs, NN_OUTPUT_SIZE);

    // 2. Calculate Output Gradient (dLoss/dLogits)
    Matrix output_gradients = matrix_create(NN_OUTPUT_SIZE, 1);
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
        double target = (i == action_index) ? 1.0 : 0.0;
        output_gradients.data[i][0] = check_double(probs[i] - target, "output_grad_base", "nn_reinforce_train") * discounted_return;
    }

    // 3. Update Weights HO and Bias O
    Matrix delta_weights_ho = matrix_multiply_scalar(matrix_dot(output_gradients, matrix_transpose(hidden_output)), -nn->lr);
    Matrix new_weights_ho = matrix_add_subtract(nn->weights_ho, delta_weights_ho, true);
    matrix_free(nn->weights_ho); nn->weights_ho = new_weights_ho;

    for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
        nn->bias_o[i] = check_double(nn->bias_o[i], "bias_o", "nn_train_upd") - check_double(output_gradients.data[i][0], "grad_o", "nn_train_upd") * nn->lr;
    }

    // 4. Calculate Hidden Errors and Update Weights IH and Bias H
    Matrix weights_ho_T = matrix_transpose(nn->weights_ho);
    Matrix hidden_errors = matrix_dot(weights_ho_T, output_gradients);

    Matrix hidden_gradients = matrix_map(hidden_output, relu_derivative); 
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

// --- Q-Learning State Discretization ---

int get_q_state_index() {
    int x_bin, y_bin, d_status_bin, d_count_bin;

    // 1. Robot X Bin (4 Bins: 0-200, 200-400, 400-600, 600-800)
    x_bin = (int)(state.robot.x / (CANVAS_WIDTH / Q_X_BINS));
    if (x_bin >= Q_X_BINS) x_bin = Q_X_BINS - 1;
    if (x_bin < 0) x_bin = 0;

    // 2. Robot Y Bin (4 Bins: 0-150, 150-300, 300-450, 450-600)
    y_bin = (int)(state.robot.y / (CANVAS_HEIGHT / Q_Y_BINS));
    if (y_bin >= Q_Y_BINS) y_bin = Q_Y_BINS - 1;
    if (y_bin < 0) y_bin = 0;

    // 3. Nearest Diamond Status Bin (2 Bins: 0: Uncollected, 1: Collected)
    d_status_bin = 1; // Assume collected if no uncollected diamonds are found
    double min_diamond_dist = INFINITY;
    for (int i = 0; i < NUM_DIAMONDS; i++) {
        Diamond* d = &state.diamonds[i];
        if (!d->collected) {
            double dist = distance_2d(state.robot.x, state.robot.y, d->x, d->y);
            if (dist < min_diamond_dist) { 
                min_diamond_dist = dist; 
                d_status_bin = 0; // Found uncollected diamond
            }
        }
    }
    // Simplification: use the state of the NEAREST diamond at the beginning of the episode as a proxy 
    // for whether the agent should be seeking diamonds (0) or the exit (1).
    // Better approximation: use the total count to set a hard transition.
    d_status_bin = (state.total_diamonds < NUM_DIAMONDS) ? 0 : 1;
    

    // 4. Diamonds Collected Count Bin (4 Bins: 0-2, 3-5, 6-8, 9-10)
    if (state.total_diamonds <= 2) d_count_bin = 0;
    else if (state.total_diamonds <= 5) d_count_bin = 1;
    else if (state.total_diamonds <= 8) d_count_bin = 2;
    else d_count_bin = 3;

    // Combine bins into a single index
    int state_index = x_bin;
    state_index = state_index * Q_Y_BINS + y_bin;
    state_index = state_index * Q_DIAMOND_STATUS_BINS + d_status_bin;
    state_index = state_index * Q_DIAMOND_COUNT_BINS + d_count_bin;

    return state_index;
}

// --- Game Logic Functions (is_potential_move_legal, etc. - unchanged) ---

void init_game_state() {
    step_count = 0;
    state.score = 0;
    state.total_diamonds = 0;

    state.robot.x = 50.0;
    state.robot.y = 50.0; 
    state.robot.size = 10.0;
    state.robot.is_alive = true;
    state.robot.has_reached_target = false;
    
    episode_buffer.count = 0;
    episode_buffer.total_score = 0;

    double obs_configs[NUM_OBSTACLES][4] = {
        {150.0, 150.0, 50.0, 250.0}, {350.0, 150.0, 50.0, 250.0},
        {550.0, 50.0, 50.0, 200.0}, {550.0, 450.0, 50.0, 100.0},
        {250.0, 400.0, 200.0, 30.0}
    };
    for (int i = 0; i < NUM_OBSTACLES; i++) {
        state.obstacles[i].x = obs_configs[i][0];
        state.obstacles[i].y = obs_configs[i][1];
        state.obstacles[i].w = obs_configs[i][2];
        state.obstacles[i].h = obs_configs[i][3];
    }

    double diamond_pos[NUM_DIAMONDS][2] = {
        {100.0, 100.0}, {300.0, 100.0}, {700.0, 100.0}, 
        {100.0, 450.0}, {250.0, 500.0}, {500.0, 500.0}, 
        {700.0, 500.0}, {50.0, 300.0},  {500.0, 350.0}, 
        {700.0, 300.0}  
    };
    for (int i = 0; i < NUM_DIAMONDS; i++) {
        state.diamonds[i].x = diamond_pos[i][0];
        state.diamonds[i].y = diamond_pos[i][1];
        state.diamonds[i].size = 8.0;
        state.diamonds[i].collected = false;
    }
    
    state.target.x = CANVAS_WIDTH - 100.0;
    state.target.y = CANVAS_HEIGHT - 100.0;
    state.target.w = 50.0;
    state.target.h = 50.0;
}

bool is_potential_move_legal(double current_x, double current_y, int action_index) {
    double next_x = current_x;
    double next_y = current_y;
    double r = state.robot.size;

    switch ((Action)action_index) {
        case ACTION_UP:    next_y -= MOVE_STEP_SIZE; break; 
        case ACTION_DOWN:  next_y += MOVE_STEP_SIZE; break;
        case ACTION_LEFT:  next_x -= MOVE_STEP_SIZE; break; 
        case ACTION_RIGHT: next_x += MOVE_STEP_SIZE; break; 
        default: return false;
    }

    if (next_x - r < BORDER_WIDTH || next_x + r > CANVAS_WIDTH - BORDER_WIDTH ||
        next_y - r < BORDER_WIDTH || next_y + r > CANVAS_HEIGHT - BORDER_WIDTH) {
        return false; 
    }
    
    for (int i = 0; i < NUM_OBSTACLES; i++) {
        Obstacle* obs = &state.obstacles[i];
        double closest_x = fmax(obs->x, fmin(next_x, obs->x + obs->w));
        double closest_y = fmax(obs->y, fmin(next_y, obs->y + obs->h));
        
        double dx = next_x - closest_x;
        double dy = next_y - closest_y;
        
        if (dx * dx + dy * dy < r * r) {
            return false; 
        }
    }
    
    return true;
}

double calculate_reward(double old_min_dist_to_goal, int diamonds_collected_this_step, bool expert_run) {
    double reward = REWARD_PER_STEP; 
    Robot* robot = &state.robot;
    
    if (!robot->is_alive) return REWARD_CRASH;
    
    if (diamonds_collected_this_step > 0) {
        reward += REWARD_COLLECT_DIAMOND * diamonds_collected_this_step;
    }

    if (robot->has_reached_target) {
        reward += REWARD_SUCCESS;
        if (state.total_diamonds < NUM_DIAMONDS) {
            reward -= (NUM_DIAMONDS - state.total_diamonds) * 50.0; 
        }
    }
    
    double min_dist_to_goal;
    double dummy_input[NN_INPUT_SIZE];
    get_state_features(dummy_input, &min_dist_to_goal); 
    
    double distance_change = old_min_dist_to_goal - min_dist_to_goal;
    
    // Progress Reward (Dense reward shaping)
    reward += REWARD_PROGRESS_SCALE * distance_change;
    
    return check_double(reward, "final_reward", "calc_reward");
}

void apply_action(Robot* robot, int action_index) {
    double dx = 0.0;
    double dy = 0.0;
    
    switch ((Action)action_index) {
        case ACTION_UP:    dy = -MOVE_STEP_SIZE; break; 
        case ACTION_DOWN:  dy = MOVE_STEP_SIZE;  break;
        case ACTION_LEFT:  dx = -MOVE_STEP_SIZE; break; 
        case ACTION_RIGHT: dx = MOVE_STEP_SIZE;  break; 
        case ACTION_INVALID: return;
    }

    robot->x += dx;
    robot->y += dy;
    
    action_history[action_history_idx] = action_index;
    action_history_idx = (action_history_idx + 1) % ACTION_HISTORY_SIZE;
}


int check_collision(Robot* robot) {
    double r = robot->size; 
    int diamonds_collected_this_step = 0;
    
    // 1. Wall Collision (Instant crash)
    if (robot->x - r < BORDER_WIDTH || robot->x + r > CANVAS_WIDTH - BORDER_WIDTH ||
        robot->y - r < BORDER_WIDTH || robot->y + r > CANVAS_HEIGHT - BORDER_WIDTH) {
        robot->is_alive = false;
        return 0; 
    }
    
    // 2. Obstacle Collision (Instant crash)
    for (int i = 0; i < NUM_OBSTACLES; i++) {
        Obstacle* obs = &state.obstacles[i];
        double closest_x = fmax(obs->x, fmin(robot->x, obs->x + obs->w));
        double closest_y = fmax(obs->y, fmin(robot->y, obs->y + obs->h));
        
        double dx = robot->x - closest_x;
        double dy = robot->y - closest_y;
        
        if (dx * dx + dy * dy < r * r) {
            robot->is_alive = false;
            return 0; 
        }
    }
    
    // 3. Diamond Collection
    for (int i = 0; i < NUM_DIAMONDS; i++) {
        Diamond* d = &state.diamonds[i];
        if (d->collected) continue;
        
        if (distance_2d(robot->x, robot->y, d->x, d->y) < robot->size + d->size) {
            d->collected = true;
            state.total_diamonds++;
            diamonds_collected_this_step++;
            state.score += (int)REWARD_COLLECT_DIAMOND;
        }
    }

    // 4. Target Area Check (Goal)
    TargetArea* target = &state.target;
    if (robot->x + r > target->x && robot->x - r < target->x + target->w &&
        robot->y + r > target->y && robot->y - r < target->y + target->h) {
        robot->has_reached_target = true;
    }
    
    return diamonds_collected_this_step;
}

void get_state_features(double* input, double* min_dist_to_goal_ptr) {
    Robot* robot = &state.robot;
    
    // The same 9 features as before
    input[0] = robot->x / CANVAS_WIDTH;  
    input[1] = robot->y / CANVAS_HEIGHT; 
    input[2] = (state.target.x + state.target.w/2.0) / CANVAS_WIDTH;
    input[3] = (state.target.y + state.target.h/2.0) / CANVAS_HEIGHT;
    
    double goal_x, goal_y;
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
    
    double goal_dx = goal_x - robot->x;
    double goal_dy = goal_y - robot->y;
    
    input[4] = check_double(goal_dx / CANVAS_WIDTH, "norm_goal_dx", "get_state_features"); 
    input[5] = check_double(goal_dy / CANVAS_HEIGHT, "norm_goal_dy", "get_state_features"); 
    
    *min_dist_to_goal_ptr = distance_2d(robot->x, robot->y, goal_x, goal_y);

    Obstacle* nearest_obs = NULL;
    double min_obs_dist = INFINITY;
    for (int i = 0; i < NUM_OBSTACLES; i++) {
        Obstacle* obs = &state.obstacles[i];
        double obs_center_x = obs->x + obs->w / 2.0;
        double obs_center_y = obs->y + obs->h / 2.0;
        double dist = distance_2d(robot->x, robot->y, obs_center_x, obs_center_y);
        if (dist < min_obs_dist) { min_obs_dist = dist; nearest_obs = obs; }
    }
    
    double obs_dx = nearest_obs ? (nearest_obs->x + nearest_obs->w / 2.0) - robot->x : 0.0;
    double obs_dy = nearest_obs ? (nearest_obs->y + nearest_obs->h / 2.0) - robot->y : 0.0;
    
    input[6] = check_double(obs_dx / CANVAS_WIDTH, "norm_obs_dx", "get_state_features");
    input[7] = check_double(obs_dy / CANVAS_HEIGHT, "norm_obs_dy", "get_state_features"); 

    input[8] = (double)state.total_diamonds / NUM_DIAMONDS;

    for (int i = 0; i < NN_INPUT_SIZE; i++) {
        if (input[i] > 1.0) input[i] = 1.0;
        if (input[i] < -1.0) input[i] = -1.0;
    }
}

// --- Q-Learning Action Selection and Update ---

int select_q_action(int state_index, double current_epsilon) {
    int legal_actions[NN_OUTPUT_SIZE];
    int num_legal = 0;
    
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
        if (is_potential_move_legal(state.robot.x, state.robot.y, i)) {
            legal_actions[num_legal++] = i;
        }
    }

    if (num_legal == 0) return ACTION_INVALID;

    double r = check_double((double)rand() / RAND_MAX, "rand_action_ql", "select_q_action");

    if (r < current_epsilon) {
        // EXPLORE: Choose a random LEGAL action
        int random_index = rand() % num_legal;
        return legal_actions[random_index];
    } else {
        // EXPLOIT: Choose the action with the maximum Q-value among legal actions
        int best_action = -1;
        double max_q_value = -INFINITY;

        for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
            bool is_legal = false;
            for(int j = 0; j < num_legal; j++) {
                if (legal_actions[j] == i) { is_legal = true; break; }
            }

            if (is_legal) {
                if (Q_table[state_index][i] > max_q_value) {
                    max_q_value = Q_table[state_index][i];
                    best_action = i;
                }
            }
        }
        
        // Safety fallback 
        if (best_action == -1) return legal_actions[0]; 

        return best_action;
    }
}

void q_learning_update(int state_old, int action, double reward, int state_new) {
    double max_q_next = -INFINITY;
    
    // Find max Q(s', a') (next state's best action)
    for (int a = 0; a < NN_OUTPUT_SIZE; a++) {
        if (Q_table[state_new][a] > max_q_next) {
            max_q_next = Q_table[state_new][a];
        }
    }
    
    // If the episode ended (crashed or reached target), max_q_next is 0 (no future reward)
    if (!state.robot.is_alive || state.robot.has_reached_target) {
        max_q_next = 0.0;
    }

    double old_q_value = Q_table[state_old][action];
    
    // Bellman Equation: Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]
    Q_table[state_old][action] = old_q_value + Q_LEARNING_ALPHA * (reward + GAMMA * max_q_next - old_q_value);
}

// --- Main Update Loop ---

void update_game_q_learning() {
    Robot* robot = &state.robot;
    
    if (!robot->is_alive || robot->has_reached_target || step_count >= MAX_EPISODE_STEPS) return; 

    // Q-Learning specific logic
    int state_old = get_q_state_index();
    
    // Epsilon decay for Q-Learning phase
    double current_epsilon = EPSILON_END;
    if (current_episode < Q_LEARNING_EPISODES) {
        double decay_rate = (EPSILON_START - EPSILON_END) / Q_LEARNING_EPISODES;
        current_epsilon = EPSILON_START - decay_rate * current_episode;
    }

    double old_min_dist_to_goal;
    double dummy_input[NN_INPUT_SIZE];
    get_state_features(dummy_input, &old_min_dist_to_goal);

    int action_index = select_q_action(state_old, current_epsilon);
    
    if (action_index == ACTION_INVALID) {
        robot->is_alive = false; // Treat as a terminal crash state
    } else {
        apply_action(robot, action_index);
    }
    
    int diamonds_collected = check_collision(robot);
    double reward = calculate_reward(old_min_dist_to_goal, diamonds_collected, false);

    int state_new = get_q_state_index();
    
    q_learning_update(state_old, action_index, reward, state_new);
    
    episode_buffer.total_score += reward;
    step_count++;
}

void update_game_nn_reinforce() {
    Robot* robot = &state.robot;
    
    if (!robot->is_alive || robot->has_reached_target || step_count >= MAX_EPISODE_STEPS) return; 

    double input[NN_INPUT_SIZE];
    double old_min_dist_to_goal;
    get_state_features(input, &old_min_dist_to_goal);

    double logit_output[NN_OUTPUT_SIZE];
    double probabilities[NN_OUTPUT_SIZE];
    nn_policy_forward(&nn, input, probabilities, logit_output);

    // Epsilon decay logic
    double current_epsilon = EPSILON_END;
    if (current_episode < EPSILON_DECAY_EPISODES) {
        double decay_rate = (EPSILON_START - EPSILON_END) / EPSILON_DECAY_EPISODES;
        current_epsilon = EPSILON_START - decay_rate * current_episode;
    }
    
    // Select action based on probability, favoring the best one (no full epsilon-greedy exploration, 
    // but the selection is probabilistic, which serves as exploration).
    // The previous implementation used `select_action` which was slightly broken with masking.
    // I will use a simple probabilistic selection combined with masking for exploration.
    int action_index = -1;
    double r = (double)rand() / RAND_MAX;
    double cumulative_prob = 0.0;
    
    // Select probabilistic action (Policy's choice)
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
        cumulative_prob += probabilities[i];
        if (r < cumulative_prob) {
            action_index = i;
            break;
        }
    }

    // Fallback or ensure legality for the final choice (Action Masking)
    if (action_index == -1 || !is_potential_move_legal(robot->x, robot->y, action_index)) {
        // Fallback to the best legal action if the probabilistic choice was illegal or failed.
        int legal_actions[NN_OUTPUT_SIZE];
        int num_legal = 0;
        for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
            if (is_potential_move_legal(robot->x, robot->y, i)) {
                legal_actions[num_legal++] = i;
            }
        }
        if (num_legal > 0) {
            action_index = legal_actions[rand() % num_legal]; // Random legal action fallback
        } else {
            action_index = ACTION_INVALID;
        }
    }

    if (action_index == ACTION_INVALID) {
        robot->is_alive = false;
        step_count++; 
    } else {
        apply_action(robot, action_index);

        int diamonds_collected = check_collision(robot);
        
        double reward = calculate_reward(old_min_dist_to_goal, diamonds_collected, false);
        
        if (episode_buffer.count < MAX_EPISODE_STEPS) {
            EpisodeStep step;
            memcpy(step.input, input, NN_INPUT_SIZE * sizeof(double));
            step.action_index = action_index;
            step.reward = reward;
            
            episode_buffer.steps[episode_buffer.count] = step;
            episode_buffer.count++;
            episode_buffer.total_score += reward;
        }
        
        step_count++;
    }
}

void run_reinforce_training() {
    if (episode_buffer.count == 0) return;

    double returns[episode_buffer.count];
    double cumulative_return = 0.0;
    
    for (int i = episode_buffer.count - 1; i >= 0; i--) {
        cumulative_return = episode_buffer.steps[i].reward + GAMMA * cumulative_return;
        returns[i] = cumulative_return;
    }
    
    double sum_returns = 0.0;
    double sum_sq_returns = 0.0;
    for (int i = 0; i < episode_buffer.count; i++) {
        sum_returns += returns[i];
        sum_sq_returns += returns[i] * returns[i];
    }
    double mean_return = sum_returns / episode_buffer.count;
    double variance = (sum_sq_returns / episode_buffer.count) - (mean_return * mean_return);
    double std_dev = sqrt(variance > 1e-6 ? variance : 1.0); 

    for (int i = 0; i < episode_buffer.count; i++) {
        double Gt = (returns[i] - mean_return) / std_dev; 
        
        nn_reinforce_train(&nn, 
                           episode_buffer.steps[i].input, 
                           episode_buffer.steps[i].action_index, 
                           -Gt); 
    }
}

void print_episode_stats(double train_time_ms, int episode_count_total, const char* phase_name) {
    Robot* robot = &state.robot;
    
    printf("====================================================\n");
    printf("%s EPISODE %d SUMMARY (Steps: %d/%d)\n", phase_name, episode_count_total, step_count, MAX_EPISODE_STEPS);
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
    printf("RL Training Time: %.3f ms\n", train_time_ms);

    if (current_rl_mode == MODE_NN_REINFORCE) {
        printf("Last %d Actions by AI (Newest to Oldest):\n", ACTION_HISTORY_SIZE);
        for (int i = 1; i <= ACTION_HISTORY_SIZE; i++) {
            int index = (action_history_idx - i + ACTION_HISTORY_SIZE) % ACTION_HISTORY_SIZE;
            if (action_history[index] != -1) { 
                printf("%s%s", action_names[action_history[index]], (i < ACTION_HISTORY_SIZE) ? ", " : "");
            }
        }
        printf("\n");
    }
    printf("====================================================\n\n");
}

void print_ascii_map() {
    printf("\n\n--- FIXED LEVEL ASCII MAP ---\n");
    printf("Legend: #=Wall, O=Obstacle, D=Diamond, S=Start (Robot), T=Target (Goal), .=Free\n");
    
    for (int r = 0; r < GRID_ROWS; r++) {
        for (int c = 0; c < GRID_COLS; c++) {
            
            char symbol = '.';
            double cell_x = (double)c * GRID_CELL_SIZE + GRID_CELL_SIZE / 2.0;
            double cell_y = (double)r * GRID_CELL_SIZE + GRID_CELL_SIZE / 2.0;

            if (r == 0 || r == GRID_ROWS - 1 || c == 0 || c == GRID_COLS - 1) {
                symbol = '#';
            }
            
            for (int i = 0; i < NUM_OBSTACLES; i++) {
                Obstacle* obs = &state.obstacles[i];
                if (cell_x >= obs->x && cell_x <= obs->x + obs->w && cell_y >= obs->y && cell_y <= obs->y + obs->h) {
                    symbol = 'O';
                    break;
                }
            }
            
            if (symbol == '.') { 
                for (int i = 0; i < NUM_DIAMONDS; i++) {
                    Diamond* d = &state.diamonds[i];
                    if ((int)(d->x / GRID_CELL_SIZE) == c && (int)(d->y / GRID_CELL_SIZE) == r) {
                        symbol = 'D';
                        break;
                    }
                }
            }

            if ((int)(state.robot.x / GRID_CELL_SIZE) == c && (int)(state.robot.y / GRID_CELL_SIZE) == r) {
                symbol = 'S';
            }
            
            TargetArea* target = &state.target;
            if (cell_x >= target->x && cell_x <= target->x + target->w && cell_y >= target->y && cell_y <= target->y + target->h) {
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

// --- Main Simulation Loop ---

int main() {
    srand((unsigned int)time(NULL)); 
    init_game_state();
    print_ascii_map();

    for(int i = 0; i < ACTION_HISTORY_SIZE; i++) { action_history[i] = -1; }

    // --- PHASE 1: Q-LEARNING INITIAL TEST RUN (1000 Episodes) ---
    
    // Initialize Q-table to 0
    memset(Q_table, 0, sizeof(Q_table)); 
    current_rl_mode = MODE_QL_TEST;
    current_episode = 0;
    double total_score_ql = 0.0;
    int success_count_ql = 0;
    
    printf("--- PHASE 1: Q-Learning Heuristic Test Run (%d Episodes) ---\n", Q_LEARNING_EPISODES);
    printf("State Space: %d, Alpha: %.2f, Gamma: %.2f\n", Q_STATE_SIZE, Q_LEARNING_ALPHA, GAMMA);

    clock_t ql_start_time = clock();
    for (int i = 0; i < Q_LEARNING_EPISODES; i++) {
        current_episode++;
        init_game_state();
        
        while (state.robot.is_alive && !state.robot.has_reached_target && step_count < MAX_EPISODE_STEPS) {
            update_game_q_learning(); 
        }

        total_score_ql += episode_buffer.total_score;
        if (state.robot.has_reached_target && state.total_diamonds == NUM_DIAMONDS) {
            success_count_ql++;
        }
    }
    clock_t ql_end_time = clock();
    double ql_runtime_ms = (double)(ql_end_time - ql_start_time) * 1000.0 / CLOCKS_PER_SEC;

    printf("\n--- Q-LEARNING TEST SUMMARY ---\n");
    printf("Total Episodes: %d\n", Q_LEARNING_EPISODES);
    printf("Total Runtime: %.3f ms\n", ql_runtime_ms);
    printf("Average Score: %.2f\n", total_score_ql / Q_LEARNING_EPISODES);
    printf("Full Success Rate (All Diamonds + Target): %.2f%% (%d/%d)\n", 
           (double)success_count_ql / Q_LEARNING_EPISODES * 100.0, success_count_ql, Q_LEARNING_EPISODES);
    printf("-------------------------------\n\n");
    
    
    // --- PHASE 2: NEURAL NETWORK REINFORCE SIMULATION ---
    
    nn_init(&nn);
    current_rl_mode = MODE_NN_REINFORCE;
    current_episode = 0; // Reset episode count for NN phase
    
    printf("--- PHASE 2: Neural Network REINFORCE Simulation (Main Run) ---\n");
    printf("Architecture: %d-%d-%d (Hidden units: %d)\n", NN_INPUT_SIZE, NN_HIDDEN_SIZE, NN_OUTPUT_SIZE, NN_HIDDEN_SIZE);
    printf("Training will run for 3 minutes (180 seconds). Stats printed every 50 episodes.\n");

    time_t start_time = time(NULL);
    const int TIME_LIMIT_SECONDS = 180; 
    
    while (time(NULL) - start_time < TIME_LIMIT_SECONDS) {
        
        current_episode++;
        init_game_state();

        // Play episode
        while (state.robot.is_alive && !state.robot.has_reached_target && step_count < MAX_EPISODE_STEPS) {
            update_game_nn_reinforce(); 
        }

        // Train and Time it
        clock_t train_start = clock();
        run_reinforce_training(); 
        clock_t train_end = clock();
        
        double train_time_ms = (double)(train_end - train_start) * 1000.0 / CLOCKS_PER_SEC;

        // Print every 50 episodes
        if (current_episode % 50 == 0) { 
            print_episode_stats(train_time_ms, current_episode, "NN REINFORCE");
        }
    }
    
    printf("\n--- TIME LIMIT REACHED. NN REINFORCE TRAINING HALTED. Total Episodes: %d ---\n", current_episode);

    // --- Cleanup ---
    matrix_free(nn.weights_ih);
    matrix_free(nn.weights_ho);
    free(nn.bias_h);
    free(nn.bias_o);
    printf("Simulation finished and memory cleaned up.\n");
    return 0;
}
