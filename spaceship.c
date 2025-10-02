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
#define MOVE_STEP_SIZE 15.0 // Robot moves this distance per action
#define MAX_EPISODE_STEPS 50 // New move limit
#define NUM_OBSTACLES 5
#define NUM_DIAMONDS 5

// NN & RL Constants
// Input Size (9): Robot X, Robot Y, Target X, Target Y, Diamond dX, Diamond dY, Obstacle dX, Obstacle dY, Collected Ratio
#define NN_INPUT_SIZE 9 
#define NN_HIDDEN_SIZE 16 // Reduced for faster training
#define NN_OUTPUT_SIZE 4  // 0:Up, 1:Down, 2:Left, 3:Right
#define NN_LEARNING_RATE 0.005
#define GAMMA 0.95 

// Reward Goals and Values
#define REWARD_PER_STEP -1.0 
#define REWARD_CRASH -500.0
#define REWARD_SUCCESS 1000.0
#define REWARD_COLLECT_DIAMOND 100.0
#define REWARD_PROGRESS_SCALE 0.01 // Small reward for reducing distance to goal

// Action History Buffer
#define ACTION_HISTORY_SIZE 10 
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

// --- C99 Utility Functions ---
void check_nan_and_stop(double value, const char* var_name, const char* context) {
    if (isnan(value)) { fprintf(stderr, "\n\nCRITICAL NAN ERROR: %s in %s is NaN. Stopping execution.\n", var_name, context); exit(EXIT_FAILURE); }
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
        sum_exp += output[i];
    }
    for (int i = 0; i < size; i++) { output[i] = check_double(output[i] / sum_exp, "softmax_output", "softmax"); }
}

// --- Matrix Functions (Same as previous version, omitted for brevity but included in the file block) ---
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
    for (int i = 0; i < size; i++) {
        m.data[i][0] = arr[i];
    }
    return m;
}

Matrix matrix_dot(Matrix A, Matrix B) {
    if (A.cols != B.rows) {
        fprintf(stderr, "Dot product dimension mismatch: %dx%d DOT %dx%d\n", A.rows, A.cols, B.rows, B.cols);
        exit(EXIT_FAILURE);
    }
    Matrix result = matrix_create(A.rows, B.cols);
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < B.cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < A.cols; k++) {
                sum += check_double(A.data[i][k], "A[i][k]", "matrix_dot") * check_double(B.data[k][j], "B[k][j]", "matrix_dot");
            }
            result.data[i][j] = check_double(sum, "dot_sum", "matrix_dot");
        }
    }
    return result;
}

Matrix matrix_transpose(Matrix m) {
    Matrix result = matrix_create(m.cols, m.rows);
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            result.data[j][i] = m.data[i][j];
        }
    }
    return result;
}

Matrix matrix_add_subtract(Matrix A, Matrix B, bool is_add) {
    if (A.rows != B.rows || A.cols != B.cols) {
        fprintf(stderr, "Add/Subtract dimension mismatch.\n");
        exit(EXIT_FAILURE);
    }
    Matrix result = matrix_create(A.rows, A.cols);
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            if (is_add) {
                result.data[i][j] = check_double(A.data[i][j], "A[i][j]", "matrix_add") + check_double(B.data[i][j], "B[i][j]", "matrix_add");
            } else {
                result.data[i][j] = check_double(A.data[i][j], "A[i][j]", "matrix_subtract") - check_double(B.data[i][j], "B[i][j]", "matrix_subtract");
            }
        }
    }
    return result;
}

Matrix matrix_multiply_scalar(Matrix A, double scalar) {
    Matrix result = matrix_create(A.rows, A.cols);
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            result.data[i][j] = check_double(A.data[i][j], "A[i][j]", "matrix_multiply_scalar") * scalar;
        }
    }
    return result;
}

Matrix matrix_multiply_elem(Matrix A, Matrix B) {
    if (A.rows != B.rows || A.cols != B.cols) {
        fprintf(stderr, "Multiply (element-wise) dimension mismatch.\n");
        exit(EXIT_FAILURE);
    }
    Matrix result = matrix_create(A.rows, A.cols);
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            result.data[i][j] = check_double(A.data[i][j], "A[i][j]", "matrix_multiply_elem") * check_double(B.data[i][j], "B[i][j]", "matrix_multiply_elem");
        }
    }
    return result;
}

Matrix matrix_map(Matrix m, double (*func)(double)) {
    Matrix result = matrix_create(m.rows, m.cols);
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            result.data[i][j] = func(m.data[i][j]);
        }
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
    Matrix inputs = array_to_matrix(input_array, NN_INPUT_SIZE);
    
    // 1. Feedforward (Calculate intermediates)
    Matrix hidden = matrix_dot(nn->weights_ih, inputs);
    for (int i = 0; i < NN_HIDDEN_SIZE; i++) hidden.data[i][0] += nn->bias_h[i];
    Matrix hidden_output = matrix_map(hidden, sigmoid);

    Matrix output_logits_m = matrix_dot(nn->weights_ho, hidden_output);
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) output_logits_m.data[i][0] += nn->bias_o[i];
    
    double logits[NN_OUTPUT_SIZE];
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) logits[i] = output_logits_m.data[i][0];
    double probs[NN_OUTPUT_SIZE];
    softmax(logits, probs, NN_OUTPUT_SIZE);

    // 2. Calculate Output Gradient (dLoss/dLogits) - Core Backprop Step
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
    
    // Fixed Obstacles (x, y, w, h)
    double obs_configs[NUM_OBSTACLES][4] = {
        {150.0, 150.0, 50.0, 250.0},
        {350.0, 150.0, 50.0, 250.0},
        {550.0, 50.0, 50.0, 200.0},
        {550.0, 450.0, 50.0, 100.0},
        {250.0, 400.0, 200.0, 30.0}
    };
    for (int i = 0; i < NUM_OBSTACLES; i++) {
        state.obstacles[i].x = obs_configs[i][0];
        state.obstacles[i].y = obs_configs[i][1];
        state.obstacles[i].w = obs_configs[i][2];
        state.obstacles[i].h = obs_configs[i][3];
    }

    // Fixed Diamonds (x, y)
    double diamond_pos[NUM_DIAMONDS][2] = {
        {100.0, 300.0}, {300.0, 100.0}, {500.0, 300.0}, {700.0, 100.0}, {700.0, 500.0}
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
    return NN_OUTPUT_SIZE - 1; // Fallback to RIGHT
}

double distance_2d(double x1, double y1, double x2, double y2) {
    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}

void get_state_features(double* input, double* min_dist_to_goal_ptr) {
    Robot* robot = &state.robot;
    
    // --- 1. Robot State (2) ---
    input[0] = robot->x / CANVAS_WIDTH;  // Normalized X position
    input[1] = robot->y / CANVAS_HEIGHT; // Normalized Y position
    
    // --- 2. Target State (2) ---
    // Target is static, but relative distance is more useful. We use absolute target pos for simplicity
    input[2] = (state.target.x + state.target.w/2.0) / CANVAS_WIDTH;
    input[3] = (state.target.y + state.target.h/2.0) / CANVAS_HEIGHT;
    
    // Determine the GOAL location (Target or Nearest Diamond)
    double goal_x, goal_y;
    
    if (state.total_diamonds < NUM_DIAMONDS) {
        // Find Nearest Diamond
        Diamond* nearest_diamond = NULL;
        double min_diamond_dist = INFINITY;
        
        for (int i = 0; i < NUM_DIAMONDS; i++) {
            Diamond* d = &state.diamonds[i];
            if (!d->collected) {
                double dist = distance_2d(robot->x, robot->y, d->x, d->y);
                if (dist < min_diamond_dist) { min_diamond_dist = dist; nearest_diamond = d; }
            }
        }
        
        // If diamonds remain, goal is the nearest diamond
        goal_x = nearest_diamond ? nearest_diamond->x : robot->x; 
        goal_y = nearest_diamond ? nearest_diamond->y : robot->y;
        
    } else {
        // All collected, goal is the target area center
        goal_x = state.target.x + state.target.w / 2.0;
        goal_y = state.target.y + state.target.h / 2.0;
    }
    
    // --- 3. Goal Distance (2) - Nearest Diamond or Target Area ---
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
        double obs_center_x = obs->x + obs->w / 2.0;
        double obs_center_y = obs->y + obs->h / 2.0;
        
        double dist = distance_2d(robot->x, robot->y, obs_center_x, obs_center_y);
        if (dist < min_obs_dist) { min_obs_dist = dist; nearest_obs = obs; }
    }
    
    // Normalized signed distance vector to nearest obstacle center
    double obs_dx = nearest_obs ? (nearest_obs->x + nearest_obs->w / 2.0) - robot->x : 0.0;
    double obs_dy = nearest_obs ? (nearest_obs->y + nearest_obs->h / 2.0) - robot->y : 0.0;
    
    input[6] = check_double(obs_dx / CANVAS_WIDTH, "norm_obs_dx", "get_state_features");
    input[7] = check_double(obs_dy / CANVAS_HEIGHT, "norm_obs_dy", "get_state_features"); 

    // --- 5. Collected Ratio (1) ---
    input[8] = (double)state.total_diamonds / NUM_DIAMONDS;

    // Clamp features between -1.0 and 1.0 (already handled by normalization above, but safe)
    for (int i = 0; i < NN_INPUT_SIZE; i++) {
        if (input[i] > 1.0) input[i] = 1.0;
        if (input[i] < -1.0) input[i] = -1.0;
    }
}

double calculate_reward(double old_min_dist_to_goal, int diamonds_collected_this_step) {
    double reward = REWARD_PER_STEP; 
    Robot* robot = &state.robot;
    
    if (!robot->is_alive) return REWARD_CRASH;
    
    // --- Collection Reward ---
    if (diamonds_collected_this_step > 0) {
        reward += REWARD_COLLECT_DIAMOND * diamonds_collected_this_step;
    }

    // --- Success Reward ---
    if (robot->has_reached_target) {
        reward += REWARD_SUCCESS;
        if (state.total_diamonds < NUM_DIAMONDS) {
            // Penalize reaching target without all diamonds (less than full success)
            reward -= (NUM_DIAMONDS - state.total_diamonds) * 50.0; 
        }
    }
    
    // --- Progress Reward (Shaping) ---
    double min_dist_to_goal;
    double dummy_input[NN_INPUT_SIZE];
    get_state_features(dummy_input, &min_dist_to_goal); // Calculate current goal distance
    
    double distance_change = old_min_dist_to_goal - min_dist_to_goal;
    
    if (distance_change > 0) {
        // Small positive reward for moving closer to the current goal
        reward += REWARD_PROGRESS_SCALE * distance_change;
    } else {
         // Small penalty for moving away
        reward += REWARD_PROGRESS_SCALE * distance_change;
    }
    
    // Penalize movement if goal is reached or robot died (though this case shouldn't happen)
    if (robot->has_reached_target || !robot->is_alive) reward = 0.0;
    
    return check_double(reward, "final_reward", "calc_reward");
}


void apply_action(Robot* robot, int action_index) {
    double dx = 0.0;
    double dy = 0.0;
    
    switch (action_index) {
        case 0: dy = -MOVE_STEP_SIZE; break; // UP (Negative Y is up)
        case 1: dy = MOVE_STEP_SIZE;  break; // DOWN
        case 2: dx = -MOVE_STEP_SIZE; break; // LEFT
        case 3: dx = MOVE_STEP_SIZE;  break; // RIGHT
    }

    robot->x += dx;
    robot->y += dy;
    
    // Record action for stats
    action_history[action_history_idx] = action_index;
    action_history_idx = (action_history_idx + 1) % ACTION_HISTORY_SIZE;
}


int check_collision(Robot* robot) {
    double r = robot->size; 
    int diamonds_collected_this_step = 0;
    
    // --- 1. Wall Collision (Instant crash) ---
    if (robot->x - r < BORDER_WIDTH || robot->x + r > CANVAS_WIDTH - BORDER_WIDTH ||
        robot->y - r < BORDER_WIDTH || robot->y + r > CANVAS_HEIGHT - BORDER_WIDTH) {
        robot->is_alive = false;
        return 0; 
    }
    
    // --- 2. Obstacle Collision (Instant crash) ---
    for (int i = 0; i < NUM_OBSTACLES; i++) {
        Obstacle* obs = &state.obstacles[i];
        
        // AABB-Circle approximate check
        double closest_x = fmax(obs->x, fmin(robot->x, obs->x + obs->w));
        double closest_y = fmax(obs->y, fmin(robot->y, obs->y + obs->h));
        
        double dx = robot->x - closest_x;
        double dy = robot->y - closest_y;
        
        if (dx * dx + dy * dy < r * r) {
            robot->is_alive = false;
            return 0; // Crash
        }
    }
    
    // --- 3. Diamond Collection ---
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

    // --- 4. Target Area Check (Goal) ---
    TargetArea* target = &state.target;
    if (robot->x + r > target->x && robot->x - r < target->x + target->w &&
        robot->y + r > target->y && robot->y - r < target->y + target->h) {
        robot->has_reached_target = true;
        // The episode will terminate in the caller (update_game)
    }
    
    return diamonds_collected_this_step;
}

void print_episode_stats(double train_time_ms) {
    Robot* robot = &state.robot;
    
    printf("====================================================\n");
    printf("EPISODE %d SUMMARY (Steps: %d/%d)\n", current_episode, step_count, MAX_EPISODE_STEPS);
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
    
    printf("Last %d Actions by AI (Newest to Oldest):\n", ACTION_HISTORY_SIZE);
    for (int i = 1; i <= ACTION_HISTORY_SIZE; i++) {
        int index = (action_history_idx - i + ACTION_HISTORY_SIZE) % ACTION_HISTORY_SIZE;
        printf("%s%s", action_names[action_history[index]], (i < ACTION_HISTORY_SIZE) ? ", " : "");
    }
    printf("\n");
    printf("====================================================\n\n");
}

void update_game(bool is_training_run) {
    Robot* robot = &state.robot;
    
    if (!robot->is_alive || robot->has_reached_target || step_count >= MAX_EPISODE_STEPS) return; 

    // --- 1. Get current state features and previous goal distance ---
    double input[NN_INPUT_SIZE];
    double old_min_dist_to_goal;
    get_state_features(input, &old_min_dist_to_goal);

    // --- 2. Feedforward and Action Selection ---
    double logit_output[NN_OUTPUT_SIZE];
    double probabilities[NN_OUTPUT_SIZE];
    nn_policy_forward(&nn, input, probabilities, logit_output);

    int action_index = select_action(probabilities);

    // --- 3. Apply Action and Check Collision ---
    apply_action(robot, action_index);

    int diamonds_collected = check_collision(robot);
    
    // --- 4. Calculate Reward and Store Step ---
    double reward = calculate_reward(old_min_dist_to_goal, diamonds_collected);
    
    if (is_training_run && episode_buffer.count < MAX_EPISODE_STEPS) {
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

void run_reinforce_training() {
    if (episode_buffer.count == 0) return;

    // --- 1. Calculate Discounted Returns (G_t) ---
    double returns[episode_buffer.count];
    double cumulative_return = 0.0;
    
    for (int i = episode_buffer.count - 1; i >= 0; i--) {
        cumulative_return = episode_buffer.steps[i].reward + GAMMA * cumulative_return;
        returns[i] = cumulative_return;
    }
    
    // --- 2. Normalize Returns (Baseline) ---
    double sum_returns = 0.0;
    double sum_sq_returns = 0.0;
    for (int i = 0; i < episode_buffer.count; i++) {
        sum_returns += returns[i];
        sum_sq_returns += returns[i] * returns[i];
    }
    double mean_return = sum_returns / episode_buffer.count;
    double variance = (sum_sq_returns / episode_buffer.count) - (mean_return * mean_return);
    double std_dev = sqrt(variance > 1e-6 ? variance : 1.0); 

    // --- 3. Train the Network using Backpropagation (REINFORCE) ---
    for (int i = 0; i < episode_buffer.count; i++) {
        double Gt = (returns[i] - mean_return) / std_dev; 
        
        nn_reinforce_train(&nn, 
                           episode_buffer.steps[i].input, 
                           episode_buffer.steps[i].action_index, 
                           -Gt); 
    }
}

// --- Main Simulation Loop ---

int main() {
    srand((unsigned int)time(NULL)); 
    nn_init(&nn);
    
    for(int i = 0; i < ACTION_HISTORY_SIZE; i++) {
        action_history[i] = 3; // Default to RIGHT
    }

    printf("--- RL 2D Robot Collector Simulation (Fixed Level / 50 Moves) ---\n");
    printf("Input Size: %d, Hidden Size: %d, Output Size: %d\n", NN_INPUT_SIZE, NN_HIDDEN_SIZE, NN_OUTPUT_SIZE);
    printf("Training will run for 3 minutes (180 seconds).\n");
    
    time_t start_time = time(NULL);
    const int TIME_LIMIT_SECONDS = 180; 
    
    // --- Time-Limited Training Phase ---
    while (time(NULL) - start_time < TIME_LIMIT_SECONDS) {
        
        current_episode++;
        init_game_state();

        // Play episode
        while (state.robot.is_alive && !state.robot.has_reached_target && step_count < MAX_EPISODE_STEPS) {
            update_game(true); // true for training run
        }

        // Train and Time it
        clock_t train_start = clock();
        run_reinforce_training(); 
        clock_t train_end = clock();
        
        double train_time_ms = (double)(train_end - train_start) * 1000.0 / CLOCKS_PER_SEC;

        // Print stats
        print_episode_stats(train_time_ms);
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