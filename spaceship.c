#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

// --- Fix for M_PI undeclared error ---
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- Global Constants ---
#define CANVAS_WIDTH 800.0
#define CANVAS_HEIGHT 600.0
#define GROUND_HEIGHT 50.0

// Landing Pad Definition
#define LANDING_PAD_X 300.0
#define LANDING_PAD_W 200.0
#define LANDING_PAD_Y (CANVAS_HEIGHT - GROUND_HEIGHT)

// Physics Constants
#define GRAVITY 0.05
#define THRUST_POWER 0.15
#define ROTATION_SPEED 0.05
#define MAX_VELOCITY_FOR_NORM 10.0
#define CRITICAL_LANDING_VELOCITY 1.5 // Max vertical speed for safe landing

// NN & RL Constants
#define NN_INPUT_SIZE 14 // Ship(5) + Diamond(3) + Obstacle(3) + Landing(2) + Collected(1)
#define NN_HIDDEN_SIZE 32
#define NN_OUTPUT_SIZE 3 // 0:Thrust, 1:Rotate Left, 2:Rotate Right
#define NN_LEARNING_RATE 0.005
#define RL_MAX_EPISODES 2000 // Number of training episodes
#define GAMMA 0.99 // Discount factor for future rewards

// Game Constants
#define NUM_OBSTACLES 5
#define NUM_DIAMONDS 10
#define SIMULATION_DT (1.0 / 60.0) // 60 FPS simulation step
#define PRINT_INTERVAL 200 // Print state every X steps

// Reward Goals and Values (Used for Reward Function Implementation)
#define REWARD_PER_STEP -0.01 
#define REWARD_CRASH -200.0
#define REWARD_SAFE_LAND 50.0
#define REWARD_SAFE_LAND_ALL_COLLECTED 500.0
#define REWARD_COLLECT_DIAMOND 50.0
#define REWARD_VELOCITY_PENALTY_SCALE -0.5 // For STABILIZE goal
#define REWARD_OBSTACLE_PROXIMITY_SCALE -5.0 // For AVOID_OBSTACLE goal

// --- C99 Utility Functions ---

void check_nan_and_stop(double value, const char* var_name, const char* context) {
    if (isnan(value)) {
        fprintf(stderr, "\n\nCRITICAL NAN ERROR: %s in %s is NaN. Stopping execution.\n", var_name, context);
        exit(EXIT_FAILURE);
    }
}

double check_double(double value, const char* var_name, const char* context) {
    check_nan_and_stop(value, var_name, context);
    return value;
}

// Function to compute Softmax (Output layer activation for policy)
void softmax(const double* input, double* output, int size) {
    double max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) max_val = input[i];
    }

    double sum_exp = 0.0;
    for (int i = 0; i < size; i++) {
        output[i] = exp(input[i] - max_val); // Stable exponential
        sum_exp += output[i];
    }

    // Normalization
    for (int i = 0; i < size; i++) {
        output[i] = check_double(output[i] / sum_exp, "softmax_output", "softmax");
    }
}

// Function to calculate the natural logarithm
double log_val(double x) {
    if (x <= 0.0) return -INFINITY; 
    return log(x);
}

// --- Data Structures ---

typedef struct {
    double x;
    double y;
} Vector2D;

typedef struct {
    double x, y;
    double w, h;
} Obstacle;

typedef struct {
    double x, y;
    double size;
    bool collected;
} Diamond;

typedef struct {
    double x, y;
    Vector2D velocity;
    double angle;
    double size;
    bool is_thrusting;
    bool is_alive;
    bool has_landed;
} Ship;

typedef struct {
    int score;
    int total_diamonds;
    Ship ship;
    Obstacle obstacles[NUM_OBSTACLES];
    Diamond diamonds[NUM_DIAMONDS];
} GameState;

// Matrix Structure
typedef struct {
    int rows;
    int cols;
    double** data;
} Matrix;

// Neural Network Structure (Weights and Biases)
typedef struct {
    Matrix weights_ih;
    Matrix weights_ho;
    double* bias_h;
    double* bias_o;
    double lr;
} NeuralNetwork;

// RL Step Structure (for REINFORCE)
typedef struct {
    double input[NN_INPUT_SIZE];
    int action_index; // The action taken (0, 1, or 2)
    double reward;
    double log_prob; // log(P(Action|State))
} EpisodeStep;

// RL Episode Buffer
typedef struct {
    EpisodeStep steps[4000]; // Max steps per episode
    int count;
    double total_score;
} Episode;

// --- Global State ---
GameState state;
NeuralNetwork nn;
Episode episode_buffer;
int current_episode = 0;

// --- Matrix Operations (Memory Managed) ---

Matrix matrix_create(int rows, int cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.data = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        m.data[i] = (double*)malloc(cols * sizeof(double));
        for (int j = 0; j < cols; j++) {
            // Xavier/He initialization (small random values)
            m.data[i][j] = check_double((((double)rand() / RAND_MAX) * 2.0 - 1.0) * sqrt(2.0 / (rows + cols)), "rand_val", "matrix_create");
        }
    }
    return m;
}

void matrix_free(Matrix m) {
    for (int i = 0; i < m.rows; i++) {
        free(m.data[i]);
    }
    free(m.data);
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

Matrix matrix_multiply_scalar(Matrix A, double scalar) {
    Matrix result = matrix_create(A.rows, A.cols);
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            result.data[i][j] = check_double(A.data[i][j], "A[i][j]", "matrix_multiply_scalar") * scalar;
        }
    }
    return result;
}

// --- NN Activation Functions ---

double sigmoid(double x) {
    double result = 1.0 / (1.0 + exp(-x));
    return check_double(result, "sigmoid_output", "sigmoid");
}

double sigmoid_derivative(double y) {
    double result = y * (1.0 - y);
    return check_double(result, "sigmoid_deriv_output", "sigmoid_derivative");
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

// --- Neural Network Methods ---

void nn_init(NeuralNetwork* nn) {
    nn->lr = check_double(NN_LEARNING_RATE, "NN_LEARNING_RATE", "nn_init");
    nn->weights_ih = matrix_create(NN_HIDDEN_SIZE, NN_INPUT_SIZE);
    nn->weights_ho = matrix_create(NN_OUTPUT_SIZE, NN_HIDDEN_SIZE);
    nn->bias_h = (double*)malloc(NN_HIDDEN_SIZE * sizeof(double));
    nn->bias_o = (double*)malloc(NN_OUTPUT_SIZE * sizeof(double));

    for (int i = 0; i < NN_HIDDEN_SIZE; i++) {
        nn->bias_h[i] = check_double((((double)rand() / RAND_MAX) * 2.0 - 1.0) * 0.01, "bias_h_val", "nn_init");
    }
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
        nn->bias_o[i] = check_double((((double)rand() / RAND_MAX) * 2.0 - 1.0) * 0.01, "bias_o_val", "nn_init");
    }
}

Matrix array_to_matrix(const double* arr, int size) {
    Matrix m = matrix_create(size, 1);
    for (int i = 0; i < size; i++) {
        m.data[i][0] = arr[i];
    }
    return m;
}

// NN Forward Pass (Policy/Action Selection)
void nn_policy_forward(NeuralNetwork* nn, const double* input_array, double* output_probabilities, double* logit_output) {
    Matrix inputs = array_to_matrix(input_array, NN_INPUT_SIZE);

    // Hidden layer (sigmoid activation)
    Matrix hidden = matrix_dot(nn->weights_ih, inputs);
    for (int i = 0; i < NN_HIDDEN_SIZE; i++) {
        hidden.data[i][0] += nn->bias_h[i];
    }
    Matrix hidden_output = matrix_map(hidden, sigmoid);

    // Output layer (linear activation - outputs logits)
    Matrix output_logits_m = matrix_dot(nn->weights_ho, hidden_output);
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
        output_logits_m.data[i][0] += nn->bias_o[i];
        logit_output[i] = output_logits_m.data[i][0];
    }
    
    // Convert logits to probabilities using Softmax
    softmax(logit_output, output_probabilities, NN_OUTPUT_SIZE);

    matrix_free(inputs);
    matrix_free(hidden);
    matrix_free(hidden_output);
    matrix_free(output_logits_m);
}

// NN Policy Gradient Update (REINFORCE Step)
void nn_reinforce_train(NeuralNetwork* nn, const double* input_array, int action_index, double discounted_return) {
    Matrix inputs = array_to_matrix(input_array, NN_INPUT_SIZE);
    
    // --- 1. Feedforward (Calculates intermediates) ---
    Matrix hidden = matrix_dot(nn->weights_ih, inputs);
    for (int i = 0; i < NN_HIDDEN_SIZE; i++) hidden.data[i][0] += nn->bias_h[i];
    Matrix hidden_output = matrix_map(hidden, sigmoid);

    Matrix output_logits_m = matrix_dot(nn->weights_ho, hidden_output);
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) output_logits_m.data[i][0] += nn->bias_o[i];
    
    // Get probabilities for gradient calculation
    double logits[NN_OUTPUT_SIZE];
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) logits[i] = output_logits_m.data[i][0];
    double probs[NN_OUTPUT_SIZE];
    softmax(logits, probs, NN_OUTPUT_SIZE);

    // --- 2. Calculate Output Gradient (dLoss/dLogits) ---
    // dLoss/dLogit_i = (Probs_i - Target_i) * Discounted_Return
    // Target is one-hot for the taken action (action_index).
    Matrix output_gradients = matrix_create(NN_OUTPUT_SIZE, 1);
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
        double target = (i == action_index) ? 1.0 : 0.0;
        // Gradient of Cross-Entropy Loss w.r.t logits for target: -(target - prob)
        // We use the gradient log(P(a|s)) * Gt
        output_gradients.data[i][0] = check_double(probs[i] - target, "output_grad_base", "nn_reinforce_train") * discounted_return;
    }

    // --- 3. Update Weights HO and Bias O ---
    Matrix delta_weights_ho = matrix_multiply_scalar(matrix_dot(output_gradients, matrix_transpose(hidden_output)), -nn->lr);
    Matrix new_weights_ho = matrix_add_subtract(nn->weights_ho, delta_weights_ho, true);
    matrix_free(nn->weights_ho); nn->weights_ho = new_weights_ho;

    for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
        nn->bias_o[i] = check_double(nn->bias_o[i], "bias_o", "nn_train_upd") - check_double(output_gradients.data[i][0], "grad_o", "nn_train_upd") * nn->lr;
    }

    // --- 4. Calculate Hidden Errors and Update Weights IH and Bias H ---
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
    
    // --- 5. Cleanup ---
    matrix_free(inputs); matrix_free(hidden); matrix_free(hidden_output);
    matrix_free(output_logits_m); matrix_free(output_gradients); matrix_free(weights_ho_T);
    matrix_free(hidden_errors); matrix_free(hidden_gradients); matrix_free(hidden_gradients_mul);
    matrix_free(delta_weights_ho); matrix_free(delta_weights_ih);
}


// --- Game Logic Functions ---

void init_game_state() {
    state.score = 0;
    state.total_diamonds = 0;

    // Ship setup
    state.ship.x = CANVAS_WIDTH / 2.0;
    state.ship.y = 50.0;
    state.ship.velocity.x = 0.0;
    state.ship.velocity.y = 0.0;
    state.ship.angle = M_PI / 2.0; 
    state.ship.size = 15.0;
    state.ship.is_thrusting = false;
    state.ship.is_alive = true;
    state.ship.has_landed = false;
    
    // Reset episode buffer
    episode_buffer.count = 0;
    episode_buffer.total_score = 0;

    // Obstacles
    state.obstacles[0] = (Obstacle){100, 300, 150, 20};
    state.obstacles[1] = (Obstacle){550, 200, 100, 20};
    state.obstacles[2] = (Obstacle){300, 450, 200, 20};
    state.obstacles[3] = (Obstacle){50, 150, 50, 50};
    state.obstacles[4] = (Obstacle){650, 400, 150, 20};

    // Diamonds
    srand((unsigned int)time(NULL) * (current_episode + 1)); 
    for (int i = 0; i < NUM_DIAMONDS; i++) {
        state.diamonds[i].x = check_double(((double)rand() / RAND_MAX) * (CANVAS_WIDTH - 40.0) + 20.0, "diamond_x", "init_game");
        state.diamonds[i].y = check_double(((double)rand() / RAND_MAX) * (CANVAS_HEIGHT - GROUND_HEIGHT - 100.0) + 20.0, "diamond_y", "init_game");
        state.diamonds[i].size = 5.0;
        state.diamonds[i].collected = false;
    }
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
    return NN_OUTPUT_SIZE - 1; // Fallback to last action
}

void get_state_features(double* input) {
    Ship* ship = &state.ship;
    
    // --- 1. Ship State Features (5) ---
    input[0] = check_double(ship->x / CANVAS_WIDTH, "norm_x", "get_state_features");
    input[1] = check_double(ship->y / CANVAS_HEIGHT, "norm_y", "get_state_features"); 
    input[2] = check_double(ship->velocity.x / MAX_VELOCITY_FOR_NORM, "norm_vx", "get_state_features"); 
    input[3] = check_double(ship->velocity.y / MAX_VELOCITY_FOR_NORM, "norm_vy", "get_state_features");
    input[4] = check_double(ship->angle / (2.0 * M_PI), "norm_angle", "get_state_features");
    
    // --- 2. Nearest Diamond Features (3) ---
    Diamond* nearest_diamond = NULL;
    double min_diamond_dist = INFINITY;
    for (int i = 0; i < NUM_DIAMONDS; i++) {
        Diamond* d = &state.diamonds[i];
        if (!d->collected) {
            double dx = check_double(ship->x - d->x, "d_dx", "get_state_features");
            double dy = check_double(ship->y - d->y, "d_dy", "get_state_features");
            double dist = check_double(sqrt(dx * dx + dy * dy), "d_dist", "get_state_features");
            if (dist < min_diamond_dist) { min_diamond_dist = dist; nearest_diamond = d; }
        }
    }
    input[5] = nearest_diamond ? check_double(nearest_diamond->x / CANVAS_WIDTH, "norm_d_x", "get_state_features") : 0.5; 
    input[6] = nearest_diamond ? check_double(nearest_diamond->y / CANVAS_HEIGHT, "norm_d_y", "get_state_features") : 0.5;
    input[7] = check_double(min_diamond_dist / CANVAS_WIDTH, "norm_d_dist", "get_state_features");
    
    // --- 3. Nearest Obstacle Features (3) ---
    Obstacle* nearest_obs = NULL;
    double min_obs_dist = INFINITY;
    for (int i = 0; i < NUM_OBSTACLES; i++) {
        Obstacle* obs = &state.obstacles[i];
        double obs_center_x = obs->x + obs->w / 2.0;
        double obs_center_y = obs->y + obs->h / 2.0;
        double dx = check_double(ship->x - obs_center_x, "o_dx", "get_state_features");
        double dy = check_double(ship->y - obs_center_y, "o_dy", "get_state_features");
        double dist = check_double(sqrt(dx * dx + dy * dy), "o_dist", "get_state_features");
        if (dist < min_obs_dist) { min_obs_dist = dist; nearest_obs = obs; }
    }
    input[8] = nearest_obs ? check_double(nearest_obs->x / CANVAS_WIDTH, "norm_o_x", "get_state_features") : 0.5; 
    input[9] = nearest_obs ? check_double(nearest_obs->y / CANVAS_HEIGHT, "norm_o_y", "get_state_features") : 0.5;
    input[10] = check_double(min_obs_dist / CANVAS_WIDTH, "norm_o_dist", "get_state_features");

    // --- 4. Landing Pad Features (2) ---
    input[11] = check_double(LANDING_PAD_X / CANVAS_WIDTH, "norm_lp_x", "get_state_features");
    input[12] = check_double(LANDING_PAD_W / CANVAS_WIDTH, "norm_lp_w", "get_state_features");
    
    // --- 5. All Collected Flag (1) ---
    input[13] = (state.total_diamonds == NUM_DIAMONDS) ? 1.0 : 0.0;

    for (int i = 0; i < NN_INPUT_SIZE; i++) {
        // Simple clipping to prevent extreme values from ruining training
        if (input[i] > 1.0) input[i] = 1.0;
        if (input[i] < -1.0) input[i] = -1.0;
        check_double(input[i], "input_val", "get_state_features_final");
    }
}

double calculate_reward(Ship* ship, double old_min_diamond_dist, int diamonds_collected_this_step) {
    double reward = REWARD_PER_STEP;
    
    if (!ship->is_alive) {
        return REWARD_CRASH;
    }
    
    // Goal 1: STABILIZE (against velocity/gravity)
    double speed_magnitude_sq = check_double(ship->velocity.x * ship->velocity.x + ship->velocity.y * ship->velocity.y, "speed_sq", "calc_reward");
    reward += REWARD_VELOCITY_PENALTY_SCALE * speed_magnitude_sq;
    
    // Goal 2 & 3: TOWARDS_DIAMOND & AVOID_OBSTACLE
    double input[NN_INPUT_SIZE];
    get_state_features(input);
    double nearest_obs_dist = input[10] * CANVAS_WIDTH; // Denormalized obstacle distance

    // AVOID_OBSTACLE: Punish heavily for being too close
    double OBSTACLE_NEAR_THRESHOLD = 50.0;
    if (nearest_obs_dist < OBSTACLE_NEAR_THRESHOLD) {
        reward += REWARD_OBSTACLE_PROXIMITY_SCALE * (1.0 - (nearest_obs_dist / OBSTACLE_NEAR_THRESHOLD));
    }
    
    // TOWARDS_DIAMOND: Reward collection
    if (diamonds_collected_this_step > 0) {
        reward += REWARD_COLLECT_DIAMOND * diamonds_collected_this_step;
    }

    // TOWARDS_DIAMOND: Small reward for progress if diamonds remain
    if (state.total_diamonds < NUM_DIAMONDS) {
        double new_min_diamond_dist = input[7] * CANVAS_WIDTH;
        double distance_change = old_min_diamond_dist - new_min_diamond_dist;
        // Reward for positive change (getting closer)
        if (distance_change > 0) {
            reward += 0.05 * distance_change / 10.0; // Small reward scaled by progress
        }
    }
    
    // Goal 4: LANDING (Final Goal)
    if (ship->has_landed) {
        if (state.total_diamonds == NUM_DIAMONDS) {
            reward += REWARD_SAFE_LAND_ALL_COLLECTED;
        } else {
            reward += REWARD_SAFE_LAND;
        }
    }
    
    return check_double(reward, "final_reward", "calc_reward");
}


void apply_action(Ship* ship, int action_index) {
    ship->is_thrusting = false; // Reset thrust
    
    if (action_index == 0) { // Thrust
        ship->is_thrusting = true;
    } else if (action_index == 1) { // Rotate Left
        ship->angle = check_double(ship->angle - ROTATION_SPEED, "ship_angle_L", "apply_action");
    } else if (action_index == 2) { // Rotate Right
        ship->angle = check_double(ship->angle + ROTATION_SPEED, "ship_angle_R", "apply_action");
    }
    
    // Normalize angle (FIX: Use M_PI constant)
    if (ship->angle > 2.0 * M_PI) ship->angle = check_double(ship->angle - 2.0 * M_PI, "ship_angle_norm", "apply_action");
    if (ship->angle < 0.0) ship->angle = check_double(ship->angle + 2.0 * M_PI, "ship_angle_norm", "apply_action");
}

void apply_thrust(Ship* ship, double dt) {
    if (!ship->is_thrusting) return;
    
    double thrust_force = check_double(THRUST_POWER * dt / SIMULATION_DT, "thrust_force", "apply_thrust");

    ship->velocity.x = check_double(ship->velocity.x + cos(ship->angle) * thrust_force, "ship_vx", "apply_thrust");
    ship->velocity.y = check_double(ship->velocity.y + sin(ship->angle) * thrust_force, "ship_vy", "apply_thrust");
}

void update_physics(Ship* ship, double dt) {
    // Apply Gravity (always down, positive y is down)
    ship->velocity.y = check_double(ship->velocity.y + GRAVITY * dt / SIMULATION_DT, "ship_vy_grav", "update_physics");

    ship->x = check_double(ship->x + ship->velocity.x * dt, "ship_x", "update_physics");
    ship->y = check_double(ship->y + ship->velocity.y * dt, "ship_y", "update_physics");

    // Wrap around horizontal edges
    if (ship->x < 0.0) ship->x = CANVAS_WIDTH;
    if (ship->x > CANVAS_WIDTH) ship->x = 0.0;
}

int check_collision(Ship* ship) {
    double ship_radius = ship->size; 
    int diamonds_collected_this_step = 0;
    
    // Obstacle collision - uses fmax and fmin
    for (int i = 0; i < NUM_OBSTACLES; i++) {
        Obstacle* obs = &state.obstacles[i];
        double closest_x = fmax(obs->x, fmin(ship->x, obs->x + obs->w));
        double closest_y = fmax(obs->y, fmin(ship->y, obs->y + obs->h));
        double dx = check_double(ship->x - closest_x, "coll_dx", "check_collision");
        double dy = check_double(ship->y - closest_y, "coll_dy", "check_collision");

        if (check_double(dx * dx + dy * dy, "coll_dist_sq", "check_collision") < (ship_radius * ship_radius)) {
            ship->is_alive = false;
            return diamonds_collected_this_step; // Crash overrides diamond collection
        }
    }

    // Diamond collection
    for (int i = 0; i < NUM_DIAMONDS; i++) {
        Diamond* d = &state.diamonds[i];
        if (d->collected) continue;
        double dx = check_double(ship->x - d->x, "diamond_dx_coll", "check_collision");
        double dy = check_double(ship->y - d->y, "diamond_dy_coll", "check_collision");
        double distance = check_double(sqrt(dx * dx + dy * dy), "diamond_dist", "check_collision");

        if (distance < ship_radius + d->size) {
            d->collected = true;
            state.total_diamonds++;
            diamonds_collected_this_step++;
            state.score += (int)REWARD_COLLECT_DIAMOND;
        }
    }
    return diamonds_collected_this_step;
}

void update_game(double dt, bool is_training_run) {
    Ship* ship = &state.ship;
    
    if (!ship->is_alive || ship->has_landed) return; 

    // --- 1. Get current state features ---
    double input[NN_INPUT_SIZE];
    get_state_features(input);
    double old_min_diamond_dist = input[7] * CANVAS_WIDTH;

    // --- 2. Feedforward and Action Selection ---
    double logit_output[NN_OUTPUT_SIZE];
    double probabilities[NN_OUTPUT_SIZE];
    nn_policy_forward(&nn, input, probabilities, logit_output);

    int action_index = select_action(probabilities);
    double log_prob_action = log_val(probabilities[action_index]);

    // --- 3. Apply Action and Physics Update ---
    apply_action(ship, action_index);
    apply_thrust(ship, dt);
    update_physics(ship, dt);

    // --- 4. Check Collision and Rewards ---
    int diamonds_collected = check_collision(ship);
    
    // --- 5. Check Landing ---
    double ground_y = LANDING_PAD_Y;
    if (ship->y + ship->size > ground_y) {
        ship->y = ground_y - ship->size;
        
        bool is_in_pad = (ship->x >= LANDING_PAD_X && ship->x <= LANDING_PAD_X + LANDING_PAD_W);
        double speed_magnitude = check_double(sqrt(ship->velocity.x * ship->velocity.x + ship->velocity.y * ship->velocity.y), "speed_mag", "update_game");
        
        // Crash if velocity is too high, or landed off pad
        if (fabs(ship->velocity.y) > CRITICAL_LANDING_VELOCITY || speed_magnitude > CRITICAL_LANDING_VELOCITY || !is_in_pad) {
            ship->is_alive = false;
        } else {
            ship->has_landed = true;
        }
        
        ship->velocity.x = 0.0;
        ship->velocity.y = 0.0;
        ship->is_thrusting = false;
    }

    // --- 6. Calculate Reward and Store Step ---
    double reward = calculate_reward(ship, old_min_diamond_dist, diamonds_collected);
    
    if (is_training_run && episode_buffer.count < 4000) {
        EpisodeStep step;
        for(int i = 0; i < NN_INPUT_SIZE; i++) step.input[i] = input[i];
        step.action_index = action_index;
        step.reward = reward;
        step.log_prob = log_prob_action;
        
        episode_buffer.steps[episode_buffer.count] = step;
        episode_buffer.count++;
        episode_buffer.total_score += reward;
    }
}

void print_state(double sim_time, int episode) {
    Ship* ship = &state.ship;
    double speed_magnitude = check_double(sqrt(ship->velocity.x * ship->velocity.x + ship->velocity.y * ship->velocity.y), "speed_mag_print", "print_state");
    
    printf("--- E%d: SIMULATION TIME: %.2fs ---\n", episode, sim_time);
    
    printf("SpaceShip State: %s | Collected: %d/%d\n", 
        !ship->is_alive ? "CRASHED" : 
        ship->has_landed ? "LANDED" : 
        "IN FLIGHT", state.total_diamonds, NUM_DIAMONDS);
    
    printf("  Position (x, y): (%.2f, %.2f) | Angle: %.2f rad\n", ship->x, ship->y, ship->angle);
    printf("  Speed (Vx, Vy, Mag): (%.2f, %.2f, %.2f)\n", 
           ship->velocity.x, ship->velocity.y, speed_magnitude);
    printf("  Accumulated Score/Reward: %.2f (Raw Score: %d)\n", episode_buffer.total_score, state.score);
    printf("--------------------------------\n\n");
}

void run_reinforce_training() {
    if (episode_buffer.count == 0) return;

    // --- 1. Calculate Discounted Returns (G_t) ---
    double returns[episode_buffer.count];
    double cumulative_return = 0.0;
    
    // Iterate backwards to calculate discounted return
    for (int i = episode_buffer.count - 1; i >= 0; i--) {
        cumulative_return = episode_buffer.steps[i].reward + GAMMA * cumulative_return;
        returns[i] = cumulative_return;
    }
    
    // --- 2. Normalize Returns (Optional but recommended for stability - Baseline) ---
    double sum_returns = 0.0;
    double sum_sq_returns = 0.0;
    for (int i = 0; i < episode_buffer.count; i++) {
        sum_returns += returns[i];
        sum_sq_returns += returns[i] * returns[i];
    }
    double mean_return = sum_returns / episode_buffer.count;
    double variance = (sum_sq_returns / episode_buffer.count) - (mean_return * mean_return);
    double std_dev = sqrt(variance > 1e-6 ? variance : 1.0); // Epsilon for stability

    // Apply policy gradient update for each step
    for (int i = 0; i < episode_buffer.count; i++) {
        // Normalized and baseline-subtracted return
        double Gt = (returns[i] - mean_return) / std_dev; 
        
        // Gradient update: -LR * Gt * gradient of log(P(a|s))
        // We pass -Gt to the nn_reinforce_train function to simplify the final calculation
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
    init_game_state();

    printf("--- RL Autonomous Lander Simulation (REINFORCE) ---\n");
    
    // --- Training Phase ---
    printf("Starting Training (%d Episodes)...\n", RL_MAX_EPISODES);
    
    double last_print_time = 0.0;
    
    for (current_episode = 1; current_episode <= RL_MAX_EPISODES; current_episode++) {
        double sim_time = 0.0;
        init_game_state();

        // Run episode until termination
        while (state.ship.is_alive && !state.ship.has_landed && sim_time < 500.0) {
            update_game(SIMULATION_DT, true); // true for training run
            sim_time += SIMULATION_DT;
        }

        // Training Step
        run_reinforce_training();
        
        // Log results
        if (current_episode % (RL_MAX_EPISODES / 10) == 0) {
            printf("[TRAINING E%d] Total Steps: %d | Final Reward: %.2f\n", 
                current_episode, episode_buffer.count, episode_buffer.total_score);
        }
    }
    
    printf("\n--- TRAINING COMPLETE. STARTING FINAL NN CONTROL SIMULATION ---\n");
    
    // --- Final NN Control Run ---
    current_episode = RL_MAX_EPISODES + 1;
    init_game_state();
    double sim_time = 0.0;
    int step_count = 0;
    
    while (state.ship.is_alive && !state.ship.has_landed && sim_time < 500.0) {
        update_game(SIMULATION_DT, false); // false for final run (no logging/storage)
        sim_time += SIMULATION_DT;
        step_count++;
        
        if (step_count % PRINT_INTERVAL == 0) {
            print_state(sim_time, current_episode);
        }
    }
    
    print_state(sim_time, current_episode);

    // --- Cleanup ---
    matrix_free(nn.weights_ih);
    matrix_free(nn.weights_ho);
    free(nn.bias_h);
    free(nn.bias_o);
    printf("Simulation finished and memory cleaned up.\n");
    return 0;
}