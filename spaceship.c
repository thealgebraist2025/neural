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

// Tunnel & Ship Constants
#define TUNNEL_WIDTH 100.0
#define SHIP_FIXED_X (CANVAS_WIDTH / 2.0)
#define LANDING_PAD_Y (CANVAS_HEIGHT - GROUND_HEIGHT)

// Physics Constants
#define GRAVITY 0.05
#define THRUST_POWER 0.35 // Increased thrust for 1D control
#define MAX_VELOCITY_FOR_NORM 15.0
#define CRITICAL_LANDING_VELOCITY 1.0 // Max vertical speed for safe landing

// Dynamic Obstacle Constants
#define OBSTACLE_VELOCITY 3.0 // Horizontal speed of obstacles

// NN & RL Constants
// Input Size (7): Y, Vy, Diamond Y-Dist, Obs Y-Dist, Obs X-Pos, Obs Width, Collected Count
#define NN_INPUT_SIZE 7 
#define NN_HIDDEN_SIZE 32
#define NN_OUTPUT_SIZE 2 // 0:Thrust Up, 1:No Thrust
#define NN_LEARNING_RATE 0.005
#define GAMMA 0.99 // Discount factor for future rewards

// Game Constants
#define NUM_OBSTACLES 5
#define NUM_DIAMONDS 10
#define SIMULATION_DT (1.0 / 60.0) // 60 FPS simulation step

// Reward Goals and Values
#define REWARD_PER_STEP -0.01 
#define REWARD_CRASH -200.0
#define REWARD_SAFE_LAND 50.0
#define REWARD_SAFE_LAND_ALL_COLLECTED 500.0
#define REWARD_COLLECT_DIAMOND 50.0
#define REWARD_VELOCITY_PENALTY_SCALE -0.2 
#define REWARD_OBSTACLE_PROXIMITY_SCALE -5.0 
#define REWARD_DIAMOND_PROGRESS_SCALE 0.05 

// Action History Buffer
#define ACTION_HISTORY_SIZE 15 
const char* action_names[NN_OUTPUT_SIZE] = {"THRUST", "PASSIVE"};

// --- Data Structures ---

typedef struct { double x, y; } Vector2D;
typedef struct { double x, y; double w, h; double vx; } Obstacle;
typedef struct { double x, y; double size; bool collected; } Diamond;
typedef struct { double x, y; double vy; double size; bool is_thrusting; bool is_alive; bool has_landed; } Ship;
typedef struct { int score; int total_diamonds; Ship ship; Obstacle obstacles[NUM_OBSTACLES]; Diamond diamonds[NUM_DIAMONDS]; } GameState;

// Matrix Structures
typedef struct { int rows; int cols; double** data; } Matrix;

// RL Step Structure
typedef struct { double input[NN_INPUT_SIZE]; int action_index; double reward; } EpisodeStep;

// RL Episode Buffer
typedef struct { EpisodeStep steps[4000]; int count; double total_score; } Episode;

// Neural Network Structure
typedef struct { Matrix weights_ih; Matrix weights_ho; double* bias_h; double* bias_o; double lr; } NeuralNetwork;


// --- Global State ---
GameState state;
NeuralNetwork nn;
Episode episode_buffer;
int current_episode = 0;
int action_history[ACTION_HISTORY_SIZE];
int action_history_idx = 0; 

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
double sigmoid(double x) {
    return check_double(1.0 / (1.0 + exp(-x)), "sigmoid_output", "sigmoid");
}
double sigmoid_derivative(double y) {
    double result = y * (1.0 - y);
    return check_double(result, "sigmoid_deriv_output", "sigmoid_derivative");
}
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
double log_val(double x) {
    if (x <= 1e-10) return log(1e-10);
    return log(x);
}

// --- Matrix Functions ---

Matrix matrix_create(int rows, int cols) {
    Matrix m; m.rows = rows; m.cols = cols;
    m.data = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        m.data[i] = (double*)malloc(cols * sizeof(double));
        for (int j = 0; j < cols; j++) {
            // Xavier/He initialization heuristic
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

// NN Forward Pass (Policy/Action Selection) - Logits and Probs
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
    
    // --- 1. Feedforward (Calculates intermediates) ---
    Matrix hidden = matrix_dot(nn->weights_ih, inputs);
    for (int i = 0; i < NN_HIDDEN_SIZE; i++) hidden.data[i][0] += nn->bias_h[i];
    Matrix hidden_output = matrix_map(hidden, sigmoid);

    Matrix output_logits_m = matrix_dot(nn->weights_ho, hidden_output);
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) output_logits_m.data[i][0] += nn->bias_o[i];
    
    double logits[NN_OUTPUT_SIZE];
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) logits[i] = output_logits_m.data[i][0];
    double probs[NN_OUTPUT_SIZE];
    softmax(logits, probs, NN_OUTPUT_SIZE);

    // --- 2. Calculate Output Gradient (dLoss/dLogits) ---
    Matrix output_gradients = matrix_create(NN_OUTPUT_SIZE, 1);
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
        double target = (i == action_index) ? 1.0 : 0.0;
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

    // Ship setup (Fixed X, only Y and Vy matter)
    state.ship.x = SHIP_FIXED_X;
    state.ship.y = CANVAS_HEIGHT / 2.0; // Start in the middle of the tunnel
    state.ship.vy = 0.0;
    state.ship.size = 15.0;
    state.ship.is_thrusting = false;
    state.ship.is_alive = true;
    state.ship.has_landed = false;
    
    // Reset episode buffer
    episode_buffer.count = 0;
    episode_buffer.total_score = 0;

    // Dynamic Obstacles
    srand((unsigned int)time(NULL) * (current_episode + 1)); 
    for (int i = 0; i < NUM_OBSTACLES; i++) {
        // Obstacles span the tunnel width (w) but move horizontally (x)
        state.obstacles[i].x = check_double(((double)rand() / RAND_MAX) * (CANVAS_WIDTH - 200.0) + 100.0, "obs_x", "init_game"); // Start within a wider range
        state.obstacles[i].y = check_double(((double)rand() / RAND_MAX) * (CANVAS_HEIGHT - GROUND_HEIGHT - 100.0) + 50.0, "obs_y", "init_game");
        state.obstacles[i].w = 100.0 + ((double)rand() / RAND_MAX) * 50.0; // Variable width
        state.obstacles[i].h = 20.0;
        // Alternate directions and speed variation
        double speed_sign = (i % 2 == 0) ? 1.0 : -1.0;
        state.obstacles[i].vx = check_double(speed_sign * (OBSTACLE_VELOCITY + ((double)rand() / RAND_MAX) * 1.0), "obs_vx", "init_game");
    }

    // Diamonds
    for (int i = 0; i < NUM_DIAMONDS; i++) {
        state.diamonds[i].x = SHIP_FIXED_X; // Fixed X at tunnel center
        // Spread diamonds vertically
        state.diamonds[i].y = check_double(((double)rand() / RAND_MAX) * (CANVAS_HEIGHT - GROUND_HEIGHT - 100.0) + 50.0, "diamond_y", "init_game");
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
    return NN_OUTPUT_SIZE - 1; // Fallback
}

void get_state_features(double* input) {
    Ship* ship = &state.ship;
    
    // --- 1. Ship State Features (2) ---
    input[0] = check_double(ship->y / CANVAS_HEIGHT, "norm_y", "get_state_features"); // Normalized Y position
    input[1] = check_double(ship->vy / MAX_VELOCITY_FOR_NORM, "norm_vy", "get_state_features"); // Normalized Vertical Velocity
    
    // --- 2. Nearest Diamond Features (1) ---
    Diamond* nearest_diamond = NULL;
    double min_diamond_y_dist = INFINITY;
    for (int i = 0; i < NUM_DIAMONDS; i++) {
        Diamond* d = &state.diamonds[i];
        if (!d->collected) {
            double y_dist = check_double(ship->y - d->y, "d_y_dist", "get_state_features");
            if (fabs(y_dist) < fabs(min_diamond_y_dist)) { min_diamond_y_dist = y_dist; nearest_diamond = d; }
        }
    }
    // Normalized Y distance (positive if diamond is below, negative if above)
    input[2] = check_double(min_diamond_y_dist / CANVAS_HEIGHT, "norm_d_y_dist", "get_state_features");
    
    // --- 3. Nearest Obstacle Features (3) ---
    Obstacle* nearest_obs = NULL;
    double min_obs_y_dist = INFINITY;
    for (int i = 0; i < NUM_OBSTACLES; i++) {
        Obstacle* obs = &state.obstacles[i];
        double obs_center_y = obs->y + obs->h / 2.0;
        double y_dist = check_double(ship->y - obs_center_y, "o_y_dist", "get_state_features");
        if (fabs(y_dist) < fabs(min_obs_y_dist)) { min_obs_y_dist = y_dist; nearest_obs = obs; }
    }
    
    // Normalized Y distance to nearest obstacle (signed)
    input[3] = check_double(min_obs_y_dist / CANVAS_HEIGHT, "norm_o_y_dist", "get_state_features"); 
    // Normalized X position of the nearest obstacle center
    input[4] = nearest_obs ? check_double((nearest_obs->x + nearest_obs->w / 2.0) / CANVAS_WIDTH, "norm_o_x", "get_state_features") : 0.5;
    // Normalized nearest obstacle width
    input[5] = nearest_obs ? check_double(nearest_obs->w / CANVAS_WIDTH, "norm_o_w", "get_state_features") : 0.0;

    // --- 4. All Collected Flag (1) ---
    input[6] = (double)state.total_diamonds / NUM_DIAMONDS;

    for (int i = 0; i < NN_INPUT_SIZE; i++) {
        if (input[i] > 1.0) input[i] = 1.0;
        if (input[i] < -1.0) input[i] = -1.0;
        check_double(input[i], "input_val", "get_state_features_final");
    }
}

double calculate_reward(Ship* ship, double old_min_diamond_y_dist, int diamonds_collected_this_step) {
    double reward = REWARD_PER_STEP; // Base step penalty
    
    if (!ship->is_alive) return REWARD_CRASH;
    
    // STABILIZE: Negative reward based on speed magnitude (only Vy matters)
    double speed_magnitude_sq = check_double(ship->vy * ship->vy, "speed_sq", "calc_reward");
    reward += REWARD_VELOCITY_PENALTY_SCALE * speed_magnitude_sq;
    
    // Get normalized state for obstacle/diamond distances
    double input[NN_INPUT_SIZE];
    get_state_features(input);
    
    // Check for imminent obstacle collision (blocking the tunnel)
    for (int i = 0; i < NUM_OBSTACLES; i++) {
        Obstacle* obs = &state.obstacles[i];
        
        // Check if the ship's Y is aligned with the obstacle's Y
        if (ship->y + ship->size > obs->y && ship->y - ship->size < obs->y + obs->h) {
            // Check if the obstacle is currently blocking the center line (SHIP_FIXED_X)
            bool is_blocking = (SHIP_FIXED_X >= obs->x && SHIP_FIXED_X <= obs->x + obs->w);
            
            if (is_blocking) {
                // Penalize based on how close the ship is vertically to the obstacle
                double y_dist = fabs(ship->y - (obs->y + obs->h / 2.0));
                double proximity_factor = 1.0 - fmin(1.0, y_dist / 50.0); // Close proximity (e.g., < 50px)
                reward += REWARD_OBSTACLE_PROXIMITY_SCALE * proximity_factor;
            }
        }
    }
    
    // TOWARDS_DIAMOND: Reward collection
    if (diamonds_collected_this_step > 0) {
        reward += REWARD_COLLECT_DIAMOND * diamonds_collected_this_step;
    }

    // TOWARDS_DIAMOND: Small reward for progress if diamonds remain (closer in Y means better)
    if (state.total_diamonds < NUM_DIAMONDS) {
        double new_min_diamond_y_dist = input[2] * CANVAS_HEIGHT;
        // Progress is reduction in absolute distance
        double distance_change = fabs(old_min_diamond_y_dist) - fabs(new_min_diamond_y_dist);
        if (distance_change > 0) {
            reward += REWARD_DIAMOND_PROGRESS_SCALE * distance_change / 10.0;
        }
    }
    
    // LANDING: Final Goal
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
    ship->is_thrusting = false; 
    
    if (action_index == 0) { // Thrust Up
        ship->is_thrusting = true;
    } else if (action_index == 1) { // No Thrust / Passive
        ship->is_thrusting = false;
    }
    
    // Record action for stats
    action_history[action_history_idx] = action_index;
    action_history_idx = (action_history_idx + 1) % ACTION_HISTORY_SIZE;
}

void apply_thrust(Ship* ship, double dt) {
    if (!ship->is_thrusting) return;
    
    // Thrust is applied upward (negative Y direction)
    double thrust_force = check_double(THRUST_POWER * dt / SIMULATION_DT, "thrust_force", "apply_thrust");
    ship->vy = check_double(ship->vy - thrust_force, "ship_vy", "apply_thrust");
}

void update_physics(Ship* ship, double dt) {
    // Apply Gravity (always down, positive y is down)
    ship->vy = check_double(ship->vy + GRAVITY * dt / SIMULATION_DT, "ship_vy_grav", "update_physics");

    // Update Y position
    ship->y = check_double(ship->y + ship->vy * dt, "ship_y", "update_physics");

    // Dynamic Obstacle Movement (Horizontal)
    for (int i = 0; i < NUM_OBSTACLES; i++) {
        Obstacle* obs = &state.obstacles[i];
        obs->x += obs->vx * dt;

        // Reverse direction upon hitting canvas edges
        if (obs->x < -obs->w || obs->x + obs->w > CANVAS_WIDTH + obs->w) {
            obs->vx *= -1.0;
            // Prevent sticking to the edge
            if (obs->x < -obs->w) obs->x = -obs->w;
            if (obs->x + obs->w > CANVAS_WIDTH + obs->w) obs->x = CANVAS_WIDTH - obs->w;
        }
    }
}

int check_collision(Ship* ship) {
    double ship_radius = ship->size; 
    int diamonds_collected_this_step = 0;
    
    // --- 1. Obstacle collision (if ship's Y is aligned AND obstacle blocks the tunnel center) ---
    for (int i = 0; i < NUM_OBSTACLES; i++) {
        Obstacle* obs = &state.obstacles[i];
        
        // Vertical check: is the ship's vertical profile overlapping the obstacle's vertical profile?
        bool y_overlap = (ship->y + ship->size > obs->y && ship->y - ship->size < obs->y + obs->h);

        // Horizontal check: is the obstacle's X range blocking the ship's fixed X?
        bool x_blocking = (SHIP_FIXED_X >= obs->x && SHIP_FIXED_X <= obs->x + obs->w);

        if (y_overlap && x_blocking) {
            ship->is_alive = false;
            return diamonds_collected_this_step; // Crash overrides diamond collection
        }
    }
    
    // --- 2. Diamond collection (Only Y proximity matters, X is fixed) ---
    for (int i = 0; i < NUM_DIAMONDS; i++) {
        Diamond* d = &state.diamonds[i];
        if (d->collected) continue;
        
        double dy = check_double(ship->y - d->y, "diamond_dy_coll", "check_collision");
        double distance = check_double(fabs(dy), "diamond_dist", "check_collision");

        if (distance < ship_radius + d->size) {
            d->collected = true;
            state.total_diamonds++;
            diamonds_collected_this_step++;
            state.score += (int)REWARD_COLLECT_DIAMOND;
        }
    }
    
    // --- 3. Screen Bounds ---
    // Upper boundary
    if (ship->y < ship->size) {
        ship->y = ship->size;
        ship->vy = 0.0;
    }

    return diamonds_collected_this_step;
}

void print_episode_stats(double sim_time, double train_time_ms) {
    Ship* ship = &state.ship;
    
    printf("====================================================\n");
    printf("EPISODE %d SUMMARY\n", current_episode);
    printf("----------------------------------------------------\n");
    printf("Termination Status: %s\n", 
        !ship->is_alive ? "CRASHED" : 
        (ship->has_landed && state.total_diamonds == NUM_DIAMONDS) ? "SUCCESSFUL LANDING (ALL DIAMONDS)" : 
        ship->has_landed ? "SAFE LANDING (PARTIAL DIAMONDS)" : 
        "TIMEOUT (Max Steps)");
    
    printf("Simulation Time: %.2fs (Steps: %d)\n", sim_time, episode_buffer.count);
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

void update_game(double dt, bool is_training_run) {
    Ship* ship = &state.ship;
    
    if (!ship->is_alive || ship->has_landed) return; 

    // --- 1. Get current state features and previous diamond proximity ---
    double input[NN_INPUT_SIZE];
    get_state_features(input);
    // input[2] holds the normalized Y distance to the nearest uncollected diamond
    double old_min_diamond_y_dist = input[2] * CANVAS_HEIGHT; 

    // --- 2. Feedforward and Action Selection ---
    double logit_output[NN_OUTPUT_SIZE];
    double probabilities[NN_OUTPUT_SIZE];
    nn_policy_forward(&nn, input, probabilities, logit_output);

    int action_index = select_action(probabilities);

    // --- 3. Apply Action, Physics, and Collision ---
    apply_action(ship, action_index);
    apply_thrust(ship, dt);
    update_physics(ship, dt);

    int diamonds_collected = check_collision(ship);
    
    // --- 4. Check Landing ---
    double ground_y = LANDING_PAD_Y;
    if (ship->y + ship->size > ground_y) {
        ship->y = ground_y - ship->size;
        
        // Landing only requires low vertical speed
        if (fabs(ship->vy) > CRITICAL_LANDING_VELOCITY) {
            ship->is_alive = false; // Crash due to high speed
        } else {
            ship->has_landed = true; // Safe landing
        }
        
        ship->vy = 0.0;
        ship->is_thrusting = false;
    }

    // --- 5. Calculate Reward and Store Step ---
    double reward = calculate_reward(ship, old_min_diamond_y_dist, diamonds_collected);
    
    if (is_training_run && episode_buffer.count < 4000) {
        EpisodeStep step;
        for(int i = 0; i < NN_INPUT_SIZE; i++) step.input[i] = input[i];
        step.action_index = action_index;
        step.reward = reward;
        
        episode_buffer.steps[episode_buffer.count] = step;
        episode_buffer.count++;
        episode_buffer.total_score += reward;
    }
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

    // --- 3. Train the Network ---
    for (int i = 0; i < episode_buffer.count; i++) {
        // Compute Advantage (A_t = G_t - V(s_t)). Here, we use G_t as the Advantage with mean/std baseline.
        double Gt = (returns[i] - mean_return) / std_dev; 
        
        // The policy gradient update uses -Gt as the magnitude of the error to encourage the action
        // (This is based on the negative log-likelihood loss function)
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
        action_history[i] = 1; // Default to PASSIVE
    }

    printf("--- RL 1D Tunnel Lander Simulation (REINFORCE) ---\n");
    printf("Training will run for 4 minutes (240 seconds).\n");
    
    time_t start_time = time(NULL);
    const int TIME_LIMIT_SECONDS = 240; 
    
    // --- Time-Limited Training Phase ---
    while (time(NULL) - start_time < TIME_LIMIT_SECONDS) {
        
        current_episode++;
        init_game_state();
        double sim_time = 0.0;

        // Play episode
        while (state.ship.is_alive && !state.ship.has_landed && episode_buffer.count < 4000) {
            update_game(SIMULATION_DT, true); // true for training run
            sim_time += SIMULATION_DT;
        }

        // Train and Time it
        clock_t train_start = clock();
        run_reinforce_training(); 
        clock_t train_end = clock();
        
        double train_time_ms = (double)(train_end - train_start) * 1000.0 / CLOCKS_PER_SEC;

        // Print stats
        print_episode_stats(sim_time, train_time_ms);
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