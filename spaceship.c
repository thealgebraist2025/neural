This C99 program simulates the spacecraft and neural network logic without a graphical user interface. It performs the full physics simulation and training logic, and prints the requested state information every 2 simulated seconds.
It includes the mandatory check_nan_and_stop function that wraps every critical floating-point calculation.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

// --- Global Constants ---
#define CANVAS_WIDTH 800.0
#define CANVAS_HEIGHT 600.0
#define GROUND_HEIGHT 50.0

#define GRAVITY 0.05
#define THRUST_POWER 0.15
#define ROTATION_SPEED 0.05
#define MAX_SAFE_VELOCITY 1.0
#define CRITICAL_VELOCITY 3.0
#define SCORE_LANDING 100
#define SCORE_DIAMOND 5

#define NN_INPUT_SIZE 10
#define NN_HIDDEN_SIZE 16
#define NN_OUTPUT_SIZE 3
#define NN_LEARNING_RATE 0.01
#define NN_MAX_PLAYTHROUGHS 50

#define NUM_OBSTACLES 5
#define NUM_DIAMONDS 10
#define SIMULATION_DT (1.0 / 60.0) // 60 FPS simulation step
#define PRINT_INTERVAL 2.0 // Print state every 2 simulated seconds

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
    int playthroughs;
    Ship ship;
    Obstacle obstacles[NUM_OBSTACLES];
    Diamond diamonds[NUM_DIAMONDS];
} GameState;

typedef struct {
    int rows;
    int cols;
    double** data;
} Matrix;

typedef struct {
    Matrix weights_ih;
    Matrix weights_ho;
    double* bias_h;
    double* bias_o;
    double lr;
} NeuralNetwork;

typedef struct {
    double input[NN_INPUT_SIZE];
    double target_output[NN_OUTPUT_SIZE];
} DataPoint;

typedef struct {
    DataPoint data[NN_MAX_PLAYTHROUGHS * 2000]; // Max data points approximation
    int count;
} DataCollector;

// --- Global State ---
GameState state;
NeuralNetwork nn;
DataCollector collector;

// --- Matrix Operations (Memory Managed) ---

Matrix matrix_create(int rows, int cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.data = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        m.data[i] = (double*)malloc(cols * sizeof(double));
        for (int j = 0; j < cols; j++) {
            m.data[i][j] = check_double((((double)rand() / RAND_MAX) * 2.0 - 1.0) * 0.01, "rand_val", "matrix_create");
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

// Element-wise addition/subtraction
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
            check_nan_and_stop(result.data[i][j], "result_val", "matrix_add_subtract");
        }
    }
    return result;
}

// Element-wise multiplication (Hadamard product)
Matrix matrix_multiply_elem(Matrix A, Matrix B) {
    if (A.rows != B.rows || A.cols != B.cols) {
        fprintf(stderr, "Multiply (element-wise) dimension mismatch.\n");
        exit(EXIT_FAILURE);
    }
    Matrix result = matrix_create(A.rows, A.cols);
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            result.data[i][j] = check_double(A.data[i][j], "A[i][j]", "matrix_multiply_elem") * check_double(B.data[i][j], "B[i][j]", "matrix_multiply_elem");
            check_nan_and_stop(result.data[i][j], "result_val", "matrix_multiply_elem");
        }
    }
    return result;
}

// Scalar multiplication
Matrix matrix_multiply_scalar(Matrix A, double scalar) {
    check_double(scalar, "scalar", "matrix_multiply_scalar");
    Matrix result = matrix_create(A.rows, A.cols);
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            result.data[i][j] = check_double(A.data[i][j], "A[i][j]", "matrix_multiply_scalar") * scalar;
            check_nan_and_stop(result.data[i][j], "result_val", "matrix_multiply_scalar");
        }
    }
    return result;
}

// --- NN Activation Functions ---

double sigmoid(double x) {
    check_double(x, "sigmoid_input", "sigmoid");
    double result = 1.0 / (1.0 + exp(-x));
    return check_double(result, "sigmoid_output", "sigmoid");
}

double sigmoid_derivative(double y) {
    check_double(y, "sigmoid_deriv_input", "sigmoid_derivative");
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

double* matrix_to_array(Matrix m, int* size) {
    if (m.cols != 1) {
        fprintf(stderr, "Matrix to array requires a column matrix.\n");
        exit(EXIT_FAILURE);
    }
    *size = m.rows;
    double* arr = (double*)malloc(m.rows * sizeof(double));
    for (int i = 0; i < m.rows; i++) {
        arr[i] = m.data[i][0];
    }
    return arr;
}

void nn_feedforward(NeuralNetwork* nn, const double* input_array, double* output_array) {
    Matrix inputs = array_to_matrix(input_array, NN_INPUT_SIZE);

    // Hidden layer
    Matrix hidden = matrix_dot(nn->weights_ih, inputs);
    for (int i = 0; i < NN_HIDDEN_SIZE; i++) {
        hidden.data[i][0] = check_double(hidden.data[i][0], "hidden_val", "nn_feedforward") + check_double(nn->bias_h[i], "bias_h", "nn_feedforward");
    }
    Matrix hidden_output = matrix_map(hidden, sigmoid);

    // Output layer
    Matrix output = matrix_dot(nn->weights_ho, hidden_output);
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
        output.data[i][0] = check_double(output.data[i][0], "output_val", "nn_feedforward") + check_double(nn->bias_o[i], "bias_o", "nn_feedforward");
    }
    Matrix final_output = matrix_map(output, sigmoid);

    for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
        output_array[i] = final_output.data[i][0];
    }

    matrix_free(inputs);
    matrix_free(hidden);
    matrix_free(hidden_output);
    matrix_free(output);
    matrix_free(final_output);
}

double nn_train(NeuralNetwork* nn, const double* input_array, const double* target_array) {
    // 1. Feedforward (calculates intermediate matrices)
    Matrix inputs = array_to_matrix(input_array, NN_INPUT_SIZE);
    Matrix hidden = matrix_dot(nn->weights_ih, inputs);
    for (int i = 0; i < NN_HIDDEN_SIZE; i++) {
        hidden.data[i][0] = check_double(hidden.data[i][0], "hidden_val", "nn_train") + check_double(nn->bias_h[i], "bias_h", "nn_train");
    }
    Matrix hidden_output = matrix_map(hidden, sigmoid);
    Matrix output = matrix_dot(nn->weights_ho, hidden_output);
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
        output.data[i][0] = check_double(output.data[i][0], "output_val", "nn_train") + check_double(nn->bias_o[i], "bias_o", "nn_train");
    }
    Matrix final_output = matrix_map(output, sigmoid);

    // 2. Output Error
    Matrix targets = array_to_matrix(target_array, NN_OUTPUT_SIZE);
    Matrix output_errors = matrix_add_subtract(targets, final_output, false); // Targets - Output

    // 3. Calculate Output Gradients and Delta Weights HO
    Matrix gradients = matrix_map(final_output, sigmoid_derivative);
    Matrix gradients_mul = matrix_multiply_elem(gradients, output_errors);
    Matrix gradients_scaled = matrix_multiply_scalar(gradients_mul, nn->lr);

    Matrix hidden_output_T = matrix_transpose(hidden_output);
    Matrix delta_weights_ho = matrix_dot(gradients_scaled, hidden_output_T);

    // 4. Update Weights HO and Bias O
    Matrix new_weights_ho = matrix_add_subtract(nn->weights_ho, delta_weights_ho, true);
    matrix_free(nn->weights_ho); nn->weights_ho = new_weights_ho;
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
        nn->bias_o[i] = check_double(nn->bias_o[i], "bias_o", "nn_train_upd") + check_double(gradients_scaled.data[i][0], "grad_scaled_o", "nn_train_upd");
    }

    // 5. Calculate Hidden Errors and Delta Weights IH
    Matrix weights_ho_T = matrix_transpose(nn->weights_ho);
    Matrix hidden_errors = matrix_dot(weights_ho_T, output_errors);

    Matrix hidden_gradients = matrix_map(hidden_output, sigmoid_derivative);
    Matrix hidden_gradients_mul = matrix_multiply_elem(hidden_gradients, hidden_errors);
    Matrix hidden_gradients_scaled = matrix_multiply_scalar(hidden_gradients_mul, nn->lr);

    Matrix inputs_T = matrix_transpose(inputs);
    Matrix delta_weights_ih = matrix_dot(hidden_gradients_scaled, inputs_T);

    // 6. Update Weights IH and Bias H
    Matrix new_weights_ih = matrix_add_subtract(nn->weights_ih, delta_weights_ih, true);
    matrix_free(nn->weights_ih); nn->weights_ih = new_weights_ih;
    for (int i = 0; i < NN_HIDDEN_SIZE; i++) {
        nn->bias_h[i] = check_double(nn->bias_h[i], "bias_h", "nn_train_upd") + check_double(hidden_gradients_scaled.data[i][0], "grad_scaled_h", "nn_train_upd");
    }

    // 7. Calculate total squared error
    double error_sum = 0.0;
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
        double error = output_errors.data[i][0];
        error_sum += check_double(error, "error_val", "nn_train_err") * check_double(error, "error_val", "nn_train_err");
    }

    // 8. Cleanup
    matrix_free(inputs); matrix_free(hidden); matrix_free(hidden_output);
    matrix_free(output); matrix_free(final_output); matrix_free(targets);
    matrix_free(output_errors); matrix_free(gradients); matrix_free(gradients_mul);
    matrix_free(gradients_scaled); matrix_free(hidden_output_T); matrix_free(delta_weights_ho);
    matrix_free(weights_ho_T); matrix_free(hidden_errors); matrix_free(hidden_gradients);
    matrix_free(hidden_gradients_mul); matrix_free(hidden_gradients_scaled); matrix_free(inputs_T);
    matrix_free(delta_weights_ih);

    return check_double(error_sum, "final_error_sum", "nn_train");
}

// --- Game Logic Functions ---

void init_game_state(bool start_fresh) {
    if (start_fresh) state.score = 0;
    
    state.ship.x = CANVAS_WIDTH / 2.0;
    state.ship.y = 50.0;
    state.ship.velocity.x = 0.0;
    state.ship.velocity.y = 0.0;
    state.ship.angle = M_PI / 2.0;
    state.ship.size = 15.0;
    state.ship.is_thrusting = false;
    state.ship.is_alive = true;
    state.ship.has_landed = false;
    
    // Obstacles
    state.obstacles[0] = (Obstacle){100, 300, 150, 20};
    state.obstacles[1] = (Obstacle){550, 200, 100, 20};
    state.obstacles[2] = (Obstacle){300, 450, 200, 20};
    state.obstacles[3] = (Obstacle){50, 150, 50, 50};
    state.obstacles[4] = (Obstacle){650, 400, 150, 20};

    // Diamonds
    srand((unsigned int)time(NULL) * (state.playthroughs + 1)); 
    for (int i = 0; i < NUM_DIAMONDS; i++) {
        state.diamonds[i].x = check_double(((double)rand() / RAND_MAX) * (CANVAS_WIDTH - 40.0) + 20.0, "diamond_x", "init_game");
        state.diamonds[i].y = check_double(((double)rand() / RAND_MAX) * (CANVAS_HEIGHT - GROUND_HEIGHT - 50.0) + 20.0, "diamond_y", "init_game");
        state.diamonds[i].size = 5.0;
        state.diamonds[i].collected = false;
    }
}

void get_nearest_features(Diamond** nearest_diamond, Obstacle** nearest_obs) {
    double min_diamond_dist_sq = INFINITY;
    double min_obs_dist_sq = INFINITY;
    *nearest_diamond = NULL;
    *nearest_obs = NULL;

    for (int i = 0; i < NUM_DIAMONDS; i++) {
        Diamond* d = &state.diamonds[i];
        if (!d->collected) {
            double dx = check_double(state.ship.x - d->x, "diamond_dx", "get_nearest_features");
            double dy = check_double(state.ship.y - d->y, "diamond_dy", "get_nearest_features");
            double dist_sq = check_double(dx * dx + dy * dy, "diamond_dist_sq", "get_nearest_features");
            if (dist_sq < min_diamond_dist_sq) {
                min_diamond_dist_sq = dist_sq;
                *nearest_diamond = d;
            }
        }
    }

    for (int i = 0; i < NUM_OBSTACLES; i++) {
        Obstacle* obs = &state.obstacles[i];
        double obs_center_x = obs->x + obs->w / 2.0;
        double obs_center_y = obs->y + obs->h / 2.0;
        double dx = check_double(state.ship.x - obs_center_x, "obs_dx", "get_nearest_features");
        double dy = check_double(state.ship.y - obs_center_y, "obs_dy", "get_nearest_features");
        double dist_sq = check_double(dx * dx + dy * dy, "obs_dist_sq", "get_nearest_features");
        if (dist_sq < min_obs_dist_sq) {
            min_obs_dist_sq = dist_sq;
            *nearest_obs = obs;
        }
    }
}

void normalize_input(const Ship* ship, const Diamond* nearest_diamond, const Obstacle* nearest_obs, double* input) {
    const double max_v = 10.0;
    input[0] = check_double(ship->x / CANVAS_WIDTH, "norm_x", "normalize_input");
    input[1] = check_double(ship->y / CANVAS_HEIGHT, "norm_y", "normalize_input"); 
    input[2] = check_double(ship->velocity.x / max_v, "norm_vx", "normalize_input"); 
    input[3] = check_double(ship->velocity.y / max_v, "norm_vy", "normalize_input");
    input[4] = check_double(ship->angle / (2.0 * M_PI), "norm_angle", "normalize_input");
    
    input[5] = nearest_diamond ? check_double(nearest_diamond->x / CANVAS_WIDTH, "norm_d_x", "normalize_input") : 0.5; 
    input[6] = nearest_diamond ? check_double(nearest_diamond->y / CANVAS_HEIGHT, "norm_d_y", "normalize_input") : 0.5;
    
    input[7] = nearest_obs ? check_double(nearest_obs->x / CANVAS_WIDTH, "norm_o_x", "normalize_input") : 0.5; 
    input[8] = nearest_obs ? check_double(nearest_obs->y / CANVAS_HEIGHT, "norm_o_y", "normalize_input") : 0.5;
    input[9] = nearest_obs ? check_double((nearest_obs->w + nearest_obs->h) / (CANVAS_WIDTH + CANVAS_HEIGHT), "norm_o_size", "normalize_input") : 0.0;
}

void get_optimal_action(const Ship* ship, double* target_output) {
    target_output[0] = 0.0; target_output[1] = 0.0; target_output[2] = 0.0;

    Diamond* nearest_diamond = NULL;
    double min_distance_sq = INFINITY;
    for (int i = 0; i < NUM_DIAMONDS; i++) {
        Diamond* d = &state.diamonds[i];
        if (!d->collected) {
            double dx = check_double(ship->x - d->x, "dx_opt", "get_optimal_action");
            double dy = check_double(d->y - ship->y, "dy_opt", "get_optimal_action");
            double dist_sq = check_double(dx * dx + dy * dy, "dist_sq_opt", "get_optimal_action");
            if (dist_sq < min_distance_sq) { min_distance_sq = dist_sq; nearest_diamond = d; }
        }
    }

    if (nearest_diamond) {
        double dx = check_double(nearest_diamond->x - ship->x, "dx_target", "get_optimal_action");
        double dy = check_double(nearest_diamond->y - ship->y, "dy_target", "get_optimal_action");
        double target_heading = check_double(atan2(dy, dx), "target_heading", "get_optimal_action");
        
        double angle_diff = check_double(target_heading - ship->angle, "angle_diff", "get_optimal_action");
        if (angle_diff > M_PI) angle_diff -= check_double(2.0 * M_PI, "2PI", "get_optimal_action");
        if (angle_diff < -M_PI) angle_diff += check_double(2.0 * M_PI, "2PI", "get_optimal_action");

        if (angle_diff > 0.1) target_output[2] = 1.0; 
        else if (angle_diff < -0.1) target_output[1] = 1.0; 

        if (fabs(angle_diff) < M_PI / 3.0) { target_output[0] = 1.0; } 
    }
}

void apply_nn_output(Ship* ship, const double* output) {
    ship->is_thrusting = output[0] > 0.5;
    bool rotate_left = output[1] > 0.5;
    bool rotate_right = output[2] > 0.5;
    
    if (rotate_left && !rotate_right) { ship->angle = check_double(ship->angle - ROTATION_SPEED, "ship_angle_L", "apply_nn_output"); } 
    else if (rotate_right && !rotate_left) { ship->angle = check_double(ship->angle + ROTATION_SPEED, "ship_angle_R", "apply_nn_output"); } 
    
    if (ship->angle > 2.0 * M_PI) ship->angle = check_double(ship->angle - 2.0 * M_PI, "ship_angle_norm", "apply_nn_output");
    if (ship->angle < 0.0) ship->angle = check_double(ship->angle + 2.0 * M_PI, "ship_angle_norm", "apply_nn_output");
}

void apply_thrust(Ship* ship, double dt) {
    if (!ship->is_thrusting) return;
    
    double thrust_force = check_double(THRUST_POWER * dt / SIMULATION_DT, "thrust_force", "apply_thrust");

    ship->velocity.x = check_double(ship->velocity.x + cos(ship->angle) * thrust_force, "ship_vx", "apply_thrust");
    ship->velocity.y = check_double(ship->velocity.y + sin(ship->angle) * thrust_force, "ship_vy", "apply_thrust");
}

void update_physics(Ship* ship, double dt) {
    ship->velocity.y = check_double(ship->velocity.y + GRAVITY * dt / SIMULATION_DT, "ship_vy_grav", "update_physics");

    ship->x = check_double(ship->x + ship->velocity.x * dt, "ship_x", "update_physics");
    ship->y = check_double(ship->y + ship->velocity.y * dt, "ship_y", "update_physics");

    // Wrap around horizontal edges
    if (ship->x < 0.0) ship->x = CANVAS_WIDTH;
    if (ship->x > CANVAS_WIDTH) ship->x = 0.0;
}

bool check_collision(Ship* ship) {
    double ship_radius = ship->size; 
    bool collision_detected = false;
    double ground_y = CANVAS_HEIGHT - GROUND_HEIGHT;

    // Obstacle collision
    for (int i = 0; i < NUM_OBSTACLES; i++) {
        Obstacle* obs = &state.obstacles[i];
        double closest_x = fmax(obs->x, fmin(ship->x, obs->x + obs->w));
        double closest_y = fmax(obs->y, fmin(ship->y, obs->y + obs->h));
        double dx = check_double(ship->x - closest_x, "coll_dx", "check_collision");
        double dy = check_double(ship->y - closest_y, "coll_dy", "check_collision");

        if (check_double(dx * dx + dy * dy, "coll_dist_sq", "check_collision") < (ship_radius * ship_radius)) {
            ship->is_alive = false;
            collision_detected = true;
            break;
        }
    }
    if (collision_detected) return true;

    // Diamond collection
    for (int i = 0; i < NUM_DIAMONDS; i++) {
        Diamond* d = &state.diamonds[i];
        if (d->collected) continue;
        double dx = check_double(ship->x - d->x, "diamond_dx_coll", "check_collision");
        double dy = check_double(ship->y - d->y, "diamond_dy_coll", "check_collision");
        double distance = check_double(sqrt(dx * dx + dy * dy), "diamond_dist", "check_collision");

        if (distance < ship_radius + d->size) {
            d->collected = true;
            state.score += SCORE_DIAMOND;
        }
    }
    return false;
}

void update_game(double dt) {
    Ship* ship = &state.ship;
    
    if (!ship->is_alive || ship->has_landed) return; 

    double input[NN_INPUT_SIZE];
    double output[NN_OUTPUT_SIZE];
    Diamond* nearest_diamond;
    Obstacle* nearest_obs;

    get_nearest_features(&nearest_diamond, &nearest_obs);
    normalize_input(ship, nearest_diamond, nearest_obs, input);

    // AI Logic
    if (state.playthroughs < NN_MAX_PLAYTHROUGHS) { // Data Collection Phase
        get_optimal_action(ship, output);
        apply_nn_output(ship, output);
        
        // Record data only if not crashed/landed *yet* (state is checked later)
        if (ship->is_alive && !ship->has_landed) {
             if (collector.count < sizeof(collector.data) / sizeof(collector.data[0])) {
                for (int i = 0; i < NN_INPUT_SIZE; i++) collector.data[collector.count].input[i] = input[i];
                for (int i = 0; i < NN_OUTPUT_SIZE; i++) collector.data[collector.count].target_output[i] = output[i];
                collector.count++;
             }
        }
    } else { // NN Control Phase
        nn_feedforward(&nn, input, output);
        apply_nn_output(ship, output);
    }
    
    apply_thrust(ship, dt);
    update_physics(ship, dt);

    if (check_collision(ship)) return; 

    // Ground Check
    double collision_y = CANVAS_HEIGHT - GROUND_HEIGHT;
    if (ship->y + ship->size > collision_y) {
        ship->y = collision_y - ship->size;

        double speed_magnitude = check_double(sqrt(ship->velocity.x * ship->velocity.x + ship->velocity.y * ship->velocity.y), "speed_mag", "update_game");
        
        if (fabs(ship->velocity.y) > CRITICAL_VELOCITY || speed_magnitude > CRITICAL_VELOCITY) {
            ship->is_alive = false;
        } else {
            ship->has_landed = true;
            state.score += SCORE_LANDING;
        }
        
        ship->velocity.x = 0.0;
        ship->velocity.y = 0.0;
        ship->is_thrusting = false;
    }

    if (!ship->is_alive || ship->has_landed) {
        if (state.playthroughs < NN_MAX_PLAYTHROUGHS) {
            state.playthroughs++;
            if (state.playthroughs < NN_MAX_PLAYTHROUGHS) {
                init_game_state(false); // Reset for next data collection run
            }
        }
    }
}

void print_state(double sim_time) {
    Ship* ship = &state.ship;
    double speed_magnitude = check_double(sqrt(ship->velocity.x * ship->velocity.x + ship->velocity.y * ship->velocity.y), "speed_mag_print", "print_state");
    
    printf("--- SIMULATION TIME: %.2fs ---\n", sim_time);
    
    printf("SpaceShip State: %s\n", 
        !ship->is_alive ? "CRASHED" : 
        ship->has_landed ? "LANDED" : 
        (state.playthroughs < NN_MAX_PLAYTHROUGHS ? "DATA COLLECTION" : "NN CONTROL"));
    
    printf("  Position (x, y): (%.2f, %.2f)\n", ship->x, ship->y);
    printf("  Speed (Vx, Vy, Mag): (%.2f, %.2f, %.2f)\n", 
           ship->velocity.x, ship->velocity.y, speed_magnitude);
    printf("  Score: %d\n", state.score);
    
    printf("\nObstacles (x, y, w, h):\n");
    for (int i = 0; i < NUM_OBSTACLES; i++) {
        Obstacle* obs = &state.obstacles[i];
        printf("  Obs %d: (%.0f, %.0f, %.0f, %.0f)\n", i + 1, obs->x, obs->y, obs->w, obs->h);
    }
    
    printf("\nDiamonds (x, y, collected):\n");
    for (int i = 0; i < NUM_DIAMONDS; i++) {
        Diamond* d = &state.diamonds[i];
        printf("  Diamond %d: (%.0f, %.0f, %s)\n", i + 1, d->x, d->y, d->collected ? "Collected" : "Available");
    }
    printf("--------------------------------\n\n");
}

void run_ai_training() {
    printf("Starting AI Sanity Unit Test (Max 2000 Epochs)...\n");
    const double TARGET_ERROR = 0.05;
    double total_error = INFINITY;
    int epoch = 0;
    const int max_epochs = 2000;

    while(total_error > TARGET_ERROR && epoch < max_epochs) {
        double current_epoch_error = 0.0;
        for (int i = 0; i < collector.count; i++) {
            current_epoch_error += nn_train(&nn, collector.data[i].input, collector.data[i].target_output);
        }
        total_error = current_epoch_error / collector.count;
        if (epoch % 200 == 0) {
            printf("[TRAINING] Epoch %d: Avg Error: %.6f\n", epoch, total_error);
        }
        epoch++;
    }

    if (total_error <= TARGET_ERROR) {
        printf("SUCCESS: NN Sanity Test PASSED. Converged in %d epochs (Final Error: %.6f)\n", epoch, total_error);
    } else {
        printf("WARNING: NN Sanity Test FAILED. Did not converge fully (Final Error: %.6f)\n", total_error);
    }
}

void run_final_training() {
    printf("Starting Final Training (5000 Epochs) with %d data points...\n", collector.count);
    const int EPOCHS = 5000;
    double final_error = 0.0;

    for (int e = 0; e < EPOCHS; e++) {
        double current_epoch_error = 0.0;
        for (int i = 0; i < collector.count; i++) {
            current_epoch_error += nn_train(&nn, collector.data[i].input, collector.data[i].target_output);
        }
        final_error = current_epoch_error / collector.count;
        if (e % 1000 == 0) {
            printf("[FINAL TRAIN] Epoch %d: Avg Error: %.6f\n", e, final_error);
        }
    }
    printf("Final training complete. Final Error: %.6f\n", final_error);
}

// --- Main Simulation Loop ---

int main() {
    srand((unsigned int)time(NULL));
    nn_init(&nn);
    init_game_state(true);
    collector.count = 0;

    printf("--- NN Autonomous Lander Simulation (C99) ---\n");
    printf("Data Collection Phase: %d playthroughs\n", NN_MAX_PLAYTHROUGHS);

    double sim_time = 0.0;
    double last_print_time = 0.0;
    
    // Initial print
    print_state(sim_time);

    // Main Simulation Loop
    while (1) {
        // --- Game Step ---
        if (state.playthroughs < NN_MAX_PLAYTHROUGHS) {
            if (state.playthroughs == NN_MAX_PLAYTHROUGHS - 1 && !state.ship.is_alive) {
                 // Last playthrough finished. Break to start training.
                 update_game(SIMULATION_DT); // One final update to handle state change
                 break;
            }
        }
        
        update_game(SIMULATION_DT);

        // --- Time Step and Printing ---
        sim_time += SIMULATION_DT;
        if (sim_time >= last_print_time + PRINT_INTERVAL) {
            print_state(sim_time);
            last_print_time = sim_time;
        }

        // --- End Condition Check for Data Collection ---
        if (state.playthroughs >= NN_MAX_PLAYTHROUGHS && state.playthroughs < NN_MAX_PLAYTHROUGHS + 1) {
            break; 
        }

        // Safety break for extremely long simulations
        if (sim_time > 10000.0 && state.playthroughs < NN_MAX_PLAYTHROUGHS) {
            printf("\nSimulation timed out during data collection.\n");
            break;
        }
        
        if (sim_time > 50000.0) {
             printf("\nSimulation timed out during NN control phase.\n");
             break;
        }
    }

    // --- Training Phase ---
    if (collector.count > 0) {
        // Run AI Sanity Training (Initial 10 playthrough data)
        run_ai_training(); 
        
        // Run Final Training (All 50 playthrough data)
        run_final_training();
    }
    
    // --- Final NN Control Run ---
    if (state.playthroughs >= NN_MAX_PLAYTHROUGHS) {
        printf("\n--- STARTING NN CONTROL SIMULATION ---\n");
        init_game_state(true);
        sim_time = 0.0;
        last_print_time = 0.0;
        
        while (state.ship.is_alive && !state.ship.has_landed && sim_time < 500.0) {
            update_game(SIMULATION_DT);
            sim_time += SIMULATION_DT;
            if (sim_time >= last_print_time + PRINT_INTERVAL) {
                print_state(sim_time);
                last_print_time = sim_time;
            }
        }
        print_state(sim_time);
    } else {
         printf("Not enough data collected for final run.\n");
    }

    // --- Cleanup ---
    matrix_free(nn.weights_ih);
    matrix_free(nn.weights_ho);
    free(nn.bias_h);
    free(nn.bias_o);
    printf("Simulation finished and memory cleaned up.\n");
    return 0;
}

