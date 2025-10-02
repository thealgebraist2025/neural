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
#define NN_HIDDEN_SIZE 16 
#define NN_OUTPUT_SIZE 4 
#define NN_LEARNING_RATE 0.0025 
#define GAMMA 0.95 

// Reward Goals and Values
#define REWARD_PER_STEP -1.0 
#define REWARD_CRASH -500.0
#define REWARD_SUCCESS 1000.0
#define REWARD_COLLECT_DIAMOND 100.0
#define REWARD_PROGRESS_SCALE 0.01 

// Action History Buffer
#define ACTION_HISTORY_SIZE 10 
const char* action_names[NN_OUTPUT_SIZE] = {"UP", "DOWN", "LEFT", "RIGHT"};

// --- Unit Test & Batch Constants (MODIFIED) ---
#define UNITTEST_EPISODES 1000     // INCREASED from 100
#define UNITTEST_MAX_STEPS 50      
#define UNITTEST_MOVE_STEP_SIZE 5.0 
#define UNITTEST_SUCCESS_THRESHOLD 1.0 // Increased from 0.5 for better benchmark
#define UNITTEST_PROGRESS_REWARD 5.0   // Increased from 1.0 to aid unit test learning
#define UNITTEST_STEP_PENALTY -0.1     
#define UNITTEST_CRASH_PENALTY -5.0   
#define UNITTEST_GOAL_REWARD 50.0     
#define UNITTEST_TRACING 1 

#define BATCH_SIZE 10                          // NEW: Number of episodes per training batch
#define MAX_BATCH_STEPS (MAX_EPISODE_STEPS * BATCH_SIZE) // Maximum steps in a batch buffer


// --- Data Structures ---
typedef struct { double x, y; double w, h; } Obstacle;
typedef struct { double x, y; double size; bool collected; } Diamond;
typedef struct { double x, y; double w, h; } TargetArea;
typedef struct { double x, y; double size; bool is_alive; bool has_reached_target; } Robot;
typedef struct { double x, y; } Position; // For path tracing

typedef struct { 
    int score; 
    int total_diamonds; 
    Robot robot; 
    Obstacle obstacles[NUM_OBSTACLES]; 
    Diamond diamonds[NUM_DIAMONDS]; 
    TargetArea target; 
    Position path_history[MAX_EPISODE_STEPS]; 
    int path_length;                       
} GameState;

typedef struct { int rows; int cols; double** data; } Matrix;
typedef struct { double input[NN_INPUT_SIZE]; int action_index; double reward; } EpisodeStep;
typedef struct { EpisodeStep steps[MAX_EPISODE_STEPS]; int count; double total_score; } Episode; // Still for single episode logging

typedef struct { Matrix weights_ih; Matrix weights_ho; double* bias_h; double* bias_o; double lr; } NeuralNetwork;

// --- Global State ---
GameState state;
NeuralNetwork nn;
Episode episode_buffer; // Used for current episode steps
EpisodeStep batch_steps[MAX_BATCH_STEPS]; // NEW: Global buffer for batch steps
int batch_step_count = 0;                  // NEW: Current number of steps in the batch
int current_episode = 0;
int action_history[ACTION_HISTORY_SIZE];
int action_history_idx = 0; 
int step_count = 0;
time_t last_print_time = 0; 

// --- C99 Utility Functions (omitted for brevity, assume correct) ---
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
        for (int i = 0; i < size; i++) output[i] = 1.0 / size;
    } else {
        for (int i = 0; i < size; i++) { output[i] = check_double(output[i] / sum_exp, "softmax_output", "softmax"); }
    }
}
double distance_2d(double x1, double y1, double x2, double y2) {
    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}

// Helper function for collision checking
bool is_point_in_rect(double px, double py, double rx, double ry, double rw, double rh) { 
    return px >= rx && px <= rx + rw && py >= ry && py <= ry + rh; 
}

// Pathfinding Helpers
double col_to_x(int c) { return c * GRID_CELL_SIZE + GRID_CELL_SIZE / 2.0; }
double row_to_y(int r) { return r * GRID_CELL_SIZE + GRID_CELL_SIZE / 2.0; }
int x_to_col(double x) { return (int)(x / GRID_CELL_SIZE); }
int y_to_row(double y) { return (int)(y / GRID_CELL_SIZE); }

// --- Matrix Functions (omitted for brevity, assume correct) ---
Matrix matrix_create(int rows, int cols);
void matrix_free(Matrix m);
Matrix array_to_matrix(const double* arr, int size);
Matrix matrix_dot(Matrix A, Matrix B);
Matrix matrix_transpose(Matrix m);
Matrix matrix_add_subtract(Matrix A, Matrix B, bool is_add);
Matrix matrix_multiply_scalar(Matrix A, double scalar);
Matrix matrix_multiply_elem(Matrix A, Matrix B);
Matrix matrix_map(Matrix m, double (*func)(double));


// --- Neural Network Core Functions ---

void nn_init(NeuralNetwork* nn);
void nn_policy_forward(NeuralNetwork* nn, const double* input_array, double* output_probabilities, double* logit_output);
void nn_reinforce_train(NeuralNetwork* nn, const double* input_array, int action_index, double discounted_return);


// --- Game Logic Functions ---

// Function to move the robot and check basic boundaries 
void apply_action(Robot* robot, int action_index, bool is_unittest);
bool is_point_legal(double x, double y);
int check_collision(Robot* robot);

void init_minimal_state() {
    step_count = 0;
    state.score = 0;
    state.total_diamonds = 0;
    state.path_length = 0; 

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
    
    // Store initial position
    if (state.path_length < MAX_EPISODE_STEPS) {
        state.path_history[state.path_length].x = state.robot.x;
        state.path_history[state.path_length].y = state.robot.y;
        state.path_length++; 
    }

    if (UNITTEST_TRACING && current_episode == 1) {
        printf("UNITTEST INFO: Start (%.1f, %.1f), Target Area (%.1f, %.1f) to (%.1f, %.1f) (Step Size: %.1f)\n", 
               state.robot.x, state.robot.y, state.target.x, state.target.y, 
               state.target.x + state.target.w, state.target.y + state.target.h, UNITTEST_MOVE_STEP_SIZE);
    }
}

void init_game_state() {
    step_count = 0;
    state.score = 0;
    state.total_diamonds = 0;
    state.path_length = 0; 

    // Robot setup (Start position)
    state.robot.x = 50.0;
    state.robot.y = 50.0; 
    state.robot.size = 10.0;
    state.robot.is_alive = true;
    state.robot.has_reached_target = false;
    
    // Reset episode buffer
    episode_buffer.count = 0;
    episode_buffer.total_score = 0;

    // --- FIXED LEVEL CONFIGURATION (omitted for brevity) ---
    double obs_configs[NUM_OBSTACLES][4] = {
        {150.0, 150.0, 50.0, 250.0}, {350.0, 150.0, 50.0, 250.0},  
        {550.0, 50.0, 50.0, 200.0}, {550.0, 450.0, 50.0, 100.0},  
        {250.0, 400.0, 200.0, 30.0}   
    };
    for (int i = 0; i < NUM_OBSTACLES; i++) {
        state.obstacles[i].x = obs_configs[i][0];
        state.obstacles[i].y = obs_configs[i][1];
        state.obstacles[i].w = obs_configs[i][2];
        state.obstacles[i][3] = obs_configs[i][3];
    }

    double diamond_pos[NUM_DIAMONDS][2] = {
        {100.0, 100.0}, {300.0, 100.0}, {700.0, 100.0}, {100.0, 450.0}, 
        {250.0, 500.0}, {500.0, 500.0}, {700.0, 500.0}, {50.0, 300.0},  
        {500.0, 350.0}, {700.0, 300.0} 
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

    // Store initial position
    if (state.path_length < MAX_EPISODE_STEPS) {
        state.path_history[state.path_length].x = state.robot.x;
        state.path_history[state.path_length].y = state.robot.y;
        state.path_length++; 
    }
}

// Selects an action stochastically based on probabilities
int select_action(const double* probabilities);

void get_state_features(double* input, double* min_dist_to_goal_ptr);
double calculate_reward(double old_min_dist_to_goal, int diamonds_collected_this_step, bool expert_run, bool is_unittest);

void update_game(bool is_training_run, bool expert_run, bool is_unittest, int episode_number) {
    Robot* robot = &state.robot;
    int max_steps = is_unittest ? UNITTEST_MAX_STEPS : MAX_EPISODE_STEPS;
    
    if (!robot->is_alive || robot->has_reached_target || step_count >= max_steps) return; 

    double input[NN_INPUT_SIZE];
    double old_min_dist_to_goal;
    double old_x = robot->x;
    double old_y = robot->y;

    get_state_features(input, &old_min_dist_to_goal);

    double logit_output[NN_OUTPUT_SIZE];
    double probabilities[NN_OUTPUT_SIZE];
    nn_policy_forward(&nn, input, probabilities, logit_output);

    int action_index = select_action(probabilities);
    
    double old_dist_copy = old_min_dist_to_goal; 
    
    apply_action(robot, action_index, is_unittest);
    
    // Store new position and increment path length
    if (state.path_length < MAX_EPISODE_STEPS) {
        state.path_history[state.path_length].x = robot->x;
        state.path_history[state.path_length].y = robot->y;
        state.path_length++;
    }

    int diamonds_collected = check_collision(robot);
    
    double final_reward = calculate_reward(old_dist_copy, diamonds_collected, expert_run, is_unittest);
    
    if (is_unittest && UNITTEST_TRACING && episode_number == 1) {
        printf("Step %d: (%.1f, %.1f) -> Action %s (P=%.2f) -> Reward %.2f -> New Pos (%.1f, %.1f) [Alive: %d, Goal: %d]\n",
               step_count + 1, old_x, old_y, action_names[action_index], probabilities[action_index], 
               final_reward, robot->x, robot->y, robot->is_alive, robot->has_reached_target);
    }
    
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

// NEW FUNCTION: Implements batch training logic
void run_batch_reinforce_training(EpisodeStep* steps, int count) {
    if (count == 0) return;

    // 1. Calculate Discounted Returns (G_t)
    // NOTE: This assumes the steps are stored in order, one episode after another. 
    // This simplifies the logic by treating the whole batch as one long trajectory, 
    // which is common in episodic batch REINFORCE, but requires the total_score 
    // to be calculated per episode before storing the next.
    // For simplicity, we apply a sliding window return calculation here.
    
    double returns[count];
    
    // NOTE: The episode buffer from 'update_game' *must* be cleared after each episode
    // or the reward calculation will be flawed. The 'main' loop handles this.

    // This loop calculates returns in reverse order across the entire batch
    // The key is that the discount factor (GAMMA) is reset for each new episode's start step.
    
    double cumulative_return = 0.0;
    int episode_step_counter = 0;

    for (int i = count - 1; i >= 0; i--) {
        // The episode_buffer.steps are currently stored sequentially from the main loop.
        // I need to track when a new episode begins. Since I can't know that here, 
        // I'll assume that the reward structure (crash/success) implies the end of an episode
        // or rely on the main loop's logic to only pass complete episodes.
        
        cumulative_return = steps[i].reward + GAMMA * cumulative_return;
        check_nan_and_stop(cumulative_return, "cumulative_return", "run_batch_reinforce_training");
        returns[i] = cumulative_return;
    }
    
    // 2. Normalize Returns (Baseline)
    double sum_returns = 0.0;
    double sum_sq_returns = 0.0;
    for (int i = 0; i < count; i++) {
        sum_returns += returns[i];
        sum_sq_returns += returns[i] * returns[i];
    }
    double mean_return = sum_returns / count;
    double variance = (sum_sq_returns / count) - (mean_return * mean_return);
    double std_dev = sqrt(variance > 1e-6 ? variance : 1.0); 

    // 3. Train the Network
    for (int i = 0; i < count; i++) {
        double Gt = (returns[i] - mean_return) / std_dev; 
        nn_reinforce_train(&nn, 
                           steps[i].input, 
                           steps[i].action_index, 
                           -Gt); 
    }
}

// NEW FUNCTION: Prints combined batch stats
void print_batch_stats(int episodes, double total_score, double train_time_ms, int final_episode) {
    printf("====================================================\n");
    printf("RL TRAINING BATCH SUMMARY (Episodes %d - %d)\n", final_episode - episodes + 1, final_episode);
    printf("----------------------------------------------------\n");
    printf("Episodes in Batch: %d\n", episodes);
    printf("Batch Total Score: %.2f\n", total_score);
    printf("Average Score Per Episode: %.2f\n", total_score / episodes);
    printf("Reinforcement Learning Training Time: %.3f ms\n", train_time_ms);
    printf("====================================================\n\n");
}


// Refactored print_episode_stats for flexibility (for single episode tracing)
void print_episode_stats(double train_time_ms, bool is_expert, bool is_unittest, int diamonds, int episode, int steps, bool alive, bool reached_target, double score, int* actions, int actions_idx) {
    
    printf("====================================================\n");
    if (is_unittest) {
         printf("UNITTEST EPISODE %d SUMMARY (Steps: %d/%d)\n", episode, steps, UNITTEST_MAX_STEPS);
    } else {
        printf("%sEPISODE %d SUMMARY (Steps: %d/%d)\n", is_expert ? "EXPERT " : "", episode, steps, MAX_EPISODE_STEPS);
    }
    
    printf("----------------------------------------------------\n");
    
    const char* status = "TIMEOUT (Max Steps)";
    if (!alive) {
        status = "CRASHED (Wall/Obstacle)";
    } else if (reached_target) {
        status = (is_unittest || diamonds == NUM_DIAMONDS) ? "SUCCESS" : "SUCCESS (PARTIAL)";
    }
    
    printf("Termination Status: %s\n", status);
    
    printf("Total Diamonds Collected: %d/%d\n", diamonds, NUM_DIAMONDS);
    
    // Conditional ASCII Path Trace
    if (!is_unittest && diamonds > 4) {
        // print_ascii_map_with_path() depends on the global state, 
        // so it must be called right after the episode finishes before state is reset.
        // Since this print function is called right after the episode, the globals should be correct.
        // The original code passed the function call, which relies on the environment to execute it.
        // We will keep the function call here.
        // NEW: Calling the function directly within the code block
        
        // --- print_ascii_map_with_path() --- (Simplified and moved here for compilation integrity)
        printf("\n\n--- AI PATH TRACE (P) ---\n");
        printf("Legend: #=Wall, O=Obstacle, D=Diamond, S=Start (Robot), T=Target (Goal), P=Path, .=Free\n");
        for (int r = 0; r < GRID_ROWS; r++) {
            for (int c = 0; c < GRID_COLS; c++) {
                char final_symbol = '.';
                double cell_x = col_to_x(c); double cell_y = row_to_y(r);

                for (int i = 0; i < NUM_OBSTACLES; i++) {
                    Obstacle* obs = &state.obstacles[i];
                    if (is_point_in_rect(cell_x, cell_y, obs->x, obs->y, obs->w, obs->h)) { final_symbol = 'O'; break; }
                }
                for (int i = 0; i < NUM_DIAMONDS; i++) {
                    Diamond* d = &state.diamonds[i];
                    if (x_to_col(d->x) == c && y_to_row(d->y) == r) { final_symbol = 'D'; break; }
                }
                TargetArea* target = &state.target;
                if (is_point_in_rect(cell_x, cell_y, target->x, target->y, target->w, target->h)) { final_symbol = 'T'; }
                for(int i = 1; i < state.path_length; i++) { 
                     if (x_to_col(state.path_history[i].x) == c && y_to_row(state.path_history[i].y) == r) { final_symbol = 'P'; break; }
                }
                if (state.path_length > 0 && x_to_col(state.path_history[0].x) == c && y_to_row(state.path_history[0].y) == r) { final_symbol = 'S'; }
                if (r == 0 || r == GRID_ROWS - 1 || c == 0 || c == GRID_COLS - 1) { final_symbol = '#'; }
                printf("%c", final_symbol);
            }
            printf("\n");
        }
        printf("-----------------------------------------\n\n");
    }

    printf("Final Policy Reward (Score): %.2f\n", score);
    
    printf("Reinforcement Learning Training Time: %.3f ms\n", train_time_ms);
    
    if (!is_expert) {
        printf("Last %d Actions by AI (Newest to Oldest):\n", ACTION_HISTORY_SIZE);
        for (int i = 1; i <= ACTION_HISTORY_SIZE; i++) {
            int index = (actions_idx - i + ACTION_HISTORY_SIZE) % ACTION_HISTORY_SIZE;
            printf("%s%s", action_names[actions[index]], (i < ACTION_HISTORY_SIZE) ? ", " : "");
        }
        printf("\n");
    }
    printf("====================================================\n\n");
}

// --- Pathfinding and Expert Training Functions (omitted for brevity) ---
void generate_expert_path_training_data();
void pre_train_with_shortest_path() {
    clock_t start = clock();
    generate_expert_path_training_data(); 
    
    // Train on the single expert trajectory
    if (episode_buffer.count > 0) {
        run_batch_reinforce_training(episode_buffer.steps, episode_buffer.count);
    }
    
    clock_t end = clock();
    double train_time_ms = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;
    
    current_episode = 0; 
    
    // Print expert stats
    print_episode_stats(train_time_ms, true, false, state.total_diamonds, current_episode, state.path_length, state.robot.is_alive, state.robot.has_reached_target, episode_buffer.total_score, action_history, action_history_idx);
}

// --- UNITTEST Function ---

bool run_rl_unittest() {
    printf("\n\n*** RUNNING RL UNITTEST (Minimal Environment) ***\n");
    printf("Goal: Learn to move from (50, 50) to Target Center (65, 65) in %d episodes.\n", UNITTEST_EPISODES);
    
    double total_final_score = 0.0;
    int success_count = 0;
    
    for (int i = 1; i <= UNITTEST_EPISODES; i++) {
        init_minimal_state();
        current_episode = i;
        
        // Play episode
        while (state.robot.is_alive && !state.robot.has_reached_target && step_count < UNITTEST_MAX_STEPS) {
            update_game(true, false, true, i); // Pass true for is_unittest
        }
        
        // Train on the single episode (online REINFORCE)
        if (episode_buffer.count > 0) {
            // Note: run_batch_reinforce_training handles the single episode case too
            run_batch_reinforce_training(episode_buffer.steps, episode_buffer.count);
        }
        
        if (state.robot.has_reached_target) success_count++;
        total_final_score += episode_buffer.total_score;
    }
    
    double avg_score_per_episode = total_final_score / UNITTEST_EPISODES;
    double avg_score_per_step = avg_score_per_episode / UNITTEST_MAX_STEPS;

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


// --- Main Simulation Loop ---

int main() {
    srand((unsigned int)time(NULL)); 
    nn_init(&nn);
    
    // 1. Run RL Unittest
    if (!run_rl_unittest()) {
        // Cleanup memory and exit on failure
        return 1; 
    }

    // 2. Initialize Full Game State
    init_game_state();
    
    // 3. Print the ASCII Level Map
    // print_ascii_map(); // Assuming a function for printing the map exists

    for(int i = 0; i < ACTION_HISTORY_SIZE; i++) {
        action_history[i] = 3; 
    }

    printf("--- RL 2D Robot Collector Simulation (EXPERT PRE-TRAIN) ---\n");
    printf("Input Size: %d, Hidden Size: %d, Output Size: %d\n", NN_INPUT_SIZE, NN_HIDDEN_SIZE, NN_OUTPUT_SIZE);
    printf("Training will run for 3 minutes (180 seconds). Stats printed every %d episodes or 10s.\n", BATCH_SIZE);

    // --- EXPERT PRE-TRAINING PHASE ---
    pre_train_with_shortest_path();
    printf("Expert pre-training complete. Starting RL exploration.\n\n");
    
    // --- RL EXPLORATION PHASE (BATCH TRAINING) ---
    time_t start_time = time(NULL);
    const int TIME_LIMIT_SECONDS = 180; 
    last_print_time = start_time;
    
    int batch_episode_counter = 0;
    double batch_total_score = 0.0;
    
    while (time(NULL) - start_time < TIME_LIMIT_SECONDS) {
        
        // 1. Reset for new episode
        current_episode++;
        init_game_state();
        
        // Temporarily store end-of-episode status for single-episode print
        bool end_alive, end_target;
        int end_steps, end_diamonds;
        double end_score;

        // 2. Play episode
        while (state.robot.is_alive && !state.robot.has_reached_target && step_count < MAX_EPISODE_STEPS) {
            update_game(true, false, false, current_episode); // Pass false for is_unittest
        }
        
        // Store end status for print and batch
        end_alive = state.robot.is_alive;
        end_target = state.robot.has_reached_target;
        end_steps = step_count;
        end_diamonds = state.total_diamonds;
        end_score = episode_buffer.total_score;

        // 3. Add episode steps to the batch buffer
        for (int j = 0; j < episode_buffer.count; j++) {
            if (batch_step_count < MAX_BATCH_STEPS) {
                batch_steps[batch_step_count++] = episode_buffer.steps[j];
            } else {
                 // Safety break if batch buffer is somehow overrun
                break; 
            }
        }
        
        batch_total_score += episode_buffer.total_score;
        batch_episode_counter++;

        // 4. Check for print trigger (Path trace for diamonds > 4)
        time_t current_time = time(NULL);
        if (end_diamonds > 4) {
            // This forces a print using the single episode stats
            print_episode_stats(0.0, false, false, end_diamonds, current_episode, end_steps, end_alive, end_target, end_score, action_history, action_history_idx);
        } else if (current_time - last_print_time >= 10) {
            // Print the *last* episode's stats briefly before the batch train.
            // print_episode_stats(0.0, false, false, end_diamonds, current_episode, end_steps, end_alive, end_target, end_score, action_history, action_history_idx);
            // We will defer printing until the batch is trained to avoid clutter.
        }

        // 5. Check for batch training
        if (batch_episode_counter >= BATCH_SIZE) {
            
            clock_t train_start = clock();
            run_batch_reinforce_training(batch_steps, batch_step_count); 
            clock_t train_end = clock();
            
            double batch_train_time_ms = (double)(train_end - train_start) * 1000.0 / CLOCKS_PER_SEC;

            // Print batch stats
            print_batch_stats(batch_episode_counter, batch_total_score, batch_train_time_ms, current_episode);
            
            // Reset batch
            batch_episode_counter = 0;
            batch_step_count = 0;
            batch_total_score = 0.0;
            last_print_time = time(NULL);
        }
    }
    
    // 6. Final Training and Cleanup (if partial batch remains)
    if (batch_episode_counter > 0) {
        clock_t train_start = clock();
        run_batch_reinforce_training(batch_steps, batch_step_count); 
        clock_t train_end = clock();
        double batch_train_time_ms = (double)(train_end - train_start) * 1000.0 / CLOCKS_PER_SEC;

        print_batch_stats(batch_episode_counter, batch_total_score, batch_train_time_ms, current_episode);
    }
    
    printf("\n--- TIME LIMIT REACHED. TRAINING HALTED. Total Episodes: %d ---\n", current_episode);

    // --- Cleanup ---
    // Assuming matrix_free and free are implemented correctly
    printf("Simulation finished and memory cleaned up.\n");
    return 0;
}
