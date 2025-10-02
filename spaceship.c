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
#define GRID_COLS (int)(CANVAS_WIDTH / GRID_CELL_SIZE) // 40
#define GRID_ROWS (int)(CANVAS_HEIGHT / GRID_CELL_SIZE) // 30
#define MAX_PATH_NODES 2000

// NN & RL Constants (Retained for Phase 2)
#define NN_INPUT_SIZE 9 
#define NN_HIDDEN_SIZE 128 
#define NN_OUTPUT_SIZE 4 
#define NN_LEARNING_RATE 0.01 
#define GAMMA 0.99 

// Exploration Constants 
#define EPSILON_START 1.0 
#define EPSILON_END 0.01 
// Q_LEARNING_EPISODES is 50000, so decay should last 50000 episodes
#define EPSILON_DECAY_EPISODES 50000 

// Reward Goals and Values
#define REWARD_PER_STEP -1.0         // Penalty for each move
#define REWARD_CRASH -500.0          // Terminal penalty
#define REWARD_SUCCESS 1000.0        // Terminal reward for reaching target
#define REWARD_COLLECT_DIAMOND 100.0 // Instant reward for diamond
#define REWARD_PROGRESS_SCALE 0.01   // Dense reward shaping for distance change

// Action History Buffer
#define ACTION_HISTORY_SIZE 10 

// Q-Learning Constants (MODIFIED FOR SOLVABILITY)
#define Q_LEARNING_EPISODES 50000 
#define Q_LEARNING_ALPHA 0.2 
#define Q_X_BINS GRID_COLS 
#define Q_Y_BINS GRID_ROWS 
#define Q_DIAMOND_DIRECTION_BINS 8 
#define Q_DIAMOND_COUNT_BINS (NUM_DIAMONDS + 1) 
#define Q_STATE_SIZE (Q_X_BINS * Q_Y_BINS * Q_DIAMOND_DIRECTION_BINS * Q_DIAMOND_COUNT_BINS) 

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

// UPDATED: Added fields to track reward contributions
typedef struct { 
    EpisodeStep steps[MAX_EPISODE_STEPS]; 
    int count; 
    double total_score; 
    double reward_total_diamonds;
    double reward_total_moves;
    double reward_terminal; // Accumulates CRASH or SUCCESS/TARGET reward
    double reward_total_progress;
} Episode;

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


// --- Utility Functions (Omitted for brevity, assumed correct) ---

double distance_2d(double x1, double y1, double x2, double y2) {
    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}

// ... (Rest of Matrix and NN utility functions)

// --- Q-Learning State Discretization ---

int get_q_state_index() {
    int x_bin, y_bin, d_direction_bin, d_count_bin;

    // 1. Robot X Bin (40 Bins)
    x_bin = (int)(state.robot.x / GRID_CELL_SIZE);
    if (x_bin >= Q_X_BINS) x_bin = Q_X_BINS - 1;
    if (x_bin < 0) x_bin = 0;

    // 2. Robot Y Bin (30 Bins)
    y_bin = (int)(state.robot.y / GRID_CELL_SIZE);
    if (y_bin >= Q_Y_BINS) y_bin = Q_Y_BINS - 1;
    if (y_bin < 0) y_bin = 0;

    // 3. Direction to Nearest Goal Bin (8 Bins)
    double goal_x, goal_y;
    Diamond* nearest_diamond = NULL;
    double min_diamond_dist = INFINITY;
    for (int i = 0; i < NUM_DIAMONDS; i++) {
        Diamond* d = &state.diamonds[i];
        if (!d->collected) {
            double dist = distance_2d(state.robot.x, state.robot.y, d->x, d->y);
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

    double angle = atan2(goal_y - state.robot.y, goal_x - state.robot.x); 
    angle = fmod(angle + 2 * M_PI, 2 * M_PI); 
    int direction_bin = (int)floor(angle / (M_PI / 4.0));
    d_direction_bin = direction_bin % Q_DIAMOND_DIRECTION_BINS;

    // 4. Diamonds Collected Count Bin (11 Bins: 0 to 10)
    d_count_bin = state.total_diamonds;

    // Combine bins into a single index
    int state_index = x_bin;
    state_index = state_index * Q_Y_BINS + y_bin;
    state_index = state_index * Q_DIAMOND_DIRECTION_BINS + d_direction_bin;
    state_index = state_index * Q_DIAMOND_COUNT_BINS + d_count_bin;

    if (state_index >= Q_STATE_SIZE) state_index = Q_STATE_SIZE - 1;
    if (state_index < 0) state_index = 0;

    return state_index;
}

// --- Game Logic Functions ---

void init_game_state() {
    step_count = 0; state.score = 0; state.total_diamonds = 0;
    state.robot.x = 50.0; state.robot.y = 50.0; state.robot.size = 10.0;
    state.robot.is_alive = true; state.robot.has_reached_target = false;
    episode_buffer.count = 0; episode_buffer.total_score = 0;
    
    // NEW: Initialize reward trackers
    episode_buffer.reward_total_diamonds = 0.0;
    episode_buffer.reward_total_moves = 0.0;
    episode_buffer.reward_terminal = 0.0;
    episode_buffer.reward_total_progress = 0.0;

    // ... (rest of init_game_state, including obstacles and diamonds setup)
    double obs_configs[NUM_OBSTACLES][4] = {
        {150.0, 150.0, 50.0, 250.0}, {350.0, 150.0, 50.0, 250.0},
        {550.0, 50.0, 50.0, 200.0}, {550.0, 450.0, 50.0, 100.0},
        {250.0, 400.0, 200.0, 30.0}
    };
    for (int i = 0; i < NUM_OBSTACLES; i++) {
        state.obstacles[i].x = obs_configs[i][0]; state.obstacles[i].y = obs_configs[i][1];
        state.obstacles[i].w = obs_configs[i][2]; state.obstacles[i].h = obs_configs[i][3];
    }
    double diamond_pos[NUM_DIAMONDS][2] = {
        {100.0, 100.0}, {300.0, 100.0}, {700.0, 100.0}, 
        {100.0, 450.0}, {250.0, 500.0}, {500.0, 500.0}, 
        {700.0, 500.0}, {50.0, 300.0},  {500.0, 350.0}, 
        {700.0, 300.0}  
    };
    for (int i = 0; i < NUM_DIAMONDS; i++) {
        state.diamonds[i].x = diamond_pos[i][0]; state.diamonds[i].y = diamond_pos[i][1];
        state.diamonds[i].size = 8.0; state.diamonds[i].collected = false;
    }
    state.target.x = CANVAS_WIDTH - 100.0; state.target.y = CANVAS_HEIGHT - 100.0;
    state.target.w = 50.0; state.target.h = 50.0;
}

bool is_potential_move_legal(double current_x, double current_y, int action_index) {
    double next_x = current_x, next_y = current_y, r = state.robot.size;
    switch ((Action)action_index) {
        case ACTION_UP:    next_y -= MOVE_STEP_SIZE; break; 
        case ACTION_DOWN:  next_y += MOVE_STEP_SIZE;  break;
        case ACTION_LEFT:  next_x -= MOVE_STEP_SIZE; break; 
        case ACTION_RIGHT: next_x += MOVE_STEP_SIZE;  break; 
        default: return false;
    }
    if (next_x - r < BORDER_WIDTH || next_x + r > CANVAS_WIDTH - BORDER_WIDTH ||
        next_y - r < BORDER_WIDTH || next_y + r > CANVAS_HEIGHT - BORDER_WIDTH) { return false; }
    for (int i = 0; i < NUM_OBSTACLES; i++) {
        Obstacle* obs = &state.obstacles[i];
        double closest_x = fmax(obs->x, fmin(next_x, obs->x + obs->w));
        double closest_y = fmax(obs->y, fmin(next_y, obs->y + obs->h));
        double dx = next_x - closest_x, dy = next_y - closest_y;
        if (dx * dx + dy * dy < r * r) { return false; }
    }
    return true;
}

void get_state_features(double* input, double* min_dist_to_goal_ptr) {
    Robot* robot = &state.robot;
    // Simplified feature calculation for dependency resolution
    double goal_x = state.target.x + state.target.w / 2.0;
    double goal_y = state.target.y + state.target.h / 2.0;
    *min_dist_to_goal_ptr = distance_2d(robot->x, robot->y, goal_x, goal_y);
}

// NEW: Step reward calculation that returns the reward components via pointers
double calculate_step_reward(double old_min_dist_to_goal, int diamonds_collected_this_step, double* diamond_contribution_ptr, double* progress_contribution_ptr) {
    double reward = 0.0; 
    
    // 1. Reward per step (Moves)
    reward += REWARD_PER_STEP; 

    *diamond_contribution_ptr = 0.0;
    *progress_contribution_ptr = 0.0;

    // 2. Diamond Collection Reward
    if (diamonds_collected_this_step > 0) { 
        double diamond_reward = REWARD_COLLECT_DIAMOND * diamonds_collected_this_step; 
        reward += diamond_reward; 
        *diamond_contribution_ptr = diamond_reward;
    }
    
    // 3. Dense Progress Reward
    double min_dist_to_goal;
    double dummy_input[NN_INPUT_SIZE];
    get_state_features(dummy_input, &min_dist_to_goal); 
    double distance_change = old_min_dist_to_goal - min_dist_to_goal;
    double progress_reward = REWARD_PROGRESS_SCALE * distance_change;
    reward += progress_reward; 
    *progress_contribution_ptr = progress_reward;
    
    return reward;
}

void apply_action(Robot* robot, int action_index) {
    double dx = 0.0, dy = 0.0;
    switch ((Action)action_index) {
        case ACTION_UP:    dy = -MOVE_STEP_SIZE; break; 
        case ACTION_DOWN:  dy = MOVE_STEP_SIZE;  break;
        case ACTION_LEFT:  dx = -MOVE_STEP_SIZE; break; 
        case ACTION_RIGHT: dx = MOVE_STEP_SIZE;  break; 
        case ACTION_INVALID: return;
    }
    robot->x += dx; robot->y += dy;
    action_history[action_history_idx] = action_index;
    action_history_idx = (action_history_idx + 1) % ACTION_HISTORY_SIZE;
}

int check_collision(Robot* robot) {
    double r = robot->size; int diamonds_collected_this_step = 0;
    if (robot->x - r < BORDER_WIDTH || robot->x + r > CANVAS_WIDTH - BORDER_WIDTH ||
        robot->y - r < BORDER_WIDTH || robot->y + r > CANVAS_HEIGHT - BORDER_WIDTH) { robot->is_alive = false; return 0; }
    for (int i = 0; i < NUM_OBSTACLES; i++) {
        Obstacle* obs = &state.obstacles[i];
        double closest_x = fmax(obs->x, fmin(robot->x, obs->x + obs->w));
        double closest_y = fmax(obs->y, fmin(robot->y, obs->y + obs->h));
        double dx = robot->x - closest_x, dy = robot->y - closest_y;
        if (dx * dx + dy * dy < r * r) { robot->is_alive = false; return 0; }
    }
    for (int i = 0; i < NUM_DIAMONDS; i++) {
        Diamond* d = &state.diamonds[i];
        if (d->collected) continue;
        if (distance_2d(robot->x, robot->y, d->x, d->y) < robot->size + d->size) {
            d->collected = true; state.total_diamonds++; diamonds_collected_this_step++; state.score += (int)REWARD_COLLECT_DIAMOND;
        }
    }
    TargetArea* target = &state.target;
    if (robot->x + r > target->x && robot->x - r < target->x + target->w &&
        robot->y + r > target->y && robot->y - r < target->y + target->h) { robot->has_reached_target = true; }
    return diamonds_collected_this_step;
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

    if ((double)rand() / RAND_MAX < current_epsilon) {
        // EXPLORE: Choose a random LEGAL action
        return legal_actions[rand() % num_legal];
    } else {
        // EXPLOIT: Choose the action with the maximum Q-value among legal actions
        int best_action = -1;
        double max_q_value = -INFINITY;

        for (int i = 0; i < num_legal; i++) {
            int action = legal_actions[i];
            if (Q_table[state_index][action] > max_q_value) {
                max_q_value = Q_table[state_index][action];
                best_action = action;
            }
        }
        return best_action;
    }
}

// UPDATED: q_learning_update now accepts a flag for terminal state
void q_learning_update(int state_old, int action, double reward, int state_new, bool is_terminal) {
    double max_q_next = 0.0;
    
    if (!is_terminal) {
        // Find max Q(s', a') (next state's best action among all actions)
        for (int a = 0; a < NN_OUTPUT_SIZE; a++) {
            if (Q_table[state_new][a] > max_q_next) {
                max_q_next = Q_table[state_new][a];
            }
        }
    }
    
    double old_q_value = Q_table[state_old][action];
    
    // Bellman Equation: Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]
    Q_table[state_old][action] = old_q_value + Q_LEARNING_ALPHA * (reward + GAMMA * max_q_next - old_q_value);
}

void update_game_q_learning() {
    Robot* robot = &state.robot;
    
    // Check if the episode is already over (in case this is called past terminal)
    if (!robot->is_alive || robot->has_reached_target || step_count >= MAX_EPISODE_STEPS) return; 

    int state_old = get_q_state_index();
    
    double current_epsilon = EPSILON_END;
    if (current_episode < EPSILON_DECAY_EPISODES) {
        double decay_rate = (EPSILON_START - EPSILON_END) / EPSILON_DECAY_EPISODES;
        current_epsilon = EPSILON_START - decay_rate * current_episode;
    }

    double old_min_dist_to_goal;
    double dummy_input[NN_INPUT_SIZE];
    get_state_features(dummy_input, &old_min_dist_to_goal);

    int action_index = select_q_action(state_old, current_epsilon);
    
    if (action_index == ACTION_INVALID) {
        robot->is_alive = false; 
    } else {
        apply_action(robot, action_index);
    }
    
    int diamonds_collected = check_collision(robot);
    
    double diamond_contrib, progress_contrib;
    // Calculate non-terminal step reward
    double reward = calculate_step_reward(old_min_dist_to_goal, diamonds_collected, &diamond_contrib, &progress_contrib);

    // Accumulate step-wise rewards
    episode_buffer.reward_total_moves += REWARD_PER_STEP;
    episode_buffer.reward_total_diamonds += diamond_contrib;
    episode_buffer.reward_total_progress += progress_contrib;

    bool is_terminal = (!robot->is_alive || robot->has_reached_target || step_count >= MAX_EPISODE_STEPS - 1);
    
    // Apply TERMINAL reward on the last step
    if (is_terminal) {
        double terminal_reward = 0.0;
        if (!robot->is_alive) {
            terminal_reward = REWARD_CRASH;
        } else if (robot->has_reached_target) {
            terminal_reward = REWARD_SUCCESS;
            if (state.total_diamonds < NUM_DIAMONDS) { 
                terminal_reward -= (NUM_DIAMONDS - state.total_diamonds) * 50.0; 
            }
        } else if (step_count >= MAX_EPISODE_STEPS - 1) {
            // Treat timeout as terminal state with no extra reward
            is_terminal = true;
        }
        reward += terminal_reward;
        episode_buffer.reward_terminal = terminal_reward;
    }
    
    int state_new = get_q_state_index();
    
    q_learning_update(state_old, action_index, reward, state_new, is_terminal);
    
    episode_buffer.total_score += reward;
    step_count++;
}

// UPDATED: print_episode_stats now includes the score breakdown
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
    
    // NEW: Print score with breakdown
    printf("Final Policy Reward (Score): %.2f (Diamonds: %.2f, Moves: %.2f, Progress: %.2f, Terminal: %.2f)\n", 
           episode_buffer.total_score,
           episode_buffer.reward_total_diamonds, 
           episode_buffer.reward_total_moves, 
           episode_buffer.reward_total_progress,
           episode_buffer.reward_terminal
    );

    printf("RL Training Time: %.3f ms\n", train_time_ms);
    printf("====================================================\n\n");
}

// --- Main Simulation Loop (Simplified to focus on QL) ---

int main() {
    srand((unsigned int)time(NULL)); 
    init_game_state();

    // ... (omitted print_ascii_map for brevity)

    // --- PHASE 1: Q-LEARNING SOLVABLE TEST RUN (50,000 Episodes) ---
    
    memset(Q_table, 0, sizeof(Q_table)); 
    current_rl_mode = MODE_QL_TEST;
    current_episode = 0;
    double total_score_ql = 0.0;
    int success_count_ql = 0;
    
    printf("--- PHASE 1: Q-Learning Solvable Test Run (%d Episodes) ---\n", Q_LEARNING_EPISODES);

    clock_t ql_start_time = clock();
    for (int i = 0; i < Q_LEARNING_EPISODES; i++) {
        current_episode++;
        init_game_state();
        
        // Play episode
        while (state.robot.is_alive && !state.robot.has_reached_target && step_count < MAX_EPISODE_STEPS) {
            update_game_q_learning(); 
        }

        total_score_ql += episode_buffer.total_score;
        if (state.robot.has_reached_target && state.total_diamonds == NUM_DIAMONDS) {
            success_count_ql++;
        }
        
        if (current_episode % 5000 == 0) {
            printf("[QL Progress] Episode %d/%d. Avg Score Last 5k: %.2f\n", 
                   current_episode, Q_LEARNING_EPISODES, total_score_ql / current_episode);
        }
    }
    clock_t ql_end_time = clock();
    double ql_runtime_ms = (double)(ql_end_time - ql_start_time) * 1000.0 / CLOCKS_PER_SEC;

    printf("\n--- Q-LEARNING SOLVABLE TEST SUMMARY ---\n");
    printf("Total Episodes: %d\n", Q_LEARNING_EPISODES);
    printf("Total Runtime: %.3f ms\n", ql_runtime_ms);
    printf("Average Score (Total): %.2f\n", total_score_ql / Q_LEARNING_EPISODES);
    printf("Full Success Rate (All Diamonds + Target): %.2f%% (%d/%d)\n", 
           (double)success_count_ql / Q_LEARNING_EPISODES * 100.0, success_count_ql, Q_LEARNING_EPISODES);
    printf("----------------------------------------\n\n");
    
    printf("Simulation finished.\n");
    return 0;
}
