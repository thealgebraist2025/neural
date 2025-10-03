#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <string.h>
#include <sys/time.h> 

// --- FIX: Define M_PI for C99 compliance with -lm ---
#define M_PI 3.14159265358979323846f

// --- RBM CONFIGURATION CONSTANTS ---
#define NUM_INPUT_INTS 8
#define BITS_PER_INT 16
#define V_SIZE (NUM_INPUT_INTS * BITS_PER_INT) // Visible Layer Size (128)
#define H_SIZE 64                              // Hidden Layer Size
#define TRAINING_SET_SIZE 10000
#define EVAL_SET_SIZE 100
#define MAX_EPOCHS 1000 

// --- HYPERPARAMETERS ---
#define LEARNING_RATE 0.01f
#define INITIAL_WEIGHT_SCALE 0.01f
#define L2_REG_FACTOR 0.0001f
#define CD_STEPS 5
#define REPORT_INTERVAL_SECONDS 10
#define PREFETCH_DISTANCE 8 // Prefetch distance in element count (e.g., for W and vectors)

// --- TIMING LIMITS ---
#define TOTAL_MAX_TIMING_MS 2000.0        // Max milliseconds for function optimization (2 seconds)
#define HARD_STOP_TIME_SECONDS 120.0f     // HARD STOP: Maximum total run time (2 minutes)

// --- TYPE AND INDEXING IMPROVEMENTS ---
typedef float V_VECTOR[V_SIZE];
typedef float H_VECTOR[H_SIZE];
#define W_IDX(i, j) ((i) * H_SIZE + (j))

// --- RBM MODEL ENCAPSULATION ---
typedef struct {
    float W[V_SIZE * H_SIZE];
    float b[V_SIZE];
    float c[H_SIZE];
} RBM_Model;

// --- FUNCTION POINTERS AND ENUMERATION FOR OPTIMIZATION ---
typedef enum {
    FUN_1_UNROLL = 1,
    FUN_2_UNROLL = 2,
    FUN_4_UNROLL = 4,
    FUN_8_UNROLL = 8
} MatrixFunctionVersion;

typedef void (*v_to_h_func_ptr)(const RBM_Model*, const float*, float*);
typedef void (*h_to_v_func_ptr)(const RBM_Model*, const float*, float*);

v_to_h_func_ptr v_to_h_ptr;
h_to_v_func_ptr h_to_v_ptr;

MatrixFunctionVersion v_to_h_version = FUN_1_UNROLL;
MatrixFunctionVersion h_to_v_version = FUN_1_UNROLL;

// --- EVALUATION STRUCTURE ---
typedef struct {
    float avg_mse;
    int error_counts[4]; 
} EvalResult;

// --- TRAINING DATA STORAGE ---
V_VECTOR training_data_v[TRAINING_SET_SIZE];

// --- UTILITY FUNCTIONS (Omitted for brevity, identical to previous) ---

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float rand_normal() {
    static int has_spare = 0;
    static float spare;
    
    if (has_spare) {
        has_spare = 0;
        return spare;
    }
    float u1, u2;
    do {
        u1 = (float)rand() / RAND_MAX;
        u2 = (float)rand() / RAND_MAX;
    } while (u1 <= 1e-6f); 

    float mag = INITIAL_WEIGHT_SCALE * sqrtf(-2.0f * logf(u1));
    spare = mag * sinf(2.0f * M_PI * u2);
    has_spare = 1;
    return mag * cosf(2.0f * M_PI * u2);
}

void encode_int_to_bits(uint16_t num, float* bits) {
    for (int i = 0; i < BITS_PER_INT; i++) {
        bits[i] = (float)((num >> i) & 1);
    }
}

uint16_t decode_bits_to_int(const float* bits) {
    uint16_t num = 0;
    for (int i = 0; i < BITS_PER_INT; i++) {
        if (bits[i] > 0.5f) {
            num |= (1 << i);
        }
    }
    return num;
}

void sort_and_encode(uint16_t* unsorted, float* sorted_bits) {
    uint16_t temp_arr[NUM_INPUT_INTS];
    memcpy(temp_arr, unsorted, NUM_INPUT_INTS * sizeof(uint16_t));
    for (int i = 0; i < NUM_INPUT_INTS - 1; i++) {
        for (int j = 0; j < NUM_INPUT_INTS - i - 1; j++) {
            if (temp_arr[j] > temp_arr[j + 1]) {
                uint16_t temp = temp_arr[j];
                temp_arr[j] = temp_arr[j + 1];
                temp_arr[j + 1] = temp;
            }
        }
    }
    for (int i = 0; i < NUM_INPUT_INTS; i++) {
        encode_int_to_bits(temp_arr[i], &sorted_bits[i * BITS_PER_INT]);
    }
}

// --- RBM CORE FUNCTIONS (Omitted for brevity, identical to previous) ---
void init_rbm(RBM_Model* model) { /* ... */ }
void sample_bernoulli(const float* prob, float* sample, int size) { /* ... */ }
void update_weights(RBM_Model* model, const float* v0, const float* h0_prob, const float* vk, const float* hk_prob) { /* ... */ }


// --- CACHE-OPTIMIZED V_TO_H IMPLEMENTATIONS WITH PREFETCHING ---

// Prefetching function for GCC/Clang
#if defined(__GNUC__) || defined(__clang__)
#define PREFETCH(addr) __builtin_prefetch(addr)
#else
#define PREFETCH(addr) ((void)0)
#endif


// V_TO_H (1X Unroll - Baseline + Prefetch)
void v_to_h_1x(const RBM_Model* model, const float* v, float* h_prob) {
    for (int j = 0; j < H_SIZE; j++) { h_prob[j] = model->c[j]; }
    
    for (int i = 0; i < V_SIZE; i++) {
        // Prefetch the weight row and output activations for the NEXT visible unit (i + 1)
        PREFETCH(&model->W[W_IDX(i + PREFETCH_DISTANCE, 0)]);
        PREFETCH(&v[i + PREFETCH_DISTANCE]);

        float v_i = v[i];
        for (int j = 0; j < H_SIZE; j++) {
            h_prob[j] += v_i * model->W[W_IDX(i, j)]; 
        }
    }
    for (int j = 0; j < H_SIZE; j++) { h_prob[j] = sigmoid(h_prob[j]); }
}

// V_TO_H (2X Unroll + Prefetch)
void v_to_h_2x(const RBM_Model* model, const float* v, float* h_prob) {
    for (int j = 0; j < H_SIZE; j++) { h_prob[j] = model->c[j]; }
    for (int i = 0; i < V_SIZE; i++) {
        PREFETCH(&model->W[W_IDX(i + PREFETCH_DISTANCE, 0)]);
        float v_i = v[i];
        int j = 0;
        for (; j < H_SIZE; j += 2) {
            // Prefetch next chunk of W
            PREFETCH(&model->W[W_IDX(i, j + PREFETCH_DISTANCE)]); 
            h_prob[j] += v_i * model->W[W_IDX(i, j)];
            h_prob[j+1] += v_i * model->W[W_IDX(i, j+1)];
        }
        for (; j < H_SIZE; j++) { h_prob[j] += v_i * model->W[W_IDX(i, j)]; }
    }
    for (int j = 0; j < H_SIZE; j++) { h_prob[j] = sigmoid(h_prob[j]); }
}

// V_TO_H (4X Unroll + Prefetch)
void v_to_h_4x(const RBM_Model* model, const float* v, float* h_prob) {
    for (int j = 0; j < H_SIZE; j++) { h_prob[j] = model->c[j]; }
    for (int i = 0; i < V_SIZE; i++) {
        PREFETCH(&model->W[W_IDX(i + PREFETCH_DISTANCE, 0)]);
        float v_i = v[i];
        int j = 0;
        for (; j < H_SIZE; j += 4) {
            PREFETCH(&model->W[W_IDX(i, j + PREFETCH_DISTANCE)]); 
            h_prob[j] += v_i * model->W[W_IDX(i, j)];
            h_prob[j+1] += v_i * model->W[W_IDX(i, j+1)];
            h_prob[j+2] += v_i * model->W[W_IDX(i, j+2)];
            h_prob[j+3] += v_i * model->W[W_IDX(i, j+3)];
        }
        for (; j < H_SIZE; j++) { h_prob[j] += v_i * model->W[W_IDX(i, j)]; }
    }
    for (int j = 0; j < H_SIZE; j++) { h_prob[j] = sigmoid(h_prob[j]); }
}

// V_TO_H (8X Unroll + Prefetch)
void v_to_h_8x(const RBM_Model* model, const float* v, float* h_prob) {
    for (int j = 0; j < H_SIZE; j++) { h_prob[j] = model->c[j]; }
    for (int i = 0; i < V_SIZE; i++) {
        PREFETCH(&model->W[W_IDX(i + PREFETCH_DISTANCE, 0)]);
        float v_i = v[i];
        int j = 0;
        for (; j < H_SIZE; j += 8) {
            PREFETCH(&model->W[W_IDX(i, j + PREFETCH_DISTANCE)]); 
            h_prob[j] += v_i * model->W[W_IDX(i, j)];
            h_prob[j+1] += v_i * model->W[W_IDX(i, j+1)];
            h_prob[j+2] += v_i * model->W[W_IDX(i, j+2)];
            h_prob[j+3] += v_i * model->W[W_IDX(i, j+3)];
            h_prob[j+4] += v_i * model->W[W_IDX(i, j+4)];
            h_prob[j+5] += v_i * model->W[W_IDX(i, j+5)];
            h_prob[j+6] += v_i * model->W[W_IDX(i, j+6)];
            h_prob[j+7] += v_i * model->W[W_IDX(i, j+7)];
        }
        for (; j < H_SIZE; j++) { h_prob[j] += v_i * model->W[W_IDX(i, j)]; }
    }
    for (int j = 0; j < H_SIZE; j++) { h_prob[j] = sigmoid(h_prob[j]); }
}


// --- UNROLLED H_TO_V IMPLEMENTATIONS WITH PREFETCHING ---

// H_TO_V (1X Unroll - Baseline + Prefetch)
void h_to_v_1x(const RBM_Model* model, const float* h_prob, float* v_prob) {
    for (int i = 0; i < V_SIZE; i++) {
        // Prefetch next output element and the next row of W
        PREFETCH(&v_prob[i + PREFETCH_DISTANCE]);
        PREFETCH(&model->W[W_IDX(i + PREFETCH_DISTANCE, 0)]);

        float activation = model->b[i];
        for (int j = 0; j < H_SIZE; j++) {
            activation += h_prob[j] * model->W[W_IDX(i, j)];
        }
        v_prob[i] = sigmoid(activation);
    }
}

// H_TO_V (2X Unroll + Prefetch)
void h_to_v_2x(const RBM_Model* model, const float* h_prob, float* v_prob) {
    for (int i = 0; i < V_SIZE; i++) {
        PREFETCH(&v_prob[i + PREFETCH_DISTANCE]);
        float activation = model->b[i];
        int j = 0;
        for (; j < H_SIZE; j += 2) {
            PREFETCH(&model->W[W_IDX(i, j + PREFETCH_DISTANCE)]);
            activation += h_prob[j] * model->W[W_IDX(i, j)];
            activation += h_prob[j+1] * model->W[W_IDX(i, j+1)];
        }
        for (; j < H_SIZE; j++) {
            activation += h_prob[j] * model->W[W_IDX(i, j)];
        }
        v_prob[i] = sigmoid(activation);
    }
}

// H_TO_V (4X Unroll + Prefetch)
void h_to_v_4x(const RBM_Model* model, const float* h_prob, float* v_prob) {
    for (int i = 0; i < V_SIZE; i++) {
        PREFETCH(&v_prob[i + PREFETCH_DISTANCE]);
        float activation = model->b[i];
        int j = 0;
        for (; j < H_SIZE; j += 4) {
            PREFETCH(&model->W[W_IDX(i, j + PREFETCH_DISTANCE)]);
            activation += h_prob[j] * model->W[W_IDX(i, j)];
            activation += h_prob[j+1] * model->W[W_IDX(i, j+1)];
            activation += h_prob[j+2] * model->W[W_IDX(i, j+2)];
            activation += h_prob[j+3] * model->W[W_IDX(i, j+3)];
        }
        for (; j < H_SIZE; j++) {
            activation += h_prob[j] * model->W[W_IDX(i, j)];
        }
        v_prob[i] = sigmoid(activation);
    }
}

// H_TO_V (8X Unroll + Prefetch)
void h_to_v_8x(const RBM_Model* model, const float* h_prob, float* v_prob) {
    for (int i = 0; i < V_SIZE; i++) {
        PREFETCH(&v_prob[i + PREFETCH_DISTANCE]);
        float activation = model->b[i];
        int j = 0;
        for (; j < H_SIZE; j += 8) {
            PREFETCH(&model->W[W_IDX(i, j + PREFETCH_DISTANCE)]);
            activation += h_prob[j] * model->W[W_IDX(i, j)];
            activation += h_prob[j+1] * model->W[W_IDX(i, j+1)];
            activation += h_prob[j+2] * model->W[W_IDX(i, j+2)];
            activation += h_prob[j+3] * model->W[W_IDX(i, j+3)];
            activation += h_prob[j+4] * model->W[W_IDX(i, j+4)];
            activation += h_prob[j+5] * model->W[W_IDX(i, j+5)];
            activation += h_prob[j+6] * model->W[W_IDX(i, j+6)];
            activation += h_prob[j+7] * model->W[W_IDX(i, j+7)];
        }
        for (; j < H_SIZE; j++) {
            activation += h_prob[j] * model->W[W_IDX(i, j)];
        }
        v_prob[i] = sigmoid(activation);
    }
}


// --- TIMING AND OPTIMAL SELECTION LOGIC (Identical to previous) ---
long long time_function(void* func_ptr, const RBM_Model* model, V_VECTOR input_v, H_VECTOR input_h, V_VECTOR output_v, H_VECTOR output_h, int is_v_to_h, double time_limit_ms) {
    struct timeval start, end;
    long long total_us = 0;
    
    v_to_h_func_ptr vh_func = (v_to_h_func_ptr)func_ptr;
    h_to_v_func_ptr hv_func = (h_to_v_func_ptr)func_ptr;

    gettimeofday(&start, NULL);

    while (total_us / 1000.0 < time_limit_ms) {
        if (is_v_to_h) {
            vh_func(model, input_v, output_h);
        } else {
            hv_func(model, input_h, output_v);
        }

        gettimeofday(&end, NULL);
        total_us = (end.tv_sec - start.tv_sec) * 1000000LL + (end.tv_usec - start.tv_usec);
    }

    return total_us;
}

void test_and_select_optimal_functions(const RBM_Model* model) {
    V_VECTOR test_v;
    H_VECTOR test_h;
    V_VECTOR result_v;
    H_VECTOR result_h;
    
    const int num_tests = 8; 
    const double time_limit_ms_per_test = TOTAL_MAX_TIMING_MS / num_tests;

    for (int i = 0; i < V_SIZE; i++) test_v[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < H_SIZE; i++) test_h[i] = (float)rand() / RAND_MAX;

    void* v_to_h_funcs[] = {v_to_h_1x, v_to_h_2x, v_to_h_4x, v_to_h_8x};
    void* h_to_v_funcs[] = {h_to_v_1x, h_to_v_2x, h_to_v_4x, h_to_v_8x};
    
    MatrixFunctionVersion versions[] = {FUN_1_UNROLL, FUN_2_UNROLL, FUN_4_UNROLL, FUN_8_UNROLL};
    const char* version_names[] = {"1x (Prefetch)", "2x Unroll", "4x Unroll", "8x Unroll"};
    int num_unrolls = sizeof(versions) / sizeof(versions[0]);

    long long min_time_vh = -1;
    long long min_time_hv = -1;
    int optimal_idx_vh = 0;
    int optimal_idx_hv = 0;

    printf("\n--- FUNCTION OPTIMIZATION BENCHMARK (Total Time Limit: %.0f ms) ---\n", TOTAL_MAX_TIMING_MS);
    printf("Visible -> Hidden (V_SIZE: %d, H_SIZE: %d) - CACHE OPTIMIZED LOOP + PREFETCH\n", V_SIZE, H_SIZE);

    // --- V_TO_H TIMING ---
    for (int i = 0; i < num_unrolls; i++) {
        long long total_us = time_function(v_to_h_funcs[i], model, test_v, test_h, result_v, result_h, 1, time_limit_ms_per_test);
        
        printf("  - %-14s : %lld us total\n", version_names[i], total_us);

        if (min_time_vh == -1 || total_us < min_time_vh) {
            min_time_vh = total_us;
            optimal_idx_vh = i;
        }
    }

    v_to_h_ptr = (v_to_h_func_ptr)v_to_h_funcs[optimal_idx_vh];
    v_to_h_version = versions[optimal_idx_vh];
    printf("-> V_to_H Optimal Version Selected: %s (Unroll: %d)\n", version_names[optimal_idx_vh], v_to_h_version);

    // --- H_TO_V TIMING ---
    printf("\nHidden -> Visible (H_SIZE: %d, V_SIZE: %d) - ROW-MAJOR LOOP + PREFETCH\n", H_SIZE, V_SIZE);
    
    for (int i = 0; i < num_unrolls; i++) {
        long long total_us = time_function(h_to_v_funcs[i], model, test_v, test_h, result_v, result_h, 0, time_limit_ms_per_test);
        
        printf("  - %-14s : %lld us total\n", version_names[i], total_us);

        if (min_time_hv == -1 || total_us < min_time_hv) {
            min_time_hv = total_us;
            optimal_idx_hv = i;
        }
    }

    h_to_v_ptr = (h_to_v_func_ptr)h_to_v_funcs[optimal_idx_hv];
    h_to_v_version = versions[optimal_idx_hv];
    printf("-> H_to_V Optimal Version Selected: %s (Unroll: %d)\n", version_names[optimal_idx_hv], h_to_v_version);
    printf("------------------------------------------------------------------------------------------------\n");
}


// --- MAIN EXECUTION LOGIC (Identical to previous, uses function pointers) ---

void generate_training_data() {
    printf("Generating %d sorted lists for training...\n", TRAINING_SET_SIZE);
    srand(12345);
    for (int i = 0; i < TRAINING_SET_SIZE; i++) {
        uint16_t unsorted_list[NUM_INPUT_INTS];
        for (int j = 0; j < NUM_INPUT_INTS; j++) {
            unsorted_list[j] = (uint16_t)(((uint32_t)rand() << 15) | rand());
        }
        sort_and_encode(unsorted_list, training_data_v[i]);
    }
    printf("Training data generation complete.\n\n");
    srand(time(NULL));
}

EvalResult run_evaluation(const RBM_Model* model) {
    EvalResult result = {0.0f, {0, 0, 0, 0}};
    float total_mse = 0.0f;

    V_VECTOR eval_v;
    H_VECTOR h0_prob;
    V_VECTOR v1_prob;
    uint16_t decoded_ints[NUM_INPUT_INTS];

    for (int i = 0; i < EVAL_SET_SIZE; i++) {
        uint16_t unsorted_list[NUM_INPUT_INTS];
        for (int j = 0; j < NUM_INPUT_INTS; j++) {
            unsorted_list[j] = (uint16_t)(((uint32_t)rand() << 15) | rand());
        }
        sort_and_encode(unsorted_list, eval_v);

        v_to_h_ptr(model, eval_v, h0_prob);
        h_to_v_ptr(model, h0_prob, v1_prob);

        float mse = 0.0f;
        for (int k = 0; k < V_SIZE; k++) {
            float diff = eval_v[k] - v1_prob[k];
            mse += diff * diff;
        }
        total_mse += (mse / V_SIZE);

        for (int j = 0; j < NUM_INPUT_INTS; j++) {
            decoded_ints[j] = decode_bits_to_int(&v1_prob[j * BITS_PER_INT]);
        }

        int sorting_errors = 0;
        for (int j = 1; j < NUM_INPUT_INTS; j++) {
            if (decoded_ints[j] < decoded_ints[j - 1]) {
                sorting_errors++;
            }
        }

        if (sorting_errors == 0) {
            result.error_counts[0]++;
        } else if (sorting_errors == 1) {
            result.error_counts[1]++;
        } else if (sorting_errors == 2) {
            result.error_counts[2]++;
        } else {
            result.error_counts[3]++;
        }
    }

    result.avg_mse = total_mse / EVAL_SET_SIZE;
    return result;
}

int main() {
    RBM_Model rbm;
    
    generate_training_data();
    init_rbm(&rbm);

    // --- OPTIMIZATION STEP ---
    test_and_select_optimal_functions(&rbm);

    printf("Starting RBM Training (CD-%d) on Sorted Data...\n", CD_STEPS);
    printf("V_SIZE: %d, H_SIZE: %d, Training Set: %d, L2: %f\n", 
           V_SIZE, H_SIZE, TRAINING_SET_SIZE, L2_REG_FACTOR);
    printf("Selected V->H Unroll: %d | Selected H->V Unroll: %d\n", v_to_h_version, h_to_v_version);
    printf("Hard Stop Time Limit: %.1f seconds.\n", HARD_STOP_TIME_SECONDS);
    printf("------------------------------------------------------------------------------------------------\n");
    printf("Epoch | Time | Train Rec. MSE | Eval Rec. MSE | Eval Error Distribution (Total Lists: %d)\n", EVAL_SET_SIZE);
    printf("      |      |                |               | 0 Err | 1 Err | 2 Err | >2 Err \n");
    printf("------------------------------------------------------------------------------------------------\n");


    V_VECTOR v0;
    H_VECTOR h0_prob; 
    V_VECTOR vk_sample; 
    H_VECTOR hk_prob;
    H_VECTOR hk_sample;
    V_VECTOR vk_prob;
    V_VECTOR vn_sample;
    H_VECTOR hn_prob;

    time_t start_time = time(NULL);
    time_t last_report_time = start_time - REPORT_INTERVAL_SECONDS;
    int epoch = 0;
    
    EvalResult eval_result = run_evaluation(&rbm);
    float elapsed_time = (float)(time(NULL) - start_time);

    printf("Baseline| %4.1fs | N/A            | %.8f | %5d | %5d | %5d | %5d\n", 
           elapsed_time, eval_result.avg_mse,
           eval_result.error_counts[0], eval_result.error_counts[1],
           eval_result.error_counts[2], eval_result.error_counts[3]);


    while (epoch < MAX_EPOCHS) {
        elapsed_time = (float)(time(NULL) - start_time);
        if (elapsed_time >= HARD_STOP_TIME_SECONDS) {
            printf("\n--- EXECUTION TIME LIMIT REACHED: %.1f seconds ---\n", HARD_STOP_TIME_SECONDS);
            break; 
        }

        float epoch_mse = 0.0f;

        for (int i = 0; i < TRAINING_SET_SIZE; i++) {
            memcpy(v0, training_data_v[i], V_SIZE * sizeof(float));

            // POSITIVE PHASE (v0 -> h0)
            v_to_h_ptr(&rbm, v0, h0_prob);

            // Initialize Gibbs chain starting state v^0
            memcpy(vk_sample, v0, V_SIZE * sizeof(float));

            // GIBBS SAMPLING CHAIN (k steps)
            for (int k = 0; k < CD_STEPS; k++) {
                v_to_h_ptr(&rbm, vk_sample, hk_prob);
                sample_bernoulli(hk_prob, hk_sample, H_SIZE);

                h_to_v_ptr(&rbm, hk_sample, vk_prob);
                sample_bernoulli(vk_prob, vk_sample, V_SIZE);
            }
            
            // Negative Phase Gradient Calculation: v^k -> h^k
            v_to_h_ptr(&rbm, vk_sample, hn_prob);
            memcpy(vn_sample, vk_sample, V_SIZE * sizeof(float));

            // Update Weights
            update_weights(&rbm, v0, h0_prob, vn_sample, hn_prob);
            
            // Calculate reconstruction error for stats (v0 vs v1_prob)
            h_to_v_ptr(&rbm, h0_prob, vk_prob);
            float mse = 0.0f;
            for(int k = 0; k < V_SIZE; k++) {
                float diff = v0[k] - vk_prob[k];
                mse += diff * diff;
            }
            epoch_mse += (mse / V_SIZE);
        }

        epoch_mse /= TRAINING_SET_SIZE;

        // --- 10-SECOND REPORTING AND EVALUATION ---
        time_t current_time = time(NULL);
        if (current_time - last_report_time >= REPORT_INTERVAL_SECONDS || epoch == MAX_EPOCHS - 1) {
            elapsed_time = (float)(current_time - start_time);
            
            eval_result = run_evaluation(&rbm);

            printf("%5d | %4.1fs | %.8f | %.8f | %5d | %5d | %5d | %5d\n", 
                   epoch + 1, elapsed_time, epoch_mse, eval_result.avg_mse,
                   eval_result.error_counts[0], eval_result.error_counts[1],
                   eval_result.error_counts[2], eval_result.error_counts[3]);
                   
            last_report_time = current_time;
        }

        epoch++;
    }

    // Final evaluation if training loop was interrupted by time limit
    if (epoch < MAX_EPOCHS) {
        eval_result = run_evaluation(&rbm);
        elapsed_time = (float)(time(NULL) - start_time);
        printf("Final   | %4.1fs | N/A            | %.8f | %5d | %5d | %5d | %5d\n", 
               elapsed_time, eval_result.avg_mse,
               eval_result.error_counts[0], eval_result.error_counts[1],
               eval_result.error_counts[2], eval_result.error_counts[3]);
    }


    printf("------------------------------------------------------------------------------------------------\n");
    printf("Training complete. Final Epoch: %d. Final Eval MSE: %.8f\n", epoch, eval_result.avg_mse);
    return 0;
}
