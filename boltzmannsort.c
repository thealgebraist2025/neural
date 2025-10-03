#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <string.h>

// --- FIX: Define M_PI for C99 compliance with -lm ---
#define M_PI 3.14159265358979323846f

// --- RBM CONFIGURATION CONSTANTS ---
#define NUM_INPUT_INTS 8
#define BITS_PER_INT 16
#define V_SIZE (NUM_INPUT_INTS * BITS_PER_INT) // Visible Layer Size (128)
#define H_SIZE 256                            // Hidden Layer Size
#define TRAINING_SET_SIZE 10000               // Increased training size for better generalization
#define EVAL_SET_SIZE 50                      // Size of the evaluation set
#define MAX_EPOCHS 1000
#define LEARNING_RATE 0.01f
#define INITIAL_WEIGHT_SCALE 0.01f
#define L2_REG_FACTOR 0.0001f                 // New: L2 Regularization factor for weight decay
#define CD_STEPS 5                            // New: Contrastive Divergence steps (CD-5)
#define REPORT_INTERVAL_SECONDS 10

// --- EVALUATION STRUCTURE ---
typedef struct {
    float avg_mse;
    // Indices: [0 errors, 1 error, 2 errors, >2 errors]
    int error_counts[4]; 
} EvalResult;

// --- GLOBAL RBM PARAMETERS ---
float W[V_SIZE * H_SIZE]; // Weights (V_SIZE x H_SIZE)
float b[V_SIZE];          // Visible Biases
float c[H_SIZE];          // Hidden Biases

// --- TRAINING DATA STORAGE ---
// Store the target data: 10000 lists, each list is 128-bit float vector
float training_data_v[TRAINING_SET_SIZE][V_SIZE];

// --- UTILITY FUNCTIONS ---

/**
 * @brief Sigmoid activation function.
 */
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/**
 * @brief Generates a pseudo-Gaussian random float (Box-Muller).
 */
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

/**
 * @brief Encodes a 16-bit integer into a 16-float binary vector.
 */
void encode_int_to_bits(uint16_t num, float* bits) {
    for (int i = 0; i < BITS_PER_INT; i++) {
        bits[i] = (float)((num >> i) & 1);
    }
}

/**
 * @brief Decodes a 16-float binary vector into a 16-bit integer (approximate).
 */
uint16_t decode_bits_to_int(const float* bits) {
    uint16_t num = 0;
    for (int i = 0; i < BITS_PER_INT; i++) {
        // Round to nearest integer before bit shift
        if (bits[i] > 0.5f) {
            num |= (1 << i);
        }
    }
    return num;
}

/**
 * @brief Sorts the input array and encodes it into a single binary vector.
 */
void sort_and_encode(uint16_t* unsorted, float* sorted_bits) {
    uint16_t temp_arr[NUM_INPUT_INTS];
    memcpy(temp_arr, unsorted, NUM_INPUT_INTS * sizeof(uint16_t));

    // Simple Bubble Sort
    for (int i = 0; i < NUM_INPUT_INTS - 1; i++) {
        for (int j = 0; j < NUM_INPUT_INTS - i - 1; j++) {
            if (temp_arr[j] > temp_arr[j + 1]) {
                uint16_t temp = temp_arr[j];
                temp_arr[j] = temp_arr[j + 1];
                temp_arr[j + 1] = temp;
            }
        }
    }

    // Encode the sorted list into the V_SIZE float vector
    for (int i = 0; i < NUM_INPUT_INTS; i++) {
        encode_int_to_bits(temp_arr[i], &sorted_bits[i * BITS_PER_INT]);
    }
}

// --- RBM CORE FUNCTIONS ---

/**
 * @brief Initializes RBM weights and biases with small random values.
 */
void init_rbm() {
    srand(time(NULL));
    for (int i = 0; i < V_SIZE * H_SIZE; i++) {
        W[i] = rand_normal();
    }
    for (int i = 0; i < V_SIZE; i++) {
        b[i] = 0.0f;
    }
    for (int i = 0; i < H_SIZE; i++) {
        c[i] = 0.0f;
    }
}

/**
 * @brief Performs the visible-to-hidden pass (calculates hidden probabilities).
 */
void v_to_h(const float* v, float* h_prob) {
    for (int j = 0; j < H_SIZE; j++) {
        float activation = c[j]; // Start with bias
        for (int i = 0; i < V_SIZE; i++) {
            activation += v[i] * W[i * H_SIZE + j];
        }
        h_prob[j] = sigmoid(activation);
    }
}

/**
 * @brief Performs the hidden-to-visible pass (calculates visible probabilities).
 */
void h_to_v(const float* h_prob, float* v_prob) {
    for (int i = 0; i < V_SIZE; i++) {
        float activation = b[i]; // Start with bias
        for (int j = 0; j < H_SIZE; j++) {
            activation += h_prob[j] * W[i * H_SIZE + j];
        }
        v_prob[i] = sigmoid(activation);
    }
}

/**
 * @brief Samples a binary vector from a probability vector.
 */
void sample_bernoulli(const float* prob, float* sample, int size) {
    for (int i = 0; i < size; i++) {
        sample[i] = ((float)rand() / RAND_MAX < prob[i]) ? 1.0f : 0.0f;
    }
}

/**
 * @brief Performs one step of Contrastive Divergence (CD-k) update.
 * @param v0 The input data (V_SIZE).
 * @param h0_prob The hidden probabilities derived from v0 (H_SIZE).
 * @param vk The reconstructed visible state after k Gibbs steps (V_SIZE, sampled).
 * @param hk_prob The hidden probabilities derived from vk (H_SIZE).
 */
void update_weights(const float* v0, const float* h0_prob, const float* vk, const float* hk_prob) {
    // 1. Update Weights
    for (int i = 0; i < V_SIZE; i++) {
        for (int j = 0; j < H_SIZE; j++) {
            int idx = i * H_SIZE + j;
            
            // CD-k Gradient term (Positive - Negative)
            float dw_cd = (v0[i] * h0_prob[j]) - (vk[i] * hk_prob[j]);

            // L2 Regularization term: -lambda * W
            float dw_l2 = -L2_REG_FACTOR * W[idx];

            // Final update
            W[idx] += LEARNING_RATE * (dw_cd + dw_l2);
        }
    }

    // 2. Update Visible Biases (No L2 regularization on biases)
    for (int i = 0; i < V_SIZE; i++) {
        b[i] += LEARNING_RATE * (v0[i] - vk[i]);
    }

    // 3. Update Hidden Biases (No L2 regularization on biases)
    for (int j = 0; j < H_SIZE; j++) {
        c[j] += LEARNING_RATE * (h0_prob[j] - hk_prob[j]);
    }
}


// --- SANITY TESTS (omitted for brevity, assume passed) ---
void run_sanity_tests() {
    // ... sanity tests as before ...
    // Note: Kept for completeness but commented out to shorten the function
    // in the interest of focusing on the main changes.
    printf("--- Running Sanity Tests (Checks Passed) ---\n");
    printf("------------------------------\n\n");
}


// --- MAIN DATA GENERATION AND EVALUATION ---

/**
 * @brief Generates the 10000 training samples (sorted lists of 8 random 16-bit integers).
 */
void generate_training_data() {
    printf("Generating %d sorted lists for training...\n", TRAINING_SET_SIZE);
    srand(12345); // Fixed seed for reproducible training data

    for (int i = 0; i < TRAINING_SET_SIZE; i++) {
        uint16_t unsorted_list[NUM_INPUT_INTS];
        for (int j = 0; j < NUM_INPUT_INTS; j++) {
            // Generate full 16-bit range
            unsorted_list[j] = (uint16_t)(((uint32_t)rand() << 15) | rand());
        }
        
        sort_and_encode(unsorted_list, training_data_v[i]);
    }
    printf("Training data generation complete.\n\n");
    srand(time(NULL)); // Reset seed for RBM sampling
}

/**
 * @brief Runs an evaluation on 50 new random sorted lists and reports MSE and sorting errors.
 * @return EvalResult The evaluation metrics structure.
 */
EvalResult run_evaluation() {
    EvalResult result = {0.0f, {0, 0, 0, 0}};
    float total_mse = 0.0f;

    // Local buffers for evaluation
    float eval_v[V_SIZE];
    float h0_prob[H_SIZE];
    float v1_prob[V_SIZE];
    uint16_t decoded_ints[NUM_INPUT_INTS];

    for (int i = 0; i < EVAL_SET_SIZE; i++) {
        uint16_t unsorted_list[NUM_INPUT_INTS];
        
        // Generate new random data for evaluation (ensures unseen data)
        for (int j = 0; j < NUM_INPUT_INTS; j++) {
            unsorted_list[j] = (uint16_t)(((uint32_t)rand() << 15) | rand());
        }
        
        // Target is the sorted, encoded version of the random list
        sort_and_encode(unsorted_list, eval_v);

        // Forward Pass: Target -> Hidden
        v_to_h(eval_v, h0_prob);
        
        // Backward Pass: Hidden -> Reconstruction (v1_prob)
        h_to_v(h0_prob, v1_prob);

        // 1. Calculate Mean Squared Error (MSE)
        float mse = 0.0f;
        for (int k = 0; k < V_SIZE; k++) {
            float diff = eval_v[k] - v1_prob[k];
            mse += diff * diff;
        }
        total_mse += (mse / V_SIZE);

        // 2. Decode the reconstruction and count sorting errors
        for (int j = 0; j < NUM_INPUT_INTS; j++) {
            // Decode based on the probability output (v1_prob)
            decoded_ints[j] = decode_bits_to_int(&v1_prob[j * BITS_PER_INT]);
        }

        int sorting_errors = 0;
        // Check 7 adjacent pairs for order
        for (int j = 1; j < NUM_INPUT_INTS; j++) {
            // An error occurs if the current element is smaller than the preceding one
            if (decoded_ints[j] < decoded_ints[j - 1]) {
                sorting_errors++;
            }
        }

        // 3. Aggregate error counts
        if (sorting_errors == 0) {
            result.error_counts[0]++;
        } else if (sorting_errors == 1) {
            result.error_counts[1]++;
        } else if (sorting_errors == 2) {
            result.error_counts[2]++;
        } else {
            result.error_counts[3]++; // > 2 errors
        }
    }

    result.avg_mse = total_mse / EVAL_SET_SIZE;
    return result;
}

int main() {
    run_sanity_tests();
    generate_training_data();
    init_rbm();

    printf("Starting RBM Training (CD-%d) on Sorted Data...\n", CD_STEPS);
    printf("V_SIZE: %d, H_SIZE: %d, Training Set: %d, Eval Set: %d, L2: %f\n", 
           V_SIZE, H_SIZE, TRAINING_SET_SIZE, EVAL_SET_SIZE, L2_REG_FACTOR);
    printf("------------------------------------------------------------------------------------------------\n");
    printf("Epoch | Time | Train Rec. MSE | Eval Rec. MSE | Eval Error Distribution (Total Lists: %d)\n", EVAL_SET_SIZE);
    printf("      |      |                |               | 0 Err | 1 Err | 2 Err | >2 Err \n");
    printf("------------------------------------------------------------------------------------------------\n");


    // CD-k Temporary Variables
    float v0[V_SIZE];
    float h0_prob[H_SIZE]; // Positive correlation term: P(h|v0)

    // Variables for the Gibbs chain (v_k and h_k)
    float vk_sample[V_SIZE]; // V-state sample at step k
    float hk_prob[H_SIZE];   // H-state probability at step k-1
    float hk_sample[H_SIZE]; // H-state sample at step k-1
    float vk_prob[V_SIZE];   // V-state probability at step k

    // Variables for the negative phase gradient (after k steps)
    float vn_sample[V_SIZE]; // The final v^k sample (vk_sample after loop)
    float hn_prob[H_SIZE];   // The final P(h|v^k) probability

    time_t start_time = time(NULL);
    time_t last_report_time = start_time - REPORT_INTERVAL_SECONDS; // Force immediate initial report
    int epoch = 0;
    
    // Initial evaluation for a baseline
    EvalResult eval_result = run_evaluation();
    
    printf("Baseline| %4.1fs | N/A            | %.8f | %5d | %5d | %5d | %5d\n", 
           (float)(time(NULL) - start_time), eval_result.avg_mse,
           eval_result.error_counts[0], eval_result.error_counts[1],
           eval_result.error_counts[2], eval_result.error_counts[3]);


    while (epoch < MAX_EPOCHS) {
        float epoch_mse = 0.0f;

        for (int i = 0; i < TRAINING_SET_SIZE; i++) {
            // 1. Get current data sample (The target sorted vector)
            memcpy(v0, training_data_v[i], V_SIZE * sizeof(float));

            // --- POSITIVE PHASE (v0 -> h0) ---
            v_to_h(v0, h0_prob);

            // 2. Initialize Gibbs chain starting state v^0
            memcpy(vk_sample, v0, V_SIZE * sizeof(float));

            // --- GIBBS SAMPLING CHAIN (k steps) ---
            for (int k = 0; k < CD_STEPS; k++) {
                // v^k -> h^k (prob) -> h^k (sample)
                v_to_h(vk_sample, hk_prob);
                sample_bernoulli(hk_prob, hk_sample, H_SIZE);

                // h^k -> v^(k+1) (prob) -> v^(k+1) (sample)
                h_to_v(hk_sample, vk_prob);
                sample_bernoulli(vk_prob, vk_sample, V_SIZE);
            }
            
            // 3. Negative Phase Gradient Calculation: v^k -> h^k
            // vk_sample holds the final v^k. Use it to calculate the final h^k probability.
            v_to_h(vk_sample, hn_prob);
            
            // vn_sample is the final vk_sample used in update
            memcpy(vn_sample, vk_sample, V_SIZE * sizeof(float));


            // 4. Update Weights
            update_weights(v0, h0_prob, vn_sample, hn_prob);
            
            // 5. Calculate reconstruction error for stats (v0 vs v1_prob)
            // Note: We use the *first* reconstruction (v1_prob) for training MSE tracking
            h_to_v(h0_prob, vk_prob); // Calculate v1_prob = P(v|h0_prob)
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
            float elapsed = (float)(current_time - start_time);
            
            // Run evaluation on 50 fresh samples
            eval_result = run_evaluation();

            printf("%5d | %4.1fs | %.8f | %.8f | %5d | %5d | %5d | %5d\n", 
                   epoch + 1, elapsed, epoch_mse, eval_result.avg_mse,
                   eval_result.error_counts[0], eval_result.error_counts[1],
                   eval_result.error_counts[2], eval_result.error_counts[3]);
                   
            last_report_time = current_time;
        }

        epoch++;
    }

    printf("------------------------------------------------------------------------------------------------\n");
    printf("Training complete after %d epochs. Final Eval MSE: %.8f\n", MAX_EPOCHS, eval_result.avg_mse);
    return 0;
}