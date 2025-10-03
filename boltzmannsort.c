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
#define TRAINING_SET_SIZE 256
#define EVAL_SET_SIZE 50                      // Size of the evaluation set
#define MAX_EPOCHS 10000
#define LEARNING_RATE 0.01f
#define INITIAL_WEIGHT_SCALE 0.01f
#define REPORT_INTERVAL_SECONDS 4

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
// Store the target data: 256 lists, each list is 128-bit float vector
float training_data_v[TRAINING_SET_SIZE][V_SIZE];

// --- UTILITY FUNCTIONS ---

/**
 * @brief Sigmoid activation function.
 * @param x Input value.
 * @return float Sigmoid output (0.0 to 1.0).
 */
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/**
 * @brief Generates a pseudo-Gaussian random float (Box-Muller).
 * @return float Random float with mean 0, variance INITIAL_WEIGHT_SCALE^2.
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
 * @param num The 16-bit integer.
 * @param bits Output array of size BITS_PER_INT (16).
 */
void encode_int_to_bits(uint16_t num, float* bits) {
    for (int i = 0; i < BITS_PER_INT; i++) {
        bits[i] = (float)((num >> i) & 1);
    }
}

/**
 * @brief Decodes a 16-float binary vector into a 16-bit integer (approximate).
 * @param bits Input array of size BITS_PER_INT (16).
 * @return uint16_t The decoded integer.
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
 * @param unsorted Array of 8 unsorted 16-bit integers.
 * @param sorted_bits Output array of size V_SIZE (128).
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
 * @param v Input visible vector (V_SIZE).
 * @param h_prob Output hidden probability vector (H_SIZE).
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
 * @param h_prob Input hidden probability vector (H_SIZE).
 * @param v_prob Output visible probability vector (V_SIZE).
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
 * @brief Performs one step of Contrastive Divergence (CD-1) update.
 */
void update_weights(const float* v0, const float* h0_prob, const float* v1, const float* h1_prob) {
    // 1. Update Weights
    for (int i = 0; i < V_SIZE; i++) {
        for (int j = 0; j < H_SIZE; j++) {
            int idx = i * H_SIZE + j;
            float positive_grad = v0[i] * h0_prob[j];
            float negative_grad = v1[i] * h1_prob[j];
            W[idx] += LEARNING_RATE * (positive_grad - negative_grad);
        }
    }

    // 2. Update Visible Biases
    for (int i = 0; i < V_SIZE; i++) {
        b[i] += LEARNING_RATE * (v0[i] - v1[i]);
    }

    // 3. Update Hidden Biases
    for (int j = 0; j < H_SIZE; j++) {
        c[j] += LEARNING_RATE * (h0_prob[j] - h1_prob[j]);
    }
}


// --- SANITY TESTS ---

void run_sanity_tests() {
    printf("--- Running Sanity Tests ---\n");

    // 1. Sigmoid Test
    if (fabsf(sigmoid(0.0f) - 0.5f) < 1e-4f && fabsf(sigmoid(10.0f) - 1.0f) < 1e-4f) {
        printf("[PASS] Sigmoid function.\n");
    } else {
        printf("[FAIL] Sigmoid function.\n");
    }

    // 2. Encoding/Decoding Test
    uint16_t original_int = 0xABCD; // 43981
    float bits[BITS_PER_INT];
    encode_int_to_bits(original_int, bits);
    uint16_t decoded_int = decode_bits_to_int(bits);
    if (original_int == decoded_int) {
        printf("[PASS] Int Encoding/Decoding (0x%X).\n", original_int);
    } else {
        printf("[FAIL] Int Encoding/Decoding. Expected 0x%X, Got 0x%X.\n", original_int, decoded_int);
    }

    // 3. Sorting Test
    uint16_t unsorted_list[NUM_INPUT_INTS] = {500, 10, 800, 5, 20, 100, 900, 1};
    float sorted_bits_out[V_SIZE];
    sort_and_encode(unsorted_list, sorted_bits_out);

    uint16_t decoded_sorted[NUM_INPUT_INTS];
    for(int i=0; i<NUM_INPUT_INTS; i++) {
        decoded_sorted[i] = decode_bits_to_int(&sorted_bits_out[i * BITS_PER_INT]);
    }

    if (decoded_sorted[0] == 1 && decoded_sorted[1] == 5 && decoded_sorted[7] == 900) {
        printf("[PASS] Sorting and Encoding.\n");
    } else {
        printf("[FAIL] Sorting and Encoding. Expected {1, ..., 900}, Got {%u, ..., %u}.\n", decoded_sorted[0], decoded_sorted[7]);
    }

    // 4. Matrix Multiplication Sanity (Positive Phase)
    init_rbm(); 
    float test_v[V_SIZE];
    float test_h_prob[H_SIZE];
    for (int i = 0; i < V_SIZE; i++) test_v[i] = 1.0f;
    
    c[0] = 5.0f; 
    v_to_h(test_v, test_h_prob);

    if (test_h_prob[0] > 0.99f) {
        printf("[PASS] V-to-H (Positive Phase) sanity.\n");
    } else {
        printf("[FAIL] V-to-H (Positive Phase) sanity. Prob[0] = %f\n", test_h_prob[0]);
    }

    printf("------------------------------\n\n");
}


// --- MAIN DATA GENERATION AND TRAINING ---

/**
 * @brief Generates the 256 training samples (sorted lists of 8 random 16-bit integers).
 */
void generate_training_data() {
    printf("Generating %d sorted lists for training...\n", TRAINING_SET_SIZE);
    srand(12345); // Fixed seed for reproducible training data

    for (int i = 0; i < TRAINING_SET_SIZE; i++) {
        uint16_t unsorted_list[NUM_INPUT_INTS];
        // Generate 8 random 16-bit positive integers
        for (int j = 0; j < NUM_INPUT_INTS; j++) {
            // Generate full 16-bit range (0 to 65535)
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
    uint16_t decoded_ints[NUM_INPUT_INTS]; // To hold the 8 decoded integers

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

    printf("Starting RBM Training (CD-1) on Sorted Data...\n");
    printf("V_SIZE: %d, H_SIZE: %d, Training Set: %d, Eval Set: %d\n", V_SIZE, H_SIZE, TRAINING_SET_SIZE, EVAL_SET_SIZE);
    printf("------------------------------------------------------------------------------------------------\n");
    printf("Epoch | Time | Train Rec. MSE | Eval Rec. MSE | Eval Error Distribution (Total Lists: %d)\n", EVAL_SET_SIZE);
    printf("      |      |                |               | 0 Err | 1 Err | 2 Err | >2 Err \n");
    printf("------------------------------------------------------------------------------------------------\n");


    // CD-1 Temporary Variables
    float v0[V_SIZE];
    float h0_prob[H_SIZE];
    float v1_prob[V_SIZE];
    float v1_sample[V_SIZE];
    float h1_prob[H_SIZE];

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

            // --- NEGATIVE PHASE (h0 -> v1 -> h1) ---
            // 2. Reconstruct visible state (v1)
            h_to_v(h0_prob, v1_prob);
            sample_bernoulli(v1_prob, v1_sample, V_SIZE); // Sample v1

            // 3. Get hidden state probabilities from reconstructed visible state (h1)
            v_to_h(v1_sample, h1_prob);

            // 4. Update Weights
            update_weights(v0, h0_prob, v1_sample, h1_prob);
            
            // 5. Calculate reconstruction error for stats (v0 vs v1_prob)
            float mse = 0.0f;
            for(int k = 0; k < V_SIZE; k++) {
                float diff = v0[k] - v1_prob[k];
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