#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define INPUT_DIM 100       // 10x10 image flattened
#define HIDDEN_DIM 10       // The dimension for the invertible matrix (N)
#define N HIDDEN_DIM        // Alias for the matrix size
#define EPOCHS 1500000       // Increased to match user environment
#define LEARNING_RATE 0.02  // INCREASED to help escape the 0.5 minimum
#define IMAGE_SIZE 10

// --- 1. Linear Algebra Utilities & Core Analysis ---

// Helper for generating uniform random float in [min, max]
float rand_uniform(float min, float max) {
    return (max - min) * ((float)rand() / RAND_MAX) + min;
}

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Matrix-Vector multiplication: y = A * x (M x N_A * N_A x 1 -> M x 1)
void mat_vec_mul(const float *A, int M, int N_A, const float *x, float *y) {
    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        for (int j = 0; j < N_A; j++) {
            sum += A[i * N_A + j] * x[j];
        }
        y[i] = sum;
    }
}

// Vector addition/subtraction (y = x + b)
void vec_add(const float *x, const float *b, int D, float *y) {
    for (int i = 0; i < D; i++) {
        y[i] = x[i] + b[i];
    }
}

// Vector normalization
float vec_normalize(float *vec, int D) {
    float norm = 0.0f;
    for (int i = 0; i < D; i++) {
        norm += vec[i] * vec[i];
    }
    norm = sqrtf(norm);
    if (norm > 1e-6) {
        for (int i = 0; i < D; i++) {
            vec[i] /= norm;
        }
    }
    return norm;
}

/**
 * Calculates the inverse of the N x N matrix W and its determinant using
 * Gaussian-Jordan elimination on an augmented matrix [W | I]. (O(N^3))
 */
float inverse_and_determinant(const float *W_in, float *W_inv) {
    float W_aug[N * 2 * N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            W_aug[i * (2 * N) + j] = W_in[i * N + j]; // Copy W
            W_aug[i * (2 * N) + j + N] = (i == j) ? 1.0f : 0.0f; // Copy I
        }
    }

    float det = 1.0f;

    for (int i = 0; i < N; i++) {
        int pivot = i;
        for (int k = i + 1; k < N; k++) {
            if (fabs(W_aug[k * (2 * N) + i]) > fabs(W_aug[pivot * (2 * N) + i])) {
                pivot = k;
            }
        }

        if (pivot != i) {
            for (int j = 0; j < 2 * N; j++) {
                float temp = W_aug[i * (2 * N) + j];
                W_aug[i * (2 * N) + j] = W_aug[pivot * (2 * N) + j];
                W_aug[pivot * (2 * N) + j] = temp;
            }
            det *= -1.0f;
        }

        float pivot_val = W_aug[i * (2 * N) + i];
        if (fabs(pivot_val) < 1e-9) {
            memset(W_inv, 0, N * N * sizeof(float));
            return 0.0f;
        }

        det *= pivot_val;

        for (int j = i; j < 2 * N; j++) {
            W_aug[i * (2 * N) + j] /= pivot_val;
        }

        for (int k = 0; k < N; k++) {
            if (k != i) {
                float factor = W_aug[k * (2 * N) + i];
                for (int j = i; j < 2 * N; j++) {
                    W_aug[k * (2 * N) + j] -= factor * W_aug[i * (2 * N) + j];
                }
            }
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            W_inv[i * N + j] = W_aug[i * (2 * N) + j + N];
        }
    }

    return det;
}

/**
 * Finds the dominant eigenvalue (largest magnitude) and its eigenvector
 * using the Power Iteration method. (O(N^2) per iteration)
 */
void power_iteration(const float *A, float *eigenvalue, float *eigenvector) {
    const int max_iterations = 50;
    const float tolerance = 1e-6f;

    for (int i = 0; i < N; i++) {
        eigenvector[i] = rand_uniform(-0.5f, 0.5f); // Use new helper
    }
    vec_normalize(eigenvector, N);

    float y[N];
    float lambda_prev = 0.0f;

    for (int iter = 0; iter < max_iterations; iter++) {
        mat_vec_mul(A, N, N, eigenvector, y);

        float lambda = 0.0f;
        for (int i = 0; i < N; i++) {
            lambda += y[i] * eigenvector[i];
        }

        if (fabs(lambda - lambda_prev) < tolerance) {
            *eigenvalue = lambda;
            return;
        }

        lambda_prev = lambda;

        memcpy(eigenvector, y, N * sizeof(float));
        vec_normalize(eigenvector, N);
    }

    *eigenvalue = lambda_prev;
}


// --- 2. Cascading Network Structures (N=10 is the constant Hidden/Invertible size) ---

// Stage 1: Input 100 -> Hidden 10 -> Output 1 (5 Networks)
typedef struct {
    float W_feat[N * INPUT_DIM]; // 10x100
    float b_feat[N];
    float W_inv[N * N]; // 10x10 (Invertible Core)
    float b_inv[N];
    float W_out[N]; // 1x10
    float b_out;
    // Internal States for Backprop
    float h1[N]; // Output of L1 (tanh activation)
    float h2[N]; // Output of L2 (linear, pre-output)
    float z_out; // Pre-sigmoid output
} NetS1;

// Stage 2: Input 5 -> Hidden 10 -> Output 1 (3 Networks)
typedef struct {
    float W_feat[N * 5]; // 10x5
    float b_feat[N];
    float W_inv[N * N]; // 10x10 (Invertible Core)
    float b_inv[N];
    float W_out[N]; // 1x10
    float b_out;
    float h1[N]; 
    float h2[N];
    float z_out;
} NetS2;

// Stage 3: Input 3 -> Hidden 10 -> Output 1 (2 Networks)
typedef struct {
    float W_feat[N * 3]; // 10x3
    float b_feat[N];
    float W_inv[N * N]; // 10x10 (Invertible Core)
    float b_inv[N];
    float W_out[N]; // 1x10
    float b_out;
    float h1[N]; 
    float h2[N];
    float z_out;
} NetS3;

// Stage 4: Input 2 -> Hidden 10 -> Output 1 (1 Network)
typedef struct {
    float W_feat[N * 2]; // 10x2
    float b_feat[N];
    float W_inv[N * N]; // 10x10 (Invertible Core)
    float b_inv[N];
    float W_out[N]; // 1x10
    float b_out;
    float h1[N]; 
    float h2[N];
    float z_out;
} NetS4;


// --- 3. Initialization ---

void init_weights(float *W, int M, int K) {
    // Glorot/Xavier uniform initialization scale: range +/- sqrt(1/fan_in)
    // This is crucial for avoiding vanishing gradients in the first layer (S1)
    float scale = sqrtf(1.0f / (float)K); 
    for (int i = 0; i < M * K; i++) W[i] = rand_uniform(-scale, scale);
}

void init_bias(float *b, int M) {
    // Small bias initialization
    for (int i = 0; i < M; i++) b[i] = rand_uniform(-0.05f, 0.05f);
}

void init_invertible_layer(float *W, float *b) {
    for (int i = 0; i < N * N; i++) W[i] = rand_uniform(-0.05f, 0.05f);
    for (int i = 0; i < N; i++) W[i * N + i] += 1.0f; // Near Identity
    init_bias(b, N);
}

void init_net_s1(NetS1 *net) {
    init_weights(net->W_feat, N, INPUT_DIM);
    init_bias(net->b_feat, N);
    init_invertible_layer(net->W_inv, net->b_inv);
    init_weights(net->W_out, 1, N); // Use W_out as 1xN vector
    init_bias(&net->b_out, 1);
}

// Initialization for S2, S3, S4 follows the same pattern, adjusted for dimensions
void init_net_s2(NetS2 *net) { init_weights(net->W_feat, N, 5); init_bias(net->b_feat, N); init_invertible_layer(net->W_inv, net->b_inv); init_weights(net->W_out, 1, N); init_bias(&net->b_out, 1); }
void init_net_s3(NetS3 *net) { init_weights(net->W_feat, N, 3); init_bias(net->b_feat, N); init_invertible_layer(net->W_inv, net->b_inv); init_weights(net->W_out, 1, N); init_bias(&net->b_out, 1); }
void init_net_s4(NetS4 *net) { init_weights(net->W_feat, N, 2); init_bias(net->b_feat, N); init_invertible_layer(net->W_inv, net->b_inv); init_weights(net->W_out, 1, N); init_bias(&net->b_out, 1); }


// --- 4. Forward Pass Functions ---

// Returns the network's output (1x1 float)
float forward_s1(NetS1 *net, const float *input) {
    mat_vec_mul(net->W_feat, N, INPUT_DIM, input, net->h1);
    vec_add(net->h1, net->b_feat, N, net->h1);
    for (int i = 0; i < N; i++) net->h1[i] = tanh(net->h1[i]); // L1 Activation

    mat_vec_mul(net->W_inv, N, N, net->h1, net->h2);
    vec_add(net->h2, net->b_inv, N, net->h2); // L2 (Invertible) Linear

    float z_out = 0.0f;
    for (int i = 0; i < N; i++) z_out += net->W_out[i] * net->h2[i];
    z_out += net->b_out;

    net->z_out = z_out; // Store pre-sigmoid for backprop
    return sigmoid(z_out);
}

float forward_s2(NetS2 *net, const float *input) {
    mat_vec_mul(net->W_feat, N, 5, input, net->h1);
    vec_add(net->h1, net->b_feat, N, net->h1);
    for (int i = 0; i < N; i++) net->h1[i] = tanh(net->h1[i]);

    mat_vec_mul(net->W_inv, N, N, net->h1, net->h2);
    vec_add(net->h2, net->b_inv, N, net->h2);

    float z_out = 0.0f;
    for (int i = 0; i < N; i++) z_out += net->W_out[i] * net->h2[i];
    z_out += net->b_out;

    net->z_out = z_out;
    return sigmoid(z_out);
}

float forward_s3(NetS3 *net, const float *input) {
    mat_vec_mul(net->W_feat, N, 3, input, net->h1);
    vec_add(net->h1, net->b_feat, N, net->h1);
    for (int i = 0; i < N; i++) net->h1[i] = tanh(net->h1[i]);

    mat_vec_mul(net->W_inv, N, N, net->h1, net->h2);
    vec_add(net->h2, net->b_inv, N, net->h2);

    float z_out = 0.0f;
    for (int i = 0; i < N; i++) z_out += net->W_out[i] * net->h2[i];
    z_out += net->b_out;

    net->z_out = z_out;
    return sigmoid(z_out);
}

float forward_s4(NetS4 *net, const float *input) {
    mat_vec_mul(net->W_feat, N, 2, input, net->h1);
    vec_add(net->h1, net->b_feat, N, net->h1);
    for (int i = 0; i < N; i++) net->h1[i] = tanh(net->h1[i]);

    mat_vec_mul(net->W_inv, N, N, net->h1, net->h2);
    vec_add(net->h2, net->b_inv, N, net->h2);

    float z_out = 0.0f;
    for (int i = 0; i < N; i++) z_out += net->W_out[i] * net->h2[i];
    z_out += net->b_out;

    net->z_out = z_out;
    return sigmoid(z_out);
}


// --- 5. Backpropagation Functions ---

// MODIFIED: Calculates and returns delta_input (size 2) to the previous stage (S3)
void backward_s4(NetS4 *net, const float *input, float delta_out, float *delta_input) {
    float grad_W_out[N];
    float delta_h2[N];
    for (int i = 0; i < N; i++) {
        grad_W_out[i] = delta_out * net->h2[i];
        delta_h2[i] = delta_out * net->W_out[i];
    }
    float grad_b_out = delta_out;

    // Backprop L2 (Invertible Core)
    float grad_W_inv[N * N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            grad_W_inv[i * N + j] = delta_h2[i] * net->h1[j];
        }
    }
    float delta_h1[N]; // dLoss/d(h1_pre_tanh)
    for (int i = 0; i < N; i++) {
        delta_h1[i] = 0.0f;
        for (int j = 0; j < N; j++) {
            delta_h1[i] += delta_h2[j] * net->W_inv[j * N + i];
        }
    }
    float *grad_b_inv = delta_h2;

    // Backprop L1 (dLoss/d(h1_pre_tanh) * tanh')
    for (int i = 0; i < N; i++) {
        delta_h1[i] *= (1.0f - net->h1[i] * net->h1[i]); // Tanh prime
    }
    float grad_W_feat[N * 2];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < 2; j++) {
            grad_W_feat[i * 2 + j] = delta_h1[i] * input[j];
        }
    }
    float *grad_b_feat = delta_h1;

    // *** Calculate delta_input (size 2) for the previous stage (S3 outputs) ***
    for (int i = 0; i < 2; i++) {
        delta_input[i] = 0.0f;
        for (int j = 0; j < N; j++) {
            delta_input[i] += delta_h1[j] * net->W_feat[j * 2 + i];
        }
    }

    // Update Weights
    for (int i = 0; i < N; i++) net->W_out[i] -= LEARNING_RATE * grad_W_out[i];
    net->b_out -= LEARNING_RATE * grad_b_out;
    for (int i = 0; i < N * N; i++) net->W_inv[i] -= LEARNING_RATE * grad_W_inv[i];
    for (int i = 0; i < N; i++) net->b_inv[i] -= LEARNING_RATE * grad_b_inv[i];
    for (int i = 0; i < N * 2; i++) net->W_feat[i] -= LEARNING_RATE * grad_W_feat[i];
    for (int i = 0; i < N; i++) net->b_feat[i] -= LEARNING_RATE * grad_b_feat[i];
}


// Generates the delta for the INPUT of this stage, to be passed back to the previous stage (NetS3 input size = 3)
void backward_s3(NetS3 *net, const float *input, float delta_out, float *delta_input) {
    float grad_W_out[N];
    float delta_h2[N];
    for (int i = 0; i < N; i++) {
        grad_W_out[i] = delta_out * net->h2[i];
        delta_h2[i] = delta_out * net->W_out[i];
    }
    float grad_b_out = delta_out;

    float grad_W_inv[N * N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            grad_W_inv[i * N + j] = delta_h2[i] * net->h1[j];
        }
    }
    float delta_h1[N];
    for (int i = 0; i < N; i++) {
        delta_h1[i] = 0.0f;
        for (int j = 0; j < N; j++) {
            delta_h1[i] += delta_h2[j] * net->W_inv[j * N + i];
        }
    }
    float *grad_b_inv = delta_h2;

    for (int i = 0; i < N; i++) {
        delta_h1[i] *= (1.0f - net->h1[i] * net->h1[i]);
    }
    float grad_W_feat[N * 3];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < 3; j++) {
            grad_W_feat[i * 3 + j] = delta_h1[i] * input[j];
        }
    }
    float *grad_b_feat = delta_h1;

    // Calculate delta_input (size 3) for the previous stage (S2 outputs)
    for (int i = 0; i < 3; i++) {
        delta_input[i] = 0.0f;
        for (int j = 0; j < N; j++) {
            delta_input[i] += delta_h1[j] * net->W_feat[j * 3 + i];
        }
    }

    // Update Weights
    for (int i = 0; i < N; i++) net->W_out[i] -= LEARNING_RATE * grad_W_out[i];
    net->b_out -= LEARNING_RATE * grad_b_out;
    for (int i = 0; i < N * N; i++) net->W_inv[i] -= LEARNING_RATE * grad_W_inv[i];
    for (int i = 0; i < N; i++) net->b_inv[i] -= LEARNING_RATE * grad_b_inv[i];
    for (int i = 0; i < N * 3; i++) net->W_feat[i] -= LEARNING_RATE * grad_W_feat[i];
    for (int i = 0; i < N; i++) net->b_feat[i] -= LEARNING_RATE * grad_b_feat[i];
}

// Backward functions for S2 and S1 follow a similar structure, adjusting dimensions (5 and 100)
void backward_s2(NetS2 *net, const float *input, float delta_out, float *delta_input) {
    float grad_W_out[N], delta_h2[N];
    for (int i = 0; i < N; i++) { grad_W_out[i] = delta_out * net->h2[i]; delta_h2[i] = delta_out * net->W_out[i]; }
    float grad_b_out = delta_out;

    float grad_W_inv[N * N];
    for (int i = 0; i < N * N; i++) grad_W_inv[i] = delta_h2[i / N] * net->h1[i % N];

    float delta_h1[N];
    for (int i = 0; i < N; i++) { delta_h1[i] = 0.0f; for (int j = 0; j < N; j++) delta_h1[i] += delta_h2[j] * net->W_inv[j * N + i]; }
    float *grad_b_inv = delta_h2;

    for (int i = 0; i < N; i++) delta_h1[i] *= (1.0f - net->h1[i] * net->h1[i]);

    float grad_W_feat[N * 5];
    for (int i = 0; i < N; i++) { for (int j = 0; j < 5; j++) grad_W_feat[i * 5 + j] = delta_h1[i] * input[j]; }
    float *grad_b_feat = delta_h1;

    for (int i = 0; i < 5; i++) {
        delta_input[i] = 0.0f;
        for (int j = 0; j < N; j++) delta_input[i] += delta_h1[j] * net->W_feat[j * 5 + i];
    }

    for (int i = 0; i < N; i++) net->W_out[i] -= LEARNING_RATE * grad_W_out[i];
    net->b_out -= LEARNING_RATE * grad_b_out;
    for (int i = 0; i < N * N; i++) net->W_inv[i] -= LEARNING_RATE * grad_W_inv[i];
    for (int i = 0; i < N; i++) net->b_inv[i] -= LEARNING_RATE * grad_b_inv[i];
    for (int i = 0; i < N * 5; i++) net->W_feat[i] -= LEARNING_RATE * grad_W_feat[i];
    for (int i = 0; i < N; i++) net->b_feat[i] -= LEARNING_RATE * grad_b_feat[i];
}

// S1 is the first stage, so it doesn't need to return a delta_input.
void backward_s1(NetS1 *net, const float *input, float delta_out) {
    float grad_W_out[N], delta_h2[N];
    for (int i = 0; i < N; i++) { grad_W_out[i] = delta_out * net->h2[i]; delta_h2[i] = delta_out * net->W_out[i]; }
    float grad_b_out = delta_out;

    float grad_W_inv[N * N];
    for (int i = 0; i < N * N; i++) grad_W_inv[i] = delta_h2[i / N] * net->h1[i % N];

    float delta_h1[N];
    for (int i = 0; i < N; i++) { delta_h1[i] = 0.0f; for (int j = 0; j < N; j++) delta_h1[i] += delta_h2[j] * net->W_inv[j * N + i]; }
    float *grad_b_inv = delta_h2;

    for (int i = 0; i < N; i++) delta_h1[i] *= (1.0f - net->h1[i] * net->h1[i]);

    float grad_W_feat[N * INPUT_DIM];
    for (int i = 0; i < N; i++) { for (int j = 0; j < INPUT_DIM; j++) grad_W_feat[i * INPUT_DIM + j] = delta_h1[i] * input[j]; }
    float *grad_b_feat = delta_h1;

    for (int i = 0; i < N; i++) net->W_out[i] -= LEARNING_RATE * grad_W_out[i];
    net->b_out -= LEARNING_RATE * grad_b_out;
    for (int i = 0; i < N * N; i++) net->W_inv[i] -= LEARNING_RATE * grad_W_inv[i];
    for (int i = 0; i < N; i++) net->b_inv[i] -= LEARNING_RATE * grad_b_inv[i];
    for (int i = 0; i < N * INPUT_DIM; i++) net->W_feat[i] -= LEARNING_RATE * grad_W_feat[i];
    for (int i = 0; i < N; i++) net->b_feat[i] -= LEARNING_RATE * grad_b_feat[i];
}


// --- 6. Data Generation ---

void make_rectangle(float *img, int is_present) {
    for (int i = 0; i < INPUT_DIM; i++) img[i] = 0.0f;

    if (is_present) {
        int start_row = rand() % (IMAGE_SIZE - 4);
        int start_col = rand() % (IMAGE_SIZE - 4);
        int width = 3 + (rand() % 3);
        int height = 3 + (rand() % 3);

        for (int r = 0; r < IMAGE_SIZE; r++) {
            for (int c = 0; c < IMAGE_SIZE; c++) {
                if (r > start_row && r < start_row + height && c > start_col && c < start_col + width) {
                    if ((r + c) % 2 == 0) {
                        img[r * IMAGE_SIZE + c] = 1.0f;
                    } else {
                        img[r * IMAGE_SIZE + c] = 0.5f;
                    }
                }
            }
        }
    }
}


// --- 7. Main Program ---

int main() {
    srand(time(NULL));

    NetS1 nets1[5];
    NetS2 nets2[3];
    NetS3 nets3[2];
    NetS4 net_final; // Only 1 network in Stage 4

    for (int i = 0; i < 5; i++) init_net_s1(&nets1[i]);
    for (int i = 0; i < 3; i++) init_net_s2(&nets2[i]);
    for (int i = 0; i < 2; i++) init_net_s3(&nets3[i]);
    init_net_s4(&net_final);

    printf("Starting Cascaded Deep Ensemble Training (5->3->2->1 structure).)\n");
    printf("Total Networks: 11. Core Invertible Matrix size: %d x %d.\n", N, N);

    float input_image[INPUT_DIM];
    float target;
    float avg_loss = 0.0f;

    // Buffers for cascading data flow
    float output_s1[5], output_s2[3], output_s3[2];
    float delta_out_s3[2]; // dLoss/d(OutputS3), calculated by backward_s4
    float delta_s3_in[3]; // dLoss/d(OutputS2), accumulated from S3 nets
    float delta_s2_in[5]; // dLoss/d(OutputS1), accumulated from S2 nets

    float W_inv_copy[N * N];
    float W_inverse[N * N];
    float dominant_eigenvalue;
    float dominant_eigenvector[N];

    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        // --- 1. Prepare Data
        int is_present = rand() % 2;
        target = (float)is_present;
        make_rectangle(input_image, is_present);

        // --- 2. Cascading Forward Pass ---
        for (int i = 0; i < 5; i++) output_s1[i] = forward_s1(&nets1[i], input_image);
        for (int i = 0; i < 3; i++) output_s2[i] = forward_s2(&nets2[i], output_s1);
        for (int i = 0; i < 2; i++) output_s3[i] = forward_s3(&nets3[i], output_s2);
        float final_output = forward_s4(&net_final, output_s3);

        // --- 3. Compute Loss
        float loss = (target - final_output) * (target - final_output);
        avg_loss = avg_loss * 0.99f + loss * 0.01f;

        // --- 4. Cascading Backward Pass ---

        // Final Layer Error (dLoss/d(z_out))
        float delta_final = (final_output - target) * final_output * (1.0f - final_output);

        // *** STAGE 4 BACKPROP (Corrected Flow) ***
        // S4 updates its weights AND returns the gradient w.r.t its input (delta_out_s3)
        backward_s4(&net_final, output_s3, delta_final, delta_out_s3); 
        
        // --- STAGE 3 BACKPROP (2 Nets) ---
        memset(delta_s3_in, 0, sizeof(delta_s3_in));
        for(int i = 0; i < 2; i++) {
            float current_delta_input[3]; // dLoss/d(OutputS2) for this single net
            // delta_out_s3[i] is the error signal (dLoss/d(OutputS3[i])) for nets3[i]
            backward_s3(&nets3[i], output_s2, delta_out_s3[i], current_delta_input); 

            // Accumulate the delta_input (size 3) for S2
            for(int j = 0; j < 3; j++) delta_s3_in[j] += current_delta_input[j];
        }

        // --- STAGE 2 BACKPROP (3 Nets) ---
        memset(delta_s2_in, 0, sizeof(delta_s2_in));
        for(int i = 0; i < 3; i++) {
            float current_delta_input[5]; // dLoss/d(OutputS1) for this single net
            // delta_s3_in[i] is the error signal (dLoss/d(OutputS2[i])) for nets2[i]
            backward_s2(&nets2[i], output_s1, delta_s3_in[i], current_delta_input); 

            // Accumulate the delta_input (size 5) for S1
            for(int j = 0; j < 5; j++) delta_s2_in[j] += current_delta_input[j];
        }

        // --- STAGE 1 BACKPROP (5 Nets) ---
        for(int i = 0; i < 5; i++) {
            // delta_s2_in[i] is the error signal (dLoss/d(OutputS1[i])) for nets1[i]
            backward_s1(&nets1[i], input_image, delta_s2_in[i]);
        }


        // --- 5. Periodic O(N^3) Analysis and Reporting ---
        if (epoch % (EPOCHS / 10) == 0) {
            printf("\n--- Epoch %d/%d ---\n", epoch, EPOCHS);
            printf("Target: %.0f | Final Output: %.4f | Smooth Loss: %.6f\n", target, final_output, avg_loss);

            // Analysis on the final stage's 10x10 Invertible Matrix
            printf("--- High-Demand Matrix Analysis (Net S4, O(N^3)) ---\n");

            // A. Inverse and Determinant (Gaussian-Jordan Elimination: O(N^3))
            memcpy(W_inv_copy, net_final.W_inv, N * N * sizeof(float));
            float det = inverse_and_determinant(W_inv_copy, W_inverse); 

            printf("   [DETERMINANT] Calculated value: %.6f\n", det);
            if (fabs(det) < 1e-6) {
                printf("   [CRITICAL] Matrix is SINGULAR (Det $\\approx 0$). Invertible constraint violation!\n");
            } else {
                printf("   [INVERSE] W_inv is invertible. (Top-Left Element of W_inv: %.4f)\n", W_inverse[0]);
            }

            // B. Dominant Eigenvalue (Power Iteration: O(N^2) per iteration)
            power_iteration(net_final.W_inv, &dominant_eigenvalue, dominant_eigenvector);

            printf("   [EIGENVALUE] Dominant $\\lambda$: %.6f\n", dominant_eigenvalue);
            printf("   [EIGENVECTOR] Dominant vector $v$ ($v_1$ to $v_3$): (%.4f, %.4f, %.4f) ...\n",
                   dominant_eigenvector[0], dominant_eigenvector[1], dominant_eigenvector[2]);
            printf("   [INTERPRET] The largest eigenvalue ($\\|\\lambda\\| > 1$) indicates directional feature expansion.\n");
        }
    }

    printf("\nTraining complete. Final smooth loss: %.6f\n", avg_loss);

    // --- Testing Example ---
    float test_image[INPUT_DIM];
    make_rectangle(test_image, 1);
    
    for (int i = 0; i < 5; i++) output_s1[i] = forward_s1(&nets1[i], test_image);
    for (int i = 0; i < 3; i++) output_s2[i] = forward_s2(&nets2[i], output_s1);
    for (int i = 0; i < 2; i++) output_s3[i] = forward_s3(&nets3[i], output_s2);
    float final_output_present = forward_s4(&net_final, output_s3);

    printf("\nTEST (Rectangle Present): Final Output = %.4f (Expected near 1.0)\n", final_output_present);

    make_rectangle(test_image, 0);
    for (int i = 0; i < 5; i++) output_s1[i] = forward_s1(&nets1[i], test_image);
    for (int i = 0; i < 3; i++) output_s2[i] = forward_s2(&nets2[i], output_s1);
    for (int i = 0; i < 2; i++) output_s3[i] = forward_s3(&nets3[i], output_s2);
    float final_output_absent = forward_s4(&net_final, output_s3);

    printf("TEST (No Rectangle): Final Output = %.4f (Expected near 0.0)\n", final_output_absent);

    return 0;
}