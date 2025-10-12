#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define INPUT_DIM 100
#define HIDDEN_DIM 20       
#define N HIDDEN_DIM        
#define EPOCHS 150000       
#define IMAGE_SIZE 10
#define GRADIENT_CLIP_MAX 0.1f 
#define PROGRESS_CHECK_INTERVAL 100 

// --- New Regularization Constants ---
#define REGULARIZATION_LAMBDA_DET 1e-5f      // Strength of the determinant penalty
#define REGULARIZATION_LAMBDA_SPARSE 1e-6f   // Strength of the L1 sparsity penalty (NEW)
#define DETERMINANT_EPSILON 1e-12f           // Small value for numerical stability near 0
// ------------------------------------

// --- 1. Utilities and Analysis ---

#define ReLU(x) ((x) > 0.0f ? (x) : 0.0f)
float rand_uniform(float min, float max) { return (max - min) * ((float)rand() / RAND_MAX) + min; }
float clip_gradient(float grad) {
    if (grad > GRADIENT_CLIP_MAX) return GRADIENT_CLIP_MAX;
    if (grad < -GRADIENT_CLIP_MAX) return -GRADIENT_CLIP_MAX;
    return grad;
}
float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
void mat_vec_mul(const float *A, int M, int N_A, const float *x, float *y) {
    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        for (int j = 0; j < N_A; j++) { sum += A[i * N_A + j] * x[j]; }
        y[i] = sum;
    }
}
void vec_add(const float *x, const float *b, int D, float *y) {
    for (int i = 0; i < D; i++) { y[i] = x[i] + b[i]; }
}

// Function to calculate Inverse and Determinant using Gaussian-Jordan elimination
float inverse_and_determinant(const float *W_in, float *W_inv) {
    float W_aug[N * 2 * N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) { W_aug[i * (2 * N) + j] = W_in[i * N + j]; W_aug[i * (2 * N) + j + N] = (i == j) ? 1.0f : 0.0f; }
    }
    float det = 1.0f;
    for (int i = 0; i < N; i++) {
        int pivot = i;
        for (int k = i + 1; k < N; k++) { if (fabs(W_aug[k * (2 * N) + i]) > fabs(W_aug[pivot * (2 * N) + i])) { pivot = k; } }
        if (fabs(W_aug[pivot * (2 * N) + i]) < 1e-9) { return 0.0f; } // Singular
        if (pivot != i) {
            for (int j = 0; j < 2 * N; j++) { float temp = W_aug[i * (2 * N) + j]; W_aug[i * (2 * N) + j] = W_aug[pivot * (2 * N) + j]; W_aug[pivot * (2 * N) + j] = temp; }
            det *= -1.0f;
        }
        float pivot_val = W_aug[i * (2 * N) + i];
        det *= pivot_val;
        for (int j = i; j < 2 * N; j++) { W_aug[i * (2 * N) + j] /= pivot_val; }
        for (int k = 0; k < N; k++) {
            if (k != i) { float factor = W_aug[k * (2 * N) + i]; for (int j = i; j < 2 * N; j++) { W_aug[k * (2 * N) + j] -= factor * W_aug[i * (2 * N) + j]; } }
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) { W_inv[i * N + j] = W_aug[i * (2 * N) + j + N]; }
    }
    return det;
}

void power_iteration(const float *A, float *eigenvalue, float *eigenvector) {
    const int max_iterations = 50;
    const float tolerance = 1e-6f;
    for (int i = 0; i < N; i++) { eigenvector[i] = rand_uniform(-0.5f, 0.5f); }
    float norm = 0.0f;
    for (int i = 0; i < N; i++) norm += eigenvector[i] * eigenvector[i];
    norm = sqrtf(norm);
    if (norm > 1e-6) { for (int i = 0; i < N; i++) eigenvector[i] /= norm; }
    float y[N];
    float lambda_prev = 0.0f;
    for (int iter = 0; iter < max_iterations; iter++) {
        mat_vec_mul(A, N, N, eigenvector, y);
        float lambda = 0.0f;
        for (int i = 0; i < N; i++) { lambda += y[i] * eigenvector[i]; }
        if (fabs(lambda - lambda_prev) < tolerance) { *eigenvalue = lambda; return; }
        lambda_prev = lambda;
        memcpy(eigenvector, y, N * sizeof(float));
        norm = 0.0f;
        for (int i = 0; i < N; i++) norm += eigenvector[i] * eigenvector[i];
        norm = sqrtf(norm);
        if (norm > 1e-6) { for (int i = 0; i < N; i++) eigenvector[i] /= norm; }
    }
    *eigenvalue = lambda_prev;
}

// --- 2. Network Structures and Result Storage ---

// Advanced Method: Cascading Ensemble with Invertible Core (W_inv)
typedef struct { float W_feat[N * INPUT_DIM]; float b_feat[N]; float W_inv[N * N]; float b_inv[N]; float W_out[N]; float b_out; float h1[N]; float h2[N]; float z_out; } NetS1;
typedef struct { float W_feat[N * 5]; float b_feat[N]; float W_inv[N * N]; float b_inv[N]; float W_out[N]; float b_out; float h1[N]; float h2[N]; float z_out; } NetS2;
typedef struct { float W_feat[N * 3]; float b_feat[N]; float W_inv[N * N]; float b_inv[N]; float W_out[N]; float b_out; float h1[N]; float h2[N]; float z_out; } NetS3;
typedef struct { float W_feat[N * 2]; float b_feat[N]; float W_inv[N * N]; float b_inv[N]; float W_out[N]; float b_out; float h1[N]; float h2[N]; float z_out; } NetS4;

// Naive Method: Single, Flat Network
typedef struct { float W1[N * INPUT_DIM]; float b1[N]; float W2[N]; float b2; float h1[N]; } NetNaive;

typedef struct {
    float smooth_loss;
    float test_present;
    float test_absent;
    float det_final;
    float lambda_final;
    const char *method_name;
    const char *optimizer_name;
    float learning_rate;
} TrainingResults;


// --- 3. Initialization (Unchanged) ---

void init_weights_he(float *W, int M, int K) {
    float scale = sqrtf(2.0f / (float)K); 
    for (int i = 0; i < M * K; i++) W[i] = rand_uniform(-scale, scale);
}
void init_bias(float *b, int M) { for (int i = 0; i < M; i++) b[i] = 0.0f; }
void init_invertible_layer(float *W, float *b) {
    for (int i = 0; i < N * N; i++) W[i] = rand_uniform(-0.05f, 0.05f);
    for (int i = 0; i < N; i++) W[i * N + i] += 1.0f;
    init_bias(b, N);
}

void init_net_s1(NetS1 *net) { init_weights_he(net->W_feat, N, INPUT_DIM); init_bias(net->b_feat, N); init_invertible_layer(net->W_inv, net->b_inv); init_weights_he(net->W_out, 1, N); init_bias(&net->b_out, 1); }
void init_net_s2(NetS2 *net) { init_weights_he(net->W_feat, N, 5); init_bias(net->b_feat, N); init_invertible_layer(net->W_inv, net->b_inv); init_weights_he(net->W_out, 1, N); init_bias(&net->b_out, 1); }
void init_net_s3(NetS3 *net) { init_weights_he(net->W_feat, N, 3); init_bias(net->b_feat, N); init_invertible_layer(net->W_inv, net->b_inv); init_weights_he(net->W_out, 1, N); init_bias(&net->b_out, 1); }
void init_net_s4(NetS4 *net) { init_weights_he(net->W_feat, N, 2); init_bias(net->b_feat, N); init_invertible_layer(net->W_inv, net->b_inv); init_weights_he(net->W_out, 1, N); init_bias(&net->b_out, 1); }
void init_net_naive(NetNaive *net) { init_weights_he(net->W1, N, INPUT_DIM); init_bias(net->b1, N); init_weights_he(net->W2, 1, N); init_bias(&net->b2, 1); }


// --- 4. Forward Pass Functions (Unchanged) ---

float forward_s1(NetS1 *net, const float *input) {
    float h1_pre[N]; mat_vec_mul(net->W_feat, N, INPUT_DIM, input, h1_pre); vec_add(h1_pre, net->b_feat, N, h1_pre); 
    for (int i = 0; i < N; i++) net->h1[i] = ReLU(h1_pre[i]); 
    float h2_pre[N]; mat_vec_mul(net->W_inv, N, N, net->h1, h2_pre); vec_add(h2_pre, net->b_inv, N, net->h2); 
    float z_out = 0.0f; for (int i = 0; i < N; i++) z_out += net->W_out[i] * net->h2[i]; z_out += net->b_out; 
    net->z_out = z_out; return sigmoid(z_out);
}
float forward_s2(NetS2 *net, const float *input) {
    float h1_pre[N]; mat_vec_mul(net->W_feat, N, 5, input, h1_pre); vec_add(h1_pre, net->b_feat, N, h1_pre); 
    for (int i = 0; i < N; i++) net->h1[i] = ReLU(h1_pre[i]); 
    float h2_pre[N]; mat_vec_mul(net->W_inv, N, N, net->h1, h2_pre); vec_add(h2_pre, net->b_inv, N, net->h2); 
    float z_out = 0.0f; for (int i = 0; i < N; i++) z_out += net->W_out[i] * net->h2[i]; z_out += net->b_out; 
    net->z_out = z_out; return sigmoid(z_out);
}
float forward_s3(NetS3 *net, const float *input) {
    float h1_pre[N]; mat_vec_mul(net->W_feat, N, 3, input, h1_pre); vec_add(h1_pre, net->b_feat, N, h1_pre); 
    for (int i = 0; i < N; i++) net->h1[i] = ReLU(h1_pre[i]); 
    float h2_pre[N]; mat_vec_mul(net->W_inv, N, N, net->h1, h2_pre); vec_add(h2_pre, net->b_inv, N, net->h2); 
    float z_out = 0.0f; for (int i = 0; i < N; i++) z_out += net->W_out[i] * net->h2[i]; z_out += net->b_out; 
    net->z_out = z_out; return sigmoid(z_out);
}
float forward_s4(NetS4 *net, const float *input) {
    float h1_pre[N]; mat_vec_mul(net->W_feat, N, 2, input, h1_pre); vec_add(h1_pre, net->b_feat, N, h1_pre); 
    for (int i = 0; i < N; i++) net->h1[i] = ReLU(h1_pre[i]); 
    float h2_pre[N]; mat_vec_mul(net->W_inv, N, N, net->h1, h2_pre); vec_add(h2_pre, net->b_inv, N, net->h2); 
    float z_out = 0.0f; for (int i = 0; i < N; i++) z_out += net->W_out[i] * net->h2[i]; z_out += net->b_out; 
    net->z_out = z_out; return sigmoid(z_out);
}
float forward_naive(NetNaive *net, const float *input) {
    float h1_pre[N]; mat_vec_mul(net->W1, N, INPUT_DIM, input, h1_pre); vec_add(h1_pre, net->b1, N, h1_pre);
    float h1_act[N]; for (int i = 0; i < N; i++) { h1_act[i] = ReLU(h1_pre[i]); net->h1[i] = h1_pre[i]; }
    float z_out = 0.0f; for (int i = 0; i < N; i++) z_out += net->W2[i] * h1_act[i]; z_out += net->b2;
    return sigmoid(z_out);
}


// --- 5. Backpropagation Functions (Modified S4) ---

// Macro for generic updates (used by S1-S4)
// REG_GRAD_W_INV now combines both Det and Sparsity gradients in run_training
#define ADVANCED_BACKWARD_UPDATES(LR, REG_GRAD_W_INV) \
    for (int i = 0; i < N; i++) net->W_out[i] -= LR * clip_gradient(grad_W_out[i]); \
    net->b_out -= LR * clip_gradient(grad_b_out); \
    /* W_inv update includes the new regularization gradient */ \
    for (int i = 0; i < N * N; i++) net->W_inv[i] -= LR * clip_gradient(grad_W_inv[i] + REG_GRAD_W_INV[i]); \
    for (int i = 0; i < N; i++) net->b_inv[i] -= LR * clip_gradient(grad_b_inv[i]); \
    for (int i = 0; i < N * IN_DIM; i++) net->W_feat[i] -= LR * clip_gradient(grad_W_feat[i]); \
    for (int i = 0; i < N; i++) net->b_feat[i] -= LR * clip_gradient(grad_b_feat[i]);

// New: S4 now accepts the pre-calculated combined regularization gradient for W_inv
void backward_s4(NetS4 *net, const float *input, float delta_out, float *delta_input, float lr, const float *reg_grad_W_inv) {
    float grad_W_out[N], delta_h2[N]; for (int i = 0; i < N; i++) { grad_W_out[i] = delta_out * net->h2[i]; delta_h2[i] = delta_out * net->W_out[i]; } float grad_b_out = delta_out;
    float grad_W_inv[N * N]; for (int i = 0; i < N * N; i++) grad_W_inv[i] = delta_h2[i / N] * net->h1[i % N];
    float delta_h1[N]; for (int i = 0; i < N; i++) { delta_h1[i] = 0.0f; for (int j = 0; j < N; j++) delta_h1[i] += delta_h2[j] * net->W_inv[j * N + i]; }
    float *grad_b_inv = delta_h2; for (int i = 0; i < N; i++) delta_h1[i] *= (net->h1[i] > 0.0f ? 1.0f : 0.0f);
    float grad_W_feat[N * 2]; for (int i = 0; i < N; i++) { for (int j = 0; j < 2; j++) grad_W_feat[i * 2 + j] = delta_h1[i] * input[j]; }
    float *grad_b_feat = delta_h1; for (int i = 0; i < 2; i++) { delta_input[i] = 0.0f; for (int j = 0; j < N; j++) delta_input[i] += delta_h1[j] * net->W_feat[j * 2 + i]; }
    #define IN_DIM 2
    ADVANCED_BACKWARD_UPDATES(lr, reg_grad_W_inv)
    #undef IN_DIM
}
// S3, S2, S1 do not use determinant/sparsity regularization, so they use a zero array
void backward_s3(NetS3 *net, const float *input, float delta_out, float *delta_input, float lr, const float *reg_grad_W_inv) {
    float grad_W_out[N], delta_h2[N]; for (int i = 0; i < N; i++) { grad_W_out[i] = delta_out * net->h2[i]; delta_h2[i] = delta_out * net->W_out[i]; } float grad_b_out = delta_out;
    float grad_W_inv[N * N]; for (int i = 0; i < N * N; i++) grad_W_inv[i] = delta_h2[i / N] * net->h1[i % N];
    float delta_h1[N]; for (int i = 0; i < N; i++) { delta_h1[i] = 0.0f; for (int j = 0; j < N; j++) delta_h1[i] += delta_h2[j] * net->W_inv[j * N + i]; }
    float *grad_b_inv = delta_h2; for (int i = 0; i < N; i++) delta_h1[i] *= (net->h1[i] > 0.0f ? 1.0f : 0.0f);
    float grad_W_feat[N * 3]; for (int i = 0; i < N; i++) { for (int j = 0; j < 3; j++) grad_W_feat[i * 3 + j] = delta_h1[i] * input[j]; }
    float *grad_b_feat = delta_h1; for (int i = 0; i < 3; i++) { delta_input[i] = 0.0f; for (int j = 0; j < N; j++) delta_input[i] += delta_h1[j] * net->W_feat[j * 3 + i]; }
    #define IN_DIM 3
    ADVANCED_BACKWARD_UPDATES(lr, reg_grad_W_inv)
    #undef IN_DIM
}
void backward_s2(NetS2 *net, const float *input, float delta_out, float *delta_input, float lr, const float *reg_grad_W_inv) {
    float grad_W_out[N], delta_h2[N]; for (int i = 0; i < N; i++) { grad_W_out[i] = delta_out * net->h2[i]; delta_h2[i] = delta_out * net->W_out[i]; } float grad_b_out = delta_out;
    float grad_W_inv[N * N]; for (int i = 0; i < N * N; i++) grad_W_inv[i] = delta_h2[i / N] * net->h1[i % N];
    float delta_h1[N]; for (int i = 0; i < N; i++) { delta_h1[i] = 0.0f; for (int j = 0; j < N; j++) delta_h1[i] += delta_h2[j] * net->W_inv[j * N + i]; }
    float *grad_b_inv = delta_h2; for (int i = 0; i < N; i++) delta_h1[i] *= (net->h1[i] > 0.0f ? 1.0f : 0.0f);
    float grad_W_feat[N * 5]; for (int i = 0; i < N; i++) { for (int j = 0; j < 5; j++) grad_W_feat[i * 5 + j] = delta_h1[i] * input[j]; }
    float *grad_b_feat = delta_h1; for (int i = 0; i < 5; i++) { delta_input[i] = 0.0f; for (int j = 0; j < N; j++) delta_input[i] += delta_h1[j] * net->W_feat[j * 5 + i]; }
    #define IN_DIM 5
    ADVANCED_BACKWARD_UPDATES(lr, reg_grad_W_inv)
    #undef IN_DIM
}
void backward_s1(NetS1 *net, const float *input, float delta_out, float lr, const float *reg_grad_W_inv) {
    float grad_W_out[N], delta_h2[N]; for (int i = 0; i < N; i++) { grad_W_out[i] = delta_out * net->h2[i]; delta_h2[i] = delta_out * net->W_out[i]; } float grad_b_out = delta_out;
    float grad_W_inv[N * N]; for (int i = 0; i < N * N; i++) grad_W_inv[i] = delta_h2[i / N] * net->h1[i % N];
    float delta_h1[N]; for (int i = 0; i < N; i++) { delta_h1[i] = 0.0f; for (int j = 0; j < N; j++) delta_h1[i] += delta_h2[j] * net->W_inv[j * N + i]; }
    float *grad_b_inv = delta_h2; for (int i = 0; i < N; i++) delta_h1[i] *= (net->h1[i] > 0.0f ? 1.0f : 0.0f);
    float grad_W_feat[N * INPUT_DIM]; for (int i = 0; i < N; i++) { for (int j = 0; j < INPUT_DIM; j++) grad_W_feat[i * INPUT_DIM + j] = delta_h1[i] * input[j]; }
    float *grad_b_feat = delta_h1;
    #define IN_DIM INPUT_DIM
    ADVANCED_BACKWARD_UPDATES(lr, reg_grad_W_inv)
    #undef IN_DIM
}

void backward_naive(NetNaive *net, const float *input, float delta_out, float lr) {
    float grad_W2[N]; float delta_h1_act[N];
    for (int i = 0; i < N; i++) { grad_W2[i] = delta_out * ReLU(net->h1[i]); delta_h1_act[i] = delta_out * net->W2[i]; }
    float grad_b2 = delta_out;
    float delta_h1_pre[N];
    for (int i = 0; i < N; i++) { delta_h1_pre[i] = delta_h1_act[i] * (net->h1[i] > 0.0f ? 1.0f : 0.0f); }
    float grad_W1[N * INPUT_DIM];
    for (int i = 0; i < N; i++) { for (int j = 0; j < INPUT_DIM; j++) { grad_W1[i * INPUT_DIM + j] = delta_h1_pre[i] * input[j]; } }
    float *grad_b1 = delta_h1_pre;
    
    // Update Weights with Clipping (using dynamic LR)
    for (int i = 0; i < N; i++) net->W2[i] -= lr * clip_gradient(grad_W2[i]);
    net->b2 -= lr * clip_gradient(grad_b2);
    for (int i = 0; i < N * INPUT_DIM; i++) net->W1[i] -= lr * clip_gradient(grad_W1[i]);
    for (int i = 0; i < N; i++) net->b1[i] -= lr * clip_gradient(grad_b1[i]);
}


// --- 6. Data Generation (Unchanged) ---

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
                    if ((r + c) % 2 == 0) { img[r * IMAGE_SIZE + c] = 1.0f; } else { img[r * IMAGE_SIZE + c] = 0.5f; }
                }
            }
        }
    }
}


// --- 7. Modular Training Function (Added Sparsity Logic) ---

TrainingResults run_training(int is_advanced, float lr, const char *method_name, const char *optimizer_name, int use_sparsity) {
    
    float input_image[INPUT_DIM];
    float target;
    float avg_loss = 0.0f;
    float prev_avg_loss = 1000.0f;
    float final_output = 0.0f;
    float final_loss = 0.0f;

    // Advanced Network components
    NetS1 nets1[5]; NetS2 nets2[3]; NetS3 nets3[2]; NetS4 net_final;
    float output_s1[5], output_s2[3], output_s3[2];
    float delta_out_s3[2], delta_s3_in[3], delta_s2_in[5];
    float W_inv_copy[N * N], W_inverse[N * N];
    float dominant_eigenvalue = 0.0f;
    float dominant_eigenvector[N];
    float det = 0.0f;
    float reg_grad_W_inv[N * N]; // Combined regularization gradient storage
    float zero_reg_grad[N*N] = {0.0f}; // Placeholder for non-final stages or Naive

    // Naive Network component
    NetNaive net_naive;

    if (is_advanced) {
        for (int i = 0; i < 5; i++) init_net_s1(&nets1[i]);
        for (int i = 0; i < 3; i++) init_net_s2(&nets2[i]);
        for (int i = 0; i < 2; i++) init_net_s3(&nets3[i]);
        init_net_s4(&net_final);
    } else {
        init_net_naive(&net_naive);
    }

    printf("\n\n#####################################################\n");
    printf("--- RUN: %s (%s Optimizer) ---\n", method_name, optimizer_name);
    printf("Learning Rate: %.4f | Hidden Dim (N): %d | Sparsity: %s\n", lr, N, use_sparsity ? "ON" : "OFF");
    printf("#####################################################\n");

    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        int is_present = rand() % 2;
        target = (float)is_present;
        make_rectangle(input_image, is_present);

        // --- Forward Pass ---
        if (is_advanced) {
            for (int i = 0; i < 5; i++) output_s1[i] = forward_s1(&nets1[i], input_image);
            for (int i = 0; i < 3; i++) output_s2[i] = forward_s2(&nets2[i], output_s1);
            for (int i = 0; i < 2; i++) output_s3[i] = forward_s3(&nets3[i], output_s2);
            final_output = forward_s4(&net_final, output_s3);
        } else {
            final_output = forward_naive(&net_naive, input_image);
        }

        // --- MSE Loss ---
        float mse_loss = (target - final_output) * (target - final_output);
        final_loss = mse_loss;
        
        // --- Regularization (Advanced Only) ---
        float reg_loss_det = 0.0f;
        float reg_loss_sparse = 0.0f;
        
        // Reset combined regularization gradient for W_inv
        memset(reg_grad_W_inv, 0, N * N * sizeof(float)); 
        
        if (is_advanced) {
            // 1. Calculate Determinant and Inverse (O(N^3))
            memcpy(W_inv_copy, net_final.W_inv, N * N * sizeof(float));
            det = inverse_and_determinant(W_inv_copy, W_inverse);

            // 2. Calculate Determinant Regularization Loss
            float det_sq_safe = det * det + DETERMINANT_EPSILON;
            reg_loss_det = REGULARIZATION_LAMBDA_DET / det_sq_safe;
            final_loss += reg_loss_det;
            
            // 3. Calculate Determinant Regularization Gradient: dL_reg_det/dW
            float dL_reg_d_det = -REGULARIZATION_LAMBDA_DET * 2.0f * det / (det_sq_safe * det_sq_safe);
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    reg_grad_W_inv[i * N + j] = dL_reg_d_det * det * W_inverse[j * N + i]; 
                }
            }

            // --- L1 Sparsity Regularization (NEW) ---
            if (use_sparsity) {
                float l1_norm = 0.0f;
                for (int i = 0; i < N * N; i++) {
                    l1_norm += fabs(net_final.W_inv[i]);
                }
                reg_loss_sparse = l1_norm * REGULARIZATION_LAMBDA_SPARSE;
                final_loss += reg_loss_sparse;
                
                // 4. Calculate Sparsity Regularization Gradient: dL_reg_sparse/dW = lambda * sign(W)
                for (int i = 0; i < N * N; i++) {
                    float sign_w = 0.0f;
                    if (net_final.W_inv[i] > 1e-6) { sign_w = 1.0f; }
                    else if (net_final.W_inv[i] < -1e-6) { sign_w = -1.0f; }
                    
                    // Additive sparsity gradient
                    reg_grad_W_inv[i] += REGULARIZATION_LAMBDA_SPARSE * sign_w; 
                }
            }
        }

        // --- Update Smooth Loss ---
        avg_loss = avg_loss * 0.99f + final_loss * 0.01f;

        // --- Backward Pass & Update ---
        float delta_final = (final_output - target) * final_output * (1.0f - final_output);
        
        if (is_advanced) {
            // S4 uses the combined gradient in reg_grad_W_inv
            backward_s4(&net_final, output_s3, delta_final, delta_out_s3, lr, reg_grad_W_inv); 
            
            // Inner stages use a zero gradient for regularization
            memset(delta_s3_in, 0, sizeof(delta_s3_in));
            for(int i = 0; i < 2; i++) { float current_delta_input[3]; backward_s3(&nets3[i], output_s2, delta_out_s3[i], current_delta_input, lr, zero_reg_grad); for(int j = 0; j < 3; j++) delta_s3_in[j] += current_delta_input[j]; }
            memset(delta_s2_in, 0, sizeof(delta_s2_in));
            for(int i = 0; i < 3; i++) { float current_delta_input[5]; backward_s2(&nets2[i], output_s1, delta_s3_in[i], current_delta_input, lr, zero_reg_grad); for(int j = 0; j < 5; j++) delta_s2_in[j] += current_delta_input[j]; }
            for(int i = 0; i < 5; i++) { backward_s1(&nets1[i], input_image, delta_s2_in[i], lr, zero_reg_grad); }
        } else {
            backward_naive(&net_naive, input_image, delta_final, lr);
        }

        // --- Sanity Check & Analysis ---
        if (epoch % PROGRESS_CHECK_INTERVAL == 0) {
            
            // Loss Progress Check
            const char *progress_status = "STAGNANT (Loss not improving)";
            if (avg_loss < prev_avg_loss - 1e-4) {
                 progress_status = "IMPROVING";
            } else if (fabs(avg_loss - prev_avg_loss) < 1e-4) {
                 progress_status = "STAGNANT (No measurable change)";
            } else {
                 progress_status = "REGRESSING (Loss increased)";
            }
            prev_avg_loss = avg_loss;

            printf("[Epoch %d] Loss: %.6f | Progress: %s\n", epoch, avg_loss, progress_status);
            
            // Advanced Network Analysis (Only for Advanced runs)
            if (is_advanced && epoch % (EPOCHS / 10) == 0) {
                printf("--- High-Demand Matrix Analysis (Net S4, O(N^3) on %d x %d) ---\n", N, N);
                // Recalculate det/eigenvalue for reporting after updates
                memcpy(W_inv_copy, net_final.W_inv, N * N * sizeof(float));
                float det_report = inverse_and_determinant(W_inv_copy, W_inverse); 
                power_iteration(net_final.W_inv, &dominant_eigenvalue, dominant_eigenvector);
                
                printf("   [DETERMINANT] Calculated value: %.6f (Penalty: %.6f)\n", det_report, reg_loss_det);
                if (use_sparsity) { printf("   [SPARSITY] L1 Loss: %.6f\n", reg_loss_sparse); }

                if (fabs(det_report) < 1e-6) { printf("   [CRITICAL] Matrix is SINGULAR (Det $\\approx 0$).\n"); }
                else { printf("   [STABLE] Matrix is invertible.\n"); }
                printf("   [EIGENVALUE] Dominant $\\lambda$: %.6f\n", dominant_eigenvalue);
            }
        }
    }
    printf("\nTraining complete. Final smooth total loss: %.6f\n", avg_loss);

    // --- Final Testing (Unchanged) ---
    float final_output_present, final_output_absent;
    
    make_rectangle(input_image, 1);
    if (is_advanced) {
        for (int i = 0; i < 5; i++) output_s1[i] = forward_s1(&nets1[i], input_image);
        for (int i = 0; i < 3; i++) output_s2[i] = forward_s2(&nets2[i], output_s1);
        for (int i = 0; i < 2; i++) output_s3[i] = forward_s3(&nets3[i], output_s2);
        final_output_present = forward_s4(&net_final, output_s3);
    } else {
        final_output_present = forward_naive(&net_naive, input_image);
    }

    make_rectangle(input_image, 0);
    if (is_advanced) {
        for (int i = 0; i < 5; i++) output_s1[i] = forward_s1(&nets1[i], input_image);
        for (int i = 0; i < 3; i++) output_s2[i] = forward_s2(&nets2[i], output_s1);
        for (int i = 0; i < 2; i++) output_s3[i] = forward_s3(&nets3[i], output_s2);
        final_output_absent = forward_s4(&net_final, output_s3);
    } else {
        final_output_absent = forward_naive(&net_naive, input_image);
    }

    printf("TEST (Rectangle Present): Final Output = %.4f (Expected near 1.0)\n", final_output_present);
    printf("TEST (No Rectangle): Final Output = %.4f (Expected near 0.0)\n", final_output_absent);

    // Final determinant value is set for the report
    if (is_advanced) {
        memcpy(W_inv_copy, net_final.W_inv, N * N * sizeof(float));
        det = inverse_and_determinant(W_inv_copy, W_inverse);
        power_iteration(net_final.W_inv, &dominant_eigenvalue, dominant_eigenvector);
    }
    
    return (TrainingResults){
        .smooth_loss = avg_loss,
        .test_present = final_output_present,
        .test_absent = final_output_absent,
        .det_final = is_advanced ? det : 0.0f,
        .lambda_final = is_advanced ? dominant_eigenvalue : 0.0f,
        .method_name = method_name,
        .optimizer_name = optimizer_name,
        .learning_rate = lr
    };
}

void print_final_summary(const TrainingResults *results[4]) {
    printf("\n\n=================================================================================================================\n");
    printf("                         COMPREHENSIVE TRAINING COMPARISON REPORT (Determinant & Sparsity Regularization)\n");
    printf("=================================================================================================================\n");
    printf("| Architecture | Optimizer | LR    | Penalty Type | Smooth Loss | Test (Present) | Test (Absent) | Det (W_inv) | Lambda (Dom) |\n");
    printf("|--------------|-----------|-------|--------------|-------------|----------------|---------------|-------------|--------------|\n");
    
    for(int i = 0; i < 4; i++) {
        const TrainingResults *res = results[i];
        printf("| %-12s | %-9s | %-5.2f | %-12s | %-11.6f | %-14.4f | %-13.4f | %-11.6f | %-12.6f |\n",
               res->method_name,
               res->optimizer_name,
               res->learning_rate,
               i == 3 ? "Det + L1" : "Det Only", // Simple way to label the new run
               res->smooth_loss,
               res->test_present,
               res->test_absent,
               res->det_final,
               res->lambda_final);
    }
    printf("=================================================================================================================\n");
    printf("NOTE: Advanced Networks use Det Reg to enforce stability. Run 4 adds the L1 Sparsity penalty.\n");
}


int main() {
    srand(time(NULL));

    // Define the 4 scenarios
    TrainingResults *results[4];

    // SCENARIO 1: Advanced Network, Aggressive Optimizer (Det Only - Expected Fail)
    results[0] = (TrainingResults*)malloc(sizeof(TrainingResults));
    *results[0] = run_training(1, 0.07f, "Advanced", "Aggressive", 0);

    // SCENARIO 2: Naive Network, Standard Optimizer (Baseline - Expected Det = 0)
    results[1] = (TrainingResults*)malloc(sizeof(TrainingResults));
    *results[1] = run_training(0, 0.03f, "Naive", "Standard", 0);
    
    // SCENARIO 3: Advanced Network, Stable Optimizer (Det Only - SUCCESS BASELINE)
    results[2] = (TrainingResults*)malloc(sizeof(TrainingResults));
    *results[2] = run_training(1, 0.01f, "Advanced", "Stable", 0);

    // SCENARIO 4: Advanced Network, Stable Optimizer (Det + L1 Sparsity - NEW TEST)
    // We are running this to see the effect on final loss and structural properties (Det/Lambda)
    results[3] = (TrainingResults*)malloc(sizeof(TrainingResults));
    *results[3] = run_training(1, 0.01f, "Advanced", "Stable (L1)", 1);

    // Print the final summary table
    print_final_summary(results);

    // Cleanup (Freeing allocated memory)
    for(int i = 0; i < 4; i++) {
        free(results[i]);
    }

    return 0;
}