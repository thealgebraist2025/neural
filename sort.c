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
#define PI 3.14159265358979323846f

// --- New Regularization Constants ---
// Run 3 (Det Only) uses 1e-5 Det, 0 Sparse
// Run 4 (Det + L1) now uses these lower values for a better chance at classification
#define REGULARIZATION_LAMBDA_DET_LOW 1e-6f  // Reduced from 1e-5f
#define REGULARIZATION_LAMBDA_SPARSE_LOW 1e-7f // Reduced from 1e-6f
#define DETERMINANT_EPSILON 1e-12f           
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

// --- 2. Network Structures and Result Storage (Updated for Test Rotated) ---

typedef struct { float W_feat[N * INPUT_DIM]; float b_feat[N]; float W_inv[N * N]; float b_inv[N]; float W_out[N]; float b_out; float h1[N]; float h2[N]; float z_out; } NetS1;
typedef struct { float W_feat[N * 5]; float b_feat[N]; float W_inv[N * N]; float b_inv[N]; float W_out[N]; float b_out; float h1[N]; float h2[N]; float z_out; } NetS2;
typedef struct { float W_feat[N * 3]; float b_feat[N]; float W_inv[N * N]; float b_inv[N]; float W_out[N]; float b_out; float h1[N]; float h2[N]; float z_out; } NetS3;
typedef struct { float W_feat[N * 2]; float b_feat[N]; float W_inv[N * N]; float b_inv[N]; float W_out[N]; float b_out; float h1[N]; float h2[N]; float z_out; } NetS4;
typedef struct { float W1[N * INPUT_DIM]; float b1[N]; float W2[N]; float b2; float h1[N]; } NetNaive;

typedef struct {
    float smooth_loss;
    float test_present;
    float test_absent;
    float test_rotated; // New metric for rotated rectangle
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
    // Step 1: Feature Extraction
    float h1_pre[N]; mat_vec_mul(net->W_feat, N, 2, input, h1_pre); vec_add(h1_pre, net->b_feat, N, h1_pre); 
    for (int i = 0; i < N; i++) net->h1[i] = ReLU(h1_pre[i]); 
    
    // Step 2: Invertible Core Layer
    float h2_pre[N]; mat_vec_mul(net->W_inv, N, N, net->h1, h2_pre); vec_add(h2_pre, net->b_inv, N, net->h2); 
    
    // Step 3: Final Output
    float z_out = 0.0f; for (int i = 0; i < N; i++) z_out += net->W_out[i] * net->h2[i]; z_out += net->b_out; 
    net->z_out = z_out; return sigmoid(z_out);
}
float forward_naive(NetNaive *net, const float *input) {
    float h1_pre[N]; mat_vec_mul(net->W1, N, INPUT_DIM, input, h1_pre); vec_add(h1_pre, net->b1, N, h1_pre);
    float h1_act[N]; for (int i = 0; i < N; i++) { h1_act[i] = ReLU(h1_pre[i]); net->h1[i] = h1_pre[i]; }
    float z_out = 0.0f; for (int i = 0; i < N; i++) z_out += net->W2[i] * h1_act[i]; z_out += net->b2;
    return sigmoid(z_out);
}

// --- 5. Symbolic Sanity Check Function (Unchanged) ---

float calculate_symbolic_output(NetS4 *net, const float *input) {
    float h1_pre[N]; mat_vec_mul(net->W_feat, N, 2, input, h1_pre); vec_add(h1_pre, net->b_feat, N, h1_pre);
    float h1[N]; for (int i = 0; i < N; i++) { h1[i] = ReLU(h1_pre[i]); }
    float h2[N]; float h2_pre[N]; mat_vec_mul(net->W_inv, N, N, h1, h2_pre); vec_add(h2_pre, net->b_inv, N, h2);
    float z_out = 0.0f; for (int i = 0; i < N; i++) { z_out += net->W_out[i] * h2[i]; } z_out += net->b_out;
    return sigmoid(z_out);
}

// --- 6. Backpropagation Functions (Simplified Updates for Readability) ---

#define ADVANCED_BACKWARD_UPDATES(LR, REG_GRAD_W_INV, IN_DIM) \
    for (int i = 0; i < N; i++) net->W_out[i] -= LR * clip_gradient(grad_W_out[i]); \
    net->b_out -= LR * clip_gradient(grad_b_out); \
    for (int i = 0; i < N * N; i++) net->W_inv[i] -= LR * clip_gradient(grad_W_inv[i] + REG_GRAD_W_INV[i]); \
    for (int i = 0; i < N; i++) net->b_inv[i] -= LR * clip_gradient(grad_b_inv[i]); \
    for (int i = 0; i < N * IN_DIM; i++) net->W_feat[i] -= LR * clip_gradient(grad_W_feat[i]); \
    for (int i = 0; i < N; i++) net->b_feat[i] -= LR * clip_gradient(grad_b_feat[i]);

void backward_s4(NetS4 *net, const float *input, float delta_out, float *delta_input, float lr, const float *reg_grad_W_inv) {
    float grad_W_out[N], delta_h2[N]; for (int i = 0; i < N; i++) { grad_W_out[i] = delta_out * net->h2[i]; delta_h2[i] = delta_out * net->W_out[i]; } float grad_b_out = delta_out;
    float grad_W_inv[N * N]; for (int i = 0; i < N * N; i++) grad_W_inv[i] = delta_h2[i / N] * net->h1[i % N];
    float delta_h1[N]; for (int i = 0; i < N; i++) { delta_h1[i] = 0.0f; for (int j = 0; j < N; j++) delta_h1[i] += delta_h2[j] * net->W_inv[j * N + i]; }
    float *grad_b_inv = delta_h2; for (int i = 0; i < N; i++) delta_h1[i] *= (net->h1[i] > 0.0f ? 1.0f : 0.0f);
    float grad_W_feat[N * 2]; for (int i = 0; i < N; i++) { for (int j = 0; j < 2; j++) grad_W_feat[i * 2 + j] = delta_h1[i] * input[j]; }
    float *grad_b_feat = delta_h1; for (int i = 0; i < 2; i++) { delta_input[i] = 0.0f; for (int j = 0; j < N; j++) delta_input[i] += delta_h1[j] * net->W_feat[j * 2 + i]; }
    ADVANCED_BACKWARD_UPDATES(lr, reg_grad_W_inv, 2)
}
void backward_s3(NetS3 *net, const float *input, float delta_out, float *delta_input, float lr, const float *reg_grad_W_inv) {
    float grad_W_out[N], delta_h2[N]; for (int i = 0; i < N; i++) { grad_W_out[i] = delta_out * net->h2[i]; delta_h2[i] = delta_out * net->W_out[i]; } float grad_b_out = delta_out;
    float grad_W_inv[N * N]; for (int i = 0; i < N * N; i++) grad_W_inv[i] = delta_h2[i / N] * net->h1[i % N];
    float delta_h1[N]; for (int i = 0; i < N; i++) { delta_h1[i] = 0.0f; for (int j = 0; j < N; j++) delta_h1[i] += delta_h2[j] * net->W_inv[j * N + i]; }
    float *grad_b_inv = delta_h2; for (int i = 0; i < N; i++) delta_h1[i] *= (net->h1[i] > 0.0f ? 1.0f : 0.0f);
    float grad_W_feat[N * 3]; for (int i = 0; i < N; i++) { for (int j = 0; j < 3; j++) grad_W_feat[i * 3 + j] = delta_h1[i] * input[j]; }
    float *grad_b_feat = delta_h1; for (int i = 0; i < 3; i++) { delta_input[i] = 0.0f; for (int j = 0; j < N; j++) delta_input[i] += delta_h1[j] * net->W_feat[j * 3 + i]; }
    ADVANCED_BACKWARD_UPDATES(lr, reg_grad_W_inv, 3)
}
void backward_s2(NetS2 *net, const float *input, float delta_out, float *delta_input, float lr, const float *reg_grad_W_inv) {
    float grad_W_out[N], delta_h2[N]; for (int i = 0; i < N; i++) { grad_W_out[i] = delta_out * net->h2[i]; delta_h2[i] = delta_out * net->W_out[i]; } float grad_b_out = delta_out;
    float grad_W_inv[N * N]; for (int i = 0; i < N * N; i++) grad_W_inv[i] = delta_h2[i / N] * net->h1[i % N];
    float delta_h1[N]; for (int i = 0; i < N; i++) { delta_h1[i] = 0.0f; for (int j = 0; j < N; j++) delta_h1[i] += delta_h2[j] * net->W_inv[j * N + i]; }
    float *grad_b_inv = delta_h2; for (int i = 0; i < N; i++) delta_h1[i] *= (net->h1[i] > 0.0f ? 1.0f : 0.0f);
    float grad_W_feat[N * 5]; for (int i = 0; i < N; i++) { for (int j = 0; j < 5; j++) grad_W_feat[i * 5 + j] = delta_h1[i] * input[j]; }
    float *grad_b_feat = delta_h1; for (int i = 0; i < 5; i++) { delta_input[i] = 0.0f; for (int j = 0; j < N; j++) delta_input[i] += delta_h1[j] * net->W_feat[j * 5 + i]; }
    ADVANCED_BACKWARD_UPDATES(lr, reg_grad_W_inv, 5)
}
void backward_s1(NetS1 *net, const float *input, float delta_out, float lr, const float *reg_grad_W_inv) {
    float grad_W_out[N], delta_h2[N]; for (int i = 0; i < N; i++) { grad_W_out[i] = delta_out * net->h2[i]; delta_h2[i] = delta_out * net->W_out[i]; } float grad_b_out = delta_out;
    float grad_W_inv[N * N]; for (int i = 0; i < N * N; i++) grad_W_inv[i] = delta_h2[i / N] * net->h1[i % N];
    float delta_h1[N]; for (int i = 0; i < N; i++) { delta_h1[i] = 0.0f; for (int j = 0; j < N; j++) delta_h1[i] += delta_h2[j] * net->W_inv[j * N + i]; }
    float *grad_b_inv = delta_h2; for (int i = 0; i < N; i++) delta_h1[i] *= (net->h1[i] > 0.0f ? 1.0f : 0.0f);
    float grad_W_feat[N * INPUT_DIM]; for (int i = 0; i < N; i++) { for (int j = 0; j < INPUT_DIM; j++) grad_W_feat[i * INPUT_DIM + j] = delta_h1[i] * input[j]; }
    float *grad_b_feat = delta_h1;
    ADVANCED_BACKWARD_UPDATES(lr, reg_grad_W_inv, INPUT_DIM)
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
    for (int i = 0; i < N; i++) net->W2[i] -= lr * clip_gradient(grad_W2[i]);
    net->b2 -= lr * clip_gradient(grad_b2);
    for (int i = 0; i < N * INPUT_DIM; i++) net->W1[i] -= lr * clip_gradient(grad_W1[i]);
    for (int i = 0; i < N; i++) net->b1[i] -= lr * clip_gradient(grad_b1[i]);
}

// --- 7. Data Generation (Modified for Rotation) ---

void make_rectangle(float *img, int is_present, int is_rotated) {
    for (int i = 0; i < INPUT_DIM; i++) img[i] = 0.0f;

    if (is_present) {
        // Define unrotated rectangle properties
        int start_row = rand() % (IMAGE_SIZE - 4);
        int start_col = rand() % (IMAGE_SIZE - 4);
        int width = 3 + (rand() % 3);
        int height = 3 + (rand() % 3);

        // Calculate center of the rectangle
        float center_r = start_row + height / 2.0f;
        float center_c = start_col + width / 2.0f;
        
        // Random angle (0 to 45 degrees, converted to radians)
        float angle = is_rotated ? rand_uniform(0.0f, 45.0f) * PI / 180.0f : 0.0f;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);

        // Iterate over every pixel in the 10x10 image grid
        for (int r = 0; r < IMAGE_SIZE; r++) {
            for (int c = 0; c < IMAGE_SIZE; c++) {
                
                // 1. Translate point (r, c) relative to the rectangle center
                float tr = r - center_r;
                float tc = c - center_c;
                
                // 2. Rotate point (tr, tc) backward by -angle
                // r_prime = tr * cos(-angle) - tc * sin(-angle)
                // c_prime = tr * sin(-angle) + tc * cos(-angle)
                // Since cos(-a) = cos(a) and sin(-a) = -sin(a):
                float r_prime = tr * cos_a + tc * sin_a;
                float c_prime = -tr * sin_a + tc * cos_a;
                
                // 3. Check if the back-rotated point falls within the unrotated boundaries
                // Check relative to the center, or relative to (0,0) in the new frame
                float half_w = width / 2.0f;
                float half_h = height / 2.0f;

                if (r_prime >= -half_h && r_prime <= half_h && 
                    c_prime >= -half_w && c_prime <= half_w) 
                {
                    // If it is inside, set pixel value (using a checkerboard pattern for complexity)
                    if (((int)roundf(r_prime * 2.0f) + (int)roundf(c_prime * 2.0f)) % 2 == 0) {
                        img[r * IMAGE_SIZE + c] = 1.0f;
                    } else {
                        img[r * IMAGE_SIZE + c] = 0.5f;
                    }
                }
            }
        }
    }
}


// --- 8. Modular Training Function (Updated for L1 constants and testing) ---

TrainingResults run_training(int is_advanced, float lr, const char *method_name, const char *optimizer_name, int use_sparsity, int perform_sanity_check) {
    
    float input_image[INPUT_DIM];
    float target;
    float avg_loss = 0.0f;
    float prev_avg_loss = 1000.0f;
    float final_output = 0.0f;
    float final_loss = 0.0f;

    NetS1 nets1[5]; NetS2 nets2[3]; NetS3 nets3[2]; NetS4 net_final;
    float output_s1[5], output_s2[3], output_s3[2];
    float delta_out_s3[2], delta_s3_in[3], delta_s2_in[5];
    float W_inv_copy[N * N], W_inverse[N * N];
    float dominant_eigenvalue = 0.0f;
    float dominant_eigenvector[N];
    float det = 0.0f;
    float reg_grad_W_inv[N * N]; 
    float zero_reg_grad[N*N] = {0.0f};

    NetNaive net_naive;

    if (is_advanced) {
        for (int i = 0; i < 5; i++) init_net_s1(&nets1[i]);
        for (int i = 0; i < 3; i++) init_net_s2(&nets2[i]);
        for (int i = 0; i < 2; i++) init_net_s3(&nets3[i]);
        init_net_s4(&net_final);
    } else {
        init_net_naive(&net_naive);
    }
    
    // Determine regularization constants based on run
    float lambda_det = (use_sparsity) ? REGULARIZATION_LAMBDA_DET_LOW : REGULARIZATION_LAMBDA_DET_LOW * 10.0f;
    float lambda_sparse = (use_sparsity) ? REGULARIZATION_LAMBDA_SPARSE_LOW : 0.0f;

    printf("\n\n#####################################################\n");
    printf("--- RUN: %s (%s Optimizer) ---\n", method_name, optimizer_name);
    printf("Learning Rate: %.4f | Hidden Dim (N): %d\n", lr, N);
    if (is_advanced) {
        printf("Penalty: Det $\\lambda$=%.0e | Sparse $\\lambda$=%.0e\n", lambda_det, lambda_sparse);
    }
    printf("#####################################################\n");

    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        int is_present = rand() % 2;
        int do_rotate = is_present && (rand() % 2); // 50% chance of rotation if present
        target = (float)is_present;
        make_rectangle(input_image, is_present, do_rotate); // Use new data generator

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
        memset(reg_grad_W_inv, 0, N * N * sizeof(float)); 
        
        if (is_advanced) {
            memcpy(W_inv_copy, net_final.W_inv, N * N * sizeof(float));
            det = inverse_and_determinant(W_inv_copy, W_inverse);

            float det_sq_safe = det * det + DETERMINANT_EPSILON;
            reg_loss_det = lambda_det / det_sq_safe;
            final_loss += reg_loss_det;
            
            float dL_reg_d_det = -lambda_det * 2.0f * det / (det_sq_safe * det_sq_safe);
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    reg_grad_W_inv[i * N + j] = dL_reg_d_det * det * W_inverse[j * N + i]; 
                }
            }

            if (use_sparsity) {
                float l1_norm = 0.0f;
                for (int i = 0; i < N * N; i++) {
                    l1_norm += fabs(net_final.W_inv[i]);
                }
                reg_loss_sparse = l1_norm * lambda_sparse;
                final_loss += reg_loss_sparse;
                
                for (int i = 0; i < N * N; i++) {
                    float sign_w = 0.0f;
                    if (net_final.W_inv[i] > 1e-6) { sign_w = 1.0f; }
                    else if (net_final.W_inv[i] < -1e-6) { sign_w = -1.0f; }
                    reg_grad_W_inv[i] += lambda_sparse * sign_w; 
                }
            }
        }

        // --- Update Smooth Loss ---
        avg_loss = avg_loss * 0.99f + final_loss * 0.01f;

        // --- Backward Pass & Update ---
        float delta_final = (final_output - target) * final_output * (1.0f - final_output);
        
        if (is_advanced) {
            backward_s4(&net_final, output_s3, delta_final, delta_out_s3, lr, reg_grad_W_inv); 
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
            const char *progress_status = (avg_loss < prev_avg_loss - 1e-4) ? "IMPROVING" : (fabs(avg_loss - prev_avg_loss) < 1e-4 ? "STAGNANT" : "REGRESSING");
            prev_avg_loss = avg_loss;
            printf("[Epoch %d] Loss: %.6f | Progress: %s\n", epoch, avg_loss, progress_status);
            
            if (is_advanced && epoch % (EPOCHS / 10) == 0) {
                memcpy(W_inv_copy, net_final.W_inv, N * N * sizeof(float));
                float det_report = inverse_and_determinant(W_inv_copy, W_inverse); 
                power_iteration(net_final.W_inv, &dominant_eigenvalue, dominant_eigenvector);
                printf("   [DETERMINANT] Calculated value: %.6f (Penalty: %.6f)\n", det_report, reg_loss_det);
                if (use_sparsity) { printf("   [SPARSITY] L1 Loss: %.6f\n", reg_loss_sparse); }
                printf("   [EIGENVALUE] Dominant $\\lambda$: %.6f\n", dominant_eigenvalue);
            }
        }
    }
    printf("\nTraining complete. Final smooth total loss: %.6f\n", avg_loss);

    // --- SYMBOLIC SANITY CHECK after training ---
    if (perform_sanity_check) {
        float test_input[2] = {1.0f, 0.5f};
        float symbolic_output = calculate_symbolic_output(&net_final, test_input);
        float actual_output = forward_s4(&net_final, test_input);
        printf("\n======================================================\n");
        printf("--- SYMBOLIC SANITY CHECK (Net S4 Forward Pass) ---\n");
        printf("Expected Symbolic Output: %.10f\n", symbolic_output);
        printf("Actual forward_s4() Output: %.10f\n", actual_output);
        if (fabs(symbolic_output - actual_output) < 1e-9) {
            printf("[SUCCESS] Symbolic and Actual Outputs match within tolerance $\\approx 1e-9$.\n");
        } else {
            printf("[FAILURE] Outputs do not match. Discrepancy: %.10f\n", fabs(symbolic_output - actual_output));
        }
        printf("======================================================\n");
    }

    // --- Final Testing (Modified to include Rotated Test Case) ---
    float final_output_present, final_output_absent, final_output_rotated;
    
    // 1. Test - Standard Rectangle
    make_rectangle(input_image, 1, 0); // is_present=1, is_rotated=0
    if (is_advanced) {
        for (int i = 0; i < 5; i++) output_s1[i] = forward_s1(&nets1[i], input_image);
        for (int i = 0; i < 3; i++) output_s2[i] = forward_s2(&nets2[i], output_s1);
        for (int i = 0; i < 2; i++) output_s3[i] = forward_s3(&nets3[i], output_s2);
        final_output_present = forward_s4(&net_final, output_s3);
    } else {
        final_output_present = forward_naive(&net_naive, input_image);
    }

    // 2. Test - Absent Rectangle
    make_rectangle(input_image, 0, 0); // is_present=0, is_rotated=0
    if (is_advanced) {
        for (int i = 0; i < 5; i++) output_s1[i] = forward_s1(&nets1[i], input_image);
        for (int i = 0; i < 3; i++) output_s2[i] = forward_s2(&nets2[i], output_s1);
        for (int i = 0; i < 2; i++) output_s3[i] = forward_s3(&nets3[i], output_s2);
        final_output_absent = forward_s4(&net_final, output_s3);
    } else {
        final_output_absent = forward_naive(&net_naive, input_image);
    }

    // 3. Test - Rotated Rectangle (NEW)
    make_rectangle(input_image, 1, 1); // is_present=1, is_rotated=1
    if (is_advanced) {
        for (int i = 0; i < 5; i++) output_s1[i] = forward_s1(&nets1[i], input_image);
        for (int i = 0; i < 3; i++) output_s2[i] = forward_s2(&nets2[i], output_s1);
        for (int i = 0; i < 2; i++) output_s3[i] = forward_s3(&nets3[i], output_s2);
        final_output_rotated = forward_s4(&net_final, output_s3);
    } else {
        final_output_rotated = forward_naive(&net_naive, input_image);
    }

    printf("TEST (Standard Present): Final Output = %.4f\n", final_output_present);
    printf("TEST (No Rectangle): Final Output = %.4f\n", final_output_absent);
    printf("TEST (Rotated Present): Final Output = %.4f\n", final_output_rotated);

    if (is_advanced) {
        memcpy(W_inv_copy, net_final.W_inv, N * N * sizeof(float));
        det = inverse_and_determinant(W_inv_copy, W_inverse);
        power_iteration(net_final.W_inv, &dominant_eigenvalue, dominant_eigenvector);
    }
    
    return (TrainingResults){
        .smooth_loss = avg_loss,
        .test_present = final_output_present,
        .test_absent = final_output_absent,
        .test_rotated = final_output_rotated, // Store new metric
        .det_final = is_advanced ? det : 0.0f,
        .lambda_final = is_advanced ? dominant_eigenvalue : 0.0f,
        .method_name = method_name,
        .optimizer_name = optimizer_name,
        .learning_rate = lr
    };
}

void print_final_summary(const TrainingResults *results[4]) {
    printf("\n\n===================================================================================================================================\n");
    printf("                         COMPREHENSIVE TRAINING COMPARISON REPORT (Rotated Data & Regularization)\n");
    printf("===================================================================================================================================\n");
    printf("| Architecture | Optimizer | LR    | Penalty Type | Smooth Loss | Test (Present) | Test (Absent) | Test (Rotated) | Det (W_inv) | Lambda (Dom) |\n");
    printf("|--------------|-----------|-------|--------------|-------------|----------------|---------------|----------------|-------------|--------------|\n");
    
    for(int i = 0; i < 4; i++) {
        const TrainingResults *res = results[i];
        const char *penalty_type = (i == 2) ? "Det Only (1e-5)" : (i == 3 ? "Det (1e-6) + L1 (1e-7)" : "None");
        
        printf("| %-12s | %-9s | %-5.2f | %-12s | %-11.6f | %-14.4f | %-13.4f | %-14.4f | %-11.6f | %-12.6f |\n",
               res->method_name,
               res->optimizer_name,
               res->learning_rate,
               penalty_type,
               res->smooth_loss,
               res->test_present,
               res->test_absent,
               res->test_rotated, // Print new metric
               res->det_final,
               res->lambda_final);
    }
    printf("===================================================================================================================================\n");
    printf("NOTE: The training data now includes randomly rotated rectangles (up to $45^{\\circ}$).\n");
    printf("The 'Det + L1' run used reduced $\\lambda$ values for this complex task.\n");
}


int main() {
    srand(time(NULL));

    TrainingResults *results[4];

    // SCENARIO 1: Advanced Network, Aggressive Optimizer (Det Only - Uses 1e-5 Det)
    results[0] = (TrainingResults*)malloc(sizeof(TrainingResults));
    *results[0] = run_training(1, 0.07f, "Advanced", "Aggressive", 0, 0);

    // SCENARIO 2: Naive Network, Standard Optimizer (Baseline - Expected Det = 0)
    results[1] = (TrainingResults*)malloc(sizeof(TrainingResults));
    *results[1] = run_training(0, 0.03f, "Naive", "Standard", 0, 0);
    
    // SCENARIO 3: Advanced Network, Stable Optimizer (Det Only - Uses 1e-5 Det & Sanity Check)
    results[2] = (TrainingResults*)malloc(sizeof(TrainingResults));
    // This run uses the original strong 1e-5 lambda_det
    *results[2] = run_training(1, 0.01f, "Advanced", "Stable", 0, 1); 

    // SCENARIO 4: Advanced Network, Stable Optimizer (Det + L1 Sparsity - Uses Lowered 1e-6 Det & 1e-7 L1)
    results[3] = (TrainingResults*)malloc(sizeof(TrainingResults));
    *results[3] = run_training(1, 0.01f, "Advanced", "Stable (L1)", 1, 0);

    print_final_summary(results);

    for(int i = 0; i < 4; i++) {
        free(results[i]);
    }

    return 0;
}