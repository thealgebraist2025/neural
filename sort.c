#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define INPUT_DIM 100
#define HIDDEN_DIM 20       
#define N HIDDEN_DIM        
#define EPOCHS 150000       
#define LEARNING_RATE 0.07  // Advanced Method LR (Aggressive)
#define NAIVE_LEARNING_RATE 0.03 // Naive Method LR (Stable)
#define IMAGE_SIZE 10
#define GRADIENT_CLIP_MAX 0.1f 

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

// O(N^3) Analysis functions (used only by Advanced Method)
float inverse_and_determinant(const float *W_in, float *W_inv) {
    float W_aug[N * 2 * N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            W_aug[i * (2 * N) + j] = W_in[i * N + j];
            W_aug[i * (2 * N) + j + N] = (i == j) ? 1.0f : 0.0f;
        }
    }
    float det = 1.0f;
    for (int i = 0; i < N; i++) {
        int pivot = i;
        for (int k = i + 1; k < N; k++) { if (fabs(W_aug[k * (2 * N) + i]) > fabs(W_aug[pivot * (2 * N) + i])) { pivot = k; } }
        if (pivot != i) {
            for (int j = 0; j < 2 * N; j++) {
                float temp = W_aug[i * (2 * N) + j];
                W_aug[i * (2 * N) + j] = W_aug[pivot * (2 * N) + j];
                W_aug[pivot * (2 * N) + j] = temp;
            }
            det *= -1.0f;
        }
        float pivot_val = W_aug[i * (2 * N) + i];
        if (fabs(pivot_val) < 1e-9) { memset(W_inv, 0, N * N * sizeof(float)); return 0.0f; }
        det *= pivot_val;
        for (int j = i; j < 2 * N; j++) { W_aug[i * (2 * N) + j] /= pivot_val; }
        for (int k = 0; k < N; k++) {
            if (k != i) {
                float factor = W_aug[k * (2 * N) + i];
                for (int j = i; j < 2 * N; j++) { W_aug[k * (2 * N) + j] -= factor * W_aug[i * (2 * N) + j]; }
            }
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

// --- 2. Network Structures ---

// Advanced Method: Cascading Ensemble with Invertible Core (W_inv)
typedef struct { float W_feat[N * INPUT_DIM]; float b_feat[N]; float W_inv[N * N]; float b_inv[N]; float W_out[N]; float b_out; float h1[N]; float h2[N]; float z_out; } NetS1;
typedef struct { float W_feat[N * 5]; float b_feat[N]; float W_inv[N * N]; float b_inv[N]; float W_out[N]; float b_out; float h1[N]; float h2[N]; float z_out; } NetS2;
typedef struct { float W_feat[N * 3]; float b_feat[N]; float W_inv[N * N]; float b_inv[N]; float W_out[N]; float b_out; float h1[N]; float h2[N]; float z_out; } NetS3;
typedef struct { float W_feat[N * 2]; float b_feat[N]; float W_inv[N * N]; float b_inv[N]; float W_out[N]; float b_out; float h1[N]; float h2[N]; float z_out; } NetS4;

// Naive Method: Single, Flat Network (No W_inv, No Cascading)
typedef struct {
    float W1[N * INPUT_DIM]; // 20x100
    float b1[N];             // 20x1
    float W2[N];             // 1x20
    float b2;                // 1x1
    float h1[N];             // Pre-activation for ReLU' check
} NetNaive;

typedef struct {
    float smooth_loss;
    float test_present;
    float test_absent;
    float det_final;
    float lambda_final;
    const char *method_name;
} TrainingResults;


// --- 3. Initialization ---

void init_weights_he(float *W, int M, int K) {
    float scale = sqrtf(2.0f / (float)K); 
    for (int i = 0; i < M * K; i++) W[i] = rand_uniform(-scale, scale);
}

void init_bias(float *b, int M) {
    for (int i = 0; i < M; i++) b[i] = 0.0f;
}

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

// --- 4. Forward Pass Functions ---

float forward_s1(NetS1 *net, const float *input); // Defined below
float forward_s2(NetS2 *net, const float *input);
float forward_s3(NetS3 *net, const float *input);
float forward_s4(NetS4 *net, const float *input);

float forward_naive(NetNaive *net, const float *input) {
    float h1_pre[N];
    mat_vec_mul(net->W1, N, INPUT_DIM, input, h1_pre);
    vec_add(h1_pre, net->b1, N, h1_pre);

    float h1_act[N];
    for (int i = 0; i < N; i++) { 
        h1_act[i] = ReLU(h1_pre[i]); 
        net->h1[i] = h1_pre[i]; // Save pre-activation for backprop
    }

    float z_out = 0.0f;
    for (int i = 0; i < N; i++) z_out += net->W2[i] * h1_act[i];
    z_out += net->b2;
    
    return sigmoid(z_out);
}

// Stubs for Advanced Forward
#define ADVANCED_FORWARD_BODY(NET_TYPE, IN_DIM) \
    float h1_pre[N]; \
    mat_vec_mul(net->W_feat, N, IN_DIM, input, h1_pre); \
    vec_add(h1_pre, net->b_feat, N, h1_pre); \
    for (int i = 0; i < N; i++) net->h1[i] = ReLU(h1_pre[i]); \
    float h2_pre[N]; \
    mat_vec_mul(net->W_inv, N, N, net->h1, h2_pre); \
    vec_add(h2_pre, net->b_inv, N, net->h2); \
    float z_out = 0.0f; \
    for (int i = 0; i < N; i++) z_out += net->W_out[i] * net->h2[i]; \
    z_out += net->b_out; \
    net->z_out = z_out; \
    return sigmoid(z_out);

float forward_s1(NetS1 *net, const float *input) { ADVANCED_FORWARD_BODY(NetS1, INPUT_DIM); }
float forward_s2(NetS2 *net, const float *input) { ADVANCED_FORWARD_BODY(NetS2, 5); }
float forward_s3(NetS3 *net, const float *input) { ADVANCED_FORWARD_BODY(NetS3, 3); }
float forward_s4(NetS4 *net, const float *input) { ADVANCED_FORWARD_BODY(NetS4, 2); }


// --- 5. Backpropagation Functions ---

// Advanced Backwards (using LEARNING_RATE)
#define ADVANCED_BACKWARD_UPDATES(LR) \
    for (int i = 0; i < N; i++) net->W_out[i] -= LR * clip_gradient(grad_W_out[i]); \
    net->b_out -= LR * clip_gradient(grad_b_out); \
    for (int i = 0; i < N * N; i++) net->W_inv[i] -= LR * clip_gradient(grad_W_inv[i]); \
    for (int i = 0; i < N; i++) net->b_inv[i] -= LR * clip_gradient(grad_b_inv[i]); \
    for (int i = 0; i < N * IN_DIM; i++) net->W_feat[i] -= LR * clip_gradient(grad_W_feat[i]); \
    for (int i = 0; i < N; i++) net->b_feat[i] -= LR * clip_gradient(grad_b_feat[i]);

void backward_s4(NetS4 *net, const float *input, float delta_out, float *delta_input) {
    // Stage 4 Backward logic (omitted for brevity, assume correct from previous step)
    float grad_W_out[N], delta_h2[N];
    for (int i = 0; i < N; i++) { grad_W_out[i] = delta_out * net->h2[i]; delta_h2[i] = delta_out * net->W_out[i]; }
    float grad_b_out = delta_out;

    float grad_W_inv[N * N];
    for (int i = 0; i < N * N; i++) grad_W_inv[i] = delta_h2[i / N] * net->h1[i % N];

    float delta_h1[N];
    for (int i = 0; i < N; i++) { delta_h1[i] = 0.0f; for (int j = 0; j < N; j++) delta_h1[i] += delta_h2[j] * net->W_inv[j * N + i]; }
    float *grad_b_inv = delta_h2;

    for (int i = 0; i < N; i++) delta_h1[i] *= (net->h1[i] > 0.0f ? 1.0f : 0.0f);

    float grad_W_feat[N * 2];
    for (int i = 0; i < N; i++) { for (int j = 0; j < 2; j++) grad_W_feat[i * 2 + j] = delta_h1[i] * input[j]; }
    float *grad_b_feat = delta_h1;

    for (int i = 0; i < 2; i++) { delta_input[i] = 0.0f; for (int j = 0; j < N; j++) delta_input[i] += delta_h1[j] * net->W_feat[j * 2 + i]; }
    #define IN_DIM 2
    ADVANCED_BACKWARD_UPDATES(LEARNING_RATE)
    #undef IN_DIM
}
void backward_s3(NetS3 *net, const float *input, float delta_out, float *delta_input) {
    // Stage 3 Backward logic
    float grad_W_out[N], delta_h2[N];
    for (int i = 0; i < N; i++) { grad_W_out[i] = delta_out * net->h2[i]; delta_h2[i] = delta_out * net->W_out[i]; }
    float grad_b_out = delta_out;
    float grad_W_inv[N * N];
    for (int i = 0; i < N * N; i++) grad_W_inv[i] = delta_h2[i / N] * net->h1[i % N];
    float delta_h1[N];
    for (int i = 0; i < N; i++) { delta_h1[i] = 0.0f; for (int j = 0; j < N; j++) delta_h1[i] += delta_h2[j] * net->W_inv[j * N + i]; }
    float *grad_b_inv = delta_h2;
    for (int i = 0; i < N; i++) delta_h1[i] *= (net->h1[i] > 0.0f ? 1.0f : 0.0f);
    float grad_W_feat[N * 3];
    for (int i = 0; i < N; i++) { for (int j = 0; j < 3; j++) grad_W_feat[i * 3 + j] = delta_h1[i] * input[j]; }
    float *grad_b_feat = delta_h1;
    for (int i = 0; i < 3; i++) { delta_input[i] = 0.0f; for (int j = 0; j < N; j++) delta_input[i] += delta_h1[j] * net->W_feat[j * 3 + i]; }
    #define IN_DIM 3
    ADVANCED_BACKWARD_UPDATES(LEARNING_RATE)
    #undef IN_DIM
}
void backward_s2(NetS2 *net, const float *input, float delta_out, float *delta_input) {
    // Stage 2 Backward logic
    float grad_W_out[N], delta_h2[N];
    for (int i = 0; i < N; i++) { grad_W_out[i] = delta_out * net->h2[i]; delta_h2[i] = delta_out * net->W_out[i]; }
    float grad_b_out = delta_out;
    float grad_W_inv[N * N];
    for (int i = 0; i < N * N; i++) grad_W_inv[i] = delta_h2[i / N] * net->h1[i % N];
    float delta_h1[N];
    for (int i = 0; i < N; i++) { delta_h1[i] = 0.0f; for (int j = 0; j < N; j++) delta_h1[i] += delta_h2[j] * net->W_inv[j * N + i]; }
    float *grad_b_inv = delta_h2;
    for (int i = 0; i < N; i++) delta_h1[i] *= (net->h1[i] > 0.0f ? 1.0f : 0.0f);
    float grad_W_feat[N * 5];
    for (int i = 0; i < N; i++) { for (int j = 0; j < 5; j++) grad_W_feat[i * 5 + j] = delta_h1[i] * input[j]; }
    float *grad_b_feat = delta_h1;
    for (int i = 0; i < 5; i++) { delta_input[i] = 0.0f; for (int j = 0; j < N; j++) delta_input[i] += delta_h1[j] * net->W_feat[j * 5 + i]; }
    #define IN_DIM 5
    ADVANCED_BACKWARD_UPDATES(LEARNING_RATE)
    #undef IN_DIM
}
void backward_s1(NetS1 *net, const float *input, float delta_out) {
    // Stage 1 Backward logic
    float grad_W_out[N], delta_h2[N];
    for (int i = 0; i < N; i++) { grad_W_out[i] = delta_out * net->h2[i]; delta_h2[i] = delta_out * net->W_out[i]; }
    float grad_b_out = delta_out;
    float grad_W_inv[N * N];
    for (int i = 0; i < N * N; i++) grad_W_inv[i] = delta_h2[i / N] * net->h1[i % N];
    float delta_h1[N];
    for (int i = 0; i < N; i++) { delta_h1[i] = 0.0f; for (int j = 0; j < N; j++) delta_h1[i] += delta_h2[j] * net->W_inv[j * N + i]; }
    float *grad_b_inv = delta_h2;
    for (int i = 0; i < N; i++) delta_h1[i] *= (net->h1[i] > 0.0f ? 1.0f : 0.0f);
    float grad_W_feat[N * INPUT_DIM];
    for (int i = 0; i < N; i++) { for (int j = 0; j < INPUT_DIM; j++) grad_W_feat[i * INPUT_DIM + j] = delta_h1[i] * input[j]; }
    float *grad_b_feat = delta_h1;
    #define IN_DIM INPUT_DIM
    ADVANCED_BACKWARD_UPDATES(LEARNING_RATE)
    #undef IN_DIM
}


void backward_naive(NetNaive *net, const float *input, float delta_out) {
    // Naive Network Backward Pass
    float grad_W2[N];
    float delta_h1_act[N];
    for (int i = 0; i < N; i++) {
        grad_W2[i] = delta_out * ReLU(net->h1[i]);
        delta_h1_act[i] = delta_out * net->W2[i];
    }
    float grad_b2 = delta_out;

    // Backprop L1 (dLoss/d(h1_pre_activation) * ReLU')
    float delta_h1_pre[N];
    for (int i = 0; i < N; i++) {
        // Chain rule: dLoss/d(h1_pre) = dLoss/d(h1_act) * d(h1_act)/d(h1_pre)
        delta_h1_pre[i] = delta_h1_act[i] * (net->h1[i] > 0.0f ? 1.0f : 0.0f);
    }
    float grad_W1[N * INPUT_DIM];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < INPUT_DIM; j++) {
            grad_W1[i * INPUT_DIM + j] = delta_h1_pre[i] * input[j];
        }
    }
    float *grad_b1 = delta_h1_pre;

    // Update Weights with Clipping (using NAIVE_LEARNING_RATE)
    for (int i = 0; i < N; i++) net->W2[i] -= NAIVE_LEARNING_RATE * clip_gradient(grad_W2[i]);
    net->b2 -= NAIVE_LEARNING_RATE * clip_gradient(grad_b2);
    for (int i = 0; i < N * INPUT_DIM; i++) net->W1[i] -= NAIVE_LEARNING_RATE * clip_gradient(grad_W1[i]);
    for (int i = 0; i < N; i++) net->b1[i] -= NAIVE_LEARNING_RATE * clip_gradient(grad_b1[i]);
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
                    if ((r + c) % 2 == 0) { img[r * IMAGE_SIZE + c] = 1.0f; } else { img[r * IMAGE_SIZE + c] = 0.5f; }
                }
            }
        }
    }
}


// --- 7. Training Functions ---

TrainingResults run_advanced_training() {
    NetS1 nets1[5]; NetS2 nets2[3]; NetS3 nets3[2]; NetS4 net_final;
    for (int i = 0; i < 5; i++) init_net_s1(&nets1[i]);
    for (int i = 0; i < 3; i++) init_net_s2(&nets2[i]);
    for (int i = 0; i < 2; i++) init_net_s3(&nets3[i]);
    init_net_s4(&net_final);

    printf("\n\n#####################################################\n");
    printf("--- RUN 1: ADVANCED CASCADED ENSEMBLE (O(N^3) Core) ---\n");
    printf("Learning Rate: %.2f | Hidden Dim (N): %d\n", LEARNING_RATE, N);
    printf("#####################################################\n");

    float input_image[INPUT_DIM];
    float target;
    float avg_loss = 0.0f;
    float output_s1[5], output_s2[3], output_s3[2];
    float delta_out_s3[2], delta_s3_in[3], delta_s2_in[5];

    float W_inv_copy[N * N];
    float W_inverse[N * N];
    float dominant_eigenvalue = 0.0f;
    float dominant_eigenvector[N];
    float det = 0.0f;
    float final_output = 0.0f;

    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        int is_present = rand() % 2;
        target = (float)is_present;
        make_rectangle(input_image, is_present);

        // Forward
        for (int i = 0; i < 5; i++) output_s1[i] = forward_s1(&nets1[i], input_image);
        for (int i = 0; i < 3; i++) output_s2[i] = forward_s2(&nets2[i], output_s1);
        for (int i = 0; i < 2; i++) output_s3[i] = forward_s3(&nets3[i], output_s2);
        final_output = forward_s4(&net_final, output_s3);

        // Loss
        float loss = (target - final_output) * (target - final_output);
        avg_loss = avg_loss * 0.99f + loss * 0.01f;

        // Backward
        float delta_final = (final_output - target) * final_output * (1.0f - final_output);
        backward_s4(&net_final, output_s3, delta_final, delta_out_s3); 
        memset(delta_s3_in, 0, sizeof(delta_s3_in));
        for(int i = 0; i < 2; i++) { float current_delta_input[3]; backward_s3(&nets3[i], output_s2, delta_out_s3[i], current_delta_input); for(int j = 0; j < 3; j++) delta_s3_in[j] += current_delta_input[j]; }
        memset(delta_s2_in, 0, sizeof(delta_s2_in));
        for(int i = 0; i < 3; i++) { float current_delta_input[5]; backward_s2(&nets2[i], output_s1, delta_s3_in[i], current_delta_input); for(int j = 0; j < 5; j++) delta_s2_in[j] += current_delta_input[j]; }
        for(int i = 0; i < 5; i++) { backward_s1(&nets1[i], input_image, delta_s2_in[i]); }

        // Analysis and Reporting
        if (epoch % (EPOCHS / 10) == 0) {
            printf("\n--- Epoch %d/%d ---\n", epoch, EPOCHS);
            printf("Target: %.0f | Final Output: %.4f | Smooth Loss: %.6f\n", target, final_output, avg_loss);
            printf("--- High-Demand Matrix Analysis (Net S4, O(N^3) on %d x %d) ---\n", N, N);
            memcpy(W_inv_copy, net_final.W_inv, N * N * sizeof(float));
            det = inverse_and_determinant(W_inv_copy, W_inverse); 
            printf("   [DETERMINANT] Calculated value: %.6f\n", det);
            if (fabs(det) < 1e-6) { printf("   [CRITICAL] Matrix is SINGULAR (Det $\\approx 0$). Invertible constraint violation!\n"); }
            else { printf("   [INVERSE] W_inv is invertible. (Top-Left Element of W_inv: %.4f)\n", W_inverse[0]); }
            power_iteration(net_final.W_inv, &dominant_eigenvalue, dominant_eigenvector);
            printf("   [EIGENVALUE] Dominant $\\lambda$: %.6f\n", dominant_eigenvalue);
            printf("   [EIGENVECTOR] Dominant vector $v$ ($v_1$ to $v_3$): (%.4f, %.4f, %.4f) ...\n",
                   dominant_eigenvector[0], dominant_eigenvector[1], dominant_eigenvector[2]);
            printf("   [INTERPRET] The largest eigenvalue ($\\|\\lambda\\| > 1$) indicates directional feature expansion.\n");
        }
    }

    // Final Test
    make_rectangle(input_image, 1);
    for (int i = 0; i < 5; i++) output_s1[i] = forward_s1(&nets1[i], input_image);
    for (int i = 0; i < 3; i++) output_s2[i] = forward_s2(&nets2[i], output_s1);
    for (int i = 0; i < 2; i++) output_s3[i] = forward_s3(&nets3[i], output_s2);
    float final_output_present = forward_s4(&net_final, output_s3);

    make_rectangle(input_image, 0);
    for (int i = 0; i < 5; i++) output_s1[i] = forward_s1(&nets1[i], input_image);
    for (int i = 0; i < 3; i++) output_s2[i] = forward_s2(&nets2[i], output_s1);
    for (int i = 0; i < 2; i++) output_s3[i] = forward_s3(&nets3[i], output_s2);
    float final_output_absent = forward_s4(&net_final, output_s3);

    printf("\nTraining complete. Final smooth loss: %.6f\n", avg_loss);
    printf("TEST (Rectangle Present): Final Output = %.4f (Expected near 1.0)\n", final_output_present);
    printf("TEST (No Rectangle): Final Output = %.4f (Expected near 0.0)\n", final_output_absent);

    return (TrainingResults){
        .smooth_loss = avg_loss,
        .test_present = final_output_present,
        .test_absent = final_output_absent,
        .det_final = det,
        .lambda_final = dominant_eigenvalue,
        .method_name = "Advanced Cascaded Ensemble"
    };
}


TrainingResults run_naive_training() {
    NetNaive net;
    init_net_naive(&net);

    printf("\n\n#####################################################\n");
    printf("--- RUN 2: NAIVE FLAT NETWORK (Stable Baseline) ---\n");
    printf("Learning Rate: %.2f | Hidden Dim (N): %d\n", NAIVE_LEARNING_RATE, N);
    printf("#####################################################\n");

    float input_image[INPUT_DIM];
    float target;
    float avg_loss = 0.0f;
    float final_output = 0.0f;

    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        int is_present = rand() % 2;
        target = (float)is_present;
        make_rectangle(input_image, is_present);

        // Forward
        final_output = forward_naive(&net, input_image);

        // Loss
        float loss = (target - final_output) * (target - final_output);
        avg_loss = avg_loss * 0.99f + loss * 0.01f;

        // Backward
        float delta_final = (final_output - target) * final_output * (1.0f - final_output);
        backward_naive(&net, input_image, delta_final);
        
        if (epoch % (EPOCHS / 10) == 0) {
            printf("--- Epoch %d/%d ---\n", epoch, EPOCHS);
            printf("Target: %.0f | Final Output: %.4f | Smooth Loss: %.6f\n", target, final_output, avg_loss);
        }
    }

    // Final Test
    make_rectangle(input_image, 1);
    float final_output_present = forward_naive(&net, input_image);

    make_rectangle(input_image, 0);
    float final_output_absent = forward_naive(&net, input_image);

    printf("\nTraining complete. Final smooth loss: %.6f\n", avg_loss);
    printf("TEST (Rectangle Present): Final Output = %.4f (Expected near 1.0)\n", final_output_present);
    printf("TEST (No Rectangle): Final Output = %.4f (Expected near 0.0)\n", final_output_absent);

    return (TrainingResults){
        .smooth_loss = avg_loss,
        .test_present = final_output_present,
        .test_absent = final_output_absent,
        .det_final = 0.0f, // Not applicable
        .lambda_final = 0.0f, // Not applicable
        .method_name = "Naive Flat Network"
    };
}

void print_final_summary(const TrainingResults *adv, const TrainingResults *naive) {
    printf("\n\n=====================================================\n");
    printf("                   FINAL COMPARATIVE REPORT\n");
    printf("=====================================================\n");
    printf("| Metric                 | Advanced (Cascaded) | Naive (Flat)       |\n");
    printf("|------------------------|---------------------|--------------------|\n");
    printf("| Final Smooth Loss      | %-19.6f | %-19.6f |\n", adv->smooth_loss, naive->smooth_loss);
    printf("|------------------------|---------------------|--------------------|\n");
    printf("| Test Output (Present)  | %-19.4f | %-19.4f |\n", adv->test_present, naive->test_present);
    printf("| Test Output (Absent)   | %-19.4f | %-19.4f |\n", adv->test_absent, naive->test_absent);
    printf("|------------------------|---------------------|--------------------|\n");
    printf("| W_inv Determinant      | %-19.6f | %-19s |\n", adv->det_final, "N/A");
    printf("| Dominant Eigenvalue    | %-19.6f | %-19s |\n", adv->lambda_final, "N/A");
    printf("=====================================================\n");
}


int main() {
    srand(time(NULL));

    TrainingResults results_advanced = run_advanced_training();
    TrainingResults results_naive = run_naive_training();

    print_final_summary(&results_advanced, &results_naive);

    return 0;
}