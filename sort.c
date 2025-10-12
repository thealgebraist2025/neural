#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define INPUT_DIM 100       // 10x10 image input
#define MAX_HIDDEN_DIM 256  // Maximum hidden layer size to accommodate all tests
#define EPOCHS 150000       
#define IMAGE_SIZE 10
#define GRADIENT_CLIP_MAX 0.1f 
#define PI 3.14159265358979323846f
#define LEARNING_RATE 0.03f 

// --- 1. Utilities ---

float ReLU(float x) { return (x) > 0.0f ? (x) : 0.0f; }
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
        // N_A is the size of the input vector x (or columns in A)
        for (int j = 0; j < N_A; j++) { sum += A[i * N_A + j] * x[j]; }
        y[i] = sum;
    }
}
void vec_add(const float *x, const float *b, int D, float *y) {
    for (int i = 0; i < D; i++) { y[i] = x[i] + b[i]; }
}

// --- 2. Network Structure and Result Storage ---

// Using MAX_HIDDEN_DIM for fixed structure size, actual size is passed to functions.
typedef struct { 
    float W1[MAX_HIDDEN_DIM * INPUT_DIM]; 
    float b1[MAX_HIDDEN_DIM]; 
    float W2[MAX_HIDDEN_DIM]; 
    float b2; 
    float h1_pre[MAX_HIDDEN_DIM]; // Pre-activation values for backprop
} NetNaive;

typedef struct {
    int size;
    float smooth_loss;
    float training_time_sec;
    float test_present;
    float test_absent;
    float test_rotated;
} BenchmarkResult;


// --- 3. Initialization ---

void init_weights_he(float *W, int M, int K) {
    float scale = sqrtf(2.0f / (float)K); 
    for (int i = 0; i < M * K; i++) W[i] = rand_uniform(-scale, scale);
}
void init_bias(float *b, int M) { for (int i = 0; i < M; i++) b[i] = 0.0f; }

void init_net_naive(NetNaive *net, int N_eff) { 
    // Initialize only the necessary parts of the arrays
    init_weights_he(net->W1, N_eff, INPUT_DIM); 
    init_bias(net->b1, N_eff); 
    init_weights_he(net->W2, 1, N_eff); 
    init_bias(&net->b2, 1); 
}


// --- 4. Forward Pass ---

float forward_naive(NetNaive *net, const float *input, int N_eff) {
    // Hidden Layer 1 (W1: N_eff x INPUT_DIM)
    // h1_pre stores the pre-activation values for the current N_eff size
    mat_vec_mul(net->W1, N_eff, INPUT_DIM, input, net->h1_pre); 
    vec_add(net->h1_pre, net->b1, N_eff, net->h1_pre);

    // Output Layer (W2: 1 x N_eff)
    float z_out = 0.0f; 
    for (int i = 0; i < N_eff; i++) {
        z_out += net->W2[i] * ReLU(net->h1_pre[i]); 
    }
    z_out += net->b2;
    return sigmoid(z_out);
}

// --- 5. Backpropagation ---

void backward_naive(NetNaive *net, const float *input, float delta_out, float lr, int N_eff) {
    float grad_W2[N_eff]; 
    float delta_h1_act[N_eff];

    // Gradients for W2 and b2, and delta for h1 activation
    for (int i = 0; i < N_eff; i++) { 
        float h1_act = ReLU(net->h1_pre[i]);
        grad_W2[i] = delta_out * h1_act; 
        delta_h1_act[i] = delta_out * net->W2[i]; 
    }
    float grad_b2 = delta_out;

    // Delta for h1 pre-activation (applying ReLU derivative)
    float delta_h1_pre[N_eff];
    for (int i = 0; i < N_eff; i++) { 
        delta_h1_pre[i] = delta_h1_act[i] * (net->h1_pre[i] > 0.0f ? 1.0f : 0.0f); 
    }

    // Gradients for W1 (N_eff x INPUT_DIM)
    float grad_W1[N_eff * INPUT_DIM];
    for (int i = 0; i < N_eff; i++) { 
        for (int j = 0; j < INPUT_DIM; j++) { 
            grad_W1[i * INPUT_DIM + j] = delta_h1_pre[i] * input[j]; 
        } 
    }
    float *grad_b1 = delta_h1_pre;

    // --- Parameter Updates ---
    
    // W2 and b2
    for (int i = 0; i < N_eff; i++) net->W2[i] -= lr * clip_gradient(grad_W2[i]);
    net->b2 -= lr * clip_gradient(grad_b2);
    
    // W1 (N_eff x INPUT_DIM)
    for (int i = 0; i < N_eff * INPUT_DIM; i++) net->W1[i] -= lr * clip_gradient(grad_W1[i]);
    
    // b1
    for (int i = 0; i < N_eff; i++) net->b1[i] -= lr * clip_gradient(grad_b1[i]);
}


// --- 6. Data Generation (Same as before) ---

void make_rectangle(float *img, int is_present, int is_rotated) {
    for (int i = 0; i < INPUT_DIM; i++) img[i] = 0.0f;

    if (is_present) {
        int start_row = rand() % (IMAGE_SIZE - 4);
        int start_col = rand() % (IMAGE_SIZE - 4);
        int width = 3 + (rand() % 3);
        int height = 3 + (rand() % 3);

        float center_r = start_row + height / 2.0f;
        float center_c = start_col + width / 2.0f;
        
        float angle = is_rotated ? rand_uniform(0.0f, 45.0f) * PI / 180.0f : 0.0f;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);

        for (int r = 0; r < IMAGE_SIZE; r++) {
            for (int c = 0; c < IMAGE_SIZE; c++) {
                
                float tr = r - center_r;
                float tc = c - center_c;
                
                float r_prime = tr * cos_a + tc * sin_a;
                float c_prime = -tr * sin_a + tc * cos_a;
                
                float half_w = width / 2.0f;
                float half_h = height / 2.0f;

                if (r_prime >= -half_h && r_prime <= half_h && 
                    c_prime >= -half_w && c_prime <= half_w) 
                {
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


// --- 7. Naive Network Benchmarking Function ---

BenchmarkResult run_naive_benchmark(int N_eff) {
    
    float input_image[INPUT_DIM];
    float target;
    float avg_loss = 0.0f;
    float final_output = 0.0f;
    float final_loss = 0.0f;
    NetNaive net;

    init_net_naive(&net, N_eff);

    // --- Training Loop ---
    clock_t start_time = clock();

    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        int is_present = rand() % 2;
        int do_rotate = is_present && (rand() % 2); // 50% chance of rotation if present
        target = (float)is_present;
        make_rectangle(input_image, is_present, do_rotate);

        // Forward Pass
        final_output = forward_naive(&net, input_image, N_eff);

        // MSE Loss
        float mse_loss = (target - final_output) * (target - final_output);
        final_loss = mse_loss;
        
        // Update Smooth Loss
        avg_loss = avg_loss * 0.99f + final_loss * 0.01f;

        // Backward Pass & Update
        float delta_final = (final_output - target) * final_output * (1.0f - final_output);
        backward_naive(&net, input_image, delta_final, LEARNING_RATE, N_eff);
    }

    clock_t end_time = clock();
    float training_time_sec = (float)(end_time - start_time) / CLOCKS_PER_SEC;

    // --- Final Testing ---
    float final_output_present, final_output_absent, final_output_rotated;
    
    // 1. Test - Standard Rectangle
    make_rectangle(input_image, 1, 0); // is_present=1, is_rotated=0
    final_output_present = forward_naive(&net, input_image, N_eff);

    // 2. Test - Absent Rectangle
    make_rectangle(input_image, 0, 0); // is_present=0, is_rotated=0
    final_output_absent = forward_naive(&net, input_image, N_eff);

    // 3. Test - Rotated Rectangle
    make_rectangle(input_image, 1, 1); // is_present=1, is_rotated=1
    final_output_rotated = forward_naive(&net, input_image, N_eff);

    printf("[Benchmark N=%-3d] Time: %.2fs | Loss: %.6f | Present: %.4f, Absent: %.4f, Rotated: %.4f\n", 
           N_eff, training_time_sec, avg_loss, final_output_present, final_output_absent, final_output_rotated);

    return (BenchmarkResult){
        .size = N_eff,
        .smooth_loss = avg_loss,
        .training_time_sec = training_time_sec,
        .test_present = final_output_present,
        .test_absent = final_output_absent,
        .test_rotated = final_output_rotated
    };
}

void print_final_summary(const BenchmarkResult *results, int count) {
    printf("\n\n===========================================================================================\n");
    printf("                  NAIVE NETWORK SCALING BENCHMARK (Hidden Layer Size vs. Performance)\n");
    printf("===========================================================================================\n");
    printf("| Hidden Neurons | Time (s) | Smooth Loss | Test (Present) | Test (Absent) | Test (Rotated) |\n");
    printf("|----------------|----------|-------------|----------------|---------------|----------------|\n");
    
    for(int i = 0; i < count; i++) {
        const BenchmarkResult *res = &results[i];
        
        printf("| %-14d | %-8.2f | %-11.6f | %-14.4f | %-13.4f | %-14.4f |\n",
               res->size,
               res->training_time_sec,
               res->smooth_loss,
               res->test_present,
               res->test_absent,
               res->test_rotated);
    }
    printf("===========================================================================================\n");
    printf("NOTE: The Naive Architecture (simple 2-layer MLP) shows classic diminishing returns in loss\n");
    printf("but increasing time complexity as the hidden layer size increases.\n");
}


int main() {
    // Disable stdout buffering for real-time log (useful for long runs)
    setbuf(stdout, NULL); 
    srand(time(NULL));

    int sizes[] = {2, 4, 8, 16, 32, 64, 128, 256};
    int num_runs = sizeof(sizes) / sizeof(sizes[0]);
    BenchmarkResult results[num_runs];

    printf("Starting Naive Network Benchmark across %d hidden layer sizes (N=2 to N=256).\n", num_runs);
    printf("Each run trains for %d epochs with LR=%.2f.\n\n", EPOCHS, LEARNING_RATE);

    for (int i = 0; i < num_runs; i++) {
        results[i] = run_naive_benchmark(sizes[i]);
    }

    print_final_summary(results, num_runs);

    return 0;
}