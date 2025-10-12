#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define INPUT_DIM 100       // 10x10 image input
#define N_VISUALIZE 4       // Hardcode N=4 for this specific run
#define MAX_HIDDEN_DIM N_VISUALIZE
#define EPOCHS 150000       
#define IMAGE_SIZE 10
#define GRADIENT_CLIP_MAX 0.1f 
#define PI 3.14159265358979323846f
#define LEARNING_RATE 0.03f 
#define L2_REG_LAMBDA 0.0001f // L2 regularization strength

// --- 1. Network Structure ---

typedef struct { 
    float W1[MAX_HIDDEN_DIM * INPUT_DIM]; 
    float b1[MAX_HIDDEN_DIM]; 
    float W2[MAX_HIDDEN_DIM]; 
    float b2; 
    float h1_pre[MAX_HIDDEN_DIM]; 
} NetConstrained;

// --- 2. Utilities and Initialization ---

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
        for (int j = 0; j < N_A; j++) { sum += A[i * N_A + j] * x[j]; }
        y[i] = sum;
    }
}
void init_weights_he(float *W, int M, int K) {
    float scale = sqrtf(2.0f / (float)K); 
    for (int i = 0; i < M * K; i++) W[i] = rand_uniform(-scale, scale);
}
void init_bias(float *b, int M) { for (int i = 0; i < M; i++) b[i] = 0.0f; }
void init_net(NetConstrained *net, int N_eff) { 
    init_weights_he(net->W1, N_eff, INPUT_DIM); 
    init_bias(net->b1, N_eff); 
    init_weights_he(net->W2, 1, N_eff); 
    init_bias(&net->b2, 1); 
}

// --- 3. Forward Pass (Common) ---

float forward_network(NetConstrained *net, const float *input, int N_eff) {
    mat_vec_mul(net->W1, N_eff, INPUT_DIM, input, net->h1_pre); 
    for(int i=0; i<N_eff; i++) net->h1_pre[i] += net->b1[i];

    float z_out = 0.0f; 
    for (int i = 0; i < N_eff; i++) {
        z_out += net->W2[i] * ReLU(net->h1_pre[i]); 
    }
    z_out += net->b2;
    return sigmoid(z_out);
}

// --- 4. Constrained Backpropagation (L2 Regularization Penalty) ---

void backward_constrained(NetConstrained *net, const float *input, float delta_out, float lr, int N_eff) {
    float grad_W2[N_eff]; 
    float delta_h1_act[N_eff];

    for (int i = 0; i < N_eff; i++) { 
        float h1_act = ReLU(net->h1_pre[i]);
        grad_W2[i] = delta_out * h1_act; 
        delta_h1_act[i] = delta_out * net->W2[i]; 
    }
    float grad_b2 = delta_out;

    float delta_h1_pre[N_eff];
    for (int i = 0; i < N_eff; i++) { 
        delta_h1_pre[i] = delta_h1_act[i] * (net->h1_pre[i] > 0.0f ? 1.0f : 0.0f); 
    }

    float grad_W1[N_eff * INPUT_DIM];
    for (int i = 0; i < N_eff; i++) { 
        for (int j = 0; j < INPUT_DIM; j++) { 
            grad_W1[i * INPUT_DIM + j] = delta_h1_pre[i] * input[j]; 
        } 
    }
    float *grad_b1 = delta_h1_pre; 

    // --- Parameter Updates (WITH L2 regularization term) ---
    
    // W2 (Add L2 penalty)
    for (int i = 0; i < N_eff; i++) {
        float total_grad_W2 = grad_W2[i] + L2_REG_LAMBDA * net->W2[i];
        net->W2[i] -= lr * clip_gradient(total_grad_W2);
    }
    // b2 (Bias is NOT regularized)
    net->b2 -= lr * clip_gradient(grad_b2);
    
    // W1 (Add L2 penalty)
    for (int i = 0; i < N_eff * INPUT_DIM; i++) {
        float total_grad_W1 = grad_W1[i] + L2_REG_LAMBDA * net->W1[i];
        net->W1[i] -= lr * clip_gradient(total_grad_W1);
    }
    
    // b1 (Bias is NOT regularized)
    for (int i = 0; i < N_eff; i++) net->b1[i] -= lr * clip_gradient(grad_b1[i]);
}


// --- 5. Data Generation (Identical to previous run) ---

void generate_image(float *img, int is_present, int is_rotated) {
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
                    img[r * IMAGE_SIZE + c] = (((int)roundf(r_prime * 2.0f) + (int)roundf(c_prime * 2.0f)) % 2 == 0) ? 1.0f : 0.5f;
                }
            }
        }
    }
    
    if (!is_present) {
        int size = (rand() % 3) + 2; 
        int max_start = IMAGE_SIZE - size;
        int start_r = rand() % (max_start > 0 ? max_start : 1);
        int start_c = rand() % (max_start > 0 ? max_start : 1);
        int orientation = rand() % 3; 

        for (int i = 0; i < size; i++) {
            int r = start_r, c = start_c;
            if (orientation == 0) c += i;          
            else if (orientation == 1) r += i;     
            else r += i, c += i;                   

            if (r >= 0 && r < IMAGE_SIZE && c >= 0 && c < IMAGE_SIZE) {
                img[r * IMAGE_SIZE + c] = 0.8f; 
            }
        }
    }
    
    int noise_pixels = (int)(INPUT_DIM * 0.20f);
    for (int i = 0; i < noise_pixels; i++) {
        int idx = rand() % INPUT_DIM;
        if (img[idx] == 0.0f) {
            img[idx] = 0.2f; 
        }
    }
}


// --- 6. Visualization Utilities ---

void print_ascii_image(const float *data, float min_val, float max_val, const char *title) {
    printf("\n%s (10x10 Feature Map):\n", title);
    
    // Scale the floating point value to an integer from 0 to 9 for ASCII visualization
    float range = max_val - min_val;
    if (range < 1e-6) range = 1.0f; 

    for (int r = 0; r < IMAGE_SIZE; r++) {
        for (int c = 0; c < IMAGE_SIZE; c++) {
            float val = data[r * IMAGE_SIZE + c];
            
            int scaled_val = (int)roundf(((val - min_val) / range) * 9.0f);
            
            if (scaled_val < 0) scaled_val = 0;
            if (scaled_val > 9) scaled_val = 9;

            printf("%d ", scaled_val);
        }
        printf("\n");
    }
}

void visualize_features(const NetConstrained *net) {
    int N_eff = N_VISUALIZE; 

    // Find the min/max overall weight to normalize all features to the same global scale
    float global_min = net->W1[0];
    float global_max = net->W1[0];

    for (int i = 0; i < N_eff * INPUT_DIM; i++) {
        if (net->W1[i] < global_min) global_min = net->W1[i];
        if (net->W1[i] > global_max) global_max = net->W1[i];
    }
    
    printf("\n\n---------------------------------------------------------------------------\n");
    printf("FEATURE MAP VISUALIZATION (CONSTRAINED/L2 N=%d)\n", N_eff);
    printf("Weights normalized from global_min=%.4f to global_max=%.4f\n", global_min, global_max);
    printf("---------------------------------------------------------------------------\n");

    // Display the first two learned features (W1 rows 0 and 1)
    
    // Feature 1
    print_ascii_image(&net->W1[0], global_min, global_max, "Feature 1 (Input Weights of Neuron 1)");
    printf("W2 Output Weight: %.4f\n", net->W2[0]);

    // Feature 2
    print_ascii_image(&net->W1[INPUT_DIM], global_min, global_max, "Feature 2 (Input Weights of Neuron 2)");
    printf("W2 Output Weight: %.4f\n", net->W2[1]);
}


// --- 7. Main Benchmark Function (Only runs N=4 Constrained) ---

void run_constrained_benchmark_and_visualize(NetConstrained *net_out) {
    int N_eff = N_VISUALIZE;
    float input_image[INPUT_DIM];
    float target;
    float avg_loss = 0.0f;

    init_net(net_out, N_eff); 

    // --- Training Loop ---
    clock_t start_time = clock();

    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        int is_present = rand() % 2;
        int do_rotate = is_present && (rand() % 2); 
        target = (float)is_present;
        
        generate_image(input_image, is_present, do_rotate);

        float final_output = forward_network(net_out, input_image, N_eff);
        float mse_loss = (target - final_output) * (target - final_output);
        avg_loss = avg_loss * 0.99f + mse_loss * 0.01f;

        // Backward Pass & Update (CONSTRAINED/L2)
        float delta_final = (final_output - target) * final_output * (1.0f - final_output);
        backward_constrained(net_out, input_image, delta_final, LEARNING_RATE, N_eff);
    }

    clock_t end_time = clock();
    float training_time_sec = (float)(end_time - start_time) / CLOCKS_PER_SEC;

    // --- Final Testing ---
    
    generate_image(input_image, 1, 0); 
    float test_present = forward_network(net_out, input_image, N_eff);

    generate_image(input_image, 0, 0); 
    float test_absent = forward_network(net_out, input_image, N_eff);

    generate_image(input_image, 1, 1); 
    float test_rotated = forward_network(net_out, input_image, N_eff);

    printf("[Constrained Benchmark N=%-3d] Time: %.2fs | Loss: %.6f | Present: %.4f, Absent (Distractor): %.4f, Rotated: %.4f\n", 
           N_eff, training_time_sec, avg_loss, test_present, test_absent, test_rotated);
}


int main() {
    setbuf(stdout, NULL); 
    srand(time(NULL));
    
    NetConstrained constrained_net;

    printf("Starting Constrained (L2-Regularized) Network Benchmark (N=4 only) for Feature Visualization.\n");
    printf("Training data includes complex linear distractors and 20%% random noise.\n\n");

    // Run the benchmark and save the trained network structure
    run_constrained_benchmark_and_visualize(&constrained_net);

    // Visualize the feature maps of the trained network
    visualize_features(&constrained_net);

    return 0;
}