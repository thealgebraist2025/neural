#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define INPUT_DIM 100       // 10x10 image input
#define N_VISUALIZE 4       // Use N=4 for the analysis
#define MAX_HIDDEN_DIM N_VISUALIZE
#define EPOCHS 150000       
#define IMAGE_SIZE 10
#define GRADIENT_CLIP_MAX 0.1f 
#define PI 3.14159265358979323846f
#define LEARNING_RATE 0.03f 
#define L2_REG_LAMBDA 0.0001f // L2 regularization strength for the constrained network

// --- 1. Network Structure ---

typedef struct { 
    float W1[MAX_HIDDEN_DIM * INPUT_DIM]; 
    float b1[MAX_HIDDEN_DIM]; 
    float W2[MAX_HIDDEN_DIM]; 
    float b2; 
    float h1_pre[MAX_HIDDEN_DIM]; 
} NetConstrained;

// --- 2. Utilities and Initialization (Common) ---

float ReLU(float x) { return (x) > 0.0f ? (x) : 0.0f; }
float rand_uniform(float min, float max) { return (max - min) * ((float)rand() / RAND_MAX) + min; }
float clip_gradient(float grad) {
    if (grad > GRADIENT_CLIP_MAX) return GRADIENT_CLIP_MAX;
    if (grad < -GRADIENT_CLIP_MAX) return -GRADIENT_CLIP_MAX;
    return grad;
}
float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// Matrix-Vector Multiplication: y = A * x (A is M x N_A, x is N_A)
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

// --- 3. Forward and Constrained Backward Passes ---

float forward_network(NetConstrained *net, const float *input, int N_eff) {
    // h1_pre = W1 * input + b1
    mat_vec_mul(net->W1, N_eff, INPUT_DIM, input, net->h1_pre); 
    for(int i=0; i<N_eff; i++) net->h1_pre[i] += net->b1[i];

    // output = sigmoid(W2 * ReLU(h1_pre) + b2)
    float z_out = 0.0f; 
    for (int i = 0; i < N_eff; i++) {
        z_out += net->W2[i] * ReLU(net->h1_pre[i]); 
    }
    z_out += net->b2;
    return sigmoid(z_out);
}

void backward_constrained(NetConstrained *net, const float *input, float delta_out, float lr, int N_eff) {
    // Gradients computation is identical to previous Constrained code
    float grad_W2[N_eff], delta_h1_act[N_eff], delta_h1_pre[N_eff], grad_W1[N_eff * INPUT_DIM];

    for (int i = 0; i < N_eff; i++) { 
        float h1_act = ReLU(net->h1_pre[i]);
        grad_W2[i] = delta_out * h1_act; 
        delta_h1_act[i] = delta_out * net->W2[i]; 
        delta_h1_pre[i] = delta_h1_act[i] * (net->h1_pre[i] > 0.0f ? 1.0f : 0.0f); 
        for (int j = 0; j < INPUT_DIM; j++) { 
            grad_W1[i * INPUT_DIM + j] = delta_h1_pre[i] * input[j]; 
        } 
    }

    // Parameter Updates (WITH L2 regularization term)
    for (int i = 0; i < N_eff; i++) {
        float total_grad_W2 = grad_W2[i] + L2_REG_LAMBDA * net->W2[i];
        net->W2[i] -= lr * clip_gradient(total_grad_W2);
    }
    net->b2 -= lr * clip_gradient(delta_out);
    for (int i = 0; i < N_eff * INPUT_DIM; i++) {
        float total_grad_W1 = grad_W1[i] + L2_REG_LAMBDA * net->W1[i];
        net->W1[i] -= lr * clip_gradient(total_grad_W1);
    }
    for (int i = 0; i < N_eff; i++) net->b1[i] -= lr * clip_gradient(delta_h1_pre[i]);
}


// --- 4. Data Generation for Training ---

void generate_training_image(float *img, int is_present, int is_rotated) {
    // Training image generation logic (omitted for brevity, identical to previous run)
    // ...
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
                    img[r * IMAGE_SIZE + c] = 1.0f;
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

// --- 5. Deterministic Test Image Generation for Visualization ---

void generate_test_rect_3x3(float *img) {
    for (int i = 0; i < INPUT_DIM; i++) img[i] = 0.0f;
    int center = IMAGE_SIZE / 2;
    for (int r = center - 1; r <= center + 1; r++) {
        for (int c = center - 1; c <= center + 1; c++) {
            if (r >= 0 && r < IMAGE_SIZE && c >= 0 && c < IMAGE_SIZE) {
                 img[r * IMAGE_SIZE + c] = 1.0f;
            }
        }
    }
}

void generate_test_noisy_blank(float *img) {
    for (int i = 0; i < INPUT_DIM; i++) img[i] = 0.0f;
    
    // Add deterministic scattered noise pattern (20% of pixels)
    int noise_pixels = (int)(INPUT_DIM * 0.20f);
    for (int i = 0; i < noise_pixels; i++) {
        img[i * 5] = 0.2f; // Use index 0, 5, 10, ... as a deterministic pattern
    }
}

void generate_test_rotated(float *img) {
    for (int i = 0; i < INPUT_DIM; i++) img[i] = 0.0f;
    
    float center_r = 4.5f; 
    float center_c = 4.5f;
    int width = 4;
    int height = 2;
    float angle = 20.0f * PI / 180.0f; 
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
                img[r * IMAGE_SIZE + c] = 1.0f;
            }
        }
    }
}


// --- 6. Visualization Utilities ---

// Prints an ASCII map where values are scaled between 0-9
void print_ascii_map(const float *data, float min_val, float max_val, const char *title) {
    printf("\n%s:\n", title);
    
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

// Prints the raw input image (scaled 0-9, assumes max is 1.0)
void print_input_image_ascii(const float *data, const char *title) {
    printf("\n--- INPUT IMAGE: %s ---\n", title);
    
    for (int r = 0; r < IMAGE_SIZE; r++) {
        for (int c = 0; c < IMAGE_SIZE; c++) {
            int scaled_val = (int)roundf(data[r * IMAGE_SIZE + c] * 9.0f);
            if (scaled_val > 9) scaled_val = 9;
            printf("%d ", scaled_val);
        }
        printf("\n");
    }
}

void visualize_features(const NetConstrained *net) {
    int N_eff = N_VISUALIZE; 

    float global_min = net->W1[0];
    float global_max = net->W1[0];

    for (int i = 0; i < N_eff * INPUT_DIM; i++) {
        if (net->W1[i] < global_min) global_min = net->W1[i];
        if (net->W1[i] > global_max) global_max = net->W1[i];
    }
    
    printf("\n\n===========================================================================\n");
    printf("CONSTRAINED/L2 LEARNED FEATURE MAPS (N=%d)\n", N_eff);
    printf("W1 Weights normalized from global_min=%.4f to global_max=%.4f\n", global_min, global_max);
    printf("===========================================================================\n");

    for (int i = 0; i < N_eff; i++) {
        char title[80];
        snprintf(title, sizeof(title), "FEATURE %d (W2 Output Weight: %.4f | Bias: %.4f)", i + 1, net->W2[i], net->b1[i]);
        print_ascii_map(&net->W1[i * INPUT_DIM], global_min, global_max, title);
    }
}

void analyze_single_test_case(const NetConstrained *net, const float *input_img, const char *title) {
    float weighted_activation_map[INPUT_DIM];
    float h_pre[N_VISUALIZE];
    int N_eff = N_VISUALIZE;

    print_input_image_ascii(input_img, title);
    
    // Calculate full forward pass to get final result
    float final_output = forward_network(net, input_img, N_eff);

    printf("\n--- FEATURE ACTIVATION MAPS & SUMS ---\n");
    for (int i = 0; i < N_eff; i++) {
        float map_min = 0.0f;
        float map_max = 0.0f;
        
        // Calculate W1[i] * input (element-wise product)
        for (int j = 0; j < INPUT_DIM; j++) {
            weighted_activation_map[j] = net->W1[i * INPUT_DIM + j] * input_img[j];
            if (weighted_activation_map[j] < map_min) map_min = weighted_activation_map[j];
            if (weighted_activation_map[j] > map_max) map_max = weighted_activation_map[j];
        }

        char map_title[100];
        snprintf(map_title, sizeof(map_title), 
                 "Weighted Activation Map (Feature %d / Min %.4f to Max %.4f)", 
                 i + 1, map_min, map_max);
        
        print_ascii_map(weighted_activation_map, map_min, map_max, map_title);
        
        // Print the pre-ReLU, pre-sigmoid sum for the feature (h_pre is calculated in forward_network)
        printf(" -> Pre-ReLU Sum (W1.x + b1): %.4f\n", net->h1_pre[i]);
    }
    printf("--------------------------------------\n");
    printf("FINAL NETWORK OUTPUT: %.4f\n", final_output);
    printf("===========================================================================\n");
}


void analyze_activation_patterns(const NetConstrained *net) {
    float rect_img[INPUT_DIM];
    float noisy_img[INPUT_DIM];
    float rotated_img[INPUT_DIM];

    generate_test_rect_3x3(rect_img);
    analyze_single_test_case(net, rect_img, "3x3 Rectangle (Perfect Match)");

    generate_test_noisy_blank(noisy_img);
    analyze_single_test_case(net, noisy_img, "Noisy Blank (Absent/Distractor Case)");

    generate_test_rotated(rotated_img);
    analyze_single_test_case(net, rotated_img, "Rotated Rectangle (Generalization Test)");
}


int main() {
    setbuf(stdout, NULL); 
    srand(time(NULL));
    
    NetConstrained constrained_net;

    printf("Starting Constrained (L2-Regularized) Network Benchmark (N=4 only) for Feature Analysis.\n");
    printf("Training data includes complex linear distractors and 20%% random noise.\n\n");

    // --- Training ---
    int N_eff = N_VISUALIZE;
    float input_image[INPUT_DIM];
    float target;
    float avg_loss = 0.0f;

    init_net(&constrained_net, N_eff); 
    clock_t start_time = clock();

    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        int is_present = rand() % 2;
        int do_rotate = is_present && (rand() % 2); 
        target = (float)is_present;
        
        generate_training_image(input_image, is_present, do_rotate);

        float final_output = forward_network(&constrained_net, input_image, N_eff);
        float mse_loss = (target - final_output) * (target - final_output);
        avg_loss = avg_loss * 0.99f + mse_loss * 0.01f;

        float delta_final = (final_output - target) * final_output * (1.0f - final_output);
        backward_constrained(&constrained_net, input_image, delta_final, LEARNING_RATE, N_eff);
    }

    clock_t end_time = clock();
    float training_time_sec = (float)(end_time - start_time) / CLOCKS_PER_SEC;

    // --- Testing ---
    generate_training_image(input_image, 1, 0); 
    float test_present = forward_network(&constrained_net, input_image, N_eff);
    generate_training_image(input_image, 0, 0); 
    float test_absent = forward_network(&constrained_net, input_image, N_eff);
    generate_training_image(input_image, 1, 1); 
    float test_rotated = forward_network(&constrained_net, input_image, N_eff);

    printf("[Constrained Benchmark N=%-3d] Time: %.2fs | Loss: %.6f | Present: %.4f, Absent (Distractor): %.4f, Rotated: %.4f\n", 
           N_eff, training_time_sec, avg_loss, test_present, test_absent, test_rotated);

    // --- Visualization ---
    visualize_features(&constrained_net);
    analyze_activation_patterns(&constrained_net);

    return 0;
}