#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define INPUT_DIM 100       
#define N_VISUALIZE 4       
#define MAX_HIDDEN_DIM N_VISUALIZE
#define EPOCHS 150000       
#define IMAGE_SIZE 10
#define GRADIENT_CLIP_MAX 0.1f 
#define PI 3.14159265358979323846f
#define LEARNING_RATE 0.03f 
#define L2_REG_LAMBDA 0.0001f 

// --- 1. Network Structure (Truncated for brevity) ---

typedef struct { 
    float W1[MAX_HIDDEN_DIM * INPUT_DIM]; 
    float b1[MAX_HIDDEN_DIM]; 
    float W2[MAX_HIDDEN_DIM]; 
    float b2; 
    float h1_pre[MAX_HIDDEN_DIM]; 
} NetConstrained;


// --- 2. Function Prototypes (Truncated for brevity) ---

float ReLU(float x);
float rand_uniform(float min, float max);
float sigmoid(float x);
void mat_vec_mul(const float *A, int M, int N_A, const float *x, float *y);

// Data Generation Prototypes
void generate_training_image(float *img, int is_present, int is_rotated);
void generate_test_rect_3x3(float *img);
void generate_test_noisy_blank(float *img);
void generate_test_rotated(float *img);

// Visualization Prototypes
void print_ascii_image(const float *data, float min_val, float max_val, const char *title, int index);
void visualize_weighted_input(const NetConstrained *net, const float *input, const char *input_title);
void visualize_features(const NetConstrained *net);
void analyze_activation_patterns(const NetConstrained *net);


// --- 3. Utilities, Forward/Backward Passes (Truncated for brevity) ---

float ReLU(float x) { return (x) > 0.0f ? (x) : 0.0f; }
float rand_uniform(float min, float max) { return (max - min) * ((float)rand() / (float)RAND_MAX) + min; }
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

float forward_network(NetConstrained *net, const float *input, int N_eff) {
    mat_vec_mul(net->W1, N_eff, INPUT_DIM, input, net->h1_pre); 
    for(int i=0; i<N_eff; i++) net->h1_pre[i] += net->b1[i];
    float z_out = 0.0f; 
    for (int i = 0; i < N_eff; i++) { z_out += net->W2[i] * ReLU(net->h1_pre[i]); }
    z_out += net->b2;
    return sigmoid(z_out);
}

void backward_constrained(NetConstrained *net, const float *input, float delta_out, float lr, int N_eff) {
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
    float grad_b2 = delta_out;
    for (int i = 0; i < N_eff; i++) {
        float total_grad_W2 = grad_W2[i] + L2_REG_LAMBDA * net->W2[i];
        net->W2[i] -= lr * clip_gradient(total_grad_W2);
    }
    net->b2 -= lr * clip_gradient(grad_b2);
    for (int i = 0; i < N_eff * INPUT_DIM; i++) {
        float total_grad_W1 = grad_W1[i] + L2_REG_LAMBDA * net->W1[i];
        net->W1[i] -= lr * clip_gradient(total_grad_W1);
    }
    for (int i = 0; i < N_eff; i++) net->b1[i] -= lr * clip_gradient(delta_h1_pre[i]);
}


// --- 4. Data Generation (Function bodies) ---

// ** CRITICAL CHANGE HERE: Fixed 4x4 rectangle for positive case **
void generate_training_image(float *img, int is_present, int is_rotated) {
    for (int i = 0; i < INPUT_DIM; i++) img[i] = 0.0f;
    
    if (is_present) {
        // Always generate a centered 4x4 rectangle (rows 3-6, cols 3-6)
        for (int r = 3; r < 7; r++) { 
            for (int c = 3; c < 7; c++) { 
                img[r * IMAGE_SIZE + c] = 1.0f;
            }
        }
    }
    
    // Non-Present/Distractor case remains the same (linear shapes)
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
    
    // Noise remains the same (20%)
    int noise_pixels = (int)(INPUT_DIM * 0.20f);
    for (int i = 0; i < noise_pixels; i++) {
        int idx = rand() % INPUT_DIM;
        if (img[idx] == 0.0f) {
            img[idx] = 0.2f; 
        }
    }
}


void generate_test_rect_3x3(float *img) { 
    for (int i = 0; i < INPUT_DIM; i++) img[i] = 0.0f;
    int center = IMAGE_SIZE / 2;
    for (int r = center - 1; r <= center + 1; r++) {
        for (int c = center - 1; c <= center + 1; c++) {
            img[r * IMAGE_SIZE + c] = 1.0f;
        }
    }
}

void generate_test_noisy_blank(float *img) { 
    for (int i = 0; i < INPUT_DIM; i++) img[i] = 0.0f;
    for (int i = 0; i < INPUT_DIM * 0.2; i++) {
        img[i * 5] = 0.2f; 
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


// --- 5. Visualization Utilities (Truncated for brevity) ---

void print_ascii_image(const float *data, float min_val, float max_val, const char *title, int index) {
    printf("\n%s (Feature %d / Min %.4f to Max %.4f):\n", title, index, min_val, max_val);
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

void visualize_weighted_input(const NetConstrained *net, const float *input, const char *input_title) {
    int N_eff = N_VISUALIZE;
    float weighted_map[INPUT_DIM];
    
    printf("\n\n--- INPUT IMAGE: %s ---\n", input_title);
    
    // Display the input image itself (0-9 scale)
    float input_min = 0.0f, input_max = 1.0f;
    for (int r = 0; r < IMAGE_SIZE; r++) {
        for (int c = 0; c < IMAGE_SIZE; c++) {
            float val = input[r * IMAGE_SIZE + c];
            int scaled_val = (int)roundf(((val - input_min) / (input_max - input_min)) * 9.0f);
            printf("%d ", scaled_val);
        }
        printf("\n");
    }
    
    printf("\n--- ELEMENT-WISE WEIGHTED INPUT MAPS (x * W1_i) ---\n");
    
    for (int i = 0; i < N_eff; i++) {
        float map_min = 0.0f, map_max = 0.0f;
        for (int j = 0; j < INPUT_DIM; j++) {
            weighted_map[j] = input[j] * net->W1[i * INPUT_DIM + j];
            if (weighted_map[j] < map_min) map_min = weighted_map[j];
            if (weighted_map[j] > map_max) map_max = weighted_map[j];
        }
        
        char title[80];
        snprintf(title, sizeof(title), "Weighted Map (Feature %d / Bias: %.4f / Output Weight: %.4f)", 
                 i + 1, net->b1[i], net->W2[i]);
        print_ascii_image(weighted_map, map_min, map_max, title, i + 1);
        
        float h_pre_sum = 0.0f;
        for (int j = 0; j < INPUT_DIM; j++) { h_pre_sum += weighted_map[j]; }
        h_pre_sum += net->b1[i]; 
        printf(" -> Pre-ReLU Sum (W1.x + b1): %.4f\n", h_pre_sum);
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
    
    printf("\n\n---------------------------------------------------------------------------\n");
    printf("CONSTRAINED/L2 FEATURE MAPS (N=%d) - Trained on FIXED 4x4 RECTANGLE\n", N_eff);
    printf("Weights normalized from global_min=%.4f to global_max=%.4f\n", global_min, global_max);
    printf("---------------------------------------------------------------------------\n");

    for (int i = 0; i < N_eff; i++) {
        char title[50];
        snprintf(title, sizeof(title), "Feature %d (Input Weights of Neuron %d)", i + 1, i + 1);
        print_ascii_image(&net->W1[i * INPUT_DIM], global_min, global_max, title, i + 1);
        printf("W2 Output Weight: %.4f, Bias: %.4f\n", net->W2[i], net->b1[i]);
    }
}

void analyze_activation_patterns(const NetConstrained *net) {
    float rect_img_4x4[INPUT_DIM]; // The new positive training case
    float rect_img_3x3[INPUT_DIM]; // Generalization test
    float noisy_img[INPUT_DIM];
    float rotated_img[INPUT_DIM];

    // Generate the exact 4x4 template used for training
    for (int i = 0; i < INPUT_DIM; i++) rect_img_4x4[i] = 0.0f;
    for (int r = 3; r < 7; r++) { for (int c = 3; c < 7; c++) { rect_img_4x4[r * IMAGE_SIZE + c] = 1.0f; } }
    visualize_weighted_input(net, rect_img_4x4, "4x4 Fixed Rectangle (Perfect Match Test)");

    // Generate other tests (3x3 is near match, rotated is generalization failure, noisy blank is distractor rejection)
    generate_test_rect_3x3(rect_img_3x3);
    visualize_weighted_input(net, rect_img_3x3, "3x3 Rectangle (Near Match/Generalization Test)");

    generate_test_noisy_blank(noisy_img);
    visualize_weighted_input(net, noisy_img, "Noisy Blank (Absent/Distractor Case)");

    generate_test_rotated(rotated_img);
    visualize_weighted_input(net, rotated_img, "Rotated Rectangle (Generalization Failure Test)");
}


// --- 6. Main Function ---

int main() {
    setbuf(stdout, NULL); 
    srand(time(NULL));
    
    NetConstrained constrained_net;
    int N_eff = N_VISUALIZE;

    init_net(&constrained_net, N_eff); 
    clock_t start_time = clock();
    float input_image[INPUT_DIM];
    float target;
    float avg_loss = 0.0f;

    // Training loop uses the new fixed 4x4 template logic
    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        int is_present = rand() % 2;
        int do_rotate = 0; // Rotation is irrelevant for fixed template
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
    // Note: The training image generation is used for testing here, but we set up custom tests in analyze_activation_patterns
    generate_training_image(input_image, 1, 0); 
    float test_present = forward_network(&constrained_net, input_image, N_eff);
    generate_training_image(input_image, 0, 0); 
    float test_absent = forward_network(&constrained_net, input_image, N_eff);
    
    // We expect the ROTATED test to fail since it was not trained on rotation
    float test_rotated = forward_network(&constrained_net, input_image, N_eff); 

    printf("Starting Constrained (L2-Regularized) Network Analysis (N=4).\n");
    printf("[Constrained Benchmark N=%-3d] Time: %.2fs | Loss: %.6f | Present (4x4): %.4f, Absent (Distractor): %.4f, Rotated: %.4f\n", 
           N_eff, training_time_sec, avg_loss, test_present, test_absent, test_rotated);

    // --- Visualization ---
    visualize_features(&constrained_net);
    analyze_activation_patterns(&constrained_net);

    return 0;
}