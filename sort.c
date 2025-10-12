#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define INPUT_DIM 1         // Single normalized integer input
#define HIDDEN_DIM 4        // N=4 hidden neurons
#define OUTPUT_DIM 9        // 9 binary outputs as requested (bits 0-8)
#define MAX_INT_VALUE 127   // Range is [0, 127]

#define EPOCHS 300000       // Increased epochs for convergence on numerical task
#define GRADIENT_CLIP_MAX 0.1f 
#define LEARNING_RATE 0.05f 
#define L2_REG_LAMBDA 0.0001f 

// --- 1. Network Structure ---

typedef struct { 
    // W1: HIDDEN_DIM x INPUT_DIM (4x1)
    float W1[HIDDEN_DIM * INPUT_DIM]; 
    float b1[HIDDEN_DIM];             // 4x1
    // W2: OUTPUT_DIM x HIDDEN_DIM (9x4)
    float W2[OUTPUT_DIM * HIDDEN_DIM]; 
    float b2[OUTPUT_DIM];             // 9x1
    
    // Activation storage
    float h1_pre[HIDDEN_DIM];         // 4x1 pre-ReLU
    float h1_act[HIDDEN_DIM];         // 4x1 ReLU output
} NetBinary;


// --- 2. Utilities and Initialization ---

float ReLU(float x) { return (x) > 0.0f ? (x) : 0.0f; }
float rand_uniform(float min, float max) { return (max - min) * ((float)rand() / (float)RAND_MAX) + min; }
float clip_gradient(float grad) {
    if (grad > GRADIENT_CLIP_MAX) return GRADIENT_CLIP_MAX;
    if (grad < -GRADIENT_CLIP_MAX) return -GRADIENT_CLIP_MAX;
    return grad;
}
float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// Matrix-Vector multiplication: y = A * x (M x N * N x 1 -> M x 1)
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

void init_net(NetBinary *net) { 
    init_weights_he(net->W1, HIDDEN_DIM, INPUT_DIM); 
    init_bias(net->b1, HIDDEN_DIM); 
    init_weights_he(net->W2, OUTPUT_DIM, HIDDEN_DIM); 
    init_bias(net->b2, OUTPUT_DIM); 
}


// --- 3. Data Generation ---

// Fills target with the 9-bit binary representation of the integer I
// Bit 0 is the least significant bit (rightmost). Bit 8 is the most significant.
void generate_binary_target(int I, float *target) {
    for (int i = 0; i < OUTPUT_DIM; i++) {
        // Bit extraction: (I >> i) & 1
        target[i] = (float)((I >> i) & 1);
    }
}


// --- 4. Forward Pass ---

void forward_network(NetBinary *net, const float *input, float *output) {
    // Hidden Layer (W1 * x + b1)
    mat_vec_mul(net->W1, HIDDEN_DIM, INPUT_DIM, input, net->h1_pre); 
    for(int i = 0; i < HIDDEN_DIM; i++) {
        net->h1_pre[i] += net->b1[i];
        net->h1_act[i] = ReLU(net->h1_pre[i]);
    }
    
    // Output Layer (W2 * h1_act + b2)
    mat_vec_mul(net->W2, OUTPUT_DIM, HIDDEN_DIM, net->h1_act, output); 
    for(int i = 0; i < OUTPUT_DIM; i++) {
        output[i] += net->b2[i];
        output[i] = sigmoid(output[i]); // Apply sigmoid to each output neuron
    }
}


// --- 5. Backward Pass ---

float backward_constrained(NetBinary *net, const float *input, const float *target, float lr) {
    float final_output[OUTPUT_DIM];
    forward_network(net, input, final_output); // Re-run forward pass to get fresh state

    float mse_loss = 0.0f;
    float delta_out[OUTPUT_DIM];
    
    // Calculate Multi-Output Loss and Output Delta
    for (int k = 0; k < OUTPUT_DIM; k++) {
        float error = final_output[k] - target[k];
        mse_loss += error * error;
        // Delta = (Error) * d(Sigmoid)
        delta_out[k] = error * final_output[k] * (1.0f - final_output[k]);
    }
    mse_loss /= OUTPUT_DIM; // Average MSE

    // Backpropagate to W2 and b2
    float delta_h1_act[HIDDEN_DIM] = {0.0f};
    for (int k = 0; k < OUTPUT_DIM; k++) {
        // Gradient for W2[k, i] = delta_out[k] * h1_act[i]
        for (int i = 0; i < HIDDEN_DIM; i++) {
            float grad_W2 = delta_out[k] * net->h1_act[i];
            float total_grad_W2 = grad_W2 + L2_REG_LAMBDA * net->W2[k * HIDDEN_DIM + i];
            net->W2[k * HIDDEN_DIM + i] -= lr * clip_gradient(total_grad_W2);
            
            // Accumulate delta for next layer (W2[k, i] * delta_out[k])
            delta_h1_act[i] += net->W2[k * HIDDEN_DIM + i] * delta_out[k];
        }
        // Update b2[k]
        float grad_b2 = delta_out[k];
        net->b2[k] -= lr * clip_gradient(grad_b2);
    }
    
    // Backpropagate to W1 and b1
    for (int i = 0; i < HIDDEN_DIM; i++) {
        // Delta Pre-ReLU = Delta Act * d(ReLU)
        float delta_h1_pre = delta_h1_act[i] * (net->h1_pre[i] > 0.0f ? 1.0f : 0.0f);
        
        // Update W1[i, j] (j is always 0 since INPUT_DIM=1)
        for (int j = 0; j < INPUT_DIM; j++) {
            float grad_W1 = delta_h1_pre * input[j];
            float total_grad_W1 = grad_W1 + L2_REG_LAMBDA * net->W1[i * INPUT_DIM + j];
            net->W1[i * INPUT_DIM + j] -= lr * clip_gradient(total_grad_W1);
        }
        
        // Update b1[i]
        net->b1[i] -= lr * clip_gradient(delta_h1_pre);
    }
    
    return mse_loss;
}


// --- 6. Testing and Visualization ---

void run_test(NetBinary *net) {
    float input[INPUT_DIM];
    float target[OUTPUT_DIM];
    float output[OUTPUT_DIM];
    
    printf("\n\n--- NETWORK PREDICTIONS (I=0 to I=127) ---\n");
    printf("I | Target (9-bit) | Prediction (9-bit thresholded)\n");
    printf("--|------------------|------------------------------------\n");
    
    float total_accuracy = 0.0f;
    int correct_predictions = 0;

    for (int I = 0; I <= MAX_INT_VALUE; I++) {
        // 1. Prepare input and target
        input[0] = (float)I / MAX_INT_VALUE;
        generate_binary_target(I, target);
        
        // 2. Run forward pass
        forward_network(net, input, output);
        
        // 3. Print results
        printf("%-2d| ", I);
        
        // Print Target (LSB to MSB)
        for (int k = OUTPUT_DIM - 1; k >= 0; k--) {
            printf("%d", (int)target[k]);
        }
        printf(" | ");

        // Print Prediction (LSB to MSB)
        int bits_correct = 0;
        for (int k = OUTPUT_DIM - 1; k >= 0; k--) {
            int predicted_bit = (output[k] > 0.5f) ? 1 : 0;
            int target_bit = (int)target[k];
            printf("%d", predicted_bit);
            if (predicted_bit == target_bit) {
                bits_correct++;
            }
        }
        printf(" (Acc: %.2f)", (float)bits_correct / OUTPUT_DIM);
        
        if (bits_correct == OUTPUT_DIM) {
            correct_predictions++;
        }
        
        printf("\n");
    }
    total_accuracy = (float)correct_predictions / (MAX_INT_VALUE + 1);
    printf("---------------------------------------------------\n");
    printf("Total Integers Correctly Converted: %d/%d (%.2f%%)\n", correct_predictions, MAX_INT_VALUE + 1, total_accuracy * 100.0f);
}


// --- 7. Main Function (Training) ---

int main() {
    setbuf(stdout, NULL); 
    srand(time(NULL));
    
    NetBinary net;
    init_net(&net); 
    
    clock_t start_time = clock();
    float input[INPUT_DIM];
    float target[OUTPUT_DIM];
    float avg_loss = 0.0f;
    float current_loss;

    printf("Starting Training for %d -> 4 -> %d Binary Converter...\n", INPUT_DIM, OUTPUT_DIM);

    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        int I = rand() % (MAX_INT_VALUE + 1); // Random integer in [0, 127]
        
        input[0] = (float)I / MAX_INT_VALUE;
        generate_binary_target(I, target);
        
        current_loss = backward_constrained(&net, input, target, LEARNING_RATE);
        avg_loss = avg_loss * 0.999f + current_loss * 0.001f;
        
        if (epoch % 30000 == 0) {
            printf("[Epoch %d] Average Loss: %.8f\n", epoch, avg_loss);
        }
    }
    clock_t end_time = clock();
    float training_time_sec = (float)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("\nTraining Complete. Time: %.2fs\n", training_time_sec);

    // Run the visualization/test
    run_test(&net);

    return 0;
}