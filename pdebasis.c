#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h> 
#include <float.h>
#include <string.h> 

// --- Configuration ---

// **Network Configuration (Text Compression / Lookup Map)**
#define N_INPUT 256            // Index 0 to 255 (One-Hot Encoded)
#define N_OUTPUT 512           // 64 characters * 8 bits/char
#define SENTENCE_LENGTH 64     // Length of output sentence
#define NUM_TRAINING_CASES 256 // Directly defined size of the dataset

// **Hidden Layer Sizes**
#define N_HIDDEN1 16 
#define N_HIDDEN2 16 
#define N_HIDDEN3 16 
#define N_HIDDEN4 16 

// **Training Parameters**
#define TRAINING_TIME_LIMIT 60.0 
#define BATCH_SIZE 32          
#define REPORT_FREQ 500             
#define INITIAL_LEARNING_RATE 0.0001 
#define REGRESSION_WEIGHT 1.0      

// **Adam Optimizer Parameters**
#define BETA1 0.9
#define BETA2 0.999
#define EPSILON 1e-8

// --- Global Data & Matrices (4 Layers) ---

// Weights and Biases 
double w_f1[N_INPUT][N_HIDDEN1];    
double b_1[N_HIDDEN1]; 
double w_12[N_HIDDEN1][N_HIDDEN2];  
double b_2[N_HIDDEN2];
double w_23[N_HIDDEN2][N_HIDDEN3];  
double b_3[N_HIDDEN3];
double w_34[N_HIDDEN3][N_HIDDEN4];  
double b_4[N_HIDDEN4];
double w_4o[N_HIDDEN4][N_OUTPUT];   
double b_o[N_OUTPUT];

// Adam State Variables
double m_w_f1[N_INPUT][N_HIDDEN1], v_w_f1[N_INPUT][N_HIDDEN1];
double m_b_1[N_HIDDEN1], v_b_1[N_HIDDEN1];
double m_w_12[N_HIDDEN1][N_HIDDEN2], v_w_12[N_HIDDEN1][N_HIDDEN2];
double m_b_2[N_HIDDEN2], v_b_2[N_HIDDEN2];
double m_w_23[N_HIDDEN2][N_HIDDEN3], v_w_23[N_HIDDEN2][N_HIDDEN3];
double m_b_3[N_HIDDEN3], v_b_3[N_HIDDEN3];
double m_w_34[N_HIDDEN3][N_HIDDEN4], v_w_34[N_HIDDEN3][N_HIDDEN4];
double m_b_4[N_HIDDEN4], v_b_4[N_HIDDEN4];
double m_w_4o[N_HIDDEN4][N_OUTPUT], v_w_4o[N_HIDDEN4][N_OUTPUT];
double m_b_o[N_OUTPUT], v_b_o[N_OUTPUT];

// Input Normalization Stats
double input_mean = 0.0;
double input_std = 1.0;

// Data Storage (NUM_TRAINING_CASES used instead of NUM_IMAGES)
char sentences[NUM_TRAINING_CASES][SENTENCE_LENGTH + 1]; 
double target_properties[NUM_TRAINING_CASES][N_OUTPUT];  
double single_images[NUM_TRAINING_CASES][N_INPUT];       

// --- Profiling Setup ---
enum FuncName {
    PROFILE_GENERATE_DATA, PROFILE_LOAD_TRAIN_CASE,
    PROFILE_FORWARD_PASS, PROFILE_BACKPROP_UPDATE, 
    PROFILE_TRAIN_NN, PROFILE_TEST_NN,
    NUM_FUNCTIONS 
};
const char *func_names[NUM_FUNCTIONS] = {
    "generate_data", "load_train_case", 
    "forward_pass", "backprop_update", 
    "train_nn", "test_nn"
};
clock_t func_times[NUM_FUNCTIONS] = {0}; 

#define START_PROFILE(func) clock_t start_##func = clock();
#define END_PROFILE(func) func_times[func] += (clock() - start_##func);

// --- Activation Functions (ReLU for hidden, Sigmoid for output) ---

double poly_activation(double z_net) { 
    return (z_net > 0) ? z_net : 0.0; // ReLU
} 
double poly_derivative(double z_net) { 
    return (z_net > 0) ? 1.0 : 0.0;   // ReLU derivative
}
double sigmoid(double z) { return 1.0 / (1.0 + exp(-z)); }
double sigmoid_derivative(double z, double output) { return output * (1.0 - output); }


// --- Data Generation Functions ---

void hardcode_sentences() {
    // Generate 256 distinct 64-character strings based on tech/comp sci topics.
    const char *base_text = 
        "The quick brown fox jumps over the lazy dog. Programming is a process "
        "that leads to the creation of executable computer programs. Computer "
        "programs contain a sequence of instructions written in a programming "
        "language. This language can be high-level or low-level. The sun rises "
        "in the east and sets in the west. Neural networks are a set of "
        "algorithms, modeled after the human brain, that are designed to "
        "recognize patterns. They are used to make predictions or classifications. "
        "The capital of France is Paris. The history of computing is fascinating "
        "and spans centuries of innovation. The current year is 2025 and AI is "
        "rapidly advancing. Compression tools encode information efficiently. "
        "A multi-task network solves several problems simultaneously. "
        "The deep architecture 4x16 implies four hidden layers of sixteen neurons."
        "The gradient descent algorithm is used to minimize the loss function. "
        "It updates the weights in the opposite direction of the gradient. "
        "Adam and RMSprop are common adaptive learning rate optimizers."
        "The universe is vast and contains many mysteries. Binary code is the "
        "fundamental language of computers, consisting only of zeros and ones. "
        "All data, including text and images, is ultimately represented in binary. "
        "Floating point numbers are used to approximate real values. Backpropagation "
        "is the core algorithm for training deep neural networks. It calculates "
        "the gradient of the loss function with respect to the weights. "
        "Optimization is the key to training large-scale models effectively. "
        "The Hessian matrix contains second-order derivative information. "
        "Eigenvalues describe the curvature of the loss landscape. "
        "Second-order methods like Newton's method use the inverse Hessian. "
        "Quasi-Newton methods approximate the inverse Hessian efficiently. "
        "L-BFGS is a memory-efficient Quasi-Newton optimization algorithm. "
        "Adaptive methods, like Adam, scale the learning rate per parameter. "
        "This scaling mimics a diagonal approximation of the Hessian inverse. "
        "The condition number of the Hessian dictates convergence speed. "
        "Ill-conditioned landscapes lead to slow, zigzagging paths in training. "
        "The complexity of computing the full Hessian inverse is prohibitive. "
        "Transforming calculations to the 'determinant space' means diagonalization. "
        "Diagonalization simplifies matrix multiplication to scalar scaling. "
        "The Fast Fourier Transform (FFT) is an example of such a transformation. "
        "It converts convolution in the time domain to multiplication in frequency. "
        "Similarly, using eigenvalues simplifies the gradient step's geometry. "
        "The eigen-space changes continuously as the network weights are updated. "
        "Therefore, one cannot simply initialize parameters in the eigen-space. "
        "Iteration and approximation are necessary to follow the changing space. "
        "The weights must be updated frequently to match the local curvature. "
        "The entire dataset must be processed efficiently for large models. "
        "Batch processing helps to average the noise in the gradient estimates. "
        "The learning rate hyperparameter controls the step size magnitude. "
        "Weight initialization is crucial for avoiding vanishing gradients. "
        "ReLU activation prevents saturation in deep network hidden layers. "
        "Sigmoid is used for the output layer when predicting binary outcomes. "
        "Binary Cross-Entropy or MSE are suitable loss functions for this task. "
        "The output of this network is a compressed text representation. "
        "A one-hot index input maps to a 64-character encoded sentence output. "
        "The network acts as a lookup table with a compressed intermediate layer. "
        "Training requires perfect reconstruction of all 256 unique sentences. "
        "The small 16-neuron bottleneck forces the network to learn compression. "
        "This is an example of an Autoencoder structure's encoding mechanism. "
        "The information is forced through a highly dimensional reduction. "
        "The challenge lies in mapping the index to the latent representation. "
        "The decoder part then reconstructs the 512 binary output bits. "
        "Accuracy is measured by checking if the entire 64-char string matches. "
        "High accuracy is expected given the finite, fixed dataset size. "
        "The system should eventually memorize the 256 input-output mappings. "
        "This showcases the immense representational power of neural networks. "
        "This sentence is unique and helps fill the 256 required slots. "
        "Another unique sentence to ensure the training data is fully utilized. "
        "The last few sentences are just fillers to reach the precise length. "
        "This is the end of the base text for data generation purposes. ";


    int base_len = strlen(base_text);

    for (int i = 0; i < NUM_TRAINING_CASES; i++) {
        // Find a unique 64-char substring 
        int start = (i * 10 + i * i) % (base_len - SENTENCE_LENGTH);
        strncpy(sentences[i], base_text + start, SENTENCE_LENGTH);
        sentences[i][SENTENCE_LENGTH] = '\0';
        
        // Ensure sentences are unique by adding a varying character at the start
        sentences[i][0] = (char)(33 + (i % 94)); // Printable ASCII: '!' to '~'
        
        // Ensure a null terminator is at the end of the 64-char block
        sentences[i][SENTENCE_LENGTH] = '\0';
    }
}

void generate_data() {
    START_PROFILE(PROFILE_GENERATE_DATA)
    hardcode_sentences();
    
    // 1. Convert sentences to binary targets and generate one-hot inputs
    for (int i = 0; i < NUM_TRAINING_CASES; i++) {
        // Input: One-Hot vector (Index i)
        for (int k = 0; k < N_INPUT; k++) single_images[i][k] = 0.0;
        single_images[i][i] = 1.0;
        
        // Target: 64 chars * 8 bits = 512 bits
        for (int c = 0; c < SENTENCE_LENGTH; c++) {
            char ascii_char = sentences[i][c];
            for (int b = 0; b < 8; b++) {
                // Check the b-th bit (from MSB to LSB: b=0 is MSB, b=7 is LSB)
                int bit_index = c * 8 + b;
                if ((ascii_char >> (7 - b)) & 1) {
                    target_properties[i][bit_index] = 1.0;
                } else {
                    target_properties[i][bit_index] = 0.0;
                }
            }
        }
    }

    // Normalization Stats (Irrelevant for One-Hot Input)
    input_mean = 0.0; 
    input_std = 1.0; 
    
    END_PROFILE(PROFILE_GENERATE_DATA)
}

void load_train_case(double input[N_INPUT], double target[N_OUTPUT]) {
    START_PROFILE(PROFILE_LOAD_TRAIN_CASE)
    int img_idx = rand() % NUM_TRAINING_CASES;
    
    // Load one-hot input (no normalization needed for 0/1)
    memcpy(input, single_images[img_idx], N_INPUT * sizeof(double)); 
    
    // Load 512-bit binary target
    memcpy(target, target_properties[img_idx], N_OUTPUT * sizeof(double)); 
    END_PROFILE(PROFILE_LOAD_TRAIN_CASE)
}


// --- NN Core Functions ---

void initialize_nn() {
    #define XAVIER_LIMIT(Nin, Nout) sqrt(6.0 / ((double)(Nin) + (Nout)))
    
    // Input (256) -> H1 (16)
    double limit_f1 = XAVIER_LIMIT(N_INPUT, N_HIDDEN1);
    for (int i = 0; i < N_INPUT; i++) for (int j = 0; j < N_HIDDEN1; j++) w_f1[i][j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * limit_f1; 
    for (int j = 0; j < N_HIDDEN1; j++) b_1[j] = 0.0; 
    
    // H1 (16) -> H2 (16)
    double limit_12 = XAVIER_LIMIT(N_HIDDEN1, N_HIDDEN2);
    for (int i = 0; i < N_HIDDEN1; i++) for (int j = 0; j < N_HIDDEN2; j++) w_12[i][j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * limit_12; 
    for (int j = 0; j < N_HIDDEN2; j++) b_2[j = 0.0; 

    // H2 (16) -> H3 (16)
    double limit_23 = XAVIER_LIMIT(N_HIDDEN2, N_HIDDEN3);
    for (int i = 0; i < N_HIDDEN2; i++) for (int j = 0; j < N_HIDDEN3; j++) w_23[i][j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * limit_23; 
    for (int j = 0; j < N_HIDDEN3; j++) b_3[j] = 0.0; 
    
    // H3 (16) -> H4 (16)
    double limit_34 = XAVIER_LIMIT(N_HIDDEN3, N_HIDDEN4);
    for (int i = 0; i < N_HIDDEN3; i++) for (int j = 0; j < N_HIDDEN4; j++) w_34[i][j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * limit_34; 
    for (int j = 0; j < N_HIDDEN4; j++) b_4[j] = 0.0; 
    
    // H4 (16) -> Output (512)
    double limit_4o = XAVIER_LIMIT(N_HIDDEN4, N_OUTPUT);
    for (int i = 0; i < N_HIDDEN4; i++) for (int j = 0; j < N_OUTPUT; j++) w_4o[i][j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * limit_4o; 
    for (int k = 0; k < N_OUTPUT; k++) b_o[k] = 0.0;
    
    // Initialize Adam states to zero
    memset(m_w_f1, 0, sizeof(m_w_f1)); memset(v_w_f1, 0, sizeof(v_w_f1)); memset(m_b_1, 0, sizeof(m_b_1)); memset(v_b_1, 0, sizeof(v_b_1));
    memset(m_w_12, 0, sizeof(m_w_12)); memset(v_w_12, 0, sizeof(v_w_12)); memset(m_b_2, 0, sizeof(m_b_2)); memset(v_b_2, 0, sizeof(v_b_2));
    memset(m_w_23, 0, sizeof(m_w_23)); memset(v_w_23, 0, sizeof(v_w_23)); memset(m_b_3, 0, sizeof(m_b_3)); memset(v_b_3, 0, sizeof(v_b_3));
    memset(m_w_34, 0, sizeof(m_w_34)); memset(v_w_34, 0, sizeof(v_w_34)); memset(m_b_4, 0, sizeof(m_b_4)); memset(v_b_4, 0, sizeof(v_b_4));
    memset(m_w_4o, 0, sizeof(m_w_4o)); memset(v_w_4o, 0, sizeof(v_w_4o)); memset(m_b_o, 0, sizeof(m_b_o)); memset(v_b_o, 0, sizeof(v_b_o));

    #undef XAVIER_LIMIT
}

// Global activation and net buffers 
double h1_net[N_HIDDEN1], h1_out[N_HIDDEN1];
double h2_net[N_HIDDEN2], h2_out[N_HIDDEN2];
double h3_net[N_HIDDEN3], h3_out[N_HIDDEN3];
double h4_net[N_HIDDEN4], h4_out[N_HIDDEN4];

void forward_pass(const double input[N_INPUT], 
                  double output_net[N_OUTPUT], double output_prob[N_OUTPUT]) {
    START_PROFILE(PROFILE_FORWARD_PASS)
    
    // --- Layer 1: Input (256) to H1 (16) ---
    for (int j = 0; j < N_HIDDEN1; j++) {
        double h_net = b_1[j];
        for (int i = 0; i < N_INPUT; i++) h_net += input[i] * w_f1[i][j]; 
        h1_net[j] = h_net;
        h1_out[j] = poly_activation(h_net); // ReLU
    }
    
    // --- Layer 2: H1 to H2 (16 -> 16) ---
    for (int j = 0; j < N_HIDDEN2; j++) {
        double h_net = b_2[j];
        for (int i = 0; i < N_HIDDEN1; i++) h_net += h1_out[i] * w_12[i][j]; 
        h2_net[j] = h_net;
        h2_out[j] = poly_activation(h_net); // ReLU
    }

    // --- Layer 3: H2 to H3 (16 -> 16) ---
    for (int j = 0; j < N_HIDDEN3; j++) {
        double h_net = b_3[j];
        for (int i = 0; i < N_HIDDEN2; i++) h_net += h2_out[i] * w_23[i][j]; 
        h3_net[j] = h_net;
        h3_out[j] = poly_activation(h_net); // ReLU
    }
    
    // --- Layer 4: H3 to H4 (16 -> 16) ---
    for (int j = 0; j < N_HIDDEN4; j++) {
        double h_net = b_4[j];
        for (int i = 0; i < N_HIDDEN3; i++) h_net += h3_out[i] * w_34[i][j]; 
        h4_net[j] = h_net;
        h4_out[j] = poly_activation(h_net); // ReLU
    }

    // --- Output Layer: H4 to Output (16 -> 512) ---
    for (int k = 0; k < N_OUTPUT; k++) {
        double o_net = b_o[k]; 
        for (int j = 0; j < N_HIDDEN4; j++) o_net += h4_out[j] * w_4o[j][k]; 
        output_net[k] = o_net;
        // Final Activation: Sigmoid for 512 binary bits
        output_prob[k] = sigmoid(o_net); 
    }
    END_PROFILE(PROFILE_FORWARD_PASS)
}

// Adam Update Function
void adam_update(double *param, double *grad, double *m, double *v, int t, double lr) {
    double beta1_t = pow(BETA1, t);
    double beta2_t = pow(BETA2, t);
    
    *m = BETA1 * (*m) + (1.0 - BETA1) * (*grad);
    *v = BETA2 * (*v) + (1.0 - BETA2) * (*grad) * (*grad);
    
    double m_hat = (*m) / (1.0 - beta1_t);
    double v_hat = (*v) / (1.0 - beta2_t);
    
    *param -= lr * m_hat / (sqrt(v_hat) + EPSILON);
}

void train_nn() {
    START_PROFILE(PROFILE_TRAIN_NN)
    double input[N_INPUT], target[N_OUTPUT];
    double output_net[N_OUTPUT], output_prob[N_OUTPUT];
    
    // Gradient Accumulators for 4 layers
    double grad_w_f1_acc[N_INPUT][N_HIDDEN1] = {0.0};
    double grad_b_1_acc[N_HIDDEN1] = {0.0};
    double grad_w_12_acc[N_HIDDEN1][N_HIDDEN2] = {0.0};
    double grad_b_2_acc[N_HIDDEN2] = {0.0};
    double grad_w_23_acc[N_HIDDEN2][N_HIDDEN3] = {0.0};
    double grad_b_3_acc[N_HIDDEN3] = {0.0};
    double grad_w_34_acc[N_HIDDEN3][N_HIDDEN4] = {0.0};
    double grad_b_4_acc[N_HIDDEN4] = {0.0};
    double grad_w_4o_acc[N_HIDDEN4][N_OUTPUT] = {0.0};
    double grad_b_o_acc[N_OUTPUT] = {0.0};
    
    double cumulative_loss_report = 0.0;
    int samples_processed_in_report = 0;
    int t = 0; // Adam time step
    int epoch = 0;
    
    clock_t start_time = clock();
    
    printf("--- TRAINING PHASE START (Adam, Deep Net: 4x16, Task: Text Compression, Time Limit: %.1f s) ---\n", 
           TRAINING_TIME_LIMIT);
    
    while ((double)(clock() - start_time) / CLOCKS_PER_SEC < TRAINING_TIME_LIMIT) {
        
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            
            load_train_case(input, target);
            forward_pass(input, output_net, output_prob);
            
            START_PROFILE(PROFILE_BACKPROP_UPDATE)
            
            double delta_o[N_OUTPUT];
            double delta_4[N_HIDDEN4], error_4[N_HIDDEN4];
            double delta_3[N_HIDDEN3], error_3[N_HIDDEN3];
            double delta_2[N_HIDDEN2], error_2[N_HIDDEN2];
            double delta_1[N_HIDDEN1], error_1[N_HIDDEN1];
            double total_sample_loss = 0.0;

            // --- 1. Output Delta & Loss Calculation (MSE/L2 Loss on 512 bits) ---
            for (int k = 0; k < N_OUTPUT; k++) {
                double error = output_prob[k] - target[k];
                // Delta for MSE/L2 loss + Sigmoid derivative
                delta_o[k] = error * sigmoid_derivative(output_net[k], output_prob[k]) * REGRESSION_WEIGHT; 
                total_sample_loss += 0.5 * error * error * REGRESSION_WEIGHT;
            }
            
            // 2. Backpropagate Errors (Output -> H4)
            for (int j = 0; j < N_HIDDEN4; j++) {
                error_4[j] = 0.0;
                for (int k = 0; k < N_OUTPUT; k++) error_4[j] += delta_o[k] * w_4o[j][k];
                delta_4[j] = error_4[j] * poly_derivative(h4_net[j]); // ReLU derivative
            }
            
            // H4 -> H3
            for (int j = 0; j < N_HIDDEN3; j++) {
                error_3[j] = 0.0;
                for (int k = 0; k < N_HIDDEN4; k++) error_3[j] += delta_4[k] * w_34[j][k]; 
                delta_3[j] = error_3[j] * poly_derivative(h3_net[j]); // ReLU derivative
            }
            
            // H3 -> H2
            for (int j = 0; j < N_HIDDEN2; j++) {
                error_2[j] = 0.0;
                for (int k = 0; k < N_HIDDEN3; k++) error_2[j] += delta_3[k] * w_23[j][k]; 
                delta_2[j] = error_2[j] * poly_derivative(h2_net[j]); // ReLU derivative
            }

            // H2 -> H1
            for (int j = 0; j < N_HIDDEN1; j++) {
                error_1[j] = 0.0;
                for (int k = 0; k < N_HIDDEN2; k++) error_1[j] += delta_2[k] * w_12[j][k]; 
                delta_1[j] = error_1[j] * poly_derivative(h1_net[j]); // ReLU derivative
            }

            // 3. Accumulate Gradients (dLoss/dW = delta * input)

            // H4 -> Output
            for (int k = 0; k < N_OUTPUT; k++) {
                grad_b_o_acc[k] += delta_o[k];
                for (int j = 0; j < N_HIDDEN4; j++) grad_w_4o_acc[j][k] += delta_o[k] * h4_out[j];
            }

            // H3 -> H4
            for (int j = 0; j < N_HIDDEN4; j++) {
                grad_b_4_acc[j] += delta_4[j];
                for (int i = 0; i < N_HIDDEN3; i++) grad_w_34_acc[i][j] += delta_4[j] * h3_out[i];
            }
            
            // H2 -> H3
            for (int j = 0; j < N_HIDDEN3; j++) {
                grad_b_3_acc[j] += delta_3[j];
                for (int i = 0; i < N_HIDDEN2; i++) grad_w_23_acc[i][j] += delta_3[j] * h2_out[i];
            }

            // H1 -> H2
            for (int j = 0; j < N_HIDDEN2; j++) {
                grad_b_2_acc[j] += delta_2[j];
                for (int i = 0; i < N_HIDDEN1; i++) grad_w_12_acc[i][j] += delta_2[j] * h1_out[i];
            }
            
            // Input (256) -> H1 (16)
            for (int j = 0; j < N_HIDDEN1; j++) {
                grad_b_1_acc[j] += delta_1[j];
                for (int i = 0; i < N_INPUT; i++) grad_w_f1_acc[i][j] += delta_1[j] * input[i];
            }
            
            END_PROFILE(PROFILE_BACKPROP_UPDATE)
            cumulative_loss_report += total_sample_loss; 
            samples_processed_in_report++;

        } // END BATCH LOOP

        // --- ADAM WEIGHT UPDATE ---
        t++; 
        double inv_batch_size = 1.0 / BATCH_SIZE;
        
        // H4 -> Output
        for (int k = 0; k < N_OUTPUT; k++) {
            double grad_b_o = grad_b_o_acc[k] * inv_batch_size;
            adam_update(&b_o[k], &grad_b_o, &m_b_o[k], &v_b_o[k], t, INITIAL_LEARNING_RATE);
            grad_b_o_acc[k] = 0.0; 
            for (int j = 0; j < N_HIDDEN4; j++) {
                double grad_w_4o = grad_w_4o_acc[j][k] * inv_batch_size;
                adam_update(&w_4o[j][k], &grad_w_4o, &m_w_4o[j][k], &v_w_4o[j][k], t, INITIAL_LEARNING_RATE);
                grad_w_4o_acc[j][k] = 0.0; 
            }
        }
        
        // H3 -> H4
        for (int j = 0; j < N_HIDDEN4; j++) {
            double grad_b_4 = grad_b_4_acc[j] * inv_batch_size;
            adam_update(&b_4[j], &grad_b_4, &m_b_4[j], &v_b_4[j], t, INITIAL_LEARNING_RATE);
            grad_b_4_acc[j] = 0.0; 
            for (int i = 0; i < N_HIDDEN3; i++) {
                double grad_w_34 = grad_w_34_acc[i][j] * inv_batch_size;
                adam_update(&w_34[i][j], &grad_w_34, &m_w_34[i][j], &v_w_34[i][j], t, INITIAL_LEARNING_RATE);
                grad_w_34_acc[i][j] = 0.0; 
            }
        }

        // H2 -> H3
        for (int j = 0; j < N_HIDDEN3; j++) {
            double grad_b_3 = grad_b_3_acc[j] * inv_batch_size;
            adam_update(&b_3[j], &grad_b_3, &m_b_3[j], &v_b_3[j], t, INITIAL_LEARNING_RATE);
            grad_b_3_acc[j] = 0.0; 
            for (int i = 0; i < N_HIDDEN2; i++) {
                double grad_w_23 = grad_w_23_acc[i][j] * inv_batch_size;
                adam_update(&w_23[i][j], &grad_w_23, &m_w_23[i][j], &v_w_23[i][j], t, INITIAL_LEARNING_RATE);
                grad_w_23_acc[i][j] = 0.0; 
            }
        }
        
        // H1 -> H2
        for (int j = 0; j < N_HIDDEN2; j++) {
            double grad_b_2 = grad_b_2_acc[j] * inv_batch_size;
            adam_update(&b_2[j], &grad_b_2, &m_b_2[j], &v_b_2[j], t, INITIAL_LEARNING_RATE);
            grad_b_2_acc[j] = 0.0; 
            for (int i = 0; i < N_HIDDEN1; i++) {
                double grad_w_12 = grad_w_12_acc[i][j] * inv_batch_size;
                adam_update(&w_12[i][j], &grad_w_12, &m_w_12[i][j], &v_w_12[i][j], t, INITIAL_LEARNING_RATE);
                grad_w_12_acc[i][j] = 0.0; 
            }
        }

        // Input -> H1
        for (int j = 0; j < N_HIDDEN1; j++) {
            double grad_b_1 = grad_b_1_acc[j] * inv_batch_size;
            adam_update(&b_1[j], &grad_b_1, &m_b_1[j], &v_b_1[j], t, INITIAL_LEARNING_RATE);
            grad_b_1_acc[j] = 0.0; 
            for (int i = 0; i < N_INPUT; i++) {
                double grad_w_f1 = grad_w_f1_acc[i][j] * inv_batch_size;
                adam_update(&w_f1[i][j], &grad_w_f1, &m_w_f1[i][j], &v_w_f1[i][j], t, INITIAL_LEARNING_RATE);
                grad_w_f1_acc[i][j] = 0.0; 
            }
        }

        epoch++; 
        
        if (epoch % REPORT_FREQ == 0) {
            double time_elapsed = (double)(clock() - start_time) / CLOCKS_PER_SEC;
            printf("  Epoch: %6d | Avg Loss: %7.6f | Time Elapsed: %5.2f s\n", 
                   epoch, cumulative_loss_report / samples_processed_in_report, time_elapsed);
            cumulative_loss_report = 0.0; 
            samples_processed_in_report = 0;
        }
    }
    double total_train_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
    printf("--- TRAINING PHASE COMPLETE (Total Epochs: %d, Total Time: %.2f s) ---\n", epoch, total_train_time);
    END_PROFILE(PROFILE_TRAIN_NN)
}

// Helper: Convert 512 probability bits to 64 ASCII characters
void convert_binary_to_ascii(const double binary_prob[N_OUTPUT], char output_text[SENTENCE_LENGTH + 1]) {
    for (int c = 0; c < SENTENCE_LENGTH; c++) {
        unsigned char ascii_char = 0;
        for (int b = 0; b < 8; b++) {
            int bit_index = c * 8 + b;
            // Round the probability (0.5 threshold)
            int bit = (binary_prob[bit_index] > 0.5) ? 1 : 0;
            // Set the b-th bit (MSB first)
            ascii_char |= (bit << (7 - b));
        }
        output_text[c] = (char)ascii_char;
    }
    output_text[SENTENCE_LENGTH] = '\0';
}

void test_nn(int n_test_total) {
    START_PROFILE(PROFILE_TEST_NN)
    double input[N_INPUT], output_net[N_OUTPUT], output_prob[N_OUTPUT]; 

    int correct_sentences = 0;
    
    printf("\n--- TESTING PHASE START (%d cases total) ---\n", n_test_total);

    for (int i = 0; i < n_test_total; i++) {
        
        int index = rand() % NUM_TRAINING_CASES;
        
        // Input: One-Hot vector
        for (int k = 0; k < N_INPUT; k++) input[k] = 0.0;
        input[index] = 1.0;
        
        forward_pass(input, output_net, output_prob);
        
        // 1. Decode Prediction
        char predicted_sentence[SENTENCE_LENGTH + 1];
        convert_binary_to_ascii(output_prob, predicted_sentence);
        
        // 2. Compare to Target
        if (strcmp(predicted_sentence, sentences[index]) == 0) {
            correct_sentences++;
        }
    }
    
    printf("\nTEST SUMMARY (Text Compression / Lookup):\n");
    printf("Total Test Cases: %d\n", n_test_total);
    printf("Full Sentence Accuracy: %d / %d (%.2f%%)\n", 
           correct_sentences, n_test_total, (double)correct_sentences / n_test_total * 100.0);
    printf("--------------------------------------------------\n");

    // Print a sample visualization
    printf("\nVISUALIZATION: Sample Text Reconstruction\n");
    int sample_index = 42; 
    
    for (int k = 0; k < N_INPUT; k++) input[k] = 0.0;
    input[sample_index] = 1.0;
    forward_pass(input, output_net, output_prob);
    
    char predicted_sentence[SENTENCE_LENGTH + 1];
    convert_binary_to_ascii(output_prob, predicted_sentence);

    printf("  Input Index: %d\n", sample_index);
    printf("  Target Sentence:   '%s'\n", sentences[sample_index]);
    printf("  Predicted Sentence:'%s'\n", predicted_sentence);
    
    END_PROFILE(PROFILE_TEST_NN)
}


void print_profiling_stats() {
    printf("\n==================================================\n");
    printf("PROFILING STATS (Accumulated CPU Time)\n");
    printf("==================================================\n");
    printf("%-25s | %15s | %10s\n", "Function", "Total Time (ms)", "Total Time (s)");
    printf("--------------------------------------------------\n");
    double total_time_sec = 0.0;
    for (int i = 0; i < NUM_FUNCTIONS; i++) {
        double time_sec = (double)func_times[i] / CLOCKS_PER_SEC;
        double time_ms = time_sec * 1000.0;
        printf("%-25s | %15.3f | %10.6f\n", func_names[i], time_ms, time_sec);
        total_time_sec += time_sec;
    }
    printf("--------------------------------------------------\n");
    printf("%-25s | %15s | %10.6f\n", "TOTAL PROFILED TIME", "", total_time_sec);
    printf("==================================================\n");
}


int main() {
    srand((unsigned int)time(NULL));

    printf("--- Text Compression/Lookup NN (Deep Architecture: 4x16 FFN) ---\n");
    
    initialize_nn();
    generate_data();
    printf("Data setup complete. %d training text cases generated (256-in, 512-out). Input Mean: %.4f, Std: %.4f\n", NUM_TRAINING_CASES, input_mean, input_std);

    // 2. Train Network
    train_nn();

    // 3. Test Network (Test on all 256 generated cases)
    test_nn(NUM_TRAINING_CASES);

    // 4. Summarize Profiling
    print_profiling_stats();

    return 0;
}
