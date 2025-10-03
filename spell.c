#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <string.h>
#include <stdarg.h>

// --- Spelling Corrector & Input Constants ---
#define MAX_WORD_LEN 10             // Network input/output size (L)
// ... (All Constants remain the same) ...
#define NN_HIDDEN_SIZE 512          
#define NN_INITIAL_LEARNING_RATE 0.005 
#define NN_LR_DECAY_RATE 0.0001     

// ... (SVG Constants remain the same) ...

// ... (Vocabulary and Encoding Arrays remain the same) ...
const char* ENGLISH_WORDS[VOCAB_SIZE] = { /* ... */ };
const char ALPHABET[] = "abcdefghijklmnopqrstuvwxyz ";
int CHAR_TO_INT[128]; 

// ... (init_char_mapping remains the same) ...

// --- Data Structures ---
typedef struct { 
    int rows; 
    int cols; 
    double** data; 
    bool is_initialized; // Sanity check flag
} Matrix;

// NEW: NeuralNetwork structure definition moved here
typedef struct {
    // Learning Rate
    double lr;

    // Weights & Biases
    Matrix weights_ih;
    Matrix weights_ho;
    double* bias_h;
    double* bias_o;

    // Forward-Pass Caches (Input/Output vectors)
    Matrix inputs;
    Matrix hidden_outputs;
    Matrix output_outputs;

    // Scratchpads & Gradients
    Matrix h_in_cache;
    Matrix output_in_cache;
    Matrix targets_cache;
    Matrix output_errors;
    
    // Gradient matrices for Backpropagation
    Matrix W_grad_ho;
    Matrix h_errors_cache;
    Matrix h_d_m_cache; // Hidden delta multiplier (derivative)
    Matrix W_grad_ih;

    // Transpose Scratchpads
    Matrix h_out_t_cache;
    Matrix weights_ho_t_cache;
    Matrix inputs_t_cache;

} NeuralNetwork;


// --- Helper Macro for Matrix Sanity Check ---
#define CHECK_MATRIX(M, func_name) \
// ... (CHECK_MATRIX remains the same) ...

// --- Matrix Utility Functions (All matrix functions remain the same) ---
Matrix matrix_create(int rows, int cols, int input_size) {
// ...
}

// ... (matrix_free, matrix_copy_in, array_to_matrix_store, etc. remain the same) ...

// --- Activation Functions (Unchanged) ---
double relu_activation(double x) { /* ... */ }
double relu_derivative(double y) { /* ... */ }
void softmax(double* arr) { /* ... */ }

// --- Encoding/Decoding Functions (Unchanged) ---
void encode_word_ohe(const char* word, double* arr) { /* ... */ }
void decode_word_ohe(const double* arr, char* word_out) { /* ... */ }
void generate_misspelled_word(const char* original_word, char* misspelled_word_out) { /* ... */ }

// --- Neural Network Functions (Now correctly defined after struct) ---

// Original line 448
void nn_init(NeuralNetwork* nn) {
// ... (nn_init body remains the same) ...
}

// Original line 484
void nn_free(NeuralNetwork* nn) {
// ... (nn_free body remains the same) ...
}

// Original line 513
void nn_forward(NeuralNetwork* nn, const double* input_array, double* output_array) {
// ... (nn_forward body remains the same) ...
}

// Original line 551
double nn_backward(NeuralNetwork* nn, const double* target_array) {
// ... (nn_backward body remains the same) ...
}

// Original line 621
double train_sequential_batch(NeuralNetwork* nn, int* word_indices, double* bp_time) {
// ... (train_sequential_batch body remains the same) ...
}

// Original line 664
double test_network(NeuralNetwork* nn, double* test_time, bool verbose, int* fixed_count) {
// ... (test_network body remains the same) ...
}

// ... (SVG Utility Functions remain the same) ...

// Original line 761
void save_network_as_svg(NeuralNetwork* nn) {
// ... (save_network_as_svg body remains the same) ...
}


// --- Main Execution (Now referencing a known struct) ---

int main() {
    // ... (srand and init_char_mapping) ...
    
    // Original line 881: Now works because NeuralNetwork is defined above
    NeuralNetwork nn;
    // Original line 882: Now works because nn_init is defined above
    nn_init(&nn);

    // ... (printf statements) ...

    time_t start_time = time(NULL);
    time_t last_test_time = start_time - TEST_INTERVAL_SECONDS; 
    // ... (rest of variable declarations) ...

    for (int epoch = 0; epoch < max_epochs; epoch++) {
        // ... (time_limit_reached, lr calculation, word_indices setup) ...
        
        // Original line 915: Now works
        last_batch_cce_loss = train_sequential_batch(&nn, word_indices, &bp_time);
        
        // ... (rest of the training loop) ...
        
        // Accesses to nn.lr and calls to test_network, save_network_as_svg, and nn_free now work correctly.
    }
    
    // ... (final test, save SVG, and nn_free) ...

    return 0;
}
