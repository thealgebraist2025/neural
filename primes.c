#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <string.h>
#include <stdarg.h> // FIX 1: Added for va_list, va_start, va_end

// --- Prime Detection & Input Constants ---
#define BASE2_BITS 17               // Binary representation inputs
#define BASE3_BITS 11               // Base 3 representation inputs
#define REMAINDER_INPUTS 4          // Modulo inputs (3, 5, 7, 11)
#define NN_INPUT_SIZE (BASE2_BITS + BASE3_BITS + REMAINDER_INPUTS) // 32

#define NUM_EXAMPLES 10000          
#define MAX_VAL_NEEDED 104729       

#define MAX_TRAINING_SECONDS 240.0  // FIX 2: Increased from 120.0 to 240.0
#define BATCH_SIZE_HALF 512         
#define BATCH_SIZE (BATCH_SIZE_HALF * 2) 
#define NUM_BATCHES (NUM_EXAMPLES / BATCH_SIZE_HALF)

// --- NN Architecture Constants (Deep and Narrow) ---
#define NN_OUTPUT_SIZE 1
#define NN_HIDDEN_SIZE 32           
#define NN_LEARNING_RATE 0.0005     

// --- SVG Constants (Updated for 5 layers of 32 nodes) ---
#define SVG_WIDTH 1200
#define SVG_HEIGHT 500
#define INITIAL_SVG_CAPACITY 2500
#define SVG_FILENAME "network.svg" // NEW: Set to network.svg
#define NODE_RADIUS 6
#define NODE_SPACING 12
#define LAYER_SPACING 200

// --- Pre-computed Data Arrays ---
int primes_array[NUM_EXAMPLES];
int composites_array[NUM_EXAMPLES];

// --- Data Structures ---
typedef struct { int rows; int cols; double** data; } Matrix;

typedef struct {
    // Weights and Biases for 4 Hidden Layers
    Matrix weights_ih1;         // Input -> Hidden 1
    Matrix weights_h1h2;        // Hidden 1 -> Hidden 2
    Matrix weights_h2h3;        // Hidden 2 -> Hidden 3 
    Matrix weights_h3h4;        // Hidden 3 -> Hidden 4 
    Matrix weights_h4o;         // Hidden 4 -> Output

    double* bias_h1;
    double* bias_h2;            
    double* bias_h3;            
    double* bias_h4;            
    double* bias_o;

    double lr;

    // Intermediate results for 5-layer backpropagation (Input + 4 Hidden + Output)
    Matrix inputs;
    Matrix hidden1_outputs;
    Matrix hidden2_outputs;
    Matrix hidden3_outputs;     
    Matrix hidden4_outputs;     
    Matrix output_outputs;
} NeuralNetwork;

// --- SVG String Management Struct (Restored) ---
typedef struct { char* str; size_t length; bool is_valid; } SvgString;
SvgString* svg_strings = NULL;
size_t svg_count = 0;
size_t svg_capacity = 0;

// --- Utility Functions ---

bool* prime_cache = NULL;
void precompute_primes_sieve() {
    prime_cache = (bool*)malloc((MAX_VAL_NEEDED + 1) * sizeof(bool));
    if (!prime_cache) {
        fprintf(stderr, "FATAL ERROR: Could not allocate memory for prime cache.\n");
        exit(EXIT_FAILURE);
    }
    memset(prime_cache, true, (MAX_VAL_NEEDED + 1) * sizeof(bool));
    prime_cache[0] = false; prime_cache[1] = false;
    for (int p = 2; p * p <= MAX_VAL_NEEDED; p++) {
        if (prime_cache[p]) {
            for (int i = p * p; i <= MAX_VAL_NEEDED; i += p)
                prime_cache[i] = false;
        }
    }
}

double generate_precomputed_data() {
    clock_t start = clock();
    int prime_count = 0;
    int composite_count = 0;

    for (int n = 2; n <= MAX_VAL_NEEDED; n++) {
        if (prime_cache[n]) {
            if (prime_count < NUM_EXAMPLES) { primes_array[prime_count++] = n; }
        } else if (n > 3) { 
            if (composite_count < NUM_EXAMPLES) { composites_array[composite_count++] = n; }
        }
        if (prime_count >= NUM_EXAMPLES && composite_count >= NUM_EXAMPLES) { break; }
    }
    clock_t end = clock();
    return ((double)(end - start)) / CLOCKS_PER_SEC;
}

void shuffle_array(int arr[], int n) {
    if (n > 1) {
        for (int i = 0; i < n - 1; i++) {
            int j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = arr[j];
            arr[j] = arr[i];
            arr[i] = t;
        }
    }
}

bool is_prime(int n) {
    if (n < 2 || n > MAX_VAL_NEEDED || !prime_cache) return false;
    return prime_cache[n];
}

void generate_nn_input(int n, double* arr) {
    int current_idx = 0;

    // 1. Base 2 (Binary) Inputs
    for (int i = 0; i < BASE2_BITS; i++) {
        arr[current_idx++] = (double)((n >> i) & 1);
    }

    // 2. Base 3 Inputs (Digits scaled 0, 1/2, 1)
    int temp_n = n;
    for (int i = 0; i < BASE3_BITS; i++) {
        int digit = temp_n % 3;
        arr[current_idx++] = (double)digit / 2.0; 
        temp_n /= 3;
    }

    // 3. Small Prime Remainders (Scaled 0.0 to 1.0)
    arr[current_idx++] = (double)(n % 3) / 2.0;
    arr[current_idx++] = (double)(n % 5) / 4.0;
    arr[current_idx++] = (double)(n % 7) / 6.0;
    arr[current_idx++] = (double)(n % 11) / 10.0;
}


// --- Matrix & NN Core Utility Functions ---
Matrix matrix_create(int rows, int cols, int input_size) {
    Matrix m; m.rows = rows; m.cols = cols;
    m.data = (double**)calloc(rows, sizeof(double*));
    double scale = (input_size > 0 && rows > 0) ? sqrt(2.0 / (input_size + rows)) : 1.0;
    for (int i = 0; i < rows; i++) {
        m.data[i] = (double*)calloc(cols, sizeof(double));
        for (int j = 0; j < cols; j++) {
            m.data[i][j] = (((double)rand() / RAND_MAX) * 2.0 - 1.0) * scale;
        }
    }
    return m;
}
void matrix_free(Matrix m) {
    if (m.data == NULL) return;
    for (int i = 0; i < m.rows; i++) free(m.data[i]);
    free(m.data);
}
void matrix_copy_in(Matrix A, const Matrix B) {
    if (A.rows != B.rows || A.cols != B.cols) {
        fprintf(stderr, "FATAL ERROR: matrix_copy_in dimensions mismatch (%dx%d vs %dx%d).\n", 
                A.rows, A.cols, B.rows, B.cols);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            A.data[i][j] = B.data[i][j];
        }
    }
}
Matrix array_to_matrix(const double* arr, int size) {
    Matrix m = matrix_create(size, 1, 0);
    for (int i = 0; i < size; i++) { m.data[i][0] = arr[i]; }
    return m;
}
Matrix matrix_dot(Matrix A, Matrix B) {
    Matrix result = matrix_create(A.rows, B.cols, 0);
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < B.cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < A.cols; k++) { sum += A.data[i][k] * B.data[k][j]; }
            result.data[i][j] = sum;
        }
    }
    return result;
}
Matrix matrix_transpose(Matrix m) {
    Matrix result = matrix_create(m.cols, m.rows, 0);
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) { result.data[j][i] = m.data[i][j]; }
    }
    return result;
}

Matrix matrix_add_subtract(Matrix A, Matrix B, bool is_add) {
    Matrix result = matrix_create(A.rows, A.cols, 0);
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            if (is_add) { result.data[i][j] = A.data[i][j] + B.data[i][j]; }
            else { result.data[i][j] = A.data[i][j] - B.data[i][j]; }
        }
    }
    return result;
}

Matrix matrix_multiply_elem(Matrix A, Matrix B) {
    Matrix result = matrix_create(A.rows, A.cols, 0);
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) { result.data[i][j] = A.data[i][j] * B.data[i][j]; }
    }
    return result;
}
Matrix matrix_multiply_scalar(Matrix A, double scalar) {
    Matrix result = matrix_create(A.rows, A.cols, 0);
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) { result.data[i][j] = A.data[i][j] * scalar; }
    }
    return result;
}
Matrix matrix_map(Matrix m, double (*func)(double)) {
    Matrix result = matrix_create(m.rows, m.cols, 0);
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) { result.data[i][j] = func(m.data[i][j]); }
    }
    return result;
}
double tanh_activation(double x) { return tanh(x); }
double tanh_derivative(double y) { return 1.0 - (y * y); }
double sigmoid_activation(double x) { return 1.0 / (1.0 + exp(-x)); }
double sigmoid_derivative(double y) { return y * (1.0 - y); }

// --- Neural Network Functions (5-Layer Architecture: I->H1->H2->H3->H4->O) ---

void nn_init(NeuralNetwork* nn) {
    nn->lr = NN_LEARNING_RATE;
    int h_size = NN_HIDDEN_SIZE;
    
    // Weights
    nn->weights_ih1 = matrix_create(h_size, NN_INPUT_SIZE, NN_INPUT_SIZE);
    nn->weights_h1h2 = matrix_create(h_size, h_size, h_size);
    nn->weights_h2h3 = matrix_create(h_size, h_size, h_size); 
    nn->weights_h3h4 = matrix_create(h_size, h_size, h_size); 
    nn->weights_h4o = matrix_create(NN_OUTPUT_SIZE, h_size, h_size);
    
    // Biases
    nn->bias_h1 = (double*)calloc(h_size, sizeof(double));
    nn->bias_h2 = (double*)calloc(h_size, sizeof(double));
    nn->bias_h3 = (double*)calloc(h_size, sizeof(double)); 
    nn->bias_h4 = (double*)calloc(h_size, sizeof(double)); 
    nn->bias_o = (double*)calloc(NN_OUTPUT_SIZE, sizeof(double));

    // Intermediate Outputs
    nn->inputs = matrix_create(NN_INPUT_SIZE, 1, 0);
    nn->hidden1_outputs = matrix_create(h_size, 1, 0);
    nn->hidden2_outputs = matrix_create(h_size, 1, 0);
    nn->hidden3_outputs = matrix_create(h_size, 1, 0); 
    nn->hidden4_outputs = matrix_create(h_size, 1, 0); 
    nn->output_outputs = matrix_create(NN_OUTPUT_SIZE, 1, 0);
}

void nn_free(NeuralNetwork* nn) {
    matrix_free(nn->weights_ih1); matrix_free(nn->weights_h1h2); 
    matrix_free(nn->weights_h2h3); matrix_free(nn->weights_h3h4); 
    matrix_free(nn->weights_h4o);

    free(nn->bias_h1); free(nn->bias_h2); free(nn->bias_h3); free(nn->bias_h4); free(nn->bias_o); 
    
    matrix_free(nn->inputs); matrix_free(nn->hidden1_outputs);
    matrix_free(nn->hidden2_outputs); matrix_free(nn->hidden3_outputs); 
    matrix_free(nn->hidden4_outputs); 
    matrix_free(nn->output_outputs);
}

void nn_forward(NeuralNetwork* nn, const double* input_array, double* output_array) {
    Matrix inputs_m = array_to_matrix(input_array, NN_INPUT_SIZE);
    int h = NN_HIDDEN_SIZE;
    
    // 1. Input -> H1
    Matrix h1_in_m = matrix_dot(nn->weights_ih1, inputs_m);
    for (int i = 0; i < h; i++) h1_in_m.data[i][0] += nn->bias_h1[i];
    Matrix h1_out_m = matrix_map(h1_in_m, tanh_activation);

    // 2. H1 -> H2
    Matrix h2_in_m = matrix_dot(nn->weights_h1h2, h1_out_m);
    for (int i = 0; i < h; i++) h2_in_m.data[i][0] += nn->bias_h2[i];
    Matrix h2_out_m = matrix_map(h2_in_m, tanh_activation);

    // 3. H2 -> H3
    Matrix h3_in_m = matrix_dot(nn->weights_h2h3, h2_out_m);
    for (int i = 0; i < h; i++) h3_in_m.data[i][0] += nn->bias_h3[i];
    Matrix h3_out_m = matrix_map(h3_in_m, tanh_activation);
    
    // 4. H3 -> H4
    Matrix h4_in_m = matrix_dot(nn->weights_h3h4, h3_out_m);
    for (int i = 0; i < h; i++) h4_in_m.data[i][0] += nn->bias_h4[i];
    Matrix h4_out_m = matrix_map(h4_in_m, tanh_activation);

    // 5. H4 -> Output
    Matrix output_in_m = matrix_dot(nn->weights_h4o, h4_out_m);
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) output_in_m.data[i][0] += nn->bias_o[i];
    Matrix output_out_m = matrix_map(output_in_m, sigmoid_activation);
    
    // Store intermediates for backprop
    matrix_copy_in(nn->inputs, inputs_m);
    matrix_copy_in(nn->hidden1_outputs, h1_out_m);
    matrix_copy_in(nn->hidden2_outputs, h2_out_m);
    matrix_copy_in(nn->hidden3_outputs, h3_out_m); 
    matrix_copy_in(nn->hidden4_outputs, h4_out_m); 
    matrix_copy_in(nn->output_outputs, output_out_m);

    // Copy result to output array
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) { output_array[i] = output_out_m.data[i][0]; }

    // Cleanup (only local temporaries)
    matrix_free(inputs_m); matrix_free(h1_in_m); matrix_free(h1_out_m); 
    matrix_free(h2_in_m); matrix_free(h2_out_m); 
    matrix_free(h3_in_m); matrix_free(h3_out_m); 
    matrix_free(h4_in_m); matrix_free(h4_out_m); 
    matrix_free(output_in_m); matrix_free(output_out_m);
}

double nn_backward(NeuralNetwork* nn, const double* target_array) {
    double mse_loss = 0.0;
    Matrix targets_m = array_to_matrix(target_array, NN_OUTPUT_SIZE);
    int h = NN_HIDDEN_SIZE;
    
    // --- 1. Output Layer ---
    Matrix output_errors_m = matrix_add_subtract(nn->output_outputs, targets_m, false);
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) { mse_loss += output_errors_m.data[i][0] * output_errors_m.data[i][0]; }
    mse_loss /= NN_OUTPUT_SIZE;
    Matrix output_d_m = matrix_map(nn->output_outputs, sigmoid_derivative);
    Matrix output_gradients_m = matrix_multiply_elem(output_errors_m, output_d_m);
    
    // Update Weights H4O and Bias O
    Matrix h4_out_t_m = matrix_transpose(nn->hidden4_outputs);
    Matrix delta_h4o_m = matrix_dot(output_gradients_m, h4_out_t_m);
    Matrix new_h4o_m = matrix_add_subtract(nn->weights_h4o, matrix_multiply_scalar(delta_h4o_m, nn->lr), false);
    matrix_copy_in(nn->weights_h4o, new_h4o_m);
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) { nn->bias_o[i] -= output_gradients_m.data[i][0] * nn->lr; }

    // --- 2. Hidden Layer 4 ---
    Matrix weights_h4o_t_m = matrix_transpose(nn->weights_h4o);
    Matrix h4_errors_m = matrix_dot(weights_h4o_t_m, output_gradients_m);
    Matrix h4_d_m = matrix_map(nn->hidden4_outputs, tanh_derivative);
    Matrix h4_gradients_m = matrix_multiply_elem(h4_errors_m, h4_d_m);
    
    // Update Weights H3H4 and Bias H4
    Matrix h3_out_t_m = matrix_transpose(nn->hidden3_outputs);
    Matrix delta_h3h4_m = matrix_dot(h4_gradients_m, h3_out_t_m);
    Matrix new_h3h4_m = matrix_add_subtract(nn->weights_h3h4, matrix_multiply_scalar(delta_h3h4_m, nn->lr), false);
    matrix_copy_in(nn->weights_h3h4, new_h3h4_m);
    for (int i = 0; i < h; i++) { nn->bias_h4[i] -= h4_gradients_m.data[i][0] * nn->lr; }

    // --- 3. Hidden Layer 3 ---
    Matrix weights_h3h4_t_m = matrix_transpose(nn->weights_h3h4);
    Matrix h3_errors_m = matrix_dot(weights_h3h4_t_m, h4_gradients_m);
    Matrix h3_d_m = matrix_map(nn->hidden3_outputs, tanh_derivative);
    Matrix h3_gradients_m = matrix_multiply_elem(h3_errors_m, h3_d_m);
    
    // Update Weights H2H3 and Bias H3
    Matrix h2_out_t_m = matrix_transpose(nn->hidden2_outputs);
    Matrix delta_h2h3_m = matrix_dot(h3_gradients_m, h2_out_t_m);
    Matrix new_h2h3_m = matrix_add_subtract(nn->weights_h2h3, matrix_multiply_scalar(delta_h2h3_m, nn->lr), false);
    matrix_copy_in(nn->weights_h2h3, new_h2h3_m);
    for (int i = 0; i < h; i++) { nn->bias_h3[i] -= h3_gradients_m.data[i][0] * nn->lr; }

    // --- 4. Hidden Layer 2 ---
    Matrix weights_h2h3_t_m = matrix_transpose(nn->weights_h2h3);
    Matrix h2_errors_m = matrix_dot(weights_h2h3_t_m, h3_gradients_m);
    Matrix h2_d_m = matrix_map(nn->hidden2_outputs, tanh_derivative);
    Matrix h2_gradients_m = matrix_multiply_elem(h2_errors_m, h2_d_m);
    
    // Update Weights H1H2 and Bias H2
    Matrix h1_out_t_m = matrix_transpose(nn->hidden1_outputs);
    Matrix delta_h1h2_m = matrix_dot(h2_gradients_m, h1_out_t_m);
    Matrix new_h1h2_m = matrix_add_subtract(nn->weights_h1h2, matrix_multiply_scalar(delta_h1h2_m, nn->lr), false);
    matrix_copy_in(nn->weights_h1h2, new_h1h2_m);
    for (int i = 0; i < h; i++) { nn->bias_h2[i] -= h2_gradients_m.data[i][0] * nn->lr; }
    
    // --- 5. Hidden Layer 1 ---
    Matrix weights_h1h2_t_m = matrix_transpose(nn->weights_h1h2);
    Matrix h1_errors_m = matrix_dot(weights_h1h2_t_m, h2_gradients_m);
    Matrix h1_d_m = matrix_map(nn->hidden1_outputs, tanh_derivative);
    Matrix h1_gradients_m = matrix_multiply_elem(h1_errors_m, h1_d_m);
    
    // Update Weights IH1 and Bias H1
    Matrix inputs_t_m = matrix_transpose(nn->inputs);
    Matrix delta_ih1_m = matrix_dot(h1_gradients_m, inputs_t_m);
    Matrix new_ih1_m = matrix_add_subtract(nn->weights_ih1, matrix_multiply_scalar(delta_ih1_m, nn->lr), false);
    matrix_copy_in(nn->weights_ih1, new_ih1_m);
    for (int i = 0; i < h; i++) { nn->bias_h1[i] -= h1_gradients_m.data[i][0] * nn->lr; }

    // Final Cleanup (Temporary matrices must be freed in a complete implementation)
    matrix_free(targets_m); 
    matrix_free(output_errors_m); matrix_free(output_d_m); matrix_free(output_gradients_m);
    matrix_free(h4_out_t_m); matrix_free(delta_h4o_m); matrix_free(new_h4o_m); 
    matrix_free(weights_h4o_t_m); matrix_free(h4_errors_m); matrix_free(h4_d_m); matrix_free(h4_gradients_m);
    matrix_free(h3_out_t_m); matrix_free(delta_h3h4_m); matrix_free(new_h3h4_m); 
    matrix_free(weights_h3h4_t_m); matrix_free(h3_errors_m); matrix_free(h3_d_m); matrix_free(h3_gradients_m);
    matrix_free(h2_out_t_m); matrix_free(delta_h2h3_m); matrix_free(new_h2h3_m); 
    matrix_free(weights_h2h3_t_m); matrix_free(h2_errors_m); matrix_free(h2_d_m); matrix_free(h2_gradients_m);
    matrix_free(h1_out_t_m); matrix_free(delta_h1h2_m); matrix_free(new_h1h2_m); 
    matrix_free(weights_h1h2_t_m); matrix_free(h1_errors_m); matrix_free(h1_d_m); matrix_free(h1_gradients_m);
    matrix_free(inputs_t_m); matrix_free(delta_ih1_m); matrix_free(new_ih1_m);
    
    return mse_loss;
}

// Sequential Batch Training Function
double train_sequential_batch(NeuralNetwork* nn, int batch_index, double* bp_time) {
    clock_t start_bp = clock();
    double total_mse = 0.0;
    double input_arr[NN_INPUT_SIZE]; 
    double target_arr_prime[] = {1.0};
    double target_arr_composite[] = {0.0};
    double output_arr[NN_OUTPUT_SIZE];
    
    int start_index = batch_index * BATCH_SIZE_HALF;
    int end_index = start_index + BATCH_SIZE_HALF;
    
    if (end_index > NUM_EXAMPLES) {
        end_index = NUM_EXAMPLES;
    }
    int current_batch_size_half = end_index - start_index;

    // --- 1. Train on Primes ---
    for (int i = start_index; i < end_index; i++) {
        generate_nn_input(primes_array[i], input_arr); 
        nn_forward(nn, input_arr, output_arr);
        total_mse += nn_backward(nn, target_arr_prime);
    }

    // --- 2. Train on Composites ---
    for (int i = start_index; i < end_index; i++) {
        generate_nn_input(composites_array[i], input_arr); 
        nn_forward(nn, input_arr, output_arr);
        total_mse += nn_backward(nn, target_arr_composite);
    }
    
    clock_t end_bp = clock();
    *bp_time = ((double)(end_bp - start_bp)) / CLOCKS_PER_SEC; 

    return total_mse / (current_batch_size_half * 2);
}

// Testing function
double test_network(NeuralNetwork* nn, double* test_time) {
    clock_t start_test = clock();
    int correct_predictions = 0;
    int total_tests = 0;
    double input_arr[NN_INPUT_SIZE]; 
    double output_arr[NN_OUTPUT_SIZE];

    for (int n = 1; n <= MAX_VAL_NEEDED; n++) {
        generate_nn_input(n, input_arr); 
        nn_forward(nn, input_arr, output_arr);
        double target = is_prime(n) ? 1.0 : 0.0;
        int classified_as_prime = (output_arr[0] > 0.5);
        int actual_is_prime = (target > 0.5);

        if (classified_as_prime == actual_is_prime) { correct_predictions++; }
        total_tests++;
    }
    
    clock_t end_test = clock();
    *test_time = ((double)(end_test - start_test)) / CLOCKS_PER_SEC;
    return ((double)correct_predictions / total_tests) * 100.0;
}


// --- SVG Utility Functions (Restored) ---

void svg_init_storage() {
    svg_capacity = INITIAL_SVG_CAPACITY;
    svg_strings = (SvgString*)malloc(svg_capacity * sizeof(SvgString));
    if (!svg_strings) {
        fprintf(stderr, "FATAL ERROR: Could not allocate memory for SVG storage.\n");
        exit(EXIT_FAILURE);
    }
    svg_count = 0;
}

void svg_add_string(const char* format, ...) {
    if (svg_count >= svg_capacity) {
        svg_capacity *= 2;
        svg_strings = (SvgString*)realloc(svg_strings, svg_capacity * sizeof(SvgString));
        if (!svg_strings) {
            fprintf(stderr, "FATAL ERROR: Could not reallocate memory for SVG storage.\n");
            exit(EXIT_FAILURE);
        }
    }

    va_list args;
    va_start(args, format);
    
    // Determine the required buffer size
    int size = vsnprintf(NULL, 0, format, args);
    va_end(args);

    // Allocate memory for the string
    svg_strings[svg_count].str = (char*)malloc(size + 1);
    svg_strings[svg_count].length = size;
    svg_strings[svg_count].is_valid = true;
    
    if (!svg_strings[svg_count].str) {
        fprintf(stderr, "FATAL ERROR: Could not allocate memory for SVG string.\n");
        exit(EXIT_FAILURE);
    }

    va_start(args, format);
    vsnprintf(svg_strings[svg_count].str, size + 1, format, args);
    va_end(args);

    svg_count++;
}

void svg_free_storage() {
    for (size_t i = 0; i < svg_count; i++) {
        free(svg_strings[i].str);
    }
    free(svg_strings);
    svg_strings = NULL;
    svg_count = 0;
    svg_capacity = 0;
}

void save_network_as_svg(NeuralNetwork* nn) {
    svg_init_storage();
    
    svg_add_string("<svg width=\"%d\" height=\"%d\" xmlns=\"http://www.w3.org/2000/svg\">\n", SVG_WIDTH, SVG_HEIGHT);
    svg_add_string("<style>\n");
    svg_add_string("  .neuron { stroke: black; stroke-width: 1; }\n");
    svg_add_string("  .positive { stroke: #0088ff; stroke-width: 0.5; }\n"); // Blue
    svg_add_string("  .negative { stroke: #ff0044; stroke-width: 0.5; }\n"); // Red
    svg_add_string("  .bias { fill: #aaaaaa; }\n"); // Grey for biases
    svg_add_string("</style>\n");

    // --- Helper for Layer Coordinates ---
    int layer_sizes[] = {NN_INPUT_SIZE, NN_HIDDEN_SIZE, NN_HIDDEN_SIZE, NN_HIDDEN_SIZE, NN_HIDDEN_SIZE, NN_OUTPUT_SIZE};
    int num_layers = sizeof(layer_sizes) / sizeof(layer_sizes[0]);
    int x_coords[num_layers];
    
    for (int i = 0; i < num_layers; i++) {
        x_coords[i] = 50 + i * LAYER_SPACING;
    }

    // --- Draw Connections (Edges) ---
    for (int i = 0; i < num_layers - 1; i++) {
        Matrix* weights = NULL;
        if (i == 0) weights = &nn->weights_ih1;
        else if (i == 1) weights = &nn->weights_h1h2;
        else if (i == 2) weights = &nn->weights_h2h3;
        else if (i == 3) weights = &nn->weights_h3h4;
        else if (i == 4) weights = &nn->weights_h4o;
        
        if (!weights) continue;
        
        int prev_size = layer_sizes[i];
        int curr_size = layer_sizes[i+1];
        
        // Calculate the starting Y coordinate to center the layer
        int y_offset_prev = SVG_HEIGHT / 2 - (prev_size * (NODE_RADIUS + NODE_SPACING)) / 2;
        int y_offset_curr = SVG_HEIGHT / 2 - (curr_size * (NODE_RADIUS + NODE_SPACING)) / 2;

        for (int j = 0; j < curr_size; j++) { // Current Layer (rows in matrix)
            for (int k = 0; k < prev_size; k++) { // Previous Layer (cols in matrix)
                double weight = weights->data[j][k];
                double abs_weight = fabs(weight);
                const char* css_class = (weight >= 0) ? "positive" : "negative";
                
                // Scale line width based on absolute weight (e.g., max 2.5)
                double stroke_width = 0.2 + abs_weight * 2.0;
                
                int x1 = x_coords[i] + NODE_RADIUS;
                int y1 = y_offset_prev + k * (NODE_RADIUS * 2 + NODE_SPACING) + NODE_RADIUS;
                int x2 = x_coords[i+1] - NODE_RADIUS;
                int y2 = y_offset_curr + j * (NODE_RADIUS * 2 + NODE_SPACING) + NODE_RADIUS;
                
                svg_add_string("<line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" class=\"%s\" style=\"stroke-width:%.2f\" />\n", 
                               x1, y1, x2, y2, css_class, stroke_width);
            }
        }
    }

    // --- Draw Nodes (Neurons) ---
    for (int i = 0; i < num_layers; i++) {
        int size = layer_sizes[i];
        // Calculate the starting Y coordinate to center the layer
        int y_offset = SVG_HEIGHT / 2 - (size * (NODE_RADIUS + NODE_SPACING)) / 2;
        
        // FIX 3: Declare cx here to resolve the undeclared error outside the inner loop
        int cx = x_coords[i]; 

        for (int j = 0; j < size; j++) {
            // int cx = x_coords[i]; // Declared outside the inner loop now
            int cy = y_offset + j * (NODE_RADIUS * 2 + NODE_SPACING) + NODE_RADIUS;
            
            const char* fill_color = "#ffffff"; // Default white
            if (i == 0) fill_color = "#cccccc"; // Input: Grey
            
            // Draw circle
            svg_add_string("<circle cx=\"%d\" cy=\"%d\" r=\"%d\" fill=\"%s\" class=\"neuron\" />\n", 
                           cx, cy, NODE_RADIUS, fill_color);

            // Add Bias Labels (for hidden and output layers)
            if (i > 0) {
                double bias = 0.0;
                if (i == 1) bias = nn->bias_h1[j];
                else if (i == 2) bias = nn->bias_h2[j];
                else if (i == 3) bias = nn->bias_h3[j];
                else if (i == 4) bias = nn->bias_h4[j];
                else if (i == 5) bias = nn->bias_o[j];
                
                // Add a small rectangle or text for the bias
                svg_add_string("<text x=\"%d\" y=\"%d\" font-size=\"8\" fill=\"#666666\" class=\"bias\">%.2f</text>\n",
                               cx + NODE_RADIUS + 2, cy + 3, bias);
            }
        }
        
        // Add Layer Labels
        const char* label = "";
        if (i == 0) label = "INPUT (32)";
        else if (i == 1) label = "HIDDEN 1 (32)";
        else if (i == 2) label = "HIDDEN 2 (32)";
        else if (i == 3) label = "HIDDEN 3 (32)";
        else if (i == 4) label = "HIDDEN 4 (32)";
        else if (i == 5) label = "OUTPUT (1)";
        
        svg_add_string("<text x=\"%d\" y=\"%d\" font-size=\"12\" font-weight=\"bold\" text-anchor=\"middle\">%s</text>\n",
                       cx, SVG_HEIGHT - 10, label);
    }

    svg_add_string("</svg>\n");

    // --- Save to File ---
    FILE* fp = fopen(SVG_FILENAME, "w");
    if (fp == NULL) {
        fprintf(stderr, "FATAL ERROR: Could not open file %s for writing.\n", SVG_FILENAME);
        svg_free_storage();
        return;
    }

    for (size_t i = 0; i < svg_count; i++) {
        if (svg_strings[i].is_valid) {
            fwrite(svg_strings[i].str, 1, svg_strings[i].length, fp);
        }
    }

    fclose(fp);
    svg_free_storage();
}

// --- Main Execution ---

int main() {
    srand((unsigned int)time(NULL));
    
    printf("Setting up Sieve and Pre-computing 10,000 Primes/Composites...\n");
    precompute_primes_sieve();
    double precomp_time = generate_precomputed_data();
    printf("Pre-computation Complete. Time taken: %.4f seconds.\n", precomp_time);

    // --- STEP 2: Initialize NN with NEW Architecture and Inputs ---
    NeuralNetwork nn;
    nn_init(&nn);

    printf("\nNeural Network Prime Detector Initialized with Deep and Narrow Architecture.\n");
    printf("Input Size: %d (17 Base 2 + 11 Base 3 + 4 Modulo)\n", NN_INPUT_SIZE);
    printf("Architecture: Input(32) -> H1(32) -> H2(32) -> H3(32) -> H4(32) -> Output(1)\n");
    printf("Learning Rate: %.4f\n", NN_LEARNING_RATE);
    printf("Batch Size: %d\n", BATCH_SIZE);
    printf("Maximum Training Time: %.0f seconds.\n", MAX_TRAINING_SECONDS); // Confirmation
    printf("------------------------------------------------------------------------------------------------\n");
    printf("Batch No. | Primes Range | Composites Range | Avg MSE (Batch) | Correctness (Total) | Backprop Time (sec)\n");
    printf("------------------------------------------------------------------------------------------------\n");
    fflush(stdout);

    time_t start_time = time(NULL);
    double current_success_rate = 0.0;
    int next_milestone_percent = 10;
    int total_batches_run = 0;
    int max_epochs = 1000; 

    for (int epoch = 0; epoch < max_epochs; epoch++) {
        bool time_limit_reached = false;
        
        shuffle_array(primes_array, NUM_EXAMPLES);
        shuffle_array(composites_array, NUM_EXAMPLES);
        printf("\n--- EPOCH %d STARTED (Data Shuffled). ---\n", epoch + 1);

        for (int i = 0; i <= NUM_BATCHES; i++) { 
            
            int start_index = i * BATCH_SIZE_HALF;
            if (start_index >= NUM_EXAMPLES) break; 

            if (difftime(time(NULL), start_time) >= MAX_TRAINING_SECONDS) {
                time_limit_reached = true;
                break;
            }
            
            double bp_time = 0.0;
            double avg_mse = train_sequential_batch(&nn, i, &bp_time);
            total_batches_run++;
            
            double test_time = 0.0;
            current_success_rate = test_network(&nn, &test_time); 
            
            int prime_end = start_index + BATCH_SIZE_HALF;
            if (prime_end > NUM_EXAMPLES) prime_end = NUM_EXAMPLES;

            // Log stats after every batch
            printf("%-9d | %4d-%-4d | %4d-%-4d | %-15.8f | %-25.2f | %-19.4f\n", 
                   total_batches_run, 
                   start_index + 1, prime_end, 
                   start_index + 1, prime_end, 
                   avg_mse, current_success_rate, bp_time);
            fflush(stdout);
            
            // Log milestones
            while (current_success_rate >= next_milestone_percent) {
                printf("--- MILESTONE REACHED --- Batch: %d | Time: %.0f sec | Correctness: %.2f%%\n",
                       total_batches_run, difftime(time(NULL), start_time), current_success_rate);
                fflush(stdout);
                
                if (next_milestone_percent == 90) { next_milestone_percent = 95; } 
                else if (next_milestone_percent >= 95) { break; } 
                else { next_milestone_percent += 10; }
            }
            
            if (current_success_rate >= 95.0) {
                 printf("--- GOAL ACHIEVED --- Correctness %.2f%% reached.\n", current_success_rate);
                 time_limit_reached = true; 
                 break; 
            }
        }
        
        if (time_limit_reached) break;
    }
    
    printf("------------------------------------------------------------------------------------------------\n");
    printf("\n#####################################################\n");
    printf("## TRAINING TERMINATED ##\n");
    printf("Final correctness: %.2f%%\n", current_success_rate);
    printf("Total Batches Run: %d\n", total_batches_run);
    printf("Total Training Time: %.0f seconds.\n", difftime(time(NULL), start_time));
    printf("#####################################################\n");
    fflush(stdout);

    // --- POST-TRAINING SVG SAVE (Restored) ---
    save_network_as_svg(&nn);
    printf("Final network SVG saved to %s.\n", SVG_FILENAME);

    // --- Cleanup ---
    if (prime_cache != NULL) {
        free(prime_cache);
    }
    nn_free(&nn);

    return 0;
}
