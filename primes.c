#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <string.h>

// --- Prime Detection Constants ---
#define BIT_DEPTH 17                // Input size: Binary representation of the number
#define NUM_EXAMPLES 10000          // We need the first 10,000 of each type
#define MAX_PRIME_TO_TEST 104729    // The 10,000th prime
#define TEST_RANGE (MAX_PRIME_TO_TEST + 1)
#define MAX_VAL_NEEDED 104729       // Max value we need to classify/store.

#define MAX_TRAINING_SECONDS 120.0  // 2 minutes limit
#define BATCH_SIZE_HALF 512         // Batch size: 512 Primes + 512 Non-Primes = 1024 total
#define BATCH_SIZE (BATCH_SIZE_HALF * 2) 
#define NUM_BATCHES (NUM_EXAMPLES / BATCH_SIZE_HALF) // 10000 / 512 = 19 full batches + 1 partial batch

// --- NN Architecture Constants ---
#define NN_INPUT_SIZE BIT_DEPTH
#define NN_OUTPUT_SIZE 1
#define NN_HIDDEN_SIZE 512
#define NN_LEARNING_RATE 0.0005     // FIXED: Reduced for stability

// --- SVG Constants ---
#define SVG_WIDTH 1200
#define SVG_HEIGHT 1600
#define INITIAL_SVG_CAPACITY 2500
#define SVG_FILENAME "network.svg"

// --- Pre-computed Data Arrays ---
int primes_array[NUM_EXAMPLES];
int composites_array[NUM_EXAMPLES];

// --- Data Structures ---
typedef struct { int rows; int cols; double** data; } Matrix;

typedef struct {
    Matrix weights_ih;
    Matrix weights_ho;
    double* bias_h;
    double* bias_o;
    double lr;
    // Intermediate results stored for backpropagation
    Matrix inputs;
    Matrix hidden_inputs;
    Matrix hidden_outputs;
    Matrix output_inputs;
    Matrix output_outputs;
} NeuralNetwork;

// --- SVG String Management Struct ---
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
            if (prime_count < NUM_EXAMPLES) {
                primes_array[prime_count++] = n;
            }
        } else if (n > 3) { // Composites start at 4
            if (composite_count < NUM_EXAMPLES) {
                composites_array[composite_count++] = n;
            }
        }
        if (prime_count >= NUM_EXAMPLES && composite_count >= NUM_EXAMPLES) {
            break;
        }
    }
    clock_t end = clock();
    return ((double)(end - start)) / CLOCKS_PER_SEC;
}

// Shuffling function for epoch-based training stability
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

void int_to_binary_array(int n, double* arr) {
    for (int i = 0; i < BIT_DEPTH; i++) {
        arr[i] = (double)((n >> i) & 1);
    }
}

void validate_precomputed_data() {
    int prime_errors = 0;
    int composite_errors = 0;
    
    printf("\n--- Data Validation (First 512 entries of each) ---\n");
    
    for (int i = 0; i < BATCH_SIZE_HALF; i++) {
        if (!is_prime(primes_array[i])) { prime_errors++; }
    }
    for (int i = 0; i < BATCH_SIZE_HALF; i++) {
        if (is_prime(composites_array[i])) { composite_errors++; }
    }
    
    printf("Validation Complete. Primes Errors: %d, Composite Errors: %d.\n", prime_errors, composite_errors);
    if (prime_errors > 0 || composite_errors > 0) {
        fprintf(stderr, "FATAL: Validation failed. Stopping execution.\n");
        exit(EXIT_FAILURE);
    }
    printf("-------------------------------------------\n");
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

// FIX: Definition for matrix_copy_in to resolve the 'undefined reference' error
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
// END FIX

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

// --- Neural Network Functions ---

void nn_init(NeuralNetwork* nn) {
    nn->lr = NN_LEARNING_RATE;
    nn->weights_ih = matrix_create(NN_HIDDEN_SIZE, NN_INPUT_SIZE, NN_INPUT_SIZE);
    nn->weights_ho = matrix_create(NN_OUTPUT_SIZE, NN_HIDDEN_SIZE, NN_HIDDEN_SIZE);
    nn->bias_h = (double*)calloc(NN_HIDDEN_SIZE, sizeof(double));
    nn->bias_o = (double*)calloc(NN_OUTPUT_SIZE, sizeof(double));
    // Pre-allocate space for intermediate matrices
    nn->inputs = matrix_create(NN_INPUT_SIZE, 1, 0);
    nn->hidden_inputs = matrix_create(NN_HIDDEN_SIZE, 1, 0);
    nn->hidden_outputs = matrix_create(NN_HIDDEN_SIZE, 1, 0);
    nn->output_inputs = matrix_create(NN_OUTPUT_SIZE, 1, 0);
    nn->output_outputs = matrix_create(NN_OUTPUT_SIZE, 1, 0);
}
void nn_free(NeuralNetwork* nn) {
    matrix_free(nn->weights_ih); matrix_free(nn->weights_ho);
    free(nn->bias_h); free(nn->bias_o);
    matrix_free(nn->inputs); matrix_free(nn->hidden_inputs);
    matrix_free(nn->hidden_outputs); matrix_free(nn->output_inputs);
    matrix_free(nn->output_outputs);
}

void nn_forward(NeuralNetwork* nn, const double* input_array, double* output_array) {
    Matrix inputs_m = array_to_matrix(input_array, NN_INPUT_SIZE);
    
    // Input -> Hidden
    Matrix hidden_in_m = matrix_dot(nn->weights_ih, inputs_m);
    for (int i = 0; i < NN_HIDDEN_SIZE; i++) hidden_in_m.data[i][0] += nn->bias_h[i];
    Matrix hidden_out_m = matrix_map(hidden_in_m, tanh_activation);

    // Hidden -> Output
    Matrix output_in_m = matrix_dot(nn->weights_ho, hidden_out_m);
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) output_in_m.data[i][0] += nn->bias_o[i];
    Matrix output_out_m = matrix_map(output_in_m, sigmoid_activation);
    
    // Store intermediates for backprop
    matrix_copy_in(nn->inputs, inputs_m);
    matrix_copy_in(nn->hidden_outputs, hidden_out_m);
    matrix_copy_in(nn->output_outputs, output_out_m);

    // Copy result to output array
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) { output_array[i] = output_out_m.data[i][0]; }

    matrix_free(inputs_m); matrix_free(hidden_in_m); matrix_free(hidden_out_m); 
    matrix_free(output_in_m); matrix_free(output_out_m);
}

double nn_backward(NeuralNetwork* nn, const double* target_array) {
    double mse_loss = 0.0;
    Matrix targets_m = array_to_matrix(target_array, NN_OUTPUT_SIZE);
    
    // Calculate Output Error and Loss
    Matrix output_errors_m = matrix_add_subtract(nn->output_outputs, targets_m, false);
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) { 
        double error = output_errors_m.data[i][0]; mse_loss += error * error; 
    }
    mse_loss /= NN_OUTPUT_SIZE;
    
    // Output Gradients and Delta WHO
    Matrix output_d_m = matrix_map(nn->output_outputs, sigmoid_derivative);
    Matrix output_gradients_m = matrix_multiply_elem(output_errors_m, output_d_m);
    Matrix hidden_out_t_m = matrix_transpose(nn->hidden_outputs);
    Matrix delta_who_m = matrix_dot(output_gradients_m, hidden_out_t_m);
    
    // Update Weights HO and Bias O
    Matrix scaled_delta_who_m = matrix_multiply_scalar(delta_who_m, nn->lr);
    Matrix new_who_m = matrix_add_subtract(nn->weights_ho, scaled_delta_who_m, false);
    matrix_copy_in(nn->weights_ho, new_who_m);
    Matrix scaled_output_grad_m = matrix_multiply_scalar(output_gradients_m, nn->lr);
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) { nn->bias_o[i] -= scaled_output_grad_m.data[i][0]; }

    // Calculate Hidden Errors and Hidden Gradients
    Matrix weights_ho_t_m = matrix_transpose(nn->weights_ho);
    Matrix hidden_errors_m = matrix_dot(weights_ho_t_m, output_gradients_m);
    Matrix hidden_d_m = matrix_map(nn->hidden_outputs, tanh_derivative);
    Matrix hidden_gradients_m = matrix_multiply_elem(hidden_errors_m, hidden_d_m);
    
    // Update Weights IH and Bias H
    Matrix inputs_t_m = matrix_transpose(nn->inputs);
    Matrix delta_wih_m = matrix_dot(hidden_gradients_m, inputs_t_m);
    Matrix scaled_delta_wih_m = matrix_multiply_scalar(delta_wih_m, nn->lr);
    Matrix new_wih_m = matrix_add_subtract(nn->weights_ih, scaled_delta_wih_m, false);
    matrix_copy_in(nn->weights_ih, new_wih_m);
    Matrix scaled_hidden_grad_m = matrix_multiply_scalar(hidden_gradients_m, nn->lr);
    for (int i = 0; i < NN_HIDDEN_SIZE; i++) { nn->bias_h[i] -= scaled_hidden_grad_m.data[i][0]; }

    // Cleanup
    matrix_free(targets_m); matrix_free(output_errors_m); matrix_free(output_d_m); matrix_free(output_gradients_m);
    matrix_free(hidden_out_t_m); matrix_free(delta_who_m); matrix_free(scaled_delta_who_m); matrix_free(new_who_m);
    matrix_free(scaled_output_grad_m); matrix_free(weights_ho_t_m); matrix_free(hidden_errors_m);
    matrix_free(hidden_d_m); matrix_free(hidden_gradients_m); matrix_free(inputs_t_m); matrix_free(delta_wih_m);
    matrix_free(scaled_delta_wih_m); matrix_free(new_wih_m); matrix_free(scaled_hidden_grad_m);
    
    return mse_loss;
}

// Sequential Batch Training Function
double train_sequential_batch(NeuralNetwork* nn, int batch_index, double* bp_time) {
    clock_t start_bp = clock();
    double total_mse = 0.0;
    double input_arr[BIT_DEPTH];
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
        int_to_binary_array(primes_array[i], input_arr);
        nn_forward(nn, input_arr, output_arr);
        total_mse += nn_backward(nn, target_arr_prime);
    }

    // --- 2. Train on Composites ---
    for (int i = start_index; i < end_index; i++) {
        int_to_binary_array(composites_array[i], input_arr);
        nn_forward(nn, input_arr, output_arr);
        total_mse += nn_backward(nn, target_arr_composite);
    }
    
    clock_t end_bp = clock();
    *bp_time = ((double)(end - start_bp)) / CLOCKS_PER_SEC;

    return total_mse / (current_batch_size_half * 2);
}

// Testing function: Measures accuracy on all numbers up to the MAX_VAL_NEEDED
double test_network(NeuralNetwork* nn, double* test_time) {
    clock_t start_test = clock();
    int correct_predictions = 0;
    int total_tests = 0;
    double input_arr[BIT_DEPTH];
    double output_arr[NN_OUTPUT_SIZE];

    for (int n = 1; n <= MAX_VAL_NEEDED; n++) {
        int_to_binary_array(n, input_arr);
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


// --- SVG Utility Functions (for visualization) ---

bool validate_svg_string(const SvgString* s) {
    return s != NULL && s->is_valid && s->str != NULL && s->length == strlen(s->str);
}
void append_svg_string(const char* new_str) {
    if (new_str == NULL) return;
    size_t len = strlen(new_str);
    if (len == 0) return;
    if (svg_count >= svg_capacity) {
        svg_capacity = (svg_capacity == 0) ? INITIAL_SVG_CAPACITY : svg_capacity * 2;
        SvgString* temp = (SvgString*)realloc(svg_strings, svg_capacity * sizeof(SvgString));
        if (temp == NULL) {
            fprintf(stderr, "Error: Failed to reallocate memory for SVG strings.\n");
            svg_capacity = 0;
            return;
        }
        svg_strings = temp;
    }
    char* str_copy = (char*)malloc(len + 1);
    if (str_copy == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for SVG string copy.\n");
        svg_strings[svg_count].is_valid = false;
        return;
    }
    strcpy(str_copy, new_str);
    svg_strings[svg_count].str = str_copy;
    svg_strings[svg_count].length = len;
    svg_strings[svg_count].is_valid = true;
    svg_count++;
}
void free_svg_strings() {
    for (size_t i = 0; i < svg_count; i++) {
        if (svg_strings[i].str != NULL) {
            free(svg_strings[i].str);
        }
    }
    if (svg_strings != NULL) {
        free(svg_strings);
    }
    svg_strings = NULL;
    svg_count = 0;
    svg_capacity = 0;
}

const char* SVG_HEADER_TEMPLATE = "<svg width=\"%d\" height=\"%d\" viewBox=\"0 0 %d %d\" xmlns=\"http://www.w3.org/2000/svg\">\n<rect width=\"100%%\" height=\"100%%\" fill=\"#FAFAFA\"/>\n<style>\n.neuron{stroke:#000;stroke-width:1;}\n.text{font:10px sans-serif; fill:#333;}\n.neg{stroke:red;}.pos{stroke:blue;}\n</style>\n";
const char* SVG_FOOTER = "\n</svg>";
const char* SVG_LAYER_LABEL_TEMPLATE = "<text x=\"%d\" y=\"%d\" class=\"text\" font-size=\"20\" font-weight=\"bold\" text-anchor=\"middle\">%s (%d)</text>\n";
const char* SVG_NEURON_TEMPLATE_IN = "<circle cx=\"%d\" cy=\"%.2f\" r=\"12\" class=\"neuron\" fill=\"#FFF\"/>\n<text x=\"%d\" y=\"%.2f\" class=\"text\" font-size=\"10\" text-anchor=\"middle\" dominant-baseline=\"central\">%d</text>\n";
const char* SVG_NEURON_TEMPLATE_OTHER = "<circle cx=\"%d\" cy=\"%.2f\" r=\"12\" class=\"neuron\" fill=\"#FFF\"/>\n<text x=\"%d\" y=\"%.2f\" class=\"text\" font-size=\"10\" text-anchor=\"middle\" dominant-baseline=\"central\">%s</text>\n";
const char* SVG_CONNECTION_TEMPLATE = "<line x1=\"%d\" y1=\"%.2f\" x2=\"%d\" y2=\"%.2f\" class=\"%s\" stroke-width=\"%.2f\" opacity=\"%.2f\"/>\n";
void generate_network_svg(NeuralNetwork* nn) {
    char buffer[512];
    snprintf(buffer, sizeof(buffer), SVG_HEADER_TEMPLATE, SVG_WIDTH, SVG_HEIGHT, SVG_WIDTH, SVG_HEIGHT);
    append_svg_string(buffer);
    const int X_IN = 100, X_HIDDEN = 600, X_OUT = 1100;
    const int Y_START = 50, Y_END = SVG_HEIGHT - Y_START;
    const double Y_TOTAL_SPACE = (double)(Y_END - Y_START);
    const double Y_SPACING_IN = Y_TOTAL_SPACE / NN_INPUT_SIZE;
    const double Y_SPACING_HIDDEN = Y_TOTAL_SPACE / NN_HIDDEN_SIZE; 
    double y_in[NN_INPUT_SIZE], y_hidden[NN_HIDDEN_SIZE], y_out[NN_OUTPUT_SIZE];

    // Input Layer 
    snprintf(buffer, sizeof(buffer), SVG_LAYER_LABEL_TEMPLATE, X_IN, Y_START - 20, "Input (Bits)", NN_INPUT_SIZE); append_svg_string(buffer);
    for (int i = 0; i < NN_INPUT_SIZE; i++) {
        y_in[i] = Y_START + i * Y_SPACING_IN + Y_SPACING_IN/2;
        snprintf(buffer, sizeof(buffer), SVG_NEURON_TEMPLATE_IN, X_IN, y_in[i], X_IN, y_in[i], i); append_svg_string(buffer);
    }
    // Hidden Layer 
    snprintf(buffer, sizeof(buffer), SVG_LAYER_LABEL_TEMPLATE, X_HIDDEN, Y_START - 20, "Hidden (Sampled)", NN_HIDDEN_SIZE); append_svg_string(buffer);
    for (int h = 0; h < NN_HIDDEN_SIZE; h++) {
        y_hidden[h] = Y_START + h * Y_SPACING_HIDDEN + Y_SPACING_HIDDEN/2;
        if (h < 20 || h > NN_HIDDEN_SIZE - 20 || h % 50 == 0) {
             snprintf(buffer, sizeof(buffer), SVG_NEURON_TEMPLATE_OTHER, X_HIDDEN, y_hidden[h], X_HIDDEN, y_hidden[h], ""); append_svg_string(buffer);
        }
    }
    // Output Layer 
    snprintf(buffer, sizeof(buffer), SVG_LAYER_LABEL_TEMPLATE, X_OUT, Y_START - 20, "Output", NN_OUTPUT_SIZE); append_svg_string(buffer);
    y_out[0] = Y_START + Y_TOTAL_SPACE / 2.0;
    snprintf(buffer, sizeof(buffer), SVG_NEURON_TEMPLATE_OTHER, X_OUT, y_out[0], X_OUT, y_out[0], "PRIME?"); append_svg_string(buffer);

    // Connections (Input -> Hidden) - Sampled
    for (int h = 0; h < NN_HIDDEN_SIZE; h += 10) { 
        for (int i = 0; i < NN_INPUT_SIZE; i++) {
            double weight = nn->weights_ih.data[h][i];
            const char* class = (weight >= 0) ? "pos" : "neg";
            double abs_weight = fabs(weight);
            double width = fmin(2.5, abs_weight * 5.0);
            double opacity = fmin(0.1, abs_weight * 0.5); 
            snprintf(buffer, sizeof(buffer), SVG_CONNECTION_TEMPLATE, 
                     X_IN + 10, y_in[i], X_HIDDEN - 10, y_hidden[h], class, width, opacity); append_svg_string(buffer);
        }
    }

    // Connections (Hidden -> Output) - All
    int o = 0; 
    for (int h = 0; h < NN_HIDDEN_SIZE; h++) {
        double weight = nn->weights_ho.data[o][h];
        const char* class = (weight >= 0) ? "pos" : "neg";
        double abs_weight = fabs(weight);
        double width = fmin(2.5, abs_weight * 5.0);
        double opacity = fmin(1.0, abs_weight * 3.0);
        snprintf(buffer, sizeof(buffer), SVG_CONNECTION_TEMPLATE, 
                 X_HIDDEN + 10, y_hidden[h], X_OUT - 10, y_out[o], class, width, opacity); append_svg_string(buffer);
    }
    append_svg_string(SVG_FOOTER);
}

void save_network_as_svg(NeuralNetwork* nn) {
    FILE *fp = fopen(SVG_FILENAME, "w");
    if (fp == NULL) { fprintf(stderr, "\nERROR: Could not open file %s for writing. Skipping SVG save.\n", SVG_FILENAME); free_svg_strings(); return; }
    generate_network_svg(nn);
    for (size_t i = 0; i < svg_count; i++) {
        if (validate_svg_string(&svg_strings[i])) { fprintf(fp, "%s", svg_strings[i].str); }
    }
    fclose(fp);
    free_svg_strings();
}


// --- Main Execution ---

int main() {
    srand((unsigned int)time(NULL));
    
    // --- STEP 1: Pre-compute Sieve and Arrays ---
    printf("Setting up Sieve and Pre-computing 10,000 Primes/Composites...\n");
    precompute_primes_sieve();
    double precomp_time = generate_precomputed_data();
    
    printf("Pre-computation Complete. Time taken: %.4f seconds.\n", precomp_time);
    printf("First 10 Primes: ");
    for(int i = 0; i < 10; i++) printf("%d ", primes_array[i]);
    printf("\nFirst 10 Composites: ");
    for(int i = 0; i < 10; i++) printf("%d ", composites_array[i]);
    printf("\n");
    
    // --- STEP 2: Validate Data ---
    validate_precomputed_data();

    // --- STEP 3: Initialize NN and Begin Sequential Training ---
    NeuralNetwork nn;
    nn_init(&nn);

    printf("Neural Network Prime Detector Initialized.\n");
    printf("Architecture: Input=%d, Hidden=%d, Output=%d\n", NN_INPUT_SIZE, NN_HIDDEN_SIZE, NN_OUTPUT_SIZE);
    printf("Learning Rate: %.4f (FIXED: Reduced for stability)\n", NN_LEARNING_RATE);
    printf("Batch Size: %d Primes + %d Composites (Total 1024)\n", BATCH_SIZE_HALF, BATCH_SIZE_HALF);
    printf("Max Training Time: %.0f seconds.\n", MAX_TRAINING_SECONDS);
    printf("Total Batches per Epoch: %d (plus a smaller last batch)\n", NUM_BATCHES);
    printf("------------------------------------------------------------------------------------------------\n");
    printf("Batch No. | Primes Range | Composites Range | Avg MSE (Batch) | Correctness (Total) | Backprop Time (sec)\n");
    printf("------------------------------------------------------------------------------------------------\n");
    fflush(stdout);

    time_t start_time = time(NULL);
    double current_success_rate = 0.0;
    int next_milestone_percent = 10;
    int total_batches_run = 0;
    int max_epochs = 1000; // Safety cap

    // Outer loop for epochs/passes over the data
    for (int epoch = 0; epoch < max_epochs; epoch++) {
        bool time_limit_reached = false;
        
        // --- Shuffle Data at Start of Epoch (CRUCIAL FIX for stability) ---
        shuffle_array(primes_array, NUM_EXAMPLES);
        shuffle_array(composites_array, NUM_EXAMPLES);
        printf("\n--- EPOCH %d STARTED (Data Shuffled). ---\n", epoch + 1);

        // Inner loop iterates through all sequential batches
        // Run NUM_BATCHES to include the 19 full batches, and an extra loop for the partial last batch.
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
    printf("Reason: %s\n", (current_success_rate >= 95.0) ? "Goal achieved." : "Time limit reached (%.0f seconds)." );
    printf("Final correctness: %.2f%%\n", current_success_rate);
    printf("Total Batches Run: %d\n", total_batches_run);
    printf("Total Training Time: %.0f seconds.\n", difftime(time(NULL), start_time));
    printf("#####################################################\n");
    fflush(stdout);

    // --- POST-TRAINING SVG SAVE ---
    save_network_as_svg(&nn);
    printf("Final network SVG saved to %s.\n", SVG_FILENAME);

    // --- Cleanup ---
    if (prime_cache != NULL) {
        free(prime_cache);
    }
    nn_free(&nn);

    return 0;
}
