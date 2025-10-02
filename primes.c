#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <string.h>

// --- Prime Detection Constants ---
// The 200th prime is 1223. We need 11 bits to represent up to 2047.
#define BIT_DEPTH 11                // Input size: Binary representation of the number
#define MAX_PRIME_TO_TEST 1223      // The 200th prime is 1223
#define TEST_RANGE (MAX_PRIME_TO_TEST + 1) // Test numbers from 1 to 1223
#define TRAINING_GOAL_PERCENT 95.0  // Target accuracy on the test set
#define MAX_EPOCHS 500000           // Max total batches (safety limit)
#define BATCH_SIZE_HALF 64          // 64 Primes + 64 Non-Primes = 128 total per batch

// --- NN Architecture Constants ---
#define NN_INPUT_SIZE BIT_DEPTH     // 11 bits input
#define NN_OUTPUT_SIZE 1            // Single output for classification (Prime/Not Prime)
#define NN_HIDDEN_SIZE 64           // Chosen hidden layer size
#define NN_LEARNING_RATE 0.005

// --- SVG Constants ---
#define SVG_WIDTH 1200
#define SVG_HEIGHT 1600
#define INITIAL_SVG_CAPACITY 2500
#define SVG_FILENAME "primesnetwork.svg"

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
typedef struct {
    char* str;
    size_t length;
    bool is_valid;
} SvgString;

// Global array and counter for dynamic SVG string collection
SvgString* svg_strings = NULL;
size_t svg_count = 0;
size_t svg_capacity = 0;

// --- Problem-Specific Utility Functions ---

// Pre-calculates primes up to MAX_PRIME_TO_TEST
bool prime_cache[TEST_RANGE];
void precompute_primes() {
    memset(prime_cache, true, sizeof(prime_cache));
    prime_cache[0] = prime_cache[1] = false;
    for (int p = 2; p * p <= MAX_PRIME_TO_TEST; p++) {
        if (prime_cache[p]) {
            for (int i = p * p; i <= MAX_PRIME_TO_TEST; i += p)
                prime_cache[i] = false;
        }
    }
}

// Checks if a number is prime using the pre-computed cache
bool is_prime(int n) {
    if (n < 2 || n > MAX_PRIME_TO_TEST) return false;
    return prime_cache[n];
}

// Converts an integer into its 11-bit binary representation (input array)
void int_to_binary_array(int n, double* arr) {
    for (int i = 0; i < BIT_DEPTH; i++) {
        // LSB is at index 0, MSB at index BIT_DEPTH-1
        arr[i] = (double)((n >> i) & 1);
    }
}

// Generates one training example
void generate_prime_nonprime_batch(double input[BIT_DEPTH], double* target) {
    int n;
    
    // Choose whether to generate a prime or non-prime number
    if (rand() % 2 == 0) {
        // Generate a prime number (Target: 1.0)
        *target = 1.0;
        do {
            n = 2 + rand() % (MAX_PRIME_TO_TEST - 1); // Range [2, 1223]
        } while (!is_prime(n));
    } else {
        // Generate a non-prime number (Target: 0.0)
        *target = 0.0;
        do {
            n = 4 + rand() % (MAX_PRIME_TO_TEST - 3); // Range [4, 1223]
        } while (is_prime(n));
    }
    
    int_to_binary_array(n, input);
}

// --- Matrix & NN Core Utility Functions (Retained from previous version) ---
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

// --- Neural Network Functions ---

void nn_init(NeuralNetwork* nn) {
    nn->lr = NN_LEARNING_RATE;
    // NN_INPUT_SIZE (11) -> NN_HIDDEN_SIZE (64)
    nn->weights_ih = matrix_create(NN_HIDDEN_SIZE, NN_INPUT_SIZE, NN_INPUT_SIZE);
    // NN_HIDDEN_SIZE (64) -> NN_OUTPUT_SIZE (1)
    nn->weights_ho = matrix_create(NN_OUTPUT_SIZE, NN_HIDDEN_SIZE, NN_HIDDEN_SIZE);
    nn->bias_h = (double*)calloc(NN_HIDDEN_SIZE, sizeof(double));
    nn->bias_o = (double*)calloc(NN_OUTPUT_SIZE, sizeof(double));
    
    // Allocate intermediate matrices
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

// nn_forward is designed for a single input/output array
void nn_forward(NeuralNetwork* nn, const double* input_array, double* output_array) {
    Matrix inputs_m = array_to_matrix(input_array, NN_INPUT_SIZE);
    matrix_copy_in(nn->inputs, inputs_m); matrix_free(inputs_m);

    // Input -> Hidden
    Matrix hidden_in_m = matrix_dot(nn->weights_ih, nn->inputs);
    for (int i = 0; i < NN_HIDDEN_SIZE; i++) hidden_in_m.data[i][0] += nn->bias_h[i];
    matrix_copy_in(nn->hidden_inputs, hidden_in_m); matrix_free(hidden_in_m);
    Matrix hidden_out_m = matrix_map(nn->hidden_inputs, tanh_activation);
    matrix_copy_in(nn->hidden_outputs, hidden_out_m); matrix_free(hidden_out_m);

    // Hidden -> Output
    Matrix output_in_m = matrix_dot(nn->weights_ho, nn->hidden_outputs);
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) output_in_m.data[i][0] += nn->bias_o[i];
    matrix_copy_in(nn->output_inputs, output_in_m); matrix_free(output_in_m);
    Matrix output_out_m = matrix_map(nn->output_inputs, sigmoid_activation);
    matrix_copy_in(nn->output_outputs, output_out_m);
    
    // Copy result to output array
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
        output_array[i] = output_out_m.data[i][0];
    }
    matrix_free(output_out_m);
}

// nn_backward is designed for a single target array
double nn_backward(NeuralNetwork* nn, const double* target_array) {
    double mse_loss = 0.0;
    Matrix targets_m = array_to_matrix(target_array, NN_OUTPUT_SIZE);
    
    // Calculate Output Error and Loss
    Matrix output_errors_m = matrix_add_subtract(nn->output_outputs, targets_m, false);
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) { 
        double error = output_errors_m.data[i][0];
        mse_loss += error * error; 
    }
    mse_loss /= NN_OUTPUT_SIZE;
    
    // Output Gradients
    Matrix output_d_m = matrix_map(nn->output_outputs, sigmoid_derivative);
    Matrix output_gradients_m = matrix_multiply_elem(output_errors_m, output_d_m);
    matrix_free(output_d_m); 
    
    // Update Weights Hidden -> Output (who) and Bias O
    Matrix hidden_out_t_m = matrix_transpose(nn->hidden_outputs);
    Matrix delta_who_m = matrix_dot(output_gradients_m, hidden_out_t_m);
    Matrix scaled_delta_who_m = matrix_multiply_scalar(delta_who_m, nn->lr);
    Matrix new_who_m = matrix_add_subtract(nn->weights_ho, scaled_delta_who_m, false);
    matrix_copy_in(nn->weights_ho, new_who_m);
    matrix_free(delta_who_m); matrix_free(scaled_delta_who_m); matrix_free(new_who_m); matrix_free(hidden_out_t_m);
    
    Matrix scaled_output_grad_m = matrix_multiply_scalar(output_gradients_m, nn->lr);
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) { nn->bias_o[i] -= scaled_output_grad_m.data[i][0]; }
    matrix_free(scaled_output_grad_m);

    // Calculate Hidden Errors (back-propagate)
    Matrix weights_ho_t_m = matrix_transpose(nn->weights_ho);
    Matrix hidden_errors_m = matrix_dot(weights_ho_t_m, output_gradients_m);
    matrix_free(weights_ho_t_m); matrix_free(output_gradients_m);
    
    // Hidden Gradients
    Matrix hidden_d_m = matrix_map(nn->hidden_outputs, tanh_derivative);
    Matrix hidden_gradients_m = matrix_multiply_elem(hidden_errors_m, hidden_d_m);
    matrix_free(hidden_errors_m); matrix_free(hidden_d_m);
    
    // Update Weights Input -> Hidden (wih) and Bias H
    Matrix inputs_t_m = matrix_transpose(nn->inputs);
    Matrix delta_wih_m = matrix_dot(hidden_gradients_m, inputs_t_m);
    Matrix scaled_delta_wih_m = matrix_multiply_scalar(delta_wih_m, nn->lr);
    Matrix new_wih_m = matrix_add_subtract(nn->weights_ih, scaled_delta_wih_m, false);
    matrix_copy_in(nn->weights_ih, new_wih_m);
    matrix_free(delta_wih_m); matrix_free(scaled_delta_wih_m); matrix_free(new_wih_m); matrix_free(inputs_t_m);

    Matrix scaled_hidden_grad_m = matrix_multiply_scalar(hidden_gradients_m, nn->lr);
    for (int i = 0; i < NN_HIDDEN_SIZE; i++) { nn->bias_h[i] -= scaled_hidden_grad_m.data[i][0]; }
    matrix_free(scaled_hidden_grad_m); matrix_free(hidden_gradients_m);
    
    matrix_free(targets_m);
    return mse_loss;
}

// Training Session (Batch)
double train_batch(NeuralNetwork* nn) {
    double total_mse = 0.0;
    double input_arr[BIT_DEPTH];
    double target_arr[NN_OUTPUT_SIZE];
    double output_arr[NN_OUTPUT_SIZE];
    
    int total_examples = BATCH_SIZE_HALF * 2;
    
    for (int i = 0; i < total_examples; i++) {
        // Generate mixed prime/non-prime example
        generate_prime_nonprime_batch(input_arr, target_arr);
        
        nn_forward(nn, input_arr, output_arr);
        total_mse += nn_backward(nn, target_arr);
    }
    return total_mse / total_examples;
}

// Testing function: Measures accuracy on all numbers up to the 200th prime (1223)
double test_network(NeuralNetwork* nn) {
    int correct_predictions = 0;
    int total_tests = 0;
    double input_arr[BIT_DEPTH];
    double output_arr[NN_OUTPUT_SIZE];

    for (int n = 1; n <= MAX_PRIME_TO_TEST; n++) {
        int_to_binary_array(n, input_arr);
        nn_forward(nn, input_arr, output_arr);
        
        // Target is 1.0 for prime, 0.0 for non-prime
        double target = is_prime(n) ? 1.0 : 0.0;
        double prediction = output_arr[0];
        
        // Classification: > 0.5 is Prime (1), <= 0.5 is Non-Prime (0)
        int classified_as_prime = (prediction > 0.5);
        int actual_is_prime = (target > 0.5);

        if (classified_as_prime == actual_is_prime) {
            correct_predictions++;
        }
        total_tests++;
    }
    
    return ((double)correct_predictions / total_tests) * 100.0;
}


// --- SVG Utility Functions (Retained from previous version) ---

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

// --- SVG Templates ---
const char* SVG_HEADER_TEMPLATE = "<svg width=\"%d\" height=\"%d\" viewBox=\"0 0 %d %d\" xmlns=\"http://www.w3.org/2000/svg\">\n<rect width=\"100%%\" height=\"100%%\" fill=\"#FAFAFA\"/>\n<style>\n.neuron{stroke:#000;stroke-width:1;}\n.text{font:10px sans-serif; fill:#333;}\n.neg{stroke:red;}.pos{stroke:blue;}\n</style>\n";
const char* SVG_FOOTER = "\n</svg>";
const char* SVG_LAYER_LABEL_TEMPLATE = "<text x=\"%d\" y=\"%d\" class=\"text\" font-size=\"20\" font-weight=\"bold\" text-anchor=\"middle\">%s (%d)</text>\n";
const char* SVG_NEURON_TEMPLATE = "<circle cx=\"%d\" cy=\"%.2f\" r=\"12\" class=\"neuron\" fill=\"#FFF\"/>\n<text x=\"%d\" y=\"%.2f\" class=\"text\" font-size=\"10\" text-anchor=\"middle\" dominant-baseline=\"central\">%s</text>\n";
const char* SVG_CONNECTION_TEMPLATE = "<line x1=\"%d\" y1=\"%.2f\" x2=\"%d\" y2=\"%.2f\" class=\"%s\" stroke-width=\"%.2f\" opacity=\"%.2f\"/>\n";

// Function to generate and collect ALL SVG parts
void generate_network_svg(NeuralNetwork* nn) {
    char buffer[512];

    // 1. SVG Header
    snprintf(buffer, sizeof(buffer), SVG_HEADER_TEMPLATE, SVG_WIDTH, SVG_HEIGHT, SVG_WIDTH, SVG_HEIGHT);
    append_svg_string(buffer);

    // 2. Constants for positioning
    const int X_IN = 100, X_HIDDEN = 600, X_OUT = 1100;
    const int Y_START = 50;
    const int Y_END = SVG_HEIGHT - Y_START;

    const double Y_TOTAL_SPACE = (double)(Y_END - Y_START);

    // Calculate spacing for 11 Input, 64 Hidden, 1 Output
    const double Y_SPACING_IN = Y_TOTAL_SPACE / NN_INPUT_SIZE;
    const double Y_SPACING_HIDDEN = Y_TOTAL_SPACE / NN_HIDDEN_SIZE;
    const double Y_SPACING_OUT = Y_TOTAL_SPACE / NN_OUTPUT_SIZE;

    // --- A. Draw Layers and Calculate Neuron Coordinates ---
    double y_in[NN_INPUT_SIZE];
    double y_hidden[NN_HIDDEN_SIZE];
    double y_out[NN_OUTPUT_SIZE];

    // Input Layer (11 Neurons)
    snprintf(buffer, sizeof(buffer), SVG_LAYER_LABEL_TEMPLATE, X_IN, Y_START - 20, "Input (Bits)", NN_INPUT_SIZE);
    append_svg_string(buffer);
    for (int i = 0; i < NN_INPUT_SIZE; i++) {
        y_in[i] = Y_START + i * Y_SPACING_IN + Y_SPACING_IN/2;
        // Neuron label is Bit Index
        snprintf(buffer, sizeof(buffer), SVG_NEURON_TEMPLATE, X_IN, y_in[i], X_IN, y_in[i], i == 0 ? "LSB" : i == NN_INPUT_SIZE - 1 ? "MSB" : "");
        append_svg_string(buffer);
    }

    // Hidden Layer (64 Neurons)
    snprintf(buffer, sizeof(buffer), SVG_LAYER_LABEL_TEMPLATE, X_HIDDEN, Y_START - 20, "Hidden", NN_HIDDEN_SIZE);
    append_svg_string(buffer);
    for (int h = 0; h < NN_HIDDEN_SIZE; h++) {
        y_hidden[h] = Y_START + h * Y_SPACING_HIDDEN + Y_SPACING_HIDDEN/2;
        snprintf(buffer, sizeof(buffer), SVG_NEURON_TEMPLATE, X_HIDDEN, y_hidden[h], X_HIDDEN, y_hidden[h], "");
        append_svg_string(buffer);
    }

    // Output Layer (1 Neuron) - Centered vertically
    snprintf(buffer, sizeof(buffer), SVG_LAYER_LABEL_TEMPLATE, X_OUT, Y_START - 20, "Output", NN_OUTPUT_SIZE);
    append_svg_string(buffer);
    y_out[0] = Y_START + Y_TOTAL_SPACE / 2.0; // Center the single output neuron
    snprintf(buffer, sizeof(buffer), SVG_NEURON_TEMPLATE, X_OUT, y_out[0], X_OUT, y_out[0], "PRIME?");
    append_svg_string(buffer);


    // --- B. Draw ALL Connections (Input -> Hidden) --- (11 * 64 = 704 connections)
    for (int h = 0; h < NN_HIDDEN_SIZE; h++) {
        for (int i = 0; i < NN_INPUT_SIZE; i++) {
            double weight = nn->weights_ih.data[h][i];
            const char* class = (weight >= 0) ? "pos" : "neg";
            double abs_weight = fabs(weight);
            double width = fmin(2.5, abs_weight * 5.0);
            double opacity = fmin(1.0, abs_weight * 3.0);

            snprintf(buffer, sizeof(buffer), SVG_CONNECTION_TEMPLATE, 
                     X_IN + 10, y_in[i],
                     X_HIDDEN - 10, y_hidden[h],
                     class, width, opacity);
            append_svg_string(buffer);
        }
    }

    // --- C. Draw ALL Connections (Hidden -> Output) --- (64 * 1 = 64 connections)
    // The output layer only has one neuron (index 0)
    int o = 0; 
    for (int h = 0; h < NN_HIDDEN_SIZE; h++) {
        double weight = nn->weights_ho.data[o][h];
        const char* class = (weight >= 0) ? "pos" : "neg";
        double abs_weight = fabs(weight);
        double width = fmin(2.5, abs_weight * 5.0);
        double opacity = fmin(1.0, abs_weight * 3.0);

        snprintf(buffer, sizeof(buffer), SVG_CONNECTION_TEMPLATE, 
                 X_HIDDEN + 10, y_hidden[h],
                 X_OUT - 10, y_out[o],
                 class, width, opacity);
        append_svg_string(buffer);
    }

    // 3. SVG Footer
    append_svg_string(SVG_FOOTER);
}


void save_network_as_svg(NeuralNetwork* nn) {
    
    FILE *fp = fopen(SVG_FILENAME, "w");
    if (fp == NULL) {
        fprintf(stderr, "\nERROR: Could not open file %s for writing. Skipping SVG save.\n", SVG_FILENAME);
        free_svg_strings();
        return;
    }

    generate_network_svg(nn);
    
    // Print all validated strings to file
    for (size_t i = 0; i < svg_count; i++) {
        if (validate_svg_string(&svg_strings[i])) {
            fprintf(fp, "%s", svg_strings[i].str);
        } else {
            fprintf(stderr, "SVG string at index %zu failed validation. Skipping.\n", i);
        }
    }
    
    fclose(fp);

    // Clean up memory
    free_svg_strings();
}


// --- Main Execution ---

int main() {
    srand((unsigned int)time(NULL));
    precompute_primes(); // Cache primes for fast checking

    NeuralNetwork nn;
    nn_init(&nn);

    printf("Neural Network Prime Detector Initialized.\n");
    printf("Test Range: 1 to %d (Contains the first 200 primes).\n", MAX_PRIME_TO_TEST);
    printf("Goal: Achieve %.2f%% accuracy on the test range.\n", TRAINING_GOAL_PERCENT);
    printf("Architecture: Input=%d (Bits), Hidden=%d, Output=%d (Prime/Not)\n",
           NN_INPUT_SIZE, NN_HIDDEN_SIZE, NN_OUTPUT_SIZE);
    fflush(stdout);

    time_t start_time = time(NULL);
    int batch_count = 0;
    double current_success_rate = 0.0;
    int next_milestone_percent = 10;
    
    // Safety limit, but the primary exit is the success rate
    while (batch_count < MAX_EPOCHS && current_success_rate < TRAINING_GOAL_PERCENT) {

        double avg_mse = train_batch(&nn);
        batch_count++;

        // --- Testing & Milestone Check (After every batch) ---
        current_success_rate = test_network(&nn);

        // Log general training stats
        printf("[Batch %d] MSE: %.8f | Correctness: %.2f%%\n", batch_count, avg_mse, current_success_rate);
        fflush(stdout);

        // Log milestones (10%, 20%, ..., 90%, 95%)
        while (current_success_rate >= next_milestone_percent) {
            printf("--- MILESTONE REACHED --- Batch: %d | Time: %.0f sec | Correctness: %.2f%%\n",
                   batch_count, difftime(time(NULL), start_time), current_success_rate);
            fflush(stdout);
            
            // Set next milestone. Handle the jump from 90% to 95%.
            if (next_milestone_percent == 90) {
                next_milestone_percent = (int)TRAINING_GOAL_PERCENT; // 95
            } else if (next_milestone_percent >= TRAINING_GOAL_PERCENT) {
                // Stop training if goal is met
                break;
            } else {
                next_milestone_percent += 10;
            }
        }
    }

    printf("\n#####################################################\n");
    printf("## TRAINING TERMINATED ##\n");
    if (current_success_rate >= TRAINING_GOAL_PERCENT) {
        printf("GOAL ACHIEVED: Correctness %.2f%% reached.\n", current_success_rate);
    } else {
        printf("MAX EPOCHS reached. Final correctness: %.2f%%\n", current_success_rate);
    }
    printf("Total Batches Run: %d\n", batch_count);
    printf("Total Training Time: %.0f seconds.\n", difftime(time(NULL), start_time));
    printf("#####################################################\n");
    fflush(stdout);

    // --- POST-TRAINING SVG SAVE ---
    save_network_as_svg(&nn);
    printf("Final network SVG saved to %s.\n", SVG_FILENAME);

    // --- Cleanup ---
    nn_free(&nn);

    return 0;
}
