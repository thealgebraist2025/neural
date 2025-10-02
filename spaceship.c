#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <string.h>

// --- Global Constants for Sorting Problem ---
#define ARRAY_SIZE 16               // The number of integers to sort
#define MAX_16BIT 65535.0           // Max value for 16-bit positive integer
#define BATCH_SIZE 100              // Number of training examples per batch
#define TEST_BATCH_SIZE 20          // Number of test examples per test run
#define TEST_INTERVAL_SECONDS 10    // Frequency of testing (in seconds)
#define LOG_INTERVAL_SECONDS 2      // Frequency of performance logging (in seconds)
#define MAX_TEST_ATTEMPTS 1000      // Max attempts to find a wrongly sorted example

// --- NN & Training Constants ---
#define NN_INPUT_SIZE ARRAY_SIZE    // 16
#define NN_OUTPUT_SIZE ARRAY_SIZE   // 16
#define NN_HIDDEN_SIZE 64           // Chosen hidden layer size
#define NN_LEARNING_RATE 0.005
#define EPOCHS 50000                // Total training batches to run

// --- SVG Constants (Adjusted for 64 neurons) ---
#define SVG_WIDTH 1200
#define SVG_HEIGHT 1600
#define INITIAL_SVG_CAPACITY 2500   // Increased capacity to hold all elements + overhead

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
    char* str;         // Pointer to the string data
    size_t length;     // Length of the string
    bool is_valid;     // Validation flag
} SvgString;

// Global array and counter for dynamic SVG string collection
SvgString* svg_strings = NULL;
size_t svg_count = 0;
size_t svg_capacity = 0;

// --- Utility Functions (Matrix Operations, NN Core Stubs, etc. - omitted for brevity in explanation) ---

// Creates and initializes a matrix with random values (using Xavier/He init scaling)
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
int compare_doubles(const void* a, const void* b) {
    if (*(const double*)a < *(const double*)b) return -1;
    if (*(const double*)a > *(const double*)b) return 1;
    return 0;
}
void generate_random_array(double* arr) {
    for (int i = 0; i < ARRAY_SIZE; i++) {
        unsigned short val = (unsigned short)(rand() % ((unsigned int)MAX_16BIT + 1));
        arr[i] = (double)val / MAX_16BIT;
    }
}
void generate_target_array(const double* input, double* target) {
    memcpy(target, input, ARRAY_SIZE * sizeof(double));
    qsort(target, ARRAY_SIZE, sizeof(double), compare_doubles);
}
void nn_init(NeuralNetwork* nn) {
    nn->lr = NN_LEARNING_RATE;
    nn->weights_ih = matrix_create(NN_HIDDEN_SIZE, NN_INPUT_SIZE, NN_INPUT_SIZE);
    nn->weights_ho = matrix_create(NN_OUTPUT_SIZE, NN_HIDDEN_SIZE, NN_HIDDEN_SIZE);
    nn->bias_h = (double*)calloc(NN_HIDDEN_SIZE, sizeof(double));
    nn->bias_o = (double*)calloc(NN_OUTPUT_SIZE, sizeof(double));
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
    matrix_copy_in(nn->inputs, inputs_m); matrix_free(inputs_m);
    Matrix hidden_in_m = matrix_dot(nn->weights_ih, nn->inputs);
    for (int i = 0; i < NN_HIDDEN_SIZE; i++) hidden_in_m.data[i][0] += nn->bias_h[i];
    matrix_copy_in(nn->hidden_inputs, hidden_in_m); matrix_free(hidden_in_m);
    Matrix hidden_out_m = matrix_map(nn->hidden_inputs, tanh_activation);
    matrix_copy_in(nn->hidden_outputs, hidden_out_m); matrix_free(hidden_out_m);
    Matrix output_in_m = matrix_dot(nn->weights_ho, nn->hidden_outputs);
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) output_in_m.data[i][0] += nn->bias_o[i];
    matrix_copy_in(nn->output_inputs, output_in_m); matrix_free(output_in_m);
    Matrix output_out_m = matrix_map(nn->output_inputs, sigmoid_activation);
    matrix_copy_in(nn->output_outputs, output_out_m);
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
        output_array[i] = output_out_m.data[i][0];
    }
    matrix_free(output_out_m);
}
double nn_backward(NeuralNetwork* nn, const double* target_array) {
    double mse_loss = 0.0;
    Matrix targets_m = array_to_matrix(target_array, NN_OUTPUT_SIZE);
    Matrix output_errors_m = matrix_add_subtract(nn->output_outputs, targets_m, false);
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) { mse_loss += output_errors_m.data[i][0] * output_errors_m.data[i][0]; }
    mse_loss /= NN_OUTPUT_SIZE;
    Matrix output_d_m = matrix_map(nn->output_outputs, sigmoid_derivative);
    Matrix output_gradients_m = matrix_multiply_elem(output_errors_m, output_d_m);
    matrix_free(output_d_m); matrix_free(output_errors_m);
    Matrix hidden_out_t_m = matrix_transpose(nn->hidden_outputs);
    Matrix delta_who_m = matrix_dot(output_gradients_m, hidden_out_t_m);
    Matrix scaled_delta_who_m = matrix_multiply_scalar(delta_who_m, nn->lr);
    Matrix new_who_m = matrix_add_subtract(nn->weights_ho, scaled_delta_who_m, false);
    matrix_copy_in(nn->weights_ho, new_who_m);
    matrix_free(delta_who_m); matrix_free(scaled_delta_who_m);
    matrix_free(new_who_m); matrix_free(hidden_out_t_m);
    Matrix scaled_output_grad_m = matrix_multiply_scalar(output_gradients_m, nn->lr);
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) { nn->bias_o[i] -= scaled_output_grad_m.data[i][0]; }
    matrix_free(scaled_output_grad_m);
    Matrix weights_ho_t_m = matrix_transpose(nn->weights_ho);
    Matrix hidden_errors_m = matrix_dot(weights_ho_t_m, output_gradients_m);
    matrix_free(weights_ho_t_m); matrix_free(output_gradients_m);
    Matrix hidden_d_m = matrix_map(nn->hidden_outputs, tanh_derivative);
    Matrix hidden_gradients_m = matrix_multiply_elem(hidden_errors_m, hidden_d_m);
    matrix_free(hidden_errors_m); matrix_free(hidden_d_m);
    Matrix inputs_t_m = matrix_transpose(nn->inputs);
    Matrix delta_wih_m = matrix_dot(hidden_gradients_m, inputs_t_m);
    Matrix scaled_delta_wih_m = matrix_multiply_scalar(delta_wih_m, nn->lr);
    Matrix new_wih_m = matrix_add_subtract(nn->weights_ih, scaled_delta_wih_m, false);
    matrix_copy_in(nn->weights_ih, new_wih_m);
    matrix_free(delta_wih_m); matrix_free(scaled_delta_wih_m);
    matrix_free(new_wih_m); matrix_free(inputs_t_m);
    Matrix scaled_hidden_grad_m = matrix_multiply_scalar(hidden_gradients_m, nn->lr);
    for (int i = 0; i < NN_HIDDEN_SIZE; i++) { nn->bias_h[i] -= scaled_hidden_grad_m.data[i][0]; }
    matrix_free(scaled_hidden_grad_m); matrix_free(hidden_gradients_m);
    matrix_free(targets_m);
    return mse_loss;
}
double train_batch(NeuralNetwork* nn) {
    double total_mse = 0.0;
    double input_arr[ARRAY_SIZE];
    double target_arr[ARRAY_SIZE];
    double output_arr[ARRAY_SIZE];
    for (int i = 0; i < BATCH_SIZE; i++) {
        generate_random_array(input_arr);
        generate_target_array(input_arr, target_arr);
        nn_forward(nn, input_arr, output_arr);
        total_mse += nn_backward(nn, target_arr);
    }
    return total_mse / BATCH_SIZE;
}
double test_network(NeuralNetwork* nn) {
    int total_pairs = TEST_BATCH_SIZE * (ARRAY_SIZE - 1);
    int correctly_sorted_pairs = 0;
    double input_arr[ARRAY_SIZE];
    double output_arr[ARRAY_SIZE];
    for (int i = 0; i < TEST_BATCH_SIZE; i++) {
        generate_random_array(input_arr);
        nn_forward(nn, input_arr, output_arr);
        for (int j = 0; j < ARRAY_SIZE - 1; j++) {
            if (output_arr[j] <= output_arr[j+1]) { correctly_sorted_pairs++; }
        }
    }
    return ((double)correctly_sorted_pairs / total_pairs) * 100.0;
}
bool find_wrongly_sorted_example(NeuralNetwork* nn, double* input, double* output) {
    for (int attempt = 0; attempt < MAX_TEST_ATTEMPTS; attempt++) {
        generate_random_array(input);
        nn_forward(nn, input, output);
        bool is_perfectly_sorted = true;
        for (int j = 0; j < ARRAY_SIZE - 1; j++) {
            if (output[j] > output[j+1]) {
                is_perfectly_sorted = false;
                break;
            }
        }
        if (!is_perfectly_sorted) { return true; }
    }
    return false;
}

// --- NEW SVG UTILITY FUNCTIONS ---

// Checks if an SvgString is safe to use
bool validate_svg_string(const SvgString* s) {
    // Check if the pointer is not NULL, the validity flag is true, 
    // the internal pointer is not NULL, and the recorded length matches strlen.
    return s != NULL && s->is_valid && s->str != NULL && s->length == strlen(s->str);
}

// Dynamically appends a string to the global SVG array
void append_svg_string(const char* new_str) {
    if (new_str == NULL) return;

    size_t len = strlen(new_str);
    if (len == 0) return;

    // Resize array if capacity is reached
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

    // Allocate memory for the new string content
    char* str_copy = (char*)malloc(len + 1);
    if (str_copy == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for SVG string copy.\n");
        svg_strings[svg_count].is_valid = false;
        return;
    }

    // Copy and populate the struct
    strcpy(str_copy, new_str);

    svg_strings[svg_count].str = str_copy;
    svg_strings[svg_count].length = len;
    svg_strings[svg_count].is_valid = true;

    svg_count++;
}

// Frees all memory used by the global SVG array
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
const char* SVG_NEURON_TEMPLATE = "<circle cx=\"%d\" cy=\"%.2f\" r=\"12\" class=\"neuron\" fill=\"#FFF\"/>\n<text x=\"%d\" y=\"%.2f\" class=\"text\" font-size=\"10\" text-anchor=\"middle\" dominant-baseline=\"central\">%d</text>\n";
const char* SVG_CONNECTION_TEMPLATE = "<line x1=\"%d\" y1=\"%.2f\" x2=\"%d\" y2=\"%.2f\" class=\"%s\" stroke-width=\"%.2f\" opacity=\"%.2f\"/>\n";

// Function to generate and collect ALL SVG parts
void generate_network_svg(NeuralNetwork* nn) {
    char buffer[512]; // Increased buffer size to handle detailed connection strings

    // 1. SVG Header
    snprintf(buffer, sizeof(buffer), SVG_HEADER_TEMPLATE, SVG_WIDTH, SVG_HEIGHT, SVG_WIDTH, SVG_HEIGHT);
    append_svg_string(buffer);

    // 2. Constants for positioning
    const int X_IN = 100, X_HIDDEN = 600, X_OUT = 1100;
    const int Y_START = 50;
    const int Y_END = SVG_HEIGHT - Y_START;

    const double Y_TOTAL_SPACE = (double)(Y_END - Y_START);

    // Spacing needs to cover the 64 neurons in the full vertical space
    const double Y_SPACING_IN = Y_TOTAL_SPACE / NN_INPUT_SIZE;
    const double Y_SPACING_HIDDEN = Y_TOTAL_SPACE / NN_HIDDEN_SIZE;
    const double Y_SPACING_OUT = Y_TOTAL_SPACE / NN_OUTPUT_SIZE;

    // --- A. Draw Layers and Calculate Neuron Coordinates ---
    double y_in[NN_INPUT_SIZE];
    double y_hidden[NN_HIDDEN_SIZE];
    double y_out[NN_OUTPUT_SIZE];

    // Input Layer
    snprintf(buffer, sizeof(buffer), SVG_LAYER_LABEL_TEMPLATE, X_IN, Y_START - 20, "Input", NN_INPUT_SIZE);
    append_svg_string(buffer);
    for (int i = 0; i < NN_INPUT_SIZE; i++) {
        y_in[i] = Y_START + i * Y_SPACING_IN + Y_SPACING_IN/2;
        snprintf(buffer, sizeof(buffer), SVG_NEURON_TEMPLATE, X_IN, y_in[i], X_IN, y_in[i], i);
        append_svg_string(buffer);
    }

    // Hidden Layer
    snprintf(buffer, sizeof(buffer), SVG_LAYER_LABEL_TEMPLATE, X_HIDDEN, Y_START - 20, "Hidden", NN_HIDDEN_SIZE);
    append_svg_string(buffer);
    for (int h = 0; h < NN_HIDDEN_SIZE; h++) {
        y_hidden[h] = Y_START + h * Y_SPACING_HIDDEN + Y_SPACING_HIDDEN/2;
        snprintf(buffer, sizeof(buffer), SVG_NEURON_TEMPLATE, X_HIDDEN, y_hidden[h], X_HIDDEN, y_hidden[h], h);
        append_svg_string(buffer);
    }

    // Output Layer
    snprintf(buffer, sizeof(buffer), SVG_LAYER_LABEL_TEMPLATE, X_OUT, Y_START - 20, "Output", NN_OUTPUT_SIZE);
    append_svg_string(buffer);
    for (int o = 0; o < NN_OUTPUT_SIZE; o++) {
        y_out[o] = Y_START + o * Y_SPACING_OUT + Y_SPACING_OUT/2;
        snprintf(buffer, sizeof(buffer), SVG_NEURON_TEMPLATE, X_OUT, y_out[o], X_OUT, y_out[o], o);
        append_svg_string(buffer);
    }

    // --- B. Draw ALL Connections (Input -> Hidden) ---
    // 16 * 64 = 1024 connections
    for (int h = 0; h < NN_HIDDEN_SIZE; h++) {
        for (int i = 0; i < NN_INPUT_SIZE; i++) {
            double weight = nn->weights_ih.data[h][i];
            // Use line color/thickness to represent weight value
            const char* class = (weight >= 0) ? "pos" : "neg";
            double abs_weight = fabs(weight);
            // Limit width to 2.5px and opacity to 1.0
            double width = fmin(2.5, abs_weight * 5.0);
            double opacity = fmin(1.0, abs_weight * 3.0);

            // X_IN (y_in[i]) to X_HIDDEN (y_hidden[h])
            snprintf(buffer, sizeof(buffer), SVG_CONNECTION_TEMPLATE, 
                     X_IN + 10, y_in[i], // Start slightly to the right of input neuron center
                     X_HIDDEN - 10, y_hidden[h], // End slightly to the left of hidden neuron center
                     class, width, opacity);
            append_svg_string(buffer);
        }
    }

    // --- C. Draw ALL Connections (Hidden -> Output) ---
    // 64 * 16 = 1024 connections
    for (int o = 0; o < NN_OUTPUT_SIZE; o++) {
        for (int h = 0; h < NN_HIDDEN_SIZE; h++) {
            double weight = nn->weights_ho.data[o][h];
            // Use line color/thickness to represent weight value
            const char* class = (weight >= 0) ? "pos" : "neg";
            double abs_weight = fabs(weight);
            double width = fmin(2.5, abs_weight * 5.0);
            double opacity = fmin(1.0, abs_weight * 3.0);

            // X_HIDDEN (y_hidden[h]) to X_OUT (y_out[o])
            snprintf(buffer, sizeof(buffer), SVG_CONNECTION_TEMPLATE, 
                     X_HIDDEN + 10, y_hidden[h], // Start slightly to the right of hidden neuron center
                     X_OUT - 10, y_out[o], // End slightly to the left of output neuron center
                     class, width, opacity);
            append_svg_string(buffer);
        }
    }

    // 3. SVG Footer
    append_svg_string(SVG_FOOTER);
}


void print_network_as_svg(NeuralNetwork* nn) {
    printf("\n\n#####################################################\n");
    printf("## NEURAL NETWORK SVG REPRESENTATION (Complete Graph) ##\n");
    printf("#####################################################\n\n");

    // Generate all SVG parts into the dynamic array
    generate_network_svg(nn);

    printf("<!-- SVG Output Start (Copy this section and view in a browser) -->\n");

    // Print all validated strings
    for (size_t i = 0; i < svg_count; i++) {
        // Validation check is mandatory before using the string
        if (validate_svg_string(&svg_strings[i])) {
            printf("%s", svg_strings[i].str);
        } else {
            fprintf(stderr, "SVG string at index %zu failed validation. Skipping.\n", i);
        }
    }

    printf("<!-- SVG Output End -->\n");

    // Clean up memory
    free_svg_strings();
}


// --- Main Execution ---

int main() {
    // Initialize random number generator
    srand((unsigned int)time(NULL));

    NeuralNetwork nn;
    nn_init(&nn);

    printf("Neural Network Sort Trainer Initialized.\n");
    printf("Architecture: Input=%d, Hidden=%d, Output=%d\n",
           NN_INPUT_SIZE, NN_HIDDEN_SIZE, NN_OUTPUT_SIZE);
    fflush(stdout);

    time_t start_time = time(NULL);
    time_t last_test_time = start_time;
    time_t last_log_time = start_time;
    int batch_count = 0;
    int batches_since_last_log = 0;

    // Force a 2-minute training limit
    time_t max_end_time = start_time + 120; // 120 seconds = 2 minutes

    // Main Training Loop
    while (batch_count < EPOCHS && time(NULL) < max_end_time) {

        double avg_mse = train_batch(&nn);
        batch_count++;
        batches_since_last_log++;

        time_t current_time = time(NULL);
        if (current_time - last_log_time >= LOG_INTERVAL_SECONDS) {
            double elapsed_log = difftime(current_time, last_log_time);
            double batches_per_sec = (double)batches_since_last_log / elapsed_log;
            int batches_per_interval = (int)round(batches_per_sec * LOG_INTERVAL_SECONDS);

            printf("[Perf Log] Batch %d | MSE: %.8f | Batches/2s (Est): %d\n",
                   batch_count, avg_mse, batches_per_interval);
            fflush(stdout);

            last_log_time = current_time;
            batches_since_last_log = 0;
        }

        if (current_time - last_test_time >= TEST_INTERVAL_SECONDS) {

            double success_rate = test_network(&nn);

            printf("[Test Result] Batch %d | Success Rate (Monotonic Pairs): %.2f%%\n",
                   batch_count, success_rate);
            fflush(stdout);

            last_test_time = current_time;
        }

        if (batch_count % 500 == 0 && current_time - last_test_time < TEST_INTERVAL_SECONDS && current_time - last_log_time < LOG_INTERVAL_SECONDS) {
            printf("[Batch %d] MSE: %.8f\n", batch_count, avg_mse);
            fflush(stdout);
        }
    }

    printf("\nTraining complete after %d batches or 2-minute limit reached.\n", batch_count);
    fflush(stdout);

    // --- POST-TRAINING ANALYSIS ---

    // 1. Final Evaluation
    double final_success_rate = test_network(&nn);
    printf("\n--- FINAL EVALUATION ---\n");
    printf("Total Batches Run: %d\n", batch_count);
    printf("Success Rate (Monotonic Pairs on %d test arrays): %.2f%%\n",
           TEST_BATCH_SIZE, final_success_rate);
    fflush(stdout);

    // 2. Find and Print a Wrongly Sorted Example
    double bad_input[ARRAY_SIZE];
    double bad_output[ARRAY_SIZE];

    printf("\n--- FINDING WRONGLY SORTED EXAMPLE ---\n");
    if (find_wrongly_sorted_example(&nn, bad_input, bad_output)) {
        printf("FOUND a wrongly sorted example:\n");
        printf("  Input Array (Unsorted):\n    [");
        for (int i = 0; i < ARRAY_SIZE; i++) {
            printf("%.4f%s", bad_input[i], (i == ARRAY_SIZE - 1) ? "" : ", ");
        }
        printf("]\n");

        printf("  Output Array (Network Prediction):\n    [");
        for (int i = 0; i < ARRAY_SIZE; i++) {
            printf("%.4f%s", bad_output[i], (i == ARRAY_SIZE - 1) ? "" : ", ");
        }
        printf("]\n");
        fflush(stdout);

    } else {
        printf("Did NOT find a wrongly sorted example in %d attempts. (Network may be performing near-perfectly.)\n", MAX_TEST_ATTEMPTS);
        fflush(stdout);
    }

    // 3. Print Network as SVG (Complete)
    print_network_as_svg(&nn);
    fflush(stdout);

    // --- Cleanup ---
    nn_free(&nn);

    return 0;
}
