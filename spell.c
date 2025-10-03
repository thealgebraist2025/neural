#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <string.h>
#include <stdarg.h>

// --- Spelling Corrector & Input Constants ---
#define MAX_WORD_LEN 10             // Network input/output size (L)
#define VOCAB_SIZE 512              // Size of the hardcoded word list
#define ALPHABET_SIZE 27            // 'a'-'z' (26) + ' ' (1) for padding/decoding (C)

#define NN_INPUT_SIZE (MAX_WORD_LEN * ALPHABET_SIZE)    // 10 * 27 = 270
#define NN_OUTPUT_SIZE (MAX_WORD_LEN * ALPHABET_SIZE)   // 10 * 27 = 270

#define NUM_TRAIN_WORDS 64          
#define MISSPELLED_PER_WORD 16      
#define CORRECT_WORD_PROPORTION 0.1 
#define BATCH_SIZE ((int)(NUM_TRAIN_WORDS * MISSPELLED_PER_WORD / (1.0 - CORRECT_WORD_PROPORTION))) 
#define CORRECT_SAMPLES_IN_BATCH (BATCH_SIZE - (NUM_TRAIN_WORDS * MISSPELLED_PER_WORD)) 

#define NUM_TEST_WORDS VOCAB_SIZE   
#define MISSPELLED_PER_TEST_WORD 16 
#define TOTAL_TESTS (NUM_TEST_WORDS * MISSPELLED_PER_TEST_WORD) 

#define MAX_TRAINING_SECONDS 240.0  
#define TEST_INTERVAL_SECONDS 30.0  

// --- NN Architecture Constants (Single Hidden Layer) ---
#define NN_HIDDEN_SIZE 512          
#define NN_INITIAL_LEARNING_RATE 0.005 
#define NN_LR_DECAY_RATE 0.0001     

// --- SVG Constants ---
#define SVG_WIDTH 1000 
#define SVG_HEIGHT 500
#define INITIAL_SVG_CAPACITY 2500
#define SVG_FILENAME "network.svg"
#define NODE_RADIUS 6
#define NODE_SPACING 12
#define LAYER_SPACING 300 

// --- Vocabulary and Encoding Arrays ---
// ... (Vocabulary omitted for brevity, it remains unchanged) ...
const char* ENGLISH_WORDS[VOCAB_SIZE] = {
    // ... 512 words here ...
    "the", "and", "but", "not", "for", "with", "you", "all", "are", "can",
    "had", "his", "out", "was", "she", "new", "day", "use", "way", "get",
    "him", "how", "man", "one", "say", "see", "two", "who", "big", "did",
    "end", "fly", "her", "job", "lot", "off", "put", "run", "sit", "ten",
    "try", "why", "act", "age", "air", "arm", "art", "ash", "awe", "bay",
    "bed", "bee", "box", "bug", "bus", "buy", "cap", "car", "cat", "cow",
    "cup", "dog", "ear", "egg", "eye", "fan", "fee", "fig", "fin", "fox",
    "fun", "gap", "gas", "gum", "hat", "hen", "ice", "ink", "jar", "key",
    "kit", "leg", "lie", "map", "mat", "men", "mix", "mud", "net", "oil",
    "pen", "pie", "pig", "pot", "ray", "red", "rib", "rod", "rug", "sea",
    "sun", "tea", "toe", "top", "van", "vet", "war", "wet", "win", "wok",
    "yet", "zoo", "able", "acid", "also", "area", "baby", "back", "ball",
    "band", "bank", "base", "bear", "best", "blue", "body", "book", "burn",
    "call", "card", "care", "case", "cent", "city", "club", "cold", "come",
    "copy", "core", "cost", "dark", "data", "dead", "deal", "door", "down",
    "draw", "each", "east", "edge", "even", "face", "fact", "fall", "farm",
    "fast", "feel", "film", "fine", "fire", "firm", "fish", "food", "foot",
    "form", "four", "free", "from", "full", "gain", "game", "give", "glad",
    "good", "grow", "hair", "half", "hand", "hard", "have", "head", "hear",
    "heat", "help", "high", "hold", "home", "hope", "hour", "huge", "idea",
    "into", "iron", "join", "just", "keep", "kids", "kind", "lack", "land",
    "last", "late", "lead", "less", "life", "like", "line", "list", "look",
    "loss", "love", "main", "make", "many", "mark", "mass", "mean", "meet",
    "mind", "miss", "mode", "more", "most", "move", "much", "name", "near",
    "need", "next", "none", "only", "open", "over", "page", "pair", "part",
    "pass", "past", "path", "peak", "plan", "play", "plus", "pool", "post",
    "pull", "rate", "read", "real", "rest", "rich", "rise", "risk", "road",
    "rock", "roll", "room", "rule", "safe", "same", "save", "seat", "seem",
    "self", "send", "show", "side", "sign", "sing", "size", "slip", "slow",
    "soil", "some", "soon", "sort", "star", "stay", "step", "stop", "such",
    "sure", "take", "talk", "team", "tell", "than", "that", "them", "then",
    "they", "thin", "this", "time", "type", "unit", "upon", "view", "wait",
    "walk", "want", "wear", "week", "well", "west", "what", "when", "will",
    "wish", "word", "work", "year", "your", "about", "above", "admit", "agree",
    "allow", "alone", "among", "apply", "begin", "board", "break", "bring",
    "build", "catch", "check", "claim", "clear", "close", "cover", "cross",
    "daily", "death", "doubt", "drive", "eight", "empty", "enjoy", "enter",
    "exist", "extra", "faith", "false", "fight", "final", "first", "force",
    "front", "glass", "great", "group", "heavy", "horse", "hotel", "house",
    "image", "index", "issue", "judge", "knife", "large", "later", "laugh",
    "layer", "learn", "leave", "level", "local", "loose", "major", "match",
    "metal", "might", "money", "month", "moral", "mouth", "music", "night",
    "north", "occur", "ocean", "offer", "order", "other", "owner", "panel",
    "paper", "party", "peace", "press", "price", "prove", "quick", "quiet",
    "range", "reach", "ready", "refer", "reply", "right", "river", "round",
    "scene", "sense", "serve", "seven", "shake", "share", "shift", "short",
    "since", "sleep", "small", "smoke", "sound", "south", "space", "speak",
    "speed", "spent", "sport", "staff", "steel", "stock", "stone", "store",
    "study", "style", "sugar", "table", "taste", "third", "those", "three",
    "throw", "total", "touch", "train", "treat", "truth", "under", "until",
    "voice", "whole", "woman", "world", "worth", "write", "wrong", "young",
    "across", "amount", "charge", "common", "damage", "design", "detail",
    "effect", "effort", "enable", "ensure", "entire", "figure", "former",
    "ground", "happen", "indeed", "inside", "involve", "island", "itself",
    "listen", "memory", "modern", "nation", "nature", "normal", "period",
    "person", "pretty", "public", "reason", "report", "return", "second",
    "simple", "social", "source", "speech", "sudden", "though", "travel",
    "unique", "visual", "volume", "within", "wonder", "writer", "ability",
    "account", "address", "advance", "against", "airline", "already", "analyst",
    
};

const char ALPHABET[] = "abcdefghijklmnopqrstuvwxyz ";
int CHAR_TO_INT[128]; 

void init_char_mapping() {
    for (int i = 0; i < 128; i++) CHAR_TO_INT[i] = -1;
    for (int i = 0; i < 26; i++) {
        CHAR_TO_INT['a' + i] = i;
    }
    CHAR_TO_INT[' '] = 26;
}

// --- Data Structures ---
typedef struct { int rows; int cols; double** data; } Matrix;

typedef struct {
    // Weights and Biases
    Matrix weights_ih;         // Input -> Hidden 
    Matrix weights_ho;         // Hidden -> Output
    double* bias_h;
    double* bias_o;

    double lr;

    // FORWARD-PASS CACHES (Intermediates)
    Matrix inputs;             // Input vector (270x1)
    Matrix hidden_outputs;     // Activated hidden layer (512x1)
    Matrix output_outputs;     // Activated output layer (270x1)
    
    // SCRATCHPAD MATRICES (Pre-allocated for temporary calculations)
    Matrix h_in_cache;         // Un-activated hidden input (512x1)
    Matrix output_in_cache;    // Un-activated output input (270x1)
    
    Matrix targets_cache;      // Target vector (270x1)
    Matrix output_errors;      // Output error (270x1)
    
    // BACKWARD-PASS CACHES (Gradients/Deltas)
    Matrix W_grad_ho;          // Gradient of HO weights (270x512)
    Matrix h_errors_cache;     // Error propagated to hidden layer (512x1)
    Matrix h_d_m_cache;        // ReLU derivative mask for hidden layer (512x1)
    Matrix W_grad_ih;          // Gradient of IH weights (512x270)
    
    // --- NEW: TRANSPOSE SCRATCHPAD MATRICES (Optimized for performance) ---
    Matrix h_out_t_cache;      // hidden_outputs^T (1x512)
    Matrix weights_ho_t_cache; // weights_ho^T (512x270)
    Matrix inputs_t_cache;     // inputs^T (1x270)
    // ----------------------------------------------------------------------
} NeuralNetwork;

// --- Matrix Utility Functions (Modified to store result) ---
Matrix matrix_create(int rows, int cols, int input_size) {
    printf("[spell.c:%d] matrix_create(%d, %d, %d)\n", __LINE__, rows, cols, input_size);
    Matrix m; m.rows = rows; m.cols = cols;
    m.data = (double**)calloc(rows, sizeof(double*));
    if (m.data == NULL) { fprintf(stderr, "FATAL ERROR: Memory allocation failed for matrix rows.\n"); exit(EXIT_FAILURE); }

    double scale = (input_size > 0 && rows > 0) ? sqrt(2.0 / (input_size + rows)) : 1.0;
    for (int i = 0; i < rows; i++) {
        m.data[i] = (double*)calloc(cols, sizeof(double));
        if (m.data[i] == NULL) { fprintf(stderr, "FATAL ERROR: Memory allocation failed for matrix cols.\n"); exit(EXIT_FAILURE); }
        // Initialize weights if input_size is provided (Xavier/He initialization)
        if (input_size > 0) {
            for (int j = 0; j < cols; j++) {
                m.data[i][j] = (((double)rand() / RAND_MAX) * 2.0 - 1.0) * scale;
            }
        }
    }
    return m;
}

void matrix_free(Matrix m) {
    printf("[spell.c:%d] matrix_free(%dx%d)\n", __LINE__, m.rows, m.cols);
    if (m.data == NULL) return;
    for (int i = 0; i < m.rows; i++) free(m.data[i]);
    free(m.data);
}

void matrix_copy_in(Matrix A, const Matrix B) {
    printf("[spell.c:%d] matrix_copy_in(%dx%d <- %dx%d)\n", __LINE__, A.rows, A.cols, B.rows, B.cols);
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

void array_to_matrix_store(const double* arr, int size, Matrix result) {
    printf("[spell.c:%d] array_to_matrix_store(size %d -> %dx%d)\n", __LINE__, size, result.rows, result.cols);
    if (result.rows != size || result.cols != 1) {
        fprintf(stderr, "FATAL ERROR: array_to_matrix_store dimensions mismatch.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < size; i++) { result.data[i][0] = arr[i]; }
}

// Stores result of A * B into the pre-allocated Matrix C
void matrix_dot_store(Matrix A, Matrix B, Matrix C) {
    printf("[spell.c:%d] matrix_dot_store(%dx%d * %dx%d -> %dx%d)\n", __LINE__, A.rows, A.cols, B.rows, B.cols, C.rows, C.cols);
    if (A.cols != B.rows || A.rows != C.rows || B.cols != C.cols) {
        fprintf(stderr, "FATAL ERROR: matrix_dot_store dimensions mismatch.\n");
        fprintf(stderr, "Requested: %dx%d * %dx%d -> %dx%d. Actual C: %dx%d\n", 
                A.rows, A.cols, B.rows, B.cols, A.rows, B.cols, C.rows, C.cols);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < B.cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < A.cols; k++) { sum += A.data[i][k] * B.data[k][j]; }
            C.data[i][j] = sum;
        }
    }
}

// Stores result of A^T into the pre-allocated Matrix C
void matrix_transpose_store(Matrix m, Matrix result) {
    printf("[spell.c:%d] matrix_transpose_store(%dx%d -> %dx%d)\n", __LINE__, m.rows, m.cols, result.rows, result.cols);
    if (m.rows != result.cols || m.cols != result.rows) {
        fprintf(stderr, "FATAL ERROR: matrix_transpose_store dimensions mismatch.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) { result.data[j][i] = m.data[i][j]; }
    }
}

// Stores result of A op B into the pre-allocated Matrix C
void matrix_add_subtract_store(Matrix A, Matrix B, Matrix C, bool is_add) {
    printf("[spell.c:%d] matrix_add_subtract_store(%dx%d %s %dx%d -> %dx%d)\n", 
           __LINE__, A.rows, A.cols, is_add ? "+" : "-", B.rows, B.cols, C.rows, C.cols);
    if (A.rows != B.rows || A.rows != C.rows || A.cols != B.cols || A.cols != C.cols) {
        fprintf(stderr, "FATAL ERROR: matrix_add_subtract_store dimensions mismatch.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            if (is_add) { C.data[i][j] = A.data[i][j] + B.data[i][j]; }
            else { C.data[i][j] = A.data[i][j] - B.data[i][j]; }
        }
    }
}

// Stores result of A * B (element-wise) into the pre-allocated Matrix C
void matrix_multiply_elem_store(Matrix A, Matrix B, Matrix C) {
    printf("[spell.c:%d] matrix_multiply_elem_store(%dx%d * %dx%d -> %dx%d)\n", __LINE__, A.rows, A.cols, B.rows, B.cols, C.rows, C.cols);
    if (A.rows != B.rows || A.rows != C.rows || A.cols != B.cols || A.cols != C.cols) {
        fprintf(stderr, "FATAL ERROR: matrix_multiply_elem_store dimensions mismatch.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) { C.data[i][j] = A.data[i][j] * B.data[i][j]; }
    }
}

// Stores result of A * scalar into the pre-allocated Matrix C
void matrix_multiply_scalar_store(Matrix A, double scalar, Matrix C) {
    printf("[spell.c:%d] matrix_multiply_scalar_store(%dx%d * %.2f -> %dx%d)\n", __LINE__, A.rows, A.cols, scalar, C.rows, C.cols);
    if (A.rows != C.rows || A.cols != C.cols) {
        fprintf(stderr, "FATAL ERROR: matrix_multiply_scalar_store dimensions mismatch.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) { C.data[i][j] = A.data[i][j] * scalar; }
    }
}

// Maps A into the pre-allocated Matrix C
void matrix_map_store(Matrix m, Matrix result, double (*func)(double)) {
    printf("[spell.c:%d] matrix_map_store(%dx%d -> %dx%d)\n", __LINE__, m.rows, m.cols, result.rows, result.cols);
    if (m.rows != result.rows || m.cols != result.cols) {
        fprintf(stderr, "FATAL ERROR: matrix_map_store dimensions mismatch.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) { result.data[i][j] = func(m.data[i][j]); }
    }
}


// --- Activation Functions (Unchanged) ---
double relu_activation(double x) { return x > 0.0 ? x : 0.0; }
double relu_derivative(double y) { return y > 0.0 ? 1.0 : 0.0; }

void softmax(double* arr) {
    double max_val = arr[0];
    for (int i = 1; i < ALPHABET_SIZE; i++) {
        if (arr[i] > max_val) max_val = arr[i];
    }
    double sum_exp = 0.0;
    for (int i = 0; i < ALPHABET_SIZE; i++) {
        arr[i] = exp(arr[i] - max_val); 
        sum_exp += arr[i];
    }
    for (int i = 0; i < ALPHABET_SIZE; i++) {
        arr[i] /= sum_exp;
    }
}


// --- Encoding/Decoding Functions ---
void encode_word_ohe(const char* word, double* arr) {
    memset(arr, 0, NN_OUTPUT_SIZE * sizeof(double));
    int len = strlen(word);
    
    for (int i = 0; i < MAX_WORD_LEN; i++) {
        // FIXED: Cast char to unsigned char to resolve [-Wchar-subscripts] warning
        int char_code = (i < len) 
            ? CHAR_TO_INT[(unsigned char)word[i]] 
            : CHAR_TO_INT[(unsigned char)' '];
            
        if (char_code == -1) char_code = CHAR_TO_INT[' '];
        
        int start_index = i * ALPHABET_SIZE;
        arr[start_index + char_code] = 1.0;
    }
}

void decode_word_ohe(const double* arr, char* word_out) {
    for (int i = 0; i < MAX_WORD_LEN; i++) {
        int start_index = i * ALPHABET_SIZE;
        double max_val = -1.0;
        int max_idx = CHAR_TO_INT[' ']; 
        
        for (int j = 0; j < ALPHABET_SIZE; j++) {
            if (arr[start_index + j] > max_val) {
                max_val = arr[start_index + j];
                max_idx = j;
            }
        }
        word_out[i] = ALPHABET[max_idx];
    }
    word_out[MAX_WORD_LEN] = '\0';

    for (int i = MAX_WORD_LEN - 1; i >= 0; i--) {
        if (word_out[i] == ' ') {
            word_out[i] = '\0';
        } else {
            break;
        }
    }
}

void generate_misspelled_word(const char* original_word, char* misspelled_word_out) {
    int len = strlen(original_word);
    if (len == 0 || len > MAX_WORD_LEN) {
        strncpy(misspelled_word_out, original_word, MAX_WORD_LEN);
        misspelled_word_out[MAX_WORD_LEN] = '\0';
        return;
    }

    strncpy(misspelled_word_out, original_word, MAX_WORD_LEN);
    misspelled_word_out[MAX_WORD_LEN] = '\0';

    int error_type = rand() % 4;
    int idx1, idx2;
    char temp_word[MAX_WORD_LEN + 2]; 

    switch (error_type) {
        case 0: // Substitution
            idx1 = rand() % len;
            char original_char = original_word[idx1];
            char new_char;
            do {
                new_char = ALPHABET[rand() % 26]; 
            } while (new_char == original_char);
            misspelled_word_out[idx1] = new_char;
            break;

        case 1: // Deletion
            if (len <= 1) return; 
            idx1 = rand() % len;
            
            for (int i = idx1; i < len; i++) {
                misspelled_word_out[i] = original_word[i + 1];
            }
            misspelled_word_out[len - 1] = '\0';
            break;

        case 2: // Insertion
            if (len >= MAX_WORD_LEN) return; 
            idx1 = rand() % (len + 1); 
            char insert_char = ALPHABET[rand() % 26]; 

            strncpy(temp_word, original_word, idx1);
            temp_word[idx1] = insert_char;
            strcpy(temp_word + idx1 + 1, original_word + idx1);
            temp_word[MAX_WORD_LEN] = '\0'; 
            
            strncpy(misspelled_word_out, temp_word, MAX_WORD_LEN);
            misspelled_word_out[MAX_WORD_LEN] = '\0';
            break;

        case 3: // Transposition
            if (len <= 1) return;
            idx1 = rand() % (len - 1); 
            idx2 = idx1 + 1;
            
            char tmp = misspelled_word_out[idx1];
            misspelled_word_out[idx1] = misspelled_word_out[idx2];
            misspelled_word_out[idx2] = tmp;
            break;
    }
}


// --- Neural Network Functions (I -> H -> O) ---

void nn_init(NeuralNetwork* nn) {
    nn->lr = NN_INITIAL_LEARNING_RATE;
    int h = NN_HIDDEN_SIZE;
    int input_size = NN_INPUT_SIZE;
    int output_size = NN_OUTPUT_SIZE;
    
    // 1. Weights and Biases (Initialized)
    nn->weights_ih = matrix_create(h, input_size, input_size); // Input -> Hidden
    nn->weights_ho = matrix_create(output_size, h, h);             // Hidden -> Output
    
    nn->bias_h = (double*)calloc(h, sizeof(double));
    nn->bias_o = (double*)calloc(output_size, sizeof(double));
    if (nn->bias_h == NULL || nn->bias_o == NULL) { fprintf(stderr, "FATAL ERROR: Bias memory allocation failed.\n"); exit(EXIT_FAILURE); }
    
    // 2. FORWARD-PASS CACHES (Pre-allocated for intermediates)
    nn->inputs = matrix_create(input_size, 1, 0);
    nn->hidden_outputs = matrix_create(h, 1, 0);
    nn->output_outputs = matrix_create(output_size, 1, 0);
    
    // 3. SCRATCHPADS & GRADIENT CACHES (Pre-allocated for calculation/storage)
    nn->h_in_cache = matrix_create(h, 1, 0);
    nn->output_in_cache = matrix_create(output_size, 1, 0);
    nn->targets_cache = matrix_create(output_size, 1, 0);
    nn->output_errors = matrix_create(output_size, 1, 0);
    
    nn->W_grad_ho = matrix_create(output_size, h, 0);
    nn->h_errors_cache = matrix_create(h, 1, 0);
    nn->h_d_m_cache = matrix_create(h, 1, 0);
    nn->W_grad_ih = matrix_create(h, input_size, 0);
    
    // 4. NEW: Initialize Transpose Scratchpads
    nn->h_out_t_cache = matrix_create(1, h, 0);             // (h x 1)^T = 1 x h
    nn->weights_ho_t_cache = matrix_create(h, output_size, 0); // (output x h)^T = h x output
    nn->inputs_t_cache = matrix_create(1, input_size, 0);   // (input x 1)^T = 1 x input
}

void nn_free(NeuralNetwork* nn) {
    // 1. Weights
    matrix_free(nn->weights_ih); 
    matrix_free(nn->weights_ho);

    free(nn->bias_h); free(nn->bias_o); 
    
    // 2. Forward Caches
    matrix_free(nn->inputs); 
    matrix_free(nn->hidden_outputs);
    matrix_free(nn->output_outputs);
    
    // 3. Scratchpads & Gradients
    matrix_free(nn->h_in_cache);
    matrix_free(nn->output_in_cache);
    matrix_free(nn->targets_cache);
    matrix_free(nn->output_errors);
    
    matrix_free(nn->W_grad_ho);
    matrix_free(nn->h_errors_cache);
    matrix_free(nn->h_d_m_cache);
    matrix_free(nn->W_grad_ih);

    // 4. NEW: Free Transpose Scratchpads
    matrix_free(nn->h_out_t_cache);
    matrix_free(nn->weights_ho_t_cache);
    matrix_free(nn->inputs_t_cache);
}

void nn_forward(NeuralNetwork* nn, const double* input_array, double* output_array) {
    int h = NN_HIDDEN_SIZE;
    int input_size = NN_INPUT_SIZE;
    int output_size = NN_OUTPUT_SIZE;
    
    // 1. Load Input into cache
    array_to_matrix_store(input_array, input_size, nn->inputs);
    
    // 2. Input -> Hidden (ReLU)
    // h_in_cache = weights_ih * inputs
    matrix_dot_store(nn->weights_ih, nn->inputs, nn->h_in_cache);
    // Add bias
    for (int i = 0; i < h; i++) nn->h_in_cache.data[i][0] += nn->bias_h[i];
    // hidden_outputs = ReLU(h_in_cache)
    matrix_map_store(nn->h_in_cache, nn->hidden_outputs, relu_activation);

    // 3. Hidden -> Output (Softmax)
    // output_in_cache = weights_ho * hidden_outputs
    matrix_dot_store(nn->weights_ho, nn->hidden_outputs, nn->output_in_cache);
    // Add bias
    for (int i = 0; i < output_size; i++) nn->output_in_cache.data[i][0] += nn->bias_o[i];
    
    // Apply Softmax on each of the 10 character blocks, storing result in output_outputs
    for (int i = 0; i < MAX_WORD_LEN; i++) {
        int start_idx = i * ALPHABET_SIZE;
        double block[ALPHABET_SIZE];
        for(int j = 0; j < ALPHABET_SIZE; j++) block[j] = nn->output_in_cache.data[start_idx + j][0];
        
        softmax(block); 
        
        for(int j = 0; j < ALPHABET_SIZE; j++) nn->output_outputs.data[start_idx + j][0] = block[j];
    }
    
    // Copy result to external output array
    for (int i = 0; i < output_size; i++) { output_array[i] = nn->output_outputs.data[i][0]; }
}


double nn_backward(NeuralNetwork* nn, const double* target_array) {
    double total_cce_loss = 0.0;
    int h = NN_HIDDEN_SIZE;
    int input_size = NN_INPUT_SIZE;
    int output_size = NN_OUTPUT_SIZE;

    // Load target into cache (targets_cache)
    array_to_matrix_store(target_array, output_size, nn->targets_cache);
    
    // --- 1. Output Layer Error (Softmax + CCE) ---
    // Error is (Output - Target) for CCE with Softmax. Stored in output_errors.
    matrix_add_subtract_store(nn->output_outputs, nn->targets_cache, nn->output_errors, false);

    // Calculate CCE loss
    for (int i = 0; i < output_size; i++) { 
        if (nn->targets_cache.data[i][0] == 1.0) { 
            double p = nn->output_outputs.data[i][0];
            if (p < 1e-12) p = 1e-12;
            total_cce_loss -= log(p); 
        }
    }
    
    // **Update Weights HO**
    // 1. Transpose hidden_outputs into h_out_t_cache (1x512)
    matrix_transpose_store(nn->hidden_outputs, nn->h_out_t_cache); 
    // W_grad_ho = output_errors (270x1) * h_out_t_cache (1x512) -> W_grad_ho (270x512)
    matrix_dot_store(nn->output_errors, nn->h_out_t_cache, nn->W_grad_ho);
    
    // 2. Apply learning rate: weights_ho = weights_ho - (W_grad_ho * lr)
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < h; j++) {
            nn->weights_ho.data[i][j] -= nn->W_grad_ho.data[i][j] * nn->lr;
        }
    }
    
    // **Update Bias O**
    for (int i = 0; i < output_size; i++) { nn->bias_o[i] -= nn->output_errors.data[i][0] * nn->lr; }

    // --- 2. Hidden Layer Gradients (ReLU) ---
    // 1. Backpropagate error: weights_ho_t_cache = weights_ho^T (512x270)
    matrix_transpose_store(nn->weights_ho, nn->weights_ho_t_cache); 
    // h_errors_cache = weights_ho_t_cache (512x270) * output_errors (270x1) -> h_errors_cache (512x1)
    matrix_dot_store(nn->weights_ho_t_cache, nn->output_errors, nn->h_errors_cache);
    
    // 2. Apply ReLU derivative: h_d_m_cache = hidden_outputs (mapped to derivative) (512x1)
    matrix_map_store(nn->hidden_outputs, nn->h_d_m_cache, relu_derivative); 
    // h_gradients_m (stored in h_errors_cache) = h_errors_cache (element-wise) * h_d_m_cache (512x1)
    matrix_multiply_elem_store(nn->h_errors_cache, nn->h_d_m_cache, nn->h_errors_cache);
    
    // **Update Weights IH**
    // 1. Transpose input into inputs_t_cache (1x270)
    matrix_transpose_store(nn->inputs, nn->inputs_t_cache); 
    // W_grad_ih = h_errors_cache (512x1) * inputs_t_cache (1x270) -> W_grad_ih (512x270)
    matrix_dot_store(nn->h_errors_cache, nn->inputs_t_cache, nn->W_grad_ih);

    // 2. Apply learning rate: weights_ih = weights_ih - (W_grad_ih * lr)
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < input_size; j++) {
            nn->weights_ih.data[i][j] -= nn->W_grad_ih.data[i][j] * nn->lr;
        }
    }

    // **Update Bias H**
    for (int i = 0; i < h; i++) { nn->bias_h[i] -= nn->h_errors_cache.data[i][0] * nn->lr; }
    
    // Return average CCE loss per character position
    return total_cce_loss / MAX_WORD_LEN;
}

// Sequential Batch Training Function (Unchanged logic)
double train_sequential_batch(NeuralNetwork* nn, int* word_indices, double* bp_time) {
    clock_t start_bp = clock();
    double total_loss = 0.0;
    double input_arr[NN_INPUT_SIZE]; 
    double target_arr[NN_OUTPUT_SIZE];
    double output_arr[NN_OUTPUT_SIZE];
    
    char work_word[MAX_WORD_LEN + 2]; 

    // 1. Train on Misspelled Words
    for (int i = 0; i < NUM_TRAIN_WORDS; i++) {
        const char* correct_word = ENGLISH_WORDS[word_indices[i]];
        
        encode_word_ohe(correct_word, target_arr);
        
        for (int j = 0; j < MISSPELLED_PER_WORD; j++) { 
            generate_misspelled_word(correct_word, work_word);
            encode_word_ohe(work_word, input_arr); 

            nn_forward(nn, input_arr, output_arr);
            total_loss += nn_backward(nn, target_arr);
        }
    }
    
    // 2. Train on Correctly Spelled Words (Identity Mapping)
    for (int i = 0; i < CORRECT_SAMPLES_IN_BATCH; i++) {
        int word_idx = word_indices[rand() % NUM_TRAIN_WORDS];
        const char* correct_word = ENGLISH_WORDS[word_idx];
        
        encode_word_ohe(correct_word, target_arr);
        encode_word_ohe(correct_word, input_arr); 

        nn_forward(nn, input_arr, output_arr);
        total_loss += nn_backward(nn, target_arr);
    }
    
    clock_t end_bp = clock();
    *bp_time = ((double)(end_bp - start_bp)) / CLOCKS_PER_SEC; 

    return total_loss / BATCH_SIZE;
}

// Testing function (Unchanged logic)
double test_network(NeuralNetwork* nn, double* test_time, bool verbose, int* fixed_count) {
    clock_t start_test = clock();
    int correct_words = 0;
    double input_arr[NN_INPUT_SIZE]; 
    double output_arr[NN_OUTPUT_SIZE];
    
    char misspelled_word[MAX_WORD_LEN + 2]; 
    char decoded_output[MAX_WORD_LEN + 1];

    for (int i = 0; i < NUM_TEST_WORDS; i++) {
        const char* correct_word = ENGLISH_WORDS[i];
        
        for (int j = 0; j < MISSPELLED_PER_TEST_WORD; j++) {
            
            generate_misspelled_word(correct_word, misspelled_word);
            
            encode_word_ohe(misspelled_word, input_arr); 
            nn_forward(nn, input_arr, output_arr);
            
            decode_word_ohe(output_arr, decoded_output);
            
            if (strcmp(correct_word, decoded_output) == 0) { 
                correct_words++; 
            } else if (verbose) {
                printf("  [Test Fail] Input: '%s' -> Output: '%s' (Target: '%s')\n", 
                       misspelled_word, decoded_output, correct_word);
            }
        }
    }
    
    clock_t end_test = clock();
    *fixed_count = correct_words;
    *test_time = ((double)correct_words / TOTAL_TESTS) * 100.0;
    
    return ((double)correct_words / TOTAL_TESTS) * 100.0;
}


// --- SVG Utility Functions (Unchanged logic) ---

typedef struct { char* str; size_t length; bool is_valid; } SvgString;
SvgString* svg_strings = NULL;
size_t svg_count = 0;
size_t svg_capacity = 0;

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
        SvgString* new_strings = (SvgString*)realloc(svg_strings, svg_capacity * sizeof(SvgString));
        if (!new_strings) {
            fprintf(stderr, "FATAL ERROR: Could not reallocate memory for SVG storage.\n");
            exit(EXIT_FAILURE);
        }
        svg_strings = new_strings;
    }

    va_list args;
    va_start(args, format);
    int size = vsnprintf(NULL, 0, format, args);
    va_end(args);

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
    svg_add_string("  .positive { stroke: #0088ff; stroke-width: 0.5; }\n"); 
    svg_add_string("  .negative { stroke: #ff0044; stroke-width: 0.5; }\n"); 
    svg_add_string("  .bias { fill: #aaaaaa; }\n"); 
    svg_add_string("</style>\n");

    int layer_sizes[] = {NN_INPUT_SIZE, NN_HIDDEN_SIZE, NN_OUTPUT_SIZE};
    int num_layers = sizeof(layer_sizes) / sizeof(layer_sizes[0]); 
    int x_coords[num_layers];
    
    for (int i = 0; i < num_layers; i++) {
        x_coords[i] = 50 + i * LAYER_SPACING;
    }
    
    int node_radius_small = 2; 
    int node_spacing_small = 2;

    for (int i = 0; i < num_layers - 1; i++) {
        Matrix* weights = (i == 0) ? &nn->weights_ih : &nn->weights_ho;
        
        int prev_size = layer_sizes[i];
        int curr_size = layer_sizes[i+1];
        
        int r1 = (i == 0) ? node_radius_small : NODE_RADIUS;
        int s1 = (i == 0) ? node_spacing_small : NODE_SPACING;
        int r2 = (i == num_layers - 2) ? node_radius_small : NODE_RADIUS;
        int s2 = (i == num_layers - 2) ? node_spacing_small : NODE_SPACING;

        int y_offset_prev = SVG_HEIGHT / 2 - (prev_size * (r1 + s1)) / 2;
        int y_offset_curr = SVG_HEIGHT / 2 - (curr_size * (r2 + s2)) / 2;
        
        int step_prev = (i == 0) ? 10 : 5;
        int step_curr = (i == num_layers - 2) ? 10 : 5;


        for (int j = 0; j < curr_size; j += step_curr) { 
            for (int k = 0; k < prev_size; k += step_prev) { 
                double weight = (i == 0) ? weights->data[j][k] : weights->data[j][k]; // Accessing HO or IH
                double abs_weight = fabs(weight);
                const char* css_class = (weight >= 0) ? "positive" : "negative";
                
                double stroke_width = 0.05 + abs_weight * 1.5;
                if (stroke_width > 1.5) stroke_width = 1.5;

                int x1 = x_coords[i] + r1;
                int y1 = y_offset_prev + k * (r1 * 2 + s1) + r1;
                int x2 = x_coords[i+1] - r2;
                int y2 = y_offset_curr + j * (r2 * 2 + s2) + r2;
                
                svg_add_string("<line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" class=\"%s\" style=\"stroke-width:%.2f\" />\n", 
                               x1, y1, x2, y2, css_class, stroke_width);
            }
        }
    }

    for (int i = 0; i < num_layers; i++) {
        int size = layer_sizes[i];
        int r = (i == 0 || i == num_layers - 1) ? node_radius_small : NODE_RADIUS;
        int s = (i == 0 || i == num_layers - 1) ? node_spacing_small : NODE_SPACING;
        int step = (i == 0 || i == num_layers - 1) ? 10 : 1; 

        int y_offset = SVG_HEIGHT / 2 - (size * (r + s)) / 2;
        int cx = x_coords[i]; 

        for (int j = 0; j < size; j += step) {
            int cy = y_offset + j * (r * 2 + s) + r;
            
            const char* fill_color = (i == 0) ? "#cccccc" : "#ffffff";
            
            svg_add_string("<circle cx=\"%d\" cy=\"%d\" r=\"%d\" fill=\"%s\" class=\"neuron\" />\n", 
                           cx, cy, r, fill_color);

            if (i == 1 && j % 100 == 0) { 
                double bias = nn->bias_h[j];
                svg_add_string("<text x=\"%d\" y=\"%d\" font-size=\"8\" fill=\"#666666\" class=\"bias\">%.2f</text>\n",
                               cx + r + 2, cy + 3, bias);
            }
        }
        
        const char* label = "";
        if (i == 0) label = "INPUT (270 OHE)";
        else if (i == 1) label = "HIDDEN (512 ReLU)";
        else if (i == 2) label = "OUTPUT (270 Softmax)";
        
        svg_add_string("<text x=\"%d\" y=\"%d\" font-size=\"12\" font-weight=\"bold\" text-anchor=\"middle\">%s</text>\n",
                       cx, SVG_HEIGHT - 10, label);
    }

    svg_add_string("</svg>\n");

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

// --- Main Execution (Unchanged logic) ---

int main() {
    srand((unsigned int)time(NULL));
    init_char_mapping();
    
    printf("Spelling Corrector Initializing with %d word vocabulary.\n", VOCAB_SIZE);
    
    NeuralNetwork nn;
    nn_init(&nn);

    printf("Architecture: Input(%d OHE) -> Hidden(%d ReLU) -> Output(%d Softmax)\n", NN_INPUT_SIZE, NN_HIDDEN_SIZE, NN_OUTPUT_SIZE);
    printf("Training Batch Size: ~%d examples/batch (%.0f%% correct words).\n", BATCH_SIZE, CORRECT_WORD_PROPORTION * 100);
    printf("Test Set Size: %d words * %d misspellings = %d examples/test.\n", NUM_TEST_WORDS, MISSPELLED_PER_TEST_WORD, TOTAL_TESTS);
    printf("Maximum Training Time: %.0f seconds. Testing occurs every %.0f seconds.\n", MAX_TRAINING_SECONDS, TEST_INTERVAL_SECONDS); 
    printf("----------------------------------------------------------------------------------------------------------------------------------\n");
    printf("Time (sec) | Batches Run | CCE Loss (Last Batch) | Spelling Errors Fixed (Absolute/%%) | Test Time (sec)\n");
    printf("----------------------------------------------------------------------------------------------------------------------------------\n");
    fflush(stdout);

    time_t start_time = time(NULL);
    time_t last_test_time = start_time - TEST_INTERVAL_SECONDS; 
    double current_success_rate = 0.0;
    int fixed_count = 0;
    double last_batch_cce_loss = 0.0;
    int next_milestone_percent = 10;
    int total_batches_run = 0;
    int max_epochs = 10000; 

    int word_indices[NUM_TRAIN_WORDS];

    for (int epoch = 0; epoch < max_epochs; epoch++) {
        bool time_limit_reached = false;
        
        double current_time_sec = difftime(time(NULL), start_time);
        nn.lr = NN_INITIAL_LEARNING_RATE / (1.0 + NN_LR_DECAY_RATE * current_time_sec);
        
        for (int i = 0; i < NUM_TRAIN_WORDS; i++) {
            word_indices[i] = rand() % VOCAB_SIZE;
        }

        double bp_time = 0.0;
        last_batch_cce_loss = train_sequential_batch(&nn, word_indices, &bp_time);
        total_batches_run++;

        time_t current_time = time(NULL);
        if (difftime(current_time, last_test_time) >= TEST_INTERVAL_SECONDS || total_batches_run == 1) {
            double test_time = 0.0;
            current_success_rate = test_network(&nn, &test_time, false, &fixed_count); 
            last_test_time = current_time;
            
            printf("%-10.0f | %-11d | %-21.8f | %-16d/%-6d (%.2f%%) | %-15.4f\n", 
                   current_time_sec,
                   total_batches_run, 
                   last_batch_cce_loss, 
                   fixed_count, TOTAL_TESTS, current_success_rate, 
                   test_time);
            fflush(stdout);
            
            while (current_success_rate >= next_milestone_percent) {
                printf("--- MILESTONE REACHED --- Correctness: %.2f%% (%d/%d) (LR: %.6f)\n", current_success_rate, fixed_count, TOTAL_TESTS, nn.lr);
                fflush(stdout);
                
                if (next_milestone_percent == 90) { next_milestone_percent = 95; } 
                else if (next_milestone_percent >= 95) { break; } 
                else { next_milestone_percent += 10; }
            }
        } 

        if (current_time_sec >= MAX_TRAINING_SECONDS || current_success_rate >= 95.0) {
            time_limit_reached = true;
        }
        
        if (time_limit_reached) break;
    }
    
    double final_test_time = 0.0;
    int final_fixed_count = 0;
    current_success_rate = test_network(&nn, &final_test_time, true, &final_fixed_count); 

    printf("----------------------------------------------------------------------------------------------------------------------------------\n");
    printf("\n#####################################################\n");
    printf("## TRAINING TERMINATED ##\n");
    printf("Final correctness (%d total tests): %d/%d (%.2f%%)\n", TOTAL_TESTS, final_fixed_count, TOTAL_TESTS, current_success_rate);
    printf("Total Batches Run: %d\n", total_batches_run);
    printf("Total Training Time: %.0f seconds.\n", difftime(time(NULL), start_time));
    printf("#####################################################\n");
    fflush(stdout);

    save_network_as_svg(&nn);
    printf("Final network SVG saved to %s.\n", SVG_FILENAME);

    nn_free(&nn);

    return 0;
}
