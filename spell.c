#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <string.h>
#include <stdarg.h>

// --- Spelling Corrector & Input Constants ---
#define MAX_WORD_LEN 10             // Network input/output size
#define VOCAB_SIZE 512              // Size of the hardcoded word list
#define ALPHABET_SIZE 27            // 'a'-'z' (26) + ' ' (1) for padding/decoding

#define NN_INPUT_SIZE MAX_WORD_LEN  // 10
#define NN_OUTPUT_SIZE MAX_WORD_LEN // 10

#define NUM_TRAIN_WORDS 32          // Number of base words in one batch (Increased from 16 to 32)
#define MISSPELLED_PER_WORD 64      // Number of misspellings per base word (Increased from 16 to 64)
#define BATCH_SIZE (NUM_TRAIN_WORDS * MISSPELLED_PER_WORD) // 2048
#define MAX_VAL_NEEDED 1            // Placeholder (not used for this problem)

#define MAX_TRAINING_SECONDS 240.0  
#define TEST_INTERVAL_SECONDS 10.0  // Check accuracy every 10 seconds

// --- NN Architecture Constants (Deep and Narrow) ---
#define NN_HIDDEN_SIZE 32           
#define NN_LEARNING_RATE 0.0005     

// --- SVG Constants (5 layers of 32 nodes) ---
#define SVG_WIDTH 1200
#define SVG_HEIGHT 500
#define INITIAL_SVG_CAPACITY 2500
#define SVG_FILENAME "network.svg"
#define NODE_RADIUS 6
#define NODE_SPACING 12
#define LAYER_SPACING 200

// --- Vocabulary and Encoding Arrays ---
const char* ENGLISH_WORDS[VOCAB_SIZE] = {
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
    "ancient", "another", "anxiety", "article", "attempt", "average", "balance",
    "billion", "brother", "capital", "century", "channel", "company", "compare",
    "concern", "culture", "current", "develop", "digital", "disease", "economy",
    "element", "evening", "example", "excited", "failure", "feeling", "finance",
    "freedom", "general", "history", "imagine", "initial", "instead", "mission",
    "nuclear", "patient", "pattern", "perform", "perhaps", "popular", "private",
    "problem", "process", "quality", "respect", "science", "section", "serious",
    "session", "similar", "society", "soldier", "station", "success", "suggest",
    "support", "teacher", "typical", "victory", "western", "whether", "without",
    "yourself", "determine", "different", "establish", "executive", "operation",
    "president", "structure", "technical", "thousands"
};

const char ALPHABET[] = "abcdefghijklmnopqrstuvwxyz ";
int CHAR_TO_INT[128]; // Map char to index (0-26)

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

    // Intermediate results for 5-layer backpropagation
    Matrix inputs;
    Matrix hidden1_outputs;
    Matrix hidden2_outputs;
    Matrix hidden3_outputs;     
    Matrix hidden4_outputs;     
    Matrix output_outputs;
} NeuralNetwork;

// --- SVG String Management Struct ---
typedef struct { char* str; size_t length; bool is_valid; } SvgString;
SvgString* svg_strings = NULL;
size_t svg_count = 0;
size_t svg_capacity = 0;

// --- Encoding/Decoding Functions ---

// Encodes a string (max 10 chars) into a double array of size 10, scaled 0.0 to ~0.96
void encode_word(const char* word, double* arr) {
    int len = strlen(word);
    for (int i = 0; i < MAX_WORD_LEN; i++) {
        if (i < len) {
            int char_code = CHAR_TO_INT[word[i]];
            if (char_code == -1) char_code = CHAR_TO_INT[' ']; // Fallback for safety
            arr[i] = (double)char_code / (ALPHABET_SIZE - 1); // Scale by 1/26.0
        } else {
            arr[i] = 0.0; // Pad with 0.0
        }
    }
}

// Decodes a double array (size 10) back into a string
void decode_word(const double* arr, char* word_out) {
    for (int i = 0; i < MAX_WORD_LEN; i++) {
        // Unscale the value and round to the nearest character code (0-26)
        int char_code = (int)round(arr[i] * (ALPHABET_SIZE - 1));
        
        // Clamp the code to the valid range
        if (char_code < 0) char_code = 0;
        if (char_code >= ALPHABET_SIZE) char_code = ALPHABET_SIZE - 1;

        word_out[i] = ALPHABET[char_code];
    }
    word_out[MAX_WORD_LEN] = '\0';

    // Trim trailing spaces
    for (int i = MAX_WORD_LEN - 1; i >= 0; i--) {
        if (word_out[i] == ' ') {
            word_out[i] = '\0';
        } else {
            break;
        }
    }
}

// Generates a misspelled word by changing one random letter
void generate_misspelled_word(const char* original_word, char* misspelled_word_out) {
    int len = strlen(original_word);
    if (len == 0 || len > MAX_WORD_LEN) {
        strncpy(misspelled_word_out, original_word, MAX_WORD_LEN);
        misspelled_word_out[MAX_WORD_LEN] = '\0';
        return;
    }

    // 1. Copy the original word
    strncpy(misspelled_word_out, original_word, MAX_WORD_LEN);
    misspelled_word_out[MAX_WORD_LEN] = '\0';
    
    // 2. Choose a random index to change (within the actual length)
    int change_idx = rand() % len;
    
    // 3. Choose a random letter that is NOT the current letter
    char original_char = original_word[change_idx];
    char new_char;
    do {
        int new_char_code = rand() % 26; // Choose 'a'-'z'
        new_char = ALPHABET[new_char_code];
    } while (new_char == original_char);
    
    // 4. Apply the change
    misspelled_word_out[change_idx] = new_char;
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

// --- Neural Network Functions (5-Layer Architecture: I->H1->H2->H3->H4->O) ---

void nn_init(NeuralNetwork* nn) {
    nn->lr = NN_LEARNING_RATE;
    int h_size = NN_HIDDEN_SIZE;
    
    // Weights (I->H1, H1->H2, H2->H3, H3->H4, H4->O)
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
    
    // 1. Input -> H1 (Tanh)
    Matrix h1_in_m = matrix_dot(nn->weights_ih1, inputs_m);
    for (int i = 0; i < h; i++) h1_in_m.data[i][0] += nn->bias_h1[i];
    Matrix h1_out_m = matrix_map(h1_in_m, tanh_activation);

    // 2. H1 -> H2 (Tanh)
    Matrix h2_in_m = matrix_dot(nn->weights_h1h2, h1_out_m);
    for (int i = 0; i < h; i++) h2_in_m.data[i][0] += nn->bias_h2[i];
    Matrix h2_out_m = matrix_map(h2_in_m, tanh_activation);

    // 3. H2 -> H3 (Tanh)
    Matrix h3_in_m = matrix_dot(nn->weights_h2h3, h2_out_m);
    for (int i = 0; i < h; i++) h3_in_m.data[i][0] += nn->bias_h3[i];
    Matrix h3_out_m = matrix_map(h3_in_m, tanh_activation);
    
    // 4. H3 -> H4 (Tanh)
    Matrix h4_in_m = matrix_dot(nn->weights_h3h4, h3_out_m);
    for (int i = 0; i < h; i++) h4_in_m.data[i][0] += nn->bias_h4[i];
    Matrix h4_out_m = matrix_map(h4_in_m, tanh_activation);

    // 5. H4 -> Output (Tanh for scaled character data)
    Matrix output_in_m = matrix_dot(nn->weights_h4o, h4_out_m);
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) output_in_m.data[i][0] += nn->bias_o[i];
    
    // Map Tanh output from (-1, 1) to (0, 1) to match input/target encoding
    Matrix output_out_m = matrix_map(output_in_m, tanh_activation);
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
        output_out_m.data[i][0] = (output_out_m.data[i][0] + 1.0) / 2.0; 
    }
    
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
    
    // Convert O back to tanh(Z)
    Matrix output_tanh_m = matrix_create(NN_OUTPUT_SIZE, 1, 0);
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
        // Tanh_output = 2 * O - 1
        output_tanh_m.data[i][0] = 2.0 * nn->output_outputs.data[i][0] - 1.0;
    }
    
    Matrix output_d_m = matrix_map(output_tanh_m, tanh_derivative); 
    matrix_free(output_tanh_m); 
    
    // Gradient relative to Tanh INPUT (Z): (O - T) * (1 - tanh^2(Z))
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
double train_sequential_batch(NeuralNetwork* nn, int* word_indices, double* bp_time) {
    clock_t start_bp = clock();
    double total_mse = 0.0;
    double input_arr[NN_INPUT_SIZE]; 
    double target_arr[NN_OUTPUT_SIZE];
    double output_arr[NN_OUTPUT_SIZE];
    
    char misspelled_word[MAX_WORD_LEN + 1];

    for (int i = 0; i < NUM_TRAIN_WORDS; i++) {
        const char* correct_word = ENGLISH_WORDS[word_indices[i]];
        
        // Encode the correct word as the target array
        encode_word(correct_word, target_arr);
        
        for (int j = 0; j < MISSPELLED_PER_WORD; j++) {
            // Generate a misspelled word
            generate_misspelled_word(correct_word, misspelled_word);
            
            // Encode the misspelled word as the input array
            encode_word(misspelled_word, input_arr); 

            // Forward and Backward pass
            nn_forward(nn, input_arr, output_arr);
            total_mse += nn_backward(nn, target_arr);
        }
    }
    
    clock_t end_bp = clock();
    *bp_time = ((double)(end_bp - start_bp)) / CLOCKS_PER_SEC; 

    return total_mse / BATCH_SIZE;
}

// Testing function
double test_network(NeuralNetwork* nn, double* test_time, bool verbose) {
    clock_t start_test = clock();
    int correct_words = 0;
    // Test on 32 words, same size as the training words sample
    int total_tests = 32; 
    double input_arr[NN_INPUT_SIZE]; 
    double output_arr[NN_OUTPUT_SIZE];
    
    char misspelled_word[MAX_WORD_LEN + 1];
    char decoded_output[MAX_WORD_LEN + 1];

    // Pick 32 random words from the vocabulary for testing
    for (int i = 0; i < total_tests; i++) {
        int word_idx = rand() % VOCAB_SIZE;
        const char* correct_word = ENGLISH_WORDS[word_idx];
        
        // Generate a test misspelling
        generate_misspelled_word(correct_word, misspelled_word);
        
        // Encode and run forward pass
        encode_word(misspelled_word, input_arr); 
        nn_forward(nn, input_arr, output_arr);
        
        // Decode the output
        decode_word(output_arr, decoded_output);
        
        // Check for correctness (word-level match)
        if (strcmp(correct_word, decoded_output) == 0) { 
            correct_words++; 
        } else if (verbose) {
            printf("  [Test Fail] Input: '%s' -> Output: '%s' (Target: '%s')\n", 
                   misspelled_word, decoded_output, correct_word);
        }
    }
    
    clock_t end_test = clock();
    *test_time = ((double)(end_test - start_test)) / CLOCKS_PER_SEC;
    return ((double)correct_words / total_tests) * 100.0;
}


// --- SVG Utility Functions (Unchanged) ---

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
        
        int cx = x_coords[i]; 

        for (int j = 0; j < size; j++) {
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
        if (i == 0) label = "INPUT (10 CHARS)";
        else if (i == 1) label = "HIDDEN 1 (32)";
        else if (i == 2) label = "HIDDEN 2 (32)";
        else if (i == 3) label = "HIDDEN 3 (32)";
        else if (i == 4) label = "HIDDEN 4 (32)";
        else if (i == 5) label = "OUTPUT (10 CHARS)";
        
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
    init_char_mapping();
    
    printf("Spelling Corrector Initializing with %d word vocabulary.\n", VOCAB_SIZE);

    // --- Initialize NN with NEW Architecture ---
    NeuralNetwork nn;
    nn_init(&nn);

    printf("Architecture: Input(%d) -> H1(32) -> H2(32) -> H3(32) -> H4(32) -> Output(%d)\n", NN_INPUT_SIZE, NN_OUTPUT_SIZE);
    printf("Training Batch Size: %d words * %d misspellings = %d examples/batch.\n", NUM_TRAIN_WORDS, MISSPELLED_PER_WORD, BATCH_SIZE);
    printf("Maximum Training Time: %.0f seconds. Testing occurs every %.0f seconds.\n", MAX_TRAINING_SECONDS, TEST_INTERVAL_SECONDS); 
    printf("--------------------------------------------------------------------------------------------------\n");
    printf("Time (sec) | Batches Run | Avg MSE (Last Batch) | Word Correctness (Test) | Test Time (sec)\n");
    printf("--------------------------------------------------------------------------------------------------\n");
    fflush(stdout);

    time_t start_time = time(NULL);
    time_t last_test_time = start_time - TEST_INTERVAL_SECONDS; // Force immediate first test
    double current_success_rate = 0.0;
    double last_batch_mse = 0.0;
    int next_milestone_percent = 10;
    int total_batches_run = 0;
    int max_epochs = 10000; 

    // Indices to select 32 random words from the vocabulary for the current batch
    int word_indices[NUM_TRAIN_WORDS];

    for (int epoch = 0; epoch < max_epochs; epoch++) {
        bool time_limit_reached = false;
        
        // --- Randomly select words for the batch ---
        for (int i = 0; i < NUM_TRAIN_WORDS; i++) {
            word_indices[i] = rand() % VOCAB_SIZE;
        }

        // --- Training Step (One batch) ---
        double bp_time = 0.0;
        last_batch_mse = train_sequential_batch(&nn, word_indices, &bp_time);
        total_batches_run++;

        // --- Testing Step (Every 10 seconds) ---
        time_t current_time = time(NULL);
        if (difftime(current_time, last_test_time) >= TEST_INTERVAL_SECONDS || total_batches_run == 1) {
            double test_time = 0.0;
            current_success_rate = test_network(&nn, &test_time, false); 
            last_test_time = current_time;
            
            // Log stats after test (Only print happens here)
            printf("%-10.0f | %-11d | %-20.8f | %-25.2f | %-15.4f\n", 
                   difftime(current_time, start_time),
                   total_batches_run, 
                   last_batch_mse, 
                   current_success_rate, 
                   test_time);
            fflush(stdout);
            
            // Log milestones
            while (current_success_rate >= next_milestone_percent) {
                printf("--- MILESTONE REACHED --- Correctness: %.2f%%\n", current_success_rate);
                fflush(stdout);
                
                if (next_milestone_percent == 90) { next_milestone_percent = 95; } 
                else if (next_milestone_percent >= 95) { break; } 
                else { next_milestone_percent += 10; }
            }
        } 

        // Check time limit or goal
        if (difftime(current_time, start_time) >= MAX_TRAINING_SECONDS) {
            time_limit_reached = true;
        }
        
        if (current_success_rate >= 95.0) {
             printf("--- GOAL ACHIEVED --- Correctness %.2f%% reached.\n", current_success_rate);
             time_limit_reached = true; 
        }
        
        if (time_limit_reached) break;
    }
    
    // Final test and log
    double final_test_time = 0.0;
    current_success_rate = test_network(&nn, &final_test_time, true); 

    printf("--------------------------------------------------------------------------------------------------\n");
    printf("\n#####################################################\n");
    printf("## TRAINING TERMINATED ##\n");
    printf("Final correctness (Tested on 32 new misspellings): %.2f%%\n", current_success_rate);
    printf("Total Batches Run: %d\n", total_batches_run);
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
