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

#define NUM_TRAIN_WORDS 64          // Increased from 32 to 64
#define MISSPELLED_PER_WORD 32      // Reduced to balance the larger NUM_TRAIN_WORDS
#define CORRECT_WORD_PROPORTION 0.1 // 10% of the batch will be correct words
#define BATCH_SIZE ((int)(NUM_TRAIN_WORDS * MISSPELLED_PER_WORD / (1.0 - CORRECT_WORD_PROPORTION))) // ~2275 examples
#define CORRECT_SAMPLES_IN_BATCH (BATCH_SIZE - (NUM_TRAIN_WORDS * MISSPELLED_PER_WORD)) // ~227

#define NUM_TEST_WORDS VOCAB_SIZE   // Test on all 512 words
#define MISSPELLED_PER_TEST_WORD 16 // 16 misspellings per test word
#define TOTAL_TESTS (NUM_TEST_WORDS * MISSPELLED_PER_TEST_WORD) // 8192

#define MAX_TRAINING_SECONDS 240.0  
#define TEST_INTERVAL_SECONDS 10.0  

// --- NN Architecture Constants (Deep and Narrow) ---
#define NN_HIDDEN_SIZE 64           // Increased hidden size to handle larger OHE input
#define NN_INITIAL_LEARNING_RATE 0.005 // Higher initial LR
#define NN_LR_DECAY_RATE 0.0001     // Time-based decay rate

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
    
    // Gradients for CCE backprop (needs to be stored)
    Matrix output_errors;
} NeuralNetwork;

// --- SVG String Management Struct (Unchanged) ---
typedef struct { char* str; size_t length; bool is_valid; } SvgString;
SvgString* svg_strings = NULL;
size_t svg_count = 0;
size_t svg_capacity = 0;

// --- Encoding/Decoding Functions ---

// Encodes a word (max 10 chars) into a ONE-HOT double array of size 270.
void encode_word_ohe(const char* word, double* arr) {
    memset(arr, 0, NN_OUTPUT_SIZE * sizeof(double));
    int len = strlen(word);
    
    for (int i = 0; i < MAX_WORD_LEN; i++) {
        int char_code;
        if (i < len) {
            char_code = CHAR_TO_INT[word[i]];
            if (char_code == -1) char_code = CHAR_TO_INT[' ']; // Safety for unknown char
        } else {
            char_code = CHAR_TO_INT[' ']; // Padding with space
        }
        
        int start_index = i * ALPHABET_SIZE;
        arr[start_index + char_code] = 1.0;
    }
}

// Decodes a ONE-HOT array (size 270) back into a string
void decode_word_ohe(const double* arr, char* word_out) {
    for (int i = 0; i < MAX_WORD_LEN; i++) {
        int start_index = i * ALPHABET_SIZE;
        double max_val = -1.0;
        int max_idx = CHAR_TO_INT[' ']; // Default to space
        
        // Find the index with the maximum value for this character position
        for (int j = 0; j < ALPHABET_SIZE; j++) {
            if (arr[start_index + j] > max_val) {
                max_val = arr[start_index + j];
                max_idx = j;
            }
        }
        word_out[i] = ALPHABET[max_idx];
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

// Generates a misspelled word by applying one random edit: SUB, DEL, INS, or TRA
void generate_misspelled_word(const char* original_word, char* misspelled_word_out) {
    int len = strlen(original_word);
    if (len == 0 || len > MAX_WORD_LEN) {
        strncpy(misspelled_word_out, original_word, MAX_WORD_LEN);
        misspelled_word_out[MAX_WORD_LEN] = '\0';
        return;
    }

    // Copy the original word
    strncpy(misspelled_word_out, original_word, MAX_WORD_LEN);
    misspelled_word_out[MAX_WORD_LEN] = '\0';

    // 4 types of errors: 0:SUB, 1:DEL, 2:INS, 3:TRA
    int error_type = rand() % 4;
    int idx1, idx2;
    char temp_word[MAX_WORD_LEN + 2]; // For insertion/deletion ease

    switch (error_type) {
        case 0: // Substitution (Change one letter, like previous)
            idx1 = rand() % len;
            char original_char = original_word[idx1];
            char new_char;
            do {
                new_char = ALPHABET[rand() % 26]; // 'a'-'z'
            } while (new_char == original_char);
            misspelled_word_out[idx1] = new_char;
            break;

        case 1: // Deletion (Remove one letter)
            if (len <= 1) return; // Cannot delete a 1-char word
            idx1 = rand() % len;
            
            // Shift characters left to fill the gap
            for (int i = idx1; i < len; i++) {
                misspelled_word_out[i] = original_word[i + 1];
            }
            misspelled_word_out[len - 1] = '\0';
            break;

        case 2: // Insertion (Insert one random letter)
            if (len >= MAX_WORD_LEN) return; // Word is already max length
            idx1 = rand() % (len + 1); // Insertion point can be 0 to len
            char insert_char = ALPHABET[rand() % 26]; // 'a'-'z'

            // Build new string in temp_word
            strncpy(temp_word, original_word, idx1);
            temp_word[idx1] = insert_char;
            strcpy(temp_word + idx1 + 1, original_word + idx1);
            temp_word[MAX_WORD_LEN] = '\0'; // Ensure it's capped
            
            strncpy(misspelled_word_out, temp_word, MAX_WORD_LEN);
            misspelled_word_out[MAX_WORD_LEN] = '\0';
            break;

        case 3: // Transposition (Swap two adjacent letters)
            if (len <= 1) return;
            idx1 = rand() % (len - 1); // First index to swap (0 to len-2)
            idx2 = idx1 + 1;
            
            char tmp = misspelled_word_out[idx1];
            misspelled_word_out[idx1] = misspelled_word_out[idx2];
            misspelled_word_out[idx2] = tmp;
            break;
    }
}


// --- Matrix & NN Core Utility Functions ---

// ReLU activation
double relu_activation(double x) { return x > 0.0 ? x : 0.0; }
// ReLU derivative
double relu_derivative(double y) { return y > 0.0 ? 1.0 : 0.0; }

// Softmax function (applied per 27-neuron block)
void softmax(double* arr) {
    // arr is a 27-element block
    double max_val = arr[0];
    for (int i = 1; i < ALPHABET_SIZE; i++) {
        if (arr[i] > max_val) max_val = arr[i];
    }

    double sum_exp = 0.0;
    for (int i = 0; i < ALPHABET_SIZE; i++) {
        // Subtract max_val for numerical stability
        arr[i] = exp(arr[i] - max_val); 
        sum_exp += arr[i];
    }
    
    // Normalize to get probabilities
    for (int i = 0; i < ALPHABET_SIZE; i++) {
        arr[i] /= sum_exp;
    }
}


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


// --- Neural Network Functions (5-Layer Architecture: I->H1->H2->H3->H4->O) ---

void nn_init(NeuralNetwork* nn) {
    nn->lr = NN_INITIAL_LEARNING_RATE;
    int h = NN_HIDDEN_SIZE;
    
    // Weights (I->H1, H1->H2, H2->H3, H3->H4, H4->O)
    nn->weights_ih1 = matrix_create(h, NN_INPUT_SIZE, NN_INPUT_SIZE);
    nn->weights_h1h2 = matrix_create(h, h, h);
    nn->weights_h2h3 = matrix_create(h, h, h); 
    nn->weights_h3h4 = matrix_create(h, h, h); 
    nn->weights_h4o = matrix_create(NN_OUTPUT_SIZE, h, h);
    
    // Biases
    nn->bias_h1 = (double*)calloc(h, sizeof(double));
    nn->bias_h2 = (double*)calloc(h, sizeof(double));
    nn->bias_h3 = (double*)calloc(h, sizeof(double)); 
    nn->bias_h4 = (double*)calloc(h, sizeof(double)); 
    nn->bias_o = (double*)calloc(NN_OUTPUT_SIZE, sizeof(double));

    // Intermediate Outputs
    nn->inputs = matrix_create(NN_INPUT_SIZE, 1, 0);
    nn->hidden1_outputs = matrix_create(h, 1, 0);
    nn->hidden2_outputs = matrix_create(h, 1, 0);
    nn->hidden3_outputs = matrix_create(h, 1, 0); 
    nn->hidden4_outputs = matrix_create(h, 1, 0); 
    nn->output_outputs = matrix_create(NN_OUTPUT_SIZE, 1, 0);
    nn->output_errors = matrix_create(NN_OUTPUT_SIZE, 1, 0);
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
    matrix_free(nn->output_errors);
}

void nn_forward(NeuralNetwork* nn, const double* input_array, double* output_array) {
    Matrix inputs_m = array_to_matrix(input_array, NN_INPUT_SIZE);
    int h = NN_HIDDEN_SIZE;
    
    // 1. Input -> H1 (ReLU)
    Matrix h1_in_m = matrix_dot(nn->weights_ih1, inputs_m);
    for (int i = 0; i < h; i++) h1_in_m.data[i][0] += nn->bias_h1[i];
    Matrix h1_out_m = matrix_map(h1_in_m, relu_activation);

    // 2. H1 -> H2 (ReLU)
    Matrix h2_in_m = matrix_dot(nn->weights_h1h2, h1_out_m);
    for (int i = 0; i < h; i++) h2_in_m.data[i][0] += nn->bias_h2[i];
    Matrix h2_out_m = matrix_map(h2_in_m, relu_activation);

    // 3. H2 -> H3 (ReLU)
    Matrix h3_in_m = matrix_dot(nn->weights_h2h3, h2_out_m);
    for (int i = 0; i < h; i++) h3_in_m.data[i][0] += nn->bias_h3[i];
    Matrix h3_out_m = matrix_map(h3_in_m, relu_activation);
    
    // 4. H3 -> H4 (ReLU)
    Matrix h4_in_m = matrix_dot(nn->weights_h3h4, h3_out_m);
    for (int i = 0; i < h; i++) h4_in_m.data[i][0] += nn->bias_h4[i];
    Matrix h4_out_m = matrix_map(h4_in_m, relu_activation);

    // 5. H4 -> Output (Softmax)
    Matrix output_in_m = matrix_dot(nn->weights_h4o, h4_out_m);
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) output_in_m.data[i][0] += nn->bias_o[i];
    
    // Apply Softmax on each of the 10 character blocks
    Matrix output_out_m = matrix_create(NN_OUTPUT_SIZE, 1, 0);
    for (int i = 0; i < MAX_WORD_LEN; i++) {
        int start_idx = i * ALPHABET_SIZE;
        double block[ALPHABET_SIZE];
        for(int j = 0; j < ALPHABET_SIZE; j++) block[j] = output_in_m.data[start_idx + j][0];
        
        softmax(block); // Softmax function modifies the block array in place
        
        for(int j = 0; j < ALPHABET_SIZE; j++) output_out_m.data[start_idx + j][0] = block[j];
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
    double total_cce_loss = 0.0;
    Matrix targets_m = array_to_matrix(target_array, NN_OUTPUT_SIZE);
    int h = NN_HIDDEN_SIZE;
    
    // --- 1. Output Layer Error (Softmax + CCE) ---
    // Error is (Output - Target) for CCE with Softmax. 
    Matrix output_errors_m = matrix_add_subtract(nn->output_outputs, targets_m, false);
    matrix_copy_in(nn->output_errors, output_errors_m); // Store for H4 error calculation

    // Calculate CCE loss
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) { 
        if (targets_m.data[i][0] == 1.0) { // Only check for the target element
            // Add a small epsilon for numerical stability if output is near 0
            double p = nn->output_outputs.data[i][0];
            if (p < 1e-12) p = 1e-12;
            total_cce_loss -= log(p); 
        }
    }
    
    // Update Weights H4O and Bias O
    Matrix h4_out_t_m = matrix_transpose(nn->hidden4_outputs);
    Matrix delta_h4o_m = matrix_dot(output_errors_m, h4_out_t_m);
    Matrix scaled_delta_h4o_m = matrix_multiply_scalar(delta_h4o_m, nn->lr);
    Matrix new_h4o_m = matrix_add_subtract(nn->weights_h4o, scaled_delta_h4o_m, false);
    matrix_copy_in(nn->weights_h4o, new_h4o_m);
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) { nn->bias_o[i] -= output_errors_m.data[i][0] * nn->lr; }

    // --- 2. Hidden Layer 4 ---
    Matrix weights_h4o_t_m = matrix_transpose(nn->weights_h4o);
    Matrix h4_errors_m = matrix_dot(weights_h4o_t_m, output_errors_m);
    Matrix h4_d_m = matrix_map(nn->hidden4_outputs, relu_derivative); // ReLU derivative
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
    Matrix h3_d_m = matrix_map(nn->hidden3_outputs, relu_derivative); // ReLU derivative
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
    Matrix h2_d_m = matrix_map(nn->hidden2_outputs, relu_derivative); // ReLU derivative
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
    Matrix h1_d_m = matrix_map(nn->hidden1_outputs, relu_derivative); // ReLU derivative
    Matrix h1_gradients_m = matrix_multiply_elem(h1_errors_m, h1_d_m);
    
    // Update Weights IH1 and Bias H1
    Matrix inputs_t_m = matrix_transpose(nn->inputs);
    Matrix delta_ih1_m = matrix_dot(h1_gradients_m, inputs_t_m);
    Matrix new_ih1_m = matrix_add_subtract(nn->weights_ih1, matrix_multiply_scalar(delta_ih1_m, nn->lr), false);
    matrix_copy_in(nn->weights_ih1, new_ih1_m);
    for (int i = 0; i < h; i++) { nn->bias_h1[i] -= h1_gradients_m.data[i][0] * nn->lr; }

    // Final Cleanup (Temporary matrices must be freed in a complete implementation)
    matrix_free(targets_m); 
    matrix_free(output_errors_m);
    matrix_free(h4_out_t_m); matrix_free(delta_h4o_m); matrix_free(scaled_delta_h4o_m); matrix_free(new_h4o_m); 
    matrix_free(weights_h4o_t_m); matrix_free(h4_errors_m); matrix_free(h4_d_m); matrix_free(h4_gradients_m);
    matrix_free(h3_out_t_m); matrix_free(delta_h2h3_m); matrix_free(new_h2h3_m); 
    matrix_free(weights_h3h4_t_m); matrix_free(h3_errors_m); matrix_free(h3_d_m); matrix_free(h3_gradients_m);
    matrix_free(h2_out_t_m); matrix_free(delta_h1h2_m); matrix_free(new_h1h2_m); 
    matrix_free(weights_h2h3_t_m); matrix_free(h2_errors_m); matrix_free(h2_d_m); matrix_free(h2_gradients_m);
    matrix_free(h1_out_t_m); matrix_free(delta_ih1_m); matrix_free(new_ih1_m); 
    matrix_free(weights_h1h2_t_m); matrix_free(h1_errors_m); matrix_free(h1_d_m); matrix_free(h1_gradients_m);
    matrix_free(inputs_t_m);
    
    // Return average CCE loss per character position
    return total_cce_loss / MAX_WORD_LEN;
}

// Sequential Batch Training Function
double train_sequential_batch(NeuralNetwork* nn, int* word_indices, double* bp_time) {
    clock_t start_bp = clock();
    double total_loss = 0.0;
    double input_arr[NN_INPUT_SIZE]; 
    double target_arr[NN_OUTPUT_SIZE];
    double output_arr[NN_OUTPUT_SIZE];
    
    char work_word[MAX_WORD_LEN + 2]; // Needs space for insertion

    // --- 1. Train on Misspelled Words ---
    for (int i = 0; i < NUM_TRAIN_WORDS; i++) {
        const char* correct_word = ENGLISH_WORDS[word_indices[i]];
        
        // Encode the correct word as the target array
        encode_word_ohe(correct_word, target_arr);
        
        for (int j = 0; j < MISSPELLED_PER_WORD; j++) {
            // Generate a misspelled word (SUB/DEL/INS/TRA)
            generate_misspelled_word(correct_word, work_word);
            
            // Encode the misspelled word as the input array
            encode_word_ohe(work_word, input_arr); 

            // Forward and Backward pass
            nn_forward(nn, input_arr, output_arr);
            total_loss += nn_backward(nn, target_arr);
        }
    }
    
    // --- 2. Train on Correctly Spelled Words (Identity Mapping) ---
    for (int i = 0; i < CORRECT_SAMPLES_IN_BATCH; i++) {
        // Select a random word from the training set
        int word_idx = word_indices[rand() % NUM_TRAIN_WORDS];
        const char* correct_word = ENGLISH_WORDS[word_idx];
        
        // Input is the correct word, target is the correct word
        encode_word_ohe(correct_word, target_arr);
        encode_word_ohe(correct_word, input_arr); 

        // Forward and Backward pass
        nn_forward(nn, input_arr, output_arr);
        total_loss += nn_backward(nn, target_arr);
    }
    
    clock_t end_bp = clock();
    *bp_time = ((double)(end_bp - start_bp)) / CLOCKS_PER_SEC; 

    return total_loss / BATCH_SIZE;
}

// Testing function: Runs 512 words * 16 misspellings = 8192 tests
double test_network(NeuralNetwork* nn, double* test_time, bool verbose, int* fixed_count) {
    clock_t start_test = clock();
    int correct_words = 0;
    double input_arr[NN_INPUT_SIZE]; 
    double output_arr[NN_OUTPUT_SIZE];
    
    char misspelled_word[MAX_WORD_LEN + 2]; // Allow for insertion errors on the word generator
    char decoded_output[MAX_WORD_LEN + 1];

    // Iterate through all hardcoded words
    for (int i = 0; i < NUM_TEST_WORDS; i++) {
        const char* correct_word = ENGLISH_WORDS[i];
        
        // Generate 16 misspellings for each word
        for (int j = 0; j < MISSPELLED_PER_TEST_WORD; j++) {
            
            // Generate a test misspelling (SUB/DEL/INS/TRA)
            generate_misspelled_word(correct_word, misspelled_word);
            
            // Encode and run forward pass
            encode_word_ohe(misspelled_word, input_arr); 
            nn_forward(nn, input_arr, output_arr);
            
            // Decode the output
            decode_word_ohe(output_arr, decoded_output);
            
            // Check for correctness (word-level match)
            if (strcmp(correct_word, decoded_output) == 0) { 
                correct_words++; 
            } else if (verbose) {
                // Only log failures in final verbose test
                printf("  [Test Fail] Input: '%s' -> Output: '%s' (Target: '%s')\n", 
                       misspelled_word, decoded_output, correct_word);
            }
        }
    }
    
    clock_t end_test = clock();
    *fixed_count = correct_words;
    *test_time = ((double)(end_test - start_test)) / CLOCKS_PER_SEC;
    
    return ((double)correct_words / TOTAL_TESTS) * 100.0;
}


// --- SVG Utility Functions (Minor update for H-size) ---

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
    
    // Adjust Y-spacing for larger I/O layers (NN_INPUT_SIZE=270, NN_OUTPUT_SIZE=270)
    int node_radius_small = 2; // Smaller radius for the large layers
    int node_spacing_small = 2;

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
        
        int r1 = (i == 0) ? node_radius_small : NODE_RADIUS;
        int s1 = (i == 0) ? node_spacing_small : NODE_SPACING;
        int r2 = (i + 1 == num_layers - 1) ? node_radius_small : NODE_RADIUS;
        int s2 = (i + 1 == num_layers - 1) ? node_spacing_small : NODE_SPACING;

        // Calculate the starting Y coordinate to center the layer
        int y_offset_prev = SVG_HEIGHT / 2 - (prev_size * (r1 + s1)) / 2;
        int y_offset_curr = SVG_HEIGHT / 2 - (curr_size * (r2 + s2)) / 2;

        for (int j = 0; j < curr_size; j += (i > 0 && i < 4) ? 1 : 10) { // Current Layer (sample less for large layers)
            for (int k = 0; k < prev_size; k += (i == 0) ? 10 : 1) { // Previous Layer
                double weight = weights->data[j][k];
                double abs_weight = fabs(weight);
                const char* css_class = (weight >= 0) ? "positive" : "negative";
                
                // Scale line width based on absolute weight
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

    // --- Draw Nodes (Neurons) ---
    for (int i = 0; i < num_layers; i++) {
        int size = layer_sizes[i];
        int r = (i == 0 || i == num_layers - 1) ? node_radius_small : NODE_RADIUS;
        int s = (i == 0 || i == num_layers - 1) ? node_spacing_small : NODE_SPACING;

        // Calculate the starting Y coordinate to center the layer
        int y_offset = SVG_HEIGHT / 2 - (size * (r + s)) / 2;
        
        int cx = x_coords[i]; 

        for (int j = 0; j < size; j += (i == 0 || i == num_layers - 1) ? 10 : 1) { // Sample less for large layers
            int cy = y_offset + j * (r * 2 + s) + r;
            
            const char* fill_color = "#ffffff"; // Default white
            if (i == 0) fill_color = "#cccccc"; // Input: Grey
            
            // Draw circle
            svg_add_string("<circle cx=\"%d\" cy=\"%d\" r=\"%d\" fill=\"%s\" class=\"neuron\" />\n", 
                           cx, cy, r, fill_color);

            // Add Bias Labels (only for hidden layers)
            if (i > 0 && i < num_layers - 1 && j % 5 == 0) { // Only sample biases
                double bias = 0.0;
                if (i == 1) bias = nn->bias_h1[j];
                else if (i == 2) bias = nn->bias_h2[j];
                else if (i == 3) bias = nn->bias_h3[j];
                else if (i == 4) bias = nn->bias_h4[j];
                
                // Add a small rectangle or text for the bias
                svg_add_string("<text x=\"%d\" y=\"%d\" font-size=\"8\" fill=\"#666666\" class=\"bias\">%.2f</text>\n",
                               cx + r + 2, cy + 3, bias);
            }
        }
        
        // Add Layer Labels
        const char* label = "";
        if (i == 0) label = "INPUT (270 OHE)";
        else if (i == 1) label = "HIDDEN 1 (64)";
        else if (i == 2) label = "HIDDEN 2 (64)";
        else if (i == 3) label = "HIDDEN 3 (64)";
        else if (i == 4) label = "HIDDEN 4 (64)";
        else if (i == 5) label = "OUTPUT (270 OHE)";
        
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
    
    // --- Initialize NN ---
    NeuralNetwork nn;
    nn_init(&nn);

    printf("Architecture: Input(%d OHE) -> H1(%d ReLU) -> H2(%d ReLU) -> H3(%d ReLU) -> H4(%d ReLU) -> Output(%d Softmax)\n", NN_INPUT_SIZE, NN_HIDDEN_SIZE, NN_HIDDEN_SIZE, NN_HIDDEN_SIZE, NN_HIDDEN_SIZE, NN_OUTPUT_SIZE);
    printf("Training Batch Size: ~%d examples/batch (%.0f%% correct words).\n", BATCH_SIZE, CORRECT_WORD_PROPORTION * 100);
    printf("Test Set Size: %d words * %d misspellings = %d examples/test.\n", NUM_TEST_WORDS, MISSPELLED_PER_TEST_WORD, TOTAL_TESTS);
    printf("Maximum Training Time: %.0f seconds. Testing occurs every %.0f seconds.\n", MAX_TRAINING_SECONDS, TEST_INTERVAL_SECONDS); 
    printf("----------------------------------------------------------------------------------------------------------------------------------\n");
    printf("Time (sec) | Batches Run | CCE Loss (Last Batch) | Spelling Errors Fixed (Absolute/%%) | Test Time (sec)\n");
    printf("----------------------------------------------------------------------------------------------------------------------------------\n");
    fflush(stdout);

    time_t start_time = time(NULL);
    time_t last_test_time = start_time - TEST_INTERVAL_SECONDS; // Force immediate first test
    double current_success_rate = 0.0;
    int fixed_count = 0;
    double last_batch_cce_loss = 0.0;
    int next_milestone_percent = 10;
    int total_batches_run = 0;
    int max_epochs = 10000; 

    // Indices to select 64 random words from the vocabulary for the current batch
    int word_indices[NUM_TRAIN_WORDS];

    for (int epoch = 0; epoch < max_epochs; epoch++) {
        bool time_limit_reached = false;
        
        // Update Learning Rate with time-based decay
        double current_time_sec = difftime(time(NULL), start_time);
        nn.lr = NN_INITIAL_LEARNING_RATE / (1.0 + NN_LR_DECAY_RATE * current_time_sec);
        
        // --- Randomly select words for the batch ---
        for (int i = 0; i < NUM_TRAIN_WORDS; i++) {
            word_indices[i] = rand() % VOCAB_SIZE;
        }

        // --- Training Step (One batch) ---
        double bp_time = 0.0;
        last_batch_cce_loss = train_sequential_batch(&nn, word_indices, &bp_time);
        total_batches_run++;

        // --- Testing Step (Every 10 seconds) ---
        time_t current_time = time(NULL);
        if (difftime(current_time, last_test_time) >= TEST_INTERVAL_SECONDS || total_batches_run == 1) {
            double test_time = 0.0;
            current_success_rate = test_network(&nn, &test_time, false, &fixed_count); 
            last_test_time = current_time;
            
            // Log stats after test (Only print happens here)
            printf("%-10.0f | %-11d | %-21.8f | %-16d/%-6d (%.2f%%) | %-15.4f\n", 
                   current_time_sec,
                   total_batches_run, 
                   last_batch_cce_loss, 
                   fixed_count, TOTAL_TESTS, current_success_rate, 
                   test_time);
            fflush(stdout);
            
            // Log milestones
            while (current_success_rate >= next_milestone_percent) {
                printf("--- MILESTONE REACHED --- Correctness: %.2f%% (%d/%d) (LR: %.6f)\n", current_success_rate, fixed_count, TOTAL_TESTS, nn.lr);
                fflush(stdout);
                
                if (next_milestone_percent == 90) { next_milestone_percent = 95; } 
                else if (next_milestone_percent >= 95) { break; } 
                else { next_milestone_percent += 10; }
            }
        } 

        // Check time limit or goal
        if (current_time_sec >= MAX_TRAINING_SECONDS) {
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

    // --- POST-TRAINING SVG SAVE ---
    save_network_as_svg(&nn);
    printf("Final network SVG saved to %s.\n", SVG_FILENAME);

    // --- Cleanup ---
    nn_free(&nn);

    return 0;
}
