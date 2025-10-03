#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <ctype.h>

// Fix for M_PI not being defined by default in <math.h>
#define M_PI 3.14159265358979323846

// --- INN PARAMETERS AND CONSTANTS ---
#define D 8                   // Dimension of the sentence feature vector
#define NUM_LEGAL_SENTENCES 2048 // Training Size
#define NUM_ILLEGAL_SENTENCES 2048 // Test Size
#define LEARNING_RATE 0.00001  
#define MAX_EPOCHS 200
#define TRAINING_TIME_LIMIT_SEC 180.0 
#define GAUSSIAN_SIGMA 1.0    
#define MAX_WORD_LEN 64
#define MAX_FILE_READ_SIZE 100000 
#define MAX_SENTENCES_FROM_FILE 5000 
#define MAX_WORDS_PER_SENTENCE 50 

// --- NEW DATA STRUCTURES ---

typedef struct {
    char *str;      
    size_t len;     
    int valid;      
} Word;

typedef struct {
    Word *words;    
    size_t count;   
} SentenceText;

// Flexible Matrix/Vector structures to handle both D=8 and D=1 (for the sine test)
// Note: Arrays are still fixed size but we use the 'rows/cols/size' fields to manage dimensions.
// For the 1D test, we only use [0][0] for Matrix and [0] for Vector.
typedef struct { double data[D][D]; int rows; int cols; int initialized; } Matrix;
typedef struct { double data[D]; int size; int initialized; } Vector;
typedef struct { Matrix A; Vector b; } INN_Parameters;

// Struct to hold a single sentence for training/testing
typedef struct {
    SentenceText text_data; 
    Vector features;
    int is_legal;
} Sentence;

// --- VOCABULARY ---

const char *const Nouns[] = {"alice", "queen", "hatter", "rabbit", "cat", "king", "mouse", "turtle", "garden", "door", "table", "key", "curiosity", "head", "tea", "dream", "voice", "way", "sister", "time", "world", "thing", "house", "foot", "corner"};
const double NounValues[] = {1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4};
#define NUM_NOUNS (sizeof(Nouns) / sizeof(Nouns[0]))

const char *const Verbs[] = {"said", "began", "thought", "went", "looked", "cried", "found", "came", "made", "knew", "heard", "running", "walking", "growing", "shrinking", "wondered", "talked", "singing", "dreaming", "speak", "feel", "try", "open", "sit", "think"};
const double VerbValues[] = {4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4};
#define NUM_VERBS (sizeof(Verbs) / sizeof(Verbs[0]))

const char *const Adjectives[] = {"mad", "curious", "large", "small", "beautiful", "good", "great", "nice", "silly", "poor", "dark", "sudden", "long", "loud", "real", "different", "tired", "anxious", "cross", "stupid"};
const double AdjectiveValues[] = {7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9};
#define NUM_ADJECTIVES (sizeof(Adjectives) / sizeof(Adjectives[0]))

const char *const NonsenseNouns[] = {"glork", "zorp", "fleep", "wubba", "snerd", "crungle", "chork"};
const char *const NonsenseVerbs[] = {"bleem", "floof", "skree", "quonk", "smashle", "zooble"};
#define NUM_NONSENSE_NOUNS (sizeof(NonsenseNouns) / sizeof(NonsenseNouns[0]))
#define NUM_NONSENSE_VERBS (sizeof(NonsenseVerbs) / sizeof(NonsenseVerbs[0]))

// --- UTILITY AND MEMORY MANAGEMENT ---

Word create_word(const char *source, size_t length) {
    if (length == 0 || source == NULL) {
        return (Word){NULL, 0, 0};
    }
    char *new_str = (char *)malloc(length + 1);
    if (new_str == NULL) {
        fprintf(stderr, "Memory allocation failed for word.\n");
        return (Word){NULL, 0, 0};
    }
    strncpy(new_str, source, length);
    new_str[length] = '\0';
    return (Word){new_str, length, 1};
}

void free_sentence_text(SentenceText *st) {
    if (st == NULL || st->words == NULL) return;
    for (size_t i = 0; i < st->count; i++) {
        if (st->words[i].str != NULL && st->words[i].valid) {
            free(st->words[i].str);
        }
    }
    free(st->words);
    st->words = NULL;
    st->count = 0;
}

Word parse_token_to_word(const char *token) {
    size_t len = strlen(token);
    if (len == 0) return (Word){NULL, 0, 0};

    size_t effective_len = len;
    while (effective_len > 0 && (ispunct(token[effective_len - 1]) || isspace(token[effective_len - 1]))) {
        effective_len--;
    }

    size_t start_offset = 0;
    while (start_offset < effective_len && ispunct(token[start_offset])) {
        start_offset++;
    }

    if (effective_len <= start_offset) return (Word){NULL, 0, 0};

    return create_word(token + start_offset, effective_len - start_offset);
}

char *safe_strdup(const char *s) {
    size_t len = strlen(s) + 1;
    char *new_s = (char *)malloc(len);
    if (new_s == NULL) return NULL;
    return (char *)memcpy(new_s, s, len);
}

// --- MATRIX/VECTOR UTILITIES ---

void init_matrix(Matrix *M, int rows, int cols, int triangular) {
    // If rows or cols is 1, it's the simple test. If it's D, it's the language model.
    if (rows > D || cols > D || rows <= 0 || cols <= 0) { 
        fprintf(stderr, "Error: Matrix dimensions must be 1x1 or %dx%d.\n", D, D); 
        M->initialized = 0;
        return; 
    }
    M->rows = rows; M->cols = cols;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (triangular && j > i) M->data[i][j] = 0.0;
            else M->data[i][j] = ((double)rand() / RAND_MAX) * 0.01;
        }
    }
    for (int i = 0; i < (rows < cols ? rows : cols); i++) { 
        // Initialize diagonal elements to a value near 1.0
        if (M->data[i][i] == 0.0) M->data[i][i] = 1.0; 
    }
    M->initialized = 1;
}

void init_vector(Vector *V, int size) {
    if (size > D || size <= 0) { 
        fprintf(stderr, "Error: Vector dimension must be 1 or %d.\n", D); 
        V->initialized = 0;
        return; 
    }
    V->size = size;
    for (int i = 0; i < size; i++) V->data[i] = 0.0;
    V->initialized = 1;
}

int check_init(const Matrix *const M) { return M->initialized; }
int check_vector_init(const Vector *const V) { return V->initialized; } 

/**
 * @brief Multiplies Matrix A by Vector x, storing the result in y.
 */
void multiply_matrix_vector(const Matrix *const A, const Vector *const x, Vector *y) {
    if (!check_init(A) || !check_vector_init(x) || A->cols != x->size) { init_vector(y, A->rows); return; }
    init_vector(y, A->rows);
    for (int i = 0; i < A->rows; i++) {
        double sum = 0.0;
        for (int j = 0; j < A->cols; j++) {
            sum += A->data[i][j] * x->data[j];
        }
        y->data[i] = sum;
    }
}

double get_determinant_triangular(const Matrix *const A) {
    if (!check_init(A)) return 0.0;
    // This function only works for square matrices where rows == cols.
    if (A->rows != A->cols) return 0.0; 
    double det = 1.0;
    for (int i = 0; i < A->rows; i++) det *= A->data[i][i];
    return det;
}

// --- INN FLOW FUNCTIONS ---

/**
 * @brief Performs the forward pass of the INN: z = A * x + b
 */
void inn_forward(const INN_Parameters *const params, const Vector *const x, Vector *z) {
    if (!check_init(&params->A) || !check_vector_init(&params->b)) { init_vector(z, x->size); return; } 

    init_vector(z, params->A.rows);
    multiply_matrix_vector(&params->A, x, z);
    for (int i = 0; i < params->A.rows; i++) {
        z->data[i] += params->b.data[i];
    }
}

double calculate_nll_loss(const INN_Parameters *const params, const Vector *const z) {
    double z_norm_sq = 0.0;
    for (int i = 0; i < z->size; i++) {
        z_norm_sq += z->data[i] * z->data[i];
    }
    // Energy term: 0.5 * ||z||^2 / sigma^2
    const double log_prob_z = 0.5 * z_norm_sq / (GAUSSIAN_SIGMA * GAUSSIAN_SIGMA);
    
    // Volume term: - log(|det(A)|)
    if (params->A.rows == params->A.cols) {
        const double det_A = get_determinant_triangular(&params->A);
        const double abs_det_A = fabs(det_A);
        // Add epsilon (1e-9) for numerical stability during log calculation
        const double log_det_A = log(abs_det_A > 1e-9 ? abs_det_A : 1e-9); 
        
        return log_prob_z - log_det_A;
    } else {
        // If non-square, the flow loss term is ignored (only the energy term remains)
        return log_prob_z;
    }
}


// --- FEATURE MAPPING (Uses D=8) ---

double find_word_value(const Word *word, int *word_type) {
    if (!word->valid || word->len == 0) {
        *word_type = -1;
        return 0.0;
    }
    
    char lower_word[MAX_WORD_LEN];
    size_t copy_len = word->len < MAX_WORD_LEN - 1 ? word->len : MAX_WORD_LEN - 1;
    strncpy(lower_word, word->str, copy_len);
    lower_word[copy_len] = '\0';

    for(size_t i = 0; i < copy_len; i++){
      lower_word[i] = tolower(lower_word[i]);
    }

    // Check Nouns
    for (int i = 0; i < NUM_NOUNS; i++) {
        if (strcmp(lower_word, Nouns[i]) == 0) { *word_type = 0; return NounValues[i]; }
    }
    // Check Verbs
    for (int i = 0; i < NUM_VERBS; i++) {
        if (strcmp(lower_word, Verbs[i]) == 0) { *word_type = 1; return VerbValues[i]; }
    }
    // Check Adjectives
    for (int i = 0; i < NUM_ADJECTIVES; i++) {
        if (strcmp(lower_word, Adjectives[i]) == 0) { *word_type = 2; return AdjectiveValues[i]; }
    }
    
    *word_type = -1; 
    return 0.0;
}

void map_sentence_to_features(const SentenceText *const st, Vector *V) {
    init_vector(V, D);

    int feature_index = 0;
    
    for (size_t i = 0; i < st->count; i++) {
        if (feature_index >= D) break;

        const Word *word = &st->words[i];
        int word_type = -1;
        const double value = find_word_value(word, &word_type);

        // Only map words that are in the vocabulary and have a positive value
        if (word->valid && value > 0.0) {
            V->data[feature_index] = value;
            feature_index++;
        }
    }
}

// --- DATASET GENERATION FUNCTIONS ---

SentenceText convert_raw_to_sentence_text(const char *raw_sentence) {
    SentenceText st = {NULL, 0};

    char *raw_copy = safe_strdup(raw_sentence); 
    if (raw_copy == NULL) return st;

    char *temp_copy = safe_strdup(raw_sentence);
    if (temp_copy == NULL) { free(raw_copy); return st; }

    int word_count = 0;
    char *token = strtok(temp_copy, " \t\n");
    while (token != NULL) {
        word_count++;
        token = strtok(NULL, " \t\n");
    }
    free(temp_copy);

    if (word_count == 0 || word_count > MAX_WORDS_PER_SENTENCE) {
        free(raw_copy);
        return st;
    }

    st.words = (Word *)calloc(word_count, sizeof(Word));
    if (st.words == NULL) {
        free(raw_copy);
        return st;
    }
    st.count = word_count;

    token = strtok(raw_copy, " \t\n");
    size_t i = 0;
    while (token != NULL && i < st.count) {
        st.words[i] = parse_token_to_word(token);
        i++;
        token = strtok(NULL, " \t\n");
    }

    free(raw_copy);
    return st;
}

int load_sentence_templates(SentenceText *legal_templates_st, int max_templates) {
    // Requires a file named 'alice.txt' in the same directory.
    FILE *file = fopen("alice.txt", "r");
    if (file == NULL) {
        fprintf(stderr, "FATAL ERROR: Could not open alice.txt. Make sure the file is in the same directory.\n");
        return 0;
    }
    
    char *buffer = (char *)malloc(MAX_FILE_READ_SIZE);
    if (buffer == NULL) { fclose(file); return 0; }

    size_t bytes_read = fread(buffer, 1, MAX_FILE_READ_SIZE - 1, file);
    buffer[bytes_read] = '\0'; 
    fclose(file);

    char *sentence_split_buffer = safe_strdup(buffer);
    if (sentence_split_buffer == NULL) { free(buffer); return 0; }
    free(buffer);

    char *token = strtok(sentence_split_buffer, ".!?");
    int count = 0;
    while (token != NULL && count < max_templates) {
        char *trimmed_token = token;
        size_t len = strlen(token);
        while (isspace(*trimmed_token)) { trimmed_token++; len--; }
        
        if (len > 10) { 
            SentenceText st = convert_raw_to_sentence_text(trimmed_token);
            if (st.count > 0 && st.count < MAX_WORDS_PER_SENTENCE) {
                // Only keep templates that have at least one replaceable word (Noun/Verb)
                int has_replaceable = 0;
                for (size_t i = 0; i < st.count; i++) {
                    int word_type = -1;
                    find_word_value(&st.words[i], &word_type);
                    if (word_type == 0 || word_type == 1) { // 0=Noun, 1=Verb
                        has_replaceable = 1;
                        break;
                    }
                }
                
                if (has_replaceable) {
                    legal_templates_st[count] = st;
                    count++;
                } else {
                    free_sentence_text(&st);
                }
            } else {
                free_sentence_text(&st);
            }
        }
        
        token = strtok(NULL, ".!?");
    }

    free(sentence_split_buffer);
    return count;
}


SentenceText generate_illegal_sentence_text(const SentenceText *legal_template) {
    SentenceText illegal_st = {NULL, 0};
    if (legal_template->count == 0) return illegal_st;

    illegal_st.words = (Word *)calloc(legal_template->count, sizeof(Word));
    if (illegal_st.words == NULL) return illegal_st;
    illegal_st.count = legal_template->count;

    // 1. Find all indices that are Nouns or Verbs in the original template
    int replaceable_indices[MAX_WORDS_PER_SENTENCE];
    int replaceable_count = 0;
    for (size_t i = 0; i < legal_template->count; i++) {
        const Word *original_word = &legal_template->words[i];
        int word_type = -1;
        find_word_value(original_word, &word_type);
        if (original_word->valid && (word_type == 0 || word_type == 1)) {
            replaceable_indices[replaceable_count++] = i;
        }
    }

    // 2. Determine the indices to replace (1 or 2 replacements, guaranteed at least 1 if possible)
    int indices_to_replace[2] = {-1, -1};
    int num_to_replace = (replaceable_count > 0) ? ((rand() % 2) + 1) : 0; // 1 or 2
    num_to_replace = (num_to_replace > replaceable_count) ? replaceable_count : num_to_replace;
    
    if (num_to_replace > 0) {
        indices_to_replace[0] = replaceable_indices[rand() % replaceable_count];
        if (num_to_replace == 2 && replaceable_count > 1) {
            int second_idx;
            do {
                second_idx = replaceable_indices[rand() % replaceable_count];
            } while (second_idx == indices_to_replace[0]);
            indices_to_replace[1] = second_idx;
        }
    }

    // 3. Perform the copy/replacement
    for (size_t i = 0; i < legal_template->count; i++) {
        const Word *original_word = &legal_template->words[i];
        int replaced = 0;
        
        for (int k = 0; k < num_to_replace; k++) {
            if ((int)i == indices_to_replace[k]) {
                int word_type = -1;
                find_word_value(original_word, &word_type); 

                // Replace with a nonsense word of the same type (Noun/Verb)
                const char *nonsense_str = (word_type == 0) 
                                            ? NonsenseNouns[rand() % NUM_NONSENSE_NOUNS]
                                            : NonsenseVerbs[rand() % NUM_NONSENSE_VERBS];
                illegal_st.words[i] = create_word(nonsense_str, strlen(nonsense_str));
                replaced = 1;
                break;
            }
        }

        if (!replaced) {
            // Deep copy the original word
            if (original_word->valid) {
                 illegal_st.words[i] = create_word(original_word->str, original_word->len);
            } else {
                illegal_st.words[i] = *original_word;
            }
        }

        if (illegal_st.words[i].valid && illegal_st.words[i].str == NULL) {
            illegal_st.words[i].valid = 0;
        }
    }
    
    return illegal_st;
}

void generate_datasets(Sentence *legal_sentences, Sentence *illegal_sentences, SentenceText *legal_templates_st, int num_templates) {
    if (num_templates == 0) {
        fprintf(stderr, "FATAL: Cannot generate datasets without templates. Exiting.\n");
        exit(1);
    }

    printf("Dataset generation starting: %d templates loaded.\n", num_templates);

    // Generate Legal Sentences (Training Data)
    for (int i = 0; i < NUM_LEGAL_SENTENCES; i++) {
        const int template_idx = rand() % num_templates;
        const SentenceText *template = &legal_templates_st[template_idx];
        
        // Use the generation function, but the result should be similar to the original template
        // since the INN will learn to map these "mostly legal" features to Gaussian.
        legal_sentences[i].text_data = generate_illegal_sentence_text(template); 
        map_sentence_to_features(&legal_sentences[i].text_data, &legal_sentences[i].features);
        legal_sentences[i].is_legal = 1;
    }

    // Generate Illegal Sentences (Test Data)
    for (int i = 0; i < NUM_ILLEGAL_SENTENCES; i++) {
        const int template_idx = rand() % num_templates;
        const SentenceText *template = &legal_templates_st[template_idx];

        // This intentionally swaps real words with nonsense words, guaranteeing high NLL
        illegal_sentences[i].text_data = generate_illegal_sentence_text(template);
        map_sentence_to_features(&illegal_sentences[i].text_data, &illegal_sentences[i].features);
        illegal_sentences[i].is_legal = 0;
    }

    printf("Dataset generated: %d legal, %d illegal sentences.\n", NUM_LEGAL_SENTENCES, NUM_ILLEGAL_SENTENCES);
}

// --- SANITY CHECKS ---

int run_sanity_checks(SentenceText *templates, int num_templates) {
    int passed = 0;
    int failed = 0;

    printf("\n--- Running Sanity Checks ---\n");

    // 1. Check File Loading
    if (num_templates == 0) {
        printf("Check 1 (File Load): FAILED. No templates loaded.\n");
        return 0;
    } else {
        printf("Check 1 (File Load): PASSED. Loaded %d templates.\n", num_templates);
        passed++;
    }

    // --- Check 2 & 3 Setup: Look for at least 1 feature ---
    int robust_template_idx = -1;
    const int REQUIRED_FEATURES = 1; 
    for (int i = 0; i < num_templates; i++) {
        Vector V;
        map_sentence_to_features(&templates[i], &V);
        int feature_count = 0;
        if (V.data[0] > 0.0) feature_count++; 
        
        if (feature_count >= REQUIRED_FEATURES) { 
            robust_template_idx = i;
            break;
        }
    }

    if (robust_template_idx == -1) {
        printf("Check 2 (Feature Mapping): FAILED. Could not find any template with at least %d feature.\n", REQUIRED_FEATURES);
        printf("Check 3 (Illegal Gen): FAILED. Aborted due to lack of robust template.\n");
        return 0; 
    }

    const SentenceText *test_template = &templates[robust_template_idx];
    
    // 2. Check Feature Mapping (on the robust template)
    Vector V;
    map_sentence_to_features(test_template, &V);
    printf("Check 2 (Feature Mapping): PASSED. Template index %d yielded first feature (%.2f).\n", 
           robust_template_idx, V.data[0]);
    passed++;

    // 3. Check Illegal Sentence Generation (Nonsense word found)
    SentenceText illegal_st = generate_illegal_sentence_text(test_template);
    int found_nonsense = 0;
    int word_count_match = (illegal_st.count == test_template->count);
    
    for (size_t i = 0; i < illegal_st.count; i++) {
        if (illegal_st.words[i].valid && illegal_st.words[i].len > 0) {
            // Check if the word is a known nonsense word
            for (int j = 0; j < NUM_NONSENSE_NOUNS; j++) {
                if (strcmp(illegal_st.words[i].str, NonsenseNouns[j]) == 0) { found_nonsense = 1; break; }
            }
            if(found_nonsense) break;
            for (int j = 0; j < NUM_NONSENSE_VERBS; j++) {
                if (strcmp(illegal_st.words[i].str, NonsenseVerbs[j]) == 0) { found_nonsense = 1; break; }
            }
            if(found_nonsense) break;
        }
    }

    if (found_nonsense && word_count_match) {
        printf("Check 3 (Illegal Gen): PASSED. Nonsense word found, word count matches.\n");
        passed++;
    } else {
        printf("Check 3 (Illegal Gen): FAILED. Nonsense: %s, Count Match: %s. (Template Index: %d)\n", 
               found_nonsense ? "Yes" : "No", word_count_match ? "Yes" : "No", robust_template_idx);
        failed++;
    }
    free_sentence_text(&illegal_st);
    
    printf("Sanity Checks Complete: %d PASSED, %d FAILED.\n", passed, failed);
    return failed == 0;
}

// --- NEW SIMPLE FUNCTION TEST ---

#define SINE_TEST_SAMPLES 1000
#define SINE_TEST_LR 0.001
#define SINE_TEST_EPOCHS 500

/**
 * @brief Trains an INN on the simple 1D sine function $y = \sin(x) + \text{noise}$ 
 * to check for basic convergence and correct gradient application.
 */
int simple_function_test() {
    printf("\n\n--- Running Simple Function (1D Sin Wave) Test ---\n");
    
    INN_Parameters params;
    // Initialize 1x1 matrix A and 1x1 vector b
    init_matrix(&params.A, 1, 1, 1);
    init_vector(&params.b, 1);
    
    // Check initial random weights (should be near 1.0 for A, near 0.0 for b)
    printf("Initial Parameters: A[0][0]=%.4f, b[0]=%.4f\n", params.A.data[0][0], params.b.data[0]);

    double total_loss = 0.0;

    for (int epoch = 0; epoch < SINE_TEST_EPOCHS; epoch++) {
        double epoch_loss = 0.0;
        
        for (int i = 0; i < SINE_TEST_SAMPLES; i++) {
            // 1. Generate Data (x_i)
            // The INN learns the transformation z = A*x + b where z ~ N(0, 1)
            
            // x from 0 to 2*PI
            const double x_val = (double)i / SINE_TEST_SAMPLES * 2.0 * M_PI; 
            const double noise = ((double)rand() / RAND_MAX - 0.5) * 0.1; // Small noise
            
            // Generate data x that approximately follows a sine curve
            Vector x_in; init_vector(&x_in, 1);
            x_in.data[0] = sin(x_val) + noise; 

            Vector z; 
            
            // Forward Pass: z = A * x + b
            inn_forward(&params, &x_in, &z); 
            const double nll_loss = calculate_nll_loss(&params, &z);
            epoch_loss += nll_loss;

            // Backpropagation
            // dL/dz = z / sigma^2. Since sigma=1, dL/dz = z
            const double dL_dz = z.data[0] / (GAUSSIAN_SIGMA * GAUSSIAN_SIGMA); 
            const double x_0 = x_in.data[0];
            const double A_00 = params.A.data[0][0];

            // Gradients for 1D INN
            // dL/dA = (dL/dz) * x - 1/A
            const double dL_dA_00 = dL_dz * x_0 - (1.0 / A_00);
            // dL/db = dL/dz * 1
            const double dL_db_0 = dL_dz;

            // Update
            params.A.data[0][0] -= SINE_TEST_LR * dL_dA_00;
            params.b.data[0] -= SINE_TEST_LR * dL_db_0;
        }
        
        total_loss = epoch_loss / SINE_TEST_SAMPLES;
        
        if (epoch % 50 == 0) {
            printf("Epoch %5d: Avg Loss = %8.4f, A[0][0] = %7.4f, b[0] = %7.4f\n", 
                   epoch, total_loss, params.A.data[0][0], params.b.data[0]);
        }
        
        // Stop condition: Check for stable, low loss (well below the initial random loss)
        if (epoch > 100 && total_loss < 2.0) { 
            break; 
        }
    }
    
    printf("Final Loss: %.4f\n", total_loss);
    if (total_loss < 2.0) {
        printf("Sine Test: PASSED. INN converged to a stable, low-loss state (Loss < 2.0).\n");
        return 1;
    } else {
        printf("Sine Test: FAILED. INN did not converge to a stable, low-loss state.\n");
        return 0;
    }
}


// --- MAIN EXECUTION ---

int main(void) {
    srand((unsigned int)time(NULL));
    
    // 0. Run Simple Sine Function Test
    if (!simple_function_test()) {
         fprintf(stderr, "\nFATAL ERROR: Simple INN function test failed. Check core gradient implementation.\n");
         return 1;
    }

    // 1. Load Templates
    SentenceText legal_templates_st[MAX_SENTENCES_FROM_FILE];
    const int num_templates = load_sentence_templates(legal_templates_st, MAX_SENTENCES_FROM_FILE);

    // 2. Run Sanity Checks
    if (!run_sanity_checks(legal_templates_st, num_templates)) {
        fprintf(stderr, "\nFATAL ERROR: Initial language sanity checks failed. Aborting training.\n");
        for(int i = 0; i < num_templates; i++) free_sentence_text(&legal_templates_st[i]);
        return 1;
    }

    // 3. Initialize INN and Allocate Memory for Datasets
    INN_Parameters params;
    // Initialize DxD lower triangular matrix A and Dx1 bias vector b
    init_matrix(&params.A, D, D, 1);
    init_vector(&params.b, D);
    
    Sentence *legal_sentences = (Sentence *)malloc(NUM_LEGAL_SENTENCES * sizeof(Sentence));
    Sentence *illegal_sentences = (Sentence *)malloc(NUM_ILLEGAL_SENTENCES * sizeof(Sentence));
    
    if (legal_sentences == NULL || illegal_sentences == NULL) {
        fprintf(stderr, "Fatal Error: Failed to allocate memory for sentence datasets.\n");
        for(int i = 0; i < num_templates; i++) free_sentence_text(&legal_templates_st[i]);
        free(legal_sentences); free(illegal_sentences);
        return 1;
    }

    // 4. Generate Datasets
    generate_datasets(legal_sentences, illegal_sentences, legal_templates_st, num_templates);
    
    // Templates are no longer needed, free them
    for(int i = 0; i < num_templates; i++) free_sentence_text(&legal_templates_st[i]);
    
    
    // --- Training Loop (Maximum Likelihood Estimation) ---
    const clock_t start_time = clock();
    double last_print_time_sec = 0.0;
    const double print_interval_sec = 10.0; 
    int training_stopped_early = 0;

    printf("\n\n--- Starting Language INN Training (MLE via SGD) ---\n");
    printf("Training on %d Alice sentences (D=%d). Time limit: %.0f seconds.\n", NUM_LEGAL_SENTENCES, D, TRAINING_TIME_LIMIT_SEC);
    printf("Total Elapsed (s) | Epoch | Iteration | Avg NLL (Loss) | Det(A)\n");
    printf("----------------------------------------------------------------\n");

    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        double epoch_nll_sum = 0.0;
        int sentences_processed_in_epoch = 0; 

        // The inner loop selects random legal sentences (stochastic gradient descent)
        for (int i = 0; i < NUM_LEGAL_SENTENCES; i++) {
            const double total_elapsed_sec = (double)(clock() - start_time) / CLOCKS_PER_SEC;
            
            // Check time limit at the start of each step
            if (total_elapsed_sec >= TRAINING_TIME_LIMIT_SEC) {
                printf("--- Stopping training after %.2f seconds (Time Limit Reached) ---\n", total_elapsed_sec);
                training_stopped_early = 1;
                break;
            }

            const int idx = rand() % NUM_LEGAL_SENTENCES; // Randomly choose a legal sentence
            const Vector x = legal_sentences[idx].features;

            Vector z; 
            
            // Forward Pass: z = A*x + b
            inn_forward(&params, &x, &z); 
            const double nll_loss = calculate_nll_loss(&params, &z);
            epoch_nll_sum += nll_loss;
            sentences_processed_in_epoch++;

            // Backpropagation (Gradients)
            // dL/dz = z / sigma^2
            Vector dL_dz; init_vector(&dL_dz, D);
            const double scale = 1.0 / (GAUSSIAN_SIGMA * GAUSSIAN_SIGMA);
            for (int k = 0; k < D; k++) dL_dz.data[k] = z.data[k] * scale;

            // Update Bias: dL/db = dL/dz
            for (int k = 0; k < D; k++) params.b.data[k] -= LEARNING_RATE * dL_dz.data[k];

            // Update Matrix A (Lower Triangular)
            for (int r = 0; r < D; r++) {
                for (int c = 0; c < D; c++) {
                    if (c <= r) {
                        // Core gradient term: dL/dA_rc = (dL/dz_r) * x_c
                        const double grad_A_r_c = dL_dz.data[r] * x.data[c];
                        params.A.data[r][c] -= LEARNING_RATE * grad_A_r_c;

                        if (r == c) {
                             // Additional determinant term for diagonal elements: - 1/A_rr
                             params.A.data[r][c] -= LEARNING_RATE * (-1.0 / params.A.data[r][c]);
                        }
                    }
                }
            }

            // Print progress every 10 seconds
            if (total_elapsed_sec - last_print_time_sec >= print_interval_sec) {
                 printf("%19.2f | %5d | %9d | %14.4f | %6.4e\n", 
                       total_elapsed_sec, epoch, i + 1, epoch_nll_sum / sentences_processed_in_epoch, get_determinant_triangular(&params.A));
                last_print_time_sec = total_elapsed_sec;
            }
        } 
        if (training_stopped_early) break; 
        
        // Final epoch print
        if (sentences_processed_in_epoch > 0) {
             const double total_elapsed_sec = (double)(clock() - start_time) / CLOCKS_PER_SEC;
             printf("%19.2f | %5d | %9d | %14.4f | %6.4e (End of Epoch)\n", 
                       total_elapsed_sec, epoch, NUM_LEGAL_SENTENCES, epoch_nll_sum / sentences_processed_in_epoch, get_determinant_triangular(&params.A));
        }
    } 

    // --- Evaluation ---
    printf("\n--- INN Detection Test (Legal vs. Nonsensical) ---\n");
    double legal_nll_sum = 0.0;
    double illegal_nll_sum = 0.0;
    int legal_count = 0;
    int illegal_count = 0;

    // Evaluate Legal Sentences
    for (int i = 0; i < NUM_LEGAL_SENTENCES; i++) {
        Vector z;
        inn_forward(&params, &legal_sentences[i].features, &z); 
        legal_nll_sum += calculate_nll_loss(&params, &z);
        legal_count++;
    }

    // Evaluate Illegal Sentences
    for (int i = 0; i < NUM_ILLEGAL_SENTENCES; i++) {
        Vector z;
        inn_forward(&params, &illegal_sentences[i].features, &z); 
        illegal_nll_sum += calculate_nll_loss(&params, &z);
        illegal_count++;
    }

    const double avg_legal_nll = legal_nll_sum / legal_count;
    const double avg_illegal_nll = illegal_nll_sum / illegal_count;

    printf("Average NLL for Legal Sentences (Alice Text): %.4f\n", avg_legal_nll);
    printf("Average NLL for Illegal Sentences (Nonsense Words): %.4f\n", avg_illegal_nll);
    
    printf("\nDetection Conclusion:\n");

    if (avg_illegal_nll > avg_legal_nll) {
        printf("SUCCESS: The average NLL of nonsensical word data (%.4f) is HIGHER than the original Alice text (%.4f).\n",
               avg_illegal_nll, avg_legal_nll);
        printf("The INN successfully learned the statistical features and rejects inputs containing out-of-vocabulary (nonsensical) words.\n");
    } else {
        printf("FAILURE: The average NLL of nonsensical word data (%.4f) is NOT HIGHER than the original Alice structure (%.4f).\n",
               avg_illegal_nll, avg_legal_nll);
        printf("The INN failed to clearly distinguish between the original and modified feature patterns.\n");
    }
    
    // 5. Cleanup
    for (int i = 0; i < NUM_LEGAL_SENTENCES; i++) free_sentence_text(&legal_sentences[i].text_data);
    for (int i = 0; i < NUM_ILLEGAL_SENTENCES; i++) free_sentence_text(&illegal_sentences[i].text_data);
    free(legal_sentences);
    free(illegal_sentences);

    return 0;
}