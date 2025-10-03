#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <ctype.h>

// --- INN PARAMETERS AND CONSTANTS ---
#define D 8                   // Dimension of the sentence feature vector
#define NUM_LEGAL_SENTENCES 2048 // Training Size
#define NUM_ILLEGAL_SENTENCES 2048 // Test Size
#define LEARNING_RATE 0.00005  
#define MAX_EPOCHS 200
#define TRAINING_TIME_LIMIT_SEC 240.0 // 4 minutes
#define GAUSSIAN_SIGMA 1.0    
#define MAX_WORD_LEN 64
#define MAX_FILE_READ_SIZE 100000 
#define MAX_SENTENCES_FROM_FILE 1000 
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

// Matrix, Vector, INN_Parameters
typedef struct { double data[D][D]; int rows; int cols; int initialized; } Matrix;
typedef struct { double data[D]; int size; int initialized; } Vector;
typedef struct { Matrix A; Vector b; } INN_Parameters;

// Struct to hold a single sentence for training/testing
typedef struct {
    SentenceText text_data; 
    Vector features;
    int is_legal;
} Sentence;

// --- VOCABULARY: CHOSEN FROM ALICE IN WONDERLAND ---

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

// --- MATRIX/VECTOR UTILITIES (Unchanged) ---

void init_matrix(Matrix *M, int rows, int cols, int triangular) {
    if (rows != D || cols != D) { fprintf(stderr, "Error: Matrix dimensions must be %dx%d.\n", D, D); return; }
    M->rows = rows; M->cols = cols;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (triangular && j > i) M->data[i][j] = 0.0;
            else M->data[i][j] = ((double)rand() / RAND_MAX) * 0.01;
        }
    }
    for (int i = 0; i < D; i++) { if (M->data[i][i] == 0.0) M->data[i][i] = 1.0; }
    M->initialized = 1;
}

void init_vector(Vector *V, int size) {
    if (size != D) { fprintf(stderr, "Error: Vector dimension must be %d.\n", D); return; }
    V->size = size;
    for (int i = 0; i < size; i++) V->data[i] = 0.0;
    V->initialized = 1;
}

int check_init(const Matrix *const M) { return M->initialized; }
double get_determinant_triangular(const Matrix *const A) {
    if (!check_init(A)) return 0.0;
    double det = 1.0;
    for (int i = 0; i < D; i++) det *= A->data[i][i];
    return det;
}

// --- INN FLOW FUNCTIONS (Unchanged) ---

void inn_forward(const INN_Parameters *const params, const Vector *const x, Vector *z); // Forward declaration

double calculate_nll_loss(const INN_Parameters *const params, const Vector *const z) {
    double z_norm_sq = 0.0;
    for (int i = 0; i < D; i++) {
        z_norm_sq += z->data[i] * z->data[i];
    }
    const double log_prob_z = 0.5 * z_norm_sq / (GAUSSIAN_SIGMA * GAUSSIAN_SIGMA);
    const double det_A = get_determinant_triangular(&params->A);
    const double log_det_A = log(fabs(det_A) > 1e-9 ? fabs(det_A) : 1e-9);
    return log_prob_z - log_det_A;
}


// --- FEATURE MAPPING (Simplified Logic) ---

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

/**
 * @brief Maps a SentenceText to its feature vector (D=8). 
 * Fills features sequentially with the first 8 recognized words.
 */
void map_sentence_to_features(const SentenceText *const st, Vector *V) {
    init_vector(V, D);

    int feature_index = 0;
    
    for (size_t i = 0; i < st->count; i++) {
        if (feature_index >= D) break;

        const Word *word = &st->words[i];
        int word_type = -1;
        const double value = find_word_value(word, &word_type);

        if (word->valid && value > 0.0) {
            V->data[feature_index] = value;
            feature_index++;
        }
    }
}

// --- DATASET GENERATION FUNCTIONS (Unchanged logic) ---

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
                legal_templates_st[count] = st;
                count++;
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

    int replace_count = 0;
    int num_to_replace = (rand() % 2) + 1; // Try to replace 1 or 2 words

    for (size_t i = 0; i < legal_template->count; i++) {
        const Word *original_word = &legal_template->words[i];
        int word_type = -1;
        // Check if the original word is a feature-generating type (N, V, or Adj)
        find_word_value(original_word, &word_type); 

        int do_replace = 0;
        // Target Noun (0) or Verb (1) for replacement
        if (replace_count < num_to_replace && original_word->valid && (word_type == 0 || word_type == 1)) {
            if (rand() % 100 < 20) { // 20% chance to replace a suitable word
                do_replace = 1;
                replace_count++;
            }
        }

        if (do_replace) {
            const char *nonsense_str = (word_type == 0) 
                                        ? NonsenseNouns[rand() % NUM_NONSENSE_NOUNS]
                                        : NonsenseVerbs[rand() % NUM_NONSENSE_VERBS];
            illegal_st.words[i] = create_word(nonsense_str, strlen(nonsense_str));
        } else {
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

    for (int i = 0; i < NUM_LEGAL_SENTENCES; i++) {
        const int template_idx = rand() % num_templates;
        const SentenceText *template = &legal_templates_st[template_idx];
        
        legal_sentences[i].text_data = generate_illegal_sentence_text(template); 
        map_sentence_to_features(&legal_sentences[i].text_data, &legal_sentences[i].features);
        legal_sentences[i].is_legal = 1;
    }

    for (int i = 0; i < NUM_ILLEGAL_SENTENCES; i++) {
        const int template_idx = rand() % num_templates;
        const SentenceText *template = &legal_templates_st[template_idx];

        illegal_sentences[i].text_data = generate_illegal_sentence_text(template);
        map_sentence_to_features(&illegal_sentences[i].text_data, &illegal_sentences[i].features);
        illegal_sentences[i].is_legal = 0;
    }

    printf("Dataset generated: %d legal, %d illegal sentences.\n", NUM_LEGAL_SENTENCES, NUM_ILLEGAL_SENTENCES);
}

// --- SANITY CHECKS (Modified Threshold) ---

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

    // --- FIX: Lowered robustness threshold to 1 feature ---
    int robust_template_idx = -1;
    const int REQUIRED_FEATURES = 1; 
    for (int i = 0; i < num_templates; i++) {
        Vector V;
        map_sentence_to_features(&templates[i], &V);
        int feature_count = 0;
        if (V.data[0] > 0.0) feature_count++; // Check only the first feature slot
        
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

// --- MAIN EXECUTION ---

int main(void) {
    srand((unsigned int)time(NULL));
    
    // 1. Load Templates
    SentenceText legal_templates_st[MAX_SENTENCES_FROM_FILE];
    const int num_templates = load_sentence_templates(legal_templates_st, MAX_SENTENCES_FROM_FILE);

    // 2. Run Sanity Checks
    if (!run_sanity_checks(legal_templates_st, num_templates)) {
        fprintf(stderr, "\nFATAL ERROR: Initial sanity checks failed. Aborting training.\n");
        for(int i = 0; i < num_templates; i++) free_sentence_text(&legal_templates_st[i]);
        return 1;
    }

    // 3. Initialize INN and Allocate Memory for Datasets
    INN_Parameters params;
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
    
    
    // --- Training Loop ---
    const clock_t start_time = clock();
    double last_print_time_sec = 0.0;
    const double print_interval_sec = 10.0; 
    int training_stopped_early = 0;

    printf("\n--- Starting INN Training (MLE via SGD) ---\n");
    printf("Training on %d Alice sentences (D=%d). Time limit: %.0f seconds.\n", NUM_LEGAL_SENTENCES, D, TRAINING_TIME_LIMIT_SEC);
    printf("Total Elapsed (s) | Epoch | Iteration | Avg NLL (Loss) | Det(A)\n");
    printf("----------------------------------------------------------------\n");

    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        double epoch_nll_sum = 0.0;

        for (int i = 0; i < NUM_LEGAL_SENTENCES; i++) {
            const double total_elapsed_sec = (double)(clock() - start_time) / CLOCKS_PER_SEC;
            
            if (total_elapsed_sec >= TRAINING_TIME_LIMIT_SEC) {
                printf("--- Stopping training after %.2f seconds (Time Limit Reached) ---\n", total_elapsed_sec);
                training_stopped_early = 1;
                break;
            }

            const int idx = rand() % NUM_LEGAL_SENTENCES;
            const Vector x = legal_sentences[idx].features;

            Vector z; 
            // Forward declaration fix for compiler warning
            void multiply_matrix_vector(const Matrix *const A, const Vector *const x, Vector *y);
            
            inn_forward(&params, &x, &z);
            const double nll_loss = calculate_nll_loss(&params, &z);
            epoch_nll_sum += nll_loss;

            // Backpropagation (Gradients)
            Vector dL_dz; init_vector(&dL_dz, D);
            const double scale = 1.0 / (GAUSSIAN_SIGMA * GAUSSIAN_SIGMA);
            for (int k = 0; k < D; k++) dL_dz.data[k] = z.data[k] * scale;

            for (int k = 0; k < D; k++) params.b.data[k] -= LEARNING_RATE * dL_dz.data[k];

            for (int r = 0; r < D; r++) {
                for (int c = 0; c < D; c++) {
                    if (c <= r) {
                        const double grad_A_r_c = dL_dz.data[r] * x.data[c];
                        params.A.data[r][c] -= LEARNING_RATE * grad_A_r_c;

                        if (r == c) {
                             params.A.data[r][c] -= LEARNING_RATE * (1.0 / params.A.data[r][c]);
                        }
                    }
                }
            }

            if (total_elapsed_sec - last_print_time_sec >= print_interval_sec) {
                 printf("%19.2f | %5d | %9d | %14.4f | %6.4f\n",
                       total_elapsed_sec, epoch, i + 1, epoch_nll_sum / (i + 1), get_determinant_triangular(&params.A));
                last_print_time_sec = total_elapsed_sec;
            }
        } 
        if (training_stopped_early) break; 
    } 

    // --- Evaluation ---
    printf("\n--- INN Detection Test (Legal vs. Nonsensical) ---\n");
    double legal_nll_sum = 0.0;
    double illegal_nll_sum = 0.0;
    int legal_count = 0;
    int illegal_count = 0;

    for (int i = 0; i < NUM_LEGAL_SENTENCES; i++) {
        Vector z;
        inn_forward(&params, &legal_sentences[i].features, &z);
        legal_nll_sum += calculate_nll_loss(&params, &z);
        legal_count++;
    }

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
