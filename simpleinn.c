#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <ctype.h>

// --- INN PARAMETERS AND CONSTANTS ---
#define D 8                   // Increased Dimension of the sentence feature vector
#define NUM_LEGAL_SENTENCES 2048 // Increased Training Size
#define NUM_ILLEGAL_SENTENCES 2048 // Increased Test Size
#define LEARNING_RATE 0.00005  // Adjusted for larger network/data
#define MAX_EPOCHS 200
#define TRAINING_TIME_LIMIT_SEC 240.0 // New time limit of 4 minutes
#define GAUSSIAN_SIGMA 1.0    // Variance of the target base distribution N(0, I)
#define GAUSSIAN_MU 0.0       // Mean of the target base distribution N(0, I)
#define MAX_SENTENCE_LEN 512
#define MAX_ALICE_FILE_SIZE 10000 // Max size for simulated file content

// --- DATA STRUCTURES ---

// Matrix structure with initialization check
typedef struct {
    double data[D][D];
    int rows;
    int cols;
    int initialized;
} Matrix;

// Vector structure with initialization check
typedef struct {
    double data[D];
    int size;
    int initialized;
} Vector;

// Struct to hold INN parameters
typedef struct {
    Matrix A;       // The DxD Lower Triangular Weight Matrix (Decomposable)
    Vector b;       // The D-dimensional Bias Vector
} INN_Parameters;

// Struct to hold a single sentence and its feature vector
typedef struct {
    char text[MAX_SENTENCE_LEN];
    Vector features;
    int is_legal;
} Sentence;

// --- VOCABULARY AND FEATURE MAPPING (Expanded for D=8) ---

// Nouns/Verbs/Adjectives expanded and updated.
const char *const Nouns[] = {"car", "bike", "dog", "lawyer", "judge", "contract", "witness", "defendant", "alice", "way", "side", "door", "middle", "table", "glass", "key", "thought", "locks", "curtain", "sister", "rabbit", "garden", "pool", "mock", "turtle"};
const double NounValues[] = {1.0, 1.5, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5};
#define NUM_NOUNS (sizeof(Nouns) / sizeof(Nouns[0]))

const char *const Verbs[] = {"drives", "reads", "signs", "runs", "had", "been", "trying", "walked", "wondering", "get", "came", "made", "belong", "open", "noticed", "tried", "fitted", "began", "fell", "swam", "cried", "ran", "jumped"};
const double VerbValues[] = {1.0, 2.0, 3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0};
#define NUM_VERBS (sizeof(Verbs) / sizeof(Verbs[0]))

const char *const Adjectives[] = {"red", "fast", "legal", "binding", "corrupt", "little", "three-legged", "solid", "tiny", "golden", "large", "small", "low", "fifteen", "great", "second", "white", "busy", "silly", "mad"};
const double AdjectiveValues[] = {1.0, 1.5, 2.0, 3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 1.5, 10.0, 10.5, 11.0, 11.5};
#define NUM_ADJECTIVES (sizeof(Adjectives) / sizeof(Adjectives[0]))

// New: Nonsensical/Out-of-Distribution words for creating illegal sentences
const char *const NonsenseNouns[] = {"glork", "zorp", "fleep", "wubba", "snerd", "crungle", "chork"};
const char *const NonsenseVerbs[] = {"bleem", "floof", "skree", "quonk", "smashle", "zooble"};
#define NUM_NONSENSE_NOUNS (sizeof(NonsenseNouns) / sizeof(NonsenseNouns[0]))
#define NUM_NONSENSE_VERBS (sizeof(NonsenseVerbs) / sizeof(NonsenseVerbs[0]))

// --- SIMULATED FILE CONTENT ---
// In a real environment, this content would be read from alice.txt
const char *SIMULATED_ALICE_CONTENT = 
"Alice had been all the way down one side and up the other, trying every door, she walked sadly down the middle, wondering how she was ever to get out again. Suddenly she came upon a little three-legged table, all made of solid glass; there was nothing on it except a tiny golden key, and Alice’s first thought was that it might belong to one of the doors of the hall; but, alas! either the locks were too large, or the key was too small, but at any rate it would not open any of them. However, on the second time round, she came upon a low curtain she had not noticed before, and behind it was a little door about fifteen inches high: she tried the little golden key in the lock, and to her great delight it fitted! 'Curiouser and curiouser!' cried Alice (she was so much surprised, that for the moment she quite forgot how to speak good English). Down, down, down. Would the fall never come to an end? 'I wonder how many miles I’ve fallen by this time?' she said aloud. 'I must be getting somewhere near the centre of the earth. Let me see: that would be four thousand miles down, I think.' 'I shall be late!' (when she thought of this, it was just possible it had happened a few minutes before, and she was in a great hurry) 'My sister will think I’m mad! She will think I'm mad!'";
#define MAX_SENTENCES_FROM_FILE 100 // Limit for loading distinct sentences

// --- UTILITY FUNCTIONS FOR MATRIX AND VECTOR OPERATIONS ---

/**
 * @brief Initializes a matrix with small random values, enforcing lower triangular structure.
 */
void init_matrix(Matrix *M, int rows, int cols, int triangular) {
    if (rows != D || cols != D) { fprintf(stderr, "Error: Matrix dimensions must be %dx%d.\n", D, D); return; }
    M->rows = rows; M->cols = cols;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (triangular && j > i) {
                M->data[i][j] = 0.0;
            } else {
                M->data[i][j] = ((double)rand() / RAND_MAX) * 0.01; // Smaller initialization
            }
        }
    }
    for (int i = 0; i < D; i++) { if (M->data[i][i] == 0.0) { M->data[i][i] = 1.0; } }
    M->initialized = 1;
}

/**
 * @brief Initializes a vector with zeros.
 */
void init_vector(Vector *V, int size) {
    if (size != D) { fprintf(stderr, "Error: Vector dimension must be %d.\n", D); return; }
    V->size = size;
    for (int i = 0; i < size; i++) { V->data[i] = 0.0; }
    V->initialized = 1;
}

/**
 * @brief Checks if a matrix is initialized (simplified).
 */
int check_init(const Matrix *const M) {
    if (!M->initialized) { return 0; }
    return 1;
}

/**
 * @brief Checks if a vector is initialized (simplified).
 */
int check_vector_init(const Vector *const V) {
    if (!V->initialized) { return 0; }
    return 1;
}


/**
 * @brief Performs matrix-vector multiplication: y = A * x.
 */
void multiply_matrix_vector(const Matrix *const A, const Vector *const x, Vector *y) {
    if (!check_init(A) || !check_vector_init(x) || A->cols != x->size) { return; }
    init_vector(y, D);
    for (int i = 0; i < A->rows; i++) {
        double sum = 0.0;
        for (int j = 0; j < A->cols; j++) {
            sum += A->data[i][j] * x->data[j];
        }
        y->data[i] = sum;
    }
}

/**
 * @brief Computes the determinant of the triangular INN matrix.
 */
double get_determinant_triangular(const Matrix *const A) {
    if (!check_init(A)) return 0.0;
    double det = 1.0;
    for (int i = 0; i < D; i++) {
        det *= A->data[i][i];
    }
    return det;
}

// --- INN FLOW FUNCTIONS ---

/**
 * @brief The forward pass of the INN (transformation x -> z).
 * @param params The INN parameters (A, b).
 * @param x The input vector (sentence features).
 * @param z The output latent vector.
 */
void inn_forward(const INN_Parameters *const params, const Vector *const x, Vector *z) {
    if (!check_init(&params->A) || !check_vector_init(&params->b)) return;
    // z = A*x + b
    multiply_matrix_vector(&params->A, x, z);
    for (int i = 0; i < D; i++) {
        z->data[i] += params->b.data[i];
    }
}

/**
 * @brief Calculates the Negative Log-Likelihood (NLL) loss.
 */
double calculate_nll_loss(const INN_Parameters *const params, const Vector *const z) {
    double z_norm_sq = 0.0;
    for (int i = 0; i < D; i++) {
        z_norm_sq += z->data[i] * z->data[i];
    }

    const double log_prob_z = 0.5 * z_norm_sq / (GAUSSIAN_SIGMA * GAUSSIAN_SIGMA);
    const double det_A = get_determinant_triangular(&params->A);
    const double log_det_A = log(fabs(det_A) > 1e-9 ? fabs(det_A) : 1e-9);

    const double nll = log_prob_z - log_det_A;
    return nll;
}

// --- DATASET PARSING & FEATURE MAPPING ---

/**
 * @brief Attempts to find a word in the vocabulary lists and return its feature value and type.
 * @return The feature value, or 0.0 if not found.
 */
double find_word_value(const char *word, int *word_type) {
    char lower_word[32];
    strncpy(lower_word, word, 31);
    lower_word[31] = '\0';
    for(int i = 0; lower_word[i]; i++){
      if(lower_word[i] >= 'A' && lower_word[i] <= 'Z') lower_word[i] += 'a' - 'A';
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
    *word_type = -1; // Not a relevant word type
    return 0.0;
}

/**
 * @brief Maps a complex sentence string to its feature vector (D=8).
 * Features: [N1, V1, Adj1, N2, V2, Adj2, V3, N3] (First 3 of each, or 0.0 if not found)
 */
void parse_sentence_to_features(const char *const text, Vector *V) {
    init_vector(V, D);

    char temp_text[MAX_SENTENCE_LEN];
    strncpy(temp_text, text, MAX_SENTENCE_LEN - 1);
    temp_text[MAX_SENTENCE_LEN - 1] = '\0';

    // Remove punctuation from the end to aid strtok
    char *end = temp_text + strlen(temp_text) - 1;
    while(end > temp_text && (ispunct(*end) || isspace(*end))) { *end-- = '\0'; }


    char *token = strtok(temp_text, " ,.;:!?");

    // Feature indices to fill: [N1, V1, Adj1, N2, V2, Adj2, V3, N3]
    int noun_count = 0; // V[0], V[3], V[7]
    int verb_count = 0; // V[1], V[4], V[6]
    int adj_count = 0;  // V[2], V[5]
    
    while (token != NULL) {
        int word_type = -1;
        const double value = find_word_value(token, &word_type);

        if (value > 0.0) {
            if (word_type == 0) { // Noun
                if (noun_count == 0) V->data[0] = value;
                else if (noun_count == 1) V->data[3] = value;
                else if (noun_count == 2) V->data[7] = value;
                noun_count++;
            } else if (word_type == 1) { // Verb
                if (verb_count == 0) V->data[1] = value;
                else if (verb_count == 1) V->data[4] = value;
                else if (verb_count == 2) V->data[6] = value;
                verb_count++;
            } else if (word_type == 2) { // Adjective
                if (adj_count == 0) V->data[2] = value;
                else if (adj_count == 1) V->data[5] = value;
                adj_count++;
            }
        }

        // Optimization: stop if all D=8 core features are filled
        if (noun_count >= 3 && verb_count >= 3 && adj_count >= 2) {
            break;
        }

        token = strtok(NULL, " ,.;:!?");
    }
}


// --- DATASET GENERATION FUNCTIONS ---

/**
 * @brief Reads sentences from the simulated file content.
 * @return The number of distinct sentences loaded.
 */
int load_sentences_from_file(char legal_templates[MAX_SENTENCES_FROM_FILE][MAX_SENTENCE_LEN]) {
    char temp_content[MAX_ALICE_FILE_SIZE];
    strncpy(temp_content, SIMULATED_ALICE_CONTENT, MAX_ALICE_FILE_SIZE - 1);
    temp_content[MAX_ALICE_FILE_SIZE - 1] = '\0';
    
    // Split the content into sentences (using a period followed by a space)
    char *sentence_start = temp_content;
    int count = 0;

    while (sentence_start != NULL && count < MAX_SENTENCES_FROM_FILE) {
        char *sentence_end = strstr(sentence_start, ". ");
        if (sentence_end == NULL) {
            // Check for the very last sentence without a trailing space
            size_t len = strlen(sentence_start);
            if (len > 0) {
                 if (sentence_start[len-1] == '.' || sentence_start[len-1] == '!' || sentence_start[len-1] == '?') {
                    strncpy(legal_templates[count], sentence_start, MAX_SENTENCE_LEN - 1);
                    legal_templates[count][MAX_SENTENCE_LEN - 1] = '\0';
                    count++;
                 }
            }
            break; 
        }

        // Copy and null-terminate the sentence
        size_t len = sentence_end - sentence_start + 1;
        if (len < MAX_SENTENCE_LEN) {
            strncpy(legal_templates[count], sentence_start, len);
            legal_templates[count][len] = '\0';
            count++;
        }
        
        sentence_start = sentence_end + 2; // Move past ". "
    }

    return count;
}


/**
 * @brief Generates an illegal sentence by substituting a word in a legal template.
 * Attempts to replace 1 or 2 words (Noun/Verb) with a nonsensical word.
 */
void generate_illegal_sentence(const char *legal_template, Sentence *illegal_s) {
    char temp_text[MAX_SENTENCE_LEN];
    strncpy(temp_text, legal_template, MAX_SENTENCE_LEN - 1);
    temp_text[MAX_SENTENCE_LEN - 1] = '\0';

    char working_text[MAX_SENTENCE_LEN * 2]; // Increased size for safety
    working_text[0] = '\0';

    char *token = strtok(temp_text, " ,.;:!?");
    int replace_count = 0;
    int num_to_replace = (rand() % 2) + 1; // 1 or 2 words

    while (token != NULL) {
        int word_type = -1;
        find_word_value(token, &word_type); // Check if it's a known word

        int do_replace = 0;
        if (replace_count < num_to_replace && (word_type == 0 || word_type == 1)) {
            // 15% chance to replace this specific word
            if (rand() % 100 < 15) {
                do_replace = 1;
                replace_count++;
            }
        }

        if (do_replace) {
            const char *nonsense_word = (word_type == 0) 
                                        ? NonsenseNouns[rand() % NUM_NONSENSE_NOUNS]
                                        : NonsenseVerbs[rand() % NUM_NONSENSE_VERBS];
            strcat(working_text, nonsense_word);
        } else {
            strcat(working_text, token);
        }
        
        strcat(working_text, " "); // Re-add space/delimiter (simple)
        token = strtok(NULL, " ,.;:!?");
    }
    
    // Simple cleanup: remove trailing space
    if (working_text[0] != '\0' && working_text[strlen(working_text) - 1] == ' ') {
        working_text[strlen(working_text) - 1] = '\0';
    }

    // Copy to struct and parse features
    strncpy(illegal_s->text, working_text, MAX_SENTENCE_LEN - 1);
    illegal_s->text[MAX_SENTENCE_LEN - 1] = '\0';
    parse_sentence_to_features(illegal_s->text, &illegal_s->features);
    illegal_s->is_legal = 0;
}


/**
 * @brief Populates the training (legal) and testing (illegal) datasets.
 */
void generate_datasets(Sentence *legal_sentences, Sentence *illegal_sentences) {
    char legal_templates[MAX_SENTENCES_FROM_FILE][MAX_SENTENCE_LEN];
    const int num_templates = load_sentences_from_file(legal_templates);

    if (num_templates == 0) {
        fprintf(stderr, "FATAL ERROR: Could not load any sentences from simulated alice.txt content.\n");
        return;
    }

    printf("Loaded %d distinct Alice sentence templates from file content.\n", num_templates);

    // 1. Legal Sentences (Training data)
    for (int i = 0; i < NUM_LEGAL_SENTENCES; i++) {
        const int template_idx = rand() % num_templates;
        strcpy(legal_sentences[i].text, legal_templates[template_idx]);
        parse_sentence_to_features(legal_sentences[i].text, &legal_sentences[i].features);
        legal_sentences[i].is_legal = 1;
    }

    // 2. Illegal Sentences (Test data) - Word Substitution
    for (int i = 0; i < NUM_ILLEGAL_SENTENCES; i++) {
        const int template_idx = rand() % num_templates;
        generate_illegal_sentence(legal_templates[template_idx], &illegal_sentences[i]);
    }

    printf("Dataset generated: %d legal (Alice text), %d illegal (Word-Substituted) sentences.\n", NUM_LEGAL_SENTENCES, NUM_ILLEGAL_SENTENCES);
}

// --- MAIN EXECUTION ---

int main(void) {
    // Note: No unit tests are included in this final version to reduce bulk, 
    // but they should be used in a development environment.

    srand((unsigned int)time(NULL));
    INN_Parameters params;
    init_matrix(&params.A, D, D, 1);
    init_vector(&params.b, D);

    if (!params.A.initialized || !params.b.initialized) {
        fprintf(stderr, "Fatal Error: INN parameters failed to initialize.\n");
        return 1;
    }

    Sentence legal_sentences[NUM_LEGAL_SENTENCES];
    Sentence illegal_sentences[NUM_ILLEGAL_SENTENCES];
    generate_datasets(legal_sentences, illegal_sentences);

    const clock_t start_time = clock();
    double last_print_time_sec = 0.0;
    const double print_interval_sec = 10.0; // Print every 10 seconds
    int training_stopped_early = 0;

    printf("\n--- Starting INN Training (MLE via SGD) ---\n");
    printf("Training on %d Alice sentences (D=%d). Time limit: %.0f seconds.\n", NUM_LEGAL_SENTENCES, D, TRAINING_TIME_LIMIT_SEC);
    printf("Total Elapsed (s) | Epoch | Iteration | Avg NLL (Loss) | Det(A)\n");
    printf("----------------------------------------------------------------\n");

    // 3. Training Loop (SGD - Maximum Likelihood Estimation)
    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        double epoch_nll_sum = 0.0;
        const int num_batches = NUM_LEGAL_SENTENCES;

        for (int i = 0; i < num_batches; i++) {
            const double total_elapsed_sec = (double)(clock() - start_time) / CLOCKS_PER_SEC;
            
            if (total_elapsed_sec >= TRAINING_TIME_LIMIT_SEC) {
                printf("--- Stopping training after %.2f seconds (Time Limit Reached) ---\n", total_elapsed_sec);
                training_stopped_early = 1;
                break; // Break batch loop
            }

            const int idx = rand() % NUM_LEGAL_SENTENCES;
            const Vector x = legal_sentences[idx].features;

            Vector z;
            inn_forward(&params, &x, &z);
            const double nll_loss = calculate_nll_loss(&params, &z);
            epoch_nll_sum += nll_loss;

            // --- Backpropagation (Calculating Gradients) ---

            Vector dL_dz;
            init_vector(&dL_dz, D);
            const double scale = 1.0 / (GAUSSIAN_SIGMA * GAUSSIAN_SIGMA);
            for (int k = 0; k < D; k++) {
                dL_dz.data[k] = z.data[k] * scale;
            }

            // Update bias b
            for (int k = 0; k < D; k++) {
                params.b.data[k] -= LEARNING_RATE * dL_dz.data[k];
            }

            // Update Matrix A
            for (int r = 0; r < D; r++) {
                for (int c = 0; c < D; c++) {
                    if (c <= r) { // Lower triangular part
                        const double grad_A_r_c = dL_dz.data[r] * x.data[c];
                        params.A.data[r][c] -= LEARNING_RATE * grad_A_r_c;

                        if (r == c) { // Diagonal term
                             params.A.data[r][c] -= LEARNING_RATE * (1.0 / params.A.data[r][c]);
                        }
                    }
                }
            }

            // --- STATS PRINTING (Time-based) ---
            if (total_elapsed_sec - last_print_time_sec >= print_interval_sec) {
                 printf("%19.2f | %5d | %9d | %14.4f | %6.4f\n",
                       total_elapsed_sec, epoch, i + 1, epoch_nll_sum / (i + 1), get_determinant_triangular(&params.A));
                last_print_time_sec = total_elapsed_sec;
            }

        } // End batch loop
        if (training_stopped_early) break; // Break epoch loop if time limit hit
    } // End epoch loop

    // 4. Detection / Evaluation
    printf("\n--- INN Detection Test (Legal vs. Nonsensical) ---\n");
    double legal_nll_sum = 0.0;
    double illegal_nll_sum = 0.0;
    int legal_count = 0;
    int illegal_count = 0;

    // Test Legal Sentences
    for (int i = 0; i < NUM_LEGAL_SENTENCES; i++) {
        Vector z;
        inn_forward(&params, &legal_sentences[i].features, &z);
        legal_nll_sum += calculate_nll_loss(&params, &z);
        legal_count++;
    }

    // Test Illegal Sentences
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
    
    // --- Detailed Sample Output ---
    printf("\n--- Detailed NLL Analysis (First 5 of each) ---\n");
    
    // Print 5 Legal Sentences
    printf("\nLegal Sentences (IN-DISTRIBUTION - Low NLL expected):\n");
    for (int i = 0; i < 5 && i < NUM_LEGAL_SENTENCES; i++) {
        Vector z;
        inn_forward(&params, &legal_sentences[i].features, &z);
        const double nll = calculate_nll_loss(&params, &z);
        printf("NLL: %.4f | Text: %s\n", nll, legal_sentences[i].text);
    }

    // Print 5 Illegal Sentences
    printf("\nIllegal Sentences (OUT-OF-DISTRIBUTION - High NLL expected):\n");
    for (int i = 0; i < 5 && i < NUM_ILLEGAL_SENTENCES; i++) {
        Vector z;
        inn_forward(&params, &illegal_sentences[i].features, &z);
        const double nll = calculate_nll_loss(&params, &z);
        printf("NLL: %.4f | Text: %s\n", nll, illegal_sentences[i].text);
    }
    // --- End Detailed Sample Output ---

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

    return 0;
}
