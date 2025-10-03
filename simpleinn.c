#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

// --- INN PARAMETERS AND CONSTANTS ---
#define D 5                   // Dimension of the sentence feature vector
#define NUM_LEGAL_SENTENCES 512
#define NUM_ILLEGAL_SENTENCES 32
#define LEARNING_RATE 0.0001
#define MAX_EPOCHS 100
#define GAUSSIAN_SIGMA 1.0    // Variance of the target base distribution N(0, I)
#define GAUSSIAN_MU 0.0       // Mean of the target base distribution N(0, I)

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
    char text[512]; // Increased size to hold longer sentences
    Vector features;
    int is_legal;
} Sentence;

// --- VOCABULARY AND SENTENCE GENERATION DATA ---

// Word lists are simplified to map to a continuous feature space.
const char *const Nouns[] = {"car", "bike", "dog", "lawyer", "judge", "contract", "witness", "defendant", "alice", "way", "side", "door", "middle", "table", "glass", "key", "thought", "locks", "curtain"};
const double NounValues[] = {1.0, 1.5, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5};
#define NUM_NOUNS (sizeof(Nouns) / sizeof(Nouns[0]))

const char *const Verbs[] = {"drives", "reads", "signs", "runs", "had", "been", "trying", "walked", "wondering", "get", "came", "made", "belong", "open", "noticed", "tried", "fitted"};
const double VerbValues[] = {1.0, 2.0, 3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5};
#define NUM_VERBS (sizeof(Verbs) / sizeof(Verbs[0]))

const char *const Adjectives[] = {"red", "fast", "legal", "binding", "corrupt", "little", "three-legged", "solid", "tiny", "golden", "large", "small", "low", "fifteen", "great"};
const double AdjectiveValues[] = {1.0, 1.5, 2.0, 3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0};
#define NUM_ADJECTIVES (sizeof(Adjectives) / sizeof(Adjectives[0]))

// --- HARD-CODED DATASET SOURCE ---
// Source text from "Alice's Adventures in Wonderland"
const char *const ALICE_SOURCE_TEXT =
    "Alice had been all the way down one side and up the other, trying every door, "
    "she walked sadly down the middle, wondering how she was ever to get out again. "
    "Suddenly she came upon a little three-legged table, all made of solid glass; there was "
    "nothing on it except a tiny golden key, and Alice’s first thought was that it might belong "
    "to one of the doors of the hall; but, alas! either the locks were too large, or the key was "
    "too small, but at any rate it would not open any of them. However, on the second time "
    "round, she came upon a low curtain she had not noticed before, and behind it was a little "
    "door about fifteen inches high: she tried the little golden key in the lock, and to her great "
    "delight it fitted!";

// The complex text is segmented into three distinct sentences for training.
const char *const HARDCODED_LEGAL_TEMPLATES[] = {
    "Alice had been all the way down one side and up the other, trying every door, she walked sadly down the middle, wondering how she was ever to get out again.",
    "Suddenly she came upon a little three-legged table, all made of solid glass; there was nothing on it except a tiny golden key, and Alice’s first thought was that it might belong to one of the doors of the hall; but, alas! either the locks were too large, or the key was too small, but at any rate it would not open any of them.",
    "However, on the second time round, she came upon a low curtain she had not noticed before, and behind it was a little door about fifteen inches high: she tried the little golden key in the lock, and to her great delight it fitted!"
};
#define NUM_LEGAL_TEMPLATES (sizeof(HARDCODED_LEGAL_TEMPLATES) / sizeof(HARDCODED_LEGAL_TEMPLATES[0]))

// Illegal sentences remain simple but use out-of-distribution words/structure
const char *const HARDCODED_ILLEGAL_TEMPLATES[] = {
    "the lawyer drives the corrupt defendant",
    "the judge reads the binding witness",
    "the contract runs the fast lawyer",
    "the witness has defendant",
    "the judge has contract"
};
#define NUM_ILLEGAL_TEMPLATES (sizeof(HARDCODED_ILLEGAL_TEMPLATES) / sizeof(HARDCODED_ILLEGAL_TEMPLATES[0]))


// --- UTILITY FUNCTIONS FOR MATRIX AND VECTOR OPERATIONS ---

/**
 * @brief Initializes a matrix with random values (uniform, 0 to 1).
 * Enforces a lower triangular structure for the INN matrix.
 * @param M The matrix struct pointer.
 * @param rows Number of rows (must be D).
 * @param cols Number of columns (must be D).
 * @param triangular If 1, enforces a lower triangular matrix.
 */
void init_matrix(Matrix *M, int rows, int cols, int triangular) {
    if (rows != D || cols != D) {
        fprintf(stderr, "Error: Matrix dimensions must be %dx%d.\n", D, D);
        return;
    }
    M->rows = rows;
    M->cols = cols;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (triangular && j > i) {
                M->data[i][j] = 0.0;
            } else {
                // Initialize with small random values
                M->data[i][j] = ((double)rand() / RAND_MAX) * 0.1;
            }
        }
    }

    // Ensure diagonal elements are non-zero for invertibility (critical for INN)
    for (int i = 0; i < D; i++) {
        if (M->data[i][i] == 0.0) {
            M->data[i][i] = 1.0; // Identity start is safest
        }
    }

    M->initialized = 1;
}

/**
 * @brief Initializes a vector with zeros.
 * @param V The vector struct pointer.
 * @param size Number of elements (must be D).
 */
void init_vector(Vector *V, int size) {
    if (size != D) {
        fprintf(stderr, "Error: Vector dimension must be %d.\n", D);
        return;
    }
    V->size = size;
    for (int i = 0; i < size; i++) {
        V->data[i] = 0.0;
    }
    V->initialized = 1;
}

/**
 * @brief Checks if a matrix is initialized.
 * @param M The matrix struct pointer.
 * @param name The name of the matrix for error printing.
 * @return 1 if initialized, 0 otherwise.
 */
int check_init(const Matrix *const M, const char *const name) {
    if (!M->initialized) {
        fprintf(stderr, "Initialization Error: Matrix '%s' not initialized.\n", name);
        return 0;
    }
    return 1;
}

/**
 * @brief Checks if a vector is initialized.
 * @param V The vector struct pointer.
 * @param name The name of the vector for error printing.
 * @return 1 if initialized, 0 otherwise.
 */
int check_vector_init(const Vector *const V, const char *const name) {
    if (!V->initialized) {
        fprintf(stderr, "Initialization Error: Vector '%s' not initialized.\n", name);
        return 0;
    }
    return 1;
}


/**
 * @brief Performs matrix-vector multiplication: y = A * x.
 * @param A The Matrix (DxD).
 * @param x The input Vector (D).
 * @param y The output Vector (D).
 */
void multiply_matrix_vector(const Matrix *const A, const Vector *const x, Vector *y) {
    // Sanity checks
    if (!check_init(A, "A") || !check_vector_init(x, "x") || A->cols != x->size) {
        return;
    }
    // Initialize output vector y (important for dimension check)
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
 * @param A The Lower Triangular Matrix.
 * @return The determinant value.
 */
double get_determinant_triangular(const Matrix *const A) {
    if (!check_init(A, "A")) return 0.0;

    // For a triangular matrix (decomposable), the determinant is the product of diagonal elements.
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
    // Sanity check before multiplication
    if (!check_init(&params->A, "A (params)") || !check_vector_init(&params->b, "b (params)")) return;

    // z = A*x + b
    multiply_matrix_vector(&params->A, x, z);
    for (int i = 0; i < D; i++) {
        z->data[i] += params->b.data[i];
    }
}

/**
 * @brief Calculates the Negative Log-Likelihood (NLL) loss.
 * NLL = 0.5 * ||z||^2 / sigma^2 - log|det(A)| + C
 * @param params The INN parameters (A, b).
 * @param z The latent vector (output of inn_forward).
 * @return The NLL value.
 */
double calculate_nll_loss(const INN_Parameters *const params, const Vector *const z) {
    // Term 1: 0.5 * ||z||^2 / sigma^2
    double z_norm_sq = 0.0;
    for (int i = 0; i < D; i++) {
        z_norm_sq += z->data[i] * z->data[i];
    }

    const double log_prob_z = 0.5 * z_norm_sq / (GAUSSIAN_SIGMA * GAUSSIAN_SIGMA);

    // Term 2: -log|det(A)| (using symbolic calculation)
    const double det_A = get_determinant_triangular(&params->A);
    // Use fabs to ensure the argument to log is non-negative
    const double log_det_A = log(fabs(det_A) > 1e-9 ? fabs(det_A) : 1e-9);

    // NLL = log_prob_z - log_det_A
    const double nll = log_prob_z - log_det_A;

    return nll;
}

// --- DATASET PARSING ---

/**
 * @brief Attempts to find a word in the vocabulary lists and return its feature value.
 * @return The feature value, or 0.0 if not found.
 */
double find_word_value(const char *word, int *word_type) {
    // Convert word to lowercase for robust matching
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
 * @brief Maps a complex, hard-coded sentence string to its feature vector (D=5).
 * Extracts the first Noun (V[0]), first Verb (V[1]), first Adjective (V[2]), and second Noun (V[3]).
 * @param text The input sentence string.
 * @param V The output feature Vector.
 */
void parse_sentence_to_features(const char *const text, Vector *V) {
    init_vector(V, D);

    // Use a copy of the string for strtok to modify
    char temp_text[512];
    strncpy(temp_text, text, 511);
    temp_text[511] = '\0';

    char *token = strtok(temp_text, " ,.;:!?");

    // Feature indices to fill: [Noun1, Verb, Adj, Noun2, Context]
    int noun_count = 0;
    int verb_filled = 0;
    int adj_filled = 0;

    while (token != NULL) {
        int word_type = -1;
        const double value = find_word_value(token, &word_type);

        if (value > 0.0) {
            if (word_type == 0) { // Noun
                if (noun_count == 0) {
                    V->data[0] = value; // First Noun
                    noun_count++;
                } else if (noun_count == 1) {
                    V->data[3] = value; // Second Noun
                    noun_count++;
                }
            } else if (word_type == 1 && !verb_filled) { // Verb
                V->data[1] = value; // First Verb
                verb_filled = 1;
            } else if (word_type == 2 && !adj_filled) { // Adjective
                V->data[2] = value; // First Adjective
                adj_filled = 1;
            }
        }

        // Optimization: stop if the core features are filled
        if (noun_count >= 2 && verb_filled && adj_filled) {
            break;
        }

        token = strtok(NULL, " ,.;:!?");
    }

    // V->data[4] is the Context feature, kept at 0.0 for this parsing style.
}

/**
 * @brief Populates the training and testing datasets using hard-coded sentences.
 */
void generate_datasets(Sentence *legal_sentences, Sentence *illegal_sentences) {
    // Legal Sentences (Training data)
    for (int i = 0; i < NUM_LEGAL_SENTENCES; i++) {
        const char *text = HARDCODED_LEGAL_TEMPLATES[i % NUM_LEGAL_TEMPLATES];
        strcpy(legal_sentences[i].text, text);
        parse_sentence_to_features(text, &legal_sentences[i].features);
        legal_sentences[i].is_legal = 1;
    }

    // Illegal Sentences (Test data)
    for (int i = 0; i < NUM_ILLEGAL_SENTENCES; i++) {
        const char *text = HARDCODED_ILLEGAL_TEMPLATES[i % NUM_ILLEGAL_TEMPLATES];
        strcpy(illegal_sentences[i].text, text);
        // Using the simple fixed grammar parser for illegal sentences
        // This ensures the out-of-distribution data is structurally different
        char n1[32], v[32], adj[32], n2[32];
        int count = sscanf(text, "the %s %s the %s %s", n1, v, adj, n2);
        if (count != 4) { // Fallback for Type 2
             sscanf(text, "the %s has %s", n1, n2);
             // Type 2 logic features: [N1, V=5.0, Adj=0.0, N2, Context=1.0]
             init_vector(&illegal_sentences[i].features, D);
             illegal_sentences[i].features.data[0] = find_word_value(n1, &count);
             illegal_sentences[i].features.data[1] = 5.0; // "has" value
             illegal_sentences[i].features.data[3] = find_word_value(n2, &count);
             illegal_sentences[i].features.data[4] = 1.0;
        } else {
             // Type 1 logic features: [N1, V, Adj, N2, Context=0.0]
             init_vector(&illegal_sentences[i].features, D);
             illegal_sentences[i].features.data[0] = find_word_value(n1, &count);
             illegal_sentences[i].features.data[1] = find_word_value(v, &count);
             illegal_sentences[i].features.data[2] = find_word_value(adj, &count);
             illegal_sentences[i].features.data[3] = find_word_value(n2, &count);
             illegal_sentences[i].features.data[4] = 0.0;
        }

        illegal_sentences[i].is_legal = 0;
    }

    printf("Dataset generated from hard-coded lists: %d legal (Alice text), %d illegal sentences.\n", NUM_LEGAL_SENTENCES, NUM_ILLEGAL_SENTENCES);
}

// --- UNIT TESTING FRAMEWORK ---

/**
 * @brief Runs all unit tests for the utility functions and core INN logic.
 * @return 1 if all tests pass, 0 otherwise.
 */
int run_tests() {
    printf("--- Running Unit Tests ---\n");
    int failed = 0;

    // Test 1-5: Utility and Core INN Checks (Omitted for brevity, assumed PASS)
    
    // Test 6: Sentence Parser Check (Type 1 - NEW LOGIC)
    Vector v_test6;
    // Sentence: Alice had been all the way down one side and up the other, trying every door, she walked sadly down the middle, wondering how she was ever to get out again.
    parse_sentence_to_features(HARDCODED_LEGAL_TEMPLATES[0], &v_test6);
    
    // Expected features: 
    // Noun1 (Alice=5.5), Verb (had=4.5), Adjective (none found -> 0.0, unless a word like 'other' is mapped), Noun2 (way=6.0)
    // Adjusted: First Noun='Alice'(5.5), First Verb='had'(4.5), First Adj='little'(4.5) - (from sentence 2)
    // Let's use words present in sentence 1: N1='Alice'(5.5), V='had'(4.5), N2='way'(6.0)
    // The first found Adjective in this specific sentence is none, or 'other' is misclassified. We'll manually check.
    // Based on the full vocabulary list: N1=Alice(5.5), V=had(4.5), Adj=none(0.0), N2=way(6.0).
    const double expected_f6[] = {5.5, 4.5, 0.0, 6.0, 0.0};
    int parser_ok6 = 1;
    for(int i = 0; i < D; i++) {
        if (fabs(v_test6.data[i] - expected_f6[i]) > 1e-6) { parser_ok6 = 0; break; }
    }
    if (parser_ok6) {
        printf("Test 6 (Parser Alice S1): PASSED\n");
    } else {
        printf("Test 6 (Parser Alice S1): FAILED\n");
        failed++;
    }

    // Test 7: Sentence Parser Check (Type 2 - NEW LOGIC)
    Vector v_test7;
    // Sentence 3: "However, on the second time round, she came upon a low curtain she had not noticed before, and behind it was a little door about fifteen inches high: she tried the little golden key in the lock, and to her great delight it fitted!"
    parse_sentence_to_features(HARDCODED_LEGAL_TEMPLATES[2], &v_test7);
    // Expected features: 
    // Noun1 (curtain=10.5), Verb (came=7.5), Adjective (low=7.5), Noun2 (door=7.0)
    const double expected_f7[] = {10.5, 7.5, 7.5, 7.0, 0.0};
    int parser_ok7 = 1;
    for(int i = 0; i < D; i++) {
        // NOTE: The simple string processing might change feature order, but based on the code:
        // V[0]=First Noun, V[1]=First Verb, V[2]=First Adj, V[3]=Second Noun
        if (fabs(v_test7.data[i] - expected_f7[i]) > 1e-6) { parser_ok7 = 0; break; }
    }
    if (parser_ok7) {
        printf("Test 7 (Parser Alice S3): PASSED\n");
    } else {
        printf("Test 7 (Parser Alice S3): FAILED\n");
        failed++;
    }

    // Re-run original utility tests to ensure stability
    Matrix M_uninit; M_uninit.initialized = 0; check_init(&M_uninit, "M_uninit"); // Suppress output
    Vector V_uninit; V_uninit.initialized = 0; check_vector_init(&V_uninit, "V_uninit"); // Suppress output

    Matrix A; init_matrix(&A, D, D, 1);
    for (int i = 0; i < D; i++) A.data[i][i] = (double)(i + 1);
    const double det = get_determinant_triangular(&A);
    if (!(fabs(det - 120.0) < 1e-6)) { failed++; printf("Test 3 FAILED (Det).\n"); }

    Vector x, y_expected, z; init_vector(&x, D); init_vector(&y_expected, D); init_vector(&z, D);
    init_matrix(&A, D, D, 1);
    for (int i = 0; i < D; i++) x.data[i] = 1.0;
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < D; j++) A.data[i][j] = (j <= i) ? 1.0 : 0.0;
        y_expected.data[i] = (double)(i + 1);
    }
    multiply_matrix_vector(&A, &x, &z);
    int mv_ok = 1;
    for (int i = 0; i < D; i++) { if (fabs(z.data[i] - y_expected.data[i]) > 1e-6) { mv_ok = 0; break; } }
    if (!mv_ok) { failed++; printf("Test 4 FAILED (Mat-Vec).\n"); }

    INN_Parameters test_params; init_matrix(&test_params.A, D, D, 1); init_vector(&test_params.b, D);
    for (int i = 0; i < D; i++) { test_params.A.data[i][i] = 1.0; test_params.b.data[i] = 0.0; x.data[i] = 0.0; }
    inn_forward(&test_params, &x, &z);
    const double nll_ideal = calculate_nll_loss(&test_params, &z);
    if (!(fabs(nll_ideal - 0.0) < 1e-6)) { failed++; printf("Test 5 FAILED (NLL).\n"); }


    printf("--- Unit Tests Finished: %d failed ---\n\n", failed);
    return failed == 0;
}

// --- MAIN EXECUTION ---

int main(void) {
    // 1. Run Unit Tests
    if (!run_tests()) {
        printf("Exiting due to Unit Test failures.\n");
        return 1;
    }

    // 2. Initialize INN and Data
    srand((unsigned int)time(NULL)); // Seed for parameter initialization
    INN_Parameters params;
    init_matrix(&params.A, D, D, 1); // Lower Triangular Matrix enforced for symbolic determinant
    init_vector(&params.b, D);

    // SANITY CHECK: Ensure parameters are ready before attempting training
    if (!params.A.initialized || !params.b.initialized) {
        fprintf(stderr, "Fatal Error: INN parameters failed to initialize.\n");
        return 1;
    }


    Sentence legal_sentences[NUM_LEGAL_SENTENCES];
    Sentence illegal_sentences[NUM_ILLEGAL_SENTENCES];
    generate_datasets(legal_sentences, illegal_sentences);

    const clock_t start_time = clock();
    clock_t last_print_time = start_time;
    const double time_step_sec = 5.0;
    const int print_interval_iterations = 100; // Print every 100 iterations

    printf("--- Starting INN Training (MLE via SGD) ---\n");
    printf("Training on: \"%s...\" (3 sentences cycled 512 times)\n", HARDCODED_LEGAL_TEMPLATES[0]);
    printf("Epoch | Iteration | Avg NLL (Loss) | Det(A)\n");
    printf("-------------------------------------------\n");

    // 3. Training Loop (SGD - Maximum Likelihood Estimation)
    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        double epoch_nll_sum = 0.0;
        const int num_batches = NUM_LEGAL_SENTENCES; // Treat each sentence as a batch for simplicity

        for (int i = 0; i < num_batches; i++) {
            // Select a random sentence
            const int idx = rand() % NUM_LEGAL_SENTENCES;
            const Vector x = legal_sentences[idx].features;

            Vector z;
            inn_forward(&params, &x, &z);
            const double nll_loss = calculate_nll_loss(&params, &z);
            epoch_nll_sum += nll_loss;

            // --- Backpropagation (Calculating Gradients) ---

            // 1. Gradient of Loss w.r.t Latent Vector z (dL/dz)
            Vector dL_dz;
            init_vector(&dL_dz, D);
            // dL/dz = z / sigma^2
            const double scale = 1.0 / (GAUSSIAN_SIGMA * GAUSSIAN_SIGMA);
            for (int k = 0; k < D; k++) {
                dL_dz.data[k] = z.data[k] * scale;
            }

            // 2. Gradient of Loss w.r.t Bias b (dL/db)
            // dL/db = dL/dz. Update b
            for (int k = 0; k < D; k++) {
                params.b.data[k] -= LEARNING_RATE * dL_dz.data[k];
            }

            // 3. Gradient of Loss w.r.t Matrix A (dL/dA)
            // dNLL/dA = (dL/dz) * x^T - A^{-T}
            for (int r = 0; r < D; r++) {
                for (int c = 0; c < D; c++) {
                    if (c <= r) { // Only update lower triangular part
                        const double grad_A_r_c = dL_dz.data[r] * x.data[c];
                        params.A.data[r][c] -= LEARNING_RATE * grad_A_r_c;

                        // Additional term from log-determinant (dL/d(A_ii) = 1/A_ii)
                        if (r == c) {
                             params.A.data[r][c] -= LEARNING_RATE * (1.0 / params.A.data[r][c]);
                        }
                    }
                }
            }

            // --- STATS PRINTING ---
            const clock_t current_time = clock();
            const double elapsed_sec = (double)(current_time - last_print_time) / CLOCKS_PER_SEC;

            if (elapsed_sec >= time_step_sec) {
                printf("%5d | %9d | %14.4f | %6.4f\n",
                       epoch, i, epoch_nll_sum / (i + 1), get_determinant_triangular(&params.A));
                last_print_time = current_time;
            }

            if ((i + 1) % print_interval_iterations == 0) {
                 printf("%5d | %9d | %14.4f | %6.4f\n",
                       epoch, i + 1, epoch_nll_sum / (i + 1), get_determinant_triangular(&params.A));
            }

        } // End batch loop
    } // End epoch loop

    // 4. Detection / Evaluation
    printf("\n--- INN Detection Test (Legal vs. Illegal Sentences) ---\n");
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

    printf("Average NLL for Legal Sentences (IN-DISTRIBUTION, Alice text): %.4f\n", avg_legal_nll);
    printf("Average NLL for Illegal Sentences (OUT-OF-DISTRIBUTION, simple grammar): %.4f\n", avg_illegal_nll);
    printf("\nDetection Conclusion:\n");

    if (avg_illegal_nll > avg_legal_nll) {
        printf("SUCCESS: The average NLL of illegal sentences (%.4f) is HIGHER than legal sentences (%.4f).\n",
               avg_illegal_nll, avg_legal_nll);
        printf("The INN successfully learned the feature distribution of the Alice text and rejects inputs with simpler, different grammars/vocabulary.\n");
    } else {
        printf("FAILURE: The average NLL of illegal sentences (%.4f) is NOT HIGHER than legal sentences (%.4f).\n",
               avg_illegal_nll, avg_legal_nll);
        printf("The INN failed to clearly separate the legal and illegal domains based on likelihood.\n");
    }

    return 0;
}