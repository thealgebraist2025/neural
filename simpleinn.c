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
    char text[128];
    Vector features;
    int is_legal;
} Sentence;

// --- VOCABULARY AND SENTENCE GENERATION DATA ---

// Word lists are simplified to map to a continuous feature space.
// The index (or a derived value) serves as the embedding.
const char *const Nouns[] = {"car", "bike", "dog", "lawyer", "judge", "contract", "witness", "defendant"};
const double NounValues[] = {1.0, 1.5, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0};
#define NUM_NOUNS 8

const char *const Verbs[] = {"drives", "reads", "signs", "runs"};
const double VerbValues[] = {1.0, 2.0, 3.0, 4.0};
#define NUM_VERBS 4

const char *const Adjectives[] = {"red", "fast", "legal", "binding", "corrupt"};
const double AdjectiveValues[] = {1.0, 1.5, 2.0, 3.0, 4.0};
#define NUM_ADJECTIVES 5

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
int check_init(const Matrix *M, const char *const name) {
    if (!M->initialized) {
        fprintf(stderr, "Initialization Error: Matrix '%s' not initialized.\n", name);
        return 0;
    }
    return 1;
}

/**
 * @brief Checks if a vector is initialized. (Fix for Test 4)
 * @param V The vector struct pointer.
 * @param name The name of the vector for error printing.
 * @return 1 if initialized, 0 otherwise.
 */
int check_vector_init(const Vector *V, const char *const name) {
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
    // Corrected to use check_vector_init for x
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

/**
 * @brief Calculates the gradient of the log-likelihood w.r.t the input vector.
 * This function is included primarily for completeness but not directly used in SGD on parameters.
 * @param A The INN Weight Matrix.
 * @param z The transformed vector z = A*x + b.
 * @param gradient The output gradient vector.
 */
void calculate_nll_gradient(const Matrix *const A, const Vector *const z, Vector *gradient) {
    // dL/dx = A^T * dL/dz. We assume mu=0 and use dL/dz = z / sigma^2

    if (!check_init(A, "A") || !check_vector_init(z, "z")) return;

    // dL/dz = z / sigma^2
    double scale = 1.0 / (GAUSSIAN_SIGMA * GAUSSIAN_SIGMA);
    Vector dL_dz;
    init_vector(&dL_dz, D);
    for (int i = 0; i < D; i++) {
        dL_dz.data[i] = z->data[i] * scale;
    }

    // dL/dx = A^T * dL/dz
    init_vector(gradient, D);
    for (int i = 0; i < D; i++) { // rows of A^T (cols of A)
        double sum = 0.0;
        for (int j = 0; j < D; j++) { // cols of A^T (rows of A)
            sum += A->data[j][i] * dL_dz.data[j]; // A[j][i] is the element of A^T
        }
        gradient->data[i] = sum;
    }
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

    double log_prob_z = 0.5 * z_norm_sq / (GAUSSIAN_SIGMA * GAUSSIAN_SIGMA);

    // Term 2: -log|det(A)| (using symbolic calculation)
    double det_A = get_determinant_triangular(&params->A);
    double log_det_A = log(fabs(det_A));

    // NLL = log_prob_z - log_det_A
    double nll = log_prob_z - log_det_A;

    return nll;
}

// --- DATASET GENERATION ---

/**
 * @brief Maps a word to its pre-defined continuous feature value.
 */
double get_word_value(const char *const word, const char *const *const word_list, const double *const value_list, int count) {
    for (int i = 0; i < count; i++) {
        if (strcmp(word, word_list[i]) == 0) {
            return value_list[i];
        }
    }
    return 0.0; // Should not happen for valid words
}

/**
 * @brief Generates a single sentence and its feature vector.
 * @param type 1 for "the NOUN VERB the ADJECTIVE NOUN", 2 for "the NOUN has NOUN".
 * @param is_legal 1 for sensible words, 0 for nonsense words (same structure, illegal semantics).
 */
Sentence generate_sentence(int type, int is_legal) {
    Sentence s;
    init_vector(&s.features, D);
    s.is_legal = is_legal;
    s.text[0] = '\0';

    // Type 1: the NOUN VERB the ADJECTIVE NOUN
    if (type == 1) {
        // Legal words have lower indices, illegal words have higher indices.
        int n1_idx = is_legal ? (rand() % 3) : (3 + rand() % (NUM_NOUNS - 3));
        int v_idx = rand() % NUM_VERBS;
        int adj_idx = rand() % NUM_ADJECTIVES;
        int n2_idx = is_legal ? (rand() % 3) : (3 + rand() % (NUM_NOUNS - 3));

        // Construct text
        sprintf(s.text, "the %s %s the %s %s",
                Nouns[n1_idx], Verbs[v_idx], Adjectives[adj_idx], Nouns[n2_idx]);

        // Construct features
        s.features.data[0] = NounValues[n1_idx];
        s.features.data[1] = VerbValues[v_idx];
        s.features.data[2] = AdjectiveValues[adj_idx];
        s.features.data[3] = NounValues[n2_idx];
        s.features.data[4] = 0.0; // Context feature for Type 1
    }
    // Type 2: the NOUN has NOUN
    else if (type == 2) {
        int n1_idx = is_legal ? (rand() % 4) : (4 + rand() % (NUM_NOUNS - 4));
        int n2_idx = is_legal ? (rand() % 4) : (4 + rand() % (NUM_NOUNS - 4));

        // Construct text
        sprintf(s.text, "the %s has %s",
                Nouns[n1_idx], Nouns[n2_idx]);

        // Construct features (mapping 'has' to a specific VerbValue for consistency)
        s.features.data[0] = NounValues[n1_idx];
        s.features.data[1] = 5.0; // Fixed value for "has"
        s.features.data[2] = 0.0; // Not applicable for Type 2
        s.features.data[3] = NounValues[n2_idx];
        s.features.data[4] = 1.0; // Context feature for Type 2
    }

    return s;
}

/**
 * @brief Generates the training and testing datasets.
 */
void generate_datasets(Sentence *legal_sentences, Sentence *illegal_sentences) {
    srand((unsigned int)time(NULL));

    // Generate Legal Sentences
    for (int i = 0; i < NUM_LEGAL_SENTENCES; i++) {
        int type = (i % 2) + 1; // Alternate between Type 1 and Type 2
        legal_sentences[i] = generate_sentence(type, 1);
    }

    // Generate Illegal Sentences (Nonsense)
    for (int i = 0; i < NUM_ILLEGAL_SENTENCES; i++) {
        int type = (i % 2) + 1; // Alternate between Type 1 and Type 2
        illegal_sentences[i] = generate_sentence(type, 0);
    }

    printf("Dataset generated: %d legal, %d illegal sentences.\n", NUM_LEGAL_SENTENCES, NUM_ILLEGAL_SENTENCES);
}

// --- UNIT TESTING FRAMEWORK ---

/**
 * @brief Runs all unit tests for the utility functions and core INN logic.
 * @return 1 if all tests pass, 0 otherwise.
 */
int run_tests() {
    printf("--- Running Unit Tests ---\n");
    int failed = 0;

    // Test 1: Initialization Check (Matrix and its error message)
    Matrix M_uninit;
    M_uninit.initialized = 0;
    if (check_init(&M_uninit, "M_uninit")) {
        printf("Test 1 (Init Check Matrix): FAILED (Should not be initialized)\n");
        failed++;
    } else {
        printf("Test 1 (Init Check Matrix): PASSED\n");
    }

    // Test 1b: Initialization Check (Vector and its error message)
    Vector V_uninit;
    V_uninit.initialized = 0;
    if (check_vector_init(&V_uninit, "V_uninit")) {
        printf("Test 1b (Init Check Vector): FAILED (Should not be initialized)\n");
        failed++;
    } else {
        printf("Test 1b (Init Check Vector): PASSED\n");
    }


    // Test 2: Matrix Initialization and Triangular Check
    Matrix A;
    init_matrix(&A, D, D, 1);
    int triangular_ok = 1;
    for (int i = 0; i < D; i++) {
        for (int j = i + 1; j < D; j++) {
            if (A.data[i][j] != 0.0) {
                triangular_ok = 0;
                break;
            }
        }
        if (!triangular_ok) break;
    }
    if (triangular_ok && A.initialized) {
        printf("Test 2 (Matrix Init): PASSED\n");
    } else {
        printf("Test 2 (Matrix Init): FAILED (Not triangular or not initialized)\n");
        failed++;
    }

    // Test 3: Determinant of Triangular Matrix (Symbolic Check)
    // Set a known diagonal: 1, 2, 3, 4, 5. Determinant should be 1*2*3*4*5 = 120.0
    for (int i = 0; i < D; i++) A.data[i][i] = (double)(i + 1);
    double det = get_determinant_triangular(&A);
    if (fabs(det - 120.0) < 1e-6) {
        printf("Test 3 (Determinant): PASSED (Value: %.1f)\n", det);
    } else {
        printf("Test 3 (Determinant): FAILED (Expected 120.0, Got %.2f)\n", det);
        failed++;
    }

    // Test 4: Matrix-Vector Multiplication (INN Forward Pass Check) - FIXED
    Vector x, y_expected, z;
    init_vector(&x, D); // Input vector x is initialized
    init_vector(&y_expected, D);
    // Set x to [1, 1, 1, 1, 1]
    for (int i = 0; i < D; i++) x.data[i] = 1.0;
    // Set A to lower triangular, all ones below/on diagonal. y[i] = i+1
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < D; j++) {
            A.data[i][j] = (j <= i) ? 1.0 : 0.0;
        }
        y_expected.data[i] = (double)(i + 1); // e.g. y[0]=1, y[4]=5
    }
    multiply_matrix_vector(&A, &x, &z);
    int mv_ok = 1;
    for (int i = 0; i < D; i++) {
        if (fabs(z.data[i] - y_expected.data[i]) > 1e-6) {
            mv_ok = 0;
            break;
        }
    }
    if (mv_ok) {
        printf("Test 4 (Matrix-Vector): PASSED\n");
    } else {
        printf("Test 4 (Matrix-Vector): FAILED\n");
        failed++;
    }

    // Test 5: INN Training on Simple Case (NLL calculation)
    INN_Parameters test_params;
    init_matrix(&test_params.A, D, D, 1);
    init_vector(&test_params.b, D);
    // Ideal case: A=I, b=0, x=0. Then z=0. NLL should be 0.
    for (int i = 0; i < D; i++) {
        test_params.A.data[i][i] = 1.0;
        test_params.A.data[i][i == 0 ? 1 : i-1] = 0.0; // Ensure off-diagonal is reset
        test_params.b.data[i] = 0.0;
        x.data[i] = 0.0;
    }
    inn_forward(&test_params, &x, &z);
    double nll_ideal = calculate_nll_loss(&test_params, &z);
    if (fabs(nll_ideal - 0.0) < 1e-6) {
        printf("Test 5 (NLL Ideal): PASSED (NLL=%.2f)\n", nll_ideal);
    } else {
        printf("Test 5 (NLL Ideal): FAILED (Expected 0.0, Got %.2f)\n", nll_ideal);
        failed++;
    }


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
    INN_Parameters params;
    init_matrix(&params.A, D, D, 1); // Lower Triangular Matrix enforced for symbolic determinant
    init_vector(&params.b, D);

    // SANITY CHECK: Ensure parameters are ready before attempting training
    if (!params.A.initialized) {
        fprintf(stderr, "Fatal Error: INN Weight Matrix (A) failed to initialize.\n");
        return 1;
    }
    if (!params.b.initialized) {
        fprintf(stderr, "Fatal Error: INN Bias Vector (b) failed to initialize.\n");
        return 1;
    }


    Sentence legal_sentences[NUM_LEGAL_SENTENCES];
    Sentence illegal_sentences[NUM_ILLEGAL_SENTENCES];
    generate_datasets(legal_sentences, illegal_sentences);

    clock_t start_time = clock();
    clock_t last_print_time = start_time;
    double time_step_sec = 5.0;
    int print_interval_iterations = 100; // Print every 100 iterations

    printf("--- Starting INN Training (MLE via SGD) ---\n");
    printf("Epoch | Iteration | Avg NLL (Loss) | Det(A)\n");
    printf("-------------------------------------------\n");

    // 3. Training Loop (SGD - Maximum Likelihood Estimation)
    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        double epoch_nll_sum = 0.0;
        int num_batches = NUM_LEGAL_SENTENCES; // Treat each sentence as a batch for simplicity

        for (int i = 0; i < num_batches; i++) {
            // Select a random sentence
            int idx = rand() % NUM_LEGAL_SENTENCES;
            const Vector x = legal_sentences[idx].features;

            Vector z;
            inn_forward(&params, &x, &z);
            double nll_loss = calculate_nll_loss(&params, &z);
            epoch_nll_sum += nll_loss;

            // --- Backpropagation (Calculating Gradients) ---

            // 1. Gradient of Loss w.r.t Latent Vector z (dL/dz)
            Vector dL_dz;
            init_vector(&dL_dz, D);
            // dL/dz = z / sigma^2
            double scale = 1.0 / (GAUSSIAN_SIGMA * GAUSSIAN_SIGMA);
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
                        double grad_A_r_c = dL_dz.data[r] * x.data[c];
                        params.A.data[r][c] -= LEARNING_RATE * grad_A_r_c;

                        // Additional term from log-determinant (dL/d(A_ii) = 1/A_ii)
                        if (r == c) {
                             params.A.data[r][c] -= LEARNING_RATE * (1.0 / params.A.data[r][c]);
                        }
                    }
                }
            }

            // --- STATS PRINTING ---
            clock_t current_time = clock();
            double elapsed_sec = (double)(current_time - last_print_time) / CLOCKS_PER_SEC;

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

    double avg_legal_nll = legal_nll_sum / legal_count;
    double avg_illegal_nll = illegal_nll_sum / illegal_count;

    printf("Average NLL for Legal Sentences (IN-DISTRIBUTION): %.4f\n", avg_legal_nll);
    printf("Average NLL for Illegal Sentences (OUT-OF-DISTRIBUTION): %.4f\n", avg_illegal_nll);
    printf("\nDetection Conclusion:\n");

    // The INN should map the trained distribution to a simple Gaussian (low NLL).
    // The illegal sentences should result in a higher NLL (low probability in the target distribution).
    if (avg_illegal_nll > avg_legal_nll) {
        printf("SUCCESS: The average NLL of illegal sentences (%.4f) is HIGHER than legal sentences (%.4f).\n",
               avg_illegal_nll, avg_legal_nll);
        printf("The INN successfully learned the legal domain's density and rejects non-conforming inputs (higher NLL = lower likelihood).\n");
    } else {
        printf("FAILURE: The average NLL of illegal sentences (%.4f) is NOT HIGHER than legal sentences (%.4f).\n",
               avg_illegal_nll, avg_legal_nll);
        printf("The INN failed to clearly separate the legal and illegal domains based on likelihood.\n");
    }

    return 0;
}