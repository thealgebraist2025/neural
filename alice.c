#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <float.h>
#include <time.h> 

// --- SCALED CONFIGURATION (For a full book corpus) ---
#define N_CONTEXTS 2000 // Max number of sliding context windows (M)
#define N_NODES 50     // Vocabulary size (N)
#define MAX_TOKENS 65536 // Maximum number of tokens to process from the book

// --- SPARSE MATRIX CONSTANTS ---
#define MAX_SPARSE_ENTRIES (N_NODES * N_NODES) 

// --- WORD INDICES (Key words tracked for the semantic analysis) ---
#define ALICE   0
#define HATTER  1
#define LIKES   2
#define MAD     3
#define QUEEN   4 
#define RABBIT  5 
#define TEA     6 
#define DREAM   7 
#define VOCAB_SIZE 8 // Number of defined words. Indices 8-49 are filler.

#define MAX_ITERATIONS 3000
#define CONVERGENCE_TOLERANCE 1e-7
#define CONTEXT_WINDOW_SIZE 8 // Sliding window size for co-occurrence

// --- FILE CONSTANT ---
const char* BOOK_FILENAME = "alice.txt";


// --- SPARSE DATA STRUCTURE ---
typedef struct {
    int row;
    int col;
    double value;
} SparseEntry;

SparseEntry C_sparse[MAX_SPARSE_ENTRIES];
int num_sparse_entries = 0; 
double D_matrix[N_CONTEXTS][N_NODES];


// --- VOCABULARY MAPPING ---
// Simple map to link words to indices.
int get_word_index(const char* word) {
    if (strcmp(word, "alice") == 0) return ALICE;
    if (strcmp(word, "hatter") == 0) return HATTER;
    if (strcmp(word, "likes") == 0) return LIKES;
    if (strcmp(word, "mad") == 0) return MAD;
    if (strcmp(word, "queen") == 0) return QUEEN;
    if (strcmp(word, "rabbit") == 0) return RABBIT;
    if (strcmp(word, "tea") == 0) return TEA;
    if (strcmp(word, "dream") == 0) return DREAM;
    
    // Simple hashing for filler words (Indices 8 to 49)
    if (strlen(word) > 2) {
        // Use a simple hash: first three chars combined with length, mapped to 8-49
        unsigned long hash = word[0] + (word[1] << 1) + (word[2] << 2) + strlen(word);
        return (hash % (N_NODES - VOCAB_SIZE)) + VOCAB_SIZE; 
    }
    return -1; // Ignore very short words/punctuation remnants (like "a", "of")
}


// --- D MATRIX BUILDER (FROM RAW TEXT FILE) ---
/* Reads the raw text file, tokenizes it, and counts co-occurrences in sliding windows
   to populate the D_matrix. M is the number of context windows processed. */
int read_book_file_and_build_D_matrix(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error: Could not open file %s. Check that 'alice.txt' exists at runtime.\n", filename);
        return 0;
    }

    // Determine file size and allocate buffer for the entire file content
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    rewind(file);

    // Safety check for excessively large files (limit to 1MB)
    if (file_size > 1024 * 1024) {
        file_size = 1024 * 1024; // Process only the first 1MB for safety/performance
    }
    
    char* full_text = (char*)malloc(file_size + 1);
    if (!full_text) {
        fprintf(stderr, "Memory allocation failed for file buffer.\n");
        fclose(file);
        return 0;
    }

    // Read file content and null-terminate
    size_t read_size = fread(full_text, 1, file_size, file);
    full_text[read_size] = '\0'; 
    fclose(file);
    
    int token_indices[MAX_TOKENS]; // Array to store the vocabulary index of each token
    int token_count = 0;
    
    // 1. Tokenization and Normalization
    char* token = strtok(full_text, " ,.?!'();:\"\n\t");
    while (token != NULL && token_count < MAX_TOKENS) {
        // Convert to lowercase
        for (char *c = token; *c; c++) {
            *c = tolower(*c);
        }
        
        int index = get_word_index(token);
        
        if (index != -1) {
            token_indices[token_count] = index;
            token_count++;
        }
        token = strtok(NULL, " ,.?!'();:\"\n\t");
    }

    // 2. Co-occurrence Counting (Sliding Window)
    int context_counter = 0;
    
    // First, clear the D_matrix
    memset(D_matrix, 0, sizeof(D_matrix));

    for (int i = 0; i < token_count - CONTEXT_WINDOW_SIZE + 1; i++) {
        if (context_counter >= N_CONTEXTS) break;
        
        // Count all words within the window [i, i + CONTEXT_WINDOW_SIZE - 1]
        for (int w = 0; w < CONTEXT_WINDOW_SIZE; w++) {
            int word_idx = token_indices[i + w];
            // Increment the word's count for this specific context window
            D_matrix[context_counter][word_idx] += 1.0;
        }
        context_counter++;
    }
    
    free(full_text); // Release the memory used for the text buffer
    return context_counter; 
}


// --- UTILITY FUNCTIONS ---

double dot_product(const double v1[], const double v2[], int n) {
    double product = 0.0;
    for (int i = 0; i < n; i++) {
        product += v1[i] * v2[i];
    }
    return product;
}

double magnitude(const double v[], int n) {
    double sum_sq = 0.0;
    for (int i = 0; i < n; i++) {
        sum_sq += v[i] * v[i];
    }
    return sqrt(sum_sq);
}

void normalize_vector(double vec[], int n) {
    double mag = magnitude(vec, n);
    if (mag > DBL_EPSILON) {
        for (int i = 0; i < n; i++) {
            vec[i] /= mag;
        }
    }
}

// --- CORE SPARSE OPERATION (O(Non-Zero Entries)) ---
// Performs a sparse matrix-vector multiplication (C * vec_in)
void sparse_matrix_vector_multiply(const double vec_in[], double vec_out[], int n_nodes) {
    for (int i = 0; i < n_nodes; i++) {
        vec_out[i] = 0.0;
    }
    
    // Only iterate over the stored non-zero entries (O(Non-Zero Count))
    for (int k = 0; k < num_sparse_entries; k++) {
        int i = C_sparse[k].row;
        int j = C_sparse[k].col;
        double val = C_sparse[k].value;
        
        vec_out[i] += val * vec_in[j];
    }
}


// --- COVARIANCE CALCULATION AND SPARSE STORAGE ---
// Calculates the Covariance Matrix C and stores it in the sparse format.
void calculate_and_store_sparse_covariance(const double D[N_CONTEXTS][N_NODES], int actual_contexts) {
    if (actual_contexts <= 1) {
        fprintf(stderr, "Error: Insufficient data (%d contexts) for covariance calculation.\n", actual_contexts);
        return;
    }

    double means[N_NODES] = {0};
    double centered_data[N_CONTEXTS][N_NODES]; 
    
    int i, j, j1, j2;
    const double N_MINUS_ONE = (double)actual_contexts - 1.0;

    // 1. Calculate means and center data
    for (j = 0; j < N_NODES; j++) {
        for (i = 0; i < actual_contexts; i++) {
            means[j] += D[i][j];
        }
        means[j] /= (double)actual_contexts;
    }

    for (i = 0; i < actual_contexts; i++) {
        for (j = 0; j < N_NODES; j++) {
            centered_data[i][j] = D[i][j] - means[j];
        }
    }

    // 2. Calculate and store the Covariance C into the sparse C_sparse format (Coordinate List)
    num_sparse_entries = 0;
    double c_val;
    for (j1 = 0; j1 < N_NODES; j1++) { 
        for (j2 = j1; j2 < N_NODES; j2++) { 
            
            // Calculate C[j1][j2] (O(M) operation)
            double sum_of_products = 0.0;
            for (i = 0; i < actual_contexts; i++) {
                sum_of_products += centered_data[i][j1] * centered_data[i][j2];
            }
            c_val = sum_of_products / N_MINUS_ONE;
            
            if (fabs(c_val) > 1e-4 && num_sparse_entries < MAX_SPARSE_ENTRIES - 2) { // Check for sparsity
                
                C_sparse[num_sparse_entries].row = j1;
                C_sparse[num_sparse_entries].col = j2;
                C_sparse[num_sparse_entries].value = c_val;
                num_sparse_entries++;
                
                if (j1 != j2) {
                    C_sparse[num_sparse_entries].row = j2;
                    C_sparse[num_sparse_entries].col = j1;
                    C_sparse[num_sparse_entries].value = c_val;
                    num_sparse_entries++;
                }
            }
        }
    }
}

// --- SPECTRAL CALCULATION: POWER ITERATION ---
double power_iteration(double dominant_eigenvector[]) {
    double b_prev[N_NODES];
    double b[N_NODES];
    double lambda = 0.0;
    double lambda_prev = 0.0;
    int k;

    for(k=0; k < N_NODES; k++) b_prev[k] = 1.0; 
    normalize_vector(b_prev, N_NODES);

    for (k = 0; k < MAX_ITERATIONS; k++) {
        sparse_matrix_vector_multiply(b_prev, b, N_NODES); 
        
        lambda = dot_product(b, b_prev, N_NODES);

        if (k > 0 && fabs(lambda - lambda_prev) < CONVERGENCE_TOLERANCE) {
            break;
        }
        lambda_prev = lambda;

        for (int i = 0; i < N_NODES; i++) {
            b_prev[i] = b[i];
        }
        normalize_vector(b_prev, N_NODES);
    }

    for (int i = 0; i < N_NODES; i++) {
        dominant_eigenvector[i] = b_prev[i];
    }

    return lambda;
}

// --- SPECTRAL SAMPLING (Log-Determinant Approximation) ---
double approximate_determinant_sampling(double lambda_dominant) {
    double total_trace = 0.0;
    double log_det_sum = 0.0;
    int i;

    // 1. Calculate the Trace(C) (Sum of Variances)
    for (i = 0; i < num_sparse_entries; i++) {
        if (C_sparse[i].row == C_sparse[i].col) {
            total_trace += C_sparse[i].value; 
        }
    }

    double remaining_trace = total_trace - lambda_dominant;
    if (remaining_trace < DBL_EPSILON) remaining_trace = DBL_EPSILON;

    double average_remaining_lambda = remaining_trace / (N_NODES - 1);

    log_det_sum += log(lambda_dominant);

    // 2. Simulate the remaining log sum (complexity)
    for (i = 1; i < N_NODES; i++) {
        double sampled_lambda = average_remaining_lambda + ( (double)rand() / RAND_MAX - 0.5 ) * (average_remaining_lambda * 0.5);

        if (sampled_lambda > DBL_EPSILON) {
            log_det_sum += log(sampled_lambda);
        }
    }

    return log_det_sum;
}


int main() {
    double lambda_dominant;
    double v_dominant[N_NODES] = {0};
    double log_determinant;
    int actual_contexts;

    srand((unsigned int)time(NULL));

    printf("--- Spectral Linguistic Solver (C99) ---\n");
    printf("Method: SPARSE Matrix Iteration + File Processing.\n");
    printf("Corpus: Read from file '%s'.\n", BOOK_FILENAME);
    printf("Target Sentence: \"Alice likes the mad hatter\"\n\n");
    
    // 1. READ BOOK FILE AND BUILD D MATRIX
    actual_contexts = read_book_file_and_build_D_matrix(BOOK_FILENAME);

    if (actual_contexts == 0) {
        printf("Analysis aborted: Could not read or process data from the file.\n");
        return 1;
    }
    
    // 2. Calculate Covariance Matrix and store it in the efficient sparse format
    calculate_and_store_sparse_covariance(D_matrix, actual_contexts);

    printf("--- Corpus Summary ---\n");
    printf("Actual Context Windows (M): %d\n", actual_contexts);
    printf("Vocabulary Size (N): %d\n", N_NODES);
    printf("Total non-zero entries (incl. symmetry): %d\n", num_sparse_entries);
    printf("Sparsity Level: %.2f%%\n", 100.0 * (1.0 - (double)num_sparse_entries / (N_NODES * N_NODES)));
    printf("------------------------\n\n");

    // 3. Calculate Dominant Feature using sparse multiplication
    lambda_dominant = power_iteration(v_dominant);

    // 4. Approximate Log-Determinant
    log_determinant = approximate_determinant_sampling(lambda_dominant);

    printf("1. CALCULATED SPECTRAL PROPERTIES:\n");
    printf("----------------------------------\n");

    printf("A) Approximate Log-Determinant (Volume/Complexity): %.4f\n\n", log_determinant);

    printf("B) Dominant Eigenvalue and Eigenvector:\n");
    printf("   Lambda_max (Variance Explained): %.4f\n", lambda_dominant);
    printf("   V_max (Semantic Feature Vector - Key Terms):\n");
    printf("     ALICE (%.4f) | HATTER (%.4f) | LIKES (%.4f) | MAD (%.4f)\n",
           v_dominant[ALICE], v_dominant[HATTER], v_dominant[LIKES], v_dominant[MAD]);
    printf("     QUEEN (%.4f) | RABBIT (%.4f) | TEA (%.4f) | DREAM (%.4f)\n\n",
           v_dominant[QUEEN], v_dominant[RABBIT], v_dominant[TEA], v_dominant[DREAM]);


    // 5. LINGUISTIC EVALUATION
    printf("2. PLAUSIBILITY CHECK: \"Alice likes the mad hatter\"\n");
    printf("---------------------------------------------------\n");

    double alice_hatter_alignment = v_dominant[ALICE] * v_dominant[HATTER];
    double likes_weight = v_dominant[LIKES];

    printf("   Alignment (Alice * Hatter): %.4f\n", alice_hatter_alignment);
    printf("   Weight of 'Likes': %.4f\n\n", likes_weight);

    printf("3. FINAL EVALUATION:\n");
    if (alice_hatter_alignment < -0.05) {
        printf("   => STRUCTURALLY IMPLAUSIBLE (Oppositional).\n");
        printf("      Reason: ALICE and HATTER occupy structurally distinct or opposing roles on the primary semantic feature derived from the corpus.\n");
    } else if (alice_hatter_alignment < 0.05) {
        printf("   => NEUTRAL/AMBIGUOUS.\n");
        printf("      Reason: Their relationship is weakly defined by the dominant spectral feature (close to zero alignment).\n");
    } else {
        printf("   => STRUCTURALLY PLAUSIBLE (Supported).\n");
        printf("      Reason: ALICE and HATTER co-occur frequently enough in similar contexts to show a positive alignment on the principal semantic feature.\n");
    }

    printf("\n--- END OF ANALYSIS ---\n");

    return 0;
}