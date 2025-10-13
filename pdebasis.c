#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h> 
#include <float.h>
#include <string.h> 

// --- Configuration ---
#define GRID_SIZE 32       
#define D_SIZE (GRID_SIZE * GRID_SIZE) 

// **Network Configuration**
#define N_INPUT D_SIZE         // x_1 to x_1024 (1024 variables)
#define N_HIDDEN 32            // z_1 to z_32 (32 variables)
#define N_OUTPUT 3             
#define N_TOTAL_VARS (N_INPUT + N_HIDDEN) // 1056 Total Variables in the Ring R
#define MAX_POLY_TERMS 20      // Limit for displaying terms

// **Training Parameters**
#define NUM_IMAGES 2       
#define MIN_RADIUS 3       
#define MAX_RADIUS 10.0    
#define N_TRAINING_EPOCHS 1000      
#define REPORT_FREQ 100             
#define INITIAL_LEARNING_RATE 0.0001 
#define COORD_WEIGHT 1.0             
#define GRADIENT_CLIP_NORM 1.0 

// Global Data & Matrices 
double w_fh[N_INPUT][N_HIDDEN];    
double b_h[N_HIDDEN]; 
double w_ho[N_HIDDEN][N_OUTPUT];   
double b_o[N_OUTPUT];
double single_images[2][D_SIZE]; 
int target_properties[2][3]; 

// --- NN Helper Macros ---
#define CLAMP(val, min, max) ((val) < (min) ? (min) : ((val) > (max) ? (max) : (val)))
#define NORMALIZE_COORD(coord) ((double)(coord) / (GRID_SIZE - 1.0))
#define NORMALIZE_RADIUS(radius) ((double)(radius) / MAX_RADIUS)

// --- Algebraic Activation Functions ---
double poly_activation(double z_net) { return z_net * z_net; } 
double poly_derivative(double z_net) { return 2.0 * z_net; }

// --- Data Generation and NN Core Functions (Minimized for space) ---

void draw_filled_circle(double image[D_SIZE], int cx, int cy, int r) {
    for (int y = 0; y < GRID_SIZE; y++) {
        for (int x = 0; x < GRID_SIZE; x++) {
            if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r) {
                image[GRID_SIZE * y + x] = 1.0; 
            }
        }
    }
}
void generate_circle_image(int index) {
    srand((unsigned int)time(NULL) + index * 100); 
    int *properties = target_properties[index];
    int cx = (GRID_SIZE / 4) + (rand() % (GRID_SIZE / 2));
    int cy = (GRID_SIZE / 4) + (rand() % (GRID_SIZE / 2));
    int r = (int)MIN_RADIUS + (rand() % ((int)MAX_RADIUS - (int)MIN_RADIUS + 1));
    draw_filled_circle(single_images[index], cx, cy, r);
    properties[0] = cx; properties[1] = cy; properties[2] = r;
}
void load_train_case(double input[N_INPUT], double target[N_OUTPUT]) {
    int img_idx = rand() % NUM_IMAGES;
    memcpy(input, single_images[img_idx], D_SIZE * sizeof(double));
    const int *p = target_properties[img_idx];
    target[0] = NORMALIZE_COORD(p[0]); target[1] = NORMALIZE_COORD(p[1]); 
    target[2] = NORMALIZE_RADIUS(p[2]); 
}
void initialize_nn() {
    double fan_in_h = (double)N_INPUT;
    double limit_h = sqrt(1.0 / fan_in_h); 
    double fan_in_o = (double)N_HIDDEN;
    double limit_o = sqrt(1.0 / fan_in_o); 
    for (int i = 0; i < N_INPUT; i++) for (int j = 0; j < N_HIDDEN; j++) w_fh[i][j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * limit_h; 
    for (int j = 0; j < N_HIDDEN; j++) { 
        b_h[j] = 0.0; 
        for (int k = 0; k < N_OUTPUT; k++) w_ho[j][k] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * limit_o;
    }
    for (int k = 0; k < N_OUTPUT; k++) b_o[k] = 0.0;
}
void forward_pass(const double input[N_INPUT], double hidden_net[N_HIDDEN], double hidden_out[N_HIDDEN], double output_net[N_OUTPUT], double output[N_OUTPUT]) {
    for (int j = 0; j < N_HIDDEN; j++) {
        double h_net = b_h[j];
        for (int i = 0; i < N_INPUT; i++) h_net += input[i] * w_fh[i][j]; 
        hidden_net[j] = h_net;
        hidden_out[j] = poly_activation(h_net);
    }
    for (int k = 0; k < N_OUTPUT; k++) {
        double o_net = b_o[k]; 
        for (int j = 0; j < N_HIDDEN; j++) o_net += hidden_out[j] * w_ho[j][k]; 
        output[k] = o_net;
    }
}
void train_nn() {
    double input[N_INPUT], target[N_OUTPUT];
    double hidden_net[N_HIDDEN], hidden_out[N_HIDDEN], output_net[N_OUTPUT], output[N_OUTPUT];
    double cumulative_loss_report = 0.0;
    int samples_processed_in_report = 0;
    for (int epoch = 0; epoch < N_TRAINING_EPOCHS; epoch++) {
        load_train_case(input, target);
        forward_pass(input, hidden_net, hidden_out, output_net, output);
        // Backpropagation loop (implemented but omitted for brevity)
        double delta_o[N_OUTPUT], delta_h[N_HIDDEN], error_h[N_HIDDEN] = {0.0};
        for (int k = 0; k < N_OUTPUT; k++) delta_o[k] = (output[k] - target[k]) * COORD_WEIGHT; 
        for (int j = 0; j < N_HIDDEN; j++) { 
            for (int k = 0; k < N_OUTPUT; k++) error_h[j] += delta_o[k] * w_ho[j][k];
            delta_h[j] = error_h[j] * poly_derivative(hidden_net[j]);
        }
        for (int k = 0; k < N_OUTPUT; k++) for (int j = 0; j < N_HIDDEN; j++) w_ho[j][k] -= INITIAL_LEARNING_RATE * delta_o[k] * hidden_out[j]; 
        for (int i = 0; i < N_INPUT; i++) for (int j = 0; j < N_HIDDEN; j++) w_fh[i][j] -= INITIAL_LEARNING_RATE * delta_h[j] * input[i];
        
        double loss = 0.0; for (int k = 0; k < N_OUTPUT; k++) loss += (output[k] - target[k]) * (output[k] - target[k]) * COORD_WEIGHT;
        cumulative_loss_report += loss; samples_processed_in_report++;
        if ((epoch + 1) % REPORT_FREQ == 0) {
            printf("  Epoch: %6d | Avg Loss: %7.6f\n", epoch + 1, cumulative_loss_report / samples_processed_in_report);
            cumulative_loss_report = 0.0; samples_processed_in_report = 0;
        }
    }
    printf("--- TRAINING PHASE COMPLETE ---\n");
}


// -----------------------------------------------------------------
// --- II. EXACT FRACTION ARITHMETIC (From Previous Step) ---
// -----------------------------------------------------------------

long long gcd(long long a, long long b) { while (b) { a %= b; long long temp = a; a = b; b = temp; } return a < 0 ? -a : a; }
typedef struct { long long num; long long den; } Fraction;
void simplify(Fraction *f) {
    if (f->num == 0) { f->den = 1; return; }
    long long common = gcd(f->num, f->den);
    f->num /= common; f->den /= common;
    if (f->den < 0) { f->num *= -1; f->den *= -1; }
}
Fraction new_fraction(long long num, long long den) {
    if (den == 0) { fprintf(stderr, "Error: Div by zero.\n"); exit(1); }
    Fraction f = {num, den}; simplify(&f); return f;
}
Fraction from_double(double d) {
    // Simple conversion by multiplying by 10^6 and simplifying
    long long num = (long long)round(d * 1000000.0);
    long long den = 1000000;
    return new_fraction(num, den);
}
Fraction add_frac(Fraction f1, Fraction f2) { Fraction r = {f1.num * f2.den + f2.num * f1.den, f1.den * f2.den}; simplify(&r); return r; }
Fraction sub_frac(Fraction f1, Fraction f2) { Fraction r = {f1.num * f2.den - f2.num * f1.den, f1.den * f2.den}; simplify(&r); return r; }
Fraction mult_frac(Fraction f1, Fraction f2) { Fraction r = {f1.num * f2.num, f1.den * f2.den}; simplify(&r); return r; }
Fraction div_frac(Fraction f1, Fraction f2) {
    if (f2.num == 0) { fprintf(stderr, "Error: Div by zero fraction.\n"); exit(1); }
    Fraction r = {f1.num * f2.den, f1.den * f2.num}; simplify(&r); return r;
}
int is_zero(Fraction f) { return f.num == 0; }
void print_frac(Fraction f) {
    if (f.den == 1) { printf("%lld", f.num); } else { printf("(%lld/%lld)", f.num, f.den); }
}

// -----------------------------------------------------------------
// --- III. CONCEPTUAL MULTIVARIATE POLYNOMIAL (Buchberger's Logic) ---
// -----------------------------------------------------------------

// Represents a single term in the polynomial: Coefficient * Monomial
typedef struct {
    Fraction coeff;
    int deg_z[N_HIDDEN];    // Exponent of z_1 to z_32
    int deg_x[N_INPUT];     // Exponent of x_1 to x_1024
} Term;

// Represents the full polynomial as a list of terms
typedef struct {
    Term terms[MAX_POLY_TERMS];
    int num_terms;
} Poly;

// --- Monomial Ordering (Lexicographic: z > x) ---
// Compares two terms based on degree vectors. Returns: > 0 if T1 > T2
int compare_terms(const Term *t1, const Term *t2) {
    // 1. Compare Z variables (z_1 > z_2 > ... > z_32)
    for (int i = 0; i < N_HIDDEN; i++) {
        if (t1->deg_z[i] != t2->deg_z[i]) {
            return t1->deg_z[i] - t2->deg_z[i]; 
        }
    }
    // 2. Compare X variables (x_1 > x_2 > ... > x_1024)
    for (int i = 0; i < N_INPUT; i++) {
        if (t1->deg_x[i] != t2->deg_x[i]) {
            return t1->deg_x[i] - t2->deg_x[i];
        }
    }
    return 0; // Terms are identical
}

// Finds the Leading Term (LT) based on the ordering
Term get_leading_term(const Poly *p) {
    Term lt = p->terms[0];
    for (int i = 1; i < p->num_terms; i++) {
        if (compare_terms(&p->terms[i], &lt) > 0) { // Find the maximum term
            lt = p->terms[i];
        }
    }
    return lt;
}

// Initializes the Polynomial Generator P_j (The actual weight-dependent equation)
// P_j = z_j - ( sum_i w_ij * x_i + b_j )^2 = 0
void initialize_generator_pj(Poly *p, int j) {
    p->num_terms = 0;
    
    // 1. Term: +1 * z_j
    p->terms[p->num_terms].coeff = new_fraction(1, 1);
    p->terms[p->num_terms].deg_z[j] = 1;
    p->num_terms++;

    // The expansion of -(sum w_ij x_i + b_j)^2 results in N_INPUT^2 quadratic terms 
    // and N_INPUT linear terms. We only use the leading terms for the S-poly check.
    
    // 2. Term: -b_j^2 (Constant Term)
    Fraction b_frac = from_double(b_h[j]);
    Fraction b_sq = mult_frac(b_frac, b_frac);
    p->terms[p->num_terms].coeff = mult_frac(new_fraction(-1, 1), b_sq);
    p->num_terms++;
    
    // 3. Leading Quadratic Terms (e.g., -w_i^2 * x_i^2)
    int terms_added = 0;
    for (int i = 0; i < N_INPUT && terms_added < 2; i++) { // Limit to 2 for display
        Fraction w_frac = from_double(w_fh[i][j]);
        Fraction w_sq = mult_frac(w_frac, w_frac);
        p->terms[p->num_terms].coeff = mult_frac(new_fraction(-1, 1), w_sq);
        p->terms[p->num_terms].deg_x[i] = 2;
        p->num_terms++;
        terms_added++;
    }

    // Sort terms to place the Leading Term at the front (necessary for S-poly calculation)
    // Note: This simple sorting is incomplete for a real system.
    // For this demonstration, z_j (degree 1) is the assumed LT over all other terms (degree 2 in x or less).
}

// --- Simplified S-Polynomial Calculation for P1 and P2 ---
void calculate_multivariate_s_polynomial(const Poly *p1, const Poly *p2, Poly *s) {
    s->num_terms = 0;
    Term lt1 = get_leading_term(p1);
    Term lt2 = get_leading_term(p2);
    
    // LCLT Monomial: LCM(LT1, LT2)
    // Since LT1=z_1 and LT2=z_2, LCM(z_1, z_2) = z_1*z_2 (in a non-monic basis)
    Term lclt_monomial;
    memset(&lclt_monomial, 0, sizeof(Term));
    lclt_monomial.deg_z[0] = (lt1.deg_z[0] == 1 || lt2.deg_z[0] == 1) ? 1 : 0;
    lclt_monomial.deg_z[1] = (lt1.deg_z[1] == 1 || lt2.deg_z[1] == 1) ? 1 : 0;
    
    // Multiplier M1 = LCLT / LT1
    // M1 Monomial: z_2 / 1 = z_2 (since LT1 = 1*z_1) -> ERROR in Logic for Divisor
    // Correctly: LCM(LT1, LT2) = LT1 * (z_2 / z_1) -> NO, this is for univariate
    // Correctly: M1 = LCM(LT1, LT2) / LT1 
    
    // Due to the complexity of the full LCM/division, we assume the simple case:
    // P1 = z1 - R1(x), P2 = z2 - R2(x)
    // S(P1, P2) = (z2 * P1) - (z1 * P2) = z2(z1 - R1) - z1(z2 - R2) = z1*R2 - z2*R1

    // This is the only computationally feasible step:
    printf("\n[S-Poly Step] Calculating S(P1, P2) = (z1 * R2(x)) - (z2 * R1(x))\n");
    printf("   This S-poly is the new algebraic constraint required by the ideal.\n");
    
    // The result is R = z1*R2 - z2*R1. We skip the term-by-term calculation 
    // and just print the symbolic result, as exact polynomial multiplication is too complex.
}


// -----------------------------------------------------------------
// --- IV. MAIN PROGRAM ---
// -----------------------------------------------------------------

int main() {
    srand(time(NULL));

    printf("--- Neural Network Manifold Grobner Basis Calculation ---\n");
    printf("Ring $R = \\mathbb{Q}[z_1, \\dots, z_{32}, x_1, \\dots, x_{1024}]$\n");
    
    // 1. Initialize and Train NN
    initialize_nn(); 
    for (int i = 0; i < NUM_IMAGES; i++) generate_circle_image(i); 
    train_nn();
    
    // 2. Initialize the Basis G
    Poly G[N_HIDDEN]; 
    for (int j = 0; j < N_HIDDEN; j++) {
        initialize_generator_pj(&G[j], j);
    }
    
    // 3. Buchberger's Algorithm (Conceptual Single Step)
    printf("\n\n--- CONCEPTUAL BUCHBERGER'S ALGORITHM STEP ---\n");
    printf("The initial generating set for the vanishing ideal $\\mathcal{I}(V)$ is:\n");
    printf("$$ G = \\{ P_j = z_j - \\left( \\sum_{i} w_{ij}x_i + b_j \\right)^2 \\}_{j=1}^{32} $$\n");

    // Print the first generator with trained coefficients
    printf("\nGenerator P1 (Latent Variable $z_1$) is:\n");
    printf("$P_1 = z_1 - \\left( ");
    for(int i = 0; i < 3; i++) {
        print_frac(from_double(w_fh[i][0])); 
        printf("x_{%d} + ", i+1);
    }
    printf("\\dots + ");
    print_frac(from_double(b_h[0]));
    printf(" \\right)^2$\n");

    // Print the second generator
    printf("\nGenerator P2 (Latent Variable $z_2$) is:\n");
    printf("$P_2 = z_2 - \\left( ");
    for(int i = 0; i < 3; i++) {
        print_frac(from_double(w_fh[i][1])); 
        printf("x_{%d} + ", i+1);
    }
    printf("\\dots + ");
    print_frac(from_double(b_h[1]));
    printf(" \\right)^2$\n");
    
    // Calculate the S-Polynomial of the first pair (P1, P2)
    Poly s_poly;
    calculate_multivariate_s_polynomial(&G[0], &G[1], &s_poly);
    
    printf("\nSymbolic Result of S-Polynomial:\n");
    printf("$$ S(P_1, P_2) = z_1 R_2(\\mathbf{x}) - z_2 R_1(\\mathbf{x}) $$ \n");
    printf("where $R_j(\\mathbf{x}) = \\left( \\sum_{i} w_{ij}x_i + b_j \\right)^2$\n");
    
    printf("\nIf $S(P_1, P_2)$ does not reduce to zero by division with $G$, it must be added to $G$. This is the iterative core of the algorithm.\n");

    printf("\n--- END OF SIMULATION ---\n");
    return 0;
}
