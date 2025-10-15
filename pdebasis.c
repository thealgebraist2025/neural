#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <random>
#include <iomanip>
#include <ranges>

using namespace std;

// --- Configuration ---
constexpr int INPUT_SIZE = 32;
constexpr int KERNEL_SIZE = 3;
constexpr int NUM_CLASSES = 4;
constexpr int NUM_ROTATIONS = 32;
constexpr int INVARIANT_FEATURES = 32; // Size of the final invariant feature vector
constexpr double REG_LAMBDA = 1e-4;   // Orthogonal Regularization strength
constexpr double LEARNING_RATE = 0.001; // Reduced learning rate for stability
constexpr int NUM_EPOCHS = 10;

// M_PI is typically defined in <cmath> (or <math.h>). We rely on the standard definition.
// We remove the line: // constexpr double M_PI = 3.14159265358979323846;

// --- Tensor Class (Minimal) ---
class Tensor {
public:
    vector<double> data;
    vector<int> shape;
    int size = 0;

    Tensor(initializer_list<int> s) : shape(s) {
        size = 1;
        for (int dim : shape) size *= dim;
        data.resize(size, 0.0);
    }
    
    // Access operator for 2D (Used for W, B, X_features, Logits)
    double& operator()(int r, int c) {
        if (shape.size() != 2) throw runtime_error("Invalid 2D access.");
        return data[r * shape[1] + c];
    }
    const double& operator()(int r, int c) const {
        return const_cast<Tensor*>(this)->operator()(r, c);
    }
    
    // Element-wise addition of another Tensor (for gradients)
    Tensor& operator+=(const Tensor& other) {
        if (size != other.size) throw runtime_error("Tensor size mismatch in +=.");
        for(size_t i = 0; i < data.size(); ++i) {
            data[i] += other.data[i];
        }
        return *this;
    }
    
    // Element-wise multiplication by a scalar (for learning rate)
    Tensor operator*(double scalar) const {
        Tensor result = *this;
        for(double& val : result.data) {
            val *= scalar;
        }
        return result;
    }

    // Element-wise subtraction of another Tensor
    Tensor operator-(const Tensor& other) const {
        Tensor result = *this;
        for(size_t i = 0; i < data.size(); ++i) {
            result.data[i] -= other.data[i];
        }
        return result;
    }
};

// Global RNG
random_device rd;
mt19937 gen(rd());
uniform_real_distribution<> distrib(-0.01, 0.01);

void initialize_weights(Tensor& W) {
    for (auto& val : W.data) {
        val = distrib(gen);
    }
}

// --- CORE Equivariant and Regularization Functions ---

// 1. Softmax and Cross-Entropy Loss
Tensor softmax_forward(const Tensor& logits) {
    Tensor probs = logits;
    double max_logit = -numeric_limits<double>::infinity();
    
    // Find max logit for stable exponentiation
    for (double val : probs.data) max_logit = max(max_logit, val);

    double sum_exp = 0.0;
    for (auto& val : probs.data) {
        val = exp(val - max_logit);
        sum_exp += val;
    }
    for (auto& val : probs.data) val /= sum_exp;
    return probs;
}

double cross_entropy_loss(const Tensor& probs, int true_class) {
    // Probs must be a single-sample tensor (1D) of size NUM_CLASSES
    return -log(probs.data[true_class] + 1e-9); // Add epsilon for stability
}

// 2. Backward Pass for Softmax + Cross-Entropy
// Output gradient dL/dLogits (single sample)
Tensor softmax_cross_entropy_backward(const Tensor& probs, int true_class) {
    Tensor dL_dLogits = probs; // dL/dLogits = probs - one_hot
    dL_dLogits.data[true_class] -= 1.0;
    return dL_dLogits;
}

// 3. Soft Orthogonal Regularization Loss ( || W W^T - I ||_F^2 )
double calculate_orthogonal_loss(const Tensor& W) {
    int R = W.shape[0];
    int C = W.shape[1];
    if (R > C) return 0.0; // Only regularize when R <= C

    double regularizer = 0.0;
    // Compute Gram Matrix G = W W^T (R x R)
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < R; ++j) {
            double dot_product = 0.0;
            for (int k = 0; k < C; ++k) {
                dot_product += W(i, k) * W(j, k);
            }
            double target = (i == j) ? 1.0 : 0.0;
            regularizer += pow(dot_product - target, 2);
        }
    }
    return REG_LAMBDA * regularizer;
}

// 4. Gradient of Soft Orthogonal Regularization
// dL_reg / dW = 2 * lambda * (W W^T - I) * W
Tensor orthogonal_grad(const Tensor& W) {
    int R = W.shape[0];
    int C = W.shape[1];
    // This function assumes W is the matrix we are regularizing (R x C)

    // Step 1: Compute G = W W^T (R x R)
    Tensor G({R, R});
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < R; ++j) {
            for (int k = 0; k < C; ++k) {
                G(i, j) += W(i, k) * W(j, k);
            }
        }
    }

    // Step 2: Compute G_err = (G - I) (R x R)
    Tensor G_err = G;
    for (int i = 0; i < R; ++i) G_err(i, i) -= 1.0;

    // Step 3: Compute dL_reg/dW = 2 * lambda * G_err * W
    // (R x R) * (R x C) -> (R x C)
    Tensor dL_reg_dW({R, C});
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
            double sum = 0.0;
            for (int k = 0; k < R; ++k) {
                sum += G_err(i, k) * W(k, j);
            }
            dL_reg_dW(i, j) = 2.0 * REG_LAMBDA * sum;
        }
    }

    return dL_reg_dW;
}

// --- Simplified G-CNN Architecture (Focus on Linear Head Training) ---

// Placeholder: Simulate the final INVARIANT_FEATURES output of the G-CNN backbone
// The input to the dense layer is this rotation-invariant vector.
Tensor simulate_gcnn_invariant_feature(int batch_size) {
    Tensor features({batch_size, INVARIANT_FEATURES});
    uniform_real_distribution<> feature_distrib(0.0, 1.0);
    for (double& val : features.data) {
        val = feature_distrib(gen);
    }
    return features;
}

class LinearClassifier {
public:
    Tensor W; // INVARIANT_FEATURES x NUM_CLASSES (R x C)
    Tensor B; // 1 x NUM_CLASSES

    LinearClassifier() 
        : W({INVARIANT_FEATURES, NUM_CLASSES}), 
          B({1, NUM_CLASSES}) 
    {
        initialize_weights(W);
        initialize_weights(B);
    }

    // Forward Pass (Matrix multiplication + Bias)
    Tensor forward(const Tensor& X) {
        int batch_size = X.shape[0];
        Tensor logits({batch_size, NUM_CLASSES});

        for (int b = 0; b < batch_size; ++b) {
            for (int c_out = 0; c_out < NUM_CLASSES; ++c_out) {
                // Start with bias
                double sum = B(0, c_out); 
                
                // Matmul: X[b, c_in] * W[c_in, c_out]
                for (int c_in = 0; c_in < INVARIANT_FEATURES; ++c_in) {
                    sum += X(b, c_in) * W(c_in, c_out);
                }
                logits(b, c_out) = sum;
            }
        }
        return logits;
    }

    // Backward Pass and SGD Update
    void backward_and_update(const Tensor& X, const Tensor& dL_dLogits, double& total_loss) {
        int batch_size = X.shape[0];
        
        // --- 1. Calculate Gradients for W and B ---
        Tensor dL_dW({INVARIANT_FEATURES, NUM_CLASSES});
        Tensor dL_dB({1, NUM_CLASSES});

        for (int c_out = 0; c_out < NUM_CLASSES; ++c_out) {
            for (int c_in = 0; c_in < INVARIANT_FEATURES; ++c_in) {
                // dL/dW[c_in, c_out] = sum_b( dL/dLogits[b, c_out] * X[b, c_in] )
                for (int b = 0; b < batch_size; ++b) {
                    dL_dW(c_in, c_out) += dL_dLogits(b, c_out) * X(b, c_in);
                }
            }
            // dL/dB[0, c_out] = sum_b( dL/dLogits[b, c_out] )
            for (int b = 0; b < batch_size; ++b) {
                dL_dB(0, c_out) += dL_dLogits(b, c_out);
            }
        }
        
        // --- 2. Add Orthogonal Regularization Gradient to W ---
        // Regularization is applied to the final classification layer weight W.
        // We constrain the weights by transposing W (W^T) and enforcing W^T W approx I 
        // (i.e., orthogonality between the input feature vectors).
        
        // The matrix for regularization is W_reg = W^T (NUM_CLASSES x INVARIANT_FEATURES)
        Tensor W_reg({W.shape[1], W.shape[0]});
        for(int r=0; r<W.shape[0]; ++r)
            for(int c=0; c<W.shape[1]; ++c)
                W_reg(c, r) = W(r, c);

        // Calculate dL/dW_reg (NUM_CLASSES x INVARIANT_FEATURES)
        Tensor dL_reg_dW_reg = orthogonal_grad(W_reg); 
        
        // Transpose gradient back to dL/dW (INVARIANT_FEATURES x NUM_CLASSES)
        Tensor dL_reg_dW({W.shape[0], W.shape[1]});
        for(int r=0; r<dL_reg_dW_reg.shape[0]; ++r)
            for(int c=0; c<dL_reg_dW_reg.shape[1]; ++c)
                dL_reg_dW(c, r) = dL_reg_dW_reg(r, c);

        // Add the regularization penalty to the total loss
        total_loss += calculate_orthogonal_loss(W_reg);

        // Add the regularization gradient
        dL_dW += dL_reg_dW;

        // --- 3. SGD Update ---
        W = W - dL_dW * (LEARNING_RATE / batch_size);
        B = B - dL_dB * (LEARNING_RATE / batch_size);
    }
};

// --- TRAINING LOOP (Simulated Data) ---
int main() {
    cout << fixed << setprecision(6);

    // Simulated data
    constexpr int BATCH_SIZE = 4;
    vector<int> Y_true = {0, 1, 2, 3}; // True classes for the batch

    LinearClassifier classifier;

    cout << "--- Soft Orthogonal Regularization Training (Simulated) ---" << endl;
    cout << "Regularization Lambda: " << REG_LAMBDA << endl;
    cout << "Learning Rate: " << LEARNING_RATE << endl;
    cout << "Matrix Constrained (R x C): " << classifier.W.shape[1] << "x" << classifier.W.shape[0] << endl;
    
    // Initial regularization loss for the transposed matrix W^T
    Tensor W_initial_reg({classifier.W.shape[1], classifier.W.shape[0]});
    for(int r=0; r<classifier.W.shape[0]; ++r)
        for(int c=0; c<classifier.W.shape[1]; ++c)
            W_initial_reg(c, r) = classifier.W(r, c);
            
    double initial_reg_loss = calculate_orthogonal_loss(W_initial_reg);
    cout << "Initial Reg Loss (||W^T W - I||_F^2): " << initial_reg_loss << endl;
    
    // TRAINING
    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
        
        // Simulate invariant features coming from the G-CNN backbone
        Tensor X_features = simulate_gcnn_invariant_feature(BATCH_SIZE);

        // 1. Forward Pass
        Tensor logits = classifier.forward(X_features);
        Tensor probs = softmax_forward(logits);

        double total_loss = 0.0;
        Tensor dL_dLogits({BATCH_SIZE, NUM_CLASSES});

        // 2. Compute Loss and initial gradient dL/dLogits
        for (int b = 0; b < BATCH_SIZE; ++b) {
            
            // --- FIX 2: Manually extract the single-sample Tensor ---
            Tensor sample_probs({NUM_CLASSES});
            size_t start_index = b * NUM_CLASSES;
            for (int c = 0; c < NUM_CLASSES; ++c) {
                sample_probs.data[c] = probs.data[start_index + c];
            }
            // --------------------------------------------------------
            
            double ce_loss = cross_entropy_loss(sample_probs, Y_true[b]);
            total_loss += ce_loss;

            // Compute dL/dLogits for this sample
            Tensor dL_dLogits_sample = softmax_cross_entropy_backward(sample_probs, Y_true[b]);

            // Store in batch gradient tensor
            for(int c=0; c < NUM_CLASSES; ++c) dL_dLogits(b, c) = dL_dLogits_sample.data[c];
        }

        // 3. Backward Pass and Update (Includes Orthogonal Gradient and Loss)
        classifier.backward_and_update(X_features, dL_dLogits, total_loss);
        
        // Report final regularization loss for the epoch
        Tensor W_current_reg({classifier.W.shape[1], classifier.W.shape[0]});
        for(int r=0; r<classifier.W.shape[0]; ++r)
            for(int c=0; c<classifier.W.shape[1]; ++c)
                W_current_reg(c, r) = classifier.W(r, c);
        double final_reg_loss = calculate_orthogonal_loss(W_current_reg);

        // Report
        cout << "\nEpoch " << epoch + 1 << " | Total Loss: " << total_loss / BATCH_SIZE 
             << " | Reg Loss: " << final_reg_loss;
    }

    // Final Check
    Tensor W_final_reg({classifier.W.shape[1], classifier.W.shape[0]});
    for(int r=0; r<classifier.W.shape[0]; ++r)
        for(int c=0; c<classifier.W.shape[1]; ++c)
            W_final_reg(c, r) = classifier.W(r, c);

    double final_reg_loss_val = calculate_orthogonal_loss(W_final_reg);
    cout << "\n\n--- Final State ---" << endl;
    cout << "Final Regularization Loss (||W^T W - I||_F^2): " << final_reg_loss_val << endl;
    
    return 0;
}
