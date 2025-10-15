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

// PI definition
constexpr double M_PI = 3.14159265358979323846;

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
    
    // Access operator for 2D (Simplified for Dense Layer W)
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
    return -log(probs.data[true_class] + 1e-9); // Add epsilon for stability
}

// 2. Backward Pass for Softmax + Cross-Entropy
// Output gradient dL/dLogits
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
    if (R > C) return Tensor({R, C}); 

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
    Tensor W; // INVARIANT_FEATURES x NUM_CLASSES
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

    // Backward Pass
    void backward(const Tensor& X, const Tensor& dL_dLogits, double& total_loss) {
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
        // Treat W as R x C where R=INVARIANT_FEATURES, C=NUM_CLASSES
        // For a true orthogonal matrix, we'd constrain the smaller dimension.
        // We will constrain W^T W approx I (C x C constraint)
        
        // For demonstration, we transpose W for the gradient calculation to enforce 
        // orthogonality across the input features (columns): W^T W approx I (C x C)
        
        // Transpose W (NUM_CLASSES x INVARIANT_FEATURES) for regularization gradient
        Tensor W_T({NUM_CLASSES, INVARIANT_FEATURES});
        for(int r=0; r<W.shape[0]; ++r)
            for(int c=0; c<W.shape[1]; ++c)
                W_T(c, r) = W(r, c);

        Tensor dL_reg_dW_T = orthogonal_grad(W_T); // This is dL/dW_T (NUM_CLASSES x INVARIANT_FEATURES)
        
        // Transpose gradient back to dL/dW (INVARIANT_FEATURES x NUM_CLASSES)
        Tensor dL_reg_dW({INVARIANT_FEATURES, NUM_CLASSES});
        for(int r=0; r<dL_reg_dW_T.shape[0]; ++r)
            for(int c=0; c<dL_reg_dW_T.shape[1]; ++c)
                dL_reg_dW(c, r) = dL_reg_dW_T(r, c);

        // Add the regularization penalty to the total loss
        total_loss += calculate_orthogonal_loss(W_T);

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
    cout << "Features Constrained (Rows/Cols): " << classifier.W.shape[1] << "x" << classifier.W.shape[0] << endl;
    
    double initial_reg_loss = calculate_orthogonal_loss({classifier.W.shape[1], classifier.W.shape[0]});
    cout << "Initial Reg Loss (W^T W): " << initial_reg_loss << endl;
    
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
            double ce_loss = cross_entropy_loss(
                Tensor({NUM_CLASSES}, probs.data | ranges::views::slice(b * NUM_CLASSES, (b + 1) * NUM_CLASSES)), 
                Y_true[b]);
            total_loss += ce_loss;

            // Compute dL/dLogits for this sample
            Tensor sample_probs({NUM_CLASSES});
            for(int c=0; c < NUM_CLASSES; ++c) sample_probs.data[c] = probs(b, c);
            Tensor dL_dLogits_sample = softmax_cross_entropy_backward(sample_probs, Y_true[b]);

            // Store in batch gradient tensor
            for(int c=0; c < NUM_CLASSES; ++c) dL_dLogits(b, c) = dL_dLogits_sample.data[c];
        }

        // 3. Backward Pass and Update (Includes Orthogonal Gradient and Loss)
        classifier.backward(X_features, dL_dLogits, total_loss);
        
        // Report
        double final_reg_loss = calculate_orthogonal_loss({classifier.W.shape[1], classifier.W.shape[0]});
        cout << "\nEpoch " << epoch + 1 << " | Total Loss: " << total_loss / BATCH_SIZE 
             << " | Reg Loss: " << final_reg_loss;
    }

    // Final Check
    double final_reg_loss = calculate_orthogonal_loss({classifier.W.shape[1], classifier.W.shape[0]});
    cout << "\n\n--- Final State ---" << endl;
    cout << "Final Regularization Loss (W^T W): " << final_reg_loss << endl;
    
    return 0;
}
