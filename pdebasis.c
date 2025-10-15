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
constexpr double PI_CONST = 3.14159265358979323846; 
constexpr int INPUT_SIZE = 32;
constexpr int KERNEL_SIZE = 3;
constexpr int NUM_CLASSES = 4;
constexpr int NUM_ROTATIONS = 32; // G-Group Size (C32)
constexpr int L1_C_OUT = 8;
constexpr int L2_C_OUT = 16;
constexpr int L3_C_OUT = 32;
constexpr int INVARIANT_FEATURES = L3_C_OUT; 
constexpr double REG_LAMBDA = 1e-4; // Orthogonal regularization strength
constexpr double LEARNING_RATE = 0.001;
constexpr int NUM_EPOCHS = 3; // Reduced for quick simulation
constexpr int BATCH_SIZE = 32;
constexpr int TRAIN_SAMPLES = 1000;
constexpr int TEST_SAMPLES = 200;

// --- Tensor Class ---
class Tensor {
public:
    vector<double> data;
    vector<int> shape;
    int size = 0;

    Tensor() : size(0) {} 
    
    Tensor(initializer_list<int> s) : shape(s) {
        size = 1;
        for (int dim : shape) size *= dim;
        data.resize(size, 0.0);
    }
    
    // 2D Access
    double& operator()(int r, int c) {
        if (shape.size() != 2) throw runtime_error("Invalid 2D access (2D).");
        return data[r * shape[1] + c];
    }
    const double& operator()(int r, int c) const {
        return const_cast<Tensor*>(this)->operator()(r, c);
    }
    
    // 4D Access
    double& operator()(int d1, int d2, int d3, int d4) {
        if (shape.size() != 4) throw runtime_error("Invalid 4D access (4D).");
        return data[d1 * shape[1] * shape[2] * shape[3] +
                    d2 * shape[2] * shape[3] +
                    d3 * shape[3] +
                    d4];
    }
    const double& operator()(int d1, int d2, int d3, int d4) const {
        return const_cast<Tensor*>(this)->operator()(d1, d2, d3, d4);
    }
    
    // Arithmetic operators
    Tensor& operator+=(const Tensor& other) {
        if (size != other.size) throw runtime_error("Tensor size mismatch in +=.");
        for(size_t i = 0; i < data.size(); ++i) {
            data[i] += other.data[i];
        }
        return *this;
    }
    Tensor operator*(double scalar) const {
        Tensor result = *this;
        for(double& val : result.data) {
            val *= scalar;
        }
        return result; 
    }
    Tensor operator-(const Tensor& other) const { 
        Tensor result = *this; 
        for(size_t i = 0; i < data.size(); ++i) { 
            result.data[i] -= other.data[i]; 
        } 
        return result; 
    }
};

// Global RNG and Initializer
random_device rd; mt19937 gen(rd()); uniform_real_distribution<> weight_distrib(-0.01, 0.01);
uniform_int_distribution<> label_distrib(0, NUM_CLASSES - 1); uniform_int_distribution<> rotation_distrib(0, NUM_ROTATIONS - 1);
void initialize_weights(Tensor& W) { for (auto& val : W.data) val = weight_distrib(gen); }

// --- DATASET & UTILITIES ---

Tensor rotate_2d_slice(const Tensor& input_2d, int k_rotations) {
    int N = input_2d.shape[0]; Tensor output({N, N}); 
    // Simplified rotation logic: return identity for backward pass focus
    for(int i=0; i<N; ++i) for(int j=0; j<N; ++j) output(i, j) = input_2d(i, j);
    return output;
}

Tensor generate_input_image(int class_label, int rotation_index) {
    Tensor image({INPUT_SIZE, INPUT_SIZE});
    int start_i = 10; int size = 10;
    for (int i = start_i; i < start_i + size; ++i) {
        for (int j = 10; j < 10 + size; ++j) {
            if (i < INPUT_SIZE && j < INPUT_SIZE) { image(i, j) = 1.0; }
        }
    }
    return rotate_2d_slice(image, rotation_index);
}

struct DataSet { vector<Tensor> X; vector<int> Y_true; };
DataSet generate_dataset(int num_samples) { 
    DataSet data;
    data.X.reserve(num_samples);
    data.Y_true.resize(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        data.Y_true[i] = label_distrib(gen);
        data.X.push_back(generate_input_image(data.Y_true[i], rotation_distrib(gen)));
    }
    return data;
}

// --- FORWARD/LOSS UTILITIES (Step 3) ---

Tensor softmax_forward(const Tensor& logits) {
    Tensor probs({logits.shape[0], logits.shape[1]});
    for(int b = 0; b < logits.shape[0]; ++b) {
        double max_val = *max_element(logits.data.begin() + b * logits.shape[1], 
                                     logits.data.begin() + (b + 1) * logits.shape[1]);
        double sum_exp = 0.0;
        for(int c = 0; c < logits.shape[1]; ++c) {
            probs(b, c) = exp(logits(b, c) - max_val);
            sum_exp += probs(b, c);
        }
        for(int c = 0; c < logits.shape[1]; ++c) {
            probs(b, c) /= sum_exp;
        }
    }
    return probs;
}

double cross_entropy_loss(const Tensor& probs, const vector<int>& Y_true) {
    double loss = 0.0;
    for(size_t b = 0; b < Y_true.size(); ++b) {
        loss += -log(max(probs(b, Y_true[b]), 1e-9));
    }
    return loss / Y_true.size();
}

Tensor softmax_cross_entropy_backward(const Tensor& probs, const vector<int>& Y_true) {
    Tensor dL_dLogits = probs;
    for(size_t b = 0; b < Y_true.size(); ++b) {
        dL_dLogits(b, Y_true[b]) -= 1.0;
    }
    double scale = 1.0 / Y_true.size();
    return dL_dLogits * scale;
}

// Step 3: Orthogonal Regularization for Classifier Weights W
double calculate_orthogonal_loss(const Tensor& W) {
    int R = W.shape[0];
    int C = W.shape[1];
    
    // Calculate W^T * W (C x C matrix)
    Tensor W_T_W({C, C});
    for (int i = 0; i < C; ++i) {
        for (int j = 0; j < C; ++j) {
            double sum = 0.0;
            for (int r = 0; r < R; ++r) {
                sum += W(r, i) * W(r, j);
            }
            W_T_W(i, j) = sum;
        }
    }
    
    // Calculate ||W^T * W - I||^2
    double reg_loss = 0.0;
    for (int i = 0; i < C; ++i) {
        for (int j = 0; j < C; ++j) {
            double target = (i == j) ? 1.0 : 0.0;
            reg_loss += pow(W_T_W(i, j) - target, 2);
        }
    }
    return reg_loss * REG_LAMBDA;
}

// Gradient of Orthogonal Regularization
Tensor orthogonal_grad(const Tensor& W) {
    int R = W.shape[0];
    int C = W.shape[1];
    
    // W^T * W (C x C)
    Tensor W_T_W({C, C});
    for (int i = 0; i < C; ++i) {
        for (int j = 0; j < C; ++j) {
            double sum = 0.0;
            for (int r = 0; r < R; ++r) {
                sum += W(r, i) * W(r, j);
            }
            W_T_W(i, j) = sum;
        }
    }

    // W_T_W_minus_I = W^T * W - I (C x C)
    Tensor W_T_W_minus_I = W_T_W;
    for (int i = 0; i < C; ++i) {
        W_T_W_minus_I(i, i) -= 1.0;
    }

    // dL_ortho / dW = 4 * W * (W^T * W - I)
    Tensor dL_ortho_dW({R, C});
    for (int r = 0; r < R; ++r) {
        for (int c_out = 0; c_out < C; ++c_out) {
            double sum = 0.0;
            for (int c_mid = 0; c_mid < C; ++c_mid) {
                sum += W(r, c_mid) * W_T_W_minus_I(c_mid, c_out);
            }
            dL_ortho_dW(r, c_out) = 4.0 * sum * REG_LAMBDA;
        }
    }
    return dL_ortho_dW;
}

// --- G-CONV CORE UTILITIES ---
// Note: Backward pooling/conv utilities from previous steps are logically preserved here.

Tensor c_g_convolution_forward(const Tensor& W, const Tensor& X, Tensor& Z_cache) {
    int H_in = X.shape[0]; int W_in = X.shape[1]; int C_out = W.shape[3]; int G_in = X.shape[3]; 
    int H_out = H_in - KERNEL_SIZE + 1; int W_out = W_in - KERNEL_SIZE + 1;
    int G_out = (G_in == 1) ? NUM_ROTATIONS : G_in;

    Z_cache = Tensor({H_out, W_out, C_out, G_out}); 
    Tensor Y({H_out, W_out, C_out, G_out});        
    
    // Placeholder content (Forward pass must be implemented fully for a real G-CNN)
    for (int h = 0; h < H_out; ++h) {
        for (int w = 0; w < W_out; ++w) {
            for (int c = 0; c < C_out; ++c) {
                for (int g = 0; g < G_out; ++g) {
                    double z = 0.1 * g + 0.01 * c + (double)h/100.0; 
                    Z_cache(h, w, c, g) = z;
                    Y(h, w, c, g) = max(0.0, z); // ReLU
                }
            }
        }
    }
    return Y;
}

Tensor global_average_pooling(const Tensor& X) {
    int H = X.shape[0]; int W = X.shape[1];
    int C = X.shape[2]; int G = X.shape[3];
    Tensor pooled_X({C, G});
    double area = H * W;
    for (int c = 0; c < C; ++c) {
        for (int g = 0; g < G; ++g) {
            double sum = 0.0;
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) { sum += X(h, w, c, g); }
            }
            pooled_X(c, g) = sum / area;
        }
    }
    return pooled_X;
}

Tensor invariant_pooling(const Tensor& X_pooled) {
    int C = X_pooled.shape[0];
    int G = X_pooled.shape[1];
    Tensor invariant_features({1, C}); 
    for (int c = 0; c < C; ++c) {
        double sum = 0.0;
        for (int g = 0; g < G; ++g) { sum += X_pooled(c, g); }
        invariant_features(0, c) = sum / G; 
    }
    return invariant_features;
}

// ... (backward_g_conv_output and backward_g_conv_core structures from Step 4.4.2 are used without modification)
struct GConvGrads {
    Tensor dL_dW; 
    Tensor dL_dX;
};

Tensor backward_g_conv_output(const Tensor& dL_dX_conv_out, const Tensor& Z_cache, int H_in, int W_in) {
    int H_out = dL_dX_conv_out.shape[0]; int W_out = dL_dX_conv_out.shape[1];
    int C_out = dL_dX_conv_out.shape[2]; int G_out = dL_dX_conv_out.shape[3];
    int pad = KERNEL_SIZE - 1; 

    Tensor dL_dZ({H_out, W_out, C_out, G_out});
    for (int h = 0; h < H_out; ++h) {
        for (int w = 0; w < W_out; ++w) {
            for (int c = 0; c < C_out; ++c) {
                for (int g = 0; g < G_out; ++g) {
                    dL_dZ(h, w, c, g) = (Z_cache(h, w, c, g) > 0.0) ? dL_dX_conv_out(h, w, c, g) : 0.0;
                }
            }
        }
    }

    Tensor dL_dX_conv_in_padded({H_in, W_in, C_out, G_out});
    for (int c = 0; c < C_out; ++c) {
        for (int g = 0; g < G_out; ++g) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    dL_dX_conv_in_padded(h + pad/2, w + pad/2, c, g) = dL_dZ(h, w, c, g);
                }
            }
        }
    }
    return dL_dX_conv_in_padded;
}

GConvGrads backward_g_conv_core(const Tensor& dL_dZ_padded, const Tensor& X_in, const Tensor& W) {
    int H_in = X_in.shape[0]; int W_in = X_in.shape[1];
    int C_in = X_in.shape[2]; int G_in = X_in.shape[3];
    int C_out = dL_dZ_padded.shape[2]; int G_out = dL_dZ_padded.shape[3];
    int K = KERNEL_SIZE;
    int H_out = H_in - K + 1; int W_out = W_in - K + 1;
    
    GConvGrads grads;
    grads.dL_dW = Tensor({K, K, C_in, C_out});
    grads.dL_dX = Tensor({H_in, W_in, C_in, G_in}); 

    Tensor dL_dZ({H_out, W_out, C_out, G_out});
    int pad = K - 1;
    for (int c_out = 0; c_out < C_out; ++c_out) {
        for (int g_out = 0; g_out < G_out; ++g_out) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    dL_dZ(h, w, c_out, g_out) = dL_dZ_padded(h + pad/2, w + pad/2, c_out, g_out);
                }
            }
        }
    }
    
    // 1. Calculate Weight Gradient (dL/dW)
    for (int g_out = 0; g_out < G_out; ++g_out) { 
        const Tensor& X_in_rotated = X_in; 
        for (int c_out = 0; c_out < C_out; ++c_out) {
            for (int c_in = 0; c_in < C_in; ++c_in) {
                for (int g_in = 0; g_in < G_in; ++g_in) {
                    for (int kh = 0; kh < K; ++kh) {
                        for (int kw = 0; kw < K; ++kw) {
                            for (int h = 0; h < H_out; ++h) {
                                for (int w = 0; w < W_out; ++w) {
                                    grads.dL_dW(kh, kw, c_in, c_out) += 
                                        X_in_rotated(h + kh, w + kw, c_in, g_in) * dL_dZ(h, w, c_out, g_out);
                                }
                            }
                        }
                    }
                }
            }
        }
    } 

    // 2. Calculate Input Gradient (dL/dX)
    for (int g_in = 0; g_in < G_in; ++g_in) {
        for (int c_in = 0; c_in < C_in; ++c_in) {
            for (int h = 0; h < H_in; ++h) {
                for (int w = 0; w < W_in; ++w) {
                    double grad_sum = 0.0;
                    for (int g_out = 0; g_out < G_out; ++g_out) {
                        for (int c_out = 0; c_out < C_out; ++c_out) {
                            for (int kh = 0; kh < K; ++kh) {
                                for (int kw = 0; kw < K; ++kw) {
                                    int h_out = h - kh;
                                    int w_out = w - kw;
                                    
                                    if (h_out >= 0 && h_out < H_out && w_out >= 0 && w_out < W_out) {
                                        double rotated_weight = W(kh, kw, c_in, c_out); 
                                        grad_sum += rotated_weight * dL_dZ(h_out, w_out, c_out, g_out);
                                    }
                                }
                            }
                        }
                    }
                    grads.dL_dX(h, w, c_in, g_in) = grad_sum;
                }
            }
        }
    }
    
    return grads;
}


// --- GCNN and Classifier Classes ---

struct GConvLayer {
    Tensor W; // Weights: K x K x C_IN x C_OUT
};

class GCNN {
public:
    GConvLayer L1, L2, L3;
    Tensor X0_cache, X1_cache, X2_cache; 
    Tensor Z1_cache, Z2_cache, Z3_cache; 
    Tensor X3_cache, X_pooled_cache;

    GCNN() 
        : L1({KERNEL_SIZE, KERNEL_SIZE, 1, L1_C_OUT}),
          L2({KERNEL_SIZE, KERNEL_SIZE, L1_C_OUT, L2_C_OUT}),
          L3({KERNEL_SIZE, KERNEL_SIZE, L2_C_OUT, L3_C_OUT}),
          // Initialize caches with correct size placeholders
          X0_cache({INPUT_SIZE, INPUT_SIZE, 1, 1}),
          X1_cache({30, 30, L1_C_OUT, NUM_ROTATIONS}),
          X2_cache({28, 28, L2_C_OUT, NUM_ROTATIONS}),
          Z1_cache({30, 30, L1_C_OUT, NUM_ROTATIONS}),
          Z2_cache({28, 28, L2_C_OUT, NUM_ROTATIONS}),
          Z3_cache({26, 26, L3_C_OUT, NUM_ROTATIONS}),
          X3_cache({26, 26, L3_C_OUT, NUM_ROTATIONS}),
          X_pooled_cache({L3_C_OUT, NUM_ROTATIONS})
    {
        initialize_weights(L1.W); initialize_weights(L2.W); initialize_weights(L3.W);
    }
    
    // Step 1 & 2: Forward Pass
    Tensor forward_backbone(const Tensor& X_input_2d) {
        // L0: Lift input
        X0_cache = Tensor({INPUT_SIZE, INPUT_SIZE, 1, 1});
        for(int i=0; i<INPUT_SIZE; ++i)
            for(int j=0; j<INPUT_SIZE; ++j)
                X0_cache(i, j, 0, 0) = X_input_2d(i, j);

        X1_cache = c_g_convolution_forward(L1.W, X0_cache, Z1_cache); 
        X2_cache = c_g_convolution_forward(L2.W, X1_cache, Z2_cache); 
        X3_cache = c_g_convolution_forward(L3.W, X2_cache, Z3_cache); 
        
        X_pooled_cache = global_average_pooling(X3_cache);
        return invariant_pooling(X_pooled_cache); 
    }

    // Step 4.4.2: Backward Pass for L1
    Tensor full_backward_L1(const Tensor& dL_dX_L1) {
        int H_in_L1 = INPUT_SIZE; 
        int W_in_L1 = INPUT_SIZE;
        Tensor dL_dZ_L1_padded = backward_g_conv_output(dL_dX_L1, Z1_cache, H_in_L1, W_in_L1);
        
        GConvGrads grads = backward_g_conv_core(dL_dZ_L1_padded, X0_cache, L1.W);
        L1.W = L1.W - grads.dL_dW * LEARNING_RATE;
        return grads.dL_dX; 
    }

    // Step 4.4.1: Backward Pass for L2
    Tensor full_backward_L2(const Tensor& dL_dX_L2) {
        int H_in_L2 = 30; 
        int W_in_L2 = 30;
        Tensor dL_dZ_L2_padded = backward_g_conv_output(dL_dX_L2, Z2_cache, H_in_L2, W_in_L2);
        
        GConvGrads grads_L2 = backward_g_conv_core(dL_dZ_L2_padded, X1_cache, L2.W);
        L2.W = L2.W - grads_L2.dL_dW * LEARNING_RATE;
        
        return full_backward_L1(grads_L2.dL_dX);
    }

    // Step 4.2, 4.3, 4.4: Backward Pass for L3 and initiates L2/L1 recursion
    void full_backward_L3(const Tensor& dL_dX_invariant) {
        // Step 4.2: Backward Pooling
        int H = X3_cache.shape[0]; int W = X3_cache.shape[1];
        int C = X3_cache.shape[2]; int G = X3_cache.shape[3];
        Tensor dL_dX_pooled({C, G});
        double inv_G = 1.0 / G;
        for (int c = 0; c < C; ++c) {
            double grad_c = dL_dX_invariant(0, c) * inv_G;
            for (int g = 0; g < G; ++g) { dL_dX_pooled(c, g) = grad_c; }
        }
        
        Tensor dL_dX_conv_L3({H, W, C, G});
        double inv_HW = 1.0 / (H * W);
        for (int c = 0; c < C; ++c) {
            for (int g = 0; g < G; ++g) {
                double grad_cg = dL_dX_pooled(c, g) * inv_HW; 
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) { dL_dX_conv_L3(h, w, c, g) = grad_cg; }
                }
            }
        }
                
        // Step 4.3: Backward ReLU & Padding
        int H_in_L3 = 28; int W_in_L3 = 28; 
        Tensor dL_dZ_L3_padded = backward_g_conv_output(dL_dX_conv_L3, Z3_cache, H_in_L3, W_in_L3);
        
        // Step 4.4: Backward Core
        GConvGrads grads_L3 = backward_g_conv_core(dL_dZ_L3_padded, X2_cache, L3.W);
        L3.W = L3.W - grads_L3.dL_dW * LEARNING_RATE;
        
        // Propagate to L2 and L1
        full_backward_L2(grads_L3.dL_dX);
    }
};

class LinearClassifier {
public:
    Tensor W, B; 
    LinearClassifier() : W({INVARIANT_FEATURES, NUM_CLASSES}), B({1, NUM_CLASSES}) { initialize_weights(W); initialize_weights(B); }
    
    Tensor forward(const Tensor& X) {
        Tensor logits({X.shape[0], NUM_CLASSES});
        for (int b = 0; b < X.shape[0]; ++b) {
            for (int c_out = 0; c_out < NUM_CLASSES; ++c_out) {
                double sum = 0.0;
                for (int c_in = 0; c_in < INVARIANT_FEATURES; ++c_in) {
                    sum += X(b, c_in) * W(c_in, c_out);
                }
                logits(b, c_out) = sum + B(0, c_out);
            }
        }
        return logits;
    }

    // Step 4.1: Backward Pass and Update (Includes Orthogonal Regularization)
    Tensor backward_and_update(const Tensor& X, const Tensor& dL_dLogits) {
        int batch_size = X.shape[0];
        Tensor dL_dW({INVARIANT_FEATURES, NUM_CLASSES});
        Tensor dL_dB({1, NUM_CLASSES});
        
        // dL/dW = X^T * dL/dLogits
        for (int c_in = 0; c_in < INVARIANT_FEATURES; ++c_in) {
            for (int c_out = 0; c_out < NUM_CLASSES; ++c_out) {
                double sum = 0.0;
                for (int b = 0; b < batch_size; ++b) {
                    sum += X(b, c_in) * dL_dLogits(b, c_out);
                }
                dL_dW(c_in, c_out) = sum;
                dL_dB(0, c_out) += dL_dLogits(0, c_out); 
            }
        }
        
        // Add Orthogonal Regularization Gradient
        Tensor dL_ortho_dW = orthogonal_grad(W);
        dL_dW += dL_ortho_dW;

        // Update Weights and Biases (SGD)
        W = W - dL_dW * LEARNING_RATE;
        B = B - dL_dB * LEARNING_RATE;

        // dL/dX_invariant = dL/dLogits * W^T
        Tensor dL_dX_invariant({batch_size, INVARIANT_FEATURES});
        for (int b = 0; b < batch_size; ++b) {
            for (int c_in = 0; c_in < INVARIANT_FEATURES; ++c_in) {
                for (int c_out = 0; c_out < NUM_CLASSES; ++c_out) {
                    dL_dX_invariant(b, c_in) += dL_dLogits(b, c_out) * W(c_in, c_out); 
                }
            }
        }
        return dL_dX_invariant; 
    }
};

// --- Step 4.5: FULL TRAINING LOOP AND EVALUATION ---

void evaluate(GCNN& backbone, LinearClassifier& classifier, const DataSet& data, const string& phase) {
    double total_loss = 0.0;
    int correct_predictions = 0;
    int num_samples = data.X.size();

    for (int i = 0; i < num_samples; ++i) {
        Tensor X_feature = backbone.forward_backbone(data.X[i]);
        Tensor logits = classifier.forward(X_feature);
        Tensor probs = softmax_forward(logits);
        
        vector<int> Y_true_single = {data.Y_true[i]};
        total_loss += cross_entropy_loss(probs, Y_true_single);
        
        int predicted_class = 0;
        double max_prob = -1.0;
        for (int c = 0; c < NUM_CLASSES; ++c) {
            if (probs(0, c) > max_prob) {
                max_prob = probs(0, c);
                predicted_class = c;
            }
        }
        if (predicted_class == data.Y_true[i]) {
            correct_predictions++;
        }
    }

    cout << "\n--- " << phase << " Stats ---" << endl;
    cout << "Accuracy: " << (double)correct_predictions / num_samples * 100.0 << " %" << endl;
    cout << "Average Loss: " << total_loss / num_samples << endl;
}


void train_model(GCNN& backbone, LinearClassifier& classifier, const DataSet& train_data, const DataSet& test_data) {
    int num_batches = train_data.X.size() / BATCH_SIZE;

    for (int epoch = 1; epoch <= NUM_EPOCHS; ++epoch) {
        double epoch_loss = 0.0;
        
        for (int b = 0; b < num_batches; ++b) {
            // 1. Prepare Batch
            int start_idx = b * BATCH_SIZE;
            Tensor X_batch_feature({BATCH_SIZE, INVARIANT_FEATURES});
            vector<int> Y_batch_true;
            Y_batch_true.reserve(BATCH_SIZE);
            
            // 2. Forward Pass (Batch must be processed sample-by-sample to populate GCNN caches)
            for (int i = 0; i < BATCH_SIZE; ++i) {
                Tensor X_feature_single = backbone.forward_backbone(train_data.X[start_idx + i]);
                // Copy single feature vector to the batch tensor
                for (int c = 0; c < INVARIANT_FEATURES; ++c) {
                    X_batch_feature(i, c) = X_feature_single(0, c);
                }
                Y_batch_true.push_back(train_data.Y_true[start_idx + i]);
            }
            
            // 3. Classifier Forward & Loss Calculation (Step 3)
            Tensor logits = classifier.forward(X_batch_feature);
            Tensor probs = softmax_forward(logits);
            double ce_loss = cross_entropy_loss(probs, Y_batch_true);
            double reg_loss = calculate_orthogonal_loss(classifier.W);
            double total_loss = ce_loss + reg_loss;
            epoch_loss += total_loss;

            // 4. Backward Pass (Step 4.1)
            Tensor dL_dLogits = softmax_cross_entropy_backward(probs, Y_batch_true);
            Tensor dL_dX_invariant = classifier.backward_and_update(X_batch_feature, dL_dLogits);

            // 5. Backbone Backward Pass (Steps 4.2, 4.3, 4.4, 4.4.1, 4.4.2)
            for (int i = 0; i < BATCH_SIZE; ++i) {
                Tensor dL_dX_single({1, INVARIANT_FEATURES});
                for (int c = 0; c < INVARIANT_FEATURES; ++c) {
                    dL_dX_single(0, c) = dL_dX_invariant(i, c);
                }
                
                // Rerun forward pass to load correct caches for the current sample
                backbone.forward_backbone(train_data.X[start_idx + i]);
                
                // Backpropagate through the entire GCNN
                backbone.full_backward_L3(dL_dX_single);
            }

            cout << "Epoch " << epoch << " | Batch " << b+1 << "/" << num_batches 
                 << " | Loss: " << total_loss << " (CE: " << ce_loss << " + Reg: " << reg_loss << ")" << endl;
        }

        cout << "\n=================================================" << endl;
        cout << "Epoch " << epoch << " Final Average Loss: " << epoch_loss / num_batches << endl;
        evaluate(backbone, classifier, test_data, "Test");
        cout << "=================================================" << endl;
    }
}


int main() {
    cout << fixed << setprecision(8);
    cout << "--- Starting G-CNN Training Simulation ---" << endl;

    // 1. Setup Model and Data
    GCNN backbone;
    LinearClassifier classifier;
    DataSet train_data = generate_dataset(TRAIN_SAMPLES);
    DataSet test_data = generate_dataset(TEST_SAMPLES);
    
    // 2. Execute Training (Step 4.5)
    train_model(backbone, classifier, train_data, test_data);
    
    // 3. Final Evaluation
    evaluate(backbone, classifier, train_data, "Final Train");
    evaluate(backbone, classifier, test_data, "Final Test");
    
    return 0;
}
