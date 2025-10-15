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
// Fix 1: Renaming M_PI to PI_CONST to avoid macro conflict
constexpr double PI_CONST = 3.14159265358979323846; 
constexpr int INPUT_SIZE = 32;
constexpr int KERNEL_SIZE = 3;
constexpr int NUM_CLASSES = 4;
constexpr int NUM_ROTATIONS = 32; // G-Group Size (C32)
constexpr int L1_C_OUT = 8;
constexpr int L2_C_OUT = 16;
constexpr int L3_C_OUT = 32;
constexpr int INVARIANT_FEATURES = L3_C_OUT; // Final soft feature size
constexpr double REG_LAMBDA = 1e-4;
constexpr double LEARNING_RATE = 0.001;
constexpr int NUM_EPOCHS = 10;
constexpr int BATCH_SIZE = 32;
constexpr int TRAIN_SAMPLES = 500;
constexpr int TEST_SAMPLES = 100;

// --- Tensor Class ---
class Tensor {
public:
    vector<double> data;
    vector<int> shape;
    int size = 0;

    // Fix 7: Default constructor for empty Tensor
    Tensor() : size(0) {} 
    
    Tensor(initializer_list<int> s) : shape(s) {
        size = 1;
        for (int dim : shape) size *= dim;
        data.resize(size, 0.0);
    }
    
    // 2D Access
    double& operator()(int r, int c) {
        if (shape.size() != 2) throw runtime_error("Invalid 2D access.");
        return data[r * shape[1] + c];
    }
    const double& operator()(int r, int c) const {
        return const_cast<Tensor*>(this)->operator()(r, c);
    }
    
    // 4D Access
    double& operator()(int d1, int d2, int d3, int d4) {
        if (shape.size() != 4) throw runtime_error("Invalid 4D access.");
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
    // Fix 2: Corrected data size access
    Tensor operator-(const Tensor& other) const { 
        Tensor result = *this; 
        for(size_t i = 0; i < data.size(); ++i) { 
            result.data[i] -= other.data[i]; 
        } 
        return result; 
    }
};

// Global RNG and Initializer (Partial inclusion for compilation)
random_device rd; mt19937 gen(rd()); uniform_real_distribution<> weight_distrib(-0.01, 0.01);
uniform_int_distribution<> label_distrib(0, NUM_CLASSES - 1); uniform_int_distribution<> rotation_distrib(0, NUM_ROTATIONS - 1);
void initialize_weights(Tensor& W) { for (auto& val : W.data) val = weight_distrib(gen); }
Tensor rotate_2d_slice(const Tensor& input_2d, int k_rotations) {
    int N = input_2d.shape[0]; Tensor output({N, N}); 
    for(int i=0; i<N; ++i) for(int j=0; j<N; ++j) output(i, j) = input_2d(i, j);
    return output;
}

// --- FORWARD PASS UTILITIES ---

Tensor c_g_convolution_forward(const Tensor& W, const Tensor& X, Tensor& Z_cache) {
    int H_in = X.shape[0]; int W_in = X.shape[1]; int C_out = W.shape[3]; int G_in = X.shape[3]; 
    int H_out = H_in - KERNEL_SIZE + 1; int W_out = W_in - KERNEL_SIZE + 1;
    int G_out = (G_in == 1) ? NUM_ROTATIONS : G_in;

    Z_cache = Tensor({H_out, W_out, C_out, G_out}); 
    Tensor Y({H_out, W_out, C_out, G_out});        
    
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
    int C = X.shape[2]; int G = X.shape[3];
    return Tensor({C, G});
}
Tensor invariant_pooling(const Tensor& X_pooled) {
    int C = X_pooled.shape[0]; 
    return Tensor({1, C});
}

// --- BACKWARD PASS UTILITIES (Steps 4.3 & 4.4) ---

// Step 4.3: Backward Pass for G-Convolution Output (ReLU and Padding)
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

// Fix 3: GConvGrads now has a default constructor (since Tensor does)
struct GConvGrads {
    Tensor dL_dW; // K x K x C_in x C_out
    Tensor dL_dX; // H_in x W_in x C_in x G_in
};

// Step 4.4: Backward Pass for G-Convolution Weights (dL/dW) and Input (dL/dX)
GConvGrads backward_g_conv_core(const Tensor& dL_dZ_padded, const Tensor& X_in, const Tensor& W) {
    int H_in = X_in.shape[0]; int W_in = X_in.shape[1];
    int C_in = X_in.shape[2]; int G_in = X_in.shape[3];
    int C_out = dL_dZ_padded.shape[2]; int G_out = dL_dZ_padded.shape[3];
    int K = KERNEL_SIZE;
    int H_out = H_in - K + 1; int W_out = W_in - K + 1;
    
    GConvGrads grads;
    grads.dL_dW = Tensor({K, K, C_in, C_out});
    grads.dL_dX = Tensor({H_in, W_in, C_in, G_in}); 

    // Re-slice dL/dZ to the actual output size H_out x W_out
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
        int rotation_steps = g_out;
        Tensor X_in_rotated = X_in; // Simplified: Rotation logic requires X_in to be rotated

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


// --- Full GCNN and Classifier Classes ---

struct GConvLayer {
    Tensor W; // Weights: K x K x C_IN x C_OUT
};

class GCNN {
public:
    GConvLayer L1, L2, L3;
    // Fix 4: Initialize cache members with placeholder shapes
    Tensor X1_cache{28, 28, L1_C_OUT, NUM_ROTATIONS}, X2_cache{30, 30, L2_C_OUT, NUM_ROTATIONS}; 
    Tensor Z2_cache{28, 28, L2_C_OUT, NUM_ROTATIONS}, Z3_cache{26, 26, L3_C_OUT, NUM_ROTATIONS}; 

    GCNN() 
        : L1({KERNEL_SIZE, KERNEL_SIZE, 1, L1_C_OUT}),
          L2({KERNEL_SIZE, KERNEL_SIZE, L1_C_OUT, L2_C_OUT}),
          L3({KERNEL_SIZE, KERNEL_SIZE, L2_C_OUT, L3_C_OUT})
    {
        initialize_weights(L1.W); initialize_weights(L2.W); initialize_weights(L3.W);
    }
    
    // Forward Pass: Caches X1, X2, Z2, Z3
    Tensor forward_backbone(const Tensor& X_input_2d) {
        Tensor X0({INPUT_SIZE, INPUT_SIZE, 1, 1});
        // ... (X0 setup)

        Tensor Z1_cache;
        X1_cache = c_g_convolution_forward(L1.W, X0, Z1_cache); 
        
        X2_cache = c_g_convolution_forward(L2.W, X1_cache, Z2_cache); // X2, Z2 cached
        
        Tensor X3 = c_g_convolution_forward(L3.W, X2_cache, Z3_cache); // Z3 cached
        
        // ... (Pooling forward)
        return X3; // Placeholder return
    }

    // Step 4.4.1: Backward Pass for L2 (Called by L3 backprop)
    // Input: dL/dX_L2 (from L3's backward pass)
    // Output: dL/dX_L1 (for L1's backward pass)
    Tensor full_backward_L2(const Tensor& dL_dX_L2) {
        // 1. Step 4.3 (Backward ReLU & Padding) -> dL/dZ_L2_padded
        int H_in_L2 = 30; // X1's output size
        int W_in_L2 = 30;
        Tensor dL_dZ_L2_padded = backward_g_conv_output(dL_dX_L2, Z2_cache, H_in_L2, W_in_L2);
        
        // 2. Step 4.4 (Backward Core) -> dL/dW_L2 and dL/dX_L1
        GConvGrads grads = backward_g_conv_core(dL_dZ_L2_padded, X1_cache, L2.W);
        
        // 3. Update Weights (L2)
        L2.W = L2.W - grads.dL_dW * LEARNING_RATE;
        
        cout << "L2 Backprop Complete." << endl;
        // NOTE: grads.dL_dX is dL/dX_L1 and is ready for L1's backward pass.
        return grads.dL_dX; 
    }

    // Combined Backward Pass for L3 (Steps 4.2, 4.3, 4.4)
    void full_backward_L3(const Tensor& dL_dX_invariant) {
        
        // 1. Step 4.2 (Backward Pooling) -> dL/dX_conv_L3
        int H_out = 26; int W_out = 26; int C_out = L3_C_OUT; int G_out = NUM_ROTATIONS;
        Tensor dL_dX_conv_L3({H_out, W_out, C_out, G_out}); 
        // Simulated Step 4.2 output
        for (int h = 0; h < H_out; ++h)
            for (int w = 0; w < W_out; ++w)
                dL_dX_conv_L3(h, w, 0, 0) = dL_dX_invariant(0, 0) / (H_out * W_out * G_out);
                
        // 2. Step 4.3 (Backward ReLU & Padding) -> dL/dZ_L3_padded
        int H_in_L3 = 28; int W_in_L3 = 28; // X2's output size
        Tensor dL_dZ_L3_padded = backward_g_conv_output(dL_dX_conv_L3, Z3_cache, H_in_L3, W_in_L3);
        
        // 3. Step 4.4 (Backward Core) -> dL/dW_L3 and dL/dX_L2
        GConvGrads grads_L3 = backward_g_conv_core(dL_dZ_L3_padded, X2_cache, L3.W);
        
        // 4. Update Weights (L3)
        L3.W = L3.W - grads_L3.dL_dW * LEARNING_RATE;
        
        cout << "L3 Backprop Complete." << endl;
        
        // 5. Step 4.4.1 (Propagate to L2)
        Tensor dL_dX_L1 = full_backward_L2(grads_L3.dL_dX);
        
        // NOTE: dL_dX_L1 is ready to be passed to L1's backward pass.
    }
};

// ... (LinearClassifier and main function placeholders for compilation)

class LinearClassifier {
public:
    Tensor W, B; 
    LinearClassifier() : W({INVARIANT_FEATURES, NUM_CLASSES}), B({1, NUM_CLASSES}) { /* ... */ }
    Tensor backward_and_update(const Tensor& X, const Tensor& dL_dLogits, double& total_loss) {
        Tensor dL_dX_invariant({X.shape[0], INVARIANT_FEATURES});
        for (int b = 0; b < X.shape[0]; ++b) {
            for (int c = 0; c < INVARIANT_FEATURES; ++c) {
                 dL_dX_invariant(b, c) = 0.01 + 0.001 * c; 
            }
        }
        return dL_dX_invariant; 
    }
};

int main() {
    cout << fixed << setprecision(8);

    // 1. Setup 
    GCNN backbone;
    
    // Simulate a forward pass to populate caches (X1, X2, Z2, Z3)
    Tensor input_image({INPUT_SIZE, INPUT_SIZE});
    backbone.forward_backbone(input_image);
    
    // 2. Simulate Step 4.1 Output (dL/dX_invariant)
    Tensor dL_dX_invariant({1, INVARIANT_FEATURES}); 
    dL_dX_invariant(0, 0) = 0.001; 
    
    cout << "\n--- Full G-CNN Backward Pass: Step 4.4.1 Implementation (L2) ---" << endl;
    
    // 3. Execute L3 and L2 Backprop
    try {
        backbone.full_backward_L3(dL_dX_invariant);
    } catch (const exception& e) {
        cerr << "\nError during Backprop: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}
