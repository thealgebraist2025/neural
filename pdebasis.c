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
constexpr double M_PI = 3.14159265358979323846;

// --- Tensor Class ---
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
    
    // 2D/4D Access (Omitted for brevity, but defined as in previous steps)
    // ...
    double& operator()(int r, int c) { return data[r * shape[1] + c]; }
    const double& operator()(int r, int c) const { return const_cast<Tensor*>(this)->operator()(r, c); }
    double& operator()(int d1, int d2, int d3, int d4) {
        return data[d1 * shape[1] * shape[2] * shape[3] + d2 * shape[2] * shape[3] + d3 * shape[3] + d4];
    }
    const double& operator()(int d1, int d2, int d3, int d4) const {
        return const_cast<Tensor*>(this)->operator()(d1, d2, d3, d4);
    }
    
    // Arithmetic operators (Omitted for brevity, but defined as in previous steps)
    // ...
    Tensor& operator+=(const Tensor& other) { for(size_t i = 0; i < data.size(); ++i) data[i] += other.data[i]; return *this; }
    Tensor operator*(double scalar) const { Tensor result = *this; for(double& val : result.data) val *= scalar; return result; }
    Tensor operator-(const Tensor& other) const { Tensor result = *this; for(size_t i = 0; i < data.data.size(); ++i) result.data[i] -= other.data[i]; return result; }
};

// Global RNG and Initializer (Omitted for brevity, but required)
// ...
random_device rd; mt19937 gen(rd()); uniform_real_distribution<> weight_distrib(-0.01, 0.01);
uniform_int_distribution<> label_distrib(0, NUM_CLASSES - 1); uniform_int_distribution<> rotation_distrib(0, NUM_ROTATIONS - 1);
void initialize_weights(Tensor& W) { for (auto& val : W.data) val = weight_distrib(gen); }

// --- FORWARD PASS UTILITIES (G-Conv structure) ---

Tensor rotate_2d_slice(const Tensor& input_2d, int k_rotations) {
    int N = input_2d.shape[0]; Tensor output({N, N}); 
    // Simplified rotation logic: return identity for backward pass focus
    for(int i=0; i<N; ++i) for(int j=0; j<N; ++j) output(i, j) = input_2d(i, j);
    return output;
}

// G-Convolution Forward Pass (Simplified, caches pre-activation value Z and input X)
Tensor c_g_convolution_forward(const Tensor& W, const Tensor& X, Tensor& Z_cache) {
    int H_in = X.shape[0]; int W_in = X.shape[1]; int C_out = W.shape[3]; int G_in = X.shape[3]; 
    int H_out = H_in - KERNEL_SIZE + 1; int W_out = W_in - KERNEL_SIZE + 1;
    int G_out = (G_in == 1) ? NUM_ROTATIONS : G_in;

    Z_cache = Tensor({H_out, W_out, C_out, G_out}); // Pre-activation
    Tensor Y({H_out, W_out, C_out, G_out});        // Activated output
    
    // Placeholder content:
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

// Global Average Pooling/Invariant Pooling (Omitted for brevity, but required)
// ...

// --- BACKWARD PASS UTILITIES (STEP 4.3 & 4.4 Implementation) ---

// Step 4.3: Backward Pass for G-Convolution Output (ReLU and Padding)
Tensor backward_g_conv_output(const Tensor& dL_dX_conv_out, const Tensor& Z_cache, int H_in, int W_in) {
    int H_out = dL_dX_conv_out.shape[0]; int W_out = dL_dX_conv_out.shape[1];
    int C_out = dL_dX_conv_out.shape[2]; int G_out = dL_dX_conv_out.shape[3];
    int pad = KERNEL_SIZE - 1; 

    // 1. Backward ReLU (dL/dX_conv_out -> dL/dZ)
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

    // 2. Backward Spatial Crop/Un-padding (dL/dZ -> dL/dX_conv_in_padded)
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

// Step 4.4: Backward Pass for G-Convolution Weights (dL/dW) and Input (dL/dX)
// dL_dZ_padded: H_in x W_in x C_out x G_out (dL/dZ, after padding)
// X_in: H_in x W_in x C_in x G_in (Input activation, X_L2 for L3)
// W: K x K x C_in x C_out (The weights of L3)
struct GConvGrads {
    Tensor dL_dW; // K x K x C_in x C_out
    Tensor dL_dX; // H_in x W_in x C_in x G_in
};

GConvGrads backward_g_conv_core(const Tensor& dL_dZ_padded, const Tensor& X_in, const Tensor& W) {
    int H_in = X_in.shape[0]; int W_in = X_in.shape[1];
    int C_in = X_in.shape[2]; int G_in = X_in.shape[3];
    int C_out = dL_dZ_padded.shape[2]; int G_out = dL_dZ_padded.shape[3];
    int K = KERNEL_SIZE;
    int H_out = H_in - K + 1; // Actual convolution output size
    int W_out = W_in - K + 1;
    
    GConvGrads grads;
    grads.dL_dW = Tensor({K, K, C_in, C_out});
    grads.dL_dX = Tensor({H_in, W_in, C_in, G_in}); // This will be the gradient passed to L2

    // The gradient dL/dZ is effectively un-padded (only the conv output area is non-zero)
    // We re-slice dL/dZ to the actual output size H_out x W_out
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
    // dL/dW_k = Sum_{h,w,c_out,g_out} ( X_rotated * dL/dZ )
    // Since W is group-invariant, we must sum over the group dimension G_out (which is $G$)
    
    for (int g_out = 0; g_out < G_out; ++g_out) { // Sum over all output rotations
        int rotation_steps = g_out; // Rotation applied in forward pass

        // Get the spatially rotated X_in (simulating the forward rotation)
        Tensor X_in_rotated({H_in, W_in, C_in, G_in});
        for(int c_in=0; c_in < C_in; ++c_in) {
            for(int g_in=0; g_in < G_in; ++g_in) {
                Tensor X_slice({H_in, W_in});
                for(int h=0; h<H_in; ++h) for(int w=0; w<W_in; ++w) X_slice(h,w) = X_in(h,w,c_in,g_in);
                Tensor X_rot_slice = rotate_2d_slice(X_slice, rotation_steps); 
                for(int h=0; h<H_in; ++h) for(int w=0; w<W_in; ++w) X_in_rotated(h,w,c_in,g_in) = X_rot_slice(h,w);
            }
        }
        
        // Standard convolution weight gradient calculation (summed over C_in and G_in)
        for (int c_out = 0; c_out < C_out; ++c_out) {
            for (int c_in = 0; c_in < C_in; ++c_in) {
                for (int g_in = 0; g_in < G_in; ++g_in) {
                    for (int kh = 0; kh < K; ++kh) {
                        for (int kw = 0; kw < K; ++kw) {
                            for (int h = 0; h < H_out; ++h) {
                                for (int w = 0; w < W_out; ++w) {
                                    // Sum over spatial location (h, w), G_in, and G_out
                                    grads.dL_dW(kh, kw, c_in, c_out) += 
                                        X_in_rotated(h + kh, w + kw, c_in, g_in) * dL_dZ(h, w, c_out, g_out);
                                }
                            }
                        }
                    }
                }
            }
        }
    } // End Sum over G_out

    // 2. Calculate Input Gradient (dL/dX)
    // dL/dX = Transposed Convolution of dL/dZ with Rotated/Flipped Weights
    
    for (int g_in = 0; g_in < G_in; ++g_in) { // For each input group element
        for (int c_in = 0; c_in < C_in; ++c_in) {
            for (int h = 0; h < H_in; ++h) {
                for (int w = 0; w < W_in; ++w) {
                    
                    double grad_sum = 0.0;
                    
                    for (int g_out = 0; g_out < G_out; ++g_out) { // Sum over all output rotations
                        int rotation_steps = g_out;
                        
                        // NOTE: True G-Conv backward pass involves permuting the input gradient 
                        // and rotating the transposed kernel. We simplify using the spatial rotation.
                        
                        for (int c_out = 0; c_out < C_out; ++c_out) {
                            for (int kh = 0; kh < K; ++kh) {
                                for (int kw = 0; kw < K; ++kw) {
                                    // Check boundary for dL_dZ
                                    int h_out = h - kh;
                                    int w_out = w - kw;
                                    
                                    if (h_out >= 0 && h_out < H_out && w_out >= 0 && w_out < W_out) {
                                        // The weight must be spatially rotated (inverse rotation) 
                                        // before correlation with the gradient dL/dZ.
                                        // The rotated weight is W_rot(kh, kw, c_in, c_out, rotation_steps)
                                        
                                        // Simplified: Assume W_rotated is simply W for $C_k$
                                        double rotated_weight = W(kh, kw, c_in, c_out); 
                                        
                                        grad_sum += rotated_weight * dL_dZ(h_out, w_out, c_out, g_out);
                                    }
                                }
                            }
                        }
                    } // End Sum over G_out
                    
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
    Tensor X2_cache; // L2 output (L3 input) for backward pass
    Tensor Z3_cache; // L3 pre-activation for ReLU backprop
    // ... (other caches)

    GCNN() 
        : L1({KERNEL_SIZE, KERNEL_SIZE, 1, L1_C_OUT}),
          L2({KERNEL_SIZE, KERNEL_SIZE, L1_C_OUT, L2_C_OUT}),
          L3({KERNEL_SIZE, KERNEL_SIZE, L2_C_OUT, L3_C_OUT})
    {
        initialize_weights(L1.W); initialize_weights(L2.W); initialize_weights(L3.W);
    }
    
    // Forward Pass: Caches X2 and Z3
    Tensor forward_backbone(const Tensor& X_input_2d) {
        // ... (X0 setup)
        Tensor X0({INPUT_SIZE, INPUT_SIZE, 1, 1});
        // ... (X0 setup)

        Tensor Z1_cache;
        Tensor X1 = c_g_convolution_forward(L1.W, X0, Z1_cache);
        
        Tensor Z2_cache;
        X2_cache = c_g_convolution_forward(L2.W, X1, Z2_cache); // X2 cached
        
        Tensor X3 = c_g_convolution_forward(L3.W, X2_cache, Z3_cache); // Z3 cached
        
        Tensor X_pooled_cache;
        // ... (Pooling forward)
        return X3; // Placeholder return
    }

    // Combined Backward Pass for L3 (Steps 4.2, 4.3, 4.4)
    // dL_dX_invariant is the gradient from the classifier (single sample)
    void full_backward_L3(const Tensor& dL_dX_invariant) {
        
        // 1. Step 4.2 (Backward Pooling) -> dL/dX_conv_L3
        int H_out = 26; int W_out = 26; int C_out = L3_C_OUT; int G_out = NUM_ROTATIONS;
        Tensor dL_dX_conv_L3({H_out, W_out, C_out, G_out}); 
        // Simulated Step 4.2 output
        for (int h = 0; h < H_out; ++h)
            for (int w = 0; w < W_out; ++w)
                dL_dX_conv_L3(h, w, 0, 0) = dL_dX_invariant(0, 0) / (H_out * W_out * G_out);
                
        // 2. Step 4.3 (Backward ReLU & Padding) -> dL/dX_conv_L2_padded
        int H_in_L3 = 28; int W_in_L3 = 28;
        Tensor dL_dZ_padded = backward_g_conv_output(dL_dX_conv_L3, Z3_cache, H_in_L3, W_in_L3);
        
        // 3. Step 4.4 (Backward Core) -> dL/dW_L3 and dL/dX_L2
        GConvGrads grads = backward_g_conv_core(dL_dZ_padded, X2_cache, L3.W);
        
        // 4. Update Weights (L3)
        L3.W = L3.W - grads.dL_dW * LEARNING_RATE;
        
        // NOTE: grads.dL_dX is dL/dX_L2 and is ready to be passed to L2's backward pass.
        cout << "\nL3 Backprop Complete." << endl;
        cout << "dL/dW_L3 L1 Norm: " << accumulate(grads.dL_dW.data.begin(), grads.dL_dW.data.end(), 0.0, [](double sum, double val){ return sum + abs(val); }) << endl;
        cout << "dL/dX_L2 L1 Norm (Next Step Input): " << accumulate(grads.dL_dX.data.begin(), grads.dL_dX.data.end(), 0.0, [](double sum, double val){ return sum + abs(val); }) << endl;
    }
};

// ... (Loss functions, Regularization functions, LinearClassifier, and Data generation omitted for brevity) ...

class LinearClassifier {
public:
    Tensor W, B; 
    LinearClassifier() : W({INVARIANT_FEATURES, NUM_CLASSES}), B({1, NUM_CLASSES}) { /* ... */ }
    Tensor backward_and_update(const Tensor& X, const Tensor& dL_dLogits, double& total_loss) {
        Tensor dL_dX_invariant({X.shape[0], INVARIANT_FEATURES});
        // Simulated dL/dX_invariant
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
    
    // Simulate a forward pass to populate caches (X2_cache, Z3_cache)
    Tensor input_image({INPUT_SIZE, INPUT_SIZE});
    Tensor X_L3_output = backbone.forward_backbone(input_image);
    
    // Check cache sizes (Must be correct for Step 4.4 to work)
    cout << "X_L2 (L3 Input) Cache Shape: " << backbone.X2_cache.shape[0] << "x" << backbone.X2_cache.shape[1] 
         << "x" << backbone.X2_cache.shape[2] << "x" << backbone.X2_cache.shape[3] << endl;
    
    // 2. Simulate Step 4.1 Output (dL/dX_invariant)
    Tensor dL_dX_invariant({1, INVARIANT_FEATURES}); 
    dL_dX_invariant(0, 0) = 0.001; // Tiny non-zero gradient
    dL_dX_invariant(0, 1) = 0.002;

    cout << "\n--- Full G-CNN Backward Pass: Step 4.4 Implementation (L3) ---" << endl;
    cout << "dL/dX_invariant Sample (Input to L3 Backward Chain): " << dL_dX_invariant(0, 0) << endl;
    
    // 3. Step 4.4 Execution (wrapped in full_backward_L3)
    try {
        backbone.full_backward_L3(dL_dX_invariant);
    } catch (const exception& e) {
        cerr << "\nError during Step 4.4: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}
