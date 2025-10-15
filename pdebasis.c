#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <random>
#include <iomanip>
#include <ranges>
#include <chrono>
#include <map>

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
constexpr double REG_LAMBDA = 1e-4; 
constexpr double LEARNING_RATE = 0.001;
constexpr int NUM_EPOCHS = 1000; // Increased to ensure time limit is hit
constexpr int BATCH_SIZE = 32;
constexpr int TRAIN_SAMPLES = 1000;
constexpr int TEST_SAMPLES = 200;
constexpr double TIME_LIMIT_SECONDS = 120.0; // Stop training after 2 minutes

// --- Profiling Globals ---
using Clock = chrono::high_resolution_clock;
using TimePoint = Clock::time_point;
map<string, double> profiling_times;
TimePoint start_time_global;

#define START_PROFILE(name) TimePoint start_##name = Clock::now();
#define END_PROFILE(name) \
    TimePoint end_##name = Clock::now(); \
    profiling_times[#name] += chrono::duration<double>(end_##name - start_##name).count();

// --- Tensor Class (Omitted internal details for brevity, assumed functional) ---
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
    
    // 4D Access (Required for G-Conv functions)
    double& operator()(int d1, int d2, int d3, int d4) {
        if (shape.size() != 4) throw runtime_error("Invalid 4D access (4D).");
        return data[d1 * shape[1] * shape[2] * shape[3] + d2 * shape[2] * shape[3] + d3 * shape[3] + d4];
    }
    const double& operator()(int d1, int d2, int d3, int d4) const {
        return const_cast<Tensor*>(this)->operator()(d1, d2, d3, d4);
    }

    // 2D Access
    double& operator()(int r, int c) {
        if (shape.size() != 2) throw runtime_error("Invalid 2D access (2D).");
        return data[r * shape[1] + c];
    }

    // Arithmetic operators (simplified to ensure minimal code length for repetition)
    Tensor& operator+=(const Tensor& other) { /* ... implementation ... */ return *this; }
    Tensor operator*(double scalar) const { /* ... implementation ... */ return *this; }
    Tensor operator-(const Tensor& other) const { /* ... implementation ... */ return *this; }
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
struct DataSet { vector<Tensor> X; vector<int> Y_true; };
Tensor generate_input_image(int class_label, int rotation_index) { /* ... implementation ... */ return Tensor({INPUT_SIZE, INPUT_SIZE}); }
DataSet generate_dataset(int num_samples) { /* ... implementation ... */ return DataSet{}; }

// --- FORWARD/LOSS UTILITIES (Simplified/Partial) ---
Tensor softmax_forward(const Tensor& logits) { /* ... implementation ... */ return Tensor({logits.shape[0], logits.shape[1]}); }
double cross_entropy_loss(const Tensor& probs, const vector<int>& Y_true) { /* ... implementation ... */ return 0.0; }
Tensor softmax_cross_entropy_backward(const Tensor& probs, const vector<int>& Y_true) { /* ... implementation ... */ return probs; }
double calculate_orthogonal_loss(const Tensor& W) { /* ... implementation ... */ return 0.0; }
Tensor orthogonal_grad(const Tensor& W) { /* ... implementation ... */ return W; }
Tensor c_g_convolution_forward(const Tensor& W, const Tensor& X, Tensor& Z_cache) { /* ... implementation ... */ return Tensor({X.shape[0] - KERNEL_SIZE + 1, X.shape[1] - KERNEL_SIZE + 1, W.shape[3], (X.shape[3] == 1) ? NUM_ROTATIONS : X.shape[3]}); }
Tensor global_average_pooling(const Tensor& X) { /* ... implementation ... */ return Tensor({X.shape[2], X.shape[3]}); }
Tensor invariant_pooling(const Tensor& X_pooled) { /* ... implementation ... */ return Tensor({1, X_pooled.shape[0]}); }

// --- BACKWARD PASS UTILITIES (Simplified/Partial) ---
struct GConvGrads { Tensor dL_dW; Tensor dL_dX; };
Tensor backward_g_conv_output(const Tensor& dL_dX_conv_out, const Tensor& Z_cache, int H_in, int W_in) { /* ... implementation ... */ return Tensor({H_in, W_in, dL_dX_conv_out.shape[2], dL_dX_conv_out.shape[3]}); }
GConvGrads backward_g_conv_core(const Tensor& dL_dZ_padded, const Tensor& X_in, const Tensor& W) { 
    GConvGrads grads;
    grads.dL_dW = Tensor({W.shape[0], W.shape[1], W.shape[2], W.shape[3]});
    grads.dL_dX = Tensor({X_in.shape[0], X_in.shape[1], X_in.shape[2], X_in.shape[3]});
    return grads; 
}


// --- GCNN and Classifier Classes ---

struct GConvLayer { Tensor W; };

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
          X0_cache({INPUT_SIZE, INPUT_SIZE, 1, 1}), X1_cache({30, 30, L1_C_OUT, NUM_ROTATIONS}),
          X2_cache({28, 28, L2_C_OUT, NUM_ROTATIONS}), Z1_cache({30, 30, L1_C_OUT, NUM_ROTATIONS}),
          Z2_cache({28, 28, L2_C_OUT, NUM_ROTATIONS}), Z3_cache({26, 26, L3_C_OUT, NUM_ROTATIONS}),
          X3_cache({26, 26, L3_C_OUT, NUM_ROTATIONS}), X_pooled_cache({L3_C_OUT, NUM_ROTATIONS})
    {
        initialize_weights(L1.W); initialize_weights(L2.W); initialize_weights(L3.W);
    }
    
    Tensor forward_backbone(const Tensor& X_input_2d) {
        START_PROFILE(forward_backbone);
        // L0: Lift input
        X0_cache = Tensor({INPUT_SIZE, INPUT_SIZE, 1, 1});
        // L1
        X1_cache = c_g_convolution_forward(L1.W, X0_cache, Z1_cache); 
        // L2
        X2_cache = c_g_convolution_forward(L2.W, X1_cache, Z2_cache); 
        // L3
        X3_cache = c_g_convolution_forward(L3.W, X2_cache, Z3_cache); 
        
        X_pooled_cache = global_average_pooling(X3_cache);
        Tensor result = invariant_pooling(X_pooled_cache); 
        END_PROFILE(forward_backbone);
        return result;
    }

    Tensor full_backward_L1(const Tensor& dL_dX_L1) {
        START_PROFILE(full_backward_L1);
        int H_in_L1 = INPUT_SIZE; int W_in_L1 = INPUT_SIZE; 
        Tensor dL_dZ_L1_padded = backward_g_conv_output(dL_dX_L1, Z1_cache, H_in_L1, W_in_L1);
        GConvGrads grads = backward_g_conv_core(dL_dZ_L1_padded, X0_cache, L1.W);
        L1.W = L1.W - grads.dL_dW * LEARNING_RATE;
        END_PROFILE(full_backward_L1);
        return grads.dL_dX; 
    }

    Tensor full_backward_L2(const Tensor& dL_dX_L2) {
        START_PROFILE(full_backward_L2);
        int H_in_L2 = 30; int W_in_L2 = 30;
        Tensor dL_dZ_L2_padded = backward_g_conv_output(dL_dX_L2, Z2_cache, H_in_L2, W_in_L2);
        GConvGrads grads_L2 = backward_g_conv_core(dL_dZ_L2_padded, X1_cache, L2.W);
        L2.W = L2.W - grads_L2.dL_dW * LEARNING_RATE;
        
        Tensor dL_dX_L0 = full_backward_L1(grads_L2.dL_dX);
        END_PROFILE(full_backward_L2);
        return dL_dX_L0;
    }

    void full_backward_L3(const Tensor& dL_dX_invariant) {
        START_PROFILE(full_backward_L3);
        // Step 4.2: Backward Pooling (Simplified)
        int C = X3_cache.shape[2]; int G = X3_cache.shape[3];
        Tensor dL_dX_conv_L3 = Tensor({X3_cache.shape[0], X3_cache.shape[1], C, G});
        
        // Step 4.3: Backward ReLU & Padding
        int H_in_L3 = 28; int W_in_L3 = 28; 
        Tensor dL_dZ_L3_padded = backward_g_conv_output(dL_dX_conv_L3, Z3_cache, H_in_L3, W_in_L3);
        
        // Step 4.4: Backward Core
        GConvGrads grads_L3 = backward_g_conv_core(dL_dZ_L3_padded, X2_cache, L3.W);
        L3.W = L3.W - grads_L3.dL_dW * LEARNING_RATE;
        
        // Propagate to L2 and L1
        full_backward_L2(grads_L3.dL_dX);
        END_PROFILE(full_backward_L3);
    }
};

class LinearClassifier {
public:
    Tensor W, B; 
    LinearClassifier() : W({INVARIANT_FEATURES, NUM_CLASSES}), B({1, NUM_CLASSES}) { initialize_weights(W); initialize_weights(B); }
    
    Tensor forward(const Tensor& X) {
        START_PROFILE(classifier_forward);
        Tensor logits({X.shape[0], NUM_CLASSES});
        // Simplified matrix multiplication
        // for (int b = 0; b < X.shape[0]; ++b) { ... }
        END_PROFILE(classifier_forward);
        return logits;
    }

    Tensor backward_and_update(const Tensor& X, const Tensor& dL_dLogits) {
        START_PROFILE(classifier_backward);
        // Simplified gradient calculation and update
        Tensor dL_dX_invariant = Tensor({X.shape[0], INVARIANT_FEATURES});
        // ... (dL/dW, dL/dB calculation and update)
        // Add Orthogonal Regularization Gradient
        Tensor dL_ortho_dW = orthogonal_grad(W);
        // W = W - dL_dW * LEARNING_RATE;
        END_PROFILE(classifier_backward);
        return dL_dX_invariant; 
    }
};

// --- Step 4.5: FULL TRAINING LOOP AND EVALUATION ---

void evaluate(GCNN& backbone, LinearClassifier& classifier, const DataSet& data, const string& phase) {
    // ... (evaluation logic, no profiling inside inner loop for performance)
}

void train_model(GCNN& backbone, LinearClassifier& classifier, const DataSet& train_data, const DataSet& test_data) {
    int num_batches = train_data.X.size() / BATCH_SIZE;

    start_time_global = Clock::now();
    
    for (int epoch = 1; epoch <= NUM_EPOCHS; ++epoch) {
        for (int b = 0; b < num_batches; ++b) {
            
            // Check for time limit
            double elapsed_time = chrono::duration<double>(Clock::now() - start_time_global).count();
            if (elapsed_time > TIME_LIMIT_SECONDS) {
                cout << "\nTraining stopped. Time limit of " << TIME_LIMIT_SECONDS << " seconds reached." << endl;
                return;
            }

            // 1. Prepare Batch (Omitted details)
            int start_idx = b * BATCH_SIZE;
            Tensor X_batch_feature({BATCH_SIZE, INVARIANT_FEATURES});
            vector<int> Y_batch_true;
            
            // 2. Forward Pass (Batch must be processed sample-by-sample)
            for (int i = 0; i < BATCH_SIZE; ++i) {
                Tensor X_feature_single = backbone.forward_backbone(train_data.X[start_idx + i]);
                // ... (copy feature to batch tensor)
                Y_batch_true.push_back(train_data.Y_true[start_idx + i]);
            }
            
            // 3. Classifier Forward & Loss Calculation (Step 3)
            Tensor logits = classifier.forward(X_batch_feature);
            Tensor probs = softmax_forward(logits);
            double total_loss = 0.0; // Simplified loss for profiling focus

            // 4. Backward Pass (Step 4.1)
            Tensor dL_dLogits = softmax_cross_entropy_backward(probs, Y_batch_true);
            Tensor dL_dX_invariant = classifier.backward_and_update(X_batch_feature, dL_dLogits);

            // 5. Backbone Backward Pass (Steps 4.2, 4.3, 4.4, 4.4.1, 4.4.2)
            for (int i = 0; i < BATCH_SIZE; ++i) {
                Tensor dL_dX_single({1, INVARIANT_FEATURES});
                // Rerun forward pass to load correct caches for the current sample
                backbone.forward_backbone(train_data.X[start_idx + i]); 
                // Backpropagate through the entire GCNN
                backbone.full_backward_L3(dL_dX_single);
            }

            cout << "Epoch " << epoch << " | Batch " << b+1 << "/" << num_batches 
                 << " | Loss: " << total_loss << endl;
        }
        cout << "\n=================================================" << endl;
        evaluate(backbone, classifier, test_data, "Test");
        cout << "=================================================" << endl;
    }
}

void summarize_profiling() {
    cout << "\n\n--- Profiling Summary (Accumulated Time) ---" << endl;
    double total_time = 0.0;
    for (const auto& pair : profiling_times) {
        total_time += pair.second;
    }
    
    // Sort and print results
    vector<pair<string, double>> sorted_profiles(profiling_times.begin(), profiling_times.end());
    sort(sorted_profiles.begin(), sorted_profiles.end(), [](const auto& a, const auto& b) {
        return a.second > b.second; 
    });

    for (const auto& pair : sorted_profiles) {
        double percentage = (pair.second / total_time) * 100.0;
        cout << fixed << setprecision(4)
             << left << setw(20) << pair.first << ": " 
             << right << setw(12) << pair.second << " s (" 
             << right << setw(6) << percentage << " %)" << endl;
    }
    cout << "Total Profiled Time: " << total_time << " s" << endl;

    // Summary of GCNN backpropagation
    double gcnn_backprop_time = profiling_times["full_backward_L1"] + profiling_times["full_backward_L2"] + profiling_times["full_backward_L3"];
    double gcnn_forward_time = profiling_times["forward_backbone"];
    
    cout << "\n--- Functional Summary ---" << endl;
    cout << fixed << setprecision(4);
    cout << "Total GCNN Forward Time (per call): " << gcnn_forward_time << " s" << endl;
    cout << "Total GCNN Backward Time (per chain): " << gcnn_backprop_time << " s" << endl;

    // The GCNN backward pass (L3 -> L2 -> L1) is expected to dominate the overall runtime, 
    // especially the `full_backward_L3` call which initiates the recursive chain.
    double expected_bottleneck = (gcnn_backprop_time / total_time) * 100.0;
    cout << "GCNN Backprop constitutes " << expected_bottleneck << " % of total time." << endl;
}


int main() {
    cout << fixed << setprecision(8);
    cout << "--- Starting G-CNN Training Simulation with 2-minute Time Limit ---" << endl;

    // 1. Setup Model and Data
    GCNN backbone;
    LinearClassifier classifier;
    // Note: The actual data generation is time-consuming, simplifying here.
    DataSet train_data = generate_dataset(TRAIN_SAMPLES);
    DataSet test_data = generate_dataset(TEST_SAMPLES);
    
    // 2. Execute Training (Step 4.5)
    train_model(backbone, classifier, train_data, test_data);
    
    // 3. Final Evaluation (Omitted final full evaluation for brevity)

    // 4. Summarize Profiling
    summarize_profiling();
    
    return 0;
}
