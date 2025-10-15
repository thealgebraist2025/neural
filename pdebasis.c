#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <random>
#include <iomanip>
#include <chrono>
#include <map>

using namespace std;

// --- Configuration (OPTIMIZED) ---
constexpr double PI_CONST = 3.14159265358979323846; 
constexpr int INPUT_SIZE = 32;
constexpr int KERNEL_SIZE = 3;
constexpr int NUM_CLASSES = 4;

// FIX 1: Reduced Rotations for Speed
constexpr int NUM_ROTATIONS = 16; 

// FIX 2: Reduced Network Width for Speed
constexpr int L1_C_OUT = 4;
constexpr int L2_C_OUT = 8;
constexpr int L3_C_OUT = 16; 

constexpr int INVARIANT_FEATURES = L3_C_OUT; 
constexpr double REG_LAMBDA = 1e-4; 

// FIX 3: Increased Learning Rate for Faster Learning
constexpr double LEARNING_RATE = 0.01;

constexpr int NUM_EPOCHS = 1000;
constexpr int BATCH_SIZE = 32;
constexpr int TRAIN_SAMPLES = 1000;
constexpr int TEST_SAMPLES = 200;
constexpr double TIME_LIMIT_SECONDS = 120.0; 

// --- Profiling Globals ---
using Clock = chrono::high_resolution_clock;
using TimePoint = Clock::time_point;
map<string, double> profiling_times;
TimePoint start_time_global;

#define START_PROFILE(name) TimePoint start_##name = Clock::now();
#define END_PROFILE(name) \
    TimePoint end_##name = Clock::now(); \
    profiling_times[#name] += chrono::duration<double>(end_##name - start_##name).count();

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
    
    // 4D Access
    double& operator()(int d1, int d2, int d3, int d4) {
        if (shape.size() != 4) throw runtime_error("Invalid 4D access (4D).");
        return data[d1 * shape[1] * shape[2] * shape[3] + d2 * shape[2] * shape[3] + d3 * shape[3] + d4];
    }
    const double& operator()(int d1, int d2, int d3, int d4) const {
        if (shape.size() != 4) throw runtime_error("Invalid 4D access (4D).");
        return data[d1 * shape[1] * shape[2] * shape[3] + d2 * shape[2] * shape[3] + d3 * shape[3] + d4];
    }

    // 2D Access
    double& operator()(int r, int c) {
        if (shape.size() != 2) throw runtime_error("Invalid 2D access (2D).");
        return data[r * shape[1] + c];
    }
    const double& operator()(int r, int c) const {
        if (shape.size() != 2) throw runtime_error("Invalid 2D access (2D).");
        return data[r * shape[1] + c];
    }

    // Arithmetic operators
    Tensor& operator+=(const Tensor& other) { 
        if (size != other.size) throw runtime_error("Tensor size mismatch in +=.");
        for(size_t i = 0; i < data.size(); ++i) { data[i] += other.data[i]; }
        return *this; 
    }
    Tensor operator*(double scalar) const { 
        Tensor result = *this; 
        for(double& val : result.data) { val *= scalar; }
        return result; 
    }
    Tensor operator-(const Tensor& other) const { 
        Tensor result = *this; 
        for(size_t i = 0; i < data.size(); ++i) { result.data[i] -= other.data[i]; }
        return result; 
    }
};

// Global RNG and Initializer
random_device rd; mt19937 gen(rd()); uniform_real_distribution<> weight_distrib(-0.01, 0.01);
uniform_int_distribution<> label_distrib(0, NUM_CLASSES - 1); uniform_int_distribution<> rotation_distrib(0, NUM_ROTATIONS - 1);
uniform_real_distribution<> coord_distrib(0, INPUT_SIZE);
uniform_real_distribution<> noise_distrib(0.0, 0.1);

void initialize_weights(Tensor& W) { for (auto& val : W.data) val = weight_distrib(gen); }

// --- DATASET & UTILITIES (Random Shapes) ---

Tensor rotate_2d_slice(const Tensor& input_2d, int k_rotations) {
    int N = input_2d.shape[0]; Tensor output({N, N}); 
    for(int i=0; i<N; ++i) for(int j=0; j<N; ++j) output(i, j) = input_2d(i, j);
    return output;
}

void draw_line(Tensor& image, int x0, int y0, int x1, int y1) {
    if (abs(x1 - x0) > abs(y1 - y0)) { 
        if (x0 > x1) swap(x0, x1);
        for (int x = x0; x <= x1; ++x) 
            if (x >= 0 && x < INPUT_SIZE && y0 >= 0 && y0 < INPUT_SIZE) 
                image(y0, x) = 1.0;
    } else { 
        if (y0 > y1) swap(y0, y1);
        for (int y = y0; y <= y1; ++y)
            if (y >= 0 && y < INPUT_SIZE && x0 >= 0 && x0 < INPUT_SIZE) 
                image(y, x0) = 1.0;
    }
}

Tensor generate_filled_circle() {
    Tensor image({INPUT_SIZE, INPUT_SIZE});
    int center_x = coord_distrib(gen);
    int center_y = coord_distrib(gen);
    int radius = 5 + (rand() % 5);
    
    for (int i = 0; i < INPUT_SIZE; ++i) {
        for (int j = 0; j < INPUT_SIZE; ++j) {
            double dist_sq = pow(i - center_y, 2) + pow(j - center_x, 2);
            if (dist_sq <= pow(radius, 2)) {
                image(i, j) = 1.0;
            }
        }
    }
    return image;
}

Tensor generate_random_line() {
    Tensor image({INPUT_SIZE, INPUT_SIZE});
    int x0 = coord_distrib(gen); int y0 = coord_distrib(gen);
    int x1 = coord_distrib(gen); int y1 = coord_distrib(gen);
    draw_line(image, x0, y0, x1, y1);
    return image;
}

Tensor generate_random_noise() {
    Tensor image({INPUT_SIZE, INPUT_SIZE});
    for (int i = 0; i < INPUT_SIZE; ++i) {
        for (int j = 0; j < INPUT_SIZE; ++j) {
            image(i, j) = noise_distrib(gen) * 10.0; 
        }
    }
    return image;
}


Tensor generate_input_image(int class_label, int rotation_index) {
    Tensor image({INPUT_SIZE, INPUT_SIZE});
    
    // Use class_label to determine the underlying shape
    if (class_label % NUM_CLASSES == 0) {
        image = generate_filled_circle();
    } else if (class_label % NUM_CLASSES == 1) {
        image = generate_random_line();
    } else if (class_label % NUM_CLASSES == 2) {
        image = generate_random_noise();
    } else { 
        image = generate_filled_circle(); 
    }

    // Apply rotation
    if (rotation_index > 0) {
        image = rotate_2d_slice(image, rotation_index);
    }
    return image;
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

// --- FORWARD/LOSS UTILITIES ---

Tensor softmax_forward(const Tensor& logits) {
    Tensor probs({logits.shape[0], logits.shape[1]});
    for(int b = 0; b < logits.shape[0]; ++b) {
        double max_val = logits(b, 0);
        for(int c = 1; c < logits.shape[1]; ++c) max_val = max(max_val, logits(b, c));
        
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

// Simplified placeholders for G-CNN math
double calculate_orthogonal_loss(const Tensor& W) { return 0.0001; } 
Tensor orthogonal_grad(const Tensor& W) { return W * 1e-6; } 
Tensor c_g_convolution_forward(const Tensor& W, const Tensor& X, Tensor& Z_cache) { 
    int H_in = X.shape[0]; int W_in = X.shape[1]; int C_out = W.shape[3]; int G_in = X.shape[3]; 
    int H_out = H_in - KERNEL_SIZE + 1; int W_out = W_in - KERNEL_SIZE + 1;
    // G_out is NUM_ROTATIONS for L1, and G_in for L2/L3 (G-Conv)
    int G_out = (G_in == 1) ? NUM_ROTATIONS : G_in; 
    Z_cache = Tensor({H_out, W_out, C_out, G_out}); 
    Tensor Y({H_out, W_out, C_out, G_out});
    
    // Placeholder content for computation time simulation
    for (int h = 0; h < H_out; ++h) {
        for (int w = 0; w < W_out; ++w) {
            for (int c = 0; c < C_out; ++c) {
                for (int g = 0; g < G_out; ++g) {
                    double z = (double)h/100.0 + W(0,0,0,c) + X(h, w, 0, g); 
                    Z_cache(h, w, c, g) = z;
                    Y(h, w, c, g) = max(0.0, z); 
                }
            }
        }
    }
    return Y; 
}
Tensor global_average_pooling(const Tensor& X) { return Tensor({X.shape[2], X.shape[3]}); }
Tensor invariant_pooling(const Tensor& X_pooled) { return Tensor({1, X_pooled.shape[0]}); }
struct GConvGrads { Tensor dL_dW; Tensor dL_dX; };
Tensor backward_g_conv_output(const Tensor& dL_dX_conv_out, const Tensor& Z_cache, int H_in, int W_in) { return Tensor({H_in, W_in, dL_dX_conv_out.shape[2], dL_dX_conv_out.shape[3]}); }
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
          // Adjusted cache sizes to match new channel counts
          X0_cache({INPUT_SIZE, INPUT_SIZE, 1, 1}), X1_cache({30, 30, L1_C_OUT, NUM_ROTATIONS}),
          X2_cache({28, 28, L2_C_OUT, NUM_ROTATIONS}), Z1_cache({30, 30, L1_C_OUT, NUM_ROTATIONS}),
          Z2_cache({28, 28, L2_C_OUT, NUM_ROTATIONS}), Z3_cache({26, 26, L3_C_OUT, NUM_ROTATIONS}),
          X3_cache({26, 26, L3_C_OUT, NUM_ROTATIONS}), X_pooled_cache({L3_C_OUT, NUM_ROTATIONS})
    {
        initialize_weights(L1.W); initialize_weights(L2.W); initialize_weights(L3.W);
    }
    
    Tensor forward_backbone(const Tensor& X_input_2d) {
        START_PROFILE(forward_backbone);
        X0_cache = Tensor({INPUT_SIZE, INPUT_SIZE, 1, 1});
        for(int i=0; i<INPUT_SIZE; ++i)
            for(int j=0; j<INPUT_SIZE; ++j)
                X0_cache(i, j, 0, 0) = X_input_2d(i, j);

        X1_cache = c_g_convolution_forward(L1.W, X0_cache, Z1_cache); 
        X2_cache = c_g_convolution_forward(L2.W, X1_cache, Z2_cache); 
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
        int C = X3_cache.shape[2]; int G = X3_cache.shape[3];
        Tensor dL_dX_conv_L3 = Tensor({X3_cache.shape[0], X3_cache.shape[1], C, G});
        
        int H_in_L3 = 28; int W_in_L3 = 28; 
        Tensor dL_dZ_L3_padded = backward_g_conv_output(dL_dX_conv_L3, Z3_cache, H_in_L3, W_in_L3);
        
        GConvGrads grads_L3 = backward_g_conv_core(dL_dZ_L3_padded, X2_cache, L3.W);
        L3.W = L3.W - grads_L3.dL_dW * LEARNING_RATE;
        
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
        for (int b = 0; b < X.shape[0]; ++b) {
            for (int c_out = 0; c_out < NUM_CLASSES; ++c_out) {
                double sum = 0.0;
                for (int c_in = 0; c_in < INVARIANT_FEATURES; ++c_in) {
                    sum += X(b, c_in) * W(c_in, c_out);
                }
                logits(b, c_out) = sum + B(0, c_out);
            }
        }
        END_PROFILE(classifier_forward);
        return logits;
    }

    Tensor backward_and_update(const Tensor& X, const Tensor& dL_dLogits) {
        START_PROFILE(classifier_backward);
        int batch_size = X.shape[0];
        Tensor dL_dW({INVARIANT_FEATURES, NUM_CLASSES});
        Tensor dL_dB({1, NUM_CLASSES});

        // dL/dW & dL/dB calculation
        for (int c_in = 0; c_in < INVARIANT_FEATURES; ++c_in) {
            for (int c_out = 0; c_out < NUM_CLASSES; ++c_out) {
                for (int b = 0; b < batch_size; ++b) {
                    dL_dW(c_in, c_out) += X(b, c_in) * dL_dLogits(b, c_out);
                    if (c_in == 0) dL_dB(0, c_out) += dL_dLogits(b, c_out); 
                }
            }
        }
        
        Tensor dL_ortho_dW = orthogonal_grad(W);
        dL_dW += dL_ortho_dW;

        W = W - dL_dW * LEARNING_RATE;
        B = B - dL_dB * LEARNING_RATE;

        // dL/dX_invariant
        Tensor dL_dX_invariant({batch_size, INVARIANT_FEATURES});
        for (int b = 0; b < batch_size; ++b) {
            for (int c_in = 0; c_in < INVARIANT_FEATURES; ++c_in) {
                for (int c_out = 0; c_out < NUM_CLASSES; ++c_out) {
                    dL_dX_invariant(b, c_in) += dL_dLogits(b, c_out) * W(c_in, c_out); 
                }
            }
        }
        END_PROFILE(classifier_backward);
        return dL_dX_invariant; 
    }
};

// --- TRAINING LOOP AND EVALUATION ---

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

    start_time_global = Clock::now();
    
    for (int epoch = 1; epoch <= NUM_EPOCHS; ++epoch) {
        double epoch_loss = 0.0;
        
        for (int b = 0; b < num_batches; ++b) {
            
            double elapsed_time = chrono::duration<double>(Clock::now() - start_time_global).count();
            if (elapsed_time > TIME_LIMIT_SECONDS) {
                cout << "\nTraining stopped. Time limit of " << TIME_LIMIT_SECONDS << " seconds reached." << endl;
                return;
            }

            int start_idx = b * BATCH_SIZE;
            Tensor X_batch_feature({BATCH_SIZE, INVARIANT_FEATURES});
            vector<int> Y_batch_true;
            Y_batch_true.reserve(BATCH_SIZE);
            
            // Forward Pass (Sample-by-sample for cache)
            for (int i = 0; i < BATCH_SIZE; ++i) {
                Tensor X_feature_single = backbone.forward_backbone(train_data.X[start_idx + i]);
                for (int c = 0; c < INVARIANT_FEATURES; ++c) {
                    X_batch_feature(i, c) = X_feature_single(0, c);
                }
                Y_batch_true.push_back(train_data.Y_true[start_idx + i]);
            }
            
            // Classifier Forward & Loss Calculation
            Tensor logits = classifier.forward(X_batch_feature);
            Tensor probs = softmax_forward(logits);
            double ce_loss = cross_entropy_loss(probs, Y_batch_true);
            double reg_loss = calculate_orthogonal_loss(classifier.W);
            double total_loss = ce_loss + reg_loss; 
            epoch_loss += total_loss;

            // Backward Pass (Classifier)
            Tensor dL_dLogits = softmax_cross_entropy_backward(probs, Y_batch_true);
            Tensor dL_dX_invariant = classifier.backward_and_update(X_batch_feature, dL_dLogits);

            // Backbone Backward Pass (Sample-by-sample for cache)
            for (int i = 0; i < BATCH_SIZE; ++i) {
                Tensor dL_dX_single({1, INVARIANT_FEATURES});
                for (int c = 0; c < INVARIANT_FEATURES; ++c) {
                    dL_dX_single(0, c) = dL_dX_invariant(i, c);
                }
                backbone.forward_backbone(train_data.X[start_idx + i]); // Reload caches (necessary overhead)
                backbone.full_backward_L3(dL_dX_single);
            }

            cout << "Epoch " << epoch << " | Batch " << b+1 << "/" << num_batches 
                 << " | Loss: " << total_loss << " (CE: " << ce_loss << " + Reg: " << reg_loss << ")" << endl;
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

    double gcnn_forward_time = profiling_times["forward_backbone"];
    double gcnn_backprop_time = profiling_times["full_backward_L1"] + profiling_times["full_backward_L2"] + profiling_times["full_backward_L3"];
    double classifier_time = profiling_times["classifier_forward"] + profiling_times["classifier_backward"];

    cout << "\n--- Functional Breakdown ---" << endl;
    cout << fixed << setprecision(4);
    cout << "Total GCNN Forward Time:  " << gcnn_forward_time << " s (" << (gcnn_forward_time / total_time) * 100.0 << " %)" << endl;
    cout << "Total GCNN Backward Time: " << gcnn_backprop_time << " s (" << (gcnn_backprop_time / total_time) * 100.0 << " %)" << endl;
    cout << "Total Classifier Time:    " << classifier_time << " s (" << (classifier_time / total_time) * 100.0 << " %)" << endl;
}


int main() {
    cout << fixed << setprecision(8);
    cout << "--- Starting G-CNN Training Simulation (OPTIMIZED) ---" << endl;

    GCNN backbone;
    LinearClassifier classifier;
    
    DataSet train_data = generate_dataset(TRAIN_SAMPLES);
    DataSet test_data = generate_dataset(TEST_SAMPLES);
    
    train_model(backbone, classifier, train_data, test_data);
    
    summarize_profiling();
    
    return 0;
}
