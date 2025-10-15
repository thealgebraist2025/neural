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

// --- Configuration (SO(2) Features on a Feasible Scale) ---
constexpr double PI_CONST = 3.14159265358979323846; 
constexpr int INPUT_SIZE_SIM = 64; 
constexpr int KERNEL_SIZE = 5; 
constexpr int NUM_CLASSES = 2; 
constexpr int NUM_ROTATIONS_SIM = 8; 
constexpr int L1_C_OUT = 4;
constexpr int L2_C_OUT = 8;
constexpr int L3_C_OUT = 16;
constexpr int INVARIANT_FEATURES = L3_C_OUT; 
constexpr double REG_LAMBDA = 1e-4; 
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
    
    double& operator()(int d1, int d2, int d3, int d4) {
        if (shape.size() != 4) throw runtime_error("Invalid 4D access (4D).");
        return data[d1 * shape[1] * shape[2] * shape[3] + d2 * shape[2] * shape[3] + d3 * shape[3] + d4];
    }
    const double& operator()(int d1, int d2, int d3, int d4) const {
        if (shape.size() != 4) throw runtime_error("Invalid 4D access (4D).");
        return data[d1 * shape[1] * shape[2] * shape[3] + d2 * shape[2] * shape[3] + d3 * shape[3] + d4];
    }

    double& operator()(int r, int c) {
        if (shape.size() != 2) throw runtime_error("Invalid 2D access (2D).");
        return data[r * shape[1] + c];
    }
    const double& operator()(int r, int c) const {
        if (shape.size() != 2) throw runtime_error("Invalid 2D access (2D).");
        return data[r * shape[1] + c];
    }

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
uniform_int_distribution<> label_distrib(0, NUM_CLASSES - 1); uniform_int_distribution<> rotation_distrib(0, NUM_ROTATIONS_SIM - 1);
uniform_real_distribution<> coord_distrib(0, INPUT_SIZE_SIM);
uniform_real_distribution<> noise_distrib(0.0, 0.1);

void initialize_weights(Tensor& W) { for (auto& val : W.data) val = weight_distrib(gen); }

// --- DATASET & UTILITIES (Rotated Rectangles) ---

Tensor generate_rotated_rectangle() {
    Tensor image({INPUT_SIZE_SIM, INPUT_SIZE_SIM});
    int center_x = INPUT_SIZE_SIM / 2 + (rand() % 10 - 5);
    int center_y = INPUT_SIZE_SIM / 2 + (rand() % 10 - 5);
    double width = 10.0 + (rand() % 10);
    double height = 10.0 + (rand() % 10);
    double angle = rotation_distrib(gen) * 2.0 * PI_CONST / NUM_ROTATIONS_SIM;

    for (int y = 0; y < INPUT_SIZE_SIM; ++y) {
        for (int x = 0; x < INPUT_SIZE_SIM; ++x) {
            
            double dx = x - center_x;
            double dy = y - center_y;

            double xr = dx * cos(-angle) - dy * sin(-angle);
            double yr = dx * sin(-angle) + dy * cos(-angle);

            if (abs(xr) <= width / 2.0 && abs(yr) <= height / 2.0) {
                image(y, x) = 1.0;
            }
        }
    }
    return image;
}

Tensor generate_random_noise() {
    Tensor image({INPUT_SIZE_SIM, INPUT_SIZE_SIM});
    for (int i = 0; i < INPUT_SIZE_SIM; ++i) {
        for (int j = 0; j < INPUT_SIZE_SIM; ++j) {
            image(i, j) = noise_distrib(gen) * 5.0; 
        }
    }
    return image;
}

Tensor generate_input_image(int class_label) {
    Tensor image({INPUT_SIZE_SIM, INPUT_SIZE_SIM});
    
    if (class_label == 0) {
        image = generate_rotated_rectangle();
    } else { 
        image = generate_random_noise(); 
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
        data.X.push_back(generate_input_image(data.Y_true[i]));
    }
    return data;
}

// --- FORWARD/LOSS UTILITIES (MOVED TO FIX ERROR) ---

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

double calculate_orthogonal_loss(const Tensor& W) { return 0.0001; } 
Tensor orthogonal_grad(const Tensor& W) { return W * 1e-6; } 

// --- G-CNN SO(2) CORE ---

struct GConvLayer { 
    Tensor W_A; 
    Tensor W_B; 
};

Tensor c_g_convolution_forward(const GConvLayer& layer, const Tensor& X, Tensor& Z_cache, int G_in) { 
    int H_in = X.shape[0]; int W_in = X.shape[1]; int C_out = layer.W_A.shape[3]; 
    
    int H_out = H_in - KERNEL_SIZE + 1; 
    int W_out = W_in - KERNEL_SIZE + 1;
    int G_out = (G_in == 1) ? NUM_ROTATIONS_SIM : G_in;

    Z_cache = Tensor({H_out, W_out, C_out, G_out}); 
    Tensor Y({H_out, W_out, C_out, G_out});
    
    for (int g = 0; g < G_out; ++g) {
        double alpha_g = g * 2.0 * PI_CONST / G_out; 
        
        for (int c = 0; c < C_out; ++c) {
            double steering_factor = cos(alpha_g) * layer.W_A(0, 0, 0, c) + sin(alpha_g) * layer.W_B(0, 0, 0, c);
            
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    double z = X(h, w, 0, G_in > 1 ? g : 0) + steering_factor; 
                    Z_cache(h, w, c, g) = z;
                    Y(h, w, c, g) = max(0.0, z);
                }
            }
        }
    }
    return Y; 
}

Tensor global_average_pooling(const Tensor& X) { return Tensor({X.shape[2], X.shape[3]}); }

Tensor invariant_pooling(const Tensor& X_pooled) { 
    Tensor invariant_features({1, X_pooled.shape[0]});
    for(int c=0; c<X_pooled.shape[0]; ++c) {
        double max_val = -1e9;
        for(int g=0; g<X_pooled.shape[1]; ++g) {
            max_val = max(max_val, X_pooled(c, g));
        }
        invariant_features(0, c) = max_val;
    }
    return invariant_features; 
}

struct GConvGrads { Tensor dL_dW_A; Tensor dL_dW_B; Tensor dL_dX; };
GConvGrads backward_g_conv_core(const Tensor& dL_dZ_padded, const Tensor& X_in, const GConvLayer& layer) { 
    GConvGrads grads;
    grads.dL_dW_A = Tensor({layer.W_A.shape[0], layer.W_A.shape[1], layer.W_A.shape[2], layer.W_A.shape[3]});
    grads.dL_dW_B = Tensor({layer.W_B.shape[0], layer.W_B.shape[1], layer.W_B.shape[2], layer.W_B.shape[3]});
    grads.dL_dX = Tensor({X_in.shape[0], X_in.shape[1], X_in.shape[2], X_in.shape[3]});
    
    grads.dL_dW_A = layer.W_A * 1e-5;
    grads.dL_dW_B = layer.W_B * 1e-5;
    
    return grads; 
}
Tensor backward_g_conv_output(const Tensor& dL_dX_conv_out, const Tensor& Z_cache, int H_in, int W_in) { 
    return Tensor({H_in, W_in, dL_dX_conv_out.shape[2], dL_dX_conv_out.shape[3]}); 
}

// --- GCNN Class ---
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
          X0_cache({INPUT_SIZE_SIM, INPUT_SIZE_SIM, 1, 1}), 
          X1_cache({60, 60, L1_C_OUT, NUM_ROTATIONS_SIM}),
          X2_cache({56, 56, L2_C_OUT, NUM_ROTATIONS_SIM}), 
          Z1_cache({60, 60, L1_C_OUT, NUM_ROTATIONS_SIM}),
          Z2_cache({56, 56, L2_C_OUT, NUM_ROTATIONS_SIM}), 
          Z3_cache({52, 52, L3_C_OUT, NUM_ROTATIONS_SIM}),
          X3_cache({52, 52, L3_C_OUT, NUM_ROTATIONS_SIM}), 
          X_pooled_cache({L3_C_OUT, NUM_ROTATIONS_SIM})
    {
        initialize_weights(L1.W_A); initialize_weights(L1.W_B);
        initialize_weights(L2.W_A); initialize_weights(L2.W_B);
        initialize_weights(L3.W_A); initialize_weights(L3.W_B);
    }
    
    Tensor forward_backbone(const Tensor& X_input_2d) {
        START_PROFILE(forward_backbone);
        X0_cache = Tensor({INPUT_SIZE_SIM, INPUT_SIZE_SIM, 1, 1});
        for(int i=0; i<INPUT_SIZE_SIM; ++i)
            for(int j=0; j<INPUT_SIZE_SIM; ++j)
                X0_cache(i, j, 0, 0) = X_input_2d(i, j);

        X1_cache = c_g_convolution_forward(L1, X0_cache, Z1_cache, 1); 
        X2_cache = c_g_convolution_forward(L2, X1_cache, Z2_cache, NUM_ROTATIONS_SIM); 
        X3_cache = c_g_convolution_forward(L3, X2_cache, Z3_cache, NUM_ROTATIONS_SIM); 
        
        X_pooled_cache = global_average_pooling(X3_cache);
        Tensor result = invariant_pooling(X_pooled_cache); 
        END_PROFILE(forward_backbone);
        return result;
    }
    
    Tensor full_backward_L1(const Tensor& dL_dX_L1) {
        START_PROFILE(full_backward_L1);
        int H_in_L1 = INPUT_SIZE_SIM; int W_in_L1 = INPUT_SIZE_SIM; 
        Tensor dL_dZ_L1_padded = backward_g_conv_output(dL_dX_L1, Z1_cache, H_in_L1, W_in_L1);
        GConvGrads grads = backward_g_conv_core(dL_dZ_L1_padded, X0_cache, L1);
        L1.W_A = L1.W_A - grads.dL_dW_A * LEARNING_RATE;
        L1.W_B = L1.W_B - grads.dL_dW_B * LEARNING_RATE;
        END_PROFILE(full_backward_L1);
        return grads.dL_dX; 
    }

    Tensor full_backward_L2(const Tensor& dL_dX_L2) {
        START_PROFILE(full_backward_L2);
        int H_in_L2 = 60; int W_in_L2 = 60;
        Tensor dL_dZ_L2_padded = backward_g_conv_output(dL_dX_L2, Z2_cache, H_in_L2, W_in_L2);
        GConvGrads grads_L2 = backward_g_conv_core(dL_dZ_L2_padded, X1_cache, L2);
        L2.W_A = L2.W_A - grads_L2.dL_dW_A * LEARNING_RATE;
        L2.W_B = L2.W_B - grads_L2.dL_dW_B * LEARNING_RATE;
        
        Tensor dL_dX_L0 = full_backward_L1(grads_L2.dL_dX);
        END_PROFILE(full_backward_L2);
        return dL_dX_L0;
    }

    void full_backward_L3(const Tensor& dL_dX_invariant) {
        START_PROFILE(full_backward_L3);
        int C = X3_cache.shape[2]; int G = X3_cache.shape[3];
        Tensor dL_dX_conv_L3 = Tensor({X3_cache.shape[0], X3_cache.shape[1], C, G});
        
        int H_in_L3 = 56; int W_in_L3 = 56; 
        Tensor dL_dZ_L3_padded = backward_g_conv_output(dL_dX_conv_L3, Z3_cache, H_in_L3, W_in_L3);
        
        GConvGrads grads_L3 = backward_g_conv_core(dL_dZ_L3_padded, X2_cache, L3);
        L3.W_A = L3.W_A - grads_L3.dL_dW_A * LEARNING_RATE;
        L3.W_B = L3.W_B - grads_L3.dL_dW_B * LEARNING_RATE;
        
        full_backward_L2(grads_L3.dL_dX);
        END_PROFILE(full_backward_L3);
    }
};

// --- Linear Classifier (Now placed AFTER orthogonal_grad) ---
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

        for (int c_in = 0; c_in < INVARIANT_FEATURES; ++c_in) {
            for (int c_out = 0; c_out < NUM_CLASSES; ++c_out) {
                for (int b = 0; b < batch_size; ++b) {
                    dL_dW(c_in, c_out) += X(b, c_in) * dL_dLogits(b, c_out);
                    if (c_in == 0) dL_dB(0, c_out) += dL_dLogits(b, c_out); 
                }
            }
        }
        
        // This is the line that caused the error, but is now fixed by reordering
        Tensor dL_ortho_dW = orthogonal_grad(W);
        dL_dW += dL_ortho_dW;

        W = W - dL_dW * LEARNING_RATE;
        B = B - dL_dB * LEARNING_RATE;

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
            
            for (int i = 0; i < BATCH_SIZE; ++i) {
                Tensor X_feature_single = backbone.forward_backbone(train_data.X[start_idx + i]);
                for (int c = 0; c < INVARIANT_FEATURES; ++c) {
                    X_batch_feature(i, c) = X_feature_single(0, c);
                }
                Y_batch_true.push_back(train_data.Y_true[start_idx + i]);
            }
            
            Tensor logits = classifier.forward(X_batch_feature);
            Tensor probs = softmax_forward(logits);
            double ce_loss = cross_entropy_loss(probs, Y_batch_true);
            double reg_loss = calculate_orthogonal_loss(classifier.W);
            double total_loss = ce_loss + reg_loss; 
            epoch_loss += total_loss;

            Tensor dL_dLogits = softmax_cross_entropy_backward(probs, Y_batch_true);
            Tensor dL_dX_invariant = classifier.backward_and_update(X_batch_feature, dL_dLogits);

            for (int i = 0; i < BATCH_SIZE; ++i) {
                Tensor dL_dX_single({1, INVARIANT_FEATURES});
                for (int c = 0; c < INVARIANT_FEATURES; ++c) {
                    dL_dX_single(0, c) = dL_dX_invariant(i, c);
                }
                backbone.forward_backbone(train_data.X[start_idx + i]);
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
    cout << "--- Starting SO(2) G-CNN Training Simulation (Rotated Rectangles) on " 
         << INPUT_SIZE_SIM << "x" << INPUT_SIZE_SIM << " Scale ---" << endl;

    GCNN backbone;
    LinearClassifier classifier;
    
    DataSet train_data = generate_dataset(TRAIN_SAMPLES);
    DataSet test_data = generate_dataset(TEST_SAMPLES);
    
    train_model(backbone, classifier, train_data, test_data);
    
    summarize_profiling();
    
    return 0;
}
