#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h> 

// --- STB Image Write Configuration ---
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Define M_PI explicitly
#define M_PI 3.14159265358979323846

// --- Global Configuration ---
#define GRID_SIZE 32        // 32x32 resolution for recognition
#define NUM_DEFORMATIONS 2  
#define NUM_VECTORS 16      
#define NUM_BINS 32         
#define NUM_FEATURES (NUM_VECTORS * NUM_BINS) 
#define PIXEL_LOSS_WEIGHT 2.5 
#define NUM_POINTS 200
#define ITERATIONS 500      
#define GRADIENT_EPSILON 0.01 
#define NUM_IDEAL_CHARS 36  
#define NUM_CONTROL_POINTS 9 
#define MAX_PIXEL_ERROR (GRID_SIZE * GRID_SIZE) 
#define TIME_LIMIT_SECONDS 240.0 

// Segmentation limits
#define MAX_SEGMENTS 100 // Maximum number of letters we expect to find

// PNG Rendering Constants (Vertical Layout)
#define PIXEL_SIZE 2    
#define IMG_SIZE (GRID_SIZE * PIXEL_SIZE) 
#define IMG_SPACING 5   
#define TEXT_HEIGHT 15  
#define SET_SPACING 25  
#define GRAPH_WIDTH 100 
#define GRAPH_HEIGHT IMG_SIZE 
#define NUM_CHANNELS 3 

// --- Data Structures ---

typedef struct { double x; double y; } Point;
typedef struct { const Point control_points[NUM_CONTROL_POINTS]; } Ideal_Curve_Params; 
typedef struct { double alpha[NUM_DEFORMATIONS]; } Deformation_Coefficients;
typedef double Generated_Image[GRID_SIZE][GRID_SIZE]; 
typedef double Feature_Vector[NUM_FEATURES]; 
typedef struct { double estimated_alpha[NUM_DEFORMATIONS]; double final_loss; double loss_history[1]; } EstimationResult; // Simplified for this context

typedef struct {
    int start; 
    int end; 
} Boundary;

typedef struct {
    int x_start, x_end; 
    int y_start, y_end; 
    Generated_Image resized_img; // The 32x32 image segment for recognition
    int best_match_index; 
    double final_loss;
    double estimated_alpha[NUM_DEFORMATIONS];
} SegmentResult;


// --- Memory Tracking Globals ---
size_t total_allocated_bytes = 0;
size_t total_freed_bytes = 0;

// --- Function Prototypes (Fixes Implicit Declarations) ---
void* safe_malloc(size_t size);
void safe_free(void *ptr, size_t size);
void draw_curve(const double alpha[NUM_DEFORMATIONS], Generated_Image img, const Ideal_Curve_Params *const ideal_params);
void extract_geometric_features(const Generated_Image img, Feature_Vector features_out);
double calculate_pixel_loss_L2(const Generated_Image generated, const Generated_Image observed);
double calculate_feature_loss_L2(const Feature_Vector generated, const Feature_Vector observed);
double calculate_combined_loss(const Generated_Image generated_img, const Feature_Vector generated_features,
                               const Generated_Image observed_img, const Feature_Vector observed_features);
void run_optimization(const Generated_Image observed_image, const Feature_Vector observed_features, 
                      int ideal_char_index, EstimationResult *result);
void set_pixel(unsigned char *buffer, int x, int y, int width, int height, unsigned char r, unsigned char g, unsigned char b);
void get_pixel_color(double intensity, int is_error_map, unsigned char *r, unsigned char *g, unsigned char *b);
void draw_text_placeholder_box(unsigned char *buffer, int buf_width, int buf_height, int x, int y, int width, int height, unsigned char r, unsigned char g, unsigned char b);
void render_single_image_to_png(unsigned char *buffer, int buf_width, int buf_height, const Generated_Image img, int x_offset, int y_offset, int is_error_map);


// --- Fixed Ideal Curves (Omitted most for brevity but ensure 'O' is fixed) ---
const char *CHAR_NAMES[NUM_IDEAL_CHARS] = {
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", 
    "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", 
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
};
const Ideal_Curve_Params IDEAL_TEMPLATES[NUM_IDEAL_CHARS] = {
    // A...
    [0] = {.control_points = {{.x = 0.5, .y = 0.1}, {.x = 0.3, .y = 0.3}, {.x = 0.2, .y = 0.5}, {.x = 0.3, .y = 0.6}, 
        {.x = 0.7, .y = 0.6}, {.x = 0.8, .y = 0.5}, {.x = 0.7, .y = 0.3}, {.x = 0.5, .y = 0.1}, {.x = 0.7, .y = 0.9} 
    }},
    // O (FIXED: Duplicated closing point to prevent superfluous internal lines)
    [14] = {.control_points = {{.x = 0.5, .y = 0.1}, {.x = 0.85, .y = 0.3}, {.x = 0.85, .y = 0.7}, 
        {.x = 0.5, .y = 0.9}, {.x = 0.15, .y = 0.7}, {.x = 0.15, .y = 0.3}, 
        {.x = 0.5, .y = 0.1}, {.x = 0.5, .y = 0.1}, {.x = 0.5, .y = 0.1}  
    }},
    // ... all 36 characters are here ...
    [35] = {.control_points = {{.x = 0.5, .y = 0.1}, {.x = 0.8, .y = 0.2}, {.x = 0.5, .y = 0.4}, {.x = 0.2, .y = 0.2}, 
        {.x = 0.5, .y = 0.1}, {.x = 0.8, .y = 0.4}, {.x = 0.8, .y = 0.9}, {.x = 0.5, .y = 0.7}, 
        {.x = 0.8, .y = 0.9} 
    }}
};


// --- Memory Wrapper Functions (Restored) ---

void* safe_malloc(size_t size) {
    void *ptr = malloc(size);
    if (ptr) {
        total_allocated_bytes += size;
    }
    return ptr;
}

void safe_free(void *ptr, size_t size) {
    if (ptr) {
        free(ptr);
        total_freed_bytes += size;
    }
}


// --- Core Recognition Functions (Restored Prototypes and Logic) ---

void apply_deformation(Point *point, const double alpha[NUM_DEFORMATIONS]) {
    point->x = point->x + alpha[0] * (point->y - 0.5);
    point->x = point->x + alpha[1] * sin(M_PI * point->y);
}

Point get_deformed_point(const double t, const Ideal_Curve_Params *const params, const double alpha[NUM_DEFORMATIONS]) {
    Point p = {.x = 0.0, .y = 0.0};
    const int N = NUM_CONTROL_POINTS; 
    const double segment_length_t = 1.0 / (N - 1); 

    int segment_index = (int)floor(t / segment_length_t);
    if (segment_index >= N - 1) segment_index = N - 2; 

    const Point P_start = params->control_points[segment_index];
    const Point P_end = params->control_points[segment_index + 1];

    const double segment_t = (t - segment_index * segment_length_t) / segment_length_t;

    p.x = P_start.x + (P_end.x - P_start.x) * segment_t;
    p.y = P_start.y + (P_end.y - P_start.y) * segment_t;

    apply_deformation(&p, alpha);

    p.x = fmax(0.0, fmin(GRID_SIZE - 1.0, p.x * GRID_SIZE));
    p.y = fmax(0.0, fmin(GRID_SIZE - 1.0, p.y * GRID_SIZE));

    return p;
}

void draw_curve(const double alpha[NUM_DEFORMATIONS], Generated_Image img, const Ideal_Curve_Params *const ideal_params) {
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            img[i][j] = 0.0;
        }
    }

    for (int i = 0; i <= NUM_POINTS; i++) {
        const double t = (double)i / NUM_POINTS;
        const Point current_p = get_deformed_point(t, ideal_params, alpha);

        const int px = (int)round(current_p.x);
        const int py = (int)round(current_p.y);

        if (px >= 0 && px < GRID_SIZE && py >= 0 && py < GRID_SIZE) {
            img[py][px] = fmin(1.0, img[py][px] + 0.5); 
        }
    }
}

void extract_geometric_features(const Generated_Image img, Feature_Vector features_out) {
    double vectors[NUM_VECTORS][2];
    for (int k = 0; k < NUM_VECTORS; k++) { 
        const double angle = 2.0 * M_PI * k / NUM_VECTORS; 
        vectors[k][0] = cos(angle);
        vectors[k][1] = sin(angle);
    }
    
    const double center = (GRID_SIZE - 1.0) / 2.0; 
    const double MAX_PROJECTION_MAGNITUDE = sqrt(center * center + center * center); 

    for (int k = 0; k < NUM_FEATURES; k++) {
        features_out[k] = 0.0;
    }

    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            const double intensity = img[i][j];
            if (intensity < 0.1) continue; 

            const double vx = (double)j - center;
            const double vy = (double)i - center;
            
            for (int k = 0; k < NUM_VECTORS; k++) {
                const double projection = (vx * vectors[k][0] + vy * vectors[k][1]);
                const double normalized_projection = projection / MAX_PROJECTION_MAGNITUDE;
                
                int bin_index = (int)floor((normalized_projection + 1.0) * (NUM_BINS / 2.0));
                
                if (bin_index < 0) bin_index = 0;
                if (bin_index >= NUM_BINS) bin_index = NUM_BINS - 1;
                
                const int feature_index = k * NUM_BINS + bin_index;
                features_out[feature_index] += intensity;
            }
        }
    }
}

double calculate_feature_loss_L2(const Feature_Vector generated, const Feature_Vector observed) {
    double loss = 0.0;
    for (int k = 0; k < NUM_FEATURES; k++) {
        const double error = observed[k] - generated[k];
        loss += error * error;
    }
    return loss;
}

double calculate_pixel_loss_L2(const Generated_Image generated, const Generated_Image observed) {
    double loss = 0.0;
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            const double error = observed[i][j] - generated[i][j];
            loss += error * error;
        }
    }
    return loss;
}

double calculate_combined_loss(const Generated_Image generated_img, const Feature_Vector generated_features,
                               const Generated_Image observed_img, const Feature_Vector observed_features) {
    
    const double feature_loss = calculate_feature_loss_L2(generated_features, observed_features);
    const double pixel_loss = calculate_pixel_loss_L2(generated_img, observed_img);
    
    return feature_loss + PIXEL_LOSS_WEIGHT * pixel_loss;
}

void calculate_gradient(const Generated_Image observed_img, const Feature_Vector observed_features, 
                        const Deformation_Coefficients *const alpha, const double loss_base, 
                        double grad_out[NUM_DEFORMATIONS], const Ideal_Curve_Params *const ideal_params) {
    
    const double epsilon = GRADIENT_EPSILON; 
    Generated_Image generated_img_perturbed; 
    Feature_Vector generated_features_perturbed; 

    for (int k = 0; k < NUM_DEFORMATIONS; k++) {
        Deformation_Coefficients alpha_perturbed = *alpha;
        alpha_perturbed.alpha[k] += epsilon; 

        draw_curve(alpha_perturbed.alpha, generated_img_perturbed, ideal_params);
        extract_geometric_features(generated_img_perturbed, generated_features_perturbed);
        
        const double loss_perturbed = calculate_combined_loss(generated_img_perturbed, generated_features_perturbed, observed_img, observed_features);

        grad_out[k] = (loss_perturbed - loss_base) / epsilon;
    }
}


void run_optimization(const Generated_Image observed_image, const Feature_Vector observed_features, 
                      int ideal_char_index, EstimationResult *result) {
    
    const Ideal_Curve_Params *ideal_params = &IDEAL_TEMPLATES[ideal_char_index];
    
    Deformation_Coefficients alpha_hat = { .alpha = {0.0, 0.0} };
    double learning_rate = 0.0000001; 
    const double min_learning_rate = 0.00000000005;
    double gradient[NUM_DEFORMATIONS];
    Generated_Image generated_image;
    Feature_Vector generated_features;
    double combined_loss;
    double prev_combined_loss = HUGE_VAL; 
    double current_feature_loss_only = HUGE_VAL;

    for (int t = 0; t <= ITERATIONS; t++) {
        draw_curve(alpha_hat.alpha, generated_image, ideal_params);
        extract_geometric_features(generated_image, generated_features);
        
        current_feature_loss_only = calculate_feature_loss_L2(generated_features, observed_features);
        combined_loss = current_feature_loss_only + PIXEL_LOSS_WEIGHT * calculate_pixel_loss_L2(generated_image, observed_image);

        if (combined_loss > prev_combined_loss * 1.001 && learning_rate > min_learning_rate) { 
            learning_rate *= 0.5;
        }

        prev_combined_loss = combined_loss;

        if (t < ITERATIONS) {
            calculate_gradient(observed_image, observed_features, &alpha_hat, combined_loss, gradient, ideal_params);
            
            double step_rate = (learning_rate > min_learning_rate) ? learning_rate : min_learning_rate;
            
            alpha_hat.alpha[0] -= step_rate * gradient[0];
            alpha_hat.alpha[1] -= step_rate * gradient[1];
        }
    }

    result->final_loss = current_feature_loss_only;
    memcpy(result->estimated_alpha, alpha_hat.alpha, sizeof(double) * NUM_DEFORMATIONS);
}

// --- PNG Rendering Functions (Restored) ---

void set_pixel(unsigned char *buffer, int x, int y, int width, int height, unsigned char r, unsigned char g, unsigned char b) {
    if (x >= 0 && x < width && y >= 0 && y < height) {
        long index = (long)y * width * NUM_CHANNELS + x * NUM_CHANNELS;
        buffer[index] = r;
        buffer[index + 1] = g;
        buffer[index + 2] = b;
    }
}

void get_pixel_color(double intensity, int is_error_map, unsigned char *r, unsigned char *g, unsigned char *b) {
    double clamped_intensity = fmax(0.0, fmin(1.0, intensity));

    if (is_error_map) {
        if (clamped_intensity > 0.3) {
            *r = 255; *g = 50; *b = 50; 
        } else if (clamped_intensity > 0.1) {
            *r = 255; *g = 150; *b = 0; 
        } else {
            *r = 0; *g = 0; *b = 0; 
        }
    } else {
        if (clamped_intensity > 0.6) {
            *r = 255; *g = 255; *b = 100; 
        } else if (clamped_intensity > 0.3) {
            *r = 100; *g = 100; *b = 100; 
        } else {
            *r = 0; *g = 0; *b = 0; 
        }
    }
}

void draw_text_placeholder_box(unsigned char *buffer, int buf_width, int buf_height, int x, int y, int width, int height, unsigned char r, unsigned char g, unsigned char b) {
    for(int py = 0; py < height; py++) {
        for(int px = 0; px < width; px++) {
            set_pixel(buffer, x + px, y + py, buf_width, buf_height, r / 3, g / 3, b / 3);
            if (py == 0 || py == height - 1 || px == 0 || px == width - 1) {
                 set_pixel(buffer, x + px, y + py, buf_width, buf_height, r, g, b);
            }
        }
    }
}

void render_single_image_to_png(unsigned char *buffer, int buf_width, int buf_height, const Generated_Image img, int x_offset, int y_offset, int is_error_map) {
    unsigned char r, g, b;
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            get_pixel_color(img[i][j], is_error_map, &r, &g, &b);
            
            for (int py = 0; py < PIXEL_SIZE; py++) {
                for (int px = 0; px < PIXEL_SIZE; px++) {
                    set_pixel(buffer, 
                              x_offset + j * PIXEL_SIZE + px, 
                              y_offset + i * PIXEL_SIZE + py, 
                              buf_width, buf_height, r, g, b);
                }
            }
        }
    }
}


// --- Image Loading and Preprocessing ---

int load_image_stb(const char *filename, double **data_out, int *width_out, int *height_out) {
    printf("--- WARNING: Skipping actual image load (stbi_load not available) ---\n");
    printf("Simulating image load with a 100x100 image containing 'TEST' like shapes.\n");

    *width_out = 100;
    *height_out = 100;
    size_t size = (size_t)(*width_out) * (*height_out);
    *data_out = (double *)safe_malloc(sizeof(double) * size);
    
    if (*data_out == NULL) return 0;

    for (int i = 0; i < *height_out; i++) {
        for (int j = 0; j < *width_out; j++) {
            (*data_out)[i * (*width_out) + j] = 0.0; // Black background
        }
    }
    
    // Simulate line 1: T
    for (int j = 10; j <= 30; j++) (*data_out)[15 * (*width_out) + j] = 1.0; // Horizontal bar
    for (int i = 15; i <= 40; i++) (*data_out)[i * (*width_out) + 20] = 1.0; // Vertical bar

    // Simulate line 1: E
    for (int j = 40; j <= 60; j++) (*data_out)[15 * (*width_out) + j] = 1.0; // Top
    for (int j = 40; j <= 60; j++) (*data_out)[25 * (*width_out) + j] = 1.0; // Middle
    for (int j = 40; j <= 60; j++) (*data_out)[35 * (*width_out) + j] = 1.0; // Bottom
    for (int i = 15; i <= 35; i++) (*data_out)[i * (*width_out) + 40] = 1.0; // Vertical

    // Simulate line 2: S
    for (int j = 10; j <= 30; j++) (*data_out)[60 * (*width_out) + j] = 1.0;
    for (int i = 60; i <= 70; i++) (*data_out)[i * (*width_out) + 10] = 1.0;
    for (int j = 10; j <= 30; j++) (*data_out)[70 * (*width_out) + j] = 1.0;
    for (int i = 70; i <= 80; i++) (*data_out)[i * (*width_out) + 30] = 1.0;
    for (int j = 10; j <= 30; j++) (*data_out)[80 * (*width_out) + j] = 1.0;


    return 1;
}

void resize_segment(const double *full_data, int full_width, int full_height, 
                    int x_start, int x_end, int y_start, int y_end, 
                    Generated_Image segment_out) {
    
    int segment_w = x_end - x_start;
    int segment_h = y_end - y_start;

    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            
            int src_y = y_start + (int)round((double)i / (GRID_SIZE - 1) * segment_h);
            int src_x = x_start + (int)round((double)j / (GRID_SIZE - 1) * segment_w);

            src_y = fmax(0, fmin(full_height - 1, src_y));
            src_x = fmax(0, fmin(full_width - 1, src_x));

            segment_out[i][j] = full_data[src_y * full_width + src_x];
        }
    }
}


// --- Histogram Segmentation Logic ---

void project_histogram(const double *full_data, int width, int height, int orientation, double *hist_out) {
    int hist_size = (orientation == 0) ? height : width; 
    
    for (int i = 0; i < hist_size; i++) {
        hist_out[i] = 0.0;
    }

    if (orientation == 0) { // Horizontal projection (sum rows -> Y-axis for lines)
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                hist_out[i] += full_data[i * width + j];
            }
        }
    } else { // Vertical projection (sum columns -> X-axis for letters)
        for (int j = 0; j < width; j++) {
            for (int i = 0; i < height; i++) {
                hist_out[j] += full_data[i * width + j];
            }
        }
    }
}

int find_zero_intervals(const double *hist, int size, int min_zero_length, double threshold, Boundary *boundaries_out) {
    int final_count = 0;
    int content_start = 0; // The start of the current content block

    for (int i = 0; i < size; i++) {
        if (hist[i] < threshold) {
            // Found a zero region, look ahead
            int zero_end = i;
            while (zero_end < size && hist[zero_end] < threshold) {
                zero_end++;
            }
            
            int zero_length = zero_end - i;

            if (zero_length >= min_zero_length) {
                // Found a significant gap, the content ends at i-1
                if (i - content_start > min_zero_length) { // Minimum content size
                    boundaries_out[final_count].start = content_start;
                    boundaries_out[final_count].end = i - 1;
                    if (final_count < MAX_SEGMENTS) final_count++;
                }
                
                // New content starts after the gap
                content_start = zero_end;
            }
            // Skip past the gap we just processed
            i = zero_end - 1; 
        }
    }
    
    // Process the final content block
    if (size - content_start > min_zero_length) {
        boundaries_out[final_count].start = content_start;
        boundaries_out[final_count].end = size - 1;
        if (final_count < MAX_SEGMENTS) final_count++;
    }

    return final_count;
}

int segment_image_naive(const double *full_data, int full_width, int full_height, SegmentResult *segments_out) {
    int total_segments = 0;
    
    // 1. Vertical Histogram Projection (Find Lines - Y-axis)
    double *h_hist = (double *)safe_malloc(sizeof(double) * full_height);
    Boundary line_boundaries[MAX_SEGMENTS];
    project_histogram(full_data, full_width, full_height, 0, h_hist);
    
    int num_lines = find_zero_intervals(h_hist, full_height, full_height / 15, 0.01, line_boundaries);
    safe_free(h_hist, sizeof(double) * full_height);
    
    printf("Segmentation: Found %d line(s).\n", num_lines);

    // 2. Horizontal Histogram Projection (Find Letters - X-axis)
    for (int l = 0; l < num_lines; l++) {
        int line_y_start = line_boundaries[l].start;
        int line_y_end = line_boundaries[l].end;
        
        // Manual vertical projection for the current line segment
        double *v_hist = (double *)safe_malloc(sizeof(double) * full_width);
        
        for (int j = 0; j < full_width; j++) {
            v_hist[j] = 0.0;
            for (int i = line_y_start; i <= line_y_end; i++) {
                v_hist[j] += full_data[i * full_width + j];
            }
        }
        
        Boundary letter_boundaries[MAX_SEGMENTS];
        // Letter segmentation: smaller minimum gap required
        int num_letters = find_zero_intervals(v_hist, full_width, full_width / 40, 0.01, letter_boundaries);
        safe_free(v_hist, sizeof(double) * full_width);
        
        printf("  Line %d (Y:[%d,%d]): Found %d letter(s).\n", l + 1, line_y_start, line_y_end, num_letters);

        // 3. Process each letter segment
        for (int c = 0; c < num_letters; c++) {
            if (total_segments >= MAX_SEGMENTS) {
                printf("  WARNING: Reached maximum segment limit (%d).\n", MAX_SEGMENTS);
                return total_segments;
            }
            
            SegmentResult *seg = &segments_out[total_segments];
            // Add a small buffer to the y-bounds to capture any ascenders/descenders lost in line segmentation
            seg->y_start = fmax(0, line_y_start - 2); 
            seg->y_end = fmin(full_height - 1, line_y_end + 2);
            seg->x_start = letter_boundaries[c].start;
            seg->x_end = letter_boundaries[c].end;
            
            // Resize and store the segment as a 32x32 image
            resize_segment(full_data, full_width, full_height, 
                           seg->x_start, seg->x_end, seg->y_start, seg->y_end, 
                           seg->resized_img);
                           
            total_segments++;
        }
    }
    
    return total_segments;
}


// --- Recognition Function ---

void recognize_segment(SegmentResult *segment) {
    Feature_Vector observed_features;
    extract_geometric_features(segment->resized_img, observed_features);
    
    double min_feature_loss = HUGE_VAL;
    int best_match_index = -1;

    EstimationResult current_result = {0}; // Zero-initialize

    for (int i = 0; i < NUM_IDEAL_CHARS; i++) {
        run_optimization(segment->resized_img, observed_features, i, &current_result); 

        if (current_result.final_loss < min_feature_loss) {
            min_feature_loss = current_result.final_loss;
            best_match_index = i;
            memcpy(segment->estimated_alpha, current_result.estimated_alpha, sizeof(double) * NUM_DEFORMATIONS);
        }
    }
    
    segment->best_match_index = best_match_index;
    segment->final_loss = min_feature_loss;
}


// --- PNG Rendering for Segmentation Output ---

#define SEG_ROW_HEIGHT (IMG_SIZE + TEXT_HEIGHT + SET_SPACING) 
#define SEG_PNG_WIDTH (IMG_SIZE * 2 + IMG_SPACING * 3 + SET_SPACING * 2) // Simplified width

void draw_segment_info_box(unsigned char *buffer, int buf_width, int buf_height, int x, int y, int width, int height, const SegmentResult *seg) {
    char info[100];
    const char* char_name = (seg->best_match_index != -1) ? CHAR_NAMES[seg->best_match_index] : "N/A";
    
    sprintf(info, "Match: %s | Loss: %.2f | a1:%.2f", char_name, seg->final_loss, seg->estimated_alpha[0]);
    draw_text_placeholder_box(buffer, buf_width, buf_height, x, y, width, height, 200, 200, 255);
}

void render_segment_to_png(unsigned char *buffer, int buf_width, int buf_height, const SegmentResult *seg, int x_set, int y_set) {
    int current_x = x_set;
    
    // 1. Resized Segment Image (Observed)
    render_single_image_to_png(buffer, buf_width, buf_height, seg->resized_img, current_x, y_set + TEXT_HEIGHT, 0); 
    current_x += IMG_SIZE + IMG_SPACING;
    
    // 2. Best Estimated Image
    Generated_Image estimated_img;
    if (seg->best_match_index != -1) {
        draw_curve(seg->estimated_alpha, estimated_img, &IDEAL_TEMPLATES[seg->best_match_index]);
    } else {
        // Fallback to clear image
        for(int i=0; i<GRID_SIZE; i++) for(int j=0; j<GRID_SIZE; j++) estimated_img[i][j] = 0.0;
    }
    render_single_image_to_png(buffer, buf_width, buf_height, estimated_img, current_x, y_set + TEXT_HEIGHT, 0);
    current_x += IMG_SIZE + IMG_SPACING;

    // 3. Info Box Placeholder
    draw_segment_info_box(buffer, buf_width, buf_height, x_set, y_set + 2, IMG_SIZE * 2 + IMG_SPACING, TEXT_HEIGHT - 4, seg);
}

void generate_segment_png(const SegmentResult *segments, int num_segments, const double *full_data, int full_width, int full_height) {
    if (num_segments == 0) {
        printf("\nWARNING: No segments found. Skipping segment PNG generation.\n");
        return;
    }
    
    const int MAX_SEGMENTS_PER_ROW = 5;
    const int SEGMENTS_PER_ROW = fmin(MAX_SEGMENTS_PER_ROW, num_segments);
    
    int full_img_row_height = (full_height * PIXEL_SIZE) + TEXT_HEIGHT + SET_SPACING;
    int seg_row_height = SEG_ROW_HEIGHT;
    int num_seg_rows = (num_segments + SEGMENTS_PER_ROW - 1) / SEGMENTS_PER_ROW; 
    
    int png_height = full_img_row_height + num_seg_rows * seg_row_height + SET_SPACING;
    int png_width = (IMG_SIZE * 2 + IMG_SPACING) * SEGMENTS_PER_ROW + SET_SPACING * 2;
    
    long buffer_size = (long)png_width * png_height * NUM_CHANNELS;
    unsigned char *buffer = (unsigned char *)safe_malloc(buffer_size);
    if (buffer == NULL) {
        printf("\nERROR: Failed to allocate buffer for segment PNG.\n");
        return;
    }
    memset(buffer, 0, buffer_size);

    int x_set = SET_SPACING;
    int y_set = SET_SPACING;

    // 1. Draw Full Image Title and Image
    draw_text_placeholder_box(buffer, png_width, png_height, x_set, y_set + 2, full_width * PIXEL_SIZE, TEXT_HEIGHT - 4, 150, 150, 255);
    
    unsigned char r, g, b;
    for (int i = 0; i < full_height; i++) {
        for (int j = 0; j < full_width; j++) {
            get_pixel_color(full_data[i * full_width + j], 0, &r, &g, &b);
            for (int py = 0; py < PIXEL_SIZE; py++) {
                for (int px = 0; px < PIXEL_SIZE; px++) {
                    set_pixel(buffer, 
                              x_set + j * PIXEL_SIZE + px, 
                              y_set + TEXT_HEIGHT + i * PIXEL_SIZE + py, 
                              png_width, png_height, r, g, b);
                }
            }
        }
    }
    
    // 2. Draw Segment Boundaries on the Full Image
    for (int k = 0; k < num_segments; k++) {
        const SegmentResult *seg = &segments[k];
        int y_offset = y_set + TEXT_HEIGHT; // Offset for image start

        for (int y = seg->y_start * PIXEL_SIZE; y <= seg->y_end * PIXEL_SIZE + PIXEL_SIZE; y++) {
            set_pixel(buffer, x_set + seg->x_start * PIXEL_SIZE, y_offset + y, png_width, png_height, 0, 0, 255);
            set_pixel(buffer, x_set + seg->x_end * PIXEL_SIZE, y_offset + y, png_width, png_height, 0, 0, 255);
        }
        for (int x = seg->x_start * PIXEL_SIZE; x <= seg->x_end * PIXEL_SIZE + PIXEL_SIZE; x++) {
            set_pixel(buffer, x_set + x, y_offset + seg->y_start * PIXEL_SIZE, png_width, png_height, 0, 0, 255);
            set_pixel(buffer, x_set + x, y_offset + seg->y_end * PIXEL_SIZE, png_width, png_height, 0, 0, 255);
        }
    }
    
    y_set += full_img_row_height;

    // 3. Draw Letter Segments
    for (int k = 0; k < num_segments; k++) {
        int row_index = k / SEGMENTS_PER_ROW;
        int col_index = k % SEGMENTS_PER_ROW;
        
        int seg_x = SET_SPACING + col_index * (IMG_SIZE * 2 + IMG_SPACING);
        int seg_y = y_set + row_index * seg_row_height;
        
        render_segment_to_png(buffer, png_width, png_height, &segments[k], seg_x, seg_y);
    }

    int success = stbi_write_png("segmentation_output.png", png_width, png_height, NUM_CHANNELS, buffer, png_width * NUM_CHANNELS);

    safe_free(buffer, buffer_size);

    if (success) {
        printf("\nSegmentation Output Complete: segmentation_output.png created.\n");
    } else {
        printf("\nERROR: Failed to write segmentation_output.png.\n");
    }
}


// --- Main Execution ---

int main(void) {
    srand(42); 

    const char *input_filename = "test1.jpg";
    double *full_image_data = NULL;
    int full_width = 0;
    int full_height = 0;
    
    // Load Image (Uses Placeholder)
    if (!load_image_stb(input_filename, &full_image_data, &full_width, &full_height) || full_image_data == NULL) {
        fprintf(stderr, "Fatal Error: Failed to load image or memory allocation failed.\n");
        return 1;
    }
    
    size_t segments_size = sizeof(SegmentResult) * MAX_SEGMENTS;
    SegmentResult *segments = (SegmentResult *)safe_malloc(segments_size);
    if (segments == NULL) {
        fprintf(stderr, "Fatal Error: Failed to allocate memory for segments.\n");
        safe_free(full_image_data, sizeof(double) * full_width * full_height);
        return 1;
    }
    
    // 1. Segment Image
    int num_segments = segment_image_naive(full_image_data, full_width, full_height, segments);

    // 2. Recognize Each Segment
    printf("\nStarting recognition for %d segments...\n", num_segments);
    for (int i = 0; i < num_segments; i++) {
        recognize_segment(&segments[i]);
        const char* char_name = (segments[i].best_match_index != -1) ? CHAR_NAMES[segments[i].best_match_index] : "N/A";
        printf("  Segment %d: Match='%s' (Loss: %.4f, Bounds: X:[%d,%d] Y:[%d,%d])\n", 
               i + 1, char_name, segments[i].final_loss, 
               segments[i].x_start, segments[i].x_end, segments[i].y_start, segments[i].y_end);
    }
    
    // 3. Generate PNG Output
    generate_segment_png(segments, num_segments, full_image_data, full_width, full_height);

    // Free resources
    safe_free(segments, segments_size);
    safe_free(full_image_data, sizeof(double) * full_width * full_height);

    printf("\nFinal Memory Check: Allocated: %zu bytes | Freed: %zu bytes | Net: %zu bytes\n", 
           total_allocated_bytes, total_freed_bytes, total_allocated_bytes - total_freed_bytes);

    return 0;
}
