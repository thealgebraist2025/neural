#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>

// --- Configuration Constants ---
#define IMAGE_WIDTH 64
#define IMAGE_HEIGHT 64
#define MAX_SITES 4096 // Maximum number of 1-pixels (64*64)
#define NUM_TEST_SITES 64 // Number of random sites to generate for the demo
#define EPSILON 1e-6 // Tolerance for floating point comparisons

// --- Data Structures ---

/**
 * @struct Site
 * Represents a 2D point (x, y) that will be used for the Voronoi calculation.
 */
typedef struct {
    double x;
    double y;
} Site;

/**
 * @struct Triangle
 * Stores the indices of the three sites forming a Delaunay triangle, 
 * along with its calculated circumcenter (which is a Voronoi vertex).
 */
typedef struct {
    int i1, i2, i3;
    Site circumcenter;
} Triangle;


// --- Geometric Utilities (The Core) ---

/**
 * @brief Checks if three sites are nearly collinear (form a very flat triangle).
 * @param p1, p2, p3 The three sites.
 * @return true if the points are collinear, false otherwise.
 */
bool is_collinear(const Site p1, const Site p2, const Site p3) {
    // Cross product (p2-p1) x (p3-p1) = (x2-x1)(y3-y1) - (y2-y1)(x3-x1)
    const double val = (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x);
    return fabs(val) < EPSILON;
}

/**
 * @brief Calculates the circumcenter of the triangle defined by p1, p2, and p3.
 * The circumcenter is a Voronoi vertex.
 * @param p1, p2, p3 The three sites.
 * @param center Pointer to the Site struct where the result will be stored.
 * @return true if the circumcenter was successfully calculated, false if collinear (degenerate).
 */
bool get_circumcenter(const Site p1, const Site p2, const Site p3, Site *center) {
    if (is_collinear(p1, p2, p3)) {
        return false; // Cannot form a valid circumcircle
    }

    // Coordinates of the sites
    const double x1 = p1.x, y1 = p1.y;
    const double x2 = p2.x, y2 = p2.y;
    const double x3 = p3.x, y3 = p3.y;

    // Denominator D = 2 * (x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2))
    const double D = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2));

    // Calculate center coordinates (ux, uy)
    const double s1 = x1 * x1 + y1 * y1;
    const double s2 = x2 * x2 + y2 * y2;
    const double s3 = x3 * x3 + y3 * y3;

    center->x = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / D;
    center->y = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / D;

    return true;
}

/**
 * @brief Checks if a test site p4 is strictly inside the circumcircle of p1, p2, p3.
 * This is the core check for the naive O(N^4) Delaunay algorithm.
 * @param p1, p2, p3 The three sites defining the circle.
 * @param p4 The site to test.
 * @return true if p4 is inside the circle, false otherwise (outside or on the boundary).
 */
bool is_in_circumcircle(const Site p1, const Site p2, const Site p3, const Site p4) {
    // Uses the determinant method (InCircle test).
    // The sign of the determinant indicates whether p4 is inside, outside, or on the circle.

    const double A = p1.x - p4.x;
    const double B = p1.y - p4.y;
    const double C = A * A + B * B;

    const double D = p2.x - p4.x;
    const double E = p2.y - p4.y;
    const double F = D * D + E * E;

    const double G = p3.x - p4.x;
    const double H = p3.y - p4.y;
    const double I = G * G + H * H;

    // Determinant calculation
    const double det = A * (E * I - F * H) - B * (D * I - F * G) + C * (D * H - E * G);

    // If the triangle (p1, p2, p3) is counter-clockwise (CCW), det > 0 means inside.
    // If the determinant is near zero, the point is cocircular.
    return det > EPSILON;
}


// --- Unit Tests ---

/**
 * @brief Runs unit tests for the core geometric functions.
 */
void run_unit_tests() {
    printf("--- Running Unit Tests ---\n");
    int failures = 0;

    // Test 1: Collinearity
    const Site c1 = {1.0, 1.0};
    const Site c2 = {2.0, 2.0};
    const Site c3 = {3.0, 3.0};
    if (!is_collinear(c1, c2, c3)) {
        printf("[FAIL] Test 1: is_collinear failed for (1,1), (2,2), (3,3).\n");
        failures++;
    } else {
        printf("[PASS] Test 1: Collinear check.\n");
    }

    // Test 2: Circumcenter (Right Triangle)
    const Site t1 = {0.0, 0.0};
    const Site t2 = {10.0, 0.0};
    const Site t3 = {0.0, 10.0};
    Site center;
    if (get_circumcenter(t1, t2, t3, &center) && fabs(center.x - 5.0) < EPSILON && fabs(center.y - 5.0) < EPSILON) {
        printf("[PASS] Test 2: Circumcenter for right triangle.\n");
    } else {
        printf("[FAIL] Test 2: Circumcenter failed. Expected (5, 5), got (%.2f, %.2f).\n", center.x, center.y);
        failures++;
    }

    // Test 3: InCircumcircle (Inside)
    const Site p_inside = {5.0, 1.0};
    if (is_in_circumcircle(t1, t2, t3, p_inside)) {
        printf("[PASS] Test 3: InCircumcircle (inside check).\n");
    } else {
        printf("[FAIL] Test 3: InCircumcircle failed. Point (5, 1) should be inside.\n");
        failures++;
    }

    // Test 4: InCircumcircle (Outside)
    const Site p_outside = {11.0, 1.0};
    if (!is_in_circumcircle(t1, t2, t3, p_outside)) {
        printf("[PASS] Test 4: InCircumcircle (outside check).\n");
    } else {
        printf("[FAIL] Test 4: InCircumcircle failed. Point (11, 1) should be outside.\n");
        failures++;
    }

    // Test 5: Circumcenter (Collinear failure)
    Site center_fail;
    if (!get_circumcenter(c1, c2, c3, &center_fail)) {
        printf("[PASS] Test 5: Circumcenter failed check (collinear).\n");
    } else {
        printf("[FAIL] Test 5: Circumcenter failed check (collinear) - it calculated a center.\n");
        failures++;
    }

    printf("--- Tests Complete: %d failure(s) ---\n\n", failures);
}


// --- Main Program Logic ---

/**
 * @brief Generates NUM_TEST_SITES random sites within the 64x64 grid area.
 * @param sites Array to store the generated sites.
 * @return The number of sites generated (NUM_TEST_SITES).
 */
int generate_sites(Site sites[]) {
    srand(time(NULL));
    for (int i = 0; i < NUM_TEST_SITES; ++i) {
        // Generate coordinates (0.0 to 63.999...)
        sites[i].x = (double)rand() / RAND_MAX * IMAGE_WIDTH;
        sites[i].y = (double)rand() / RAND_MAX * IMAGE_HEIGHT;
    }
    printf("Generated %d random sites for the demo.\n", NUM_TEST_SITES);
    return NUM_TEST_SITES;
}

/**
 * @brief Implements the naive O(N^4) Delaunay Triangulation algorithm.
 * It iterates over every triplet of sites and checks the empty circumcircle property
 * against every other site.
 * @param sites Array of sites.
 * @param count Number of sites.
 * @param triangles Array to store found Delaunay triangles.
 * @return The number of Delaunay triangles found.
 */
int find_delaunay_triangulation(const Site sites[], const int count, Triangle triangles[]) {
    int triangle_count = 0;

    // O(N^3) loop for selecting the three points (i, j, k)
    for (int i = 0; i < count; ++i) {
        for (int j = i + 1; j < count; ++j) {
            for (int k = j + 1; k < count; ++k) {
                
                // Check for degeneracy
                if (is_collinear(sites[i], sites[j], sites[k])) {
                    continue;
                }

                bool is_delaunay = true;

                // O(N) loop for checking the Empty Circumcircle property against point 'l'
                for (int l = 0; l < count; ++l) {
                    if (l != i && l != j && l != k) {
                        if (is_in_circumcircle(sites[i], sites[j], sites[k], sites[l])) {
                            is_delaunay = false;
                            break; // Not a Delaunay triangle
                        }
                    }
                }

                if (is_delaunay) {
                    Site center;
                    // Calculate the circumcenter (Voronoi vertex) and store the triangle
                    if (get_circumcenter(sites[i], sites[j], sites[k], &center)) {
                        triangles[triangle_count].i1 = i;
                        triangles[triangle_count].i2 = j;
                        triangles[triangle_count].i3 = k;
                        triangles[triangle_count].circumcenter = center;
                        triangle_count++;
                    }
                }
            }
        }
    }
    printf("Delaunay Triangulation complete. Found %d triangles.\n", triangle_count);
    return triangle_count;
}

/**
 * @brief Saves the Delaunay triangles and Voronoi vertices to an SVG file.
 * The SVG output contains the Delaunay edges (triangles) and the Voronoi vertices (circumcenters).
 * @param sites Array of sites.
 * @param site_count Number of sites.
 * @param triangles Array of Delaunay triangles.
 * @param triangle_count Number of triangles.
 */
void save_svg(const Site sites[], const int site_count, const Triangle triangles[], const int triangle_count) {
    FILE *fp = fopen("voronoi.svg", "w");
    if (fp == NULL) {
        perror("Error opening voronoi.svg");
        return;
    }

    // SVG Header
    fprintf(fp, "<svg width=\"%d\" height=\"%d\" viewBox=\"0 0 %d %d\" xmlns=\"http://www.w3.org/2000/svg\">\n",
            IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT);
    fprintf(fp, "<rect width=\"100%%\" height=\"100%%\" fill=\"#f5f5f5\"/>\n");

    // 1. Draw Delaunay Triangles (The requested "triangles" visualization)
    fprintf(fp, "<!-- Delaunay Triangles -->\n");
    for (int i = 0; i < triangle_count; ++i) {
        const Site p1 = sites[triangles[i].i1];
        const Site p2 = sites[triangles[i].i2];
        const Site p3 = sites[triangles[i].i3];

        fprintf(fp, "<polygon points=\"%.2f,%.2f %.2f,%.2f %.2f,%.2f\" ",
                p1.x, p1.y, p2.x, p2.y, p3.x, p3.y);
        fprintf(fp, "fill=\"none\" stroke=\"#4f46e5\" stroke-width=\"0.1\" opacity=\"0.5\"/>\n");
    }

    // 2. Draw Voronoi Vertices (Circumcenters)
    fprintf(fp, "<!-- Voronoi Vertices (Circumcenters) -->\n");
    for (int i = 0; i < triangle_count; ++i) {
        const Site c = triangles[i].circumcenter;
        fprintf(fp, "<circle cx=\"%.2f\" cy=\"%.2f\" r=\"0.4\" fill=\"#059669\"/>\n", c.x, c.y);
    }
    
    // 3. Draw Sites
    fprintf(fp, "<!-- Input Sites -->\n");
    for (int i = 0; i < site_count; ++i) {
        fprintf(fp, "<circle cx=\"%.2f\" cy=\"%.2f\" r=\"0.3\" fill=\"#ef4444\"/>\n", sites[i].x, sites[i].y);
    }

    // SVG Footer
    fprintf(fp, "</svg>\n");
    fclose(fp);

    printf("Output saved to voronoi.svg (Image Size: %dx%d)\n", IMAGE_WIDTH, IMAGE_HEIGHT);
    printf("SVG contains: Delaunay Triangles, Voronoi Vertices, and Input Sites.\n");
}

/**
 * @brief Main function to run the program.
 */
int main() {
    // 1. Run unit tests
    run_unit_tests();

    // 2. Mock image/site generation (as per request for 64 sites)
    Site sites[NUM_TEST_SITES];
    const int site_count = generate_sites(sites);

    // 3. Find Delaunay Triangulation (O(N^4) naive algorithm)
    // The maximum number of triangles is 2N - 2 - boundary edges, safely allocate space.
    Triangle triangles[2 * NUM_TEST_SITES]; 
    const int triangle_count = find_delaunay_triangulation(sites, site_count, triangles);

    // 4. Save results to SVG
    save_svg(sites, site_count, triangles, triangle_count);

    return 0;
}