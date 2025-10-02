/**
 * @file github_commit_manual.c
 * @brief Commits an embedded SVG string to a GitHub repository using C99 by manually
 * constructing the required data and HTTPS request, implemented using OpenSSL
 * and POSIX Sockets for the networking stubs.
 *
 * NOTE: This program REQUIRES linking with -lssl, -lcrypto, and (on some systems) -lws2_32.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// --- NETWORK LIBRARIES REQUIRED FOR IMPLEMENTATION ---
#include <openssl/ssl.h>
#include <openssl/err.h>
#include <openssl/bio.h>

// POSIX Sockets headers
#ifdef _WIN32
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "ws2_32.lib")
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <netdb.h> // <-- FIX: Added to define struct addrinfo and getaddrinfo/freeaddrinfo
    #include <unistd.h> // for close()
#endif

// --- GITHUB CONFIGURATION ---
#define GITHUB_TOKEN "YOUR_GITHUB_PERSONAL_ACCESS_TOKEN"
#define REPO_OWNER "your-repo-owner"
#define REPO_NAME "your-repository-name"
#define FILE_PATH "images/embedded_c99_logo.svg" // Target path in GitHub repo
#define COMMIT_MESSAGE "Add embedded SVG via manual C99 HTTP request"
#define BRANCH_NAME "main"
#define GITHUB_HOST "api.github.com"
#define GITHUB_PORT 443 // HTTPS standard port

// --- EMBEDDED SVG TEST STRING ---
#define SVG_TEST_CONTENT "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"120\" height=\"60\"><rect width=\"120\" height=\"60\" fill=\"#3b82f6\" rx=\"10\"/><text x=\"60\" y=\"38\" font-family=\"Inter, sans-serif\" font-size=\"18\" text-anchor=\"middle\" fill=\"#ffffff\">C99 SVG</text></svg>"

// --- PURE C99: BASE64 ENCODING ---

static const char b64_table[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

char *base64_encode(const unsigned char *data, size_t input_length, size_t *output_length) {
    *output_length = 4 * ((input_length + 2) / 3);
    char *encoded_data = (char *)malloc(*output_length + 1);
    if (!encoded_data) return NULL;

    size_t i = 0, j = 0;
    for (i = 0; i < input_length;) {
        unsigned int octet_a = i < input_length ? (unsigned int)data[i++] : 0;
        unsigned int octet_b = i < input_length ? (unsigned int)data[i++] : 0;
        unsigned int octet_c = i < input_length ? (unsigned int)data[i++] : 0;

        unsigned int triple = (octet_a << 16) + (octet_b << 8) + octet_c;

        encoded_data[j++] = b64_table[(triple >> 18) & 0x3F];
        encoded_data[j++] = b64_table[(triple >> 12) & 0x3F];
        encoded_data[j++] = b64_table[(triple >> 6) & 0x3F];
        encoded_data[j++] = b64_table[(triple >> 0) & 0x3F];
    }

    int mod = input_length % 3;
    if (mod > 0) {
        if (mod == 1) {
            encoded_data[*output_length - 2] = '=';
        }
        encoded_data[*output_length - 1] = '=';
    }
    encoded_data[*output_length] = '\0';
    return encoded_data;
}

// --- PURE C99: JSON PAYLOAD GENERATION ---

char *generate_json_payload(const char *content_b64) {
    size_t content_len = strlen(content_b64);
    size_t fixed_overhead = 512;
    size_t buffer_size = content_len + fixed_overhead;
    char *payload = (char *)malloc(buffer_size);

    if (!payload) return NULL;

    snprintf(payload, buffer_size,
             "{\"message\":\"%s\",\"content\":\"%s\",\"branch\":\"%s\"}",
             COMMIT_MESSAGE, content_b64, BRANCH_NAME);

    return payload;
}

// --- PURE C99: HTTP REQUEST ASSEMBLY ---

char *generate_http_request(const char *json_payload) {
    const char *path_format = "/repos/%s/%s/contents/%s";
    char request_path[256];
    snprintf(request_path, sizeof(request_path), path_format, REPO_OWNER, REPO_NAME, FILE_PATH);

    const char *template_format =
        "PUT %s HTTP/1.1\r\n"
        "Host: %s\r\n"
        "Authorization: Bearer %s\r\n"
        "User-Agent: C99-Manual-Client\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %zu\r\n"
        "Accept: application/vnd.github.v3+json\r\n"
        "Connection: close\r\n"
        "\r\n"
        "%s"; // Body

    size_t payload_len = strlen(json_payload);
    size_t total_size = 1024 + payload_len; // Conservative estimate

    char *http_request = (char *)malloc(total_size);
    if (!http_request) return NULL;

    snprintf(http_request, total_size, template_format,
             request_path,
             GITHUB_HOST,
             GITHUB_TOKEN,
             payload_len,
             json_payload);

    return http_request;
}


// --- IMPLEMENTED NETWORK FUNCTIONS (POSIX SOCKETS + OPENSSL) ---

#ifdef _WIN32
    #define CLOSE_SOCKET(s) closesocket(s)
#else
    #define CLOSE_SOCKET(s) close(s)
#endif


/**
 * @brief IMPLEMENTED: Connects a TCP socket to the specified host and port.
 * @return The socket file descriptor on success, -1 on failure.
 */
int tcp_connect(const char *host, int port) {
    int sockfd = -1;
    struct addrinfo hints, *servinfo, *p;
    char port_str[6];
    snprintf(port_str, sizeof(port_str), "%d", port);

#ifdef _WIN32
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        fprintf(stderr, "tcp_connect error: WSAStartup failed.\n");
        return -1;
    }
#endif

    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    if (getaddrinfo(host, port_str, &hints, &servinfo) != 0) {
        fprintf(stderr, "tcp_connect error: getaddrinfo failed for %s. %s\n", host, gai_strerror(getaddrinfo(host, port_str, &hints, &servinfo)));
        return -1;
    }

    for (p = servinfo; p != NULL; p = p->ai_next) {
        if ((sockfd = socket(p->ai_family, p->ai_socktype, p->ai_protocol)) == -1) {
            continue;
        }
        if (connect(sockfd, p->ai_addr, p->ai_addrlen) == -1) {
            CLOSE_SOCKET(sockfd);
            sockfd = -1;
            continue;
        }
        break; // Successfully connected
    }

    freeaddrinfo(servinfo);

    if (sockfd == -1) {
        fprintf(stderr, "tcp_connect error: Failed to connect to %s:%d.\n", host, port);
    } else {
        printf("NETWORK SUCCESS: TCP connection established to %s:%d (FD: %d).\n", host, port, sockfd);
    }

    return sockfd;
}

/**
 * @brief IMPLEMENTED: Wraps the established TCP socket with TLS/SSL using OpenSSL.
 * @return A pointer to the SSL session object on success, NULL on failure.
 */
void *ssl_connect(int socket_fd) {
    SSL_CTX *ctx = NULL;
    SSL *ssl = NULL;

    // 1. Initialize OpenSSL (once per program)
    SSL_library_init();
    SSL_load_error_strings();
    OpenSSL_add_all_algorithms();

    // 2. Create SSL Context
    ctx = SSL_CTX_new(TLS_client_method());
    if (!ctx) {
        fprintf(stderr, "ssl_connect error: Failed to create SSL context.\n");
        ERR_print_errors_fp(stderr);
        return NULL;
    }

    // 3. Create SSL connection state
    ssl = SSL_new(ctx);
    if (!ssl) {
        fprintf(stderr, "ssl_connect error: Failed to create SSL object.\n");
        SSL_CTX_free(ctx);
        return NULL;
    }

    // 4. Associate the socket with the SSL structure
    if (!SSL_set_fd(ssl, socket_fd)) {
        fprintf(stderr, "ssl_connect error: Failed to set socket FD.\n");
        SSL_free(ssl);
        SSL_CTX_free(ctx);
        return NULL;
    }

    // 5. Perform the TLS handshake
    if (SSL_connect(ssl) <= 0) {
        fprintf(stderr, "ssl_connect error: TLS handshake failed.\n");
        ERR_print_errors_fp(stderr);
        SSL_free(ssl);
        SSL_CTX_free(ctx);
        return NULL;
    }

    // NOTE: In a clean program, SSL_CTX_free is only called after all SSL objects
    // associated with it are freed. We rely on the implicit linkage here.
    printf("NETWORK SUCCESS: TLS/SSL handshake completed.\n");
    // We don't free ctx here, as SSL_free implicitly decrements its reference count.
    return (void *)ssl;
}

/**
 * @brief IMPLEMENTED: Sends data securely over the established TLS session.
 * @return The number of bytes sent on success, -1 on error.
 */
int ssl_send(void *ssl_session, const char *buffer, size_t len) {
    SSL *ssl = (SSL *)ssl_session;
    int bytes_sent = SSL_write(ssl, buffer, (int)len);

    if (bytes_sent <= 0) {
        fprintf(stderr, "ssl_send error: Failed to write data.\n");
        ERR_print_errors_fp(stderr);
        return -1;
    }

    return bytes_sent;
}

/**
 * @brief IMPLEMENTED: Receives data securely over the established TLS session.
 * @return The number of bytes received on success, 0 on connection close, -1 on error.
 */
int ssl_recv(void *ssl_session, char *buffer, size_t len) {
    SSL *ssl = (SSL *)ssl_session;
    // Attempt a single read
    int bytes_received = SSL_read(ssl, buffer, (int)len);

    if (bytes_received < 0) {
        int err = SSL_get_error(ssl, bytes_received);
        if (err == SSL_ERROR_ZERO_RETURN) {
            // Connection closed gracefully
            return 0;
        }
        fprintf(stderr, "ssl_recv error: Failed to read data (SSL Error: %d).\n", err);
        ERR_print_errors_fp(stderr);
        return -1;
    }

    return bytes_received;
}

/**
 * @brief IMPLEMENTED: Disconnects and frees the TLS/SSL session.
 * @return 0 on success, -1 on error.
 */
int ssl_disconnect(void *ssl_session) {
    SSL *ssl = (SSL *)ssl_session;
    if (ssl) {
        // Initiate the close-notify alert
        SSL_shutdown(ssl);
        SSL_free(ssl);
        printf("NETWORK CLEANUP: TLS/SSL session disconnected.\n");
        return 0;
    }
    return -1;
}

/**
 * @brief IMPLEMENTED: Closes the TCP socket connection.
 * @return 0 on success, -1 on error.
 */
int tcp_close(int socket_fd) {
    if (socket_fd >= 0) {
        CLOSE_SOCKET(socket_fd);
        printf("NETWORK CLEANUP: TCP socket (FD: %d) closed.\n", socket_fd);

#ifdef _WIN32
        WSACleanup();
#endif
        return 0;
    }
    return -1;
}

// --- MAIN EXECUTION LOGIC ---

int main(void) {
    char *encoded_svg_content = NULL;
    char *json_payload = NULL;
    char *http_request = NULL;
    int socket_fd = -1;
    void *ssl_session = NULL;
    int return_code = EXIT_FAILURE;

    // Data is read directly from the in-memory constant string
    const unsigned char *raw_svg_data = (const unsigned char *)SVG_TEST_CONTENT;
    long raw_svg_size = (long)strlen(SVG_TEST_CONTENT);

    printf("--- GitHub File Committer (C99 Data Preparation) ---\n");

    // Check for placeholder token
    if (strcmp(GITHUB_TOKEN, "YOUR_GITHUB_PERSONAL_ACCESS_TOKEN") == 0) {
         fprintf(stderr, "CRITICAL: Please replace GITHUB_TOKEN in the source code with a valid Personal Access Token.\n");
         goto cleanup;
    }

    // 1. Base64 Encode In-Memory Data
    encoded_svg_content = base64_encode(raw_svg_data, raw_svg_size, &(size_t){0});
    if (!encoded_svg_content) {
        fprintf(stderr, "Error: Base64 encoding failed (Memory allocation error).\n");
        goto cleanup;
    }
    printf("1. SVG data read (Length: %ld bytes) and Base64 encoded successfully.\n", raw_svg_size);

    // 2. Generate JSON Payload
    json_payload = generate_json_payload(encoded_svg_content);
    if (!json_payload) {
        fprintf(stderr, "Error: JSON payload generation failed (Memory allocation error).\n");
        goto cleanup;
    }
    printf("2. JSON payload generated (Length: %zu bytes).\n", strlen(json_payload));

    // 3. Generate Full HTTP Request
    http_request = generate_http_request(json_payload);
    if (!http_request) {
        fprintf(stderr, "Error: HTTP request generation failed (Memory allocation error).\n");
        goto cleanup;
    }
    printf("3. Complete HTTP PUT request assembled (Length: %zu bytes).\n", strlen(http_request));
    printf("\n--- Start Network Transmission (Implemented with OpenSSL/Sockets) ---\n");

    // --- 4. NETWORK STACK (Real Implementation) ---

    // 4a. Connect TCP
    socket_fd = tcp_connect(GITHUB_HOST, GITHUB_PORT);
    if (socket_fd < 0) {
        fprintf(stderr, "CRITICAL: TCP connection failed.\n");
        goto cleanup;
    }

    // 4b. Connect SSL/TLS
    ssl_session = ssl_connect(socket_fd);
    if (!ssl_session) {
        fprintf(stderr, "CRITICAL: TLS/SSL handshake failed.\n");
        goto cleanup;
    }

    // 4c. Send Request
    size_t request_len = strlen(http_request);
    int sent_bytes = ssl_send(ssl_session, http_request, request_len);

    if (sent_bytes != (int)request_len) {
        fprintf(stderr, "CRITICAL: Failed to send full HTTP request (%d/%zu bytes sent).\n", sent_bytes, request_len);
        goto cleanup;
    }
    printf("4. Request successfully sent.\n");

    // 4d. Receive Response
    char response_buffer[4096];
    memset(response_buffer, 0, sizeof(response_buffer));
    // We might need to loop here for a full response in a real client, but for simplicity, we try a single read.
    int bytes_received = ssl_recv(ssl_session, response_buffer, sizeof(response_buffer) - 1);

    if (bytes_received > 0) {
        // Look for the HTTP status line to check for success (201 Created)
        if (strncmp(response_buffer, "HTTP/1.1 201 Created", 18) == 0) {
            printf("5. HTTP Commit SUCCESS: Received 201 Created response.\n");
            return_code = EXIT_SUCCESS;
        } else {
             // Extract and report the actual status code
             char status_line[128] = {0};
             // Find the end of the first line (the status line)
             char *end_of_line = strstr(response_buffer, "\r\n");
             size_t len_to_copy = end_of_line ? (size_t)(end_of_line - response_buffer) : bytes_received;

             if (len_to_copy > sizeof(status_line) - 1) {
                 len_to_copy = sizeof(status_line) - 1;
             }
             strncpy(status_line, response_buffer, len_to_copy);
             status_line[len_to_copy] = '\0';

             fprintf(stderr, "CRITICAL: GitHub API request failed. Received status: %s\n", status_line);
             if (strstr(status_line, "401 Unauthorized")) {
                 fprintf(stderr, "HINT: Check if your GITHUB_TOKEN is valid and has 'repo' permissions.\n");
             }
        }
    } else if (bytes_received == 0) {
        fprintf(stderr, "CRITICAL: Connection closed by peer before full response received.\n");
    } else { // bytes_received < 0
        fprintf(stderr, "CRITICAL: Error receiving response from server.\n");
    }

// --- Cleanup ---
cleanup:
    // Network Cleanup
    if (ssl_session) {
        ssl_disconnect(ssl_session);
    }
    if (socket_fd >= 0) {
        tcp_close(socket_fd);
    }

    // Memory Cleanup
    if (encoded_svg_content) free(encoded_svg_content);
    if (json_payload) free(json_payload);
    if (http_request) free(http_request);

    printf("\n--- Program finished with exit code %d ---\n", return_code);

    return return_code;
}