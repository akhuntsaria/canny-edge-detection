#include "kernels.h"

#define M_PI 3.141592f

#define STRONG_EDGE 255
#define WEAK_EDGE 128
#define NO_EDGE 0

__global__ void gaussianBlur(float* kernel, unsigned char* source, unsigned char* target, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x,
        y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int totalSize = width * height;
    float weightedSum = 0;
    for (int i = -2;i <= 2;i++) {
        for (int j = -2;j <= 2;j++) {
            int idx = (y + j) * width + x + i;
            if (idx >= 0 && idx < totalSize) {
                // Get the flat index + move indices from [-2,2] to [0,4] for the kernel
                int kernelIdx = (i + 2) * 5 + (j + 2);

                weightedSum += (int)source[idx] * kernel[kernelIdx];
            }
        }
    }

    target[y * width + x] = weightedSum;
}

__global__ void grayscale(unsigned char* rgbData, unsigned char* grayData, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int avg = 0;
    for (int ch = 0;ch < 3;ch++) { // RGB channels
        int idx = y * width * 3 + x * 3 + ch;
        avg += (int)rgbData[idx];
    }
    avg /= 3;

    grayData[y * width + x] = avg;
}

__global__ void hysteresis(unsigned char* img, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int totalSize = width * height;
    int idx = y * width + x;
    if (img[idx] == STRONG_EDGE) {
        img[idx] = STRONG_EDGE;  // Strong edge is retained
    }
    else if (img[idx] >= WEAK_EDGE) {
        // Check if it is connected to any strong edge
        bool connected = false;
        for (int i = -1; i <= 1 && !connected; i++) {
            for (int j = -1; j <= 1; j++) {
                int neighborIdx = (y + j) * width + x + i;
                if (connected >= 0 && neighborIdx < totalSize && img[neighborIdx] == STRONG_EDGE) {
                    connected = true;
                    break;
                }
            }
        }
        img[idx] = connected ? STRONG_EDGE : NO_EDGE;
    }
    else {
        img[idx] = NO_EDGE;  // Suppress weak edges not connected to strong ones
    }
}

__global__ void intensityGradient(unsigned char* img, int width, int height, float* magnitudes, float* directions) {
    int x = blockIdx.x * blockDim.x + threadIdx.x,
        y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int kernelX[3][3] = {
        {1, 0, -1},
        {2, 0, -2},
        {1, 0, -1},
    }, kernelY[3][3] = {
        { 1,  2,  1},
        { 0,  0,  0},
        {-1, -2, -1},
    };

    int totalSize = width * height;
    int Gx = 0,
        Gy = 0;
    for (int i = -1;i <= 1;i++) {
        for (int j = -1;j <= 1;j++) {
            int idx = (y + j) * width + x + i;
            if (idx >= 0 && idx < totalSize) {
                // Move indices from [-1,1] to [0,2] for the kernel
                Gx += (int)img[idx] * kernelX[i + 1][j + 1];
                Gy += (int)img[idx] * kernelY[i + 1][j + 1];
            }
        }
    }

    int idx = y * width + x;

    float magnitude = sqrt((float)Gx * Gx + Gy * Gy);
    magnitudes[idx] = magnitude;

    float direction = atan2f(Gy, Gx);
    direction = direction * 180.0f / M_PI; // Convert to degrees
    if (direction < 0.0f) {
        direction += 180.0f;
    }
    directions[idx] = direction;
}

__global__ void nonMaximumSuppression(unsigned char* img,
                                      int width,
                                      int height,
                                      float* directions,
                                      float* magnitudes,
                                      float lowThreshold,
                                      float highThreshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x,
        y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // Non-maximum suppression, compare with neighboring pixels
    float neighbor1,
        neighbor2,
        direction = directions[idx];
    if (direction < 22.5f || direction >= 157.5f) { // North-South
        neighbor1 = y > 0 ? magnitudes[(y - 1) * width + x] : 0.0f;
        neighbor2 = y < height - 1 ? magnitudes[(y + 1) * width + x] : 0.0f;
    }
    else if (direction < 67.5f) { // North-East to South-West
        neighbor1 = x > 0 && y > 0 ? magnitudes[(y - 1) * width + x - 1] : 0.0f;
        neighbor2 = x < width - 1 && y < height - 1 ? magnitudes[(y + 1) * width + x + 1] : 0.0f;
    }
    else if (direction < 112.5f) { // East-West
        neighbor1 = x > 0 ? magnitudes[y * width + x - 1] : 0.0f;
        neighbor2 = x < width - 1 ? magnitudes[y * width + x + 1] : 0.0f;
    }
    else { // North-West to South-East
        neighbor1 = x > 0 && y < height - 1 ? magnitudes[(y + 1) * width + x - 1] : 0.0f;
        neighbor2 = x < width - 1 && y > 0 ? magnitudes[(y - 1) * width + x + 1] : 0.0f;
    }

    // Preserve the current pixel if it's the maximum
    float currentMagnitude = magnitudes[idx];
    if (currentMagnitude > neighbor1 && currentMagnitude > neighbor2) {
        // Double threshold
        if (currentMagnitude > highThreshold) {
            img[idx] = STRONG_EDGE;
        }
        else if (currentMagnitude > lowThreshold) {
            img[idx] = WEAK_EDGE;
        }
        else {
            img[idx] = NO_EDGE; // Suppressed pixel
        }
    }
    else {
        img[idx] = 0;
    }
}
