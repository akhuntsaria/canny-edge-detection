#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <limits.h>
#include <opencv2/core.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>

#define M_PI 3.141592f

// Convert an index, image[channel][i][j] to flat[idx]
__device__ __host__ int getIdx(int width, int channels, int channel, int i, int j) {
    return j * width * channels + i * channels + channel;
}

__global__ void gaussianBlurKernel(float* kernel, unsigned char* source, unsigned char* target, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x,
        y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int totalSize = width * height * channels;
        for (int ch = 0;ch < channels;ch++) {
            float weightedSum = 0;
            for (int i = -2;i <= 2;i++) {
                for (int j = -2;j <= 2;j++) {
                    int idx = (y + j) * width * channels + (x + i) * channels;
                    idx += ch;
                    if (idx >= 0 && idx < totalSize) {
                        // Get the flat index + move indices from [-2,2] to [0,4] for the kernel
                        int kernelIdx = (i + 2) * 5 + (j + 2);

                        weightedSum += (int)source[idx] * kernel[kernelIdx];
                    }
                }
            }

            int centerIdx = y * width * channels + x * channels;
            target[centerIdx + ch] = weightedSum;
        }
    }
}

__global__ void grayscaleKernel(unsigned char* imageData, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int channelAvg = 0;
        for (int ch = 0;ch < channels;ch++) {
            channelAvg += (int)imageData[getIdx(width, channels, ch, x, y)];
        }
        channelAvg /= 3;
        for (int ch = 0;ch < channels;ch++) {
            imageData[getIdx(width, channels, ch, x, y)] = channelAvg;
        }
    }
}

__global__ void generateMagnitudesKernel(unsigned char* source, float* target, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x,
        y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int kernelX[3][3] = {
            {1, 0, -1},
            {2, 0, -2},
            {1, 0, -1},
        }, kernelY[3][3] = {
            { 1,  2,  1},
            { 0,  0,  0},
            {-1, -2, -1},
        };

        int totalSize = width * height * channels;
        for (int ch = 0;ch < channels;ch++) {
            int Gx = 0,
                Gy = 0;
            for (int i = -1;i <= 1;i++) {
                for (int j = -1;j <= 1;j++) {
                    int idx = getIdx(width, channels, ch, x + i, y + j);
                    if (idx >= 0 && idx < totalSize) {
                        // Move indices from [-1,1] to [0,2] for the kernel
                        Gx += (int)source[idx] * kernelX[i + 1][j + 1];
                        Gy += (int)source[idx] * kernelY[i + 1][j + 1];
                    }
                }
            }

            float magnitude = sqrt((float)Gx * Gx + Gy * Gy);
            target[getIdx(width, channels, ch, x, y)] = magnitude;
        }
    }
}

__global__ void sobelKernel(unsigned char* source, unsigned char* target, float* magnitudes, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x,
        y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int kernelX[3][3] = {
            {1, 0, -1},
            {2, 0, -2},
            {1, 0, -1},
        }, kernelY[3][3] = {
            { 1,  2,  1},
            { 0,  0,  0},
            {-1, -2, -1},
        };

        int totalSize = width * height * channels;
        for (int ch = 0;ch < channels;ch++) {
            int Gx = 0,
                Gy = 0;
            for (int i = -1;i <= 1;i++) {
                for (int j = -1;j <= 1;j++) {
                    int idx = getIdx(width, channels, ch, x + i, y + j);
                    if (idx >= 0 && idx < totalSize) {
                        //TODO don't generate twice
                        // Move indices from [-1,1] to [0,2] for the kernel
                        Gx += (int)source[idx] * kernelX[i + 1][j + 1];
                        Gy += (int)source[idx] * kernelY[i + 1][j + 1];
                    }
                }
            }

            // Intensity gradient
            float currentMagnitude = sqrtf(Gx * Gx + Gy * Gy);

            // Calculate the direction of the gradient
            float direction = atan2f(Gy, Gx);
            direction = direction * 180.0f / M_PI; // Convert to degrees
            if (direction < 0.0f) {
                direction += 180.0f;
            }

            // Non-maximum suppression
            // Compare with neighboring pixels
            float neighbor1, neighbor2;
            if (direction < 22.5f || direction >= 157.5f) { // North-South
                neighbor1 = y > 0 ? magnitudes[getIdx(width, channels, ch, x, y - 1)] : 0.0f;
                neighbor2 = y < height - 1 ? magnitudes[getIdx(width, channels, ch, x, y + 1)] : 0.0f;
            }
            else if (direction < 67.5f) { // North-East to South-West
                neighbor1 = x > 0 && y > 0 ? magnitudes[getIdx(width, channels, ch, x - 1, y - 1)] : 0.0f;
                neighbor2 = x < width - 1 && y < height - 1 ? magnitudes[getIdx(width, channels, ch, x + 1, y + 1)] : 0.0f;
            }
            else if (direction < 112.5f) { // East-West
                neighbor1 = x > 0 ? magnitudes[getIdx(width, channels, ch, x - 1, y)] : 0.0f;
                neighbor2 = x < width - 1 ? magnitudes[getIdx(width, channels, ch, x + 1, y)] : 0.0f;
            }
            else { // North-West to South-East
                neighbor1 = x > 0 && y < height - 1 ? magnitudes[getIdx(width, channels, ch, x - 1, y + 1)] : 0.0f;
                neighbor2 = x < width - 1 && y > 0 ? magnitudes[getIdx(width, channels, ch, x + 1, y - 1)] : 0.0f;
            }

            // Preserve the current pixel if it's the maximum
            int idx = getIdx(width, channels, ch, x, y);
            if (currentMagnitude > neighbor1 && currentMagnitude > neighbor2 && currentMagnitude > 100.0f) {
                // Double thresholding
                float lowThreshold = 0.1f * 255;
                float highThreshold = 0.3f * 255;
                if (currentMagnitude > highThreshold) {
                    target[idx] = 255; // Strong edge pixel
                }
                else if (currentMagnitude > lowThreshold) {
                    target[idx] = 128; // Weak edge pixel
                }
                else {
                    target[idx] = 0; // Suppressed pixel
                }
            }
            else {
                target[idx] = 0;
            }
        }
    }
}

int main(int argc, char* argv[])
{
    const char* imagePath = NULL;

    for (int i = 1; i < argc; i++) {
        char* pos = strchr(argv[i], '=');
        if (pos != NULL) {
            if (strncmp(argv[i], "image", pos - argv[i]) == 0) {
                imagePath = pos + 1;
                continue;
            }
        }
    }

    if (imagePath == NULL) {
        imagePath = "gizzard.jpg";
    }

    cv::Mat image = cv::imread(imagePath, cv::IMREAD_ANYCOLOR);

    if (image.empty()) {
        printf("Could not open or find the image\n");
        return -1;
    }

    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::imshow("Original", image);

    int width = image.cols,
        height = image.rows,
        channels = image.channels(); // Blue, green, red, etc.

    size_t imageDataSize = image.total() * image.channels();
    unsigned char* deviceImageData;
    cudaMalloc(&deviceImageData, imageDataSize);
    cudaMemcpy(deviceImageData, image.data, imageDataSize, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16, 1);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y, 1);

    printf("blockSize: (%d,%d,%d)\n", blockSize.x, blockSize.y, blockSize.z);
    printf("gridSize: (%d,%d,%d)\n", gridSize.x, gridSize.y, gridSize.z);

    printf("Applying grayscale\n");
    grayscaleKernel << <gridSize, blockSize >> > (deviceImageData, width, height, channels);

    printf("Applying Gaussian blur\n");
    float kernel[5][5]{},
        kernelSum = .0f,
        sigma = .75f;

    for (int x = -2;x <= 2;x++) {
        for (int y = -2;y <= 2;y++) {
            int i = x + 2,
                j = y + 2;
            kernel[i][j] = 1 / (2 * M_PI * sigma * sigma) * exp(-((float)x * x + y * y) / 2 * sigma * sigma);
            kernelSum += kernel[i][j];
        }
    }

    // Normalize
    for (int i = 0;i < 5;i++) {
        for (int j = 0;j < 5;j++) {
            kernel[i][j] /= kernelSum;
        }
    }

    printf("Gaussian blur kernel:\n");
    for (int i = 0;i < 5;i++) {
        for (int j = 0;j < 5;j++) {
            printf("%f ", kernel[i][j]);
        }
        printf("\n");
    }

    int kernelDataSize = 5 * 5 * sizeof(float);
    float* deviceKernel;
    cudaMalloc(&deviceKernel, kernelDataSize);
    cudaMemcpy(deviceKernel, (float*)kernel, kernelDataSize, cudaMemcpyHostToDevice);

    unsigned char* deviceImageDataCopy;
    cudaMalloc(&deviceImageDataCopy, imageDataSize);
    cudaMemcpy(deviceImageDataCopy, deviceImageData, imageDataSize, cudaMemcpyDeviceToDevice);

    gaussianBlurKernel << <gridSize, blockSize >> > (deviceKernel, deviceImageDataCopy, deviceImageData, width, height, channels);

    cudaFree(deviceKernel);

    printf("Applying Sobel operator\n");
    float* deviceMagnitudes;
    cudaMalloc(&deviceMagnitudes, imageDataSize * sizeof(float));
    generateMagnitudesKernel << <gridSize, blockSize >> > (deviceImageData, deviceMagnitudes, width, height, channels);

    cudaMemcpy(deviceImageDataCopy, deviceImageData, imageDataSize, cudaMemcpyDeviceToDevice);
    sobelKernel << <gridSize, blockSize >> > (deviceImageDataCopy, deviceImageData, deviceMagnitudes, width, height, channels);

    cudaFree(deviceImageDataCopy);
    cudaFree(deviceMagnitudes);

    unsigned char* hostImageData = (unsigned char*)malloc(imageDataSize);
    cudaMemcpy(hostImageData, deviceImageData, imageDataSize, cudaMemcpyDeviceToHost);

    cudaFree(deviceImageData);

    cv::namedWindow("Canny edge detection", cv::WINDOW_AUTOSIZE);
    cv::Mat modifiedImage = cv::Mat(height, width, CV_8UC3, hostImageData);
    cv::imshow("Canny edge detection", modifiedImage);
    cv::waitKey(0);

    //free(hostImageData);
    return 0;
}
