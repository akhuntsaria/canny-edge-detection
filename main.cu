#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <limits.h>
#include <opencv2/core.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>

#define M_PI 3.141592

const int KEY_P = 112;
const int KEY_Q = 113;

bool isPaused = false;

// Color + 2D index, image[channel][i][j], to 1D
__device__ __host__ int get1dIdx(int width, int channels, int channel, int i, int j) {
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

__global__ void gradientMagnitudeThresholdingKernel(unsigned char* source, unsigned char* target, int width, int height, int channels, int threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x,
        y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        for (int ch = 0;ch < channels;ch++) {
            int idx = y * width * channels + x * channels;
            target[idx + ch] = target[idx + ch] >= threshold ? target[idx + ch] : 0;
        }
    }
}

__global__ void grayscaleKernel(unsigned char* imageData, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width * channels + x * channels;
        int channelAvg = 0;
        for (int ch = 0;ch < channels;ch++) {
            channelAvg += (int)imageData[idx + ch];
        }
        channelAvg /= 3;
        for (int ch = 0;ch < channels;ch++) {
            imageData[idx + ch] = channelAvg;
        }
    }
}

void handleKeyboardInput() {
    int key = cv::waitKey(100);

    if (key != -1) {
        printf("Pressed %d\n", key);

        if (key == KEY_Q) {
            exit(0);
        }

        if (key == KEY_P) {
            isPaused = !isPaused;
            if (isPaused) {
                printf("Paused. Press 'p' to resume or 'q' to exit.\n");
            }
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
                    int idx = (y + j) * width * channels + (x + i) * channels;
                    idx += ch;
                    if (idx >= 0 && idx < totalSize) {
                        // Move indices from [-1,1] to [0,2] for the kernel
                        Gx += (int)source[idx] * kernelX[i + 1][j + 1];
                        Gy += (int)source[idx] * kernelY[i + 1][j + 1];
                    }
                }
            }

            float magnitude = sqrt((float)Gx * Gx + Gy * Gy);
            int idx = y * width * channels + x * channels;
            target[idx + ch] = magnitude;
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
                    int idx = (y + j) * width * channels + (x + i) * channels;
                    idx += ch;
                    if (idx >= 0 && idx < totalSize) {
                        // Move indices from [-1,1] to [0,2] for the kernel
                        Gx += (int)source[idx] * kernelX[i + 1][j + 1];
                        Gy += (int)source[idx] * kernelY[i + 1][j + 1];
                    }
                }
            }

            float magnitude = sqrt((float)Gx * Gx + Gy * Gy);

            float angle = atan2f((float)Gy, (float)Gx) * 180 / M_PI;
            angle = fmodf(angle + 180, 180); // Normalize angle to [0, 180)

            // Non-maximum suppression
            bool isMax = true;
            if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle < 180)) {
                if (x > 0 && x < width - 1) {
                    isMax = magnitude >= magnitudes[get1dIdx(width, channels, ch, x - 1, y)] && magnitude >= magnitudes[get1dIdx(width, channels, ch, x + 1, y)];
                }
            }
            else if (angle >= 22.5 && angle < 67.5) {
                if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
                    isMax = magnitude >= magnitudes[get1dIdx(width, channels, ch, x - 1, y - 1)] && magnitude >= magnitudes[get1dIdx(width, channels, ch, x + 1, y + 1)];
                }
            }
            else if (angle >= 67.5 && angle < 112.5) {
                if (y > 0 && y < height - 1) {
                    isMax = magnitude >= magnitudes[get1dIdx(width, channels, ch, x, y - 1)] && magnitude >= magnitudes[get1dIdx(width, channels, ch, x, y + 1)];
                }
            }
            else if (angle >= 112.5 && angle < 157.5) {
                if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
                    isMax = magnitude >= magnitudes[get1dIdx(width, channels, ch, x + 1, y - 1)] && magnitude >= magnitudes[get1dIdx(width, channels, ch, x - 1, y + 1)];
                }
            }

            // Suppress the gradient magnitude if it's below the threshold or not a local maximum
            if (!isMax) {
                magnitude = 0;
            }
            else if (magnitude > 255) {
                magnitude = 255;
            }

            int idx = y * width * channels + x * channels;
            target[idx + ch] = (int)magnitude;
        }
    }
}

int main(int argc, char* argv[])
{
    const char* currentFilter = NULL;
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

    unsigned char* hostImageData;

    dim3 blockSize(16, 16, 1);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y, 1);

    printf("blockSize: (%d,%d,%d)\n", blockSize.x, blockSize.y, blockSize.z);
    printf("gridSize: (%d,%d,%d)\n", gridSize.x, gridSize.y, gridSize.z);

    cv::namedWindow(currentFilter, cv::WINDOW_AUTOSIZE);

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

    hostImageData = (unsigned char*)malloc(imageDataSize);
    cudaMemcpy(hostImageData, deviceImageData, imageDataSize, cudaMemcpyDeviceToHost);

    cudaFree(deviceImageDataCopy);
    cudaFree(deviceMagnitudes);
    
    cv::Mat modifiedImage = cv::Mat(height, width, CV_8UC3, hostImageData);

    cv::imwrite("modified.bmp", modifiedImage);

    cv::imshow(currentFilter, modifiedImage);

    cv::waitKey(0);

    cudaFree(deviceImageData);
    free(hostImageData);
    return 0;
}
