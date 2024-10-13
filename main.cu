#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <sstream>
#include <string>

#define CHANNELS 3 // RGB

#define M_PI 3.141592f

#define STRONG_EDGE 255
#define WEAK_EDGE 128
#define NO_EDGE 0

using namespace cv;
using namespace std;

int lowThreshold = 20, // %
    highThreshold = 60;

// Convert an index, img[channel][i][j] to flat[idx]
__device__ __host__ int getIdx(int width, int channel, int i, int j) {
    return j * width * CHANNELS + i * CHANNELS + channel;
}

__global__ void gaussianBlur(float* kernel, unsigned char* source, unsigned char* target, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x,
        y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int totalSize = width * height * CHANNELS;
        for (int ch = 0;ch < CHANNELS;ch++) {
            float weightedSum = 0;
            for (int i = -2;i <= 2;i++) {
                for (int j = -2;j <= 2;j++) {
                    int idx = getIdx(width, ch, x + i, y + j);
                    if (idx >= 0 && idx < totalSize) {
                        // Get the flat index + move indices from [-2,2] to [0,4] for the kernel
                        int kernelIdx = (i + 2) * 5 + (j + 2);

                        weightedSum += (int)source[idx] * kernel[kernelIdx];
                    }
                }
            }

            target[getIdx(width, ch, x, y)] = weightedSum;
        }
    }
}

__global__ void grayscale(unsigned char* imgData, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int channelAvg = 0;
        for (int ch = 0;ch < CHANNELS;ch++) {
            channelAvg += (int)imgData[getIdx(width, ch, x, y)];
        }
        channelAvg /= 3;
        for (int ch = 0;ch < CHANNELS;ch++) {
            imgData[getIdx(width, ch, x, y)] = channelAvg;
        }
    }
}

// Calculate gradient magnitudes and directions here to avoid do it again in the Sobel operator kernel
__global__ void intensityGradient(unsigned char* source, int width, int height, float* magnitudes, float* directions) {
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

        int totalSize = width * height * CHANNELS;
        for (int ch = 0;ch < CHANNELS;ch++) {
            int Gx = 0,
                Gy = 0;
            for (int i = -1;i <= 1;i++) {
                for (int j = -1;j <= 1;j++) {
                    int idx = getIdx(width, ch, x + i, y + j);
                    if (idx >= 0 && idx < totalSize) {
                        // Move indices from [-1,1] to [0,2] for the kernel
                        Gx += (int)source[idx] * kernelX[i + 1][j + 1];
                        Gy += (int)source[idx] * kernelY[i + 1][j + 1];
                    }
                }
            }

            int idx = getIdx(width, ch, x, y),

            float magnitude = sqrt((float)Gx * Gx + Gy * Gy);
            magnitudes[idx] = magnitude;

            float direction = atan2f(Gy, Gx);
            direction = direction * 180.0f / M_PI; // Convert to degrees
            if (direction < 0.0f) {
                direction += 180.0f;
            }
            directions[idx] = direction;
        }
    }
}

__global__ void hysteresis(unsigned char* source, unsigned char* target, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int totalSize = width * height * CHANNELS;
    for (int ch = 0;ch < CHANNELS;ch++) {
        int idx = getIdx(width, ch, x, y);
        if (source[idx] == STRONG_EDGE) {
            target[idx] = STRONG_EDGE;  // Strong edge is retained
        }
        else if (source[idx] >= WEAK_EDGE) {
            // Check if it is connected to any strong edge
            bool connectedToStrongEdge = false;
            for (int i = -1; i <= 1 && !connectedToStrongEdge; i++) {
                for (int j = -1; j <= 1; j++) {
                    int neighborIdx = getIdx(width, ch, x + i, y + j);
                    if (neighborIdx >= 0 && neighborIdx < totalSize && source[neighborIdx] == STRONG_EDGE) {
                        connectedToStrongEdge = true;
                        break;
                    }
                }
            }
            target[idx] = connectedToStrongEdge ? STRONG_EDGE : NO_EDGE;
        }
        else {
            target[idx] = NO_EDGE;  // Suppress weak edges not connected to strong ones
        }
    }
}

__global__ void sobel(unsigned char* source, 
                      unsigned char* target, 
                      int width, 
                      int height,
                      float* magnitudes,
                      float* directions,
                      float lowThreshold, 
                      float highThreshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x,
        y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        for (int ch = 0;ch < CHANNELS;ch++) { 
            int idx = getIdx(width, ch, x, y);

            // Non-maximum suppression, compare with neighboring pixels
            float neighbor1,
                neighbor2,
                direction = directions[idx];
            if (direction < 22.5f || direction >= 157.5f) { // North-South
                neighbor1 = y > 0 ? magnitudes[getIdx(width, ch, x, y - 1)] : 0.0f;
                neighbor2 = y < height - 1 ? magnitudes[getIdx(width, ch, x, y + 1)] : 0.0f;
            }
            else if (direction < 67.5f) { // North-East to South-West
                neighbor1 = x > 0 && y > 0 ? magnitudes[getIdx(width, ch, x - 1, y - 1)] : 0.0f;
                neighbor2 = x < width - 1 && y < height - 1 ? magnitudes[getIdx(width, ch, x + 1, y + 1)] : 0.0f;
            }
            else if (direction < 112.5f) { // East-West
                neighbor1 = x > 0 ? magnitudes[getIdx(width, ch, x - 1, y)] : 0.0f;
                neighbor2 = x < width - 1 ? magnitudes[getIdx(width, ch, x + 1, y)] : 0.0f;
            }
            else { // North-West to South-East
                neighbor1 = x > 0 && y < height - 1 ? magnitudes[getIdx(width, ch, x - 1, y + 1)] : 0.0f;
                neighbor2 = x < width - 1 && y > 0 ? magnitudes[getIdx(width, ch, x + 1, y - 1)] : 0.0f;
            }

            // Preserve the current pixel if it's the maximum
            float currentMagnitude = magnitudes[idx];
            if (currentMagnitude > neighbor1 && currentMagnitude > neighbor2) {
                // Double thresholding
                if (currentMagnitude > highThreshold) {
                    target[idx] = STRONG_EDGE;
                }
                else if (currentMagnitude > lowThreshold) {
                    target[idx] = WEAK_EDGE;
                }
                else {
                    target[idx] = NO_EDGE; // Suppressed pixel
                }
            }
            else {
                target[idx] = 0;
            }
        }
    }
}

void renderModified(Mat& img) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int width = img.cols,
        height = img.rows,
        channels = img.channels(); // Blue, green, red, etc.

    size_t imgDataSize = img.total() * img.channels();
    unsigned char* deviceImgData;
    cudaMalloc(&deviceImgData, imgDataSize);
    cudaMemcpy(deviceImgData, img.data, imgDataSize, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16, 1);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y, 1);

    //TODO log once
    //printf("blockSize: (%d,%d,%d)\n", blockSize.x, blockSize.y, blockSize.z);
    //printf("gridSize: (%d,%d,%d)\n", gridSize.x, gridSize.y, gridSize.z);

    grayscale << <gridSize, blockSize >> > (deviceImgData, width, height);

    // Gaussian blur
    //TODO calculate once
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

    //TODO print once
    /*printf("Gaussian blur kernel:\n");
    for (int i = 0;i < 5;i++) {
        for (int j = 0;j < 5;j++) {
            printf("%f ", kernel[i][j]);
        }
        printf("\n");
    }*/

    int kernelDataSize = 5 * 5 * sizeof(float);
    float* deviceKernel;
    cudaMalloc(&deviceKernel, kernelDataSize);
    cudaMemcpy(deviceKernel, (float*)kernel, kernelDataSize, cudaMemcpyHostToDevice);

    unsigned char* deviceImgDataCopy;
    cudaMalloc(&deviceImgDataCopy, imgDataSize);
    cudaMemcpy(deviceImgDataCopy, deviceImgData, imgDataSize, cudaMemcpyDeviceToDevice);

    gaussianBlur << <gridSize, blockSize >> > (deviceKernel, deviceImgDataCopy, deviceImgData, width, height);

    cudaFree(deviceKernel);

    // Intensity gradient
    float *deviceMagnitudes, *deviceDirections;
    cudaMalloc(&deviceMagnitudes, imgDataSize * sizeof(float));
    cudaMalloc(&deviceDirections, imgDataSize * sizeof(float));
    intensityGradient << <gridSize, blockSize >> > (deviceImgData, width, height, deviceMagnitudes, deviceDirections);

    cudaMemcpy(deviceImgDataCopy, deviceImgData, imgDataSize, cudaMemcpyDeviceToDevice);
    sobel << <gridSize, blockSize >> > (deviceImgDataCopy, 
                                        deviceImgData, 
                                        width, 
                                        height, 
                                        deviceMagnitudes, 
                                        deviceDirections, 
                                        lowThreshold / 100.0f * 255.0f, 
                                        highThreshold / 100.0f * 255.0f);

    cudaFree(deviceMagnitudes);
    cudaFree(deviceDirections);

    // Hysteresis and copy to host
    cudaMemcpy(deviceImgDataCopy, deviceImgData, imgDataSize, cudaMemcpyDeviceToDevice);
    hysteresis << <gridSize, blockSize >> > (deviceImgDataCopy, deviceImgData, width, height);

    cudaFree(deviceImgDataCopy);

    unsigned char* hostImgData = (unsigned char*)malloc(imgDataSize);
    cudaMemcpy(hostImgData, deviceImgData, imgDataSize, cudaMemcpyDeviceToHost);

    cudaFree(deviceImgData);
    
    // Measure time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Low threshold: %d, high: %d, took %dms\n", lowThreshold, highThreshold, (int)milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Show
    Mat modifiedImg = Mat(height, width, CV_8UC3, hostImgData);
    imshow("Canny edge detection", modifiedImg);

    free(hostImgData);
}

void onTrackbar(int, void* userdata) {
    if (lowThreshold > highThreshold) {
        printf("Let's be reasonable here\n");
        highThreshold = lowThreshold;
        setTrackbarPos("High (%)", "Canny edge detection", highThreshold);
        return;
    }

    Mat* img = static_cast<Mat*>(userdata);
    renderModified(*img);
}

int main(int argc, char* argv[])
{
    string imgPath = "image.jpg";
    string propPath = "config.properties";
    ifstream propFile(propPath);
    if (propFile.is_open()) {
        string line;
        while (getline(propFile, line)) {
            size_t equalsPos = line.find('=');
            if (equalsPos != string::npos) {
                string key = line.substr(0, equalsPos);
                string value = line.substr(equalsPos + 1);

                if (key == "image") {
                    imgPath = value;
                    break;
                }
            }
        }
    }
    else {
        printf("Could not open or find the properties file: %s\n", propPath);
    }

    Mat img = imread(imgPath, IMREAD_ANYCOLOR);

    if (img.empty()) {
        printf("Could not open or find the image: %s\n", imgPath);
        return 1;
    }

    namedWindow("Original", WINDOW_AUTOSIZE);
    imshow("Original", img);

    namedWindow("Canny edge detection", WINDOW_AUTOSIZE);
    createTrackbar("Low (%)", "Canny edge detection", &lowThreshold, 100, onTrackbar, &img);
    createTrackbar("High (%)", "Canny edge detection", &highThreshold, 100, onTrackbar, &img);

    // First render
    onTrackbar(0, &img);

    while (true) {
        char key = (char)waitKey(100);
        if (key == 'q') {
            break;
        }
    }

    return 0;
}
