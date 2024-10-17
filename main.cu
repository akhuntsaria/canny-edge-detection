#define _USE_MATH_DEFINES
#include <cmath>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>

#include "kernels.h"

#define checkCudaError() _checkCudaError(__FILE__, __LINE__, __func__)

using namespace cv;
using namespace std;

bool headless = false;
int lowThreshold = 20, // %
    highThreshold = 60;

dim3 blockSize;
dim3 gridSize;

// Memory reserved on the device and passed from preprocessing to the (repeated) final stage
float* devDirections;
float* devMagnitudes;
unsigned char* devImgData;

inline void _checkCudaError(const char* file, int line, const char* function) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error at %s:%d, in function %s: %s\n",
            file, line, function, cudaGetErrorString(err));
        exit(1);
    }
}

// Grayscale, Gaussian blur and calculating of the intensity gradient
void preprocess(Mat& img) {
    int width = img.cols,
        height = img.rows;
    size_t rgbDataSize = img.total() * img.channels(),
        imgDataSize = img.total(); // One channel, for grayscale

    unsigned char* devRgbData;
    cudaMalloc(&devRgbData, rgbDataSize);
    cudaMemcpy(devRgbData, img.data, rgbDataSize, cudaMemcpyHostToDevice);

    cudaMalloc(&devImgData, imgDataSize);

    blockSize = dim3(16, 16, 1);
    gridSize = dim3((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y, 1);

    printf("blockSize: (%d,%d,%d)\n", blockSize.x, blockSize.y, blockSize.z);
    printf("gridSize: (%d,%d,%d)\n", gridSize.x, gridSize.y, gridSize.z);

    grayscale << <gridSize, blockSize >> > (devRgbData, width, height, devImgData);

    cudaFree(devRgbData);

    // Gaussian blur
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

    for (int i = 0;i < 5;i++) {
        for (int j = 0;j < 5;j++) {
            kernel[i][j] /= kernelSum; // Normalize
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
    float* devKernel;
    cudaMalloc(&devKernel, kernelDataSize);
    cudaMemcpy(devKernel, (float*)kernel, kernelDataSize, cudaMemcpyHostToDevice);

    unsigned char* devImgDataCopy;
    cudaMalloc(&devImgDataCopy, imgDataSize);
    cudaMemcpy(devImgDataCopy, devImgData, imgDataSize, cudaMemcpyDeviceToDevice);

    gaussianBlur << <gridSize, blockSize >> > (devKernel, devImgDataCopy, devImgData, width, height);

    cudaFree(devKernel);
    cudaFree(devImgDataCopy);

    // Intensity gradient
    cudaMalloc(&devMagnitudes, imgDataSize * sizeof(float));
    cudaMalloc(&devDirections, imgDataSize * sizeof(float));
    intensityGradient << <gridSize, blockSize >> > (devImgData, width, height, devMagnitudes, devDirections);
}

// Non-maximum suppression, double thresholding and hysteresis
void finalStage(Mat& img) {
    int width = img.cols,
        height = img.rows;
    size_t imgDataSize = img.total();

    // Non-maximum supression and measure time
    unsigned char* devImgDataCopy;
    cudaMalloc(&devImgDataCopy, imgDataSize);
    cudaMemcpy(devImgDataCopy, devImgData, imgDataSize, cudaMemcpyDeviceToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    nonMaximumSuppression << <gridSize, blockSize >> > (devImgDataCopy,
        width,
        height,
        devDirections,
        devMagnitudes, 
        lowThreshold / 100.0f * 255.0f, 
        highThreshold / 100.0f * 255.0f);
    checkCudaError();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("nonMaximumSuppression took %.2fms\n", elapsed);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Hysteresis and copy to host
    hysteresis << <gridSize, blockSize >> > (devImgDataCopy, width, height);

    unsigned char* hostImgData;
    cudaHostAlloc((void**)&hostImgData, imgDataSize, cudaHostAllocDefault); // Pinned memory
    cudaMemcpy(hostImgData, devImgDataCopy, imgDataSize, cudaMemcpyDeviceToHost);

    cudaFree(devImgDataCopy);

    // Show or write to a file
    Mat modifiedImg = Mat(height, width, CV_8UC1, hostImgData);
    if (headless) {
        imwrite("output.bmp", modifiedImg);
        printf("Image saved as output.bmp\n");
    }
    else {
        imshow("Canny edge detection", modifiedImg);
    }

    cudaFreeHost(hostImgData);
}

void onTrackbar(int, void* userdata) {
    if (lowThreshold > highThreshold) {
        printf("Let's be reasonable here\n");
        highThreshold = lowThreshold;
        setTrackbarPos("High (%)", "Canny edge detection", highThreshold);
        return;
    }

    printf("Low threshold: %d, high: %d\n", lowThreshold, highThreshold);

    Mat* img = static_cast<Mat*>(userdata);
    finalStage(*img);
}

int main()
{
    /*string imgPath = "image.jpg",
        propPath = "config.properties";
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
                }
                else if (key == "headless") {
                    headless = (value == "true");
                }
                else if (key == "lowThreshold") {
                    lowThreshold = stoi(value);
                }
                else if (key == "highThreshold") {
                    highThreshold = stoi(value);
                }
            }
        }
    }
    else {
        printf("Could not open or find the properties file: %s\n", propPath.c_str());
    }

    Mat img = imread(imgPath, IMREAD_ANYCOLOR);

    if (img.empty()) {
        printf("Could not open or find the image: %s\n", imgPath.c_str());
        return 1;
    }

    if (!headless) {
        namedWindow("Original", WINDOW_AUTOSIZE);
        imshow("Original", img);
    }

    preprocess(img);

    if (headless) {
        finalStage(img);
    }
    else {
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
    }

    cudaFree(devImgData);
    cudaFree(devDirections);
    cudaFree(devMagnitudes);

    return 0;*/

    VideoCapture cap("(output002)2053420-uhd_3840_2160_30fps.mp4");
    if (!cap.isOpened()) {
        cerr << "Error opening video file." << endl;
        return -1;
    }

    Mat frame, gray, edges;

    // Loop through each frame of the video
    while (cap.read(frame)) {
        // Convert the frame to grayscale
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Apply Canny edge detection
        Canny(gray, edges, 50, 100); // Adjust thresholds as needed

        // Display the edges
        imshow("Canny Edge Detection", edges);
        if (waitKey(30) >= 0) break;  // Exit on key press
    }

    // Release the video capture and close windows
    cap.release();
    destroyAllWindows();
    return 0;
}
