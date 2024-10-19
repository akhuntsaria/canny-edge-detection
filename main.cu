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
string media = "image"; // or video

dim3 blockSize = dim3(16, 16, 1);

float* devGaussKernel; // Stored on the device

inline void _checkCudaError(const char* file, int line, const char* function) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error at %s:%d, in function %s: %s\n",
            file, line, function, cudaGetErrorString(err));
        exit(1);
    }
}

void canny(Mat& img) {
    cudaEvent_t start, stop;
    if (media != "video") {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    }

    int width = img.cols,
        height = img.rows;
    size_t rgbDataSize = img.total() * img.channels(),
        grayDataSize = img.total(); // One channel, for grayscale

    // Grayscale
    unsigned char* devRgbData;
    cudaMalloc(&devRgbData, rgbDataSize);
    cudaMemcpy(devRgbData, img.data, rgbDataSize, cudaMemcpyHostToDevice);

    unsigned char* devGrayData;
    cudaMalloc(&devGrayData, grayDataSize);

    dim3 gridSize = dim3((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y, 1);

    grayscale << <gridSize, blockSize >> > (devRgbData, devGrayData, width, height);
    cudaFree(devRgbData);

    // Gaussian blur
    unsigned char* devGrayDataCopy;
    cudaMalloc(&devGrayDataCopy, grayDataSize);
    cudaMemcpy(devGrayDataCopy, devGrayData, grayDataSize, cudaMemcpyDeviceToDevice);

    gaussianBlur << <gridSize, blockSize >> > (devGaussKernel, devGrayDataCopy, devGrayData, width, height);
    cudaFree(devGrayDataCopy);

    // Intensity gradient
    float *devDirections, *devMagnitudes;
    cudaMalloc(&devMagnitudes, grayDataSize * sizeof(float));
    cudaMalloc(&devDirections, grayDataSize * sizeof(float));
    intensityGradient << <gridSize, blockSize >> > (devGrayData, width, height, devMagnitudes, devDirections);

    nonMaximumSuppression << <gridSize, blockSize >> > (devGrayData,
        width,
        height,
        devDirections,
        devMagnitudes, 
        lowThreshold / 100.0f * 255.0f, 
        highThreshold / 100.0f * 255.0f);
    checkCudaError();
    cudaFree(devDirections);
    cudaFree(devMagnitudes);

    // Hysteresis and copy to host
    hysteresis << <gridSize, blockSize >> > (devGrayData, width, height);

    unsigned char* hostImgData;
    cudaHostAlloc((void**)&hostImgData, grayDataSize, cudaHostAllocDefault); // Pinned memory
    cudaMemcpy(hostImgData, devGrayData, grayDataSize, cudaMemcpyDeviceToHost);

    cudaFree(devGrayData);

    // Show or write to a file
    Mat modifiedImg = Mat(height, width, CV_8UC1, hostImgData);
    if (headless) {
        if (media == "image") {
            imwrite("output.bmp", modifiedImg);
            printf("Image saved as output.bmp\n");
        }
        else {
            // Convert and copy image data, needed for video output
            cvtColor(modifiedImg, img, COLOR_GRAY2BGR);
        }
    }
    else {
        imshow("Canny edge detection", modifiedImg);
    }

    cudaFreeHost(hostImgData);

    if (media != "video") {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsed;
        cudaEventElapsedTime(&elapsed, start, stop);
        printf("Canny took %.2fms\n", elapsed);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
}

void initGauss() {
    float kernel[5][5]{},
        sum = .0f,
        sigma = .75f;

    for (int x = -2;x <= 2;x++) {
        for (int y = -2;y <= 2;y++) {
            int i = x + 2,
                j = y + 2;
            kernel[i][j] = (float)(1 / (2 * M_PI * sigma * sigma) * exp(-(x * x + y * y) / 2.0f * sigma * sigma));
            sum += kernel[i][j];
        }
    }

    for (int i = 0;i < 5;i++) {
        for (int j = 0;j < 5;j++) {
            kernel[i][j] /= sum; // Normalize
        }
    }

    printf("Gaussian blur kernel:\n");
    for (int i = 0;i < 5;i++) {
        for (int j = 0;j < 5;j++) {
            printf("%.2f ", kernel[i][j]);
        }
        printf("\n");
    }

    int kernelDataSize = 5 * 5 * sizeof(float);
    cudaMalloc(&devGaussKernel, kernelDataSize);
    cudaMemcpy(devGaussKernel, (float*)kernel, kernelDataSize, cudaMemcpyHostToDevice);
}

void onTrackbar(int, void* userdata) {
    if (lowThreshold > highThreshold) {
        printf("Let's be reasonable here\n");
        highThreshold = lowThreshold;
        setTrackbarPos("High (%)", "Canny edge detection", highThreshold);
        return;
    }

    printf("Low threshold: %d, high: %d\n", lowThreshold, highThreshold);

    if (media == "image") {
        Mat* img = (Mat*)userdata;
        canny(*img);
    }
}

void handleImage(string& inPath) {
    Mat img = imread(inPath, IMREAD_ANYCOLOR);

    if (img.empty()) {
        printf("Could not open the image: %s\n", inPath.c_str());
        return;
    }

    if (headless) {
        canny(img);
    }
    else {
        namedWindow("Original", WINDOW_AUTOSIZE);
        imshow("Original", img);

        namedWindow("Canny edge detection", WINDOW_AUTOSIZE);
        createTrackbar("Low (%)", "Canny edge detection", &lowThreshold, 100, onTrackbar, &img);
        createTrackbar("High (%)", "Canny edge detection", &highThreshold, 100, onTrackbar, &img);

        onTrackbar(0, &img); // First render

        printf("Press q to quit\n");
        while (true) {
            if ((char)waitKey(30) == 'q') break;
        }
    }
}

void handleVideo(string& inPath) {
    VideoCapture cap(inPath);
    if (!cap.isOpened()) {
        printf("Could not open the video: %s\n", inPath.c_str());
        return;
    }

    Mat frame;

    if (headless) {
        int width = (int)cap.get(CAP_PROP_FRAME_WIDTH),
            height = (int)cap.get(CAP_PROP_FRAME_HEIGHT),
            fps = (int)cap.get(cv::CAP_PROP_FPS);
        printf("%dx%d %dfps\n", width, height, fps);

        VideoWriter output("output.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(width, height));
        if (!output.isOpened()) {
            printf("Could not open video for writing: output.avi\n");
            return;
        }

        int frameId = 0;
        while (cap.read(frame)) {
            if (++frameId % 100 == 0) {
                printf("Processing frame %d\n", frameId);
            }
            canny(frame);
            output.write(frame);
        }

        output.release();
    }
    else {
        printf("Press q to quit\n");

        namedWindow("Canny edge detection", WINDOW_AUTOSIZE);
        createTrackbar("Low (%)", "Canny edge detection", &lowThreshold, 100, onTrackbar, nullptr);
        createTrackbar("High (%)", "Canny edge detection", &highThreshold, 100, onTrackbar, nullptr);

        while (cap.read(frame)) {
            canny(frame);
            if ((char)waitKey(30) == 'q') break;
        }
    }

    
    cap.release();
}

int main()
{
    // Config
    string inPath = "in.jpg",
        propPath = "config.properties";
    ifstream propFile(propPath);
    if (propFile.is_open()) {
        string line;
        while (getline(propFile, line)) {
            size_t equalsPos = line.find('=');
            if (equalsPos != string::npos) {
                string key = line.substr(0, equalsPos);
                string value = line.substr(equalsPos + 1);

                if (key == "headless") {
                    headless = (value == "true");
                }
                else if (key == "media") {
                    media = value;
                }
                else if (key == "in") {
                    inPath = value;
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

    initGauss(); // Common for all kernel executions

    // The meat
    if (media == "image") {
        handleImage(inPath);
    }
    else {
        handleVideo(inPath);
    }

    cudaFree(devGaussKernel);
    return 0;
}
