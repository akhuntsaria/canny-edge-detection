#pragma once

__global__ void gaussianBlur(float* kernel, unsigned char* source, unsigned char* target, int width, int height);
__global__ void grayscale(unsigned char* imgData, unsigned char* grayData, int width, int height);
__global__ void hysteresis(unsigned char* target, int width, int height);
__global__ void intensityGradient(unsigned char* source, int width, int height, float* magnitudes, float* directions);
__global__ void nonMaximumSuppression(unsigned char* img, int width, int height, float* directions, float* magnitudes, float lowThreshold, float highThreshold);

