#include <iostream>
#include <stdio.h>
#include <cuda_profiler_api.h>

#include "travel_table.h"

#define BLOCK 128

#define PHI 1.61803398875
#define EARTH_RADIUS 40075.0
#define MAX_DEPTH 750
#define DEPTH_RESOLUTION 1.0

struct preliminary_hypocenter_t{
    float err;
    int correct;
};

__device__ void moveOnGlobe(float fromLat, float fromLon, float angle, float distance, float* lat, float* lon)
{
    // calculate angles
    float delta = distance / EARTH_RADIUS;
    float theta = fromLat;
    float phi = fromLon;
    float gamma = angle;

    // calculate sines and cosines
    float c_theta = cosf(theta);
    float s_theta = sinf(theta);
    float c_phi = cosf(phi);
    float s_phi = sinf(phi);
    float c_delta = cosf(delta);
    float s_delta = sinf(delta);
    float c_gamma = cosf(gamma);
    float s_gamma = sinf(gamma);

    // calculate end vector
    float x = c_delta * c_theta * c_phi - s_delta * (s_theta * c_phi * c_gamma + s_phi * s_gamma);
    float y = c_delta * c_theta * s_phi - s_delta * (s_theta * s_phi * c_gamma - c_phi * s_gamma);
    float z = s_delta * c_theta * c_gamma + c_delta * s_theta;

    // calculate end lat long
    *lat = asinf(z);
    *lon = atan2f(y, x);
}

__device__ void calculateParams(int points, int index, float maxDist, float fromLat, float fromLon, float* lat, float* lon, float* dist) {
    float ang = 2 * M_PI / (PHI * PHI) * index;
    *dist = sqrtf(index) * (maxDist / sqrtf(points));
    moveOnGlobe(fromLat, fromLon, ang, *dist, lat, lon);
}

__global__ void evaluateHypocenter(size_t points, float maxDist, float fromLat, float fromLon)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float depth = MAX_DEPTH * (blockIdx.y / (float)blockDim.y); 
    float lat, lon, dist;
    calculateParams(points, index, maxDist, fromLat, fromLon, &lat, &lon, &dist);
}

void hypocenter_search(size_t points, float maxDist, float fromLat, float fromLon)
{

    dim3 blocks = {(unsigned int)ceil(points / BLOCK), (unsigned int)ceil(MAX_DEPTH / DEPTH_RESOLUTION), 1};
    dim3 threads = {BLOCK, 1, 1};
    printf("%d %d %d\n", blocks.x, blocks.y, blocks.z);
    printf("%d %d %d\n", threads.x, threads.y, threads.z);
    printf("total points: %lld\n", (((long long)(blocks.x * blocks.y * blocks.z)) * (long long)(threads.x * threads.y * threads.z)));
    evaluateHypocenter<<<blocks, threads>>>(points, maxDist, fromLat, fromLon);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();  // add
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << "("<<+err<<")" << std::endl; // add
    }else {
        std::cout << "Success" << std::endl;
    }

    cudaProfilerStop();
}

int main()
{
    hypocenter_search(100000000, 10000.0, 0, 0);

    cudaDeviceReset();

    return 0;
}