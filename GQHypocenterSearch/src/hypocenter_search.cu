#include <iostream>
#include <stdio.h>
#include <cuda_profiler_api.h>
#include <jni.h>

#include "travel_table.hpp"
#include "globalquake.hpp"
#include "geo_utils.hpp"
#include "globalquake_jni_GQNativeFunctions.h"

#define BLOCK 128
#define PHI 1.61803398875
#define DEPTH_RESOLUTION 1.0

#define STATION_FILEDS 4

float* travel_table_device;

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

__global__ void evaluateHypocenter(size_t points, float maxDist, float fromLat, float fromLon, float max_depth)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float depth = max_depth * (blockIdx.y / (float)blockDim.y); 
    float lat, lon, dist;
    calculateParams(points, index, maxDist, fromLat, fromLon, &lat, &lon, &dist);
}

bool run_hypocenter_search(float* stations, size_t station_count, size_t points, float maxDist, float fromLat, float fromLon)
{
    bool success = true;
    size_t station_array_size = sizeof(float) * station_count * STATION_FILEDS;
    float* d_stations;
    success &= cudaMalloc(&d_stations, station_array_size) == cudaSuccess;
    success &= cudaMemcpy(d_stations, stations, station_array_size, cudaMemcpyHostToDevice) == cudaSuccess;

    dim3 blocks = {(unsigned int)ceil(points / BLOCK), (unsigned int)ceil(max_depth / DEPTH_RESOLUTION), 1};
    dim3 threads = {BLOCK, 1, 1};
    
    printf("%d %d %d\n", blocks.x, blocks.y, blocks.z);
    printf("%d %d %d\n", threads.x, threads.y, threads.z);
    printf("total points: %lld\n", (((long long)(blocks.x * blocks.y * blocks.z)) * (long long)(threads.x * threads.y * threads.z)));

    if(success) evaluateHypocenter<<<blocks, threads>>>(points, maxDist, fromLat, fromLon, max_depth);
    success &= cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();

    if(d_stations) cudaFree(d_stations);

    return success;
}


JNIEXPORT jfloatArray JNICALL Java_globalquake_jni_GQNativeFunctions_findHypocenter
  (JNIEnv *env, jclass, jobjectArray stations, jfloat fromLat, jfloat fromLon, jlong points, jfloat maxDist){
    size_t station_count = env->GetArrayLength(stations);
    
    float* stationsArray = static_cast<float*>(malloc(sizeof(float) * station_count * STATION_FILEDS));
    if(!stationsArray){
        goto cleanup;
    }

    for(int i = 0; i < station_count; i++){
        jfloatArray oneDim = (jfloatArray)env->GetObjectArrayElement(stations, i);
        jfloat *element = env->GetFloatArrayElements(oneDim, 0);
        
        for(int j = 0; j < STATION_FILEDS; j++){
            stationsArray[i * STATION_FILEDS + j] = element[j];    
        }
    }

    run_hypocenter_search(stationsArray, station_count, points, maxDist, fromLat, fromLon);

    if(stationsArray) free(stationsArray);

    cleanup:

    for (int i = 0; i < station_count; i++) {
        jfloatArray oneDim = (jfloatArray) env->GetObjectArrayElement(stations, i);
        jfloat *elements = env->GetFloatArrayElements(oneDim, 0);

        env->ReleaseFloatArrayElements(oneDim, elements, 0);
        env->DeleteLocalRef(oneDim);
    }

    return nullptr;
}

/*
 * Class:     globalquake_jni_GQNativeFunctions
 * Method:    initCUDA
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_globalquake_jni_GQNativeFunctions_initCUDA
      (JNIEnv *, jclass){
    size_t table_size = sizeof(float) * table_columns * table_rows;
    bool success = true;
    success &= cudaMalloc(&travel_table_device, table_size) == cudaSuccess;
    success &= cudaMemcpy(travel_table_device, p_wave_table, table_size, cudaMemcpyHostToDevice) == cudaSuccess;
    return success;
}
