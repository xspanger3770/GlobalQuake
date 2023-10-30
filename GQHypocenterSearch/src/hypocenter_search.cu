#include <iostream>
#include <stdio.h>
#include <cuda_profiler_api.h>
#include <jni.h>

#include "travel_table.hpp"
#include "globalquake.hpp"
#include "geo_utils.hpp"
#include "globalquake_jni_GQNativeFunctions.h"

/**
 * STATION:
 * lat | lon | alt | pwave
 * 
 * HYPOCENTER:
 * correct | err | index | origin
*/

#define STATION_FILEDS 4
#define HYPOCENTER_FILEDS 4
#define BLOCK 240
#define SHARED_TRAVEL_TABLE_SIZE 1024

#define PHI 1.61803398875f

/*struct preliminary_hypocenter_t {
    float origin;
    int index;
    float totalErr;
    int correct;
};*/

bool cuda_initialised = false;
float depth_resolution;

float* travel_table_device;
float* f_results_device;

__device__ void moveOnGlobe(float fromLat, float fromLon, float angle, float distance, float* lat, float* lon)
{
    // calculate angles
    float delta = distance / EARTH_CIRCUMFERENCE;
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

// source: https://developer.nvidia.com/blog/fast-great-circle-distance-calculation-cuda-c/
// everything is in radians
__device__ __host__ float haversine (float lat1, float lon1, 
                            float lat2, float lon2)
{
    float dlat, dlon, c1, c2, d1, d2, a, t;

    c1 = cosf (lat1);
    c2 = cosf (lat2);
    dlat = lat2 - lat1;
    dlon = lon2 - lon1;
    d1 = sinf (dlat);
    d2 = sinf (dlon);
    t = d2 * d2 * c1 * c2;
    a = d1 * d1 + t;
    return asinf (fminf (1.0f, sqrtf (a)));
}

void sanity(){
    float RADIANS = M_PIf / 360.0;

    float lat1 = 0;
    float lon1 = 0;

    float lat2 = 10 * RADIANS;
    float lon2 = 10 * RADIANS;

    printf("ang dist = %f\n", haversine(lat1, lon1, lat2, lon2));
}

__device__ void calculateParams(int points, int index, float maxDist, float fromLat, float fromLon, float* lat, float* lon, float* dist) {
    float ang = 2.0f * M_PIf / (PHI * PHI) * index;
    *dist = sqrtf(index) * (maxDist / sqrtf(points));
    moveOnGlobe(fromLat, fromLon, ang, *dist, lat, lon);
}

const float K = (SHARED_TRAVEL_TABLE_SIZE - 1.0) / MAX_ANG;

__device__ float table_interpolate(float* s_travel_table, float ang) {
    float index = ang * K;

    int index1 = fminf(0, floor(index));
    int index2 = fmaxf(index1 + 1, SHARED_TRAVEL_TABLE_SIZE - 1.0);

    float t = index - index1;
    return (1 - t) * s_travel_table[index1] + t * s_travel_table[index2];
}

__device__ inline float* h_correct(float* hypoc){
    return &hypoc[0];
}

__device__ inline float* h_err(float* hypoc, int grid_size){
    return &hypoc[1 * grid_size];
}

__device__ inline float* h_index(float* hypoc, int grid_size){
    return &hypoc[2 * grid_size];
}

__device__ inline float* h_origin(float* hypoc, int grid_size){
    return &hypoc[3 * grid_size];
}

__device__ void reduce(float *a, float *b, int grid_size){
    if(*h_correct(b) > *h_correct(a) || (*h_correct(b) == *h_correct(a) && *h_err(b, grid_size) < *h_err(a, grid_size))){
        *h_correct(a) = *h_correct(b);
        *h_err(a,grid_size) = *h_err(b, grid_size);
        *h_origin(a, grid_size) = *h_origin(b, grid_size);
        *h_index(a, grid_size) = *h_index(b, grid_size);
    }
}

__global__ void evaluateHypocenter(float* results, float* travel_table, float* stations, 
    float* station_distances, int station_count, int points, float maxDist, float fromLat, float fromLon, float max_depth)
{
    extern __shared__ float s_stations[];
    __shared__ float s_travel_table[SHARED_TRAVEL_TABLE_SIZE];
    __shared__ float s_results[BLOCK * HYPOCENTER_FILEDS];

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    float depth = max_depth * (blockIdx.y / (float)blockDim.y); 
        
    for(int tt_iteration = 0; tt_iteration < ceilf(SHARED_TRAVEL_TABLE_SIZE / blockDim.x); tt_iteration++) {
        int s_index = tt_iteration * blockDim.x + threadIdx.x;
        if(s_index < SHARED_TRAVEL_TABLE_SIZE){
            s_travel_table[s_index] = travel_table[blockIdx.y * SHARED_TRAVEL_TABLE_SIZE + s_index];
        }
    }
    
    for(int station_iteration = 0; station_iteration < ceilf((station_count * STATION_FILEDS) / blockDim.x); station_iteration++){
        int index = station_iteration * blockDim.x * STATION_FILEDS + threadIdx.x;

        if(index < station_count * STATION_FILEDS) {
            s_stations[index] = stations[index];
        }
    }

    __syncthreads();

    if(index >= points){
        return;
    }

    float origin = 0;

    int station_index = threadIdx.x % station_count;
    for(int i = 0; i < station_count; i++, station_index++) {
        if(station_index >= station_count){
            station_index = 0;
        }
        float ang_dist = station_distances[index + i * points];
        float s_pwave = s_stations[station_index + 3 * station_count];
        float expected_travel_time = table_interpolate(s_travel_table, ang_dist);
        origin += s_pwave - expected_travel_time;
    }

    origin /= station_count;

    float err = 0.0;
    float correct = 0;

    station_index = threadIdx.x % station_count;
    for(int i = 0; i < station_count; i++, station_index++) {
        if(station_index >= station_count){
            station_index = 0;
        }
        float ang_dist = station_distances[index + i * points];
        float s_pwave = s_stations[station_index + 3 * station_count];
        float expected_travel_time = table_interpolate(s_travel_table, ang_dist);
        float expected_origin = s_pwave - expected_travel_time;
        float _err = (expected_origin - origin);
        _err *= 2.0f;

        err += _err;
        if(_err < 2.0){
            correct++;
        } 
    }

    s_results[threadIdx.x + blockDim.x * 0] = correct;
    s_results[threadIdx.x + blockDim.x * 1] = err;
    s_results[threadIdx.x + blockDim.x * 2] = index;
    s_results[threadIdx.x + blockDim.x * 3] = origin;
    
    // implementation 3 from slides
    for (unsigned int s = blockDim.x / 2 ; s > 0 ;s >>= 1) {
        if (threadIdx.x < s) {
            reduce(&s_results[threadIdx.x], &s_results[threadIdx.x + s], blockDim.x);
            __syncthreads();
        }
    }

    if(threadIdx.x == 0){
        int idx = blockIdx.y * gridDim.x + blockIdx.x;
        results[idx + 0 * (gridDim.x * gridDim.y)] = s_results[0 * blockDim.x];
        results[idx + 1 * (gridDim.x * gridDim.y)] = s_results[1 * blockDim.x];
        results[idx + 2 * (gridDim.x * gridDim.y)] = s_results[2 * blockDim.x];
        results[idx + 3 * (gridDim.x * gridDim.y)] = s_results[3 * blockDim.x];
    }
}

#define BLOCK_REDUCE 256

__global__ void results_reduce(float* out, float* in, int total_size){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= total_size){
        return;
    }
    __shared__ float s_results[HYPOCENTER_FILEDS * BLOCK_REDUCE];

    s_results[threadIdx.x + BLOCK_REDUCE * 0] = in[index + total_size * 0];
    s_results[threadIdx.x + BLOCK_REDUCE * 1] = in[index + total_size * 1];
    s_results[threadIdx.x + BLOCK_REDUCE * 2] = in[index + total_size * 2];
    s_results[threadIdx.x + BLOCK_REDUCE * 3] = in[index + total_size * 3];
    __syncthreads();

    // implementation 3 from slides
    for (unsigned int s = blockDim.x / 2 ; s > 0 ;s >>= 1) {
        if (threadIdx.x < s) {
            reduce(&s_results[threadIdx.x], &s_results[threadIdx.x + s], blockDim.x);
            __syncthreads();
        }
    }

    if(threadIdx.x == 0){
        int idx = blockIdx.y * gridDim.x + blockIdx.x;
        out[idx + 0 * (gridDim.x * gridDim.y)] = s_results[0 * blockDim.x];
        out[idx + 1 * (gridDim.x * gridDim.y)] = s_results[1 * blockDim.x];
        out[idx + 2 * (gridDim.x * gridDim.y)] = s_results[2 * blockDim.x];
        out[idx + 3 * (gridDim.x * gridDim.y)] = s_results[3 * blockDim.x];
    }
}

__global__ void calculate_station_distances(float* station_distances, float* point_locations, float* stations, int station_count, int points, float maxDist, float fromLat, float fromLon){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float lat, lon, dist;
    
    calculateParams(points, index, maxDist, fromLat, fromLon, &lat, &lon, &dist);

    point_locations[index] = lat;
    point_locations[index + points] = lon;

    int station_index = threadIdx.x % station_count;
    for(int i = 0; i < station_count; i++, station_index++) {
        if(station_index >= station_count){
            station_index = 0;
        }
        float s_lat = stations[station_index + 0 * station_count];
        float s_lon = stations[station_index + 1 * station_count];
        float ang_dist = haversine(lat, lon, s_lat, s_lon);
        station_distances[index + i * points] = ang_dist;
    }
}

bool run_hypocenter_search(float* stations, size_t station_count, size_t points, float depth_resolution, float maxDist, float fromLat, float fromLon, float* final_result)
{
    float* d_stations;
    float* d_stations_distances;
    float* d_point_locations;
    float* d_temp_results;
    bool success = true;
    
    dim3 blocks = {(unsigned int)ceil(points / BLOCK), (unsigned int)ceil(max_depth / depth_resolution) + 1, 1};
    dim3 threads = {BLOCK, 1, 1};
    
    size_t station_array_size = sizeof(float) * station_count * STATION_FILEDS;
    size_t station_distances_array_size = sizeof(float) * station_count * points;
    size_t point_locations_array_size = sizeof(float) * 2 * points;
    size_t temp_results_array_elements = ceil((blocks.x * blocks.y * blocks.z) / BLOCK_REDUCE);

    printf("station array size %.2fkB\n", station_array_size / (1024.0));
    printf("station distances array size %.2fkB\n", station_distances_array_size / (1024.0));
    printf("point locations array size %.2fkB\n", point_locations_array_size / (1024.0));
    printf("temp results array size %.2fkB\n", (sizeof(float) * HYPOCENTER_FILEDS * temp_results_array_elements) / (1024.0));
    
    success &= cudaMalloc(&d_stations, station_array_size) == cudaSuccess;
    success &= cudaMemcpy(d_stations, stations, station_array_size, cudaMemcpyHostToDevice) == cudaSuccess;
    success &= cudaMalloc(&d_stations_distances, station_distances_array_size) == cudaSuccess;
    success &= cudaMalloc(&d_point_locations, station_distances_array_size) == cudaSuccess;

    printf("%d %d %d\n", blocks.x, blocks.y, blocks.z);
    printf("%d %d %d\n", threads.x, threads.y, threads.z);
    printf("total points: %lld\n", (((long long)(blocks.x * blocks.y * blocks.z)) * (long long)(threads.x * threads.y * threads.z)));

    const int block2 = 128;
    const int block_count2 = ceil(points / block2);

    if(success) calculate_station_distances<<<block_count2, block2>>>
        (d_stations_distances, d_point_locations, d_stations, station_count, points, maxDist, fromLat, fromLon);
    
    success &= cudaDeviceSynchronize() == cudaSuccess;
    cudaError err;

    success = (err = cudaGetLastError()) == cudaSuccess;
    if(err != cudaSuccess){
        printf("ERROR IN HYPOCS!!!! %s\n", cudaGetErrorString(err));
    }
    
    if(success) evaluateHypocenter<<<blocks, threads, sizeof(float) * STATION_FILEDS * station_count>>>
        (f_results_device, travel_table_device, d_stations, d_stations_distances, station_count, points, maxDist, fromLat, fromLon, max_depth);

    success &= cudaMalloc(&d_temp_results, sizeof(float) * HYPOCENTER_FILEDS * temp_results_array_elements) == cudaSuccess;
    success &= cudaDeviceSynchronize() == cudaSuccess;
    
    success = (err = cudaGetLastError()) == cudaSuccess;
    if(err != cudaSuccess){
        printf("ERROR IN HYPOCS!!!! %s\n", cudaGetErrorString(err));
    }

    size_t current_result_count = blocks.x * blocks.y * blocks.z;
    while(success && current_result_count > 1){
        dim3 blcks = {(unsigned int)ceil(current_result_count / static_cast<double>(BLOCK_REDUCE)), 1, 1};

        printf("REDUCING... now %ld to %d\n", current_result_count, blcks.x);
        
        results_reduce<<<blcks, BLOCK_REDUCE>>>(d_temp_results, f_results_device, current_result_count);
        success &= cudaDeviceSynchronize();

        success = (err = cudaGetLastError()) == cudaSuccess;
        if(err != cudaSuccess){
            printf("ERROR IN HYPOCS!!!! %s\n", cudaGetErrorString(err));
        }

        current_result_count = blcks.x;

        if(current_result_count == 1){
            success &= cudaMemcpy(final_result, d_temp_results, HYPOCENTER_FILEDS * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
        } else {
            success &= cudaMemcpy(f_results_device,d_temp_results, current_result_count, cudaMemcpyDeviceToDevice) == cudaSuccess;
        }
    }


    if(d_stations) cudaFree(d_stations);
    if(d_stations_distances) cudaFree(d_stations_distances);
    if(d_point_locations) cudaFree(d_point_locations);

    printf("Result = %d\n", success);

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

    printf("run\n");

    float final_result[HYPOCENTER_FILEDS];

    if(run_hypocenter_search(stationsArray, station_count, points, depth_resolution, maxDist, fromLat, fromLon, final_result)){
        printf("FINAL RESULT %f %f %f %f\n", final_result[0], final_result[1], final_result[2], final_result[3]);
    }

    cleanup:
    if(stationsArray) free(stationsArray);


    for (int i = 0; i < station_count; i++) {
        jfloatArray oneDim = (jfloatArray) env->GetObjectArrayElement(stations, i);
        jfloat *elements = env->GetFloatArrayElements(oneDim, 0);

        env->ReleaseFloatArrayElements(oneDim, elements, 0);
        env->DeleteLocalRef(oneDim);
    }

    return nullptr;
}

void prepare_travel_table(float* fitted_travel_table, int rows) {
    for(int row = 0; row < rows; row++){
        for(int column = 0; column < SHARED_TRAVEL_TABLE_SIZE; column++){
            fitted_travel_table[row * SHARED_TRAVEL_TABLE_SIZE + column] = 
                p_interpolate(
                    column / (SHARED_TRAVEL_TABLE_SIZE - 1.0) * MAX_ANG, 
                    (row / (rows-1.0)) * max_depth
                );
        }
    }
}

/*
 * Class:     globalquake_jni_GQNativeFunctions
 * Method:    initCUDA
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_globalquake_jni_GQNativeFunctions_initCUDA
      (JNIEnv *, jclass, jlong max_points, jfloat _depth_resolution){
    bool success = true;
    depth_resolution = _depth_resolution;
    
    dim3 blocks = {(unsigned int)ceil(max_points / BLOCK), (unsigned int)ceil(max_depth / depth_resolution) + 1, 1};
    
    size_t table_size = sizeof(float) * blocks.y * SHARED_TRAVEL_TABLE_SIZE;
    size_t results_size = sizeof(float) * HYPOCENTER_FILEDS * (blocks.x * blocks.y * blocks.z);

    printf("Table array has size %ld\n", (table_size));
    printf("Results array has size %.2fMB\n", (results_size / (1024.0*1024.0)));
    
    success &= cudaMalloc(&travel_table_device, table_size) == cudaSuccess;
    success &= cudaMemcpy(travel_table_device, p_wave_table, table_size, cudaMemcpyHostToDevice) == cudaSuccess;

    success &= cudaMalloc(&f_results_device, results_size) == cudaSuccess;

    printf("Cuda malloc done\n");

    float* fitted_travel_table = static_cast<float*>(malloc(table_size));

    if(fitted_travel_table == nullptr){
        success = false;
        perror("malloc");
    } else {
        printf("Prepare table\n");
        prepare_travel_table(fitted_travel_table, blocks.y);
        success &= cudaMemcpy(travel_table_device, fitted_travel_table, table_size, cudaMemcpyHostToDevice) == cudaSuccess;
        if(!success){
            printf("memcpy f\n");
        }
    }

    if(fitted_travel_table) free(fitted_travel_table);

    printf("init result = %d\n", success);
    cuda_initialised = success;
    return success;
}
