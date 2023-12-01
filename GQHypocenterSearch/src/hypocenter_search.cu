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
 * PRELIMINARY_HYPOCENTER:
 * err | correct (int) | index (int) | origin | depth
 * 
 * RESULT_HYPOCENTER:
 * lat, lon, depth, origin
*/


#define BLOCK_HYPOCS 240
#define BLOCK_REDUCE 256
#define BLOCK_DISTANCES 64

#define STATION_FILEDS 4
#define HYPOCENTER_FILEDS 5
#define SHARED_TRAVEL_TABLE_SIZE 2048

#define PHI2 2.618033989f

#define MUL 1.30f
#define ADD 2.0f

struct depth_profile_t{
    float depth_resolution;
    float* device_travel_table;
};

bool cuda_initialised = false;
float max_depth_resolution;

int depth_profile_count;
depth_profile_t* depth_profiles = nullptr;
float* f_results_device = nullptr;

size_t total_table_size;

void print_err(const char* msg) {
    cudaError err= cudaGetLastError();
    TRACE(2, "%s failed: %s (%d)\n", msg, cudaGetErrorString(err), err);
}

__host__ void moveOnGlobe(float fromLat, float fromLon, float angle, float angular_distance, float* lat, float* lon)
{
    // calculate angles
    float delta = angular_distance;
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

__device__ void moveOnGlobeDevice(float fromLat, float fromLon, float angle, float angular_distance, float* lat, float* lon)
{
    // calculate angles
    float delta = angular_distance;
    float theta = fromLat;
    float phi = fromLon;
    float gamma = angle;

    // calculate sines and cosines
    float c_theta = __cosf(theta);
    float s_theta = __sinf(theta);
    float c_phi = __cosf(phi);
    float s_phi = __sinf(phi);
    float c_delta = __cosf(delta);
    float s_delta = __sinf(delta);
    float c_gamma = __cosf(gamma);
    float s_gamma = __sinf(gamma);

    // calculate end vector
    float x = c_delta * c_theta * c_phi - s_delta * (s_theta * c_phi * c_gamma + s_phi * s_gamma);
    float y = c_delta * c_theta * s_phi - s_delta * (s_theta * s_phi * c_gamma - c_phi * s_gamma);
    float z = s_delta * c_theta * c_gamma + c_delta * s_theta;

    // calculate end lat long
    *lat = asinf(z);
    *lon = atan2f(y, x);
}

// everything is in radians
__device__ float haversine (float lat1, float lon1, 
                            float lat2, float lon2)
{
    float dlat = lat2 - lat1;
    float dlon = lon2 - lon1;

    // Haversine formula
    float a = powf(__sinf(dlat / 2.0f), 2.0f) + __cosf(lat1) * __cosf(lat2) * powf(__sinf(dlon / 2.0f), 2.0f);
    float c = 2.0f * atan2f(sqrtf(a), sqrtf(1.0f - a));

    return c; // Angular distance in radians
}

// everything in radians
void calculateParams(int points, int index, float maxDist, float fromLat, float fromLon, float* lat, float* lon, float* dist) {
    float ang = (2.0f * PI * (float)index) / PHI2;
    *dist = sqrtf(index) * (maxDist / sqrtf(points - 1.0f));
    moveOnGlobe(fromLat, fromLon, ang, *dist, lat, lon);
}

__device__ void calculateParamsDevice(int points, int index, float maxDist, float fromLat, float fromLon, float* lat, float* lon, float* dist) {
    float ang = (2.0f * PI * (float)index) / PHI2;
    *dist = sqrtf(index) * (maxDist / sqrtf(points - 1.0f));
    moveOnGlobeDevice(fromLat, fromLon, ang, *dist, lat, lon);
}

const float K = (SHARED_TRAVEL_TABLE_SIZE - 1.0f) / MAX_ANG;

__device__ float table_interpolate(float* s_travel_table, float ang) {
    float index = ang * K;

    if(index >= SHARED_TRAVEL_TABLE_SIZE - 1.0f) {
        return s_travel_table[SHARED_TRAVEL_TABLE_SIZE - 1]; // some
    }

    int index1 = (int) index;
    int index2 = index1 + 1;

    float t = index - index1;
    return (1.0f - t) * s_travel_table[index1] + t * s_travel_table[index2];
}

__device__ inline float* h_err(float* hypoc, int grid_size){
    return &hypoc[0 * grid_size];
}

__device__ inline int* h_correct(float* hypoc, int grid_size){
    return (int*)&hypoc[1 * grid_size];
}

__device__ inline float* h_index(float* hypoc, int grid_size){
    return &hypoc[2 * grid_size];
}

__device__ inline float* h_origin(float* hypoc, int grid_size){
    return &hypoc[3 * grid_size];
}

__device__ inline float* h_depth(float* hypoc, int grid_size){
    return &hypoc[4 * grid_size];
}

__device__ void reduce(float *a, float *b, int grid_size){
    float err_a = *h_err(a, grid_size);
    float err_b = *h_err(b, grid_size);
    
    int correct_a = *h_correct(a, grid_size);
    int correct_b = *h_correct(b, grid_size);

    /*bool swap = correct_b > correct_a * MUL || (correct_b >= correct_a / MUL && 
        (correct_b / (err_b + ADD) > correct_a / (err_a + ADD))
    );*/

    bool swap = (correct_b / (err_b + ADD) > correct_a / (err_a + ADD));

    if(swap){
        *h_err(a,grid_size) = *h_err(b, grid_size);
        *h_correct(a,grid_size) = *h_correct(b, grid_size);
        *h_origin(a, grid_size) = *h_origin(b, grid_size);
        *h_index(a, grid_size) = *h_index(b, grid_size);
        *h_depth(a, grid_size) = *h_depth(b, grid_size);
    }
}

__global__ void evaluateHypocenter(float* results, float* travel_table, float* stations, 
    float* station_distances, int station_count, int points, float maxDist, float max_depth, float p_wave_threshold)
{
    extern __shared__ float s_stations[];
    __shared__ float s_travel_table[SHARED_TRAVEL_TABLE_SIZE];
    __shared__ float s_results[BLOCK_HYPOCS * HYPOCENTER_FILEDS];

    int point_index = blockIdx.x * blockDim.x + threadIdx.x;

    float depth = max_depth * (blockIdx.y / (float)(gridDim.y - 1.0f)); 
        
    for(int tt_iteration = 0; tt_iteration < ceilf(SHARED_TRAVEL_TABLE_SIZE / static_cast<float>(blockDim.x)); tt_iteration++) {
        int s_index = tt_iteration * blockDim.x + threadIdx.x;
        if(s_index < SHARED_TRAVEL_TABLE_SIZE){
            s_travel_table[s_index] = travel_table[blockIdx.y * SHARED_TRAVEL_TABLE_SIZE + s_index];
        }
    }
    
    for(int station_iteration = 0; station_iteration < ceilf(static_cast<float>(station_count * 1) / blockDim.x); station_iteration++){
        int index = station_iteration * blockDim.x + threadIdx.x;

        if(index < station_count * 1) {
            s_stations[index] = stations[index + 3 * station_count]; // we care only P wave
        }
    }

    __syncthreads();

    if(point_index >= points){
        return;
    }

    int j = blockIdx.y % station_count;
    int station_index2 = (threadIdx.x + j) % station_count;
    float final_origin = 0.0f;
        
    {
        float ang_dist = station_distances[point_index + j * points];
        float s_pwave = s_stations[station_index2];
        float expected_travel_time = table_interpolate(s_travel_table, ang_dist);
        float predicted_origin = s_pwave - expected_travel_time;

        final_origin = predicted_origin;
    }

    int station_index = threadIdx.x % station_count;
    float err = 0.0;
    int correct = station_count;

    for(int i = 0; i < station_count; i++, station_index++) {
        if(station_index >= station_count){
            station_index = 0;
        }
        float ang_dist = station_distances[point_index + i * points];
        float s_pwave = s_stations[station_index];
        float expected_travel_time = table_interpolate(s_travel_table, ang_dist);
        float predicted_origin = s_pwave - expected_travel_time;

        float _err = fabsf(predicted_origin - final_origin);

        if (_err > p_wave_threshold) {
            correct--;
            _err = (_err - p_wave_threshold) * 0.05f + p_wave_threshold;
        }

        err += _err * _err;
    }

    s_results[threadIdx.x + blockDim.x * 0] = err;
    *(int*)&s_results[threadIdx.x + blockDim.x * 1] = correct;
    *(int*)(&s_results[threadIdx.x + blockDim.x * 2]) = point_index;
    s_results[threadIdx.x + blockDim.x * 3] = final_origin;
    s_results[threadIdx.x + blockDim.x * 4] = depth;

    __syncthreads();
    
    // implementation 3 from slides
    for (unsigned int s = blockDim.x / 2 ; s > 0 ;s >>= 1) {
        if (threadIdx.x < s && blockDim.x * blockIdx.x + threadIdx.x + s < points) {
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
        results[idx + 4 * (gridDim.x * gridDim.y)] = s_results[4 * blockDim.x];
    }
}

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
    s_results[threadIdx.x + BLOCK_REDUCE * 4] = in[index + total_size * 4];
    __syncthreads();

    // implementation 3 from slides
    for (unsigned int s = blockDim.x / 2 ; s > 0 ;s >>= 1) {
        if (threadIdx.x < s && blockDim.x * blockIdx.x + threadIdx.x + s < total_size) {
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
        out[idx + 4 * (gridDim.x * gridDim.y)] = s_results[4 * blockDim.x];
    }
}

__global__ void calculate_station_distances(
        float* station_distances, float* stations, int station_count, int points, float maxDist, float fromLat, float fromLon){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index >= points){
        return;
    }

    float lat, lon, dist;
    
    calculateParamsDevice(points, index, maxDist, fromLat, fromLon, &lat, &lon, &dist);

    int station_index = threadIdx.x % station_count;
    for(int i = 0; i < station_count; i++, station_index++) {
        if(station_index >= station_count){
            station_index = 0;
        }
        float s_lat = stations[station_index + 0 * station_count];
        float s_lon = stations[station_index + 1 * station_count];
        float ang_dist = haversine(lat, lon, s_lat, s_lon) * 180.0f / PI; // because travel table is in degrees
        station_distances[index + i * points] = ang_dist;
    }
}

void prepare_travel_table(float* fitted_travel_table, int rows) {
    for(int row = 0; row < rows; row++){
        for(int column = 0; column < SHARED_TRAVEL_TABLE_SIZE; column++){
            fitted_travel_table[row * SHARED_TRAVEL_TABLE_SIZE + column] = 
                p_interpolate(
                    column / (SHARED_TRAVEL_TABLE_SIZE - 1.0) * MAX_ANG, 
                    (row / (rows - 1.0)) * max_depth
                );
        }
    }
}

// returns (accurately) estimated total GPU memory allocation size given the parameters
size_t get_total_allocation_size(size_t points, size_t station_count, float depth_resolution)
{
    size_t result = total_table_size;

    dim3 blocks = {(unsigned int)ceil(static_cast<float>(points) / BLOCK_HYPOCS), (unsigned int)ceil(max_depth / depth_resolution) + 1, 1};
    
    size_t station_array_size = sizeof(float) * station_count * STATION_FILEDS;
    size_t station_distances_array_size = sizeof(float) * station_count * points;
    size_t results_size = sizeof(float) * HYPOCENTER_FILEDS * (blocks.x * blocks.y * blocks.z);  

    size_t temp_results_array_elements = ceil((blocks.x * blocks.y * blocks.z) / static_cast<float>(BLOCK_REDUCE));
    size_t temp_results_array_size = (sizeof(float) * HYPOCENTER_FILEDS * temp_results_array_elements);

    result += station_array_size;
    result += station_distances_array_size;
    result += results_size;
    result += temp_results_array_size;    

    return result;
}


JNIEXPORT jlong JNICALL Java_globalquake_jni_GQNativeFunctions_getAllocationSize(JNIEnv *, jclass, jint points, jint stations, jfloat depth_resolution) {
    return get_total_allocation_size(points, stations, depth_resolution);
}

bool run_hypocenter_search(float* stations, size_t station_count, size_t points, int depth_profile_index, 
    float maxDist, float fromLat, float fromLon, float* final_result, float p_wave_threshold)
{
    if(depth_profile_index < 0 || depth_profile_index >= depth_profile_count){
        TRACE(2, "Error! Invalid depth profile: %d!\n", depth_profile_index);
        return false;
    }

    depth_profile_t* depth_profile = &depth_profiles[depth_profile_index];

    float* d_stations;
    float* d_stations_distances;
    float* d_temp_results;

    if(points < 2){
        TRACE(2, "Error! at least 2 points needed!\n");
        return false;
    }

    if(station_count < 3){
        TRACE(2, "Error! at least 3 stations needed!\n");
        return false;
    }

    bool success = true;
    
    dim3 blocks = {(unsigned int)ceil(static_cast<float>(points) / BLOCK_HYPOCS), (unsigned int)ceil(max_depth / depth_profile->depth_resolution) + 1, 1};
    dim3 threads = {BLOCK_HYPOCS, 1, 1};

    if(blocks.y < 2){
        TRACE(2, "Error! at least 2 depth points needed!\n");
        return false;
    }
    
    size_t station_array_size = sizeof(float) * station_count * STATION_FILEDS;
    size_t station_distances_array_size = sizeof(float) * station_count * points;
    size_t results_size = sizeof(float) * HYPOCENTER_FILEDS * (blocks.x * blocks.y * blocks.z);  

    size_t temp_results_array_elements = ceil((blocks.x * blocks.y * blocks.z) / static_cast<float>(BLOCK_REDUCE));
    size_t current_result_count = blocks.x * blocks.y * blocks.z;

    const int block_count2 = ceil(static_cast<float>(points) / BLOCK_DISTANCES);

    TRACE(1, "station array size (%ld stations) %.2fkB\n", station_count, station_array_size / (1024.0));
    TRACE(1, "station distances array size %.2fkB\n", station_distances_array_size / (1024.0));
    TRACE(1, "temp results array size %.2fkB\n", (sizeof(float) * HYPOCENTER_FILEDS * temp_results_array_elements) / (1024.0));
    TRACE(1, "Results array has size %.2fMB\n", (results_size / (1024.0*1024.0)));
    
    success &= cudaMalloc(&d_stations, station_array_size) == cudaSuccess;
    success &= cudaMemcpy(d_stations, stations, station_array_size, cudaMemcpyHostToDevice) == cudaSuccess;
    success &= cudaMalloc(&d_stations_distances, station_distances_array_size) == cudaSuccess;
    success &= cudaMalloc(&d_temp_results, sizeof(float) * HYPOCENTER_FILEDS * temp_results_array_elements) == cudaSuccess;
    success &= cudaMalloc(&f_results_device, results_size) == cudaSuccess;
    
    if(!success){
        print_err("Hypocs initialisation");
        goto cleanup;
    }

    TRACE(1, "Grid size: %d %d %d\n", blocks.x, blocks.y, blocks.z);
    TRACE(1, "Block size: %d %d %d\n", threads.x, threads.y, threads.z);
    TRACE(1, "total points: %lld\n", (((long long)(blocks.x * blocks.y * blocks.z)) * (long long)(threads.x * threads.y * threads.z)));

    if(success) calculate_station_distances<<<block_count2, BLOCK_DISTANCES>>>
        (d_stations_distances, d_stations, station_count, points, maxDist, fromLat, fromLon);
    
    success &= cudaDeviceSynchronize() == cudaSuccess;
    
    if(!success){
        print_err("Calculate station distances");
        goto cleanup;
    }
    
    if(success) evaluateHypocenter<<<blocks, threads, sizeof(float) * station_count>>>
        (f_results_device, depth_profile->device_travel_table, d_stations, d_stations_distances, station_count, points, maxDist, max_depth, p_wave_threshold);

    success &= cudaDeviceSynchronize() == cudaSuccess;
    
    if(!success){
        print_err("Hypocenter search");
        goto cleanup;
    }

    while(success && current_result_count > 1){
        dim3 blcks = {(unsigned int)ceil(current_result_count / static_cast<double>(BLOCK_REDUCE)), 1, 1};
        TRACE(1, "Reducing... from %ld to %d\n", current_result_count, blcks.x);
        
        results_reduce<<<blcks, BLOCK_REDUCE>>>(d_temp_results, f_results_device, current_result_count);
        success &= cudaDeviceSynchronize() == cudaSuccess;

        if(!success){
            print_err("Reduce");
            goto cleanup;
        }

        current_result_count = blcks.x;

        float local_result[HYPOCENTER_FILEDS];

        if(current_result_count == 1){
            success &= cudaMemcpy(local_result, d_temp_results, HYPOCENTER_FILEDS * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;

            float lat, lon, u_dist;
            calculateParams(points, *(int*)&local_result[2], maxDist, fromLat, fromLon, &lat, &lon, &u_dist);

            final_result[0] = lat;
            final_result[1] = lon;
            final_result[2] = local_result[4];
            final_result[3] = local_result[3];
        } else {
            success &= cudaMemcpy(f_results_device, d_temp_results, current_result_count * HYPOCENTER_FILEDS * sizeof(float), cudaMemcpyDeviceToDevice) == cudaSuccess;
        }

        if(!success){
            print_err("CUDA memcpy");
            goto cleanup;
        }
    }

    cleanup:

    if(d_stations) cudaFree(d_stations);
    if(d_stations_distances) cudaFree(d_stations_distances);
    if(d_temp_results) cudaFree(d_temp_results);
    if(f_results_device) cudaFree(f_results_device);

    return success;
}


JNIEXPORT jfloatArray JNICALL Java_globalquake_jni_GQNativeFunctions_findHypocenter
  (JNIEnv *env, jclass, jfloatArray stations, jfloat fromLat, jfloat fromLon, jlong points, int depthResProfile, jfloat maxDist, jfloat p_wave_threshold) {
    size_t station_count = env->GetArrayLength(stations) / STATION_FILEDS;
    
    bool success = false;
    
    float* stationsArray = static_cast<float*>(malloc(sizeof(float) * station_count * STATION_FILEDS));
    if(!stationsArray){
        perror("malloc");
        goto cleanup;
    }
    
    for(int i = 0; i < env->GetArrayLength(stations); i++){        
        stationsArray[i] = env->GetFloatArrayElements(stations, 0)[i];
    }

    float final_result[HYPOCENTER_FILEDS];

    success = run_hypocenter_search(stationsArray, station_count, points, depthResProfile, maxDist, fromLat, fromLon, final_result, p_wave_threshold);

    cleanup:
    if(stationsArray) free(stationsArray);

    jfloat *elements = env->GetFloatArrayElements(stations, 0);
    env->ReleaseFloatArrayElements(stations, elements, 0);

    jfloatArray result = nullptr;

    if(success){
        result = env->NewFloatArray(4);

        if(result != nullptr){
            env->SetFloatArrayRegion(result, 0, 4, final_result);
        }
    }

    return result;
}

bool initDepthProfiles(float* resols, int count){
    max_depth_resolution = max_depth;
    depth_profile_count = count;

    depth_profiles = static_cast<depth_profile_t*>(malloc(count * sizeof(depth_profile_t)));
    if(depth_profiles == nullptr){
        perror("malloc");
        return false;
    }

    total_table_size = 0;

    for(int i = 0; i < depth_profile_count; i++) {
        float depthRes = resols[i];
        if(depthRes < max_depth_resolution){
            max_depth_resolution = depthRes;
        }

        depth_profiles[i].depth_resolution = depthRes;

        int rows = (unsigned int)ceil(max_depth / depthRes) + 1;
        size_t table_size = sizeof(float) * rows * SHARED_TRAVEL_TABLE_SIZE;
        total_table_size += table_size;

        TRACE(1, "Creating depth profile with resolution %.2fkm (%.2fkB)\n", depthRes, table_size / 1024.0);

        // todo fitted array
        if(cudaMalloc(&depth_profiles[i].device_travel_table, table_size) != cudaSuccess){
            print_err("CUDA malloc");
            return false;
        }

        float* fitted_travel_table = static_cast<float*>(malloc(table_size));

        if(fitted_travel_table == nullptr){
            perror("malloc");
            return false;
        } else {
            prepare_travel_table(fitted_travel_table, rows);
            if(cudaMemcpy(depth_profiles[i].device_travel_table, fitted_travel_table, table_size, cudaMemcpyHostToDevice) != cudaSuccess){
                print_err("CUDA memcpy");
                free(fitted_travel_table);
                return false;
            }

            free(fitted_travel_table);
            fitted_travel_table = NULL;
        }
    }

    return true;
}

/*
 * Class:     globalquake_jni_GQNativeFunctions
 * Method:    initCUDA
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_globalquake_jni_GQNativeFunctions_initCUDA
      (JNIEnv *env, jclass, jfloatArray depth_profiles_array){
    bool success = true;

    if(depth_profiles_array != nullptr && depth_profiles == nullptr) {
        int depth_profile_count = env->GetArrayLength(depth_profiles_array);
        float depthResols[depth_profile_count];
        for(int i = 0; i < depth_profile_count; i++){
            depthResols[i] = env->GetFloatArrayElements(depth_profiles_array, 0)[i];
        }

        success &= initDepthProfiles(depthResols, depth_profile_count);
        env->ReleaseFloatArrayElements(depth_profiles_array, env->GetFloatArrayElements(depth_profiles_array, 0), 0);
    }
    
    cuda_initialised = success;
    return success;
}
