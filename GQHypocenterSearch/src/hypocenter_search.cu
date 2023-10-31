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
 * correct | err | index | origin
 * 
 * RESULT_HYPOCENTER:
 * lat, lon, depth, origin
*/


#define BLOCK_HYPOCS 240
#define BLOCK_REDUCE 256
#define BLOCK_DISTANCES 64

#define STATION_FILEDS 4
#define HYPOCENTER_FILEDS 5
#define SHARED_TRAVEL_TABLE_SIZE 1024

#define PHI2 (2.618033989f)

/*struct preliminary_hypocenter_t {
    float origin;
    int index;
    float totalErr;
    int correct;
};*/

bool cuda_initialised = false;
float max_depth_resolution;

float* travel_table_device;
float* f_results_device;

__host__ __device__ void moveOnGlobe(float fromLat, float fromLon, float angle, float angular_distance, float* lat, float* lon)
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

// everything is in radians
__device__ __host__ float haversine (float lat1, float lon1, 
                            float lat2, float lon2)
{
    float dlat = lat2 - lat1;
    float dlon = lon2 - lon1;

    // Haversine formula
    float a = powf(sinf(dlat / 2.0f), 2.0f) + cosf(lat1) * cosf(lat2) * powf(sinf(dlon / 2.0f), 2.0f);
    float c = 2.0f * atan2f(sqrtf(a), sqrtf(1.0f - a));

    return c; // Angular distance in radians
}

// everything in radians
__device__ __host__ void calculateParams(int points, int index, float maxDist, float fromLat, float fromLon, float* lat, float* lon, float* dist) {
    float ang = (2.0f * M_PIf * (float)index) / PHI2;
    *dist = sqrtf(index) * (maxDist / sqrtf(points - 1.0f));
    moveOnGlobe(fromLat, fromLon, ang, *dist, lat, lon);
}

const float K = (SHARED_TRAVEL_TABLE_SIZE - 1.0f) / MAX_ANG;

__device__ float table_interpolate(float* s_travel_table, float ang) {
    float index = ang * K;

    if(index < 0){
        return s_travel_table[0];
    } else if(index >= SHARED_TRAVEL_TABLE_SIZE - 1.0f) {
        return s_travel_table[SHARED_TRAVEL_TABLE_SIZE - 1];
    }

    int index1 = (int) index;
    int index2 = index1 + 1;

    float t = index - index1;
    return (1.0f - t) * s_travel_table[index1] + t * s_travel_table[index2];
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

__device__ inline float* h_depth(float* hypoc, int grid_size){
    return &hypoc[4 * grid_size];
}

__device__ void reduce(float *a, float *b, int grid_size){
    if(*h_correct(b) > *h_correct(a) || (*h_correct(b) == *h_correct(a) && *h_err(b, grid_size) < *h_err(a, grid_size))){
        *h_correct(a) = *h_correct(b);
        *h_err(a,grid_size) = *h_err(b, grid_size);
        *h_origin(a, grid_size) = *h_origin(b, grid_size);
        *h_index(a, grid_size) = *h_index(b, grid_size);
        *h_depth(a, grid_size) = *h_depth(b, grid_size);
    }
}

__global__ void evaluateHypocenter(float* results, float* travel_table, float* stations, 
    float* station_distances, int station_count, int points, float maxDist, float fromLat, float fromLon, float max_depth)
{
    extern __shared__ float s_stations[];
    __shared__ float s_travel_table[SHARED_TRAVEL_TABLE_SIZE];
    __shared__ float s_results[BLOCK_HYPOCS * HYPOCENTER_FILEDS];

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    float depth = max_depth * (blockIdx.y / (float)(gridDim.y - 1.0f)); 
        
    for(int tt_iteration = 0; tt_iteration < ceilf(SHARED_TRAVEL_TABLE_SIZE / static_cast<float>(blockDim.x)); tt_iteration++) {
        int s_index = tt_iteration * blockDim.x + threadIdx.x;
        if(s_index < SHARED_TRAVEL_TABLE_SIZE){
            s_travel_table[s_index] = travel_table[blockIdx.y * SHARED_TRAVEL_TABLE_SIZE + s_index];
        }
    }
    
    for(int station_iteration = 0; station_iteration < ceilf(static_cast<float>(station_count * STATION_FILEDS) / blockDim.x); station_iteration++){
        int index = station_iteration * blockDim.x + threadIdx.x;

        if(index < station_count * STATION_FILEDS) {
            s_stations[index] = stations[index];
        }
    }

    __syncthreads();

    if(index >= points){
        return;
    }

    float last_origin = 0.0f;
    float final_origin = 0.0f;
    float err = 0.0;
    float correct = 0;

    int station_index = threadIdx.x % station_count;
    for(int i = 0; i < station_count; i++, station_index++) {
        if(station_index >= station_count){
            station_index = 0;
        }
        float ang_dist = station_distances[index + i * points];
        float s_pwave = s_stations[station_index + 3 * station_count];
        float expected_travel_time = table_interpolate(s_travel_table, ang_dist);
        float predicted_origin = s_pwave - expected_travel_time;

        if(i > 0){
            float _err = predicted_origin - last_origin;
            err += _err * _err;
            if(_err < 5) {
                correct++;
            }
        }

        if(i == station_count / 2){
            final_origin = predicted_origin;
        }

        last_origin = predicted_origin;
    }

    s_results[threadIdx.x + blockDim.x * 0] = correct;
    s_results[threadIdx.x + blockDim.x * 1] = err;
    s_results[threadIdx.x + blockDim.x * 2] = index;
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

        /*printf("TBLCKS %f %f %f %f %f\n", 
            s_results[0 * blockDim.x],
            s_results[1 * blockDim.x],
            s_results[2 * blockDim.x],
            s_results[3 * blockDim.x],
            s_results[4 * blockDim.x]
        );*/
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

__global__ void calculate_station_distances(float* station_distances, float* point_locations, float* stations, int station_count, int points, float maxDist, float fromLat, float fromLon){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index >= points){
        return;
    }

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
        float ang_dist = haversine(lat, lon, s_lat, s_lon) * 180.0f / M_PIf; // because travel table is in degrees
        station_distances[index + i * points] = ang_dist; // special precomputed value 
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

bool run_hypocenter_search(float* stations, size_t station_count, size_t points, float depth_resolution, float maxDist, float fromLat, float fromLon, float* final_result)
{
    float* d_stations;
    float* d_stations_distances;
    float* d_point_locations;
    float* d_temp_results;

    if(points < 2){
        printf("ERR!! at least 2 points needed!\n");
        return false;
    }

    bool success = true;
    
    dim3 blocks = {(unsigned int)ceil(static_cast<float>(points) / BLOCK_HYPOCS), (unsigned int)ceil(max_depth / depth_resolution) + 1, 1};
    dim3 threads = {BLOCK_HYPOCS, 1, 1};

    if(blocks.y < 2){
        printf("ERR!! at least 2 depth points needed!\n");
        return false;
    }
    
    size_t station_array_size = sizeof(float) * station_count * STATION_FILEDS;
    size_t station_distances_array_size = sizeof(float) * station_count * points;
    size_t point_locations_array_size = sizeof(float) * 2 * points;
    size_t temp_results_array_elements = ceil((blocks.x * blocks.y * blocks.z) / static_cast<float>(BLOCK_REDUCE));
    size_t table_size = sizeof(float) * blocks.y * SHARED_TRAVEL_TABLE_SIZE;

    printf("station array size (%ld stations) %.2fkB\n", station_count, station_array_size / (1024.0));
    printf("station distances array size %.2fkB\n", station_distances_array_size / (1024.0));
    printf("point locations array size %.2fkB\n", point_locations_array_size / (1024.0));
    printf("temp results array size %.2fkB\n", (sizeof(float) * HYPOCENTER_FILEDS * temp_results_array_elements) / (1024.0));
    printf("travel time table array size %.2fkB\n", (table_size) / (1024.0));
    
    success &= cudaMalloc(&d_stations, station_array_size) == cudaSuccess;
    success &= cudaMemcpy(d_stations, stations, station_array_size, cudaMemcpyHostToDevice) == cudaSuccess;
    success &= cudaMalloc(&d_stations_distances, station_distances_array_size) == cudaSuccess;
    success &= cudaMalloc(&d_point_locations, point_locations_array_size) == cudaSuccess;

    printf("%d %d %d\n", blocks.x, blocks.y, blocks.z);
    printf("%d %d %d\n", threads.x, threads.y, threads.z);
    printf("total points: %lld\n", (((long long)(blocks.x * blocks.y * blocks.z)) * (long long)(threads.x * threads.y * threads.z)));

    const int block_count2 = ceil(static_cast<float>(points) / BLOCK_DISTANCES);

    if(success) calculate_station_distances<<<block_count2, BLOCK_DISTANCES>>>
        (d_stations_distances, d_point_locations, d_stations, station_count, points, maxDist, fromLat, fromLon);

    // in the meantime we can prepare travel table

    cudaError err;
    
    success &= cudaMalloc(&travel_table_device, table_size) == cudaSuccess;
    float* fitted_travel_table = static_cast<float*>(malloc(table_size));

    if(fitted_travel_table == nullptr){
        success = false;
        perror("malloc");
    } else {
        printf("Prepare table\n");
        prepare_travel_table(fitted_travel_table, blocks.y);
        success &= (err = cudaMemcpy(travel_table_device, fitted_travel_table, table_size, cudaMemcpyHostToDevice)) == cudaSuccess;
        if(!success){
            printf("memcpy %s\n", cudaGetErrorString(err));
        }
    }
    
    success &= cudaDeviceSynchronize() == cudaSuccess;
    
    success &= (err = cudaGetLastError()) == cudaSuccess;
    if(err != cudaSuccess){
        printf("ERROR IN STATD!!!! %s\n", cudaGetErrorString(err));
    }
    
    if(success) evaluateHypocenter<<<blocks, threads, sizeof(float) * STATION_FILEDS * station_count>>>
        (f_results_device, travel_table_device, d_stations, d_stations_distances, station_count, points, maxDist, fromLat, fromLon, max_depth);

    success &= cudaMalloc(&d_temp_results, sizeof(float) * HYPOCENTER_FILEDS * temp_results_array_elements) == cudaSuccess;
    success &= cudaDeviceSynchronize() == cudaSuccess;
    
    success &= (err = cudaGetLastError()) == cudaSuccess;
    if(err != cudaSuccess){
        printf("ERROR IN HYPOCS!!!! %s\n", cudaGetErrorString(err));
    }

    size_t current_result_count = blocks.x * blocks.y * blocks.z;
    while(success && current_result_count > 1){
        dim3 blcks = {(unsigned int)ceil(current_result_count / static_cast<double>(BLOCK_REDUCE)), 1, 1};

        printf("REDUCING [%d]... now %ld to %d\n",success, current_result_count, blcks.x);
        
        results_reduce<<<blcks, BLOCK_REDUCE>>>(d_temp_results, f_results_device, current_result_count);
        success &= cudaDeviceSynchronize() == cudaSuccess;

        success &= (err = cudaGetLastError()) == cudaSuccess;
        if(err != cudaSuccess){
            printf("ERROR IN REDUCE!!!! %s\n", cudaGetErrorString(err));
        }

        current_result_count = blcks.x;

        /**
         * PRELIMINARY_HYPOCENTER:
         * correct | err | index | origin | depth
         * 
         * RESULT_HYPOCENTER:
         * lat, lon, depth, origin
        */

        float local_result[HYPOCENTER_FILEDS];

        if(current_result_count == 1){
            success &= cudaMemcpy(local_result, d_temp_results, HYPOCENTER_FILEDS * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;

            cudaDeviceSynchronize();

            printf("local %f %f %f %f %f\n", local_result[0], local_result[1], local_result[2], local_result[3], local_result[4]);

            float lat, lon, u_dist;
            calculateParams(points, local_result[2], maxDist, fromLat, fromLon, &lat, &lon, &u_dist);
            final_result[0] = lat;
            final_result[1] = lon;
            final_result[2] = local_result[4];
            final_result[3] = local_result[3];
        } else {
            success &= cudaMemcpy(f_results_device, d_temp_results, current_result_count * HYPOCENTER_FILEDS * sizeof(float), cudaMemcpyDeviceToDevice) == cudaSuccess;
        }

        success &= (err = cudaGetLastError()) == cudaSuccess;
        if(!success){
            printf("ERROR IN MEMCPY!!!! %s\n", cudaGetErrorString(err));
        }
    }


    if(fitted_travel_table) free(fitted_travel_table);
    if(travel_table_device) cudaFree(travel_table_device);
    if(d_stations) cudaFree(d_stations);
    if(d_stations_distances) cudaFree(d_stations_distances);
    if(d_point_locations) cudaFree(d_point_locations);

    printf("Result = %d\n", success);

    return success;
}


JNIEXPORT jfloatArray JNICALL Java_globalquake_jni_GQNativeFunctions_findHypocenter
  (JNIEnv *env, jclass, jfloatArray stations, jfloat fromLat, jfloat fromLon, jlong points, jfloat depthRes, jfloat maxDist) {
    size_t station_count = env->GetArrayLength(stations) / STATION_FILEDS;
    
    float* stationsArray = static_cast<float*>(malloc(sizeof(float) * station_count * STATION_FILEDS));

    bool success = false;

    if(!stationsArray){
        goto cleanup;
    }

    for(int i = 0; i < env->GetArrayLength(stations); i++){        
        stationsArray[i] = env->GetFloatArrayElements(stations, 0)[i];
    }

    printf("run\n");

    float final_result[HYPOCENTER_FILEDS];

    success = run_hypocenter_search(stationsArray, station_count, points, depthRes, maxDist, fromLat, fromLon, final_result);

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

/*
 * Class:     globalquake_jni_GQNativeFunctions
 * Method:    initCUDA
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_globalquake_jni_GQNativeFunctions_initCUDA
      (JNIEnv *, jclass, jlong max_points, jfloat _max_depth_resolution){
    bool success = true;
    max_depth_resolution = _max_depth_resolution;
    
    dim3 blocks = {(unsigned int)ceil(static_cast<float>(max_points) / BLOCK_HYPOCS), (unsigned int)ceil(max_depth / _max_depth_resolution) + 1, 1};
    
    size_t results_size = sizeof(float) * HYPOCENTER_FILEDS * (blocks.x * blocks.y * blocks.z);  

    printf("Results array has size %.2fMB\n", (results_size / (1024.0*1024.0)));
    
    success &= cudaMalloc(&f_results_device, results_size) == cudaSuccess;

    printf("Cuda malloc done\n");

    printf("init result = %d\n", success);
    cuda_initialised = success;
    return success;
}
