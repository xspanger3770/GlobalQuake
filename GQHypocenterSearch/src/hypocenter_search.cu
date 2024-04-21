#include <cuda_profiler_api.h>
#include <iostream>
#include <jni.h>
#include <stdio.h>

#include "globalquake.hpp"
#include "globalquake_jni_GQNativeFunctions.h"
#include "travel_table.hpp"

#define BLOCK_REDUCE 256
#define BLOCK_DISTANCES 64
#define SHARED_TRAVEL_TABLE_SIZE 256

#define STATION_FILEDS 4
#define HYPOCENTER_FILEDS 4
/**
 * STATION:
 * lat | lon | alt | pwave
 * 
 * PRELIMINARY_HYPOCENTER (STEP 1):
 * heuristic | position | origin
 * 
 * PRELIMINARY_HYPOCENTER (STEP 2):
 * heuristic | index (int) | depth | origin
 * 
 * RESULT_HYPOCENTER:
 * lat, lon, depth, origin
*/

#define MAX_ANG_VIRTUAL (181.0f)
#define PHI2 2.618033989f
#define PI 3.14159256f

struct depth_profile_t
{
    float depth_resolution;
    float *device_travel_table;
};

bool cuda_initialised = false;
float max_depth_resolution;

int depth_profile_count;
depth_profile_t *depth_profiles = nullptr;
float *f_results_device = nullptr;

size_t total_travel_table_size;

void print_err(const char *msg) {
    cudaError err = cudaGetLastError();
    TRACE(2, "%s failed: %s (%d)\n", msg, cudaGetErrorString(err), err);
}

__host__ void move_on_globe(float from_lat, float from_lon, float angle, float angular_distance, float *lat, float *lon) {
    // calculate angles
    float delta = angular_distance;
    float theta = from_lat;
    float phi = from_lon;
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

__device__ void move_on_globe_device(float from_lat, float from_lon, float angle, float angular_distance, float *lat, float *lon) {
    // calculate angles
    float delta = angular_distance;
    float theta = from_lat;
    float phi = from_lon;
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
__device__ float haversine(float lat1, float lon1, float lat2, float lon2) {
    float dlat = lat2 - lat1;
    float dlon = lon2 - lon1;

    // Haversine formula
    float v1 = __sinf(dlat / 2.0f);
    float v2 = __sinf(dlon / 2.0f);
    float a = v1 * v1 + __cosf(lat1) * __cosf(lat2) * v2 * v2;

    float y = __fsqrt_rn(a);
    float x = __fsqrt_rn(1.0f - a);
    float c = atan2f(y, x); // bottleneck of station distances calculation

    return c * 2.0f; // Angular distance in radians
}

// everything in radians
void calculate_params(int points, int index, float max_dist, float from_lat, float from_lon, float *lat, float *lon, float *dist) {
    float ang = (2.0f * PI * (float) index) / PHI2;
    *dist = sqrtf(index) * (max_dist / sqrtf(points - 1.0f));
    move_on_globe(from_lat, from_lon, ang, *dist, lat, lon);
}

__device__ void calculate_params_device(int points, int index, float max_dist, float from_lat, float from_lon, float *lat, float *lon, float *dist) {
    float ang = (2.0f * PI * (float) index) / PHI2;
    *dist = __fsqrt_rn(index) * (max_dist / __fsqrt_rn(points - 1.0f));
    move_on_globe_device(from_lat, from_lon, ang, *dist, lat, lon);
}

__device__ float travel_table_interpolate(float *s_travel_table, float index) {
    int index1 = (int) index;
    int index2 = index1 + 1;

    float t = index - index1;
    return (1.0f - t) * s_travel_table[index1] + t * s_travel_table[index2];
}

__device__ inline float *hypocenter_heuristic(float *hypocenter, int grid_size) {
    return &hypocenter[0 * grid_size];
}

__device__ inline float *hypocenter_index(float *hypocenter, int grid_size) {
    return &hypocenter[1 * grid_size];
}

__device__ inline float *hypocenter_depth_index(float *hypocenter, int grid_size) {
    return &hypocenter[2 * grid_size];
}

__device__ inline float *hypocenter_origin(float *hypocenter, int grid_size) {
    return &hypocenter[3 * grid_size];
}

__device__ inline float heuristic(float correct, float err) {
    return (correct * correct) / (err * err);
}

__device__ void reduce(float *hypocenter_a, float *hypocenter_b, int grid_size) {
    float heuristic_a = *hypocenter_heuristic(hypocenter_a, grid_size);
    float heuristic_b = *hypocenter_heuristic(hypocenter_b, grid_size);

    bool swap = heuristic_b > heuristic_a;

    if (swap) {
        *hypocenter_heuristic(hypocenter_a, grid_size) = *hypocenter_heuristic(hypocenter_b, grid_size);
        *hypocenter_depth_index(hypocenter_a, grid_size) = *hypocenter_depth_index(hypocenter_b, grid_size);
        *hypocenter_index(hypocenter_a, grid_size) = *hypocenter_index(hypocenter_b, grid_size);
        *hypocenter_origin(hypocenter_a, grid_size) = *hypocenter_origin(hypocenter_b, grid_size);
    }
}

__global__ void evaluate_hypocenter(float *results,
        float *travel_table,
        float *stations,
        float *station_distances,
        float *station_distances_across,
        int station_count,
        int points,
        float max_dist,
        float p_wave_threshold) {
    extern __shared__ float s_stations[];
    __shared__ float s_travel_table[SHARED_TRAVEL_TABLE_SIZE * TILE];
    __shared__ float s_results[BLOCK_HYPOCS * HYPOCENTER_FILEDS];

    int point_index = blockIdx.x * blockDim.x + threadIdx.x;


    for (int station_iteration = 0; station_iteration < ceilf(static_cast<float>(station_count * 1) / blockDim.x); station_iteration++) {
        int index = station_iteration * blockDim.x + threadIdx.x;

        if (index < station_count * 1) {
            s_stations[index] = stations[index + 3 * station_count]; // we care only P wave
        }
    }

    for(int tile = 0; tile < TILE; tile++) {
        for (int tt_iteration = 0; tt_iteration < ceilf((SHARED_TRAVEL_TABLE_SIZE) / static_cast<float>(blockDim.x)); tt_iteration++) {
            int s_index = tt_iteration * blockDim.x + threadIdx.x;
            if (s_index < SHARED_TRAVEL_TABLE_SIZE) {
                s_travel_table[tile *SHARED_TRAVEL_TABLE_SIZE + s_index] = travel_table[(blockIdx.y * TILE + tile) * SHARED_TRAVEL_TABLE_SIZE + s_index];
            }
        }
    }

    __syncthreads();

    float origins[TILE];

    int j = (point_index) % station_count;

    // trick with changing station that is being used for origin calculation
    {
        float ang_dist = station_distances_across[point_index];
        float s_pwave = s_stations[j];

        for(int tile = 0; tile < TILE; tile++) {
            float expected_travel_time = travel_table_interpolate(&s_travel_table[tile * SHARED_TRAVEL_TABLE_SIZE], ang_dist);
            float predicted_origin = s_pwave - expected_travel_time;

            origins[tile] = predicted_origin;
        }
    }
    
    float err[TILE];
    float correct[TILE];
    for(int tile = 0; tile < TILE; tile++) {
        err[tile] = 0.0f;
        correct[tile] = 0.0f;
    }

    for (int i = 0; i < station_count; i++) {
        float ang_dist = station_distances[point_index + i * points];
        float s_pwave = s_stations[i];

        for(int tile = 0; tile < TILE; tile++) {
            float expected_travel_time = travel_table_interpolate(&s_travel_table[tile * SHARED_TRAVEL_TABLE_SIZE], ang_dist);
            float predicted_origin = s_pwave - expected_travel_time;

            float _err = fabsf(predicted_origin - origins[tile]);
            correct[tile] += fmaxf(0.0f, p_wave_threshold - _err); // divide by p_wave_threshold at the end! ! actually we dont have to
            err[tile] += _err;
        }
    }

    int best_tile = 0;
    float best_heuristic = heuristic(correct[0], err[0]);
    
    #if TILE > 1
    for(int tile = 1; tile < TILE; tile++) {
        float h = heuristic(correct[tile], err[tile]);
        if(h > best_heuristic){
            best_heuristic = h;
            best_tile = tile;
        }
    }
    #endif
    
    float depth = blockIdx.y * TILE + best_tile;
            
    s_results[threadIdx.x + blockDim.x * 0] = best_heuristic;
    *(int *) (&s_results[threadIdx.x + blockDim.x * 1]) = point_index;
    s_results[threadIdx.x + blockDim.x * 2] = depth;
    s_results[threadIdx.x + blockDim.x * 3] = origins[best_tile];

    __syncthreads();

    // implementation 3 from slides
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && blockDim.x * blockIdx.x + threadIdx.x + s < points) {
            reduce(&s_results[threadIdx.x], &s_results[threadIdx.x + s], blockDim.x);
            __syncthreads();
        }
    }

    if (threadIdx.x == 0) {
        int idx = (blockIdx.y) * gridDim.x + blockIdx.x;

        results[idx + 0 * (gridDim.x * gridDim.y)] = s_results[0 * blockDim.x]; // heuristic
        results[idx + 1 * (gridDim.x * gridDim.y)] = s_results[1 * blockDim.x]; // point_index
        results[idx + 2 * (gridDim.x * gridDim.y)] = s_results[2 * blockDim.x]; // depth
        results[idx + 3 * (gridDim.x * gridDim.y)] = s_results[3 * blockDim.x]; // origin
    }
}

__global__ void results_reduce(float *out, float *in, int total_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_size) {
        return;
    }
    __shared__ float s_results[HYPOCENTER_FILEDS * BLOCK_REDUCE];

    s_results[threadIdx.x + BLOCK_REDUCE * 0] = in[index + total_size * 0];
    s_results[threadIdx.x + BLOCK_REDUCE * 1] = in[index + total_size * 1];
    s_results[threadIdx.x + BLOCK_REDUCE * 2] = in[index + total_size * 2];
    s_results[threadIdx.x + BLOCK_REDUCE * 3] = in[index + total_size * 3];
    __syncthreads();

    // implementation 3 from slides
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && blockDim.x * blockIdx.x + threadIdx.x + s < total_size) {
            reduce(&s_results[threadIdx.x], &s_results[threadIdx.x + s], blockDim.x);
            __syncthreads();
        }
    }

    if (threadIdx.x == 0) {
        int idx = blockIdx.y * gridDim.x + blockIdx.x;
        out[idx + 0 * (gridDim.x * gridDim.y)] = s_results[0 * blockDim.x];
        out[idx + 1 * (gridDim.x * gridDim.y)] = s_results[1 * blockDim.x];
        out[idx + 2 * (gridDim.x * gridDim.y)] = s_results[2 * blockDim.x];
        out[idx + 3 * (gridDim.x * gridDim.y)] = s_results[3 * blockDim.x];
    }
}

const float ANGLE_TO_INDEX = (SHARED_TRAVEL_TABLE_SIZE - 1.0f) / MAX_ANG_VIRTUAL;

__global__ void precompute_station_distances(
        float *station_distances, float* station_distances_across, float *stations, int station_count, int points, float max_dist, float from_lat, float from_lon) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= points) {
        return;
    }

    float lat, lon, dist;

    calculate_params_device(points, index, max_dist, from_lat, from_lon, &lat, &lon, &dist);

    int j = index % station_count;

    for (int i = 0; i < station_count; i++) {        
        float s_lat = stations[i + 0 * station_count];
        float s_lon = stations[i + 1 * station_count];
        float ang_dist = haversine(lat, lon, s_lat, s_lon) * 180.0f / PI;  // because travel table is in degrees
        float ang_index = ang_dist * ANGLE_TO_INDEX; // precompute;
        station_distances[index + i * points] = ang_index;

        if( i == j ) {
            station_distances_across[index] = ang_index;
        }
    }
}

void prepare_travel_table(float *fitted_travel_table, int rows) {
    for (int row = 0; row < rows; row++) {
        for (int column = 0; column < SHARED_TRAVEL_TABLE_SIZE; column++) {
            fitted_travel_table[row * SHARED_TRAVEL_TABLE_SIZE + column] =
                    p_wave_interpolate(column / (SHARED_TRAVEL_TABLE_SIZE - 1.0) * MAX_ANG_VIRTUAL, (row / (rows - 1.0)) * table_max_depth);
        }
    }
}

// returns (accurately) estimated total GPU memory allocation size given the parameters
size_t get_total_allocation_size(size_t points, size_t station_count, float depth_resolution) {
    size_t result = total_travel_table_size;

    dim3 blocks = { (unsigned int) ceil(static_cast<float>(points) / BLOCK_HYPOCS), (unsigned int) ceil(table_max_depth / (depth_resolution * TILE)) + 1, 1 };

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

bool run_hypocenter_search(float *stations,
        size_t station_count,
        size_t points,
        int depth_profile_index,
        float max_dist,
        float from_lat,
        float from_lon,
        float *final_result,
        float p_wave_threshold) {
    if (depth_profile_index < 0 || depth_profile_index >= depth_profile_count) {
        TRACE(2, "Error! Invalid depth profile index: %d!\n", depth_profile_index);
        return false;
    }

    depth_profile_t *depth_profile = &depth_profiles[depth_profile_index];

    float *device_stations;
    float *device_stations_distances;
    float *device_stations_distances_across;
    float *device_temp_results;

    if (points < 2) {
        TRACE(2, "Error! at least 2 points needed!\n");
        return false;
    }

    if (station_count < 3) {
        TRACE(2, "Error! at least 3 stations needed!\n");
        return false;
    }

    points += (BLOCK_HYPOCS - points % BLOCK_HYPOCS);

    bool success = true;

    dim3 blocks = {
        (unsigned int) ceil(static_cast<float>(points) / BLOCK_HYPOCS), (unsigned int) ceil(table_max_depth / (depth_profile->depth_resolution * TILE)) + 1, 1
    };
    dim3 threads = { BLOCK_HYPOCS, 1, 1 };

    if (blocks.y < 2) {
        TRACE(2, "Error! at least 2 depth points needed!\n");
        return false;
    }

    size_t station_array_size = sizeof(float) * station_count * STATION_FILEDS;
    size_t station_distances_array_size = sizeof(float) * station_count * points;
    size_t station_distances_array_size_across = sizeof(float) * points;
    size_t results_size = sizeof(float) * HYPOCENTER_FILEDS * (blocks.x * (blocks.y) * blocks.z);

    size_t temp_results_array_elements = ceil((blocks.x * (blocks.y ) * blocks.z) / static_cast<float>(BLOCK_REDUCE));
    size_t current_result_count = blocks.x * (blocks.y) * blocks.z;

    const int block_count = ceil(static_cast<float>(points) / BLOCK_DISTANCES);

    TRACE(1, "Station array size (%ld stations) %.2fkB\n", station_count, station_array_size / (1024.0));
    TRACE(1, "Station distances array size %.2fMB\n", station_distances_array_size / (1024.0 * 1024.0));
    TRACE(1, "Temp results array size %.2fkB\n", (sizeof(float) * HYPOCENTER_FILEDS * temp_results_array_elements) / (1024.0));
    TRACE(1, "Results array has size %.2fMB\n", (results_size / (1024.0 * 1024.0)));

    success &= cudaMalloc(&device_stations, station_array_size) == cudaSuccess;
    success &= cudaMemcpy(device_stations, stations, station_array_size, cudaMemcpyHostToDevice) == cudaSuccess;
    success &= cudaMalloc(&device_stations_distances, station_distances_array_size) == cudaSuccess;
    success &= cudaMalloc(&device_stations_distances_across, station_distances_array_size_across) == cudaSuccess;
    success &= cudaMalloc(&device_temp_results, sizeof(float) * HYPOCENTER_FILEDS * temp_results_array_elements) == cudaSuccess;
    success &= cudaMalloc(&f_results_device, results_size) == cudaSuccess;

    if (!success) {
        print_err("Hypocenter search initialisation");
        goto cleanup;
    }

    TRACE(1, "Grid size: %d %d %d\n", blocks.x, blocks.y, blocks.z);
    TRACE(1, "Block size: %d %d %d\n", threads.x, threads.y, threads.z);
    TRACE(1, "Total points: %lld\n", (((long long) (blocks.x * blocks.y * blocks.z)) * (long long) (threads.x * threads.y * threads.z)));

    if (success) {
        precompute_station_distances<<<block_count, BLOCK_DISTANCES>>>(
                device_stations_distances, device_stations_distances_across, device_stations, station_count, points, max_dist, from_lat, from_lon);
    }

    success &= cudaDeviceSynchronize() == cudaSuccess;

    if (!success) {
        print_err("Calculate station distances");
        goto cleanup;
    }

    if (success) {
        evaluate_hypocenter<<<blocks, threads, sizeof(float) * station_count>>>(f_results_device,
                depth_profile->device_travel_table,
                device_stations,
                device_stations_distances,
                device_stations_distances_across,
                station_count,
                points,
                max_dist,
                p_wave_threshold);
    }

    success &= cudaDeviceSynchronize() == cudaSuccess;

    if (!success) {
        print_err("Hypocenter search");
        goto cleanup;
    }

    while (success && current_result_count > 1) {
        dim3 blocks_reduce = { (unsigned int) ceil(current_result_count / static_cast<double>(BLOCK_REDUCE)), 1, 1 };
        TRACE(1, "Reducing... from %ld to %d\n", current_result_count, blocks_reduce.x);

        results_reduce<<<blocks_reduce, BLOCK_REDUCE>>>(device_temp_results, f_results_device, current_result_count);
        success &= cudaDeviceSynchronize() == cudaSuccess;

        if (!success) {
            print_err("Reduce");
            goto cleanup;
        }

        current_result_count = blocks_reduce.x;

        float local_result[HYPOCENTER_FILEDS];

        if (current_result_count == 1) {
            success &= cudaMemcpy(local_result, device_temp_results, HYPOCENTER_FILEDS * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;

            float lat, lon, u_dist;
            calculate_params(points, *(int *) &local_result[1], max_dist, from_lat, from_lon, &lat, &lon, &u_dist);

            double depth = table_max_depth * (local_result[2] / (float) (blocks.y * TILE - 1.0f));

            final_result[0] = lat; // lat
            final_result[1] = lon; // lon
            final_result[2] = depth;
            final_result[3] = local_result[3]; // origin
        } else {
            success &= cudaMemcpy(f_results_device, device_temp_results, current_result_count * HYPOCENTER_FILEDS * sizeof(float), cudaMemcpyDeviceToDevice) ==
                    cudaSuccess;
        }

        if (!success) {
            print_err("CUDA memcpy");
            goto cleanup;
        }
    }

cleanup:

    if (device_stations) {
        success &= cudaFree(device_stations) == cudaSuccess;
    }
    if (device_stations_distances_across) {
        success &= cudaFree(device_stations_distances_across) == cudaSuccess;
    }
    if (device_stations_distances) {
        success &= cudaFree(device_stations_distances) == cudaSuccess;
    }
    if (device_temp_results) {
        success &= cudaFree(device_temp_results) == cudaSuccess;
    }
    if (f_results_device) {
        success &= cudaFree(f_results_device) == cudaSuccess;
    }

    return success;
}

JNIEXPORT jfloatArray JNICALL Java_globalquake_jni_GQNativeFunctions_findHypocenter(JNIEnv *env,
        jclass,
        jfloatArray stations,
        jfloat from_lat,
        jfloat from_lon,
        jlong points,
        int depth_resolution_profile_id,
        jfloat max_dist,
        jfloat p_wave_threshold) {
    size_t station_count = env->GetArrayLength(stations) / STATION_FILEDS;

    float *stations_array = static_cast<float *>(malloc(sizeof(float) * station_count * STATION_FILEDS));
    if (!stations_array) {
        perror("malloc");
        return nullptr;
    }

    jfloat *elements = env->GetFloatArrayElements(stations, 0);
    for (int i = 0; i < station_count * STATION_FILEDS; i++) {
        stations_array[i] = elements[i];
    }

    env->ReleaseFloatArrayElements(stations, elements, 0);

    float final_result[HYPOCENTER_FILEDS];

    bool success = run_hypocenter_search(
            stations_array, station_count, points, depth_resolution_profile_id, max_dist, from_lat, from_lon, final_result, p_wave_threshold);

    free(stations_array);

    jfloatArray result = nullptr;

    if (success) {
        result = env->NewFloatArray(4);

        if (result != nullptr) {
            env->SetFloatArrayRegion(result, 0, 4, final_result);
        }
    }

    return result;
}

bool init_depth_profiles(float *resols, int count) {
    max_depth_resolution = table_max_depth;
    depth_profile_count = count;

    depth_profiles = static_cast<depth_profile_t *>(malloc(count * sizeof(depth_profile_t)));
    if (depth_profiles == nullptr) {
        perror("malloc");
        return false;
    }

    total_travel_table_size = 0;

    for (int i = 0; i < depth_profile_count; i++) {
        float depth_resolution = resols[i];
        if (depth_resolution < max_depth_resolution) {
            max_depth_resolution = depth_resolution;
        }

        depth_profiles[i].depth_resolution = depth_resolution;

        int rows = (unsigned int) ceil(table_max_depth / depth_resolution) + 1;
        size_t table_size = sizeof(float) * rows * SHARED_TRAVEL_TABLE_SIZE;
        total_travel_table_size += table_size;

        TRACE(1, "Creating depth profile with resolution %.2fkm (%.2fkB)\n", depth_resolution, table_size / 1024.0);

        // todo fitted array
        if (cudaMalloc(&depth_profiles[i].device_travel_table, table_size) != cudaSuccess) {
            print_err("CUDA malloc");
            return false;
        }

        float *fitted_travel_table = static_cast<float *>(malloc(table_size));

        if (fitted_travel_table == nullptr) {
            perror("malloc");
            return false;
        } else {
            prepare_travel_table(fitted_travel_table, rows);
            if (cudaMemcpy(depth_profiles[i].device_travel_table, fitted_travel_table, table_size, cudaMemcpyHostToDevice) != cudaSuccess) {
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
JNIEXPORT jboolean JNICALL Java_globalquake_jni_GQNativeFunctions_initCUDA(JNIEnv *env, jclass, jfloatArray depth_profiles_array) {
    bool success = true;

    if (depth_profiles_array != nullptr && depth_profiles == nullptr) {
        int depth_profile_count = env->GetArrayLength(depth_profiles_array);
        jfloat *depth_resolutions_array = env->GetFloatArrayElements(depth_profiles_array, 0);

        float depth_resolutions[depth_profile_count];
        for (int i = 0; i < depth_profile_count; i++) {
            depth_resolutions[i] = depth_resolutions_array[i];
        }

        env->ReleaseFloatArrayElements(depth_profiles_array, depth_resolutions_array, 0);

        success &= init_depth_profiles(depth_resolutions, depth_profile_count);
    }

    cuda_initialised = success;
    return success;
}
