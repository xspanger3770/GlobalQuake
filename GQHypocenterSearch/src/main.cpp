#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>

#include "globalquake.hpp"
#include "globalquake_jni_GQNativeFunctions.h"
#include "travel_table.hpp"
#include <sys/time.h>

void append_to_csv(const char *filename, int best_pps, int block_hypocs, int tile) {
    FILE *file = fopen(filename, "a"); // Open the file in append mode

    // Check if the file was opened successfully
    if (file == NULL) {
        printf("Error opening file %s\n", filename);
        return;
    }

    // Append data to the file in CSV format
    fprintf(file, "%d;%d;%d\n", block_hypocs, tile, best_pps);

    // Close the file
    fclose(file);
}

int main() {
    table_max_depth = 750.0;
    float depth_resolution = 0.5;

    int len1 = table_max_depth / depth_resolution + 1;
    int len2 = 1501;

    int points = 100 * 1000;

    table_rows = len1;
    table_columns = len2;

    p_wave_travel_table = static_cast<float *>(malloc(sizeof(float) * len1 * len2));
    if (!p_wave_travel_table) {
        perror("malloc");
        return 1;
    }

    float resols[] = { depth_resolution };
    if (!init_depth_profiles(resols, 1)) {
        printf("Failure!\n");
        return 1;
    }

    Java_globalquake_jni_GQNativeFunctions_initCUDA(nullptr, nullptr, nullptr);

    int st_c = 50;
    float stations[st_c * 4];

    float a = 999.0;

    static const float RADIANS = 3.14159 / 180.0;

    for (int station = 0; station < st_c; station++) {
        stations[station + 0 * st_c] = (float) rand() / (float) (RAND_MAX / 360.0) * RADIANS;
        stations[station + 1 * st_c] = (float) rand() / (float) (RAND_MAX / 180.0) * RADIANS;
        stations[station + 2 * st_c] = 420.0;
        stations[station + 3 * st_c] = (float) rand() / (float) (RAND_MAX / 10000.0);
    }

    float final_result[4];

    int tests = TESTS;

    double best_pps = 0;

    const char* filename = "../autotune_results.csv";
    
    for(int i = 0; i < tests; i++){
        struct timeval t1, t2;

        gettimeofday(&t1, 0);

        double time = 0.0;

        if (!run_hypocenter_search(stations, st_c, points, 0, 90.0 * RADIANS, 0, 0, final_result, 2.2f)) {
            printf("Error!\n");
            goto cleanup;
        }
        
        gettimeofday(&t2, 0);
        time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;

        double pps = ((points * 1000.0) / time);
        printf("Standard test with %d points (BLOCK=%d, TILE=%d), 50 stations and 0.5km depth resolution: %.1fms @ %.1fpps @ %.1fpscps\n", points, BLOCK_HYPOCS, TILE,time, pps, ((points * 1000.0 * st_c) / time));

        if(pps > best_pps){
            best_pps = pps;
        }
    }

    printf("best: %.2fpps\n", best_pps);

    append_to_csv(filename, best_pps, BLOCK_HYPOCS, TILE);

    printf("allocation size %.2fkB\n", get_total_allocation_size(points, st_c, depth_resolution) / 1024.0);

    cleanup:
    if (p_wave_travel_table) {
        free(p_wave_travel_table);
    }

    return 0;
}