#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "globalquake.hpp"
#include "globalquake_jni_GQNativeFunctions.h"
#include "travel_table.hpp"

int main() {
    table_max_depth = 750.0;
    float depth_resolution = 1.0;

    int len1 = table_max_depth / depth_resolution + 1;
    int len2 = 1501;

    int points = 50 * 1000;

    table_rows = len1;
    table_columns = len2;

    p_wave_travel_table = static_cast<float *>(malloc(sizeof(float) * len1 * len2));
    if (!p_wave_travel_table) {
        perror("malloc");
        return 1;
    }

    float resols[] = { 1.0 };
    if (!init_depth_profiles(resols, 1)) {
        printf("Failure!\n");
        return 1;
    }

    Java_globalquake_jni_GQNativeFunctions_initCUDA(nullptr, nullptr, nullptr);

    int st_c = 100;
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

    if (run_hypocenter_search(stations, st_c, points, 0, 90.0 * RADIANS, 0, 0, final_result, 2.2f)) {
        printf("FINAL RESULT %f %f %f %f\n", final_result[0], final_result[1], final_result[2], final_result[3]);
    }

    if (p_wave_travel_table) {
        free(p_wave_travel_table);
    }

    return 0;
}