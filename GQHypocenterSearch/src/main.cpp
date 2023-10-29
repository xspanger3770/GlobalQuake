#include <stdio.h>
#include <stdlib.h>

#include "travel_table.hpp"
#include "globalquake_jni_GQNativeFunctions.h"
#include "globalquake.hpp"

int main()
{
    max_depth = 750.0;
    float depth_resolution = 5.0;

    int len1 = max_depth / depth_resolution + 1;
    int len2 = 1501;

    int points = 100 * 1000;
    
    table_rows = len1;
    table_columns = len2;

    p_wave_table = static_cast<float*>(malloc(sizeof(float) * len1 * len2));

    Java_globalquake_jni_GQNativeFunctions_initCUDA(nullptr, nullptr, points, 5.0);

    int st_c = 64;
    float stations[st_c * 4];

    run_hypocenter_search(stations, st_c, points, 5.0, 10000.0, 0, 0);

    free(p_wave_table);

    return 0;
}