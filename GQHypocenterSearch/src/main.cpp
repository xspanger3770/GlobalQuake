#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "travel_table.hpp"
#include "globalquake_jni_GQNativeFunctions.h"
#include "globalquake.hpp"

int main()
{
    max_depth = 750.0;
    float depth_resolution = 5.0;

    int len1 = max_depth / depth_resolution + 1;
    int len2 = 1501;

    int points = 1000 * 1000;
    
    table_rows = len1;
    table_columns = len2;

    p_wave_table = static_cast<float*>(malloc(sizeof(float) * len1 * len2));
    if(!p_wave_table){
        perror("malloc");
        return 1;
    }

    Java_globalquake_jni_GQNativeFunctions_initCUDA(nullptr, nullptr, points, depth_resolution);

    int st_c = 64;
    float stations[st_c * 4];

    float a = 999.0;

    static const float RADIANS = M_PIf / 360.0;

    for(int station = 0; station < st_c; station++){
        stations[station * 4 + 0] = 50.262 * RADIANS;
        stations[station * 4 + 1] = 17.262 * RADIANS;
        stations[station * 4 + 2] = 420.0;
        stations[station * 4 + 3] = 10;
    }

    float final_result[4];

    if(run_hypocenter_search(stations, st_c, points, depth_resolution, 90.0 * RADIANS, 0, 0, final_result)){
        printf("FINAL RESULT %f %f %f %f\n", final_result[0], final_result[1], final_result[2], final_result[3]);
    }

    if(p_wave_table) free(p_wave_table);

    return 0;
}