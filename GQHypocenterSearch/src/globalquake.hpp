#ifndef _GLOBALQUAKE_H
#define _GLOBALQUAKE_H

#include "cstddef"

#define TRACE_LEVEL 2
#define TRACE(p, x, ...)              \
    do {                              \
        if (p >= TRACE_LEVEL)         \
            printf(x, ##__VA_ARGS__); \
    } while (0)

bool run_hypocenter_search(float *stations,
        size_t station_count,
        size_t points,
        int depth_resolution_index,
        float max_dist,
        float from_lat,
        float from_lon,
        float *final_result,
        float p_wave_threshold);

bool init_depth_profiles(float *resols, int count);

#endif // _GLOBALQUAKE_H