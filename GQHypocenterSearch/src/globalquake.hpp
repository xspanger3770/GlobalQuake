#ifndef _GQ_H
#define _GQ_H

#include "cstddef"

#define TRACE_LEVEL 2
#define TRACE(p, x, ...)  \
    do { if(p >= TRACE_LEVEL) printf(x, ##__VA_ARGS__); } while(0)

bool run_hypocenter_search(float* stations, size_t station_count, size_t points, int depth_resolution_index, float maxDist, float fromLat, float fromLon, float* final_result);

bool initDepthProfiles(float* resols, int count);

#endif