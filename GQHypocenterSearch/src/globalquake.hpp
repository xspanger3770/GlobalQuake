#ifndef _GQ_H
#define _GQ_H

#include "cstddef"

bool run_hypocenter_search(float* stations, size_t station_count, size_t points, float depth_resolution, float maxDist, float fromLat, float fromLon);

#endif