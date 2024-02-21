#ifndef _TRAVEL_TABLE_H
#define _TRAVEL_TABLE_H

#include <cstdint>

#define MAX_ANG 150.0f

extern float *p_wave_travel_table;
extern int32_t table_columns;
extern int32_t table_rows;
extern float table_max_depth;

extern float p_wave_interpolate(float ang, float depth);

#endif