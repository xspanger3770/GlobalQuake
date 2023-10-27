#ifndef _TRAVEL_TABLE_H
#define _TRAVEL_TABLE_H

#include <cstdint>

#define MAX_ANG 150.0

extern float* p_wave_table;
extern int32_t table_width;
extern int32_t table_height;
extern float max_depth;

extern bool is_initialised(void);

#endif