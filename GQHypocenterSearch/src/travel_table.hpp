#ifndef _TRAVEL_TABLE_H
#define _TRAVEL_TABLE_H

#include <vector>
#include <memory>
#include <iostream>
#include <iomanip>

const float P_S_MAX_ANGLE = 150.0;
const float PKP_MIN_ANGLE = 140.0;
const float PKP_MAX_ANGLE = 180.0;

using table = std::vector<std::vector<float>>;

class TravelTable {
public:
    float maxDepth;
    float depthResolution;
    float angularResolution;

    table p_wave;
    table s_wave;
    table pkp_wave;
    
    TravelTable(float maxDepth, float depthResolution, float angularResolution) : 
        maxDepth{maxDepth}, depthResolution{depthResolution}, angularResolution{angularResolution} {
            resizeTable(p_wave, 0, P_S_MAX_ANGLE);
            resizeTable(s_wave, 0, P_S_MAX_ANGLE);
            resizeTable(pkp_wave, PKP_MIN_ANGLE, PKP_MAX_ANGLE);
        };
    
    void resizeTable(table& table, double minAngle, double maxAngle){
        int size_x = static_cast<int>((maxAngle - minAngle) / angularResolution) + 1;
        int size_y = static_cast<int>(maxDepth / depthResolution) + 1;
        table.resize(size_x);
        for(int i = 0; i < size_x; i++){
            table[i].resize(size_y);
        }

        std::cout << "Travel time table created of size " << std::fixed << std::setprecision(2) << (size_x * size_y * sizeof(float)) / (1024 * 1024.0)<< "MB" << std::endl;
    }
};

extern std::unique_ptr<TravelTable> loaded_travel_table;

int createTravelTable(float maxDepth, float depthResolution, float angularResolution);

#endif