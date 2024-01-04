#include <jni.h>
#include <stdlib.h>
#include <math.h>

#include "travel_table.hpp"
#include "globalquake_jni_GQNativeFunctions.h"

float* p_wave_table;
int32_t table_rows;
int32_t table_columns;
float max_depth;
bool travel_table_initialised = false;

bool is_initialised(void){
    return travel_table_initialised;
}

float p_interpolate(float ang, float depth) {
    float row = (depth / max_depth) * (table_rows - 1.0);
    float column = (ang / MAX_ANG) * (table_columns - 1.0);

    int row_floor = floor(row);
    int col_floor = floor(column);
    int row_ceil = fmin(table_rows - 1, row_floor + 1);
    int col_ceil = fmin(table_columns - 1, col_floor + 1);

    //if(row_floor < 0 || col_floor < 0 || row_ceil >= table_rows || col_ceil >= table_columns){
        //printf("%d %d %d %d [%d %d]\n", row_floor, col_floor, row_ceil, col_ceil, table_rows, table_columns);
        //exit(1);
    //}

    
    float row_frac = row - row_floor;
    float col_frac = column - col_floor;

    float q11 = p_wave_table[row_floor * table_columns + col_floor];
    float q12 = p_wave_table[row_floor * table_columns + col_ceil];
    float q21 = p_wave_table[row_ceil * table_columns + col_floor];
    float q22 = p_wave_table[row_ceil * table_columns + col_ceil];

    float interpolated_value = (1 - row_frac) * ((1 - col_frac) * q11 + col_frac * q12) +
                              row_frac * ((1 - col_frac) * q21 + col_frac * q22);

    return interpolated_value;
}

/*
 * Class:     globalquake_jni_GQNativeFunctions
 * Method:    isInitialized
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_globalquake_jni_GQNativeFunctions_isTravelTableReady
  (JNIEnv *, jclass){
    return is_initialised();
  }

static void releaseMatrixArray(JNIEnv *env, jobjectArray matrix) {
    int size = env->GetArrayLength(matrix);
    for (int i = 0; i < size; i++) {
        jfloatArray oneDim = (jfloatArray) env->GetObjectArrayElement(matrix, i);
        jfloat *elements = env->GetFloatArrayElements(oneDim, 0);

        env->ReleaseFloatArrayElements(oneDim, elements, 0);
        env->DeleteLocalRef(oneDim);
    }
}

/*
 * Class:     globalquake_jni_GQNativeFunctions
 * Method:    initPTravelTable
 * Signature: ([[FF)V
 */
JNIEXPORT jboolean JNICALL Java_globalquake_jni_GQNativeFunctions_copyPTravelTable(
        JNIEnv *env, jclass cls, jobjectArray table, jfloat _max_depth) {
    max_depth = _max_depth;

    int len1 = env->GetArrayLength(table);
    jfloatArray dim =  (jfloatArray)env->GetObjectArrayElement(table, 0);
    int len2 = env->GetArrayLength(dim);
    
    table_rows = len1;
    table_columns = len2;
    
    if(is_initialised()){
        free(p_wave_table);
    }
    
    p_wave_table = static_cast<float*>(malloc(sizeof(float) * len1 * len2));
    if(p_wave_table == nullptr){
        perror("malloc");
        return false;
    }

    for(int i=0; i<len1; ++i){
        jfloatArray oneDim = (jfloatArray)env->GetObjectArrayElement(table, i);
        jfloat *element = env->GetFloatArrayElements(oneDim, 0);

        for(int j=0; j<len2; ++j) {
            p_wave_table[i * len2 + j] = element[j];
        }
    }

    releaseMatrixArray(env, table);

    travel_table_initialised = true;
    return true;
}
