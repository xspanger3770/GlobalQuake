#include <jni.h>
#include <math.h>
#include <stdlib.h>

#include "globalquake_jni_GQNativeFunctions.h"
#include "travel_table.hpp"

float *p_wave_travel_table;
int32_t table_rows;
int32_t table_columns;
float table_max_depth;
bool travel_table_initialised = false;

float p_wave_interpolate(float ang, float depth) {
    float row = (depth / table_max_depth) * (table_rows - 1.0);
    float column = (ang / MAX_ANG) * (table_columns - 1.0);

    int row_floor = fmin(table_rows - 2, floor(row));
    int col_floor = fmin(table_columns - 2, floor(column));
    int row_ceil = row_floor + 1;
    int col_ceil = col_floor + 1;

    float row_frac = row - row_floor;
    float col_frac = column - col_floor;

    float q11 = p_wave_travel_table[row_floor * table_columns + col_floor];
    float q12 = p_wave_travel_table[row_floor * table_columns + col_ceil];
    float q21 = p_wave_travel_table[row_ceil * table_columns + col_floor];
    float q22 = p_wave_travel_table[row_ceil * table_columns + col_ceil];

    float interpolated_value = (1 - row_frac) * ((1 - col_frac) * q11 + col_frac * q12) + row_frac * ((1 - col_frac) * q21 + col_frac * q22);

    return interpolated_value;
}

/*
 * Class:     globalquake_jni_GQNativeFunctions
 * Method:    isInitialized
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_globalquake_jni_GQNativeFunctions_isTravelTableReady(JNIEnv *, jclass) {
    return travel_table_initialised;
}

static void release_matrix(JNIEnv *env, jobjectArray matrix) {
    int size = env->GetArrayLength(matrix);
    for (int i = 0; i < size; i++) {
        jfloatArray array = (jfloatArray) env->GetObjectArrayElement(matrix, i);
        jfloat *elements = env->GetFloatArrayElements(array, 0);

        env->ReleaseFloatArrayElements(array, elements, 0);
        env->DeleteLocalRef(array);
    }
}

/*
 * Class:     globalquake_jni_GQNativeFunctions
 * Method:    initPTravelTable
 * Signature: ([[FF)V
 */
JNIEXPORT jboolean JNICALL Java_globalquake_jni_GQNativeFunctions_copyPTravelTable(JNIEnv *env, jclass cls, jobjectArray table, jfloat max_depth) {
    table_max_depth = max_depth;

    table_rows = env->GetArrayLength(table);
    jfloatArray dim = (jfloatArray) env->GetObjectArrayElement(table, 0);
    table_columns = env->GetArrayLength(dim);

    if (travel_table_initialised) {
        free(p_wave_travel_table);
    }

    p_wave_travel_table = static_cast<float *>(malloc(sizeof(float) * table_rows * table_columns));
    if (p_wave_travel_table == nullptr) {
        perror("malloc");
        return false;
    }

    for (int i = 0; i < table_rows; ++i) {
        jfloatArray oneDim = (jfloatArray) env->GetObjectArrayElement(table, i);
        jfloat *element = env->GetFloatArrayElements(oneDim, 0);

        for (int j = 0; j < table_columns; ++j) {
            p_wave_travel_table[i * table_columns + j] = element[j];
        }
    }

    release_matrix(env, table);

    travel_table_initialised = true;
    return true;
}
