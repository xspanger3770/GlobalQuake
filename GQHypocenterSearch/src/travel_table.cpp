#include <jni.h>
#include <stdlib.h>

#include "travel_table.hpp"
#include "globalquake_jni_GQNativeFunctions.h"

float* p_wave_table;
int32_t table_width;
int32_t table_height;
float max_depth;
bool travel_time_initialised = false;

bool is_initialised(void){
    return travel_time_initialised;
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

JNIEXPORT void JNICALL Java_globalquake_jni_GQNativeFunctions_initPTravelTable(JNIEnv *env, jclass cls, jobjectArray table, jint w, jint h, jfloat d) {
    table_width = w;
    table_height = h;
    max_depth = d;

    int len1 = env->GetArrayLength(table);
    jfloatArray dim =  (jfloatArray)env->GetObjectArrayElement(table, 0);
    int len2 = env->GetArrayLength(dim);
    
    if(travel_time_initialised){
        free(p_wave_table);
    }
    
    p_wave_table = static_cast<float*>(malloc(sizeof(float) * len1 * len2));
    if(p_wave_table == nullptr){
        perror("malloc");
        return;
    }

    for(int i=0; i<len1; ++i){
        jfloatArray oneDim = (jfloatArray)env->GetObjectArrayElement(table, i);
        jfloat *element = env->GetFloatArrayElements(oneDim, 0);

        for(int j=0; j<len2; ++j) {
            p_wave_table[i * len1 + j]= element[j];
        }
    }

    releaseMatrixArray(env, table);
}
