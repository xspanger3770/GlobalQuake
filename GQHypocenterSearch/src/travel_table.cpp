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

/*
 * Class:     globalquake_jni_GQNativeFunctions
 * Method:    isInitialized
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_globalquake_jni_GQNativeFunctions_isInitialized
  (JNIEnv *, jclass){
    return is_initialised();
  }

/*
 * Class:     globalquake_jni_GQNativeFunctions
 * Method:    querryTable
 * Signature: (DD)F
 */
JNIEXPORT jfloat JNICALL Java_globalquake_jni_GQNativeFunctions_querryTable
  (JNIEnv *env, jclass, jdouble ang, jdouble depth){
    int i = depth / max_depth * (table_height - 1);
    int j = ang / MAX_ANG * (table_width - 1);
    float val = p_wave_table[i * table_width + j];
    return val;
  }

/*
 * Class:     globalquake_jni_GQNativeFunctions
 * Method:    initPTravelTable
 * Signature: ([[FF)V
 */
JNIEXPORT void JNICALL Java_globalquake_jni_GQNativeFunctions_initPTravelTable(JNIEnv *env, jclass cls, jobjectArray table, jfloat d) {
    max_depth = d;

    int len1 = env->GetArrayLength(table);
    jfloatArray dim =  (jfloatArray)env->GetObjectArrayElement(table, 0);
    int len2 = env->GetArrayLength(dim);
    
    table_height = len1;
    table_width = len2;
    
    if(is_initialised()){
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

    travel_time_initialised = true;
}
