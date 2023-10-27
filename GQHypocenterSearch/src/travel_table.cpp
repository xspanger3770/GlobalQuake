#include <jni.h>
#include <stdlib.h>

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
JNIEXPORT jboolean JNICALL Java_globalquake_jni_GQNativeFunctions_isTravelTableReady
  (JNIEnv *, jclass){
    return is_initialised();
  }

/*
 * Class:     globalquake_jni_GQNativeFunctions
 * Method:    initPTravelTable
 * Signature: ([[FF)V
 */
JNIEXPORT jboolean JNICALL Java_globalquake_jni_GQNativeFunctions_copyPTravelTable(JNIEnv *env, jclass cls, jobjectArray table, jfloat d) {
    max_depth = d;

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
            p_wave_table[i * len1 + j] = element[j];
        }
    }

    releaseMatrixArray(env, table);

    travel_table_initialised = true;
    return true;
}
