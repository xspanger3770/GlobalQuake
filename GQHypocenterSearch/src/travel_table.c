#include "travel_table.h"
#include "globalquake_jni_GQNativeFunctions.h"

float* p_wave_table;

JNIEXPORT jint JNICALL Java_globalquake_jni_GQNativeFunctions_getTheUltimateAnswer(JNIEnv *, jclass, jint) {
    return 42;
}

