#include <jni.h>
#include "hnswindex.h"
#include "mathlib.h"

extern "C" {

JNIEXPORT jfloat JNICALL Java_com_criteo_hnsw_MathLib_innerProductDistance
        (JNIEnv *env, jclass jobj, jlong pointer, jobject vector1_buffer, jobject vector2_buffer) {
    const auto *vector1 = static_cast<float *>(env->GetDirectBufferAddress(vector1_buffer));
    const auto *vector2 = static_cast<float *>(env->GetDirectBufferAddress(vector2_buffer));
    return ((MathLib *) pointer)->innerProductDistance(vector1, vector2);
}

JNIEXPORT jfloat JNICALL Java_com_criteo_hnsw_MathLib_l2DistanceSquared
        (JNIEnv *env, jclass jobj, jlong pointer, jobject vector1_buffer, jobject vector2_buffer) {
    const auto *vector1 = static_cast<float *>(env->GetDirectBufferAddress(vector1_buffer));
    const auto *vector2 = static_cast<float *>(env->GetDirectBufferAddress(vector2_buffer));
    return ((MathLib *) pointer)->l2DistanceSquared(vector1, vector2);
}

JNIEXPORT jfloat JNICALL Java_com_criteo_hnsw_MathLib_l2NormSquared
        (JNIEnv *env, jclass jobj, jlong pointer, jobject vector1_buffer) {
    const auto *vector1 = static_cast<float *>(env->GetDirectBufferAddress(vector1_buffer));
    return ((MathLib *) pointer)->l2NormSquared(vector1);
}

JNIEXPORT jlong JNICALL Java_com_criteo_hnsw_MathLib_create(JNIEnv *env, jclass jobj, jint dim) {
    return (jlong) new MathLib(dim);
}

JNIEXPORT void JNICALL Java_com_criteo_hnsw_MathLib_destroy(JNIEnv *env, jclass jobj, jlong pointer) {
    delete ((MathLib *) pointer);
}

}
