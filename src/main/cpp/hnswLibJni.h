/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class com_criteo_recommendation_knn_HnswLib */

#ifndef _Included_com_criteo_hnsw_HnswLib
#define _Included_com_criteo_hnsw_HnswLib
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     com_criteo_recommendation_knn_HnswLib
 * Method:    createAngular
 * Signature: (I)J
 */
JNIEXPORT jlong JNICALL Java_com_criteo_hnsw_HnswLib_createAngular
  (JNIEnv *, jclass, jint);

/*
 * Class:     com_criteo_recommendation_knn_HnswLib
 * Method:    createEuclidean
 * Signature: (I)J
 */
JNIEXPORT jlong JNICALL Java_com_criteo_hnsw_HnswLib_createEuclidean
  (JNIEnv *, jclass, jint);

/*
 * Class:     com_criteo_recommendation_knn_HnswLib
 * Method:    createInnerProduct
 * Signature: (I)J
 */
JNIEXPORT jlong JNICALL Java_com_criteo_hnsw_HnswLib_createInnerProduct
  (JNIEnv *, jclass, jint);

/*
 * Class:     com_criteo_recommendation_knn_HnswLib
 * Method:    createKendall
 * Signature: (I)J
 */
JNIEXPORT jlong JNICALL Java_com_criteo_hnsw_HnswLib_createKendall
  (JNIEnv *, jclass, jint);

/*
 * Class:     com_criteo_recommendation_knn_HnswLib
 * Method:    createEuclideanF16
 * Signature: (I)J
 */
JNIEXPORT jlong JNICALL Java_com_criteo_hnsw_HnswLib_createEuclideanF16
  (JNIEnv *, jclass, jint);

/*
 * Class:     com_criteo_recommendation_knn_HnswLib
 * Method:    destroy
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_criteo_hnsw_HnswLib_destroy
  (JNIEnv *, jclass, jlong);

/*
 * Class:     com_criteo_recommendation_knn_HnswLib
 * Method:    initNewIndex
 * Signature: (JJJJJ)V
 */
JNIEXPORT void JNICALL Java_com_criteo_hnsw_HnswLib_initNewIndex
  (JNIEnv *, jclass, jlong, jlong, jlong, jlong, jlong);

/*
 * Class:     com_criteo_recommendation_knn_HnswLib
 * Method:    setEf
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_criteo_hnsw_HnswLib_setEf
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     com_criteo_recommendation_knn_HnswLib
 * Method:    saveIndex
 * Signature: (JLjava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_com_criteo_hnsw_HnswLib_saveIndex
  (JNIEnv *, jclass, jlong, jstring);

/*
 * Class:     com_criteo_recommendation_knn_HnswLib
 * Method:    loadIndex
 * Signature: (JLjava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_com_criteo_hnsw_HnswLib_loadIndex
  (JNIEnv *, jclass, jlong, jstring);

/*
 * Class:     com_criteo_recommendation_knn_HnswLib
 * Method:    addItem
 * Signature: (J[FJ)V
 */
JNIEXPORT void JNICALL Java_com_criteo_hnsw_HnswLib_addItem
  (JNIEnv *, jclass, jlong, jfloatArray, jlong);

/*
 * Class:     com_criteo_recommendation_knn_HnswLib
 * Method:    getNbItems
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_criteo_hnsw_HnswLib_getNbItems
  (JNIEnv *, jclass, jlong);

JNIEXPORT jobject JNICALL Java_com_criteo_hnsw_HnswLib_getItem(JNIEnv *env, jclass jobj, jlong pointer, jlong label);

JNIEXPORT jint JNICALL Java_com_criteo_hnsw_HnswLib_search(JNIEnv *env, jclass jobj, jlong pointer, jobject query_buffer, jlong k, jobject items_result_buffer, jobject distance_result_buffer, jobjectArray result_vectors);

JNIEXPORT void JNICALL Java_com_criteo_hnsw_HnswLib_decodeItem(JNIEnv *env, jclass jobj, jlong pointer, jobject src_buffer, jobject dst_buffer);

JNIEXPORT jfloat JNICALL Java_com_criteo_hnsw_HnswLib_getDistanceBetweenLabels(JNIEnv *env, jclass jobj, jlong pointer, jlong label1, jlong label2);

JNIEXPORT jfloat JNICALL Java_com_criteo_hnsw_HnswLib_getDistanceBetweenVectors(JNIEnv *env, jclass jobj, jlong pointer, jobject vector1_buffer, jobject vector2_buffer);

JNIEXPORT jlongArray JNICALL Java_com_criteo_hnsw_HnswLib_getLabels(JNIEnv *env, jclass jobj, jlong pointer);

#ifdef __cplusplus
}
#endif
#endif
