#include "hnswLibJni.h"
#include "hnswindex.h"
#include "hnswlib.h"

/*
 * Class:     com_criteo_recommendation_knn_HnswLib
 * Method:    create
 * Signature: (I)J
 */
JNIEXPORT jlong JNICALL Java_com_criteo_hnsw_HnswLib_create(JNIEnv *env, jclass jobj, jint dim, jint distance, jint precision) {
    return (jlong)new Index<float>((Distance)distance, dim, (Precision)precision);
}

/*
 * Class:     com_criteo_recommendation_knn_HnswLib
 * Method:    destroy
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_criteo_hnsw_HnswLib_destroy(JNIEnv *env, jclass jobj, jlong pointer) {
    delete ((Index<float> *)pointer);
}

/*
 * Class:     com_criteo_recommendation_knn_HnswLib
 * Method:    initNewIndex
 * Signature: (JJJJJ)V
 */
JNIEXPORT void JNICALL Java_com_criteo_hnsw_HnswLib_initNewIndex
(JNIEnv *env, jclass jobj, jlong pointer, jlong max_elements, jlong M, jlong ef_construction, jlong random_seed) {
    ((Index<float> *)pointer)->initNewIndex((size_t)max_elements, (size_t) M, (size_t) ef_construction, (size_t) random_seed);
}

/*
 * Class:     com_criteo_recommendation_knn_HnswLib
 * Method:    initBruteforce
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_criteo_hnsw_HnswLib_initBruteforce
(JNIEnv *env, jclass jobj, jlong pointer, jlong max_elements) {
    ((Index<float> *)pointer)->initBruteforce((size_t)max_elements);
}

/*
 * Class:     com_criteo_recommendation_knn_HnswLib
 * Method:    setEf
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_criteo_hnsw_HnswLib_setEf(JNIEnv *env, jclass jobj, jlong pointer, jlong ef) {
    auto index = (Index<float> *)pointer;
    auto hnsw = (hnswlib::HierarchicalNSW<float> *)index->appr_alg;
    hnsw->ef_ = (size_t) ef;
}

/*
 * Class:     com_criteo_recommendation_knn_HnswLib
 * Method:    saveIndex
 * Signature: (JLjava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_com_criteo_hnsw_HnswLib_saveIndex(JNIEnv *env, jclass jobj, jlong pointer, jstring path) {
    const char *path_to_index = env->GetStringUTFChars(path, NULL);
    ((Index<float> *)pointer)->saveIndex(path_to_index);
    env->ReleaseStringUTFChars(path, path_to_index);
}

/*
 * Class:     com_criteo_recommendation_knn_HnswLib
 * Method:    loadIndex
 * Signature: (JLjava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_com_criteo_hnsw_HnswLib_loadIndex(JNIEnv *env, jclass jobj, jlong pointer, jstring path) {
    const char *path_to_index = env->GetStringUTFChars(path, NULL);
    ((Index<float> *)pointer)->loadIndex(path_to_index);
    env->ReleaseStringUTFChars(path, path_to_index);
}

/*
 * Class:     com_criteo_recommendation_knn_HnswLib
 * Method:    loadBruteforce
 * Signature: (JLjava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_com_criteo_hnsw_HnswLib_loadBruteforce(JNIEnv *env, jclass jobj, jlong pointer, jstring path) {
    const char *path_to_index = env->GetStringUTFChars(path, NULL);
    ((Index<float> *)pointer)->loadBruteforce(path_to_index);
    env->ReleaseStringUTFChars(path, path_to_index);
}

/*
 * Class:     com_criteo_recommendation_knn_HnswLib
 * Method:    addItem
 * Signature: (J[FJ)V
 */
JNIEXPORT void JNICALL Java_com_criteo_hnsw_HnswLib_addItem(JNIEnv *env, jclass jobj, jlong pointer, jfloatArray vector, jlong label) {
    auto hnsw = (Index<float> *)pointer;
    auto dim = hnsw->dim;
    std::vector<float> elements(dim);
    auto elements_data = elements.data();
    env->GetFloatArrayRegion(vector, 0, dim, elements_data);
    hnsw->addItem(elements_data, (size_t) label);
}

/*
 * Class:     com_criteo_recommendation_knn_HnswLib
 * Method:    getNbItems
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_criteo_hnsw_HnswLib_getNbItems(JNIEnv *env, jclass jobj, jlong pointer) {
    return ((Index<float> *)pointer)->getNbItems();
}

JNIEXPORT jobject JNICALL Java_com_criteo_hnsw_HnswLib_getItem(JNIEnv *env, jclass jobj, jlong pointer, jlong label) {
    auto hnsw = (Index<float> *) pointer;
    auto data_ptr = hnsw->getItem(label);
    if (data_ptr == nullptr) {
        return nullptr;
    } else {
        auto buffer = env->NewDirectByteBuffer(data_ptr, hnsw->data_size);
        return buffer;
    }
}

JNIEXPORT jboolean JNICALL Java_com_criteo_hnsw_HnswLib_decodeItem(JNIEnv *env, jclass jobj, jlong pointer, jobject src_buffer, jobject dst_buffer) {
    auto hnsw = (Index<float> *)pointer;
    auto src = static_cast<uint16_t*>(env->GetDirectBufferAddress(src_buffer));
    auto dst = static_cast<float*>(env->GetDirectBufferAddress(dst_buffer));
    if (hnsw->precision == Float16) {
        hnsw->decodeFloat16(src, dst);
        return (jboolean)true;
    }
    return (jboolean)false;
}

JNIEXPORT jint JNICALL Java_com_criteo_hnsw_HnswLib_search(JNIEnv *env, jclass jobj, jlong pointer, jobject query_buffer, jlong k, jobject items_result_buffer, jobject distance_result_buffer, jobjectArray result_vectors) {
    auto *hnsw = (Index<float> *) pointer;
    auto *query_buffer_address = static_cast<float *>(env->GetDirectBufferAddress(query_buffer));
    auto *items_result_address = static_cast<size_t *>(env->GetDirectBufferAddress(items_result_buffer));
    auto *distance_result_address = static_cast<float *>(env->GetDirectBufferAddress(distance_result_buffer));
    std::vector<float*> item_pointers(k);
    auto result_count = hnsw->knnQuery(query_buffer_address, items_result_address, distance_result_address, item_pointers.data(), k);
    for(int i = 0; i < result_count; i++) {
        auto vector_buffer = env->NewDirectByteBuffer(item_pointers[i], hnsw->data_size);
        env->SetObjectArrayElement(result_vectors, (jsize)i, (jobject)vector_buffer);
    }
    return result_count;
}

JNIEXPORT jfloat JNICALL Java_com_criteo_hnsw_HnswLib_getDistanceBetweenLabels(JNIEnv *env, jclass jobj, jlong pointer, jlong label1, jlong label2) {
    return ((Index<float> *)pointer)->getDistanceBetweenLabels(label1, label2);
}

JNIEXPORT jfloat JNICALL Java_com_criteo_hnsw_HnswLib_getDistanceBetweenVectors(JNIEnv *env, jclass jobj, jlong pointer, jobject vector1_buffer, jobject vector2_buffer) {
    auto *vector1 = static_cast<float *>(env->GetDirectBufferAddress(vector1_buffer));
    auto *vector2 = static_cast<float *>(env->GetDirectBufferAddress(vector2_buffer));
    return ((Index<float> *)pointer)->getDistanceBetweenVectors(vector1, vector2);
}

JNIEXPORT jlongArray JNICALL Java_com_criteo_hnsw_HnswLib_getLabels(JNIEnv *env, jclass jobj, jlong pointer) {
    auto hnsw = (Index<float> *) pointer;
    auto labels = hnsw->getLabels();
    jlongArray result = env->NewLongArray(labels.size());
    env->SetLongArrayRegion(result, 0, labels.size(), (jlong *) labels.data());
    return result;
}