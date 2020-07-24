#include <jni.h>
#include "hnswindex.h"
#include "hnswlib.h"

extern "C" {

JNIEXPORT jlong JNICALL Java_com_criteo_hnsw_HnswLib_create(JNIEnv *env, jclass jobj, jint dim, jint distance, jint precision) {
    return (jlong)new Index<float>((Distance)distance, dim, (Precision)precision);
}

JNIEXPORT void JNICALL Java_com_criteo_hnsw_HnswLib_destroy(JNIEnv *env, jclass jobj, jlong pointer) {
    delete ((Index<float> *)pointer);
}

JNIEXPORT void JNICALL Java_com_criteo_hnsw_HnswLib_initNewIndex
(JNIEnv *env, jclass jobj, jlong pointer, jlong max_elements, jlong M, jlong ef_construction, jlong random_seed) {
    ((Index<float> *)pointer)->initNewIndex((size_t)max_elements, (size_t) M, (size_t) ef_construction, (size_t) random_seed);
}

JNIEXPORT void JNICALL Java_com_criteo_hnsw_HnswLib_initBruteforce
(JNIEnv *env, jclass jobj, jlong pointer, jlong max_elements) {
    ((Index<float> *)pointer)->initBruteforce((size_t)max_elements);
}

JNIEXPORT void JNICALL Java_com_criteo_hnsw_HnswLib_enableBruteforceSearch
(JNIEnv *env, jclass jobj, jlong pointer) {
    ((Index<float> *)pointer)->enableBruteforceSearch();
}

JNIEXPORT void JNICALL Java_com_criteo_hnsw_HnswLib_setEf(JNIEnv *env, jclass jobj, jlong pointer, jlong ef) {
    auto index = (Index<float> *)pointer;
    auto hnsw = (hnswlib::HierarchicalNSW<float> *)index->appr_alg;
    hnsw->ef_ = (size_t) ef;
}

JNIEXPORT void JNICALL Java_com_criteo_hnsw_HnswLib_saveIndex(JNIEnv *env, jclass jobj, jlong pointer, jstring path) {
    const char *path_to_index = env->GetStringUTFChars(path, NULL);
    ((Index<float> *)pointer)->saveIndex(path_to_index);
    env->ReleaseStringUTFChars(path, path_to_index);
}

JNIEXPORT void JNICALL Java_com_criteo_hnsw_HnswLib_loadIndex(JNIEnv *env, jclass jobj, jlong pointer, jstring path) {
    const char *path_to_index = env->GetStringUTFChars(path, NULL);
    ((Index<float> *)pointer)->loadIndex(path_to_index);
    env->ReleaseStringUTFChars(path, path_to_index);
}

JNIEXPORT void JNICALL Java_com_criteo_hnsw_HnswLib_loadBruteforce(JNIEnv *env, jclass jobj, jlong pointer, jstring path) {
    const char *path_to_index = env->GetStringUTFChars(path, NULL);
    ((Index<float> *)pointer)->loadBruteforce(path_to_index);
    env->ReleaseStringUTFChars(path, path_to_index);
}

JNIEXPORT void JNICALL Java_com_criteo_hnsw_HnswLib_addItem(JNIEnv *env, jclass jobj, jlong pointer, jfloatArray vector, jlong label) {
    auto hnsw = (Index<float> *)pointer;
    auto dim = hnsw->dim;
    std::vector<float> elements(dim);
    auto elements_data = elements.data();
    env->GetFloatArrayRegion(vector, 0, dim, elements_data);
    hnsw->addItem(elements_data, (size_t) label);
}

JNIEXPORT void JNICALL Java_com_criteo_hnsw_HnswLib_addItemBuffer(JNIEnv *env, jclass jobj, jlong pointer, jobject vector_buff, jlong label) {
    auto hnsw = (Index<float> *)pointer;
    auto vector_ptr = static_cast<float*>(env->GetDirectBufferAddress(vector_buff));
    hnsw->addItem(vector_ptr, (size_t) label);
}

JNIEXPORT void JNICALL Java_com_criteo_hnsw_HnswLib_trainEncodingSpace(JNIEnv *env, jclass jobj, jlong pointer, jfloatArray vector) {
    auto hnsw = (Index<float> *)pointer;
    auto dim = hnsw->dim;
    std::vector<float> elements(dim);
    auto elements_data = elements.data();
    env->GetFloatArrayRegion(vector, 0, dim, elements_data);
    hnsw->space->train(elements_data);
}

JNIEXPORT void JNICALL Java_com_criteo_hnsw_HnswLib_trainEncodingSpaceBuffer(JNIEnv *env, jclass jobj, jlong pointer, jobject vector_buff) {
    auto hnsw = (Index<float> *)pointer;
    auto vector_ptr = static_cast<float*>(env->GetDirectBufferAddress(vector_buff));
    hnsw->space->train(vector_ptr);
}

JNIEXPORT jlong JNICALL Java_com_criteo_hnsw_HnswLib_getNbItems(JNIEnv *env, jclass jobj, jlong pointer) {
    return ((Index<float> *)pointer)->getNbItems();
}

JNIEXPORT jlong JNICALL Java_com_criteo_hnsw_HnswLib_encodingNeedsTraining(JNIEnv *env, jclass jobj, jlong pointer) {
    return ((Index<float> *)pointer)->space->needs_initialization();
}

JNIEXPORT jobject JNICALL Java_com_criteo_hnsw_HnswLib_getItem(JNIEnv *env, jclass jobj, jlong pointer, jlong label) {
    auto hnsw = (Index<float> *) pointer;
    auto data_ptr = hnsw->getItem(label);
    if (data_ptr == nullptr) {
        return nullptr;
    } else {
        auto buffer = env->NewDirectByteBuffer(data_ptr, hnsw->space->get_data_size());
        return buffer;
    }
}

JNIEXPORT void JNICALL Java_com_criteo_hnsw_HnswLib_decode(JNIEnv *env, jclass jobj, jlong pointer, jobject src_buffer, jobject dst_buffer) {
    auto hnsw = (Index<float> *)pointer;
    auto src = static_cast<void*>(env->GetDirectBufferAddress(src_buffer));
    auto dst = static_cast<float*>(env->GetDirectBufferAddress(dst_buffer));
    hnsw->decode(src, dst);
}

JNIEXPORT void JNICALL Java_com_criteo_hnsw_HnswLib_encode(JNIEnv *env, jclass jobj, jlong pointer, jobject src_buffer, jobject dst_buffer) {
    auto hnsw = (Index<float> *)pointer;
    auto src = static_cast<float*>(env->GetDirectBufferAddress(src_buffer));
    auto dst = static_cast<void*>(env->GetDirectBufferAddress(dst_buffer));
    hnsw->encode(src, dst);
}

JNIEXPORT jint JNICALL Java_com_criteo_hnsw_HnswLib_search(JNIEnv *env, jclass jobj, jlong pointer, jobject query_buffer, jlong k, jobject items_result_buffer, jobject distance_result_buffer, jobjectArray result_vectors, jboolean bruteforce_search) {
    auto *hnsw = (Index<float> *) pointer;
    auto *query_buffer_address = static_cast<float *>(env->GetDirectBufferAddress(query_buffer));
    auto *items_result_address = static_cast<size_t *>(env->GetDirectBufferAddress(items_result_buffer));
    auto *distance_result_address = static_cast<float *>(env->GetDirectBufferAddress(distance_result_buffer));
    std::vector<float*> item_pointers(k);
    size_t result_count;
    if(bruteforce_search) {
        result_count = hnsw->knnQuery<true>(query_buffer_address, items_result_address, distance_result_address, item_pointers.data(), k);
    } else {
        result_count = hnsw->knnQuery<false>(query_buffer_address, items_result_address, distance_result_address, item_pointers.data(), k);
    }
    const auto data_size = hnsw->space->get_data_size();
    for(int i = 0; i < result_count; i++) {
        auto vector_buffer = env->NewDirectByteBuffer(item_pointers[i], data_size);
        env->SetObjectArrayElement(result_vectors, (jsize)i, (jobject)vector_buffer);
    }
    return result_count;
}

JNIEXPORT jint JNICALL Java_com_criteo_hnsw_HnswLib_getPrecision(JNIEnv *env, jclass jobj, jlong pointer) {
    return (jint)((Index<float> *)pointer)->precision;
}

JNIEXPORT jint JNICALL Java_com_criteo_hnsw_HnswLib_getDimension(JNIEnv *env, jclass jobj, jlong pointer) {
    return (jint)((Index<float> *)pointer)->dim;
}

JNIEXPORT jint JNICALL Java_com_criteo_hnsw_HnswLib_getMetric(JNIEnv *env, jclass jobj, jlong pointer) {
    return (jint)((Index<float> *)pointer)->distance;
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

}