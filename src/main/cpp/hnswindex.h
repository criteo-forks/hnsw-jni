#pragma once
#include <iostream>
#include <tuple>
#include "hnswlib.h"

#ifndef KNN_JNI_HNSW_INDEX_H
#define KNN_JNI_HNSW_INDEX_H


enum Distance {
    Euclidean = 1,
    Angular = 2,
    InnerProduct = 3,
    Kendall = 4
};

enum Precision {
    Float32 = 1,
    Float16 = 2,
    Float8 = 3,
    num_values,
};

template<typename dist_t, typename data_t=float>
class Index {
public:
    Index(Distance distance, const int dim, const Precision precision) :
        dim(dim), precision(precision), distance(distance) {
        switch (distance) {
            case Euclidean:
                switch (precision) {
                    case Float8: space = new hnswlib::L2TrainedSpace<uint8_t>(dim); break;
                    case Float16: space = new hnswlib::L2Space<uint16_t>(dim); break;
                    case Float32:
                    default: space = new hnswlib::L2Space<float>(dim); break;
                }
                break;
            case Angular:
            case InnerProduct:
                switch (precision) {
                    case Float8: space = new hnswlib::InnerProductTrainedSpace<uint8_t>(dim); break;
                    case Float16: space = new hnswlib::InnerProductSpace<uint16_t>(dim); break;
                    case Float32:
                    default: space = new hnswlib::InnerProductSpace<float>(dim); break;
                }
                break;
            case Kendall:
                space = new hnswlib::KendallSpace(dim);
                if (precision != Float32) {
                    std::cerr<<"Warning: Kendal distance does not support other precision than float32\n";
                }
                break;
            default:
                throw std::runtime_error("Distance not supported: " + std::to_string(distance));
        }
        normalize = distance == Angular;
        decode_func_float16 = hnswlib::get_fast_encode_func<uint16_t, float>(dim);
        encode_func_float16 = hnswlib::get_fast_encode_func<float, uint16_t>(dim);
        decode_func_float8 = hnswlib::get_fast_decode_trained_func(dim);
        encode_func_float8 = hnswlib::encode_trained_vector<float, uint8_t, hnswlib::encode_component, 1>;
    }

    void initNewIndex(const size_t maxElements, const size_t M, const size_t efConstruction, const size_t random_seed) {
        setAlgorithm(new hnswlib::HierarchicalNSW<dist_t>(space, maxElements, M, efConstruction, random_seed));
    }

    void initBruteforce(const size_t maxElements) {
        setAlgorithm(new hnswlib::BruteforceSearch<dist_t>(space, maxElements));
    }

    void enableBruteforceSearch() {
        brute_alg = new hnswlib::BruteforceSearchAlg<dist_t>(space);
    }

    void setAlgorithm(hnswlib::AlgorithmInterface<dist_t> * algo) {
        if (appr_alg) {
            std::cerr<<"Warning: Setting index for an already inited index. Old index is being deallocated.\n";
            delete appr_alg;
        }
        appr_alg = algo;
        label_lookup_ = appr_alg->getLabelLookup();
    }

    void saveIndex(const std::string &path_to_index) {
        appr_alg->saveIndex(path_to_index);
    }

    void loadIndex(const std::string &path_to_index) {
        auto algo = new hnswlib::HierarchicalNSW<dist_t>(space);
        switch (precision) {
            case Float32: algo->template loadAndDecode<float, float, size_t>(path_to_index, space, nullptr); break;
            case Float16: algo->template loadAndDecode<float, uint16_t, size_t>(path_to_index, space, encode_func_float16); break;
            case Float8:  algo->template loadAndDecode<float, uint8_t, hnswlib::TrainParams>(path_to_index, space, encode_func_float8); break;
            default: throw std::runtime_error("Unsupported precision " + std::to_string(precision));
        }
        setAlgorithm(algo);
    }

    // TODO: Unify with loadIndex
    void loadBruteforce(const std::string &path_to_index) {
        auto algo = new hnswlib::BruteforceSearch<dist_t>(space, 0);
        switch (precision) {
            case Float32: algo->template loadAndDecode<float, float, size_t>(path_to_index, space, nullptr); break;
            case Float16: algo->template loadAndDecode<float, uint16_t, size_t>(path_to_index, space, encode_func_float16); break;
            case Float8:  algo->template loadAndDecode<float, uint8_t, hnswlib::TrainParams>(path_to_index, space, encode_func_float8); break;
            default: throw std::runtime_error("Unsupported precision " + std::to_string(precision));
        }
        setAlgorithm(algo);
    }

    void normalizeVector(dist_t *data, dist_t *norm_array){
        dist_t norm = 1.0f / (getL2Norm(data) + 1e-30f);
        for(int i = 0; i < dim; i++) {
            norm_array[i] = data[i]*norm;
        }
    }

    inline float getL2Norm(dist_t *data) {
        dist_t norm=0.0f;
        for(int i=0; i < dim; i++) {
            norm+= data[i]*data[i];
        }
        return sqrtf(norm);
    }

    void addItem(dist_t* vector, size_t id) {
        std::vector<dist_t> norm_array;
        std::vector<char> encoded_vector;
        const auto normalized_data = normalizeItem(vector, norm_array);
        const auto vector_data = encodeItem(normalized_data, encoded_vector);
        appr_alg->addPoint(vector_data, (size_t) id);
    }

    size_t getNbItems() {
        return appr_alg->getNbItems();
    }

    void* getItem(size_t label) {
        hnswlib::tableint label_c;
        auto search = label_lookup_->find(label);
        if (search == label_lookup_->end()) {
            return nullptr;
        }
        label_c = search->second;
        return appr_alg->getDataByInternalId(label_c);
    }

    std::vector<size_t> getLabels() {
        std::vector<size_t> labels;
        for(auto & iter : *label_lookup_) {
            labels.push_back(iter.first);
        }
        return labels;
    }

    dist_t* normalizeItem(dist_t* item, std::vector<dist_t>& norm_array) {
        if(normalize) {
            norm_array.resize(dim);
            normalizeVector(static_cast<dist_t*>(item), norm_array.data());
            return norm_array.data();
        }
        return item;
    }

    void* encodeItem(dist_t* item, std::vector<char>& encoded_vector) {
        if (precision == Float32) {
            return item;
        }
        encoded_vector.resize(space->get_data_size());
        return encode(item, reinterpret_cast<void*>(encoded_vector.data()));
    }

    void* encode(dist_t* src, void* dst) {
        const auto param = space->get_dist_func_param();
        switch (precision) {
            case Float32: return src;
            case Float16: encode_func_float16(src, reinterpret_cast<uint16_t *>(dst), static_cast<const size_t*>(param)); return dst;
            case Float8:  encode_func_float8(src, reinterpret_cast<uint8_t *>(dst), static_cast<const hnswlib::TrainParams*>(param)); return dst;
            default: throw std::runtime_error("Unsupported precision " + std::to_string(precision));
        }
    }

    dist_t* decode(void* src, dist_t* dst) {
        const auto param = space->get_dist_func_param();
        switch (precision) {
            case Float32: return static_cast<dist_t*>(src);
            case Float16: decode_func_float16(reinterpret_cast<uint16_t *>(src), dst, static_cast<const size_t*>(param)); return dst;
            case Float8:  decode_func_float8(reinterpret_cast<uint8_t *>(src), dst, static_cast<const hnswlib::TrainParams*>(param)); return dst;
            default: throw std::runtime_error("Unsupported precision " + std::to_string(precision));
        }
    }

    /**
     * `knnQuery` - runs knn search on the query vector and returns k closest neighbours with
     * distance from query and pointer to each result.
     *
     * `bruteforce_search` = true is use to compute fast recall/precison of the index which is
     * already loaded in memory avoiding double vectors footprint on memory and improving speed.
     *
     *  * `query` - query vector (float[dim]) in the index space
     *  * `result_labels` (out) - array of labels of nearest neighbours (size_t[k])
     *  * `result_distances` (out) - array of distances from query to result items (float[k])
     *  * `results_pointers` (out) - array of pointers to results (float*[k])
     *  * `k` - number of neighbours to retrieve
     *
     * Returns: number of neighbours returned (<= k)
     **/
    template<bool bruteforce_search=false>
    size_t knnQuery(dist_t* query, size_t* result_labels, dist_t* result_distances, data_t** results_pointers, size_t k) {
        std::vector<dist_t> norm_array;
        const auto query_data = normalizeItem(query, norm_array);

        std::priority_queue<std::pair<dist_t, hnswlib::tableint >> result;
        if(!bruteforce_search) {
            result = appr_alg->searchKnn(query_data, k);
        } else {
            result = brute_alg->searchKnn(query_data, k, appr_alg);
        }
        const auto nbResults = result.size();

        for (int i = nbResults - 1; i >= 0; i--) {
            auto &result_tuple = result.top();
            result_distances[i] = result_tuple.first;
            result_labels[i] = (size_t)appr_alg->getExternalLabel(result_tuple.second);
            results_pointers[i] = (data_t*)appr_alg->getDataByInternalId(result_tuple.second);
            result.pop();
        }
        return nbResults;
    }

    dist_t getDistanceBetweenLabels(size_t label1, size_t label2) {
        return getDistanceBetweenVectors(getItem(label1), getItem(label2));
    }

    dist_t getDistanceBetweenVectors(void* vector1, void* vector2) {
        const auto func = space->get_dist_func();
        return func(vector1, vector2, &dim);
    }

    hnswlib::SpaceInterface<float>* space;
    const size_t dim;
    bool normalize = false;
    hnswlib::DECODEFUNC<dist_t, uint16_t, size_t> encode_func_float16;
    hnswlib::DECODEFUNC<uint16_t, dist_t, size_t> decode_func_float16;
    hnswlib::DECODEFUNC<dist_t, uint8_t, hnswlib::TrainParams> encode_func_float8;
    hnswlib::DECODEFUNC<uint8_t, dist_t, hnswlib::TrainParams> decode_func_float8;
    const Precision precision;
    const Distance distance;
    hnswlib::AlgorithmInterface<dist_t> * appr_alg = nullptr;
    hnswlib::BruteforceSearchAlg<dist_t> * brute_alg = nullptr;
    std::unordered_map<hnswlib::labeltype, hnswlib::tableint> * label_lookup_ = nullptr;

    ~Index() {
        delete space;
        if (brute_alg)
            delete brute_alg;
        if (appr_alg)
            delete appr_alg;
    }
};

#endif