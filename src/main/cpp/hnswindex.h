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
};

template<typename dist_t, typename data_t=float>
class Index {
public:
    Index(hnswlib::SpaceInterface<float> *space, const int dim, bool normalize = false, const Precision precision = Float32) :
        space(space), dim(dim), normalize(normalize), data_size(space->get_data_size()), precision(precision) {
        appr_alg = NULL;
    }

    Index(Distance distance, const int dim, const Precision precision = Float32) :
        dim(dim), precision(precision) {
        if (precision == Float16) {
            if(distance == Euclidean) {
                auto spaceF16 = new hnswlib::L2SpaceF16(dim);
                encode_func_ = spaceF16->get_encode_func();
                decode_func_ = spaceF16->get_decode_func();
                space = spaceF16;
            }
            else {
                throw std::runtime_error(std::to_string(precision) + " precision does not support distance " + std::to_string(distance));
            }
        } else {
            if(distance == Euclidean) {
                space = new hnswlib::L2Space(dim);
            }
            else if (distance == Angular) {
                space = new hnswlib::InnerProductSpace(dim);
                normalize = true;
            }
            else if (distance == InnerProduct) {
                space = new hnswlib::InnerProductSpace(dim);
            }
            else if (distance == Kendall) {
                space = new hnswlib::KendallSpace(dim);
            }
            else {
                throw std::runtime_error("Distance not supported: " + std::to_string(distance));
            }
        }
        data_size = space->get_data_size();
        appr_alg = NULL;
    }

    void initNewIndex(const size_t maxElements, const size_t M, const size_t efConstruction, const size_t random_seed) {
        if (appr_alg) {
            throw new std::runtime_error("The index is already initiated.");
        }
        appr_alg = new hnswlib::HierarchicalNSW<dist_t>(space, maxElements, M, efConstruction, random_seed);
    }

    void saveIndex(const std::string &path_to_index) {
        appr_alg->saveIndex(path_to_index);
    }

    void loadIndex(const std::string &path_to_index) {
        if (appr_alg) {
            std::cerr<<"Warning: Calling load_index for an already inited index. Old index is being deallocated.\n";
            delete appr_alg;
        }
        appr_alg = new hnswlib::HierarchicalNSW<dist_t>(space, path_to_index, false, 0);
    }

    void normalizeVector(dist_t *data, dist_t *norm_array){
        dist_t norm=0.0f;
        for(int i=0;i<dim;i++)
            norm+=data[i]*data[i];
        norm= 1.0f / (sqrtf(norm) + 1e-30f);
        for(int i=0;i<dim;i++)
            norm_array[i]=data[i]*norm;
    }

    void addItem(dist_t* vector, size_t id) {
        dist_t* vector_data = vector;
        std::vector<dist_t> norm_array;
        if(normalize) {
            norm_array.reserve(dim);
            normalizeVector(vector_data, norm_array.data());
            vector_data = norm_array.data();
        }
        std::vector<dist_t> encoded_vector;
        if (precision == Float16) {
            encoded_vector.reserve(dim);
            encodeItem(vector_data, encoded_vector.data());
            vector_data = encoded_vector.data();
        }
        appr_alg->addPoint(vector_data, (size_t) id);
    }

    size_t getNbItems() {
        return appr_alg->cur_element_count;
    }

    void* getItem(size_t label) {
        hnswlib::tableint label_c;
        auto search = appr_alg->label_lookup_.find(label);
        if (search == appr_alg->label_lookup_.end()) {
            return nullptr;
        }
        label_c = search->second;
        return appr_alg->getDataByInternalId(label_c);
    }

    inline void decodeItem(const void* src, void* dst) {
        decode_func_(src, dst, appr_alg->dist_func_param_);
    }

    inline void encodeItem(const void* src, void* dst) {
        encode_func_(src, dst, appr_alg->dist_func_param_);
    }

    std::vector<size_t> getLabels() {
        std::vector<size_t> labels;
        for(auto kv : appr_alg->label_lookup_) {
            labels.push_back(kv.first);
        }
        return labels;
    }

    /**
     * `knnQuery` - runs knn search on the query vector and returns k closest neighbours with
     * distance from query and pointer to each result
     *
     *  * `query` - query vector (float[dim]) in the index space
     *  * `result_labels` (out) - array of labels of nearest neighbours (size_t[k])
     *  * `result_distances` (out) - array of distances from query to result items (float[k])
     *  * `results_pointers` (out) - array of pointers to results (float*[k])
     *  * `k` - number of neighbours to retrieve
     *
     * Returns: number of neighbours returned (<= k)
     **/
    size_t knnQuery(dist_t * query, size_t * result_labels, dist_t * result_distances, data_t** results_pointers, size_t k) {
        dist_t* query_data = query;
        std::vector<dist_t> norm_array;
        if(normalize) {
            norm_array.reserve(dim);
            normalizeVector(query_data, norm_array.data());
            query_data = norm_array.data();
        }
        std::vector<dist_t> encoded_vector;
        if (precision == Float16) {
            encoded_vector.reserve(dim);
            // TODO: Add asymmetric distance computation to avoid f32->f16->f32*n conversions for query vector
            encodeItem(query_data, encoded_vector.data());
            query_data = encoded_vector.data();
        }
        std::priority_queue<std::pair<dist_t, hnswlib::tableint >> result = appr_alg->searchKnn(
                (void *) query_data, k);
        size_t nbResults = result.size();

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
        return appr_alg->fstdistfunc_(vector1, vector2, appr_alg->dist_func_param_);
    }

    hnswlib::SpaceInterface<float> *space;
    int dim;
    int data_size;
    bool normalize = false;
    hnswlib::ENCODEFUNC encode_func_;
    hnswlib::DECODEFUNC decode_func_;
    Precision precision = Float32;
    hnswlib::HierarchicalNSW<dist_t> *appr_alg;

    ~Index() {
        delete space;
        if (appr_alg)
            delete appr_alg;
    }
};

#endif