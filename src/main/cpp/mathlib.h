#pragma once
#include <vector>
#include "hnswlib.h"

#ifndef HNSWLIB_JNI_MATH_UTILS_H
#define HNSWLIB_JNI_MATH_UTILS_H

class MathLib {
public:
    explicit MathLib(const int dim) : dimension(dim) {
        ip_dist_func = hnswlib::InnerProductSpace<float>(dim).get_dist_func();
        l2_dist_func = hnswlib::L2Space<float>(dim).get_dist_func();
        // TODO: Replace with optimized l2 norm function. Currently we compute l2 norm as l2 distance from zero vector
        zero_vector.assign(dim, 0.f);
        zero_vector_ptr = zero_vector.data();
        dim_ptr = &dimension;
    }

    float innerProductDistance(const float* vector1, const float* vector2) {
        // specific to dotProduct distance definition in hnswlib https://github.com/nmslib/hnswlib/blob/master/hnswlib/space_ip.h#L13
        return 1.f - ip_dist_func(vector1, vector2, dim_ptr);
    }

    float l2DistanceSquared(const float* vector1, const float* vector2) {
        return l2_dist_func(vector1, vector2, dim_ptr);
    }

    float l2NormSquared(const float* vector1) {
        return l2_dist_func(vector1, zero_vector_ptr, dim_ptr);
    }

private:
    const size_t dimension;
    hnswlib::DISTFUNC<float> l2_dist_func;
    hnswlib::DISTFUNC<float> ip_dist_func;
    std::vector<float> zero_vector;
    const void* dim_ptr;
    const float* zero_vector_ptr;
};

#endif
