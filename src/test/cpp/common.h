#pragma once
#include "doctest.h"
#include "hnswindex.h"
#include <cmath>
#include <tuple>

#ifndef HNSWLIB_TEST_COMMON
#define HNSWLIB_TEST_COMMON
const int seed = 42;

// Reimplemented from Approx(double) for floats
static inline bool is_approx_equal(float lhs, float rhs, float epsilon) {
    const auto absolute_error = std::abs(lhs - rhs);
    const auto expected_error = epsilon * (1 + std::max(std::fabs(lhs), std::fabs(rhs)));
    return absolute_error < expected_error;
}

static inline float get_random_float(float min, float max) {
    return min + static_cast <float> (rand()) / static_cast <float> (RAND_MAX/(max-min));
}

static inline void to_float8(const std::vector<float> &src, std::vector<uint8_t> &dst, const hnswlib::TrainParams* params) {
    hnswlib::encode_trained_vector<float, uint8_t, hnswlib::encode_component, 1>(src.data(), dst.data(), params);
}

static inline void to_float32(const size_t dim, const std::vector<uint8_t> &src, std::vector<float> &dst, const hnswlib::TrainParams* params) {
    const auto decode_func = hnswlib::get_fast_decode_trained_func(dim);
    decode_func(src.data(), dst.data(), params);
}

static inline void to_float16(const size_t dim, const std::vector<float> &src, std::vector<uint16_t> &dst) {
    const auto encode_func = hnswlib::get_fast_encode_func<float, uint16_t>(dim);
    encode_func(src.data(), dst.data(), &dim);
}

static inline void to_float32(const size_t dim, const std::vector<uint16_t> &src, std::vector<float> &dst) {
    const auto decode_func = hnswlib::get_fast_encode_func<uint16_t, float>(dim);
    decode_func(src.data(), dst.data(), &dim);
}
#endif
