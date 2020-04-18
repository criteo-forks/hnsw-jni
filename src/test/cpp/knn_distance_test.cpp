#include "doctest.h"
#include "hnswindex.h"
#include<cmath>

const int seed = 42;

float get_distance(void* a, void* b, int32_t dim, hnswlib::SpaceInterface<float>* space) {
    auto dist_func = space->get_dist_func();
    auto dist_func_param = space->get_dist_func_param();
    return dist_func(a, b, dist_func_param);
}

void to_float16(hnswlib::DecoderFloat16* decoder, std::vector<float> &src, std::vector<uint16_t> &dst) {
    auto encode_func = decoder->get_encode_func();
    auto dist_func_param = decoder->get_dist_func_param();
    encode_func(src.data(), dst.data(), dist_func_param);
}

void to_float32(hnswlib::DecoderFloat16* decoder, std::vector<uint16_t> &src, std::vector<float> &dst) {
    auto decode_func = decoder->get_decode_func();
    auto dist_func_param = decoder->get_dist_func_param();
    decode_func(src.data(), dst.data(), dist_func_param);
}

float p2(float x) { return pow(x, 2); }

// Reimplemented from Approx(double) for floats
float is_approx_euqal(float lhs, float rhs, float epsilon) {
    auto absolute_error = std::fabs(lhs - rhs);
    auto expected_error = epsilon * (1 + std::max(std::fabs(lhs), std::fabs(rhs)));
    CAPTURE("Absolute error:" << absolute_error);
    CAPTURE("Expected error:" << expected_error);
    return absolute_error < expected_error;
}

TEST_CASE("Check Float16 encoding-decoding on small vectors (1 - 16)") {
    std::vector<float> a {0,     1,     1.001, 2,     2.009, 3,     4,     6,     0.5,   -0.45, 0.3,  0.2,  -0.2, 0.1,  0.005, 0.006};
    std::vector<float> e {1E-30, 1E-30, 3E-5,  1E-30, 4E-4,  1E-30, 1E-30, 1E-30, 1E-30, 2E-4,  2E-4, 3E-4, 3E-4, 3E-4, 1E-5,  1E-30};

    for (int32_t dim = 1; dim < a.size(); dim++) {
        CAPTURE(dim);
        auto decoder = new hnswlib::DecoderFloat16(dim);
        std::vector<uint16_t> a_f16(dim);
        to_float16(decoder, a, a_f16);
        std::vector<float> a_f32(dim);
        to_float32(decoder, a_f16, a_f32);
        for (int32_t i = 0; i < dim; i++) {
            auto actual = a_f32[i];
            auto expected = a[i];
            auto epsilon = e[i];
            CAPTURE(epsilon);
            REQUIRE(is_approx_euqal(actual, expected, epsilon));
        }
        delete decoder;
    }
}

TEST_CASE("Check Float16 encoding-decoding on large vectors") {
    std::vector<int32_t> dimensions = {16, 17, 32, 34, 54, 100, 256, 300, 1000, 1024};
    srand(seed);
    auto epsilon = 1E-3;
    for (size_t i = 0; i < dimensions.size(); i++) {
        auto dim = dimensions[i];
        CAPTURE(dim);
        CAPTURE(epsilon);
        std::vector<float> a(dim);
        for(int j = 0; j < dim; j++) {
            a[j] = (float) rand() / RAND_MAX;
        }
        auto decoder = new hnswlib::DecoderFloat16(dim);
        std::vector<uint16_t> a_f16(dim);
        to_float16(decoder, a, a_f16);
        std::vector<float> a_f32(dim);
        to_float32(decoder, a_f16, a_f32);
        for (int32_t i = 0; i < dim; i++) {
            auto decoded_f16 = a_f32[i];
            auto expected = a[i];
            REQUIRE(is_approx_euqal(decoded_f16, expected, epsilon));
        }
        delete decoder;
    }
}

TEST_CASE("Check L2 distance computation on small vectors") {
    std::vector<float> a {1.0, 2.0, 3.0, 0.5, 0.1,  0.2, -0.2,  0.005, 1.001};
    std::vector<float> b {2.0, 4.0, 6.0, 0.0, 0.3, -0.2, -0.45, 0.006, 2.009};
    std::unordered_map<int32_t, std::pair<float, float>> expected_distances = {
    /* dim | epsilon | expected_distance */
        {1,  {1E-30,   1.0f}},
        {2,  {1E-30,   1.0f + p2(2)}},
        {3,  {1E-30,   1.0f + p2(2) + p2(3)}},
        {4,  {1E-30,   1.0f + p2(2) + p2(3) + p2(0.5)}},
        {5,  {1E-5 ,   1.0f + p2(2) + p2(3) + p2(0.5) + p2(0.2)}},
        {6,  {1E-5 ,   1.0f + p2(2) + p2(3) + p2(0.5) + p2(0.2) + p2(0.4)}},
        {7,  {1E-5 ,   1.0f + p2(2) + p2(3) + p2(0.5) + p2(0.2) + p2(0.4) + p2(-0.25)}},
        {8,  {1E-5 ,   1.0f + p2(2) + p2(3) + p2(0.5) + p2(0.2) + p2(0.4) + p2(-0.25) + p2(0.001)}},
        {9,  {1E-4 ,   1.0f + p2(2) + p2(3) + p2(0.5) + p2(0.2) + p2(0.4) + p2(-0.25) + p2(0.001) + p2(1.008)}},
    };
    for (std::pair<int32_t, std::pair<float, float>> element : expected_distances) {
        auto dim = element.first;
        auto decoder = new hnswlib::DecoderFloat16(dim);
        CAPTURE(dim);
        float epsilon = element.second.first;
        float expected_distance = element.second.second;
        INFO("float32 (no error)");
        hnswlib::SpaceInterface<float>* space = new hnswlib::L2Space(dim);
        auto result = get_distance(a.data(), b.data(), dim, space);
        REQUIRE_EQ(result, expected_distance);
        CAPTURE("float16. Allowed Error: " << epsilon);
        delete space;
        space = new hnswlib::L2SpaceF16(dim);
        std::vector<uint16_t> a_f16(dim), b_f16(dim);

        to_float16(decoder, a, a_f16);
        to_float16(decoder, b, b_f16);

        result = get_distance(a_f16.data(), b_f16.data(), dim, space);
        REQUIRE(is_approx_euqal(result, expected_distance, epsilon));
        delete space;
        delete decoder;
    }
}

TEST_CASE ("Check L2 distance computation on large vectors") {
    std::unordered_map<int32_t, std::pair<float, float>> expected_distances = {
    /* dim    | epsilon_f32 | epsilon_f16 */
        {16,       {1E-7,      1E-4}},
        {17,       {1E-7,      2E-3}},
        {32,       {1E-7,      1E-3}},
        {34,       {1E-7,      1E-3}},
        {54,       {1E-7,      1E-3}},
        {100,      {1E-6,      1E-3}},
        {256,      {1E-6,      1E-3}},
        {300,      {1E-6,      1E-3}},
        {1000,     {1E-6,      1E-3}},
        {1024,     {1E-6,      1E-3}},
    };
    srand(seed);
    const float component_diff = 0.1f;
    for (std::pair<int32_t, std::pair<float, float>> element : expected_distances) {
        auto dim = element.first;
        CAPTURE(dim);
        auto decoder = new hnswlib::DecoderFloat16(dim);
        std::vector<float> a(dim), b(dim);
        for(int j = 0; j < dim; j++) {
            a[j] = (float) rand() / RAND_MAX;
            b[j] = a[j] + component_diff;
        }
        float expected_distance = dim*p2(component_diff);

        float epsilon32 = element.second.first;
        CAPTURE("float32 Allowed Error: " << epsilon32);

        hnswlib::SpaceInterface<float>* space = new hnswlib::L2Space(dim);
        auto result = get_distance(a.data(), b.data(), dim, space);
        REQUIRE(is_approx_euqal(result, expected_distance, epsilon32));
        delete space;

        float epsilon16 = element.second.second;
        CAPTURE("float16. Allowed Error: " << epsilon16);

        space = new hnswlib::L2SpaceF16(dim);
        std::vector<uint16_t> a_f16(dim), b_f16(dim);
        to_float16(decoder, a, a_f16);
        to_float16(decoder, b, b_f16);
        result = get_distance(a_f16.data(), b_f16.data(), dim, space);
        REQUIRE(is_approx_euqal(result, expected_distance, epsilon16));
        delete space;
        delete decoder;
    }
}