#include "doctest.h"
#include "hnswindex.h"
#include <cmath>
#include <tuple>

const int seed = 42;

float get_distance(const void* a, const void* b, int32_t dim, const hnswlib::SpaceInterface<float>* space) {
    const auto sp = const_cast<hnswlib::SpaceInterface<float>*>(space);
    const auto dist_func = sp->get_dist_func();
    const auto dist_func_param = sp->get_dist_func_param();
    return dist_func(a, b, dist_func_param);
}

void to_float16(const hnswlib::DecoderFloat16* decoder, const std::vector<float> &src, std::vector<uint16_t> &dst) {
    const auto encode_func = decoder->get_encode_func();
    const auto dist_func_param = decoder->get_dist_func_param();
    encode_func(src.data(), dst.data(), dist_func_param);
}

void to_float32(const hnswlib::DecoderFloat16* decoder, const std::vector<uint16_t> &src, std::vector<float> &dst) {
    const auto decode_func = decoder->get_decode_func();
    const auto dist_func_param = decoder->get_dist_func_param();
    decode_func(src.data(), dst.data(), dist_func_param);
}

float p2(float x) { return pow(x, 2); }

// Reimplemented from Approx(double) for floats
float is_approx_euqal(float lhs, float rhs, float epsilon) {
    const auto absolute_error = std::fabs(lhs - rhs);
    const auto expected_error = epsilon * (1 + std::max(std::fabs(lhs), std::fabs(rhs)));
    return absolute_error < expected_error;
}

TEST_CASE("Check Float16 encoding-decoding on small vectors (1 - 16)") {
    const std::vector<float> a {0,     1,     1.001, 2,     2.009, 3,     4,     6,     0.5,   -0.45, 0.3,  0.2,  -0.2, 0.1,  0.005, 0.006};
    const std::vector<float> e {1E-30, 1E-30, 3E-5,  1E-30, 4E-4,  1E-30, 1E-30, 1E-30, 1E-30, 2E-4,  2E-4, 3E-4, 3E-4, 3E-4, 1E-5,  1E-30};

    for (size_t dim = 1; dim < a.size(); dim++) {
        CAPTURE(dim);
        const auto decoder = hnswlib::DecoderFloat16{dim};
        std::vector<uint16_t> a_f16(dim);
        to_float16(&decoder, a, a_f16);
        std::vector<float> a_f32(dim);
        to_float32(&decoder, a_f16, a_f32);
        for (auto i = 0; i < dim; i++) {
            auto actual = a_f32[i];
            auto expected = a[i];
            auto epsilon = e[i];
            CAPTURE(epsilon);
            REQUIRE(is_approx_euqal(actual, expected, epsilon));
        }
    }
}

TEST_CASE("Check Float16 encoding-decoding on large vectors") {
    const std::vector<size_t> dimensions = {16, 17, 32, 34, 54, 100, 256, 300, 1000, 1024};
    srand(seed);
    const auto epsilon = 1E-3f;
    for (auto i = 0; i < dimensions.size(); i++) {
        const auto dim = dimensions[i];
        CAPTURE(dim);
        CAPTURE(epsilon);
        std::vector<float> a(dim);
        for(int j = 0; j < dim; j++) {
            a[j] = (float) rand() / RAND_MAX;
        }
        const auto decoder = hnswlib::DecoderFloat16(dim);
        std::vector<uint16_t> a_f16(dim);
        to_float16(&decoder, a, a_f16);
        std::vector<float> a_f32(dim);
        to_float32(&decoder, a_f16, a_f32);
        for (int32_t i = 0; i < dim; i++) {
            auto decoded_f16 = a_f32[i];
            auto expected = a[i];
            REQUIRE(is_approx_euqal(decoded_f16, expected, epsilon));
        }
    }
}

TEST_CASE("Check L2 distance computation on small vectors") {
    const std::vector<float> a {1.0, 2.0, 3.0, 0.5, 0.1,  0.2, -0.2,  0.005, 1.001};
    const std::vector<float> b {2.0, 4.0, 6.0, 0.0, 0.3, -0.2, -0.45, 0.006, 2.009};
    const std::vector<std::tuple<size_t, float, float>> expected_distances {
        /*              dim | epsilon | expected_distance */
        std::make_tuple(1UL,  1E-30f,   1.0f),
        std::make_tuple(2UL,  1E-30f,   1.0f + p2(2.f)),
        std::make_tuple(3UL,  1E-30f,   1.0f + p2(2.f) + p2(3.f)),
        std::make_tuple(4UL,  1E-30f,   1.0f + p2(2.f) + p2(3.f) + p2(0.5f)),
        std::make_tuple(5UL,  1E-5f ,   1.0f + p2(2.f) + p2(3.f) + p2(0.5f) + p2(0.2f)),
        std::make_tuple(6UL,  1E-5f ,   1.0f + p2(2.f) + p2(3.f) + p2(0.5f) + p2(0.2f) + p2(0.4f)),
        std::make_tuple(7UL,  1E-5f ,   1.0f + p2(2.f) + p2(3.f) + p2(0.5f) + p2(0.2f) + p2(0.4f) + p2(-0.25f)),
        std::make_tuple(8UL,  1E-5f ,   1.0f + p2(2.f) + p2(3.f) + p2(0.5f) + p2(0.2f) + p2(0.4f) + p2(-0.25f) + p2(0.001f)),
        std::make_tuple(9UL,  1E-4f ,   1.0f + p2(2.f) + p2(3.f) + p2(0.5f) + p2(0.2f) + p2(0.4f) + p2(-0.25f) + p2(0.001f) + p2(1.008f)),
    };
    for (const auto &element : expected_distances) {
        const auto dim = std::get<0>(element);
        const auto epsilon = std::get<1>(element);
        const auto expected_distance = std::get<2>(element);
        const auto decoder = hnswlib::DecoderFloat16{dim};
        CAPTURE(dim);
        INFO("float32 (no error)");
        const auto space32 = hnswlib::L2Space(dim);
        const auto result32 = get_distance(a.data(), b.data(), dim, &space32);
        CAPTURE(result32);
        CAPTURE(expected_distance);
        REQUIRE_EQ(result32, expected_distance);
        CAPTURE("float16. Allowed Error: " << epsilon);
        const auto space16 = hnswlib::L2SpaceF16(dim);
        std::vector<uint16_t> a_f16(dim), b_f16(dim);

        to_float16(&decoder, a, a_f16);
        to_float16(&decoder, b, b_f16);

        const auto result16 = get_distance(a_f16.data(), b_f16.data(), dim, &space16);
        CAPTURE(result16);
        REQUIRE(is_approx_euqal(result16, expected_distance, epsilon));
    }
}

TEST_CASE ("Check L2 distance computation on large vectors") {
    const std::vector<std::tuple<size_t, float, float>> expected_distances {
        /*              dim    | epslon_f32 | epsilon_f16 */
        std::make_tuple(16UL,       1E-7f,      1E-3f),
        std::make_tuple(17UL,       1E-7f,      2E-3f),
        std::make_tuple(32UL,       1E-7f,      1E-3f),
        std::make_tuple(34UL,       1E-7f,      1E-3f),
        std::make_tuple(54UL,       1E-7f,      1E-3f),
        std::make_tuple(100UL,      1E-6f,      1E-3f),
        std::make_tuple(256UL,      1E-6f,      1E-3f),
        std::make_tuple(300UL,      1E-6f,      1E-3f),
        std::make_tuple(1000UL,     1E-6f,      1E-3f),
        std::make_tuple(1024UL,     1E-6f,      1E-3f),
    };
    srand(seed);
    const auto component_diff = 0.1f;
    for (const auto &element : expected_distances) {
        const auto dim = std::get<0>(element);
        const auto epsilon32 = std::get<1>(element);
        const auto epsilon16 = std::get<2>(element);
        CAPTURE(dim);
        const auto decoder = hnswlib::DecoderFloat16{dim};
        std::vector<float> a(dim), b(dim);
        for(int j = 0; j < dim; j++) {
            a[j] = (float) rand() / RAND_MAX;
            b[j] = a[j] + component_diff;
        }
        const auto expected_distance = dim*p2(component_diff);
        CAPTURE("float32 Allowed Error: " << epsilon32);

        const auto space32 = hnswlib::L2Space(dim);
        const auto result32 = get_distance(a.data(), b.data(), dim, &space32);
        CAPTURE(expected_distance);
        CAPTURE(result32);
        REQUIRE(is_approx_euqal(result32, expected_distance, epsilon32));
        CAPTURE("float16. Allowed Error: " << epsilon16);

        const auto space16 = hnswlib::L2SpaceF16(dim);
        std::vector<uint16_t> a_f16(dim), b_f16(dim);
        to_float16(&decoder, a, a_f16);
        to_float16(&decoder, b, b_f16);
        const auto result16 = get_distance(a_f16.data(), b_f16.data(), dim, &space16);
        CAPTURE(result16);
        REQUIRE(is_approx_euqal(result16, expected_distance, epsilon16));
    }
}

TEST_CASE("Check Inner Product distance computation on small vectors") {
    const std::vector<float> a {1.0, 2.0, 3.0, 0.5, 0.1,  0.2, -0.2,  0.005, 1.001};
    const std::vector<float> b {2.0, 4.0, 6.0, 0.0, 0.3, -0.2, -0.45, 0.006, 2.009};
    const std::vector<std::tuple<size_t, float, float>> expected_distances = {
        /*              dim | epsilon | expected_distance */
        std::make_tuple(1UL,  1E-30f,   1.f - (2.f)),
        std::make_tuple(2UL,  1E-30f,   1.f - (2.f + 8.f)),
        std::make_tuple(3UL,  1E-30f,   1.f - (2.f + 8.f + 18.f)),
        std::make_tuple(4UL,  1E-30f,   1.f - (2.f + 8.f + 18.f + 0.f)),
        std::make_tuple(5UL,  1E-6f ,   1.f - (2.f + 8.f + 18.f + 0.f + 0.03f)),
        std::make_tuple(6UL,  1E-6f ,   1.f - (2.f + 8.f + 18.f + 0.f + 0.03f + -0.04f)),
        std::make_tuple(7UL,  1E-6f ,   1.f - (2.f + 8.f + 18.f + 0.f + 0.03f + -0.04f + 0.09f)),
        std::make_tuple(8UL,  1E-5f ,   1.f - (2.f + 8.f + 18.f + 0.f + 0.03f + -0.04f + 0.09f + 3E-5f)),
        std::make_tuple(9UL,  1E-4f ,   1.f - (2.f + 8.f + 18.f + 0.f + 0.03f + -0.04f + 0.09f + 3E-5f + 2.011009f)),
    };
    for (const auto &element : expected_distances) {
        const auto dim = std::get<0>(element);
        const auto epsilon = std::get<1>(element);
        const auto expected_distance = std::get<2>(element);
        CAPTURE(dim);
        INFO("float32 (no error)");
        const auto space32 = hnswlib::InnerProductSpace(dim);
        const auto result32 = get_distance(a.data(), b.data(), dim, &space32);
        CAPTURE(expected_distance);
        CAPTURE(result32);
        REQUIRE_EQ(result32, expected_distance);
        CAPTURE("float16. Allowed Error: " << epsilon);

        const auto decoder = hnswlib::DecoderFloat16{dim};
        const auto space16 = hnswlib::InnerProductSpaceF16(dim);
        std::vector<uint16_t> a_f16(dim), b_f16(dim);

        to_float16(&decoder, a, a_f16);
        to_float16(&decoder, b, b_f16);

        const auto result16 = get_distance(a_f16.data(), b_f16.data(), dim, &space16);
        CAPTURE(result16);
        REQUIRE(is_approx_euqal(result16, expected_distance, epsilon));
    }
}

TEST_CASE ("Check Inner Product distance computation on large vectors") {
    const std::vector<std::tuple<size_t, float, float>> expected_distances = {
        /*              dim    | epsilon_f32 | epsilon_f16 */
        std::make_tuple(16UL,       1E-6f,      3E-4f),
        std::make_tuple(17UL,       1E-6f,      3E-4f),
        std::make_tuple(32UL,       1E-6f,      3E-4f),
        std::make_tuple(34UL,       1E-6f,      3E-4f),
        std::make_tuple(54UL,       1E-6f,      3E-4f),
        std::make_tuple(100UL,      1E-6f,      3E-4f),
        std::make_tuple(256UL,      1E-5f,      3E-4f),
        std::make_tuple(300UL,      1E-4f,      3E-4f),
        std::make_tuple(1000UL,     1E-4f,      1E-3f),
        std::make_tuple(1024UL,     1E-4f,      1E-3f),
    };
    srand(seed);
    const auto component_ratio = 2.05f;
    for (const auto &element : expected_distances) {
        const auto dim = std::get<0>(element);
        float epsilon32 = std::get<1>(element);
        float epsilon16 = std::get<2>(element);
        CAPTURE(dim);
        std::vector<float> a(dim), b(dim);
        for(int j = 0; j < dim; j++) {
            a[j] = ((float) rand() / RAND_MAX ) / (float)dim;
            b[j] = (component_ratio / a[j]) / (float)dim;
        }
        const auto expected_distance = 1 - component_ratio;
        CAPTURE("float32 Allowed Error: " << epsilon32);

        const auto space32 = hnswlib::InnerProductSpace(dim);
        auto result32 = get_distance(a.data(), b.data(), dim, &space32);
        CAPTURE(expected_distance);
        CAPTURE(result32);
        REQUIRE(is_approx_euqal(result32, expected_distance, epsilon32));

        const auto decoder = hnswlib::DecoderFloat16{dim};
        CAPTURE("float16. Allowed Error: " << epsilon16);

        const auto space16 = hnswlib::InnerProductSpaceF16(dim);
        std::vector<uint16_t> a_f16(dim), b_f16(dim);
        to_float16(&decoder, a, a_f16);
        to_float16(&decoder, b, b_f16);
        auto result16 = get_distance(a_f16.data(), b_f16.data(), dim, &space16);
        CAPTURE(result16);
        REQUIRE(is_approx_euqal(result16, expected_distance, epsilon16));
    }
}