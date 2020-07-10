#include "common.h"

TEST_CASE("MinMaxRange tests") {
    const auto dim = 3;
    const auto n = 4;
    std::vector<std::vector<float>> data {
        std::vector<float> {10, 11, 13.5},
        std::vector<float> {20, -1, 22.02},
        std::vector<float> {35, 0.1, 12},
        std::vector<float> {0.4, 10, 17.1},
    };

    SUBCASE("Min/max set correctly") {
        const std::vector<float> expected_min {0.4, -1, 12};
        const std::vector<float> expected_max {35, 11, 22.02};
        hnswlib::MinMaxRange range(dim);
        for (auto &row: data) {
            range.add(row.data());
        }
        for (int i = 0; i < dim; i++) {
            CAPTURE(i);
            CAPTURE(expected_min[i]);
            CAPTURE(expected_max[i]);
            REQUIRE_EQ(expected_min[i], range.min_[i]);
            REQUIRE_EQ(expected_max[i], range.max_[i]);
        }
    }

    SUBCASE("Decode params: min - as is, diff - norm by 2^8-1=255 (for float8)") {
        const std::vector<float> expected_diff {
            34.6f / 255,
            12.f / 255,
            10.02f / 255
        };

        const std::vector<float> expected_min {
            0.4f,
            -1.f,
            12.f
        };

        hnswlib::MinMaxRange range(dim);
        for (auto &row: data) {
            range.add(row.data());
        }
        const auto params = range.get_trained_params<uint8_t>();
        for (int i = 0; i < dim; i++) {
            CAPTURE(i);
            CAPTURE(expected_min[i]);
            CAPTURE(expected_diff[i]);
            REQUIRE_EQ(expected_min[i], params->min[i]);
            REQUIRE_EQ(expected_diff[i], params->diff[i]);
        }
        delete params;
    }
}

TEST_CASE("Float8 encoding-decoding on small vectors (1 - 16)") {
    const std::vector<float> a   {0.9,   0.5,   -0.24528086, 0.013, 0.67, 1.337, -0.14,  1.001,  0.5,  -0.45, 0.3,   0.2,  -0.2, 0.1,   0.05f,  0.1506};
    const std::vector<float> min {0.1,   -0.8,  -0.27057678, -0.4,  -0.9, -1.45, -1.565, -0.22,  0.1,  -0.45, 0.01,  -0.9, -0.3, -1.f,  0.f,    -2.4};
    const std::vector<float> max {1.7,   1.25,  0.25777041,  0.5,   1.3,  2.033, 0.34,   1.001,  0.7,  -0.45, 0.789, 3.4,  0.32, 0.55f, 0.08f,  0.16};
    const std::vector<float> e   {4e-3,  2e-3,  4e-4,        6e-5,  2e-4, 6e-4,  2e-3,   1e-30,  1e-7, 1e-30,  2e-4, 4e-3, 4e-4, 2e-4,  2e-4,   7e-4f};
    srand(seed);
    for (size_t dim = 1; dim <= a.size(); dim++) {
        CAPTURE(dim);
        hnswlib::MinMaxRange range(dim);
        range.add(min.data());
        range.add(max.data());
        const auto params = range.get_trained_params<uint8_t>();
        std::vector<uint8_t> a_f8(dim);
        to_float8(a, a_f8, params);
        std::vector<float> a_f32(dim);
        to_float32(dim, a_f8, a_f32, params);
        for (auto i = 0; i < dim; i++) {
            auto actual = a_f32[i];
            auto expected = a[i];
            auto epsilon = e[i];
            CAPTURE(epsilon);
            CAPTURE(actual);
            CAPTURE(expected);
            CAPTURE(min[i]);
            CAPTURE(max[i]);
            REQUIRE(max[i] >= actual);
            REQUIRE(min[i] <= actual);
            const auto res = is_approx_equal(actual, expected, epsilon);
            REQUIRE(res);
        }
    }
}

TEST_CASE("Float8 encoding-decoding on large vectors") {
    const std::vector<size_t> dimensions = {16, 17, 32, 34, 54, 100, 256, 300, 1000, 1024};
    srand(seed);
    const auto epsilon = 4e-3;
    for (const auto &dim : dimensions) {
        CAPTURE(dim);
        CAPTURE(epsilon);
        std::vector<float> a(dim), min(dim), max(dim);
        for(int i = 0; i < dim; i++) {
            min[i] = get_random_float(-1, 1);
            max[i] = get_random_float(min[i], 1);
            a[i] = get_random_float(min[i], max[i]);
        }
        hnswlib::MinMaxRange range(min, max);
        const auto params = range.get_trained_params<uint8_t>();
        std::vector<uint8_t> a_f8(dim);
        to_float8(a, a_f8, params);
        std::vector<float> a_f32(dim);
        to_float32(dim, a_f8, a_f32, params);
        for(int i = 0; i < dim; i++) {
            auto actual = a_f32[i];
            auto expected = a[i];
            CAPTURE(epsilon);
            CAPTURE(actual);
            CAPTURE(expected);
            CAPTURE(min[i]);
            CAPTURE(max[i]);
            REQUIRE(is_approx_equal(actual, expected, epsilon));
        }
    }
}
