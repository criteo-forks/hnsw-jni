#include "common.h"

TEST_CASE("Search Euclidean in the index with <= K items should return all items with their distances") {
    const int M = 15;
    const int efConstruction = 1000;

    int32_t nbItems = 20;
    int32_t K = 20;
    std::vector<int32_t> dims {101, 128};
    std::vector<Precision> precisions {Float32, Float16};
    std::vector<float> epsilons(Precision::num_values);
    epsilons[Float32] = 5E-6f;
    epsilons[Float16] = 5E-4f;

    for(auto dim: dims) {
        for (auto precision: precisions) {
            auto hnsw = Index<float>(Euclidean, dim, precision);
            const auto epsilon = epsilons[precision];
            CAPTURE(precision);
            CAPTURE(epsilon);
            hnsw.initNewIndex(nbItems, M, efConstruction, seed);

            for (int id = 0; id < nbItems; id++) {
                std::vector<float> item(dim);
                for(int i = 0; i < dim; i++) {
                    item[i] = 1.f / ((float)id + 1.f) + 1.f / ((float)i + 1.f);
                }
                hnsw.addItem(item.data(), id);
            }
            const auto query_label = 0;
            auto query = hnsw.getItem(0);
            std::vector<float> decoded_query(dim);
            query = hnsw.decode(query, decoded_query.data());
            std::vector<size_t> labels(K);
            std::vector<float> distances(K);
            std::vector<float*> pointers(K);
            const auto nb_results = hnsw.knnQuery(static_cast<float*>(query), labels.data(), distances.data(), pointers.data(), K);
            const auto distance_to_self = distances[0];
            const auto self = labels[0];
            REQUIRE_EQ(0.0f, distance_to_self);
            REQUIRE_EQ(query_label, self);
            for (int i = 1; i < nb_results; i++) {
                const auto label_actual = labels[i];
                const auto label_expected = i;
                REQUIRE_EQ(label_expected, label_actual);
                const auto dist_actual = distances[i];
                const auto dist_expected = dim * pow(1.f - 1.f / ((float)label_expected + 1.f), 2);
                CAPTURE(dist_expected);
                CAPTURE(dist_actual);
                REQUIRE(is_approx_equal(dist_actual, dist_expected, epsilon));
            }
        }
    }
}

TEST_CASE("Search InnerProduct in the index with <= K items should return all items with their distances") {
    const int M = 15;
    const int efConstruction = 1000;

    int32_t nbItems = 20;
    int32_t K = 20;
    std::vector<int32_t> dims {101, 128};
    std::vector<Precision> precisions {Float32, Float16};
    std::vector<float> epsilons(Precision::num_values);
    epsilons[Float32] = 5E-6f;
    epsilons[Float16] = 5E-4f;
    for(auto dim: dims) {
        for (auto precision: precisions) {
            auto hnsw = Index<float>(InnerProduct, dim, precision);
            const auto epsilon = epsilons[precision];
            CAPTURE(precision);
            CAPTURE(epsilon);
            hnsw.initNewIndex(nbItems, M, efConstruction, seed);

            for (int id = 0; id < nbItems; id++) {
                std::vector<float> item(dim);
                for(int i = 0; i < dim; i++) {
                    item[i] = 1.f / ((float)id  + 1.f);
                }
                hnsw.addItem(item.data(), id);
            }
            const auto query_label = 0;
            auto query = hnsw.getItem(0);
            std::vector<float> decoded_query(dim);
            query = hnsw.decode(static_cast<char*>(query), decoded_query.data());
            std::vector<size_t> labels(K);
            std::vector<float> distances(K);
            std::vector<float*> pointers(K);
            const auto nb_results = hnsw.knnQuery(static_cast<float*>(query), labels.data(), distances.data(), pointers.data(), K);
            const auto distance_to_self = distances[0];
            const auto self = labels[0];
            REQUIRE_EQ(1.f - dim * 1.f, distance_to_self);
            REQUIRE_EQ(query_label, self);
            for (int i = 1; i < nb_results; i++) {
                const auto label_actual = labels[i];
                const auto label_expected = i;
                REQUIRE_EQ(label_expected, label_actual);
                const auto dist_actual = distances[i];
                const auto dist_expected = 1 - dim * 1.f / (label_expected + 1.f);
                CAPTURE(dist_expected);
                CAPTURE(dist_actual);
                REQUIRE(is_approx_equal(dist_actual, dist_expected, epsilon));
            }
        }
    }
}