#include "common.h"

TEST_CASE("Serialize and deserialize indices") {
    const int M = 15;
    const int efConstruction = 1000;

    int32_t nbItems = 1000;
    std::vector<int32_t> dims {101, 128};
    const std::vector<std::tuple<Precision, Distance>> indices {
        std::make_tuple(Float16, Euclidean),
        std::make_tuple(Float16, InnerProduct),
        std::make_tuple(Float32, Euclidean),
        std::make_tuple(Float32, InnerProduct),
    };
    const auto epsilon32 = 1E-30f;
    const auto epsilon16 = 5E-4f;
    for(auto dim: dims) {
        for (auto item: indices) {
            const auto precision = std::get<0>(item);
            const auto distance = std::get<1>(item);
            auto hnsw = Index<float>(distance, dim, precision);
            CAPTURE(dim);
            CAPTURE(distance);
            CAPTURE(precision);
            CAPTURE(epsilon32);
            CAPTURE(epsilon16);
            hnsw.initNewIndex(nbItems, M, efConstruction, seed);
            REQUIRE_EQ(0, hnsw.getNbItems());
            for (int id = 0; id < nbItems; id++) {
                float value = 0;
                if (id > 0) {
                    value = 1 / (float) id;
                }
                std::vector<float> item(dim, value);
                hnsw.addItem(item.data(), id);
            }
            REQUIRE_EQ(nbItems, hnsw.getNbItems());
            const auto indexPath = "./hnsw-" + std::to_string(precision) + "-" + std::to_string(distance) + ".bin";
            hnsw.saveIndex(indexPath);

            // Loading in the same format
            auto hnsw_iso = Index<float>(distance, dim, precision);
            hnsw_iso.loadIndex(indexPath);

            // Loading as float16
            auto hnsw16 = Index<float>(distance, dim, Float16);
            hnsw16.loadIndex(indexPath);

            const std::vector<Index<float> *> loaded_indices{
                &hnsw_iso,
                &hnsw16,
            };

            for (auto loaded_index: loaded_indices) {
                REQUIRE_EQ(nbItems, loaded_index->getNbItems());
                for (size_t id = 0; id < nbItems; id++) {
                    float expected = 0;
                    if (id > 0) {
                        expected = 1 / (float) id;
                    }
                    auto item_ptr = loaded_index->getItem(id);
                    std::vector<float> item(dim);
                    auto epsilon = loaded_index->precision == Float32? epsilon32 : epsilon16;
                    item_ptr = loaded_index->decode(item_ptr, item.data());
                    const auto *item_ptr_float32 = reinterpret_cast<float *>(item_ptr);
                    for (size_t i = 0; i < dim; i++) {
                        auto actual = item_ptr_float32[i];
                        REQUIRE(expected == doctest::Approx(actual).epsilon(epsilon));
                    }
                }
            }
        }
    }
}

TEST_CASE("Deserialize Float32 as Float8 indices") {
    const int M = 15;
    const int efConstruction = 1000;

    int32_t nbItems = 1000;
    std::vector<int32_t> dims {101, 128};
    const std::vector<Distance> distances {
        InnerProduct,
        Euclidean,
    };
    const auto epsilon = 4E-3f;
    for(auto dim: dims) {
        for (auto distance: distances) {
            auto hnsw = Index<float>(distance, dim, Float32);
            CAPTURE(dim);
            CAPTURE(distance);
            hnsw.initNewIndex(nbItems, M, efConstruction, seed);
            auto range = hnswlib::MinMaxRange(dim);
            std::vector<float> min(dim), max(dim);
            for(int i = 0; i < dim; i++) {
                min[i] = get_random_float(-1, 1);
                max[i] = get_random_float(min[i], 1);
            }
            REQUIRE_EQ(0, hnsw.getNbItems());
            std::vector<std::vector<float>> vectors;
            for (int id = 0; id < nbItems; id++) {
                std::vector<float> item(dim);
                for(int i = 0; i < dim; i++) {
                    item[i] = get_random_float(min[i], max[i]);
                }
                vectors.push_back(item);
                range.add(item.data());
                hnsw.addItem(item.data(), id);
            }
            REQUIRE_EQ(nbItems, hnsw.getNbItems());
            const auto indexPath = "./hnsw-dist" + std::to_string(distance) + ".bin";
            hnsw.saveIndex(indexPath);

            // Loading as float8 providing range explicitly
            auto hnsw_float8_explicit = Index<float>(distance, dim, Float8);
            hnsw_float8_explicit.space->initialize_params(&range);
            hnsw_float8_explicit.loadIndex(indexPath);

            // Loading as float8 without range
            auto hnsw_float8_implicit = Index<float>(distance, dim, Float8);
            hnsw_float8_implicit.loadIndex(indexPath);

            const std::vector<Index<float> *> loaded_indices{
                &hnsw_float8_explicit,
                &hnsw_float8_implicit,
            };

            for (auto loaded_index: loaded_indices) {
                REQUIRE_EQ(nbItems, loaded_index->getNbItems());
                for (size_t id = 0; id < nbItems; id++) {
                    const auto expected = vectors[id];
                    const auto item = loaded_index->getItem(id);
                    std::vector<float> item_v(dim);
                    const auto item_ptr = loaded_index->decode(item, item_v.data());
                    for (size_t i = 0; i < dim; i++) {
                        CAPTURE(expected[i]);
                        CAPTURE(item_ptr[i]);
                        CAPTURE(range.min_[i]);
                        CAPTURE(range.max_[i]);
                        REQUIRE(expected[i] == doctest::Approx(item_ptr[i]).epsilon(epsilon));
                    }
                }
            }
        }
    }
}
