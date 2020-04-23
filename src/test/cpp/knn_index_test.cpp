#include "doctest.h"
#include "hnswindex.h"
#include <tuple>

TEST_CASE("Serialize and deserialize indices") {
    const int seed = 42;
    const int M = 15;
    const int efConstruction = 1000;

    int32_t nbItems = 1000;
    int32_t dim = 100;
    const std::vector<std::tuple<Precision, Distance>> indices {
        std::make_tuple(Float16, Euclidean),
        std::make_tuple(Float16, InnerProduct),
        std::make_tuple(Float32, Euclidean),
        std::make_tuple(Float32, InnerProduct),
    };
    const auto epsilon32 = 1E-30f;
    const auto epsilon16 = 5E-4f;
    for(auto item: indices) {
        const auto precision = std::get<0>(item);
        const auto distance = std::get<1>(item);
        auto hnsw = Index<float>(distance, dim, precision);
        CAPTURE(distance);
        CAPTURE(precision);
        CAPTURE(epsilon32);
        CAPTURE(epsilon16);
        hnsw.initNewIndex(nbItems, M, efConstruction, seed);
        REQUIRE_EQ(0, hnsw.getNbItems());
        for(int id = 0; id < nbItems; id++) {
            float value = 0;
            if(id > 0) {
                value = 1/(float)id;
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

        const std::vector<Index<float>*> loaded_indices {
            &hnsw_iso,
            &hnsw16,
        };

        for(auto loaded_index: loaded_indices) {
            REQUIRE_EQ(nbItems, loaded_index->getNbItems());
            for(size_t id = 0; id < nbItems; id++) {
                float expected = 0;
                if(id > 0) {
                    expected = 1/(float)id;
                }
                auto item_ptr = loaded_index->getItem(id);
                std::vector<float> item(dim);
                auto epsilon = epsilon32;
                if(loaded_index->precision == Float16) {
                    loaded_index->decodeFloat16((uint16_t*)item_ptr, item.data());
                    item_ptr = item.data();
                    epsilon = epsilon16;
                }
                float* item_ptr_float32 = (float*)item_ptr;
                for(size_t i = 0; i < dim; i++) {
                    auto actual = item_ptr_float32[i];
                    REQUIRE(expected == doctest::Approx(actual).epsilon(epsilon));
                }
            }
        }
    }
}
