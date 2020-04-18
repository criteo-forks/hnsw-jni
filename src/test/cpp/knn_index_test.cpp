#include "doctest.h"
#include "hnswindex.h"

TEST_CASE("Serialize and deserialize indices") {
    const int seed = 42;
    const int M = 15;
    const int efConstruction = 1000;

    int32_t nbItems = 1000;
    int32_t dim = 100;
    std::vector<std::pair<Precision, Distance>> p {
        std::make_pair(Float16, Euclidean),
        std::make_pair(Float32, Euclidean),
        std::make_pair(Float32, InnerProduct),
    };
    std::vector<float> e {
        5E-4,
        1E-30,
        1E-30,
    };
    for(size_t j = 0; j < p.size(); j++) {
        auto precision = p[j].first;
        auto distance = p[j].second;
        auto hnsw = new Index<float>(distance, dim, precision);
        auto epsilon = e[j];
        CAPTURE(distance);
        CAPTURE(precision);
        CAPTURE(epsilon);
        hnsw->initNewIndex(nbItems, M, efConstruction, seed);
        REQUIRE_EQ(0, hnsw->getNbItems());
        for(int id = 0; id < nbItems; id++) {
            float value = 0;
            if(id > 0) {
                value = 1/(float)id;
            }
            std::vector<float> item(dim, value);
            hnsw->addItem(item.data(), id);
        }
        REQUIRE_EQ(nbItems, hnsw->getNbItems());
        std::string indexPath = "./hnsw-" + std::to_string(precision) + "-" + std::to_string(distance) + ".bin";
        hnsw->saveIndex(indexPath);

        hnsw->loadIndex(indexPath);
        REQUIRE_EQ(nbItems, hnsw->getNbItems());
        for(size_t id = 0; id < nbItems; id++) {
            float expected = 0;
            if(id > 0) {
                expected = 1/(float)id;
            }
            auto item_ptr = hnsw->getItem(id);
            std::vector<float> item(dim);
            if(precision == Float16) {
                hnsw->decodeFloat16((uint16_t*)item_ptr, item.data());
                item_ptr = item.data();
            }
            float* item_ptr_float32 = (float*)item_ptr;
            for(size_t i = 0; i < dim; i++) {
                auto actual = item_ptr_float32[i];
                REQUIRE(expected == doctest::Approx(actual).epsilon(epsilon));
            }
        }
        delete hnsw;
    }
}