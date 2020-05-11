#pragma once
#include <unordered_map>
#include <fstream>
#include <memory>

namespace hnswlib {

    template<typename dist_t>
    class BruteforceSearchAlg {
        public:
            explicit BruteforceSearchAlg(SpaceInterface <dist_t> *s) {
                fstdistfunc_ = s->get_dist_func();
                dist_func_param_ = s->get_dist_func_param();
            }

            std::priority_queue<std::pair<dist_t, tableint >> searchKnn(const void *query_data, size_t k, const AlgorithmInterface<dist_t>* index) const {
                std::priority_queue<std::pair<dist_t, tableint>> topResults;
                const auto nbItems = index->getNbItems();
                for (auto i = 0; i < nbItems; i++) {
                    const auto dist = fstdistfunc_(query_data, index->getDataByInternalId(i), dist_func_param_);
                    if (i < k || dist <= topResults.top().first) {
                        topResults.push(std::pair<dist_t, tableint>(dist, i));
                        if (topResults.size() > k)
                            topResults.pop();
                    }
                }
                return topResults;
            }

        private:
            DISTFUNC <dist_t> fstdistfunc_;
            void *dist_func_param_;
    };

    template<typename dist_t>
    class BruteforceSearch : public AlgorithmInterface<dist_t> {
    public:
        explicit BruteforceSearch(SpaceInterface <dist_t> *s)
            : BruteforceSearch(s, 0) {
        }

        BruteforceSearch(SpaceInterface<dist_t> *s, const std::string &location)
            : BruteforceSearch(s, 0) {
            loadIndex(location, s);
        }

        BruteforceSearch(SpaceInterface <dist_t> *s, size_t maxElements = 0) {
            maxelements_ = maxElements;
            data_size_ = s->get_data_size();
            alg_ = std::unique_ptr<BruteforceSearchAlg<dist_t>>(new BruteforceSearchAlg<dist_t>(s));
            size_per_element_ = data_size_ + sizeof(labeltype);
            data_ = (char *) malloc(maxElements * size_per_element_);
            cur_element_count = 0;
        }

        ~BruteforceSearch() {
            free(data_);
        }

        char *data_;
        size_t maxelements_;
        size_t cur_element_count;
        size_t size_per_element_;

        size_t data_size_;
        DISTFUNC <dist_t> fstdistfunc_;
        void *dist_func_param_;
        std::unique_ptr<BruteforceSearchAlg<dist_t>> alg_;

        std::unordered_map<labeltype, tableint> dict_external_to_internal;

        void addPoint(void *datapoint, labeltype label) {
            if(dict_external_to_internal.count(label))
                throw std::runtime_error("Ids have to be unique");


            if (cur_element_count >= maxelements_) {
                throw std::runtime_error("The number of elements exceeds the specified limit\n");
            };
            memcpy(data_ + size_per_element_ * cur_element_count + data_size_, &label, sizeof(labeltype));
            memcpy(getDataByInternalId(cur_element_count), datapoint, data_size_);
            dict_external_to_internal[label]=cur_element_count;

            cur_element_count++;
        };

        void removePoint(labeltype cur_external) {
            size_t cur_c=dict_external_to_internal[cur_external];

            dict_external_to_internal.erase(cur_external);

            labeltype label=*((labeltype*)(data_ + size_per_element_ * (cur_element_count-1) + data_size_));
            dict_external_to_internal[label]=cur_c;
            memcpy(data_ + size_per_element_ * cur_c,
                   data_ + size_per_element_ * (cur_element_count-1),
                   data_size_+sizeof(labeltype));
            cur_element_count--;
        }


        std::priority_queue<std::pair<dist_t, tableint >> searchKnn(const void *query_data, size_t k) const {
            return alg_->searchKnn(query_data, k, this);
        }

        void saveIndex(const std::string &location) {
            std::ofstream output(location, std::ios::binary);
            std::streampos position;

            writeBinaryPOD(output, maxelements_);
            writeBinaryPOD(output, size_per_element_);
            writeBinaryPOD(output, cur_element_count);

            output.write(data_, maxelements_ * size_per_element_);
            output.close();
        }

        void loadIndex(const std::string &location, SpaceInterface<dist_t> *s) {
            std::ifstream input(location, std::ios::binary);
            std::streampos position;

            readBinaryPOD(input, maxelements_);
            readBinaryPOD(input, size_per_element_);
            readBinaryPOD(input, cur_element_count);

            data_ = (char *) malloc(maxelements_ * size_per_element_);
            input.read(data_, maxelements_ * size_per_element_);
            input.close();

            for (size_t i = 0; i < cur_element_count; i++) {
                dict_external_to_internal[getExternalLabel(i)]=i;
            }
        }

        inline char *getDataByInternalId(tableint internal_id) const {
            return (data_ + internal_id * size_per_element_);
        }

        inline labeltype getExternalLabel(tableint internal_id) const {
            return *((labeltype *) (data_ + size_per_element_ * internal_id + data_size_));
        }

        inline size_t getNbItems() const {
            return cur_element_count;
        }

        inline std::unordered_map<labeltype, tableint> * getLabelLookup() {
            return &dict_external_to_internal;
        }
    };
}
