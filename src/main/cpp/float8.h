#pragma once
#include "hnswlib.h"
#include <limits>
#include <memory>
#include <cassert>

namespace hnswlib {

    struct TrainParams {
        const size_t dim;
        const float* min;
        const float* diff;
        const bool is_copy;

        explicit TrainParams(const float* min, const float* diff, const size_t dim, const bool is_copy=false)
        : dim(dim), min(min), diff(diff), is_copy(is_copy) {}

        explicit TrainParams(const std::vector<float> &min_v, const std::vector<float> &diff_v)
        : dim(min_v.size()), is_copy(false) {
            const auto data_size = sizeof(float) * dim;
            void *min_ptr, *diff_ptr;
            if (
                posix_memalign(&min_ptr, 32, data_size) ||
                posix_memalign(&diff_ptr, 32, data_size)
            ) {
                throw std::runtime_error("Couldn't allocate TrainParams buffers");
            }

            memcpy(min_ptr, min_v.data(), data_size);
            memcpy(diff_ptr, diff_v.data(), data_size);
            min = static_cast<float*>(min_ptr);
            diff = static_cast<float*>(diff_ptr);
        }

        virtual ~TrainParams() {
            /* TODO: Store both min and diff within the space itself and avoid passing context as argument
             * However this requires some refactoring. For now we need this hack to avoid releasing memory
             * by when creating copies of context in `_resiguals` methods. */
            if(!is_copy) {
                free(const_cast<float *>(min));
                free(const_cast<float *>(diff));
            }
        }
    };


    class MinMaxRange {
        public:
        size_t nb_examples;

        explicit MinMaxRange(size_t dim)
            : nb_examples(0)
            , min_(dim, std::numeric_limits<float>::max())
            , max_(dim, std::numeric_limits<float>::lowest())
            , dim_(dim)
        {
            assert(dim > 0);
        }

        explicit MinMaxRange(
            const std::vector<float> &min,
            const std::vector<float> &max
        )
            : nb_examples(2)
            , min_(min)
            , max_(max)
            , dim_(min.size())
        {
            assert(!min.empty());
            assert(min.size() == max.size());
        }

        void add(const float* vector) {
            std::transform(vector, vector + dim_, min_.cbegin(), min_.begin(), (binary_op)std::min);
            std::transform(vector, vector + dim_, max_.cbegin(), max_.begin(), (binary_op)std::max);
            nb_examples += 1;
        }

        template<typename TARGET>
        TrainParams* get_trained_params() {
            std::vector<float> diff(dim_);
            const auto max_value = std::numeric_limits<TARGET>::max();
            std::transform(
                max_.cbegin(), max_.cend(),
                min_.cbegin(),
                diff.begin(),
                [&max_value](float max_i, float min_i) {
                    // Normalizing diff by max value of scalar type (255 for uint8_t/float8)
                    // to avoid extra multiplication at decoding time
                    return (max_i - min_i) / max_value;
                }
            );
            return new TrainParams(min_, diff);
        }

        std::vector<float> min_;
        std::vector<float> max_;

        private:
        size_t dim_;
        using binary_op = const float& (*) (const float&, const float&);
    };

    // Encoding F32 -> F8
    // This method is called at index load time only so, doesn't require high performance
    // which is also quite hard to achieve using AVX only due to lack of specialized convert
    // instructions from uint8_t to float.
    static inline void encode_component(const float* src, uint8_t *dst, const float* min, const float *diff) {
        // TODO/OPT: align to the border values of  [0, 255] if less /greater
        *dst = static_cast<uint8_t>(round((*src - *min) / *diff));
    }

    template<typename SRC, typename DST, void(*encode_func)(const SRC*, DST*, const float*, const float*), int step>
    static inline void encode_trained_vector(const SRC* src, DST* dst, const TrainParams* param_ptr) {
        const auto *end = src + param_ptr->dim;
        auto min = param_ptr->min;
        auto diff = param_ptr->diff;
        while (src < end) {
            encode_func(src, dst, min, diff);
            src += step;
            dst += step;
            min += step;
            diff += step;
        }
    }

    static inline float load_component(const float *src, const float* min, const float *diff) {
        return *src;
    }

    // Decode F8 -> F32
    static inline float load_component(const uint8_t *src, const float* min, const float *diff) {
        return *diff * *src + *min;
    }

    static inline void encode_component(const uint8_t *src, float* dst, const float* min, const float *diff) {
        *dst = load_component(src, min, diff);
    }

#ifdef USE_AVX
    static inline __m256 load_component_avx(const float *src, const float* min, const float *diff) {
        return _mm256_loadu_ps(src);
    }

    static inline __m256 load_component_avx(const uint8_t *src, const float* min, const float *diff) {
        const auto c8 = *(uint64_t*)(src);
        __m128i c4lo = _mm_cvtepu8_epi32 (_mm_set1_epi32(c8));
        __m128i c4hi = _mm_cvtepu8_epi32 (_mm_set1_epi32(c8 >> 32));
        __m256i i8 = _mm256_castsi128_si256 (c4lo);
        i8 = _mm256_insertf128_si256 (i8, c4hi, 1);
        const auto src_f32 = _mm256_cvtepi32_ps(i8);
        const auto diff_f32 = _mm256_load_ps(diff);
        const auto min_f32 = _mm256_load_ps(min);
        return diff_f32 * src_f32 + min_f32;
    }

    // Decode F8 -> F32
    static inline void encode_component_avx(const uint8_t *src, float* dst, const float* min, const float *diff) {
        const auto f32 = load_component_avx(src, min, diff);
        _mm256_storeu_ps(dst, f32);
    }

    static void decode_trained_vector_avx_residuals(const uint8_t* src, float* dst, const TrainParams* param_ptr) {
        const auto qty = param_ptr->dim;
        const auto qty8 = qty >> 3 << 3;
        TrainParams params8(param_ptr->min, param_ptr->diff, qty8, true);
        encode_trained_vector<uint8_t, float, encode_component_avx, 8>(src, dst, &params8);
        const auto qty_left = qty - qty8;
        TrainParams params_left(param_ptr->min + qty8, param_ptr->diff + qty8, qty_left, true);
        encode_trained_vector<uint8_t, float, encode_component, 1>(src + qty8, dst + qty8, &params_left);
    }
#endif

#if defined(USE_SSE) || defined(USE_AVX)
    static inline __m128 load_component_sse(const float *src, const float* min, const float *diff) {
        return _mm_loadu_ps(src);
    }

    static inline __m128 load_component_sse(const uint8_t *src, const float* min, const float *diff) {
        const auto u4 = *(uint32_t*)(src);
        const auto i4 = _mm_cvtepu8_epi32 (_mm_set1_epi32(u4));
        const auto src_f32 = _mm_cvtepi32_ps(i4);
        const auto diff_f32 = _mm_load_ps(diff);
        const auto min_f32 = _mm_load_ps(min);
        return diff_f32 * src_f32 + min_f32;
    }

    // Decode F8 -> F32
    static inline void encode_component_sse(const uint8_t *src, float* dst, const float* min, const float *diff) {
        const auto f32 = load_component_sse(src, min, diff);
        _mm_storeu_ps(dst, f32);
    }

    static void decode_trained_vector_sse_residuals(const uint8_t* src, float* dst, const TrainParams* param_ptr) {
        const auto qty = param_ptr->dim;
        const auto qty4 = qty >> 2 << 2;
        TrainParams params8(param_ptr->min, param_ptr->diff, qty4, true);
        encode_trained_vector<uint8_t, float, encode_component_sse, 4>(src, dst, &params8);
        const auto qty_left = qty - qty4;
        TrainParams params_left(param_ptr->min + qty4, param_ptr->diff + qty4, qty_left, true);
        encode_trained_vector<uint8_t, float, encode_component, 1>(src + qty4, dst + qty4, &params_left);
    }
#endif

    static inline
    DECODEFUNC<uint8_t, float, TrainParams> get_fast_decode_trained_func(size_t dim) {
        auto func = encode_trained_vector<uint8_t, float, encode_component, 1>;
#if defined(USE_SSE)
        if (dim % 4 == 0) func = encode_trained_vector<uint8_t, float, encode_component_sse, 4>;
        else if (dim > 4) func = decode_trained_vector_sse_residuals;
#endif
#if defined(USE_AVX)
        if (dim % 8 == 0) func = encode_trained_vector<uint8_t, float, encode_component_avx, 8>;
        else if (dim > 8) func = decode_trained_vector_avx_residuals;
#endif
        return func;
    }
}