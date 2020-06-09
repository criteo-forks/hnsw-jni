#pragma once
#include <algorithm>
#include "float16.h"

namespace hnswlib {

    template<typename src_type, typename dst_type, dst_type(*func)(src_type)>
    static void inline
    encode_decode_vector(const src_type* src, dst_type* dst, const size_t qty) {
        std::transform(src, src + qty, dst, func);
    }

    static inline float load_component(const float *component) {
        return *component;
    }

    static inline float load_component(const uint16_t *component) {
        return decode_fp16(*component);
    }

#if defined(USE_AVX)

    static inline __m256 load_component_avx(const float *component) {
        return _mm256_loadu_ps(component);
    }

    static inline __m256 load_component_avx(const uint16_t *component) {
        const auto tmp = _mm_loadu_si128((const __m128i *) component);
        return _mm256_cvtph_ps(tmp);
    }
#endif

#if defined(USE_SSE) || defined(USE_AVX)

    static inline __m128 load_component_sse(const float *component) {
        return _mm_loadu_ps(component);
    }

    static inline __m128 load_component_sse(const uint16_t *component) {
        const auto tmp = _mm_loadu_si128((const __m128i*)component);
        return _mm_cvtph_ps(tmp);
    }
#endif
}