#pragma once
#include <algorithm>
#include "float16.h"

namespace hnswlib {
    static inline float load_component(const float *component) {
        return *component;
    }

    static inline float load_component(const uint16_t *component) {
        return decode_fp16(*component);
    }

    // Encoding F32 -> F16
    static inline void encode_component(const float* src, uint16_t *dst) {
        *dst = encode_fp16(*src);
    }

    // Decoding F16 -> F32
    static inline void encode_component(const uint16_t* src, float *dst) {
        *dst = decode_fp16(*src);
    }

    template<typename SRC, typename DST, void(*encode_func)(const SRC*, DST*), int step>
    static void
    encode_decode_vector(const SRC* src, DST* dst, const size_t* qty_ptr) {
        const SRC *end = src + *qty_ptr;
        while (src < end) {
            encode_func(src, dst);
            src += step;
            dst += step;
        }
    }

#if defined(USE_AVX)

    static inline __m256 load_component_avx(const float *component) {
        return _mm256_loadu_ps(component);
    }

    static inline __m256 load_component_avx(const uint16_t *component) {
        const auto tmp = _mm_loadu_si128((const __m128i *) component);
        return _mm256_cvtph_ps(tmp);
    }

    // Encoding F32 -> F16
    static inline void encode_component_avx(const float* src, uint16_t *dst) {
        const auto f32 = load_component_avx(src);
        const auto f16 = _mm256_cvtps_ph(f32, FC16_CONVERSION_FLAGS);
        _mm_storeu_si128((__m128i*)dst, f16);
    }

    // Decoding F16 -> F32
    static inline void encode_component_avx(const uint16_t* src, float *dst) {
        const auto f32 = load_component_avx(src);
        _mm256_storeu_ps(dst, f32);
    }

    template<typename SRC, typename DST>
    static void
    encode_decode_vector_avx_residuals(const SRC* src, DST* dst, const size_t* qty_ptr) {
        const auto qty = *qty_ptr;
        const auto qty8 = qty >> 3 << 3;
        encode_decode_vector<SRC, DST, encode_component_avx, 8>(src, dst, &qty8);

        const auto qty_left = qty - qty8;
        encode_decode_vector<SRC, DST, encode_component, 1>(src + qty8, dst + qty8, &qty_left);
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

    // Encoding F32 -> F16
    static inline void encode_component_sse(const float* src, uint16_t *dst) {
        const auto f32 = load_component_sse(src);
        const auto f16 = _mm_cvtps_ph(f32, FC16_CONVERSION_FLAGS);
        _mm_storel_epi64((__m128i*)dst, f16);
    }

    // Decoding F16 -> F32
    static inline void encode_component_sse(const uint16_t* src, float *dst) {
        const auto f32 = load_component_sse(src);
        _mm_storeu_ps(dst, f32);
    }

    template<typename SRC, typename DST>
    static void
    encode_decode_vector_sse_residuals(const SRC* src, DST* dst, const size_t* qty_ptr) {
        const auto qty = *qty_ptr;
        const auto qty4 = qty >> 2 << 2;
        encode_decode_vector<SRC, DST, encode_component_sse, 4>(src, dst, &qty4);
        const auto qty_left = qty - qty4;
        encode_decode_vector<SRC, DST, encode_component, 1>(src + qty4, dst + qty4, &qty_left);
    }
#endif

    template<typename SRC, typename DST>
    static inline
    DECODEFUNC<SRC, DST, size_t> get_fast_encode_func(size_t dim) {
        auto func = encode_decode_vector<SRC, DST, encode_component, 1>;
#if defined(USE_SSE)
        if (dim % 4 == 0) func = encode_decode_vector<SRC, DST, encode_component_sse, 4>;
        else if (dim > 4) func = encode_decode_vector_sse_residuals<SRC, DST>;
#endif
#if defined(USE_AVX)
        if (dim % 8 == 0) func = encode_decode_vector<SRC, DST, encode_component_avx, 8>;
        else if (dim > 8) func = encode_decode_vector_avx_residuals<SRC, DST>;
#endif
        return func;
    }
}