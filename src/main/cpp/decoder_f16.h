#pragma once
#include "hnswlib.h"

namespace hnswlib {

    static void
    encode_vector_float16(const float* src, uint16_t* dst, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        for (size_t i = 0; i < qty; i++) {
            *dst = encode_fp16(*src);
            src++;
            dst++;
        }
    }

    static void
    decode_float16_vector(const uint16_t* src, float* dst, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        for (size_t i = 0; i < qty; i++) {
            *dst = decode_fp16(*src);
            src++;
            dst++;
        }
    }


#if defined(USE_AVX)

    static void
    encode_vector_float16_SIMD16(const float* src, uint16_t* dst, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        size_t qty16 = qty >> 4;
        const float *end = src + (qty16 << 4);
        __m128i f16;
        __m256 f32;
        while (src < end) {
            f32 = _mm256_loadu_ps(src);
            src += 8;
            f16 = _mm256_cvtps_ph(f32, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
            _mm_storeu_si128((__m128i*)dst, f16);
            dst +=8;

            f32 = _mm256_loadu_ps(src);
            src += 8;
            f16 = _mm256_cvtps_ph(f32, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
            _mm_storeu_si128((__m128i*)dst, f16);
            dst +=8;
        }
    }

    static void
    decode_float16_vector_SIMD16(const uint16_t* src, float* dst, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        size_t qty16 = qty >> 4;
        const uint16_t *end = src + (qty16 << 4);
        __m128i f16;
        __m256 f32;
        while (src < end) {
            f16 = _mm_loadu_si128((__m128i*)src);
            src += 8;
            f32 = _mm256_cvtph_ps(f16);
            _mm256_storeu_ps(dst, f32);
            dst +=8;

            f16 = _mm_loadu_si128((__m128i*)src);
            src += 8;
            f32 = _mm256_cvtph_ps(f16);
            _mm256_storeu_ps(dst, f32);
            dst +=8;
        }
    }

#elif defined(USE_SSE)

    static void
    encode_vector_float16_SIMD16(const float* src, uint16_t* dst, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        size_t qty16 = qty >> 4;
        const float *end = src + (qty16 << 4);
        __m128i f16;
        __m128 f32;
        while (src < end) {
            f32 = _mm_loadu_ps(src);
            src += 4;
            f16 = _mm_cvtps_ph(f32, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
            _mm_storel_epi64((__m128i*)dst, f16);
            dst += 4;

            f32 = _mm_loadu_ps(src);
            src += 4;
            f16 = _mm_cvtps_ph(f32, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
            _mm_storel_epi64((__m128i*)dst, f16);
            dst += 4;

            f32 = _mm_loadu_ps(src);
            src += 4;
            f16 = _mm_cvtps_ph(f32, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
            _mm_storel_epi64((__m128i*)dst, f16);
            dst += 4;

            f32 = _mm_loadu_ps(src);
            src += 4;
            f16 = _mm_cvtps_ph(f32, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
            _mm_storel_epi64((__m128i*)dst, f16);
            dst += 4;
        }
    }

    static void
    decode_float16_vector_SIMD16(const uint16_t* src, float* dst, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        size_t qty16 = qty >> 4;
        const uint16_t *end = src + (qty16 << 4);
        __m128i f16;
        __m128 f32;
        while (src < end) {
            f16 = _mm_loadu_si128((const __m128i*)src);
            src += 4;
            f32 = _mm_cvtph_ps(f16);
            _mm_storeu_ps(dst, f32);
            dst +=4;

            f16 = _mm_loadu_si128((const __m128i*)src);
            src += 4;
            f32 = _mm_cvtph_ps(f16);
            _mm_storeu_ps(dst, f32);
            dst +=4;

            f16 = _mm_loadu_si128((const __m128i*)src);
            src += 4;
            f32 = _mm_cvtph_ps(f16);
            _mm_storeu_ps(dst, f32);
            dst +=4;

            f16 = _mm_loadu_si128((const __m128i*)src);
            src += 4;
            f32 = _mm_cvtph_ps(f16);
            _mm_storeu_ps(dst, f32);
            dst +=4;
        }
    }
#endif

#if defined(USE_SSE) || defined(USE_AVX)
    static void
    encode_vector_float16_SIMD16Residuals(const float* src, uint16_t* dst, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        size_t qty16 = qty >> 4 << 4;
        encode_vector_float16_SIMD16(src, dst, &qty16);

        size_t qty_left = qty - qty16;
        encode_vector_float16(src + qty16, dst + qty16, &qty_left);
    }

    static void
    decode_float16_vector_SIMD16Residuals(const uint16_t* src, float* dst, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        size_t qty16 = qty >> 4 << 4;
        decode_float16_vector_SIMD16(src, dst, &qty16);

        size_t qty_left = qty - qty16;
        decode_float16_vector(src + qty16, dst + qty16, &qty_left);
    }
#endif

#ifdef USE_SSE

    static void
    encode_vector_float16_SIMD4(const float* src, uint16_t* dst, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        size_t qty4 = qty >> 2;
        const float *end = src + (qty4 << 2);
        __m128i f16;
        __m128 f32;
        while (src < end) {
            f32 = _mm_loadu_ps(src);
            src += 4;
            f16 = _mm_cvtps_ph(f32, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
            _mm_storel_epi64((__m128i*)dst, f16);
            dst += 4;
        }
    }

    static void
    decode_float16_vector_SIMD4(const uint16_t* src, float* dst, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        size_t qty4 = qty >> 2;
        const uint16_t *end = src + (qty4 << 2);
        __m128i f16;
        __m128 f32;
        while (src < end) {
            f16 = _mm_loadu_si128((const __m128i*)src);
            src += 4;
            f32 = _mm_cvtph_ps(f16);
            _mm_storeu_ps(dst, f32);
            dst +=4;
        }
    }

    static void
    encode_vector_float16_SIMD4Residuals(const float* src, uint16_t* dst, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        size_t qty4 = qty >> 2 << 2;

        encode_vector_float16_SIMD4(src, dst, &qty4);
        size_t qty_left = qty - qty4;

        encode_vector_float16(src + qty4, dst + qty4, &qty_left);
    }

    static void
    decode_float16_vector_SIMD4Residuals(const uint16_t* src, float* dst, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        size_t qty4 = qty >> 2 << 2;

        decode_float16_vector_SIMD4(src, dst, &qty4);
        size_t qty_left = qty - qty4;

        decode_float16_vector(src + qty4, dst + qty4, &qty_left);
    }
#endif

    template<typename SRC, typename DST>
    using ENCODEFUNC = void(*)(const SRC *, DST *, const void *);

    template<typename SRC, typename DST>
    using DECODEFUNC = void(*)(const DST *, SRC *, const void *);

    class DecoderFloat16 {

        ENCODEFUNC<float, uint16_t> fast_encode_func_;
        DECODEFUNC<float, uint16_t> fast_decode_func_;
        size_t data_size_;
        size_t dim_;
    public:
        DecoderFloat16(size_t dim) {
            fast_encode_func_ = encode_vector_float16;
            fast_decode_func_ = decode_float16_vector;
        #if defined(USE_SSE) || defined(USE_AVX)
            if (dim % 16 == 0) {
                fast_encode_func_ = encode_vector_float16_SIMD16;
                fast_decode_func_ = decode_float16_vector_SIMD16;
            }
            else if (dim % 4 == 0) {
                fast_encode_func_ = encode_vector_float16_SIMD4;
                fast_decode_func_ = decode_float16_vector_SIMD4;
            }
            else if (dim > 16) {
                fast_encode_func_ = encode_vector_float16_SIMD16Residuals;
                fast_decode_func_ = decode_float16_vector_SIMD16Residuals;
            }
            else if (dim > 4) {
                fast_encode_func_ = encode_vector_float16_SIMD4Residuals;
                fast_decode_func_ = decode_float16_vector_SIMD4Residuals;
            }
        #endif
            dim_ = dim;
            data_size_ = dim * sizeof(float);
        }

        size_t get_data_size() {
            return data_size_;
        }

        ENCODEFUNC<float, uint16_t> get_encode_func() {
            return fast_encode_func_;
        }

        DECODEFUNC<float, uint16_t> get_decode_func() {
            return fast_decode_func_;
        }

        void *get_dist_func_param() {
            return &dim_;
        }

        ~DecoderFloat16() {}
    };
}