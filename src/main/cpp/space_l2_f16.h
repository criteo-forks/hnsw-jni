#pragma once
#include "hnswlib.h"

namespace hnswlib {

    static float L2SqrF16(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        uint16_t *pVect1 = (uint16_t *) pVect1v;
        uint16_t *pVect2 = (uint16_t *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);
        float res = 0;
        for (size_t i = 0; i < qty; i++) {
            float a = decode_fp16(*pVect1);
            float b = decode_fp16(*pVect2);

            float t = a - b;
            pVect1++;
            pVect2++;
            res += t * t;
        }
        return (res);
    }

    static void encode_vector_float16(const void* from, const void* to, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        float* src = (float*)from;
        uint16_t* dst = (uint16_t*)to;
        for (size_t i = 0; i < qty; i++) {
            *dst = encode_fp16(*src);
            src++;
            dst++;
        }
    }

    static void decode_float16_vector(const void* from, const void* to, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        uint16_t* src = (uint16_t*)from;
        float* dst = (float*)to;
        for (size_t i = 0; i < qty; i++) {
            *dst = decode_fp16(*src);
            src++;
            dst++;
        }
    }

#if defined(USE_AVX)

    // Favor using AVX if available.
    static float
    L2SqrSIMD16ExtF16(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        uint16_t *pVect1 = (uint16_t *) pVect1v;
        uint16_t *pVect2 = (uint16_t *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);
        float PORTABLE_ALIGN32 TmpRes[8];
        size_t qty16 = qty >> 4;

        const uint16_t *pEnd1 = pVect1 + (qty16 << 4);

        __m128i v1f16, v2f16;
        __m256 diff, v1, v2;
        __m256 sum = _mm256_set1_ps(0);

        while (pVect1 < pEnd1) {
            v1f16 = _mm_loadu_si128((const __m128i*)pVect1);
            v1 = _mm256_cvtph_ps(v1f16);
            pVect1 += 8;
            v2f16 = _mm_loadu_si128((const __m128i*)pVect2);
            v2 = _mm256_cvtph_ps(v2f16);
            pVect2 += 8;
            diff = _mm256_sub_ps(v1, v2);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

            v1f16 = _mm_loadu_si128((const __m128i*)pVect1);
            v1 = _mm256_cvtph_ps(v1f16);
            pVect1 += 8;
            v2f16 = _mm_loadu_si128((const __m128i*)pVect2);
            v2 = _mm256_cvtph_ps(v2f16);
            pVect2 += 8;
            diff = _mm256_sub_ps(v1, v2);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
        }

        _mm256_store_ps(TmpRes, sum);
        float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
        size_t qty_left = qty - (qty16 << 4);
        float res_tail = L2SqrF16(pVect1, pVect2, &qty_left);
        return (res + res_tail);
    }

    static void encode_vector_float16_SIMD16(const void* from, const void* to, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        float* src = (float*)from;
        uint16_t* dst = (uint16_t*)to;
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
        size_t qty_left = qty - (qty16 << 4);
        encode_vector_float16(src, dst, &qty_left);
    }

    static void decode_float16_vector_SIMD16(const void* from, const void* to, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        uint16_t* src = (uint16_t*)from;
        float* dst = (float*)to;
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
        size_t qty_left = qty - (qty16 << 4);
        decode_float16_vector(src, dst, &qty_left);
    }

#elif defined(USE_SSE)

    static float
    L2SqrSIMD16ExtF16(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        uint16_t *pVect1 = (uint16_t *) pVect1v;
        uint16_t *pVect2 = (uint16_t *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);
        float PORTABLE_ALIGN32 TmpRes[8];
        size_t qty16 = qty >> 4;

        const uint16_t *pEnd1 = pVect1 + (qty16 << 4);

        __m128i v1f16, v2f16;
        __m128 diff, v1, v2;
        __m128 sum = _mm_set1_ps(0);

        while (pVect1 < pEnd1) {
            v1f16 = _mm_loadu_si128((const __m128i*)pVect1);
            v1 = _mm_cvtph_ps(v1f16);
            pVect1 += 4;
            v2f16 = _mm_loadu_si128((const __m128i*)pVect2);
            v2 = _mm_cvtph_ps(v2f16);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));


            v1f16 = _mm_loadu_si128((const __m128i*)pVect1);
            v1 = _mm_cvtph_ps(v1f16);
            pVect1 += 4;
            v2f16 = _mm_loadu_si128((const __m128i*)pVect2);
            v2 = _mm_cvtph_ps(v2f16);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));


            v1f16 = _mm_loadu_si128((const __m128i*)pVect1);
            v1 = _mm_cvtph_ps(v1f16);
            pVect1 += 4;
            v2f16 = _mm_loadu_si128((const __m128i*)pVect2);
            v2 = _mm_cvtph_ps(v2f16);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

            v1f16 = _mm_loadu_si128((const __m128i*)pVect1);
            v1 = _mm_cvtph_ps(v1f16);
            pVect1 += 4;
            v2f16 = _mm_loadu_si128((const __m128i*)pVect2);
            v2 = _mm_cvtph_ps(v2f16);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        }

        _mm_store_ps(TmpRes, sum);
        float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
        size_t qty_left = qty - (qty16 << 4);
        float res_tail = L2SqrF16(pVect1, pVect2, &qty_left);
        return (res + res_tail);
    }

    static void encode_vector_float16_SIMD16(const void* from, const void* to, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        float* src = (float*)from;
        uint16_t* dst = (uint16_t*)to;
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
        size_t qty_left = qty - (qty16 << 4);
        encode_vector_float16(src, dst, &qty_left);
    }

    static void decode_float16_vector_SIMD16(const void* from, const void* to, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        uint16_t* src = (uint16_t*)from;
        float* dst = (float*)to;
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
        size_t qty_left = qty - (qty16 << 4);
        decode_float16_vector(src, dst, &qty_left);
    }
#endif


#ifdef USE_SSE
    static float
    L2SqrSIMD4ExtF16(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float PORTABLE_ALIGN32 TmpRes[8];
        uint16_t *pVect1 = (uint16_t *) pVect1v;
        uint16_t *pVect2 = (uint16_t *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);


        size_t qty4 = qty >> 2;

        const uint16_t *pEnd1 = pVect1 + (qty4 << 2);

        __m128i v1f16, v2f16;
        __m128 diff, v1, v2;
        __m128 sum = _mm_set1_ps(0);

        while (pVect1 < pEnd1) {
            v1f16 = _mm_loadu_si128((const __m128i*)pVect1);
            v1 = _mm_cvtph_ps(v1f16);
            pVect1 += 4;
            v2f16 = _mm_loadu_si128((const __m128i*)pVect2);
            v2 = _mm_cvtph_ps(v2f16);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        }
        _mm_store_ps(TmpRes, sum);
        float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
        size_t qty_left = qty - (qty4 << 2);
        float res_tail = L2SqrF16(pVect1, pVect2, &qty_left);

        return (res + res_tail);
    }

    static void encode_vector_float16_SIMD4(const void* from, const void* to, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        float* src = (float*)from;
        uint16_t* dst = (uint16_t*)to;
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
        size_t qty_left = qty - (qty4 << 2);
        encode_vector_float16(src, dst, &qty_left);
    }

    static void decode_float16_vector_SIMD4(const void* from, const void* to, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        uint16_t* src = (uint16_t*)from;
        float* dst = (float*)to;
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
        size_t qty_left = qty - (qty4 << 2);
        decode_float16_vector(src, dst, &qty_left);
    }
#endif
    using ENCODEFUNC = void(*)(const void *, const void *, const void *);
    using DECODEFUNC = void(*)(const void *, const void *, const void *);

    class L2SpaceF16 : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        ENCODEFUNC fast_encode_func_;
        DECODEFUNC fast_decode_func_;
        size_t data_size_;
        size_t dim_;
    public:
        L2SpaceF16(size_t dim) {
            fstdistfunc_ = L2SqrF16;
            fast_encode_func_ = encode_vector_float16;
            fast_decode_func_ = decode_float16_vector;
        #if defined(USE_SSE) || defined(USE_AVX)
            if (dim >= 16) {
                fstdistfunc_ = L2SqrSIMD16ExtF16;
                fast_encode_func_ = encode_vector_float16_SIMD16;
                fast_decode_func_ = decode_float16_vector_SIMD16;
            }
            else if (dim >= 4) {
                fstdistfunc_ = L2SqrSIMD4ExtF16;
                fast_encode_func_ = encode_vector_float16_SIMD4;
                fast_decode_func_ = decode_float16_vector_SIMD4;
            }
        #endif

            dim_ = dim;
            data_size_ = dim * sizeof(uint16_t);
        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<float> get_dist_func() {
            return fstdistfunc_;
        }

        ENCODEFUNC get_encode_func() {
            return fast_encode_func_;
        }

        DECODEFUNC get_decode_func() {
            return fast_decode_func_;
        }

        void *get_dist_func_param() {
            return &dim_;
        }

        ~L2SpaceF16() {}
    };
}