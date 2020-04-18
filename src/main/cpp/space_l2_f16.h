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
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
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
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
    }

#endif

#if defined(USE_SSE) || defined(USE_AVX)
    static float
    L2SqrSIMD16ExtF16Residuals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        size_t qty16 = qty >> 4 << 4;
        float res = L2SqrSIMD16ExtF16(pVect1v, pVect2v, &qty16);
        uint16_t *pVect1 = (uint16_t *) pVect1v + qty16;
        uint16_t *pVect2 = (uint16_t *) pVect2v + qty16;

        size_t qty_left = qty - qty16;
        float res_tail = L2SqrF16(pVect1, pVect2, &qty_left);
        return (res + res_tail);
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
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
    }

    static float
    L2SqrSIMD4ExtF16Residuals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        size_t qty4 = qty >> 2 << 2;

        float res = L2SqrSIMD4ExtF16(pVect1v, pVect2v, &qty4);
        size_t qty_left = qty - qty4;

        uint16_t *pVect1 = (uint16_t *) pVect1v + qty4;
        uint16_t *pVect2 = (uint16_t *) pVect2v + qty4;
        float res_tail = L2SqrF16(pVect1, pVect2, &qty_left);

        return (res + res_tail);
    }
#endif

    class L2SpaceF16 : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;
        size_t dim_;
    public:
        L2SpaceF16(size_t dim) {
            fstdistfunc_ = L2SqrF16;
        #if defined(USE_SSE) || defined(USE_AVX)
            if (dim % 16 == 0)
                fstdistfunc_ = L2SqrSIMD16ExtF16;
            else if (dim % 4 == 0)
                fstdistfunc_ = L2SqrSIMD4ExtF16;
            else if (dim > 16)
                fstdistfunc_ = L2SqrSIMD16ExtF16Residuals;
            else if (dim > 4)
                fstdistfunc_ = L2SqrSIMD4ExtF16Residuals;
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

        void *get_dist_func_param() {
            return &dim_;
        }

        ~L2SpaceF16() {}
    };
}