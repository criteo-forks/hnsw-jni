#pragma once
#include "hnswlib.h"

namespace hnswlib {

    static float
    InnerProductF16(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        auto pVect1 = static_cast<const uint16_t*>(pVect1v);
        auto pVect2 = static_cast<const uint16_t*>(pVect2v);
        const auto qty = *static_cast<const size_t*>(qty_ptr);
        float res = 0;
        for (auto i = 0; i < qty; i++) {
            float a = decode_fp16(*pVect1);
            float b = decode_fp16(*pVect2);
            pVect1++;
            pVect2++;
            res += a * b;
        }
        return (1.0f - res);

    }

#if defined(USE_AVX)

// Favor using AVX if available.
    static float
    InnerProductSIMD4ExtF16(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        auto pVect1 = static_cast<const uint16_t*>(pVect1v);
        auto pVect2 = static_cast<const uint16_t*>(pVect2v);
        const auto qty = *static_cast<const size_t*>(qty_ptr);
        const auto qty8 = qty >> 3;
        const auto qty4 = qty >> 2;

        const auto pEnd1 = pVect1 + (qty8 << 3);
        const auto pEnd2 = pVect1 + (qty4 << 2);

        __m128i v1f16, v2f16;
        __m256 v1, v2;
        __m256 sum256 = _mm256_set1_ps(0);

        while (pVect1 < pEnd1) {

            v1f16 = _mm_loadu_si128((const __m128i*)pVect1);
            v1 = _mm256_cvtph_ps(v1f16);
            pVect1 += 8;
            v2f16 = _mm_loadu_si128((const __m128i*)pVect2);
            v2 = _mm256_cvtph_ps(v2f16);
            pVect2 += 8;
            sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));
        }
        __m128 v1r, v2r;
        __m128 sum = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
        while (pVect1 < pEnd2) {
            v1f16 = _mm_loadu_si128((const __m128i*)pVect1);
            v1r = _mm_cvtph_ps(v1f16);
            pVect1 += 4;
            v2f16 = _mm_loadu_si128((const __m128i*)pVect2);
            v2r = _mm_cvtph_ps(v2f16);
            pVect2 += 4;
            sum = _mm_add_ps(sum, _mm_mul_ps(v1r, v2r));
        }

        float PORTABLE_ALIGN32 TmpRes[4];
        _mm_store_ps(TmpRes, sum);
        float dot_product = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
        return 1.0f - dot_product;
    }

#elif defined(USE_SSE)

    static float
    InnerProductSIMD4ExtF16(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        auto pVect1 = static_cast<const uint16_t*>(pVect1v);
        auto pVect2 = static_cast<const uint16_t*>(pVect2v);
        const auto qty = *static_cast<const size_t*>(qty_ptr);
        const auto qty4 = qty >> 2;

        const auto pEnd = pVect1 + (qty4 << 2);

        __m128i v1f16, v2f16;
        __m128 v1, v2;
        __m128 sum = _mm_set1_ps(0);

        while (pVect1 < pEnd) {
            v1f16 = _mm_loadu_si128((const __m128i*)pVect1);
            v1 = _mm_cvtph_ps(v1f16);
            pVect1 += 4;
            v2f16 = _mm_loadu_si128((const __m128i*)pVect2);
            v2 = _mm_cvtph_ps(v2f16);
            pVect2 += 4;
            sum = _mm_add_ps(sum, _mm_mul_ps(v1, v2));
        }
        float PORTABLE_ALIGN32 TmpRes[4];
        _mm_store_ps(TmpRes, sum);
        float dot_product = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

        return 1.0f - dot_product;
    }

#endif

#if defined(USE_AVX)

    static float
    InnerProductSIMD8ExtF16(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        auto pVect1 = static_cast<const uint16_t*>(pVect1v);
        auto pVect2 = static_cast<const uint16_t*>(pVect2v);
        const auto qty = *static_cast<const size_t*>(qty_ptr);
        const auto qty8 = qty >> 3;

        const auto pEnd1 = pVect1 + (qty8 << 3);

        __m128i v1f16, v2f16;
        __m256 v1, v2;
        __m256 sum = _mm256_set1_ps(0);

        while (pVect1 < pEnd1) {
            v1f16 = _mm_loadu_si128((const __m128i*)pVect1);
            v1 = _mm256_cvtph_ps(v1f16);
            pVect1 += 8;
            v2f16 = _mm_loadu_si128((const __m128i*)pVect2);
            v2 = _mm256_cvtph_ps(v2f16);
            pVect2 += 8;
            sum = _mm256_add_ps(sum, _mm256_mul_ps(v1, v2));
        }

        float PORTABLE_ALIGN32 TmpRes[8];
        _mm256_store_ps(TmpRes, sum);
        float dot_product = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

        return 1.0f - dot_product;
    }

#elif defined(USE_SSE)

      static float
      InnerProductSIMD8ExtF16(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        auto pVect1 = static_cast<const uint16_t*>(pVect1v);
        auto pVect2 = static_cast<const uint16_t*>(pVect2v);
        const auto qty = *static_cast<const size_t*>(qty_ptr);
        const auto qty8 = qty >> 3;

        const auto pEnd1 = pVect1 + (qty8 << 3);

        __m128i v1f16, v2f16;
        __m128 v1, v2;
        __m128 sum = _mm_set1_ps(0);

        while (pVect1 < pEnd1) {
            v1f16 = _mm_loadu_si128((const __m128i*)pVect1);
            v1 = _mm_cvtph_ps(v1f16);
            pVect1 += 4;
            v2f16 = _mm_loadu_si128((const __m128i*)pVect2);
            v2 = _mm_cvtph_ps(v2f16);
            pVect2 += 4;
            sum = _mm_add_ps(sum, _mm_mul_ps(v1, v2));

            v1f16 = _mm_loadu_si128((const __m128i*)pVect1);
            v1 = _mm_cvtph_ps(v1f16);
            pVect1 += 4;
            v2f16 = _mm_loadu_si128((const __m128i*)pVect2);
            v2 = _mm_cvtph_ps(v2f16);
            pVect2 += 4;
            sum = _mm_add_ps(sum, _mm_mul_ps(v1, v2));
        }
        float PORTABLE_ALIGN32 TmpRes[4];
        _mm_store_ps(TmpRes, sum);
        float dot_product = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

        return 1.0f - dot_product;
    }

#endif

#if defined(USE_SSE) || defined(USE_AVX)
    static float
    InnerProductSIMD8ExtF16Residuals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        const auto qty = *static_cast<const size_t*>(qty_ptr);
        const auto qty8 = qty >> 3 << 3;
        float res = InnerProductSIMD8ExtF16(pVect1v, pVect2v, &qty8);
        const auto pVect1 = static_cast<const uint16_t *>(pVect1v) + qty8;
        const auto pVect2 = static_cast<const uint16_t *>(pVect2v) + qty8;

        const auto qty_left = qty - qty8;
        const auto res_tail = InnerProductF16(pVect1, pVect2, &qty_left);
        return res + res_tail - 1.0f;
    }

    static float
    InnerProductSIMD4ExtF16Residuals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        const auto qty = *static_cast<const size_t*>(qty_ptr);
        const auto qty4 = qty >> 2 << 2;

        const auto res = InnerProductSIMD4ExtF16(pVect1v, pVect2v, &qty4);
        const auto qty_left = qty - qty4;

        const auto pVect1 = static_cast<const uint16_t *>(pVect1v) + qty4;
        const auto pVect2 = static_cast<const uint16_t *>(pVect2v) + qty4;
        const auto res_tail = InnerProductF16(pVect1, pVect2, &qty_left);

        return res + res_tail - 1.0f;
    }
#endif

    class InnerProductSpaceF16 : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        const size_t data_size_;
        size_t dim_;
    public:
        InnerProductSpaceF16(size_t dim)
        : data_size_(dim * sizeof(uint16_t))
        , dim_(dim) {
            fstdistfunc_ = InnerProductF16;
    #if defined(USE_AVX) || defined(USE_SSE)
            if (dim % 4 == 0)
                fstdistfunc_ = InnerProductSIMD4ExtF16;
            if (dim % 8 == 0)
                fstdistfunc_ = InnerProductSIMD8ExtF16;
            else if (dim > 8)
                fstdistfunc_ = InnerProductSIMD8ExtF16Residuals;
            else if (dim > 4)
                fstdistfunc_ = InnerProductSIMD4ExtF16Residuals;
    #endif
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

        ~InnerProductSpaceF16() {}
    };
}
