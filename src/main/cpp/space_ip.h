#pragma once
#include "hnswlib.h"

namespace hnswlib {

    template<typename TARG1, typename TARG2>
    static float
    InnerProduct(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        auto pVect1 = static_cast<const TARG1*>(pVect1v);
        auto pVect2 = static_cast<const TARG2*>(pVect2v);
        const auto qty = *static_cast<const size_t*>(qty_ptr);

        float res = 0;

        for (size_t i = 0; i < qty; i++) {
            res += load_component(pVect1) * load_component(pVect2);
            pVect1++;
            pVect2++;
        }
        return (1.0f - res);
    }

#if defined(USE_AVX)

// Favor using AVX if available.
    template<typename TARG1, typename TARG2>
    static float
    InnerProductSIMD4Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        auto pVect1 = static_cast<const TARG1*>(pVect1v);
        auto pVect2 = static_cast<const TARG2*>(pVect2v);
        const auto qty = *static_cast<const size_t*>(qty_ptr);
        const auto qty8 = qty >> 3;
        const auto qty4 = qty >> 2;

        const auto pEnd1 = pVect1 + (qty8 << 3);
        const auto pEnd2 = pVect1 + (qty4 << 2);

        __m256 sum256 = _mm256_set1_ps(0);

        while (pVect1 < pEnd1) {

            __m256 v1 = load_component_avx(pVect1);
            pVect1 += 8;
            __m256 v2 = load_component_avx(pVect2);
            pVect2 += 8;
            sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));
        }

        __m128 v1, v2;
        __m128 sum_prod = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));

        while (pVect1 < pEnd2) {
            v1 = load_component_sse(pVect1);
            pVect1 += 4;
            v2 = load_component_sse(pVect2);
            pVect2 += 4;
            sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
        }
        float PORTABLE_ALIGN32 TmpRes[8];
        _mm_store_ps(TmpRes, sum_prod);
        float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];;
        return 1.0f - sum;
}

#elif defined(USE_SSE)

    template<typename TARG1, typename TARG2>
    static float
    InnerProductSIMD4Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        auto pVect1 = static_cast<const TARG1*>(pVect1v);
        auto pVect2 = static_cast<const TARG2*>(pVect2v);
        const auto qty = *static_cast<const size_t*>(qty_ptr);

        const auto pEnd2 = pVect1 + qty;

        __m128 v1, v2;
        __m128 sum_prod = _mm_set1_ps(0);

        while (pVect1 < pEnd2) {
            v1 = load_component_sse(pVect1);
            pVect1 += 4;
            v2 = load_component_sse(pVect2);
            pVect2 += 4;
            sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
        }
        float PORTABLE_ALIGN32 TmpRes[8];
        _mm_store_ps(TmpRes, sum_prod);
        float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

        return 1.0f - sum;
    }

#endif

#if defined(USE_AVX)

    template<typename TARG1, typename TARG2>
    static float
    InnerProductSIMD8Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        auto pVect1 = static_cast<const TARG1*>(pVect1v);
        auto pVect2 = static_cast<const TARG2*>(pVect2v);
        const auto qty = *static_cast<const size_t*>(qty_ptr);
        const auto pEnd1 = pVect1 + qty;

        __m256 sum256 = _mm256_set1_ps(0);

        while (pVect1 < pEnd1) {

            __m256 v1 = load_component_avx(pVect1);
            pVect1 += 8;
            __m256 v2 = load_component_avx(pVect2);
            pVect2 += 8;
            sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));
        }

        float PORTABLE_ALIGN32 TmpRes[8];
        _mm256_store_ps(TmpRes, sum256);
        float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

        return 1.0f - sum;
    }

#elif defined(USE_SSE)


      template<typename TARG1, typename TARG2>
      static float
      InnerProductSIMD8Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        auto pVect1 = static_cast<const TARG1*>(pVect1v);
        auto pVect2 = static_cast<const TARG2*>(pVect2v);
        const auto qty = *static_cast<const size_t*>(qty_ptr);
        const auto pEnd1 = pVect1 + qty;

        __m128 v1, v2;
        __m128 sum_prod = _mm_set1_ps(0);

        while (pVect1 < pEnd1) {
            v1 = load_component_sse(pVect1);
            pVect1 += 4;
            v2 = load_component_sse(pVect2);
            pVect2 += 4;
            sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

            v1 = load_component_sse(pVect1);
            pVect1 += 4;
            v2 = load_component_sse(pVect2);
            pVect2 += 4;
            sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
        }
        float PORTABLE_ALIGN32 TmpRes[8];
        _mm_store_ps(TmpRes, sum_prod);
        float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

        return 1.0f - sum;
    }

#endif

#if defined(USE_SSE) || defined(USE_AVX)

    template<typename TARG1, typename TARG2>
    static float
    InnerProductSIMD8ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        const auto qty = *static_cast<const size_t*>(qty_ptr);
        const auto qty8 = qty >> 3 << 3;
        const auto res = InnerProductSIMD8Ext<TARG1, TARG2>(pVect1v, pVect2v, &qty8);
        const auto pVect1 = static_cast<const TARG1*>(pVect1v) + qty8;
        const auto pVect2 = static_cast<const TARG2*>(pVect2v) + qty8;

        const auto qty_left = qty - qty8;
        const auto res_tail = InnerProduct<TARG1, TARG2>(pVect1, pVect2, &qty_left);
        return res + res_tail - 1.0f;
    }

    template<typename TARG1, typename TARG2>
    static float
    InnerProductSIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        const auto qty = *static_cast<const size_t*>(qty_ptr);
        const auto qty4 = qty >> 2 << 2;
        const auto res = InnerProductSIMD4Ext<TARG1, TARG2>(pVect1v, pVect2v, &qty4);
        const auto qty_left = qty - qty4;
        const auto pVect1 = static_cast<const TARG1*>(pVect1v) + qty4;
        const auto pVect2 = static_cast<const TARG2*>(pVect2v) + qty4;
        const auto res_tail = InnerProduct<TARG1, TARG2>(pVect1, pVect2, &qty_left);

        return res + res_tail - 1.0f;
    }
#endif

    template<typename TCOMPR=float>
    class InnerProductSpace : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        const size_t data_size_;
        size_t dim_;
    public:
        explicit InnerProductSpace(size_t dim)
        : data_size_(dim * sizeof(TCOMPR))
        , dim_(dim) {
            fstdistfunc_ = InnerProduct<TCOMPR, TCOMPR>;
    #if defined(USE_AVX) || defined(USE_SSE)
            if (dim % 4 == 0) {
                fstdistfunc_ = InnerProductSIMD4Ext<TCOMPR, TCOMPR>;
            }
            if (dim % 8 == 0) {
                fstdistfunc_ = InnerProductSIMD8Ext<TCOMPR, TCOMPR>;
            }
            else if (dim > 8) {
                fstdistfunc_ = InnerProductSIMD8ExtResiduals<TCOMPR, TCOMPR>;
            }
            else if (dim > 4) {
                fstdistfunc_ = InnerProductSIMD4ExtResiduals<TCOMPR, TCOMPR>;
            }
    #endif
        }

        size_t get_data_size() override {
            return data_size_;
        }

        DISTFUNC<float> get_dist_func() override {
            return fstdistfunc_;
        }

        void *get_dist_func_param() override {
            return &dim_;
        }

        ~InnerProductSpace() override = default;
    };


}
