#pragma once
#include "hnswlib.h"

namespace hnswlib {

    template<typename TARG1, typename TARG2>
    static float
    InnerProduct_trained(const void *pVect1v, const void *pVect2v, const void *diff_ptr) {
        auto pVect1 = static_cast<const TARG1*>(pVect1v);
        auto pVect2 = static_cast<const TARG2*>(pVect2v);
        const auto params = static_cast<const TrainParams*>(diff_ptr);
        const auto qty = params->dim;
        auto pMin = params->min;
        auto pDiff = params->diff;
        const auto pEnd = pVect1 + qty;
        float res = 0;

        while (pVect1 < pEnd) {
            const auto v1 = load_component(pVect1, pMin, pDiff);
            const auto v2 = load_component(pVect2, pMin, pDiff);
            res += v1 * v2;
            pVect1++;
            pVect2++;
            pMin++;
            pDiff++;
        }
        return (1.0f - res);
    }

#if defined(USE_AVX)

// Favor using AVX if available.
    template<typename TARG1, typename TARG2>
    static float
    InnerProductSIMD4Ext_trained(const void *pVect1v, const void *pVect2v, const void *diff_ptr) {
        auto pVect1 = static_cast<const TARG1*>(pVect1v);
        auto pVect2 = static_cast<const TARG2*>(pVect2v);
        const auto params = static_cast<const TrainParams*>(diff_ptr);
        const auto qty = params->dim;
        auto pMin = params->min;
        auto pDiff = params->diff;
        const auto qty8 = qty >> 3;
        const auto qty4 = qty >> 2;

        const auto pEnd1 = pVect1 + (qty8 << 3);
        const auto pEnd2 = pVect1 + (qty4 << 2);

        __m256 sum256 = _mm256_set1_ps(0);

        while (pVect1 < pEnd1) {

            __m256 v1 = load_component_avx(pVect1, pMin, pDiff);
            __m256 v2 = load_component_avx(pVect2, pMin, pDiff);
            pVect1 += 8;
            pVect2 += 8;
            pMin += 8;
            pDiff += 8;
            sum256 += v1 * v2;
        }

        __m128 v1, v2;
        __m128 sum_prod = _mm256_extractf128_ps(sum256, 0) + _mm256_extractf128_ps(sum256, 1);

        while (pVect1 < pEnd2) {
            v1 = load_component_sse(pVect1, pMin, pDiff);
            v2 = load_component_sse(pVect2, pMin, pDiff);
            pVect1 += 4;
            pVect2 += 4;
            pMin += 4;
            pDiff += 4;
            sum_prod += v1 * v2;
        }
        float PORTABLE_ALIGN32 TmpRes[8];
        _mm_store_ps(TmpRes, sum_prod);
        float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];;
        return 1.0f - sum;
}

#elif defined(USE_SSE)

    template<typename TARG1, typename TARG2>
    static float
    InnerProductSIMD4Ext_trained(const void *pVect1v, const void *pVect2v, const void *diff_ptr) {
        auto pVect1 = static_cast<const TARG1*>(pVect1v);
        auto pVect2 = static_cast<const TARG2*>(pVect2v);
        const auto params = static_cast<const TrainParams*>(diff_ptr);
        const auto qty = params->dim;
        auto pMin = params->min;
        auto pDiff = params->diff;

        const auto pEnd2 = pVect1 + qty;
        __m128 sum_prod = _mm_set1_ps(0);

        while (pVect1 < pEnd2) {
            const auto v1 = load_component_sse(pVect1, pMin, pDiff);
            const auto v2 = load_component_sse(pVect2, pMin, pDiff);
            pVect1 += 4;
            pVect2 += 4;
            pMin += 4;
            pDiff += 4;
            sum_prod += v1 * v2;
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
    InnerProductSIMD8Ext_trained(const void *pVect1v, const void *pVect2v, const void *diff_ptr) {
        auto pVect1 = static_cast<const TARG1*>(pVect1v);
        auto pVect2 = static_cast<const TARG2*>(pVect2v);
        const auto params = static_cast<const TrainParams*>(diff_ptr);
        const auto qty = params->dim;
        auto pMin = params->min;
        auto pDiff = params->diff;
        const auto pEnd1 = pVect1 + qty;

        __m256 sum256 = _mm256_set1_ps(0);

        while (pVect1 < pEnd1) {
            __m256 v1 = load_component_avx(pVect1, pMin, pDiff);
            __m256 v2 = load_component_avx(pVect2, pMin, pDiff);
            pVect1 += 8;
            pVect2 += 8;
            pMin += 8;
            pDiff += 8;

            sum256 += v1 * v2;
        }

        float PORTABLE_ALIGN32 TmpRes[8];
        _mm256_store_ps(TmpRes, sum256);
        float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

        return 1.0f - sum;
    }

#endif

#if defined(USE_SSE) || defined(USE_AVX)

    template<typename TARG1, typename TARG2>
    static float
    InnerProductSIMD8ExtResiduals_trained(const void *pVect1v, const void *pVect2v, const void *diff_ptr) {
        const auto params = static_cast<const TrainParams*>(diff_ptr);
        const auto qty = params->dim;
        const auto qty8 = qty >> 3 << 3;
        TrainParams params8(params->min, params->diff, qty8, true);
        const auto res = InnerProductSIMD8Ext_trained<TARG1, TARG2>(pVect1v, pVect2v, &params8);
        const auto pVect1 = static_cast<const TARG1*>(pVect1v) + qty8;
        const auto pVect2 = static_cast<const TARG2*>(pVect2v) + qty8;

        const auto qty_left = qty - qty8;
        TrainParams params_left(params->min + qty8, params->diff + qty8, qty_left, true);
        const auto res_tail = InnerProduct_trained<TARG1, TARG2>(pVect1, pVect2, &params_left);
        return res + res_tail - 1.0f;
    }

    template<typename TARG1, typename TARG2>
    static float
    InnerProductSIMD4ExtResiduals_trained(const void *pVect1v, const void *pVect2v, const void *diff_ptr) {
        const auto params = static_cast<const TrainParams*>(diff_ptr);
        const auto qty = params->dim;
        const auto qty4 = qty >> 2 << 2;
        TrainParams params4(params->min, params->diff, qty4, true);
        const auto res = InnerProductSIMD4Ext_trained<TARG1, TARG2>(pVect1v, pVect2v, &params4);
        const auto qty_left = qty - qty4;
        const auto pVect1 = static_cast<const TARG1*>(pVect1v) + qty4;
        const auto pVect2 = static_cast<const TARG2*>(pVect2v) + qty4;
        TrainParams params_left(params->min + qty4, params->diff + qty4, qty_left, true);
        const auto res_tail = InnerProduct_trained<TARG1, TARG2>(pVect1, pVect2, &params_left);

        return res + res_tail - 1.0f;
    }
#endif

    template<typename TCOMPR=uint8_t>
    class InnerProductTrainedSpace : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        DISTFUNC<float> fstdist_search_func_;
        size_t dim_;
        MinMaxRange range_per_component;

    public:
        explicit InnerProductTrainedSpace(size_t dim)
        : dim_(dim), range_per_component(dim) {
            fstdistfunc_ = InnerProduct_trained<TCOMPR, TCOMPR>;
            fstdist_search_func_ = InnerProduct_trained<float, TCOMPR>;
    #if defined(USE_AVX) || defined(USE_SSE)
            if (dim % 4 == 0) {
                fstdistfunc_ = InnerProductSIMD4Ext_trained<TCOMPR, TCOMPR>;
                fstdist_search_func_ = InnerProductSIMD4Ext_trained<float, TCOMPR>;
            }
            if (dim % 8 == 0) {
                fstdistfunc_ = InnerProductSIMD8Ext_trained<TCOMPR, TCOMPR>;
                fstdist_search_func_ = InnerProductSIMD8Ext_trained<float, TCOMPR>;
            }
            else if (dim > 8) {
                fstdistfunc_ = InnerProductSIMD8ExtResiduals_trained<TCOMPR, TCOMPR>;
                fstdist_search_func_ = InnerProductSIMD8ExtResiduals_trained<float, TCOMPR>;
            }
            else if (dim > 4) {
                fstdistfunc_ = InnerProductSIMD4ExtResiduals_trained<TCOMPR, TCOMPR>;
                fstdist_search_func_ = InnerProductSIMD4ExtResiduals_trained<float, TCOMPR>;
            }
    #endif
        }

        size_t get_data_size() override {
            return dim_ * sizeof(TCOMPR);
        }

        DISTFUNC<float> get_dist_func() override {
            return fstdistfunc_;
        }

        DISTFUNC<float> get_search_dist_func() const override {
            return fstdist_search_func_;
        }

        bool needs_initialization() const override {
            return range_per_component.nb_examples == 0;
        }

        void train(const float* vectors) override {
            range_per_component.add(vectors);
        }

        void *get_dist_func_param() override {
            return range_per_component.get_trained_params<TCOMPR>();
        }

        void initialize_params(const void* range_ptr) override {
            range_per_component = *static_cast<const MinMaxRange*>(range_ptr);
        }

        ~InnerProductTrainedSpace() override = default;
    };


}
