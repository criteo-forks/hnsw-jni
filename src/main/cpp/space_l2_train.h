#pragma once
#include "encoding.h"
#include "hnswlib.h"

namespace hnswlib {

    template<typename TARG1, typename TARG2>
    static float
    L2Sqr_trained(const void *pVect1v, const void *pVect2v, const void *diff_ptr) {
        auto pVect1 = static_cast<const TARG1*>(pVect1v);
        auto pVect2 = static_cast<const TARG2*>(pVect2v);
        const auto params = static_cast<const TrainParams*>(diff_ptr);
        const auto qty = params->dim;
        auto pMin = params->min;
        auto pDiff = params->diff;

        float res = 0;
        for (size_t i = 0; i < qty; i++) {
            const auto v1 = load_component(pVect1, pMin, pDiff);
            const auto v2 = load_component(pVect2, pMin, pDiff);
            const auto t = v1 - v2;
            res += t * t;
            pVect1++;
            pVect2++;
            pMin++;
            pDiff++;
        }
        return res;
    }

#if defined(USE_AVX)

    // Favor using AVX if available.

    template<typename TARG1, typename TARG2>
    static float
    L2SqrSIMD8Ext_trained(const void *pVect1v, const void *pVect2v, const void *diff_ptr) {
        auto pVect1 = static_cast<const TARG1*>(pVect1v);
        auto pVect2 = static_cast<const TARG2*>(pVect2v);
        const auto params = static_cast<const TrainParams*>(diff_ptr);
        const auto qty = params->dim;
        auto pMin = params->min;
        auto pDiff = params->diff;
        const auto pEnd1 = pVect1 + qty;
        auto sum = _mm256_set1_ps(0);

        while (pVect1 < pEnd1) {
            const auto v1 = load_component_avx(pVect1, pMin, pDiff);
            const auto v2 = load_component_avx(pVect2, pMin, pDiff);
            const auto t = v1 - v2;
            sum += t * t;
            pVect1 += 8;
            pVect2 += 8;
            pMin += 8;
            pDiff += 8;
        }

        float PORTABLE_ALIGN32 TmpRes[8];
        _mm256_store_ps(TmpRes, sum);
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
    }

    template<typename TARG1, typename TARG2>
    static float
    L2SqrSIMD8ExtResiduals_trained(const void *pVect1v, const void *pVect2v, const void *diff_ptr) {
        const auto params = static_cast<const TrainParams*>(diff_ptr);
        const auto qty = params->dim;
        const auto qty8 = qty >> 3 << 3;
        TrainParams params8(params->min, params->diff, qty8, true);
        const auto res = L2SqrSIMD8Ext_trained<TARG1, TARG2>(pVect1v, pVect2v, &params8);
        const auto pVect1 = static_cast<const TARG1*>(pVect1v) + qty8;
        const auto pVect2 = static_cast<const TARG2*>(pVect2v) + qty8;

        const auto qty_left = qty - qty8;
        TrainParams params_left(params->min + qty8, params->diff + qty8, qty_left, true);
        const auto res_tail = L2Sqr_trained<TARG1, TARG2>(pVect1, pVect2, &params_left);
        return (res + res_tail);
    }
#endif


#ifdef USE_SSE
    template<typename TARG1, typename TARG2>
    static float
    L2SqrSIMD4Ext_trained(const void *pVect1v, const void *pVect2v, const void *diff_ptr) {
        auto pVect1 = static_cast<const TARG1*>(pVect1v);
        auto pVect2 = static_cast<const TARG2*>(pVect2v);
        const auto params = static_cast<const TrainParams*>(diff_ptr);
        const auto qty = params->dim;
        auto pMin = params->min;
        auto pDiff = params->diff;

        const auto pEnd1 = pVect1 + qty;

        __m128 sum = _mm_set1_ps(0);

        while (pVect1 < pEnd1) {
            const auto v1 = load_component_sse(pVect1, pMin, pDiff);
            const auto v2 = load_component_sse(pVect2, pMin, pDiff);
            pVect1 += 4;
            pVect2 += 4;
            pMin += 4;
            pDiff += 4;
            const auto t = v1 - v2;
            sum += t * t;
        }
        float PORTABLE_ALIGN32 TmpRes[8];
        _mm_store_ps(TmpRes, sum);
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
    }

    template<typename TARG1, typename TARG2>
    static float
    L2SqrSIMD4ExtResiduals_trained(const void *pVect1v, const void *pVect2v, const void *diff_ptr) {
        const auto params = static_cast<const TrainParams*>(diff_ptr);
        const auto qty = params->dim;
        const auto qty4 = qty >> 2 << 2;
        TrainParams params4(params->min, params->diff, qty4, true);
        const auto res = L2SqrSIMD4Ext_trained<TARG1, TARG2>(pVect1v, pVect2v, &params4);
        const auto qty_left = qty - qty4;
        const auto pVect1 = static_cast<const TARG1*>(pVect1v) + qty4;
        const auto pVect2 = static_cast<const TARG2*>(pVect2v) + qty4;
        TrainParams params_left(params->min + qty4, params->diff + qty4, qty_left, true);
        const auto res_tail = L2Sqr_trained<TARG1, TARG2>(pVect1, pVect2, &params_left);

        return (res + res_tail);
    }
#endif

    template<typename TCOMPR=float>
    class L2TrainedSpace : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        DISTFUNC<float> fstdist_search_func_;
        const size_t data_size_;
        size_t dim_;
        MinMaxRange range_per_component;
        TrainParams params = TrainParams(dim_);

    public:
        explicit L2TrainedSpace(size_t dim)
        : data_size_(dim * sizeof(TCOMPR))
        , dim_(dim), range_per_component(dim) {
            fstdistfunc_ = L2Sqr_trained<TCOMPR, TCOMPR>;
            fstdist_search_func_ = L2Sqr_trained<float, TCOMPR>;
        #if defined(USE_SSE) || defined(USE_AVX)
            if (dim % 8 == 0) {
                fstdistfunc_ = L2SqrSIMD8Ext_trained<TCOMPR, TCOMPR>;
                fstdist_search_func_ = L2SqrSIMD8Ext_trained<float, TCOMPR>;
            }
            else if (dim % 4 == 0) {
                fstdistfunc_ = L2SqrSIMD4Ext_trained<TCOMPR, TCOMPR>;
                fstdist_search_func_ = L2SqrSIMD4Ext_trained<float, TCOMPR>;
            }
            else if (dim > 8) {
                fstdistfunc_ = L2SqrSIMD8ExtResiduals_trained<TCOMPR, TCOMPR>;
                fstdist_search_func_ = L2SqrSIMD8ExtResiduals_trained<float, TCOMPR>;
            }
            else if (dim > 4) {
                fstdistfunc_ = L2SqrSIMD4ExtResiduals_trained<TCOMPR, TCOMPR>;
                fstdist_search_func_ = L2SqrSIMD4ExtResiduals_trained<float, TCOMPR>;
            }
        #endif
        }

        size_t get_data_size() override {
            return data_size_;
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
            range_per_component.update_trained_params<TCOMPR>(params);
        }

        void *get_dist_func_param() override {
            return &params;
        }

        ~L2TrainedSpace() override = default;
    };


}