#pragma once
#include "encoding.h"
#include "hnswlib.h"

namespace hnswlib {

    template<typename TARG1, typename TARG2>
    static float
    L2Sqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        auto pVect1 = static_cast<const TARG1*>(pVect1v);
        auto pVect2 = static_cast<const TARG2*>(pVect2v);
        const auto qty = *static_cast<const size_t*>(qty_ptr);

        float res = 0;
        for (size_t i = 0; i < qty; i++) {
            const auto t = load_component(pVect1) - load_component(pVect2);
            pVect1++;
            pVect2++;
            res += t * t;
        }
        return (res);
    }

#if defined(USE_AVX)

    // Favor using AVX if available.

    template<typename TARG1, typename TARG2>
    static float
    L2SqrSIMD8Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        auto pVect1 = static_cast<const TARG1*>(pVect1v);
        auto pVect2 = static_cast<const TARG2*>(pVect2v);
        const auto qty = *static_cast<const size_t*>(qty_ptr);

        const auto pEnd1 = pVect1 + qty;

        __m256 diff, v1, v2;
        __m256 sum = _mm256_set1_ps(0);

        while (pVect1 < pEnd1) {
            v1 = load_component_avx(pVect1);
            pVect1 += 8;
            v2 = load_component_avx(pVect2);
            pVect2 += 8;
            diff = _mm256_sub_ps(v1, v2);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
        }

        float PORTABLE_ALIGN32 TmpRes[8];
        _mm256_store_ps(TmpRes, sum);
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
    }

    template<typename TARG1, typename TARG2>
    static float
    L2SqrSIMD8ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        const auto qty = *static_cast<const size_t*>(qty_ptr);
        const auto qty8 = qty >> 3 << 3;
        const auto res = L2SqrSIMD8Ext<TARG1, TARG2>(pVect1v, pVect2v, &qty8);
        const auto pVect1 = static_cast<const TARG1*>(pVect1v) + qty8;
        const auto pVect2 = static_cast<const TARG2*>(pVect2v) + qty8;

        const auto qty_left = qty - qty8;
        const auto res_tail = L2Sqr<TARG1, TARG2>(pVect1, pVect2, &qty_left);
        return (res + res_tail);
    }
#endif


#ifdef USE_SSE
    template<typename TARG1, typename TARG2>
    static float
    L2SqrSIMD4Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        auto pVect1 = static_cast<const TARG1*>(pVect1v);
        auto pVect2 = static_cast<const TARG2*>(pVect2v);
        const auto qty = *static_cast<const size_t*>(qty_ptr);
        const auto pEnd1 = pVect1 + qty;

        __m128 diff, v1, v2;
        __m128 sum = _mm_set1_ps(0);

        while (pVect1 < pEnd1) {
            v1 = load_component_sse(pVect1);
            pVect1 += 4;
            v2 = load_component_sse(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        }
        float PORTABLE_ALIGN32 TmpRes[8];
        _mm_store_ps(TmpRes, sum);
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
    }

    template<typename TARG1, typename TARG2>
    static float
    L2SqrSIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        const auto qty = *static_cast<const size_t*>(qty_ptr);
        const auto qty4 = qty >> 2 << 2;
        const auto res = L2SqrSIMD4Ext<TARG1, TARG2>(pVect1v, pVect2v, &qty4);
        const auto qty_left = qty - qty4;
        const auto pVect1 = static_cast<const TARG1*>(pVect1v) + qty4;
        const auto pVect2 = static_cast<const TARG2*>(pVect2v) + qty4;
        const auto res_tail = L2Sqr<TARG1, TARG2>(pVect1, pVect2, &qty_left);

        return (res + res_tail);
    }
#endif

    template<typename TCOMPR=float>
    class L2Space : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        DISTFUNC<float> fstdist_search_func_;
        const size_t data_size_;
        size_t dim_;
    public:
        explicit L2Space(size_t dim)
        : data_size_(dim * sizeof(TCOMPR))
        , dim_(dim) {
            fstdistfunc_ = L2Sqr<TCOMPR, TCOMPR>;
            fstdist_search_func_ = L2Sqr<float, TCOMPR>;
        #if defined(USE_SSE) || defined(USE_AVX)
            if (dim % 8 == 0) {
                fstdistfunc_ = L2SqrSIMD8Ext<TCOMPR, TCOMPR>;
                fstdist_search_func_ = L2SqrSIMD8Ext<float, TCOMPR>;
            }
            else if (dim % 4 == 0) {
                fstdistfunc_ = L2SqrSIMD4Ext<TCOMPR, TCOMPR>;
                fstdist_search_func_ = L2SqrSIMD4Ext<float, TCOMPR>;
            }
            else if (dim > 8) {
                fstdistfunc_ = L2SqrSIMD8ExtResiduals<TCOMPR, TCOMPR>;
                fstdist_search_func_ = L2SqrSIMD8ExtResiduals<float, TCOMPR>;
            }
            else if (dim > 4) {
                fstdistfunc_ = L2SqrSIMD4ExtResiduals<TCOMPR, TCOMPR>;
                fstdist_search_func_ = L2SqrSIMD4ExtResiduals<float, TCOMPR>;
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

        void *get_dist_func_param() override {
            return &dim_;
        }

        ~L2Space() override = default;
    };

    static int
    L2SqrI(const void *__restrict pVect1, const void *__restrict pVect2, const void *__restrict qty_ptr) {

        auto qty = *static_cast<const size_t*>(qty_ptr);
        int res = 0;
        unsigned char *a = (unsigned char *) pVect1;
        unsigned char *b = (unsigned char *) pVect2;

        qty = qty >> 2;
        for (size_t i = 0; i < qty; i++) {

            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;


        }

        return (res);

    }

    class L2SpaceI : public SpaceInterface<int> {

        DISTFUNC<int> fstdistfunc_;
        size_t data_size_;
        size_t dim_;
    public:
        L2SpaceI(size_t dim) {
            fstdistfunc_ = L2SqrI;
            dim_ = dim;
            data_size_ = dim * sizeof(unsigned char);
        }

        size_t get_data_size() override {
            return data_size_;
        }

        DISTFUNC<int> get_dist_func() override {
            return fstdistfunc_;
        }

        DISTFUNC<int> get_search_dist_func() const override {
            return fstdistfunc_;
        }


        void *get_dist_func_param() override {
            return &dim_;
        }

        ~L2SpaceI() override = default;
    };


}