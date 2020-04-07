#pragma once
#include "hnswlib.h"
#include <iostream>


namespace hnswlib {

    static float
    Kendall(const void *pVect1, const void *pVect2, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        float res = 0;
        float v1, v2, u1, u2;
        for (unsigned i = 0; i < qty-1; i++) {
            for (unsigned j = i+1; j < qty; j++) {
                v1 = ((float *) pVect1)[i];
                v2 = ((float *) pVect1)[j];
                u1 = ((float *) pVect2)[i];
                u2 = ((float *) pVect2)[j];
                if (((v1 < v2) && (u1 < u2)) || ((v1 > v2) && (u1 > u2))) {
                    res += 1;
                }
                else {
                    res -= 1;
                }
            }
        }
        res /= qty*(qty-1)/2;
        return (1.0f - res);
    }

    class KendallSpace : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;
        size_t dim_;
    public:
        KendallSpace(size_t dim) {
            fstdistfunc_ = Kendall;
            dim_ = dim;
            data_size_ = dim * sizeof(float);
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

    ~KendallSpace() {}
    };
}
