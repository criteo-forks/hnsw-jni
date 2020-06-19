#pragma once

namespace hnswlib {
    const auto FC16_CONVERSION_FLAGS = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
    #ifdef USE_F16C
    static inline uint16_t encode_fp16 (float x) {
        __m128 xf = _mm_set1_ps (x);
        __m128i xi = _mm_cvtps_ph (xf, FC16_CONVERSION_FLAGS);
        return _mm_cvtsi128_si32 (xi);
    }

    static inline float decode_fp16 (uint16_t x) {
        __m128i xi = _mm_set1_epi16 (x);
        __m128 xf = _mm_cvtph_ps (xi);
        return _mm_cvtss_f32 (xf);
    }

    #else

    // Copy if needed from https://github.com/facebookresearch/faiss/blob/da24fcc56eb203262cf209671d39e64e40744a96/impl/ScalarQuantizer.cpp#L222

    #endif
}