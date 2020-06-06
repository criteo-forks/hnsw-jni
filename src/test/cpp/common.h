#pragma once
#include "doctest.h"
#include "hnswindex.h"
#include <cmath>
#include <tuple>

#ifndef HNSWLIB_TEST_COMMON
#define HNSWLIB_TEST_COMMON
const int seed = 42;

// Reimplemented from Approx(double) for floats
static bool is_approx_equal(float lhs, float rhs, float epsilon) {
    const auto absolute_error = std::abs(lhs - rhs);
    const auto expected_error = epsilon * (1 + std::max(std::fabs(lhs), std::fabs(rhs)));
    return absolute_error < expected_error;
}

#endif
