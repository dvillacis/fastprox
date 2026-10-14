// Factories for every operator family. Parameter arrays arrive already
// copied into std::vector<double>; the factories validate their values.
//
// SPDX-License-Identifier: MIT
#pragma once

#include "fastprox.h"

#include <cstdint>
#include <vector>

namespace fp {

using Vec = std::vector<double>;

// pointwise.cpp
fp_status make_soft(double lambda, Vec w, fp_op **out);
fp_status make_elastic_net(double l1, double l2, fp_op **out);
fp_status make_huber(double lambda, double delta, fp_op **out);
fp_status make_box(Vec lo, Vec hi, fp_op **out);
fp_status make_nonneg(fp_op **out);

// group.cpp
fp_status make_group_l2(double lambda, std::vector<int64_t> groups, int64_t ngroups, Vec c,
                        fp_op **out);

// projections.cpp
fp_status make_l2_ball(double radius, fp_op **out);
fp_status make_simplex(double radius, fp_op **out);
fp_status make_l1_ball(double radius, Vec w, fp_op **out);
fp_status make_linf(double lambda, fp_op **out);

// isotonic.cpp
fp_status make_isotonic(Vec w, bool increasing, fp_op **out);

// tv1d.cpp
fp_status make_tv1d(Vec w, fp_tv1d_method method, fp_op **out);

inline bool finite_nonneg(double v) noexcept { return v >= 0.0 && v <= 1.7976931348623157e308; }

}  // namespace fp
