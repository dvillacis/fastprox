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
fp_status make_log_barrier(double lambda, fp_op **out);
fp_status make_l1_box(double lambda, Vec w, Vec lo, Vec hi, fp_op **out);

// group.cpp
fp_status make_group_l2(double lambda, std::vector<int64_t> groups, int64_t ngroups, Vec c,
                        fp_op **out);
fp_status make_tree_l2(double l1, std::vector<int64_t> ptr, std::vector<int64_t> idx, Vec eta,
                       fp_op **out);

// radial.cpp: norm composition of a scalar map (Huber, SCAD, MCP; scalar is
// owned and deleted on failure) and the second-order cone.
fp_status make_group_radial(fp_op *scalar, const char *name, std::vector<int64_t> groups,
                            int64_t ngroups, Vec c, fp_op **out);
fp_status make_soc(fp_op **out);

// projections.cpp
fp_status make_l2_ball(double radius, fp_op **out);
fp_status make_simplex(double radius, fp_op **out);
fp_status make_l1_ball(double radius, Vec w, fp_op **out);
fp_status make_linf(double lambda, fp_op **out);
fp_status make_max(double lambda, fp_op **out);

// hyperplane.cpp
fp_status make_hyperplane(Vec a, double b, Vec lo, Vec hi, bool inequality, fp_op **out);
fp_status make_sum_largest(double lambda, int64_t k, bool absolute, fp_op **out);

// isotonic.cpp
fp_status make_isotonic(Vec w, bool increasing, fp_op **out);

// tv1d.cpp
fp_status make_tv1d(Vec w, fp_tv1d_method method, fp_op **out);

// tv_ball.cpp
fp_status make_tv1d_ball(double alpha, Vec w, fp_op **out);

// tv1d_l2.cpp
fp_status make_tv1d_l2(Vec w, fp_op **out);

// nonconvex.cpp
fp_status make_l0(double lambda, Vec w, fp_op **out);
fp_status make_lq(double lambda, double q, Vec w, fp_op **out);
fp_status make_scad(double lambda, double a, Vec w, fp_op **out);  // w empty: unweighted
fp_status make_mcp(double lambda, double b, Vec w, fp_op **out);
fp_status make_log(double lambda, double eps, fp_op **out);

// tv_aniso.cpp
fp_status make_tv_aniso(int ndim, Vec w, fp_op **out);

// tv_iso.cpp
fp_status make_tv_iso(int ndim, Vec w, fp_op **out);

// graph_tv.cpp: edges holds 2m node indices (u_0, v_0, u_1, v_1, ...).
fp_status make_graph_tv(int64_t n, int64_t m, const int64_t *edges, Vec w, fp_op **out);

inline bool finite_nonneg(double v) noexcept { return v >= 0.0 && v <= 1.7976931348623157e308; }

}  // namespace fp
