/* Compiles fastprox.h as strict C99 (-Werror -pedantic) and runs a smoke test
 * through the public API. */
#include "fastprox.h"

#include <math.h>
#include <stdio.h>

static int failures = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, \
                    #cond);                                                  \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

static fp_array vec(double *data, const int64_t *shape) {
    fp_array a;
    a.data = data;
    a.dtype = FP_F64;
    a.ndim = 1;
    a.shape = shape;
    a.strides = NULL;
    a.device = FP_DEVICE_CPU;
    return a;
}

int main(void) {
    double x[5] = {3.0, -0.5, 1.0, -2.0, 0.25};
    double z[5];
    int64_t n = 5;
    fp_array xa = vec(x, &n), za = vec(z, &n);
    fp_op *op = NULL;
    fp_info info;
    double sum = 0.0;
    int i;

    CHECK(fp_abi_version() == FP_ABI_VERSION);
    CHECK(fp_op_soft(1.0, NULL, &op) == FP_OK);
    CHECK(fp_prox(op, 1.0, &xa, &za, NULL, NULL, &info) == FP_OK);
    CHECK(z[0] == 2.0 && z[1] == 0.0 && z[2] == 0.0 && z[3] == -1.0 && z[4] == 0.0);
    CHECK(info.gap == 0.0);
    fp_op_free(op);

    CHECK(fp_op_simplex(1.0, &op) == FP_OK);
    CHECK(fp_prox(op, 1.0, &xa, &za, NULL, NULL, NULL) == FP_OK);
    for (i = 0; i < 5; ++i) {
        CHECK(z[i] >= 0.0);
        sum += z[i];
    }
    CHECK(fabs(sum - 1.0) < 1e-14);
    fp_op_free(op);

    CHECK(fp_op_tv1d(NULL, FP_TV1D_AUTO, &op) == FP_OK);
    CHECK(fp_prox(op, -1.0, &xa, &za, NULL, NULL, NULL) == FP_EPARAM);
    fp_op_free(op);

    CHECK(fp_op_simplex(-1.0, &op) == FP_EPARAM && op == NULL);
    CHECK(fp_strerror(FP_ENOMEM)[0] != '\0');

    if (failures) fprintf(stderr, "%d failure(s)\n", failures);
    return failures != 0;
}
