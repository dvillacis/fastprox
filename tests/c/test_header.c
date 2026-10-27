/* Compiles fastprox.h as strict C99 (-Werror -pedantic) and runs a smoke test
 * through the public API. */
#include "fastprox.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

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
    /* The library's version string is the build's (CMakeLists.txt project()
       and FASTPROX_VERSION_SUFFIX), and begins with the header's macros. */
    {
        char head[32];
        snprintf(head, sizeof head, "%d.%d.%d", FP_VERSION_MAJOR, FP_VERSION_MINOR, FP_VERSION_PATCH);
        CHECK(strcmp(fp_version_string(), FP_EXPECTED_VERSION) == 0);
        CHECK(strncmp(fp_version_string(), head, strlen(head)) == 0);
    }
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

    /* C enums are ints, so a caller can pass any value in an enum argument or
       field; the library reads the int and rejects values outside the enum. */
    xa.dtype = (fp_dtype)7;
    CHECK(fp_op_soft(1.0, NULL, &op) == FP_OK);
    CHECK(fp_prox(op, 1.0, &xa, &za, NULL, NULL, NULL) == FP_EDTYPE);
    xa.dtype = FP_F64;
    xa.device = (fp_device)10;
    CHECK(fp_prox(op, 1.0, &xa, &za, NULL, NULL, NULL) == FP_EINVAL);
    xa.device = FP_DEVICE_CPU;
    CHECK(fp_device_available((fp_device)10) == 0 && fp_device_available((fp_device)-1) == 0);
    CHECK(fp_op_device_check(op, (fp_device)10, NULL) == FP_EINVAL);
    CHECK(fp_op_spectral_check(op, (fp_spectral_kind)5) == FP_EINVAL);
    fp_op_free(op);
    CHECK(fp_op_tv1d(NULL, (fp_tv1d_method)7, &op) == FP_EPARAM && op == NULL);

    /* Structure of the derivative (since 1.0.2): soft thresholding of x keeps
       entries 0 and 3, as above. */
    {
        fp_options o;
        fp_workspace *ws = NULL;
        int kind = -1;
        int8_t state[5], sign[5];
        int32_t labels[5];
        int64_t count = 0, size = 0, nfixed = 0, nregions = 0;
        fp_array sa = vec(NULL, &n), ga = vec(NULL, &n), la = vec(NULL, &n);
        sa.data = state;
        sa.dtype = FP_I8;
        ga.data = sign;
        ga.dtype = FP_I8;
        la.data = labels;
        la.dtype = FP_I32;
        fp_options_init(&o);
        o.keep_structure = 1;
        CHECK(fp_op_soft(1.0, NULL, &op) == FP_OK);
        CHECK(fp_op_struct_kind(op, &kind) == FP_OK && kind == FP_STRUCT_PROJECTOR);
        CHECK(fp_workspace_create(op, &xa, &o, &ws) == FP_OK);
        CHECK(fp_prox(op, 1.0, &xa, &za, &o, ws, NULL) == FP_OK);
        CHECK(fp_workspace_units(ws, &count, &size) == FP_OK && count == 1 && size == 5);
        CHECK(fp_workspace_states(ws, &sa, &nfixed) == FP_OK && nfixed == 3);
        CHECK(state[0] == FP_ENTRY_FREE && state[1] == FP_ENTRY_ZERO && state[3] == FP_ENTRY_FREE);
        CHECK(fp_workspace_regions(ws, &la, &ga, &nregions, 1) == FP_OK && nregions == 2);
        CHECK(labels[0] == 0 && labels[1] == -1 && labels[3] == 1 && sign[3] == 1 && sign[4] == 0);
        CHECK(fp_workspace_constraint(ws, NULL, NULL, 0) == FP_OK);
        la.dtype = FP_F64;
        CHECK(fp_workspace_regions(ws, &la, NULL, NULL, 0) == FP_EDTYPE);
        fp_workspace_free(ws);
        fp_op_free(op);
        CHECK(FP_STRUCT_NONE == 0 && FP_STRUCT_WEIGHTED == 2 && FP_STRUCT_CURVED == 3);
        CHECK(FP_ENTRY_LOWER == 2 && FP_ENTRY_UPPER == 3);
    }

    if (failures) fprintf(stderr, "%d failure(s)\n", failures);
    return failures != 0;
}
