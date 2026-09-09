/* See cpfloat_wrap.h. */
#include "cpfloat_wrap.h"

#include <math.h>
#include <stdlib.h>

#include "cpfloat_binary64.h"

struct cpfw_ctx {
    optstruct* opts;
};

cpfw_ctx* cpfw_create(int precision, int emin, int emax, enum cpfw_rm rm) {
    optstruct* opts = init_optstruct();
    opts->precision = precision;
    opts->emin = emin;
    opts->emax = emax;
    opts->subnormal = CPFLOAT_SUBN_USE;
    opts->explim = CPFLOAT_EXPRANGE_TARG;
    opts->round = (cpfloat_rounding_t) rm;
    opts->flip = CPFLOAT_SOFTERR_NO;
    opts->p = 0;
    if (cpfloat_validate_optstruct(opts) < 0) {
        free_optstruct(opts);
        return NULL;
    }
    cpfw_ctx* ctx = malloc(sizeof(*ctx));
    ctx->opts = opts;
    return ctx;
}

void cpfw_free(cpfw_ctx* ctx) {
    if (ctx) {
        free_optstruct(ctx->opts);
        free(ctx);
    }
}

double cpfw_round(cpfw_ctx* ctx, double x) {
    double r;
    cpfloat(&r, &x, 1, ctx->opts);
    return r;
}

double cpfw_add(cpfw_ctx* ctx, double x, double y) {
    double r;
    cpf_add(&r, &x, &y, 1, ctx->opts);
    return r;
}

double cpfw_sub(cpfw_ctx* ctx, double x, double y) {
    double r;
    cpf_sub(&r, &x, &y, 1, ctx->opts);
    return r;
}

double cpfw_mul(cpfw_ctx* ctx, double x, double y) {
    double r;
    cpf_mul(&r, &x, &y, 1, ctx->opts);
    return r;
}

double cpfw_div(cpfw_ctx* ctx, double x, double y) {
    double r;
    cpf_div(&r, &x, &y, 1, ctx->opts);
    return r;
}

double cpfw_sqrt(cpfw_ctx* ctx, double x) {
    double r;
    cpf_sqrt(&r, &x, 1, ctx->opts);
    return r;
}

/* CPFloat has no fma; apply its recipe (binary64 op, then round) by hand. */
double cpfw_fma(cpfw_ctx* ctx, double x, double y, double z) {
    double r = fma(x, y, z);
    cpfloat(&r, &r, 1, ctx->opts);
    return r;
}

void cpfw_add_n(cpfw_ctx* ctx, double* r, const double* x, const double* y, size_t n) {
    cpf_add(r, x, y, n, ctx->opts);
}

void cpfw_sub_n(cpfw_ctx* ctx, double* r, const double* x, const double* y, size_t n) {
    cpf_sub(r, x, y, n, ctx->opts);
}

void cpfw_mul_n(cpfw_ctx* ctx, double* r, const double* x, const double* y, size_t n) {
    cpf_mul(r, x, y, n, ctx->opts);
}

void cpfw_div_n(cpfw_ctx* ctx, double* r, const double* x, const double* y, size_t n) {
    cpf_div(r, x, y, n, ctx->opts);
}

void cpfw_sqrt_n(cpfw_ctx* ctx, double* r, const double* x, size_t n) {
    cpf_sqrt(r, x, n, ctx->opts);
}

void cpfw_fma_n(cpfw_ctx* ctx, double* r, const double* x, const double* y, const double* z, size_t n) {
    for (size_t i = 0; i < n; i++) {
        r[i] = fma(x[i], y[i], z[i]);
    }
    cpfloat(r, r, n, ctx->opts);
}
