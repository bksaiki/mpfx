/**
 * @file cpfloat_wrap.h
 * @brief C shim around CPFloat for the benchmarks.
 *
 * CPFloat is a header-only C library that does not compile as C++, so the
 * benchmark reaches it through this translation unit. Numbers are stored in
 * `double`; CPFloat computes each operation natively in binary64 and then
 * rounds the result to the target format (double rounding).
 */
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cpfw_ctx cpfw_ctx;

/// Rounding modes, mirroring `cpfloat_rounding_t`.
enum cpfw_rm {
    CPFW_RND_NA = -1,  ///< nearest, ties away from zero
    CPFW_RND_NE = 1,   ///< nearest, ties to even
    CPFW_RND_TP = 2,   ///< toward +inf
    CPFW_RND_TN = 3,   ///< toward -inf
    CPFW_RND_TZ = 4,   ///< toward zero
    CPFW_RND_OD = 7    ///< to odd
};

/// Target format: `precision` significand bits (incl. implicit bit), exponent
/// range `[emin, emax]`, subnormals enabled. Returns NULL if CPFloat rejects it.
cpfw_ctx* cpfw_create(int precision, int emin, int emax, enum cpfw_rm rm);
void cpfw_free(cpfw_ctx* ctx);

// Scalar operations (one CPFloat call per element).
double cpfw_round(cpfw_ctx* ctx, double x);
double cpfw_add(cpfw_ctx* ctx, double x, double y);
double cpfw_sub(cpfw_ctx* ctx, double x, double y);
double cpfw_mul(cpfw_ctx* ctx, double x, double y);
double cpfw_div(cpfw_ctx* ctx, double x, double y);
double cpfw_sqrt(cpfw_ctx* ctx, double x);
double cpfw_fma(cpfw_ctx* ctx, double x, double y, double z);

// Array operations (one CPFloat call for all `n` elements).
void cpfw_add_n(cpfw_ctx* ctx, double* r, const double* x, const double* y, size_t n);
void cpfw_sub_n(cpfw_ctx* ctx, double* r, const double* x, const double* y, size_t n);
void cpfw_mul_n(cpfw_ctx* ctx, double* r, const double* x, const double* y, size_t n);
void cpfw_div_n(cpfw_ctx* ctx, double* r, const double* x, const double* y, size_t n);
void cpfw_sqrt_n(cpfw_ctx* ctx, double* r, const double* x, size_t n);
void cpfw_fma_n(cpfw_ctx* ctx, double* r, const double* x, const double* y, const double* z, size_t n);

#ifdef __cplusplus
}
#endif
