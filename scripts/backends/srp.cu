// Turbulence backend (op mode, sr_impl="call"): stochastic rounding of float32 results
// to a reduced precision of TURB_PREC significant bits (24 = float32, 11 = fp16-like,
// 8 = bf16-like), mode name `srp`. Group B, precision sweep (#31).
//
// The exact result x = r + d (r = RN(x) in float32, d its exact error) is rounded to one
// of its two neighbours on the TURB_PREC-bit grid of r's binade, the upper one with
// probability (x - lo) / ulp: unbiased, like Verificarlo's SR at virtual precision t.
// At TURB_PREC = 24 this is ordinary float32 stochastic rounding (the default `rr`).
// float64 operations, if any, use the reference `rr` rounding.
//
// The precision is a compile-time constant: precision_sweep.py writes a copy of this file
// with TURB_PREC defined, one per precision.

#ifndef TURB_PREC
#define TURB_PREC 24
#endif

#include "turbulence_sr.h"

__device__ __forceinline__ float srp(float r, float d, TURB_RNG_PARAMS) {
  uint32_t bits = turb_bits(r);
  uint32_t expo = (bits >> 23) & 0xffu;
  // zero, subnormal, inf, nan, or a grid finer than float32's: plain stochastic rounding
  if (expo <= (uint32_t)(TURB_PREC - 1) || expo == 0xffu || TURB_PREC >= 24)
    return turb_round(r, d, TURB_RR, TURB_RNG);
  float sign = (bits >> 31) ? -1.0f : 1.0f;
  float a = sign * r;                                                    // |r|
  float ulp = turb_f32((expo - (uint32_t)(TURB_PREC - 1)) << 23);      // 2^(e - (t - 1))
  uint32_t mask = ~((1u << (24 - TURB_PREC)) - 1u);                    // keep t-1 mantissa bits
  float lo = turb_f32(turb_bits(a) & mask);                            // a rounded toward zero
  float frac = ((a - lo) + sign * d) / ulp;                            // in (-1, 1)
  if (frac < 0.0f) {
    lo -= ulp;
    frac += 1.0f;
  }
  float u = turb_uniform_f32(turb_philox4x32_10(TURB_RNG));
  return sign * (u < frac ? lo + ulp : lo);
}

TURBULENCE_EXPORT float __turbulence_srp_add_f32(float a, float b, TURB_RNG_PARAMS) {
  float d, r = turb_two_sum(a, b, &d);
  return srp(r, d, TURB_RNG);
}
TURBULENCE_EXPORT float __turbulence_srp_sub_f32(float a, float b, TURB_RNG_PARAMS) {
  float d, r = turb_two_sum(a, -b, &d);
  return srp(r, d, TURB_RNG);
}
TURBULENCE_EXPORT float __turbulence_srp_mul_f32(float a, float b, TURB_RNG_PARAMS) {
  float d, r = turb_two_prod(a, b, &d);
  return srp(r, d, TURB_RNG);
}
TURBULENCE_EXPORT float __turbulence_srp_div_f32(float a, float b, TURB_RNG_PARAMS) {
  float d, r = turb_div_err(a, b, &d);
  return srp(r, d, TURB_RNG);
}
TURBULENCE_EXPORT float __turbulence_srp_fma_f32(float a, float b, float c, TURB_RNG_PARAMS) {
  float d, r = turb_fma_err(a, b, c, &d);
  return srp(r, d, TURB_RNG);
}

// float64 -> float32 conversion (ShallowFBCSPNet's kernels have one): same rounding.
TURBULENCE_EXPORT float __turbulence_srp_trunc_f64_f32(double x, TURB_RNG_PARAMS) {
  float r = (float)x;
  float d = (float)(x - (double)r);  // exact error in double; its float value sets the probability
  return srp(r, d, TURB_RNG);
}

#define RR64(op, expr)                                                                   \
  TURBULENCE_EXPORT double __turbulence_srp_##op##_f64(double a, double b, TURB_RNG_PARAMS) { \
    double d, r = expr;                                                                     \
    return turb_round(r, d, TURB_RR, TURB_RNG);                                        \
  }
RR64(add, turb_two_sum(a, b, &d))
RR64(sub, turb_two_sum(a, -b, &d))
RR64(mul, turb_two_prod(a, b, &d))
RR64(div, turb_div_err(a, b, &d))
TURBULENCE_EXPORT double __turbulence_srp_fma_f64(double a, double b, double c, TURB_RNG_PARAMS) {
  double d, r = turb_fma_err(a, b, c, &d);
  return turb_round(r, d, TURB_RR, TURB_RNG);
}
