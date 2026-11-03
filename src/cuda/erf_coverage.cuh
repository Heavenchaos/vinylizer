#pragma once

// DiffBMP-equivalent erf-softened coverage with runtime sigma
// Gaussian blur of a binary mask = erf(SDF / (sqrt(2)*sigma))
// sigma is a runtime parameter (annealed during optimization)

#define RSQRT2PI 0.3989422804014327f  // 1/sqrt(2*pi)
#define SQRT2_F 1.4142135623730951f

// Forward: erf-softened coverage
// sdf > 0 -> inside (coverage -> 1)
// sdf < 0 -> outside (coverage -> 0)
__device__ __forceinline__ float erf_coverage(float sdf, float sigma) {
    float inv_sqrt2_sigma = 1.0f / (SQRT2_F * sigma);
    return 0.5f * (1.0f + erff(sdf * inv_sqrt2_sigma));
}

// Backward: d(coverage)/d(sdf) = (1/sqrt(2pi)) * (1/sigma) * exp(-sdf^2/(2*sigma^2))
__device__ __forceinline__ float d_erf_coverage(float sdf, float sigma) {
    float x = sdf / sigma;
    return RSQRT2PI / sigma * expf(-0.5f * x * x);
}
