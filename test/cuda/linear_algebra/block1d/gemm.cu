#include <gtest/gtest.h>

#include "../../../../vendor/nvidia/cuda/cuPrintf.cu"
// project headers
#include <msc/cuda/array>
#include <msc/cuda/linear_algebra/block1d/gemm>


// types
typedef unsigned long ulong;


// macros
#define SMALL_DIMV 5
#define SMALL_DIMM SMALL_DIMV*SMALL_DIMV



template<typename T>
__global__
void test_gemm (const ulong m, const ulong n, const ulong l, const T alpha, const T * const a, const T * const b, const T beta, T * const c) {
	CUDA::linear_algebra::block1D::gemm(m, n, l, alpha, a, b, beta, c);
}


TEST(SmallFloatTestCase, GEMMtest) {
	const float a[SMALL_DIMM] = {
		 1,  0,  0,  0,  0,
		 2,  3,  0,  0,  0,
		 4,  5,  6,  0,  0,
		 7,  8,  9, 10,  0,
		11, 12, 13, 14, 15
	};
	const float s[SMALL_DIMM] = {// expected
		  1,   0,   0,   0,   0,
		  8,   9,   0,   0,   0,
		 38,  45,  36,   0,   0,
		129, 149, 144, 100,   0,
		350, 393, 399, 350, 225
	};
	const float alpha = 1.0;
	const float beta = 0.0;
	float c[SMALL_DIMM] = {0};

	CUDA::array<float> d_a(a, SMALL_DIMM);
	CUDA::array<float> d_c(c, SMALL_DIMM);

	test_gemm<<< 1 , SMALL_DIMV >>>(SMALL_DIMV, SMALL_DIMV, SMALL_DIMV, alpha, d_a.get_pointer(), d_a.get_pointer(), beta, d_c.get_pointer());
	HANDLE_LAST_ERROR();

	d_c.to_host(c);

	for (ulong i = 0; i < SMALL_DIMM; ++i)
		EXPECT_FLOAT_EQ(c[i], s[i]);
}
