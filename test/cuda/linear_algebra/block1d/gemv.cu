#include <gtest/gtest.h>

// project headers
#include <msc/cuda/array>
#include <msc/cuda/linear_algebra/block1d/gemv>

// CUDA headers
#include "../../../../vendor/nvidia/cuda/cuPrintf.cu"


// types
typedef unsigned long ulong;


// macros
#define SMALL_DIMV 5
#define SMALL_DIMM SMALL_DIMV*SMALL_DIMV



template<typename T>
__global__
void test_gemv (const ulong m, const ulong n, const T * const a, const T * const x, T * const y) {
	CUDA::linear_algebra::block1D::gemv(m, n, a, x, y);
}


TEST(SmallFloatTestCase, GemvTest) {
	const float a[SMALL_DIMM] = {
		 1,  0,  0,  0,  0,
		 2,  3,  0,  0,  0,
		 4,  5,  6,  0,  0,
		 7,  8,  9, 10,  0,
		11, 12, 13, 14, 15
	};
	const float x[SMALL_DIMV] = {  16,  17,  18,  19,  20 };
	const float s[SMALL_DIMV] = { 475, 533, 539, 470, 300 };// expected
	float y[SMALL_DIMM] = {0};

	CUDA::array<float> d_a(a, SMALL_DIMM);
	CUDA::array<float> d_x(x, SMALL_DIMV);
	CUDA::array<float> d_y(y, SMALL_DIMV);

	cudaPrintfInit();

	test_gemv<<< 1 , SMALL_DIMV >>>(SMALL_DIMV, SMALL_DIMV, d_a.get_pointer(), d_x.get_pointer(), d_y.get_pointer());
	HANDLE_LAST_ERROR();

	d_y.to_host(y);

	for (ulong i = 0; i < SMALL_DIMV; ++i)
		EXPECT_FLOAT_EQ(y[i], s[i]);
}
