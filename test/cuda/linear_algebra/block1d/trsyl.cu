#include <gtest/gtest.h>

#include "../../../../vendor/nvidia/cuda/cuPrintf.cu"
// project headers
#include <msc/cuda/array>
#include <msc/cuda/linear_algebra/block1d/trsyl>


// types
typedef unsigned long ulong;


// macros
#define SMALL_DIMV 2
#define SMALL_DIMM SMALL_DIMV*SMALL_DIMV



template<typename T>
__global__
void test_trsyl (const ulong m, const ulong n, const T * const a, const T * const b, T * const c) {
	CUDA::linear_algebra::block1D::trsyl(m, n, a, b, c);
}


TEST(SmallFloatTestCase, GemvTest) {
	const float a[SMALL_DIMM] = {
		1, 0,
		2, 3
	};
	const float b[SMALL_DIMM] = {
		6,  0,
		9, 10
	};
	float c[SMALL_DIMM] = {
		38, 45,
		129, 149
	};
	const float s[SMALL_DIMM] = {// expected
		4, 5,
		7, 8
	};

	CUDA::array<float> d_a(a, SMALL_DIMM);
	CUDA::array<float> d_b(b, SMALL_DIMM);
	CUDA::array<float> d_c(c, SMALL_DIMM);

	cudaPrintfInit();

	test_trsyl<<< 1 , SMALL_DIMV >>>(SMALL_DIMV, SMALL_DIMV, d_a.get_pointer(), d_b.get_pointer(), d_c.get_pointer());
	HANDLE_LAST_ERROR();

	d_c.to_host(c);

	cudaDeviceSynchronize();
	cudaPrintfDisplay(stdout, true);
	cudaPrintfEnd();

	for (ulong i = 0; i < SMALL_DIMM; ++i)
		EXPECT_FLOAT_EQ(s[i], c[i]);
}
