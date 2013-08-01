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


TEST(SmallFloatTestCase, TRSYLtest) {
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


TEST(Float7x3, TRSYLtest) {
	const ulong m = 7;
	const ulong n = 3;

	const float a[m * m] = {
		 1,  0,  0,  0,  0,  0,  0,
		 2,  3,  0,  0,  0,  0,  0,
		 4,  5,  6,  0,  0,  0,  0,
		 7,  8,  9, 10,  0,  0,  0,
		11, 12, 13, 14, 15,  0,  0,
		16, 17, 18, 19, 20, 21,  0,
		22, 23, 24, 25, 26, 27, 28
	};
	const float b[n * n] = {
		36,  0,  0,
		44, 45,  0,
		53, 54, 55
	};
	float c[m * n] = {
		3158, 3360, 3471,  3455, 3273, 2883, 2240,
		5559, 5854, 6034,  6055, 5870, 5429, 4679,
		9250, 9663, 9934, 10010, 9835, 9350, 8493
	};
	const float s[m * n] = {
		29, 30, 31, 32, 33, 34, 35,
		37, 38, 39, 40, 41, 42, 43,
		46, 47, 48, 49, 50, 51, 52
	};

	CUDA::array<float> d_a(a, m * m);
	CUDA::array<float> d_b(b, n * n);
	CUDA::array<float> d_c(c, m * n);

	cudaPrintfInit();

	test_trsyl<<< 1 , m >>>(m, n, d_a.get_pointer(), d_b.get_pointer(), d_c.get_pointer());
	HANDLE_LAST_ERROR();

	d_c.to_host(c);

	cudaDeviceSynchronize();
	cudaPrintfDisplay(stdout, true);
	cudaPrintfEnd();

	for (ulong i = 0; i < m * n; ++i)
		EXPECT_FLOAT_EQ(s[i], c[i]);
}
