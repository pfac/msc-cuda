#include <gtest/gtest.h>

// #include "../../../../vendor/nvidia/cuda/cuPrintf.cu"
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

	// cudaPrintfInit();

	test_gemm<<< 1 , SMALL_DIMV >>>(SMALL_DIMV, SMALL_DIMV, SMALL_DIMV, alpha, d_a.get_pointer(), d_a.get_pointer(), beta, d_c.get_pointer());
	HANDLE_LAST_ERROR();

	d_c.to_host(c);

	// cudaDeviceSynchronize();
	// cudaPrintfDisplay(stdout, true);
	// cudaPrintfEnd();

	for (ulong i = 0; i < SMALL_DIMM; ++i)
		EXPECT_FLOAT_EQ(c[i], s[i]);
}


TEST(Float7x7x2case01, GEMMtest) {
	const ulong l = 7;
	const ulong m = 7;
	const ulong n = 2;

	const float a[l * m] = {
		29, 30, 31, 32, 33, 34, 35,
		37, 38, 39, 40, 41, 42, 43,
		46, 47, 48, 49, 50, 51, 52,
		56, 57, 58, 59, 60, 61, 62,
		67, 68, 69, 70, 71, 72, 73,
		79, 80, 81, 82, 83, 84, 85,
		92, 93, 94, 95, 96, 97, 98
	};
	const float b[m * n] = {
		414, 415, 416, 417, 418, 419, 420,
		443, 444, 445, 446, 447, 448, 449
	};
	float c[l * n] = {
		1750904, 1762299, 1772469, 1781000, 1787475, 1791474, 1792574,
		2072225, 2084868, 2096199, 2105775, 2113150, 2117875, 2119498
	};
	const float s[l * n] = {
		1581308, 1589784, 1597035, 1602647, 1606203, 1607283, 1605464,
		1890855, 1900376, 1908585, 1915039, 1919292, 1920895, 1919396
	};

	const float alpha = -1.0;
	const float beta = 1.0;

	// cudaPrintfInit();

	CUDA::array<float> d_a(a, l * m);
	CUDA::array<float> d_b(b, m * n);
	CUDA::array<float> d_c(c, l * n);

	test_gemm<<< 1 , l >>>(l, n, m, alpha, d_a.get_pointer(), d_b.get_pointer(), beta, d_c.get_pointer());
	HANDLE_LAST_ERROR();

	d_c.to_host(c);

	// cudaDeviceSynchronize();
	// cudaPrintfDisplay(stdout, true);
	// cudaPrintfEnd();

	for (ulong i = 0; i < l * n; ++i)
		EXPECT_FLOAT_EQ(s[i], c[i]);
}
