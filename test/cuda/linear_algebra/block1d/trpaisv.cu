#include <gtest/gtest.h>

// project headers
#include <msc/cuda/array>
#include <msc/cuda/linear_algebra/block1d/trpaisv>


// types
typedef unsigned long ulong;


// macros
#define SMALL_DIMV 4
#define SMALL_DIMM SMALL_DIMV*SMALL_DIMV



template<typename T>
__global__
void test_trpaisv (const ulong n, const T * const a, const T alpha, T * const x) {
	CUDA::linear_algebra::block1D::trpaisv(n, a, alpha, x);
}


TEST(Small01FloatTestCase, TRPAISVtest) {
	const float a[SMALL_DIMM] = {
		 1,  0,  0,  0,
		 1, -2,  0,  0,
		-1, -3,  2,  0,
		 4,  1, -3,  2
	};
	const float alpha = 0;
	float x[SMALL_DIMV] = { 8,  5, 0, 4 };
	const float s[SMALL_DIMV] = { 9, -6, 3, 2 };// expected

	CUDA::array<float> d_a(a, SMALL_DIMM);
	CUDA::array<float> d_x(x, SMALL_DIMV);

	test_trpaisv<<< 1 , SMALL_DIMV >>>(SMALL_DIMV, d_a.get_pointer(), alpha, d_x.get_pointer());
	HANDLE_LAST_ERROR();

	d_x.to_host(x);


	for (ulong i = 0; i < SMALL_DIMV; ++i)
		EXPECT_FLOAT_EQ(s[i], x[i]);
}

TEST(Float7x7, TRPAISVtest) {
	const ulong m = 7;
	const float a[m * m] = {
		 1,  0,  0,  0,  0,  0,  0,
		 2,  3,  0,  0,  0,  0,  0,
		 4,  5,  6,  0,  0,  0,  0,
		 7,  8,  9, 10,  0,  0,  0,
		11, 12, 13, 14, 15,  0,  0,
		16, 17, 18, 19, 20, 21,  0,
		22, 23, 24, 25, 26, 27, 28
	};
	const float alpha = 36;
	float x[m] = { 3158, 3360, 3471, 3455, 3273, 2883, 2240 };
	const float s[m] = { 29, 30, 31, 32, 33, 34, 35 };// expected

	CUDA::array<float> d_a(a, m * m);
	CUDA::array<float> d_x(x, m);

	test_trpaisv<<< 1 , m >>>(m, d_a.get_pointer(), alpha, d_x.get_pointer());
	HANDLE_LAST_ERROR();

	d_x.to_host(x);


	for (ulong i = 0; i < m; ++i)
		EXPECT_FLOAT_EQ(s[i], x[i]);
}
