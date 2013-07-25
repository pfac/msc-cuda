#include <gtest/gtest.h>

// project headers
#include <msc/cuda/array>
#include <msc/cuda/linear_algebra/block1d/trsv>


// types
typedef unsigned long ulong;


// macros
#define SMALL_DIMV 4
#define SMALL_DIMM SMALL_DIMV*SMALL_DIMV



template<typename T>
__global__
void test_trsv (const ulong n, const T * const a, T * const x) {
	CUDA::linear_algebra::block1D::trsv(n, a, x);
}


TEST(SmallFloatTestCase, GemvTest) {
	const float a[SMALL_DIMM] = {
		 1,  0,  0,  0,
		 1, -2,  0,  0,
		-1, -3,  2,  0,
		 4,  1, -3,  2
	};
	float x[SMALL_DIMV] = { 8,  5, 0, 4 };
	const float s[SMALL_DIMV] = { 9, -6, 3, 2 };// expected

	CUDA::array<float> d_a(a, SMALL_DIMM);
	CUDA::array<float> d_x(x, SMALL_DIMV);

	test_trsv<<< 1 , SMALL_DIMV >>>(SMALL_DIMV, d_a.get_pointer(), d_x.get_pointer());
	HANDLE_LAST_ERROR();

	d_x.to_host(x);

	for (ulong i = 0; i < SMALL_DIMV; ++i)
		EXPECT_FLOAT_EQ(s[i], x[i]);
}
