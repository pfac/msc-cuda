#include <gtest/gtest.h>

// project headers
#include <msc/cuda/array>
#include <msc/cuda/linear_algebra/block1d/apy>


// types
typedef unsigned long ulong;


// macros
#define SMALL_DIMV 5
#define SMALL_DIMM SMALL_DIMV*SMALL_DIMV



template<typename T>
__global__
void test_apy (const ulong n, const T alpha, T * const y, const ulong ldy) {
	CUDA::linear_algebra::block1D::apy(n, alpha, y, ldy);
}


TEST(SmallFloatMatrixTestCase, SapyTest) {
	float y_matrix[SMALL_DIMM] = {
		 1,  2,  3,  4,  5,
		 6,  7,  8,  9, 10,
		11, 12, 13, 14, 15,
		16, 17, 18, 19, 20,
		21, 22, 23, 24, 25
	};
	const float a = 26;
	const float s[SMALL_DIMM] = {// expected
		27,  2,  3,  4,  5,
		 6, 33,  8,  9, 10,
		11, 12, 39, 14, 15,
		16, 17, 18, 45, 20,
		21, 22, 23, 24, 51
	};

	CUDA::array<float> d_y_matrix(y_matrix, SMALL_DIMM);

	test_apy<<< 1 , SMALL_DIMV >>>(SMALL_DIMV, a, d_y_matrix.get_pointer(), SMALL_DIMV + 1);
	HANDLE_LAST_ERROR();

	d_y_matrix.to_host(y_matrix);

	for (ulong i = 0; i < SMALL_DIMM; ++i)
		EXPECT_FLOAT_EQ(s[i], y_matrix[i]);
}
