#include <gtest/gtest.h>

#include <msc/matrix>

#define DIMV 4
#define DIMM DIMV*DIMV

TEST(BackSubstitutionTestCase, BackSubstitutionTest) {
	const float a[DIMM] = {// the matrix A (column-major)
		 1,  0,  0, 0,
		 1, -2,  0, 0,
		-1, -3,  2, 0,
		 4,  1, -3, 2
	};
	const float b[DIMV] = { 8,  5, 0, 4 };// the (column) vector b
	const float s[DIMV] = { 9, -6, 3, 2 };// the expected solution
	float x[4];// the (unknown) (column) vector x

	back_substitution(a, b, x);

	for (unsigned i = 0; i < DIMV; ++i)
		EXPECT_EQ(x[i], s[i]);
}
