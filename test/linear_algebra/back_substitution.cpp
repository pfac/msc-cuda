#include <gtest/gtest.h>

#include <msc/linear_algebra/back_substitution>

#define DIMV 4
#define DIMM DIMV*DIMV

TEST(BackSubstitutionBbecomesX, BackSubstitutionTest) {
	const float a[DIMM] = {// the matrix A (column-major)
		 1,  0,  0, 0,
		 1, -2,  0, 0,
		-1, -3,  2, 0,
		 4,  1, -3, 2
	};
	float b[DIMV] = { 8,  5, 0, 4 };// the (column) vector b
	const float s[DIMV] = { 9, -6, 3, 2 };// the expected solution
	// float x[4];// the (unknown) (column) vector x

	back_substitution(a, b, DIMV);

	for (unsigned i = 0; i < DIMV; ++i)
		EXPECT_EQ(b[i], s[i]);
}
