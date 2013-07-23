#include <gtest/gtest.h>

// project headers
#include <msc/core/gpu/point>


// macros
#define SMALL_DIMV 5
#define SMALL_DIMM SMALL_DIMV*SMALL_DIMV

TEST(PointFloatSmallTestCase, PointTest) {
	float t[SMALL_DIMM] = {
		  1,   0,   0,   0,   0,
		  8,   9,   0,   0,   0,
		 38,  45,  36,   0,   0,
		129, 149, 144, 100,   0,
		350, 393, 399, 350, 225
	};
	const float s[SMALL_DIMM] = {
		 1,  0,  0,  0,  0,
		 2,  3,  0,  0,  0,
		 4,  5,  6,  0,  0,
		 7,  8,  9, 10,  0,
		11, 12, 13, 14, 15
	};

	point::sqrtm(t, SMALL_DIMV);

	for (ulong i = 0; i < SMALL_DIMM; ++i)
		EXPECT_FLOAT_EQ(t[i], s[i]);
}
