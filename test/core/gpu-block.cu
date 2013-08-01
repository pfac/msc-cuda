#include <gtest/gtest.h>

// project headers
#include <msc/core/gpu/block>


// macros
#define SMALL_DIMV 5
#define SMALL_DIMM SMALL_DIMV*SMALL_DIMV
#define SMALL_BLCK 2

#define IDIVCEIL(x,y) (((x) + (y) - 1) / (y))



TEST(MainDiagonalTestCase, BlockSmallTest) {
	float t[SMALL_DIMM] = {// host data
		  1,   0,   0,   0,   0,
		  8,   9,   0,   0,   0,
		 38,  45,  36,   0,   0,
		129, 149, 144, 100,   0,
		350, 393, 399, 350, 225
	};
	const float s[SMALL_DIMM] = { 
		  1,   0,   0,   0,   0,
		  2,   3,   0,   0,   0,
		 38,  45,   6,   0,   0,
		129, 149,   9,  10,   0,
		350, 393, 399, 350,  15
	};
	float h_t[SMALL_DIMM];

	// blockify host data
	blockify(t, SMALL_DIMV, SMALL_DIMV, SMALL_BLCK, SMALL_BLCK, h_t);

	CUDA::array<float> d_t(h_t, SMALL_DIMM);// device data

	__sqrtm_d0<<< IDIVCEIL(SMALL_DIMV, SMALL_BLCK) , SMALL_BLCK >>>(d_t.get_pointer(), SMALL_DIMV, SMALL_BLCK, IDIVCEIL(SMALL_DIMV, SMALL_BLCK));
	HANDLE_LAST_ERROR();

	d_t.to_host(h_t);

	// unblockify
	unblockify(h_t, SMALL_DIMV, SMALL_DIMV, SMALL_BLCK, SMALL_BLCK, t);

	for (ulong i = 0; i < SMALL_DIMM; ++i)
		EXPECT_FLOAT_EQ(s[i], t[i]);
}
