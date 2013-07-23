#include <gtest/gtest.h>

// CUDA headers
#include <cuda_runtime.h>

TEST(DeviceQueryTest,GetDeviceCountNoException) {
	EXPECT_NO_THROW({
		int device_count;
		cudaGetDeviceCount(&device_count);
	});
}

TEST(DeviceQueryTest,GetDeviceCountAtLeastOne) {
	int device_count;
	EXPECT_NO_THROW({
		cudaGetDeviceCount(&device_count);
	});
	EXPECT_GE(device_count, 1);
}
