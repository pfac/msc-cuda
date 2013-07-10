// project headers
#include <msc/cuda/core>
#include <msc/cuda/exceptions/no_device>

// CUDA headers
#include <cuda_runtime.h>

// std C++ headers
#include <iostream>


// macros
#define HANDLE_ERROR(e) handle_error(e, __FILE__, __LINE__)
#define endl '\n'



namespace CUDA {

	bool queried = false;
	vector<device> devices;



	inline
	void handle_error (cudaError_t error, const char * file, const ulong line) throw(int) {
		if (error != cudaSuccess) {
			std::cerr << cudaGetErrorString(error) << endl;
			throw error;
		}
	}



	ulong get_device_count () {
		if (queried)
			return devices.size();
		
		int d_count = 0;
		HANDLE_ERROR( cudaGetDeviceCount(&d_count) );
		return d_count;
	}


	void query_devices (bool required = false) {
		const ulong d_count = get_device_count();
		if (required && !d_count)
			throw exceptions::no_device();

		devices.resize(d_count);

		#pragma omp parallel for
		for (ulong d = 0; d < d_count; ++d) {
			// HANDLE_ERROR( cudaSetDevice(d) );

			// get the device properties
			HANDLE_ERROR( cudaGetDeviceProperties(&devices[d], d) );
		}

		queried = true;
	}



	const device& get_device (ulong i) {
		return devices[i];
	}



}
