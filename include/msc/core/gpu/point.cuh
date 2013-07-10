#ifndef ___MSC__CORE__GPU__POINT_CUH___
#define ___MSC__CORE__GPU__POINT_CUH___

// project headers
#include <msc/cuda/array>
#include <msc/cuda/error>
#include "../../../../vendor/nvidia/cuda/cuPrintf.cu"



// namespace point {



	template<typename T>
	__global__
	void _sqrtm_d0 (T * const t, const ulong m) {
		const ulong stride = m + 1;
		for (ulong e = blockIdx.x * blockDim.x + threadIdx.x; e < m; e += blockDim.x * gridDim.x) {
			const ulong i = e * stride;
			t[i] = sqrt(t[i]);
		}
	}



	template<typename T>
	__host__
	void sqrtm (T * const h_t, const ulong m) {
		CUDA::array<T> d_t(h_t, m * m);
		const ulong threads_per_block = 32;
		const ulong blocks = (m + threads_per_block - 1) / threads_per_block;

		

		_sqrtm_d0<<< blocks , threads_per_block >>>(d_t.get_pointer(), m);
		HANDLE_LAST_ERROR();

		d_t.to_host(h_t);
	}



// }



#endif//___MSC__CORE__GPU__POINT_CUH___
