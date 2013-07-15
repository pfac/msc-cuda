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
			const ulong idx = e * stride;
			t[idx] = sqrt(t[idx]);
		}
	}


	template<typename T>
	__global__
	void _sqrtm_d1 (T * const t, const ulong m) {
		const ulong stride = m + 1;
		for (ulong elem = blockIdx.x * blockDim.x + threadIdx.x; elem < m; elem += blockDim.x * gridDim.x) {
			const ulong ii = elem;
			const ulong jj = elem + 1;
			const ulong idx = ii * m + jj;// row-major
			// const ulong idx = jj * m + ii;// col-major
			const T f = t[ii * stride];
			const T g = t[jj * stride];

			t[idx] = t[idx] / (f + g);
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

		_sqrtm_d1<<< blocks , threads_per_block >>>(d_t.get_pointer(), m);
		HANDLE_LAST_ERROR();

		d_t.to_host(h_t);
	}



// }



#endif//___MSC__CORE__GPU__POINT_CUH___
