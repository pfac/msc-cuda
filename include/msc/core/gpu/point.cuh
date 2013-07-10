#ifndef ___MSC__CORE__GPU__POINT_HPP___
#define ___MSC__CORE__GPU__POINT_HPP___

// project headers
#include <msc/cuda/array>



// namespace point {



	template<typename T>
	__global__
	void _sqrtm_d0 (T * const t, const ulong m) {
		const ulong stride = m + 1;
		const ulong e = blockIdx.x;
		const ulong i = e * stride;
		t[i] = sqrt(t[i]);
	}



	template<typename T>
	__host__
	void sqrtm (T const * h_t, const ulong m) {
		CUDA::array<T> d_t(h_t, m);
		const ulong threads_per_block = 32;
		const ulong blocks = (m + threads_per_block - 1) / threads_per_block;

		_sqrtm_d0<<< blocks , threads_per_block >>>(d_t.get_pointer(), m);
	}



// }



#endif//___MSC__CORE__GPU__POINT_HPP___
