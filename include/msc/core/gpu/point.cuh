#ifndef ___MSC__CORE__GPU__POINT_CUH___
#define ___MSC__CORE__GPU__POINT_CUH___

// project headers
#include <msc/cuda/array>
#include <msc/cuda/error>
#include "../../../../vendor/nvidia/cuda/cuPrintf.cu"

// CUK headers
#include <cuk/stopwatch>



namespace point {



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
	void _sqrtm_d1 (T * const t, const ulong tDim) {
		const ulong stride = tDim + 1;
		const ulong diagcount = tDim - 1;
		for (ulong elem = blockIdx.x * blockDim.x + threadIdx.x; elem < diagcount; elem += blockDim.x * gridDim.x) {
			const ulong ii = elem;
			const ulong jj = elem + 1;
			const ulong idx = jj * tDim + ii;
			const T f = t[ii * stride];
			const T g = t[jj * stride];

			t[idx] = t[idx] / (f + g);
		}
	}



	template<typename T>
	__global__
	void __solve_dependencies (const ulong dd, T * const t, const ulong tDim) {
		const ulong diagcount = tDim - dd;
		for (ulong elem = blockIdx.x * blockDim.x + threadIdx.x; elem < diagcount; elem += blockDim.x * gridDim.x) {
			const ulong ii = elem;
			const ulong jj = elem + dd;
			const ulong idx = jj * tDim + ii;

			const ulong kk_limits[2] = { ii + 1, jj - 1 };
			for (ulong kk = kk_limits[0]; kk <= kk_limits[1]; ++kk) {
				const ulong ccIdx = jj * tDim + kk;
				const ulong rrIdx = kk * tDim + ii;

				const T cc = t[ccIdx];
				const T rr = t[rrIdx];

				t[idx] -= rr * cc;
			}
		}
	}



	template<typename T>
	__global__
	void __sqrtm_d (const ulong dd, T * const t, const ulong tDim) {
		const ulong stride = tDim + 1;
		const ulong elem_count = tDim - dd;
		for (ulong elem = blockIdx.x * blockDim.x + threadIdx.x; elem < elem_count; elem += blockDim.x * gridDim.x) {
			const ulong ii = elem;
			const ulong jj = elem + dd;
			const ulong idx = jj * tDim + ii;

			const T f = t[ii * stride];
			const T g = t[jj * stride];

			t[idx] = t[idx] / (f + g);
		}
	}



	template<typename T>
	__host__
	void _sqrtm_d (const ulong dd, T * const t, const ulong m, const ulong blocks, const ulong threads_per_block) {
		__solve_dependencies<<< blocks, threads_per_block >>>(dd, t, m);
		__sqrtm_d<<< blocks, threads_per_block >>>(dd, t, m);
	}



	template<typename T>
	__host__
	void sqrtm (T * const h_t, const ulong m, double& nanoseconds) {
		cuk::stopwatch sw;
		CUDA::array<T> d_t(h_t, m * m);
		T * const ptr = d_t.get_pointer();
		const ulong threads_per_block = 32;
		const ulong blocks = (m + threads_per_block - 1) / threads_per_block;

		sw.start();

		_sqrtm_d0<<< blocks , threads_per_block >>>(ptr, m);
		HANDLE_LAST_ERROR();

		_sqrtm_d1<<< blocks , threads_per_block >>>(ptr, m);
		HANDLE_LAST_ERROR();

		for (ulong dd = 2; dd < m; ++dd)
			_sqrtm_d(dd, ptr, m, blocks, threads_per_block);

		sw.stop();

		d_t.to_host(h_t);
		nanoseconds = sw.ns();
	}



}



#endif//___MSC__CORE__GPU__POINT_CUH___
