#ifndef ___MSC__CUDA__LINEAR_ALGEBRA__BLOCK1D__GEMM_CUH___
#define ___MSC__CUDA__LINEAR_ALGEBRA__BLOCK1D__GEMM_CUH___


namespace CUDA { namespace linear_algebra { namespace block1D {


	template<typename T>
	__device__
	void gemm (const ulong m, const ulong n, const ulong l, const T alpha, const T * const a, const T * const b, const T beta, T * const c) {
		for (ulong j = 0; j < n; ++j) {
			for (ulong i = threadIdx.x; i < m; i += blockDim.x)
				c[j * n + i] *= beta;

			for (ulong k = 0; k < l; ++k) {
				for (ulong i = threadIdx.x; i < m; i += blockDim.x)
					c[j * n + i] += alpha * a[k * l + i] * b[j * n + k];
			}
		}
	}


}}}


#endif//___MSC__CUDA__LINEAR_ALGEBRA__BLOCK1D__GEMM_CUH___
