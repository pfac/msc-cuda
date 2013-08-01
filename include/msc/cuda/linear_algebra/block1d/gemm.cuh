#ifndef ___MSC__CUDA__LINEAR_ALGEBRA__BLOCK1D__GEMM_CUH___
#define ___MSC__CUDA__LINEAR_ALGEBRA__BLOCK1D__GEMM_CUH___


namespace CUDA { namespace linear_algebra { namespace block1D {


	/* C = alpha * A * B + beta * C */
	template<typename T>
	__device__
	void gemm (const ulong m, const ulong n, const ulong l, const T alpha, const T * const a, const T * const b, const T beta, T * const c) {
		for (ulong j = 0; j < n; ++j) {
			for (ulong i = threadIdx.x; i < m; i += blockDim.x) {
				c[j * m + i] *= beta;
			}

			for (ulong k = 0; k < l; ++k) {
				for (ulong i = threadIdx.x; i < m; i += blockDim.x) {
					c[j * m + i] += alpha * a[k * m + i] * b[j * l + k];
				}
			}
		}
	}


}}}


#endif//___MSC__CUDA__LINEAR_ALGEBRA__BLOCK1D__GEMM_CUH___
