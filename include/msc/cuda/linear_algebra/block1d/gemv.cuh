#ifndef ___MSC__CUDA__LINEAR_ALGEBRA__BLOCK1D__GEMV_CUH___
#define ___MSC__CUDA__LINEAR_ALGEBRA__BLOCK1D__GEMV_CUH___


// types
typedef unsigned long ulong;


namespace CUDA { namespace linear_algebra { namespace block1D {


	/* y = Ax + y */
	template<typename T>
	__device__
	void gemv (const ulong m, const ulong n, const T * const a, const T * const x, T * const y) {
		for (ulong colIdx = 0; colIdx < n; ++colIdx) {
			for (ulong rowIdx = threadIdx.x; rowIdx < m; rowIdx += blockDim.x) {
				y[rowIdx] += a[colIdx * m + rowIdx] * x[colIdx];
			}
		}
	}


}}}


#endif//___MSC__CUDA__LINEAR_ALGEBRA__BLOCK1D__GEMV_CUH___
