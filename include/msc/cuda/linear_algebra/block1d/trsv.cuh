#ifndef ___MSC__CUDA__LINEAR_ALGEBRA__BLOCK1D__TRSV_CUH___
#define ___MSC__CUDA__LINEAR_ALGEBRA__BLOCK1D__TRSV_CUH___


namespace CUDA { namespace linear_algebra { namespace block1D {


	/* Ax = b */
	template<typename T>
	__device__
	void trsv (const ulong n, const T * const a, T * const x) {
		for (ulong col = n; col; --col) {
			const ulong colIdx = col - 1;
			if (threadIdx.x == 0)
				x[colIdx] /= a[colIdx * n + colIdx];

			for (ulong rowIdx = threadIdx.x; rowIdx < colIdx; rowIdx += blockDim.x) {
				const ulong idx = colIdx * n + rowIdx;
				x[rowIdx] -= x[colIdx] * a[idx];
			}
		}
	}


}}}


#endif//___MSC__CUDA__LINEAR_ALGEBRA__BLOCK1D__TRSV_CUH___
