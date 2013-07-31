#ifndef ___MSC__CUDA__LINEAR_ALGEBRA__CUH___
#define ___MSC__CUDA__LINEAR_ALGEBRA__CUH___


namepspace CUDA { namespace LinearAlgebra { namespace block {


	/* AX - XB = C */
	template<typename T>
	__device__
	void bartel_stewart (const T * const a, const T * const b, const ulong m, const ulong n, T * const c) {
		for (ulong colIdx = 0; colIdx < n; colIdx) {
			// C(:,k) += C(:,0:k-1) * B(0:k-1,k)
			gemv(c, m, colIdx + 1, b + colIdx * n, c + colIdx * m);
		}
	}


}}}


#endif//___MSC__CUDA__LINEAR_ALGEBRA__CUH___
