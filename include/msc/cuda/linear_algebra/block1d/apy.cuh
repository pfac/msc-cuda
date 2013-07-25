#ifndef ___MSC__CUDA__LINEAR_ALGEBRA__BLOCK1D__SAPY_CUH___
#define ___MSC__CUDA__LINEAR_ALGEBRA__BLOCK1D__SAPY_CUH___


namespace CUDA { namespace linear_algebra { namespace block1D {


	/* y = aIy */
	template<typename T>
	__device__
	void apy (const ulong n, const T alpha, T * const y, const ulong ldy) {
		for (ulong i = threadIdx.x; i < n; i += blockDim.x) {
			y[i * ldy] += alpha;
		}
	}


}}}


#endif//___MSC__CUDA__LINEAR_ALGEBRA__BLOCK1D__SAPY_CUH___
