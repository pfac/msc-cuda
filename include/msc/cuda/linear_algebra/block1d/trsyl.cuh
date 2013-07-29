#ifndef ___MSC__CUDA__LINEAR_ALGEBRA__BLOCK1D__TRSYL_HPP___
#define ___MSC__CUDA__LINEAR_ALGEBRA__BLOCK1D__TRSYL_HPP___

// project headers
#include <msc/cuda/linear_algebra/block1d/gemv>
#include <msc/cuda/linear_algebra/block1d/trpaisv>


namespace CUDA { namespace linear_algebra { namespace block1D {


	/* AX - XB = C */
	template<typename T>
	__device__
	void trsyl (const ulong m, const ulong n, const T * const a, const T * const b, T * const c) {
		{// column 0
			const ulong colIdx = 0;
			trpaisv(m, a, b[colIdx * n + colIdx], c + colIdx * n);
		}

		for (ulong colIdx = 1; colIdx < n; ++colIdx) {
			T * column = c + colIdx * n;
			gemv(m, colIdx, -1.0f, c, b + colIdx * n, column);
			trpaisv(m, a, b[colIdx * n + colIdx], column);
		}
	}
	

}}}


#endif//___MSC__CUDA__LINEAR_ALGEBRA__BLOCK1D__TRSYL_HPP___
