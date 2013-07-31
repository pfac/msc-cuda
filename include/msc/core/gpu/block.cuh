#ifndef ___MSC__CORE__GPU__BLOCK_CUH___
#define ___MSC__CORE__GPU__BLOCK_CUH___

// project headers
#include <msc/core/cpu/blockify.h>
#include <msc/core/cpu/unblockify.h>
#include <msc/cuda/array>
#include <msc/cuda/linear_algebra/block1d/gemm>
#include <msc/cuda/linear_algebra/block1d/trsyl>
#include "../../../../vendor/nvidia/cuda/cuPrintf.cu"

// std C++ headers
#include <algorithm>
#include <limits>


// names
using std::min;
using std::numeric_limits;


// types
typedef unsigned long ulong;


// macros
#define IDIVCEIL(x,y) (((x) + (y) - 1) / (y))
#define MIN(x,y) ((x) < (y) ? (x) : (y))
#define endl '\n'


/** Solve the main diagonal of a matrix.
 *  This function uses the threads uses only the threads of a particular
 * block to solve the main diagonal of a given matrix. Meant solve a
 * particular block using the point method.
 *
 * \param mat The matrix to solve. This function changes the values in this matrix to those of the solution.
 * \param matDim The dimension of mat, which is assumed to be squared and contiguous in memory.
 */
template<typename T>
__device__
void ____sqrtm_d0 (T * const mat, const ulong matDim) {
	const ulong blckDim = blockDim.x;
	const ulong tid = threadIdx.x;

	const ulong stride = matDim + 1;

	for (ulong elem = tid; elem < matDim; elem += blckDim) {
		const ulong idx = elem * stride;
		mat[idx] = sqrt(mat[idx]);
	}
}


/** Solve the first super-diagonal of a matrix.
 *  This function uses the threads uses only the threads of a particular
 * block to solve the first super-diagonal of a given matrix. Meant solve
 * a particular block using the point method.
 *
 * \param mat The matrix to solve. This function changes the values in this matrix to those of the solution.
 * \param matDim The dimension of mat, which is assumed to be squared and contiguous in memory.
 */
template<typename T>
__device__
void ____sqrtm_d1 (T * const mat, const ulong matDim) {
	const ulong stride = matDim + 1;
	const ulong elem_count = matDim - 1;
	for (ulong elem = threadIdx.x; elem < elem_count; elem += blockDim.x) {
		const ulong ii = elem;
		const ulong jj = elem + 1;
		const ulong idx = jj * matDim + ii;

		const T f = mat[ii * stride];
		const T g = mat[jj * stride];

		mat[idx] = mat[idx] / (f + g);
	}
}


/** Solve the dependencies of the n-th super-diagonal of a matrix, where
 * n > 1.
 *  This function uses the threads uses only the threads of a particular
 * block to solve the dependencies of the n-th super-diagonal of a given
 * matrix. Meant solve a particular block using the point method.
 *
 * \param superDiagIdx The index of the super-diagonal. The main diagonal
 *                     has index 0, the first super-diagonal has index 1,
 *                     and so on. NOTE: n=0 and n=1 should not be solved
 *                     using this function.
 *
 * \param mat The matrix to solve. This function changes the values in
 *            this matrix to those of the solution.
 *
 * \param matDim The dimension of mat, which is assumed to be squared
 *               and contiguous in memory.
 */
template<typename T>
__device__
void _____solve_dependencies (const ulong superDiagIdx, T * const mat, const ulong matDim) {
	const ulong elem_count = matDim - superDiagIdx;
	for (ulong elem = threadIdx.x; elem < elem_count; elem += blockDim.x) {
		const ulong ii = elem;
		const ulong jj = elem + superDiagIdx;
		const ulong idx = jj * matDim + ii;

		const ulong kk_limits[2] = { ii + 1, jj - 1 };
		for (ulong kk = kk_limits[0]; kk <= kk_limits[1]; ++kk) {
			const T cc = mat[jj * matDim + kk];
			const T rr = mat[kk * matDim + ii];
			mat[idx] -= cc * rr;
		}
	}
}


/** Solve the n-th super-diagonal of a matrix, where n > 1.
 *  This function uses the threads uses only the threads of a particular
 * block to solve the n-th super-diagonal of a given matrix. Meant solve
 * a particular block using the point method.
 *  Each element of the n-th diagonal is assumed to contain the proper
 * value after solving it's dependencies. See _____solve_dependencies for
 * further details.
 *
 * \param superDiagIdx The index of the super-diagonal. The main diagonal
 *                     has index 0, the first super-diagonal has index 1,
 *                     and so on. NOTE: n=0 and n=1 should not be solved
 *                     using this function.
 *
 * \param mat The matrix to solve. This function changes the values in
 *            this matrix to those of the solution.
 *
 * \param matDim The dimension of mat, which is assumed to be squared
 *               and contiguous in memory.
 */
template<typename T>
__device__
void _____sqrtm_d (const ulong superDiagIdx, T * const mat, const ulong matDim) {
	const ulong stride = matDim + 1;
	const ulong elem_count = matDim - superDiagIdx;
	for (ulong elem = threadIdx.x; elem < elem_count; elem += blockDim.x) {
		const ulong ii = elem;
		const ulong jj = elem + superDiagIdx;
		const ulong idx = jj * matDim + ii;

		const T f = mat[ii * stride];
		const T g = mat[jj * stride];

		mat[idx] = mat[idx] / (f + g);
	}
}


/** Solve the n-th super-diagonal of a matrix, where n > 1.
 *  This function uses the threads uses only the threads of a particular
 * block to solve the n-th super-diagonal of a given matrix. Meant solve
 * a particular block using the point method.
 *  Each element of the n-th diagonal is assumed to contain the proper
 * value after solving it's dependencies. See _____solve_dependencies for
 * further details.
 *
 * \param superDiagIdx The index of the super-diagonal. The main diagonal
 *                     has index 0, the first super-diagonal has index 1,
 *                     and so on. NOTE: n=0 and n=1 should not be solved
 *                     using this function.
 *
 * \param mat The matrix to solve. This function changes the values in
 *            this matrix to those of the solution.
 *
 * \param matDim The dimension of mat, which is assumed to be squared
 *               and contiguous in memory.
 */
template<typename T>
__device__
void ____sqrtm_d (const ulong superDiagIdx, T * const mat, const ulong matDim) {
	_____solve_dependencies(superDiagIdx, mat, matDim);
	__syncthreads();
	_____sqrtm_d(superDiagIdx, mat, matDim);
}



template<typename T>
__device__
void ___sqrtm (T * const mat, const ulong matDim) {
	____sqrtm_d0(mat, matDim);
	____sqrtm_d1(mat, matDim);

	for (ulong superDiagIdx = 2; superDiagIdx < matDim; ++superDiagIdx)
		____sqrtm_d(superDiagIdx, mat, matDim);
}


template<typename T>
__global__
void __sqrtm_d0 (T * const mat, const ulong matDim, const ulong block_count) {
	const ulong bid = blockIdx.x;

	for (ulong blckIdx = bid; blckIdx < block_count; blckIdx += gridDim.x) {
		const ulong ii[2] = {
			blckIdx * blockDim.x,
			MIN((blckIdx + 1) * blockDim.x, matDim) - 1
		};

		const ulong blckDim = ii[1] - ii[0] + 1;
		const ulong idx = ii[0] * matDim + ii[0] * blckDim;

		___sqrtm(mat + idx, blckDim);
	}
}


template<typename T>
__global__
void __sqrtm_d1 (T * const mat, const ulong matDim, const ulong block_count) {
	const ulong bid = blockIdx.x;

	for (ulong blckIdx = bid; blckIdx < block_count; blckIdx += gridDim.x) {
		const ulong ii[2] = {
			blckIdx * blockDim.x,
			(blckIdx + 1) * blockDim.x - 1
		};
		const ulong jj[2] = {
			(blckIdx + 1) * blockDim.x,
			MIN((blckIdx + 2) * blockDim.x, matDim) - 1
		};

		const ulong ii_count = blockDim.x;
		const ulong jj_count = jj[1] - jj[0] + 1;

		const ulong f_first = ii[0] * matDim + ii[0] * ii_count;
		const ulong g_first = jj[0] * matDim + jj[0] * jj_count;
		const ulong c_first = jj[0] * matDim + ii[0] * jj_count;

		const T * const f = mat + f_first;// Tii
		const T * const g = mat + g_first;// Tjj
		T * const c = mat + c_first;// Tij

		CUDA::linear_algebra::block1D::trsyl(ii_count, jj_count, f, g, c);
	}
}


template<typename T>
__global__
void __sqrtm_d (const ulong d, T * const mat, const ulong matDim, const ulong block_count) {
	const ulong bid = blockIdx.x;

	for (ulong blckIdx = bid; blckIdx < block_count; blckIdx += gridDim.x) {
		const ulong ii[2] = {
			blckIdx * blockDim.x,
			(blckIdx + 1) * blockDim.x - 1
		};
		const ulong jj[2] = {
			(blckIdx + d) * blockDim.x,
			MIN((blckIdx + d + 1) * blockDim.x, matDim) - 1
		};

		const ulong ii_count = blockDim.x;
		const ulong jj_count = jj[1] - jj[0] + 1;

		const ulong f_first = ii[0] * matDim + ii[0] * ii_count;
		const ulong g_first = jj[0] * matDim + jj[0] * jj_count;
		const ulong c_first = jj[0] * matDim + ii[0] * jj_count;

		const T * const f = mat + f_first;// Tii
		const T * const g = mat + g_first;// Tjj
		T * const c = mat + c_first;// Tij

		for (ulong depIdx = 1; depIdx < d; ++depIdx) {
			const ulong kk[2] = {
				(blckIdx + depIdx) * blockDim.x,
				(blckIdx + depIdx + 1) * blockDim.x - 1
			};

			const ulong a_first = kk[0] * matDim + ii[0] * blockDim.x;
			const ulong b_first = jj[0] * matDim + kk[0] * jj_count;

			const T * a = mat + a_first;
			const T * b = mat + b_first;

			CUDA::linear_algebra::block1D::gemm(ii_count, jj_count, ii_count, T(-1), a, b, T(1), c);
		}

		CUDA::linear_algebra::block1D::trsyl(ii_count, jj_count, f, g, c);
	}
}


template<typename T>
__host__
void _sqrtm (T * const host_data, const ulong matDim, const ulong block_size, const ulong cuda_threads_per_block, const ulong cuda_blocks) {
	const ulong block_count = IDIVCEIL(matDim, cuda_threads_per_block);
	CUDA::array<T> device_data(host_data, matDim * matDim);
	T * const ptr = device_data.get_pointer();

	__sqrtm_d0<<< cuda_blocks , cuda_threads_per_block >>>(ptr, matDim, block_count);
	HANDLE_LAST_ERROR();

	__sqrtm_d1<<< cuda_blocks , cuda_threads_per_block >>>(ptr, matDim, block_count - 1);
	HANDLE_LAST_ERROR();

	for (ulong dd = 2; dd < block_count; ++dd) {
		__sqrtm_d<<< cuda_blocks , cuda_threads_per_block >>>(dd, ptr, matDim, block_count - dd);
		HANDLE_LAST_ERROR();
	}

	device_data.to_host(host_data);
}


template<typename T>
__host__
void sqrtm (T * const host_data, const ulong matDim, const ulong block_size, const ulong cuda_threads_per_block, const ulong cuda_blocks) {
	T * const blockified_host_data = new T[matDim * matDim];
	blockify(host_data, matDim, matDim, block_size, block_size, blockified_host_data);
	_sqrtm(blockified_host_data, matDim, block_size, cuda_threads_per_block, cuda_blocks);
	unblockify(blockified_host_data, matDim, matDim, block_size, block_size,  host_data);
	delete[] blockified_host_data;
}


template<typename T>
__host__
void sqrtm (T * const host_data, const ulong matDim, const ulong block_size, const ulong cuda_threads_per_block) {
	sqrtm(host_data, matDim, block_size, cuda_threads_per_block, IDIVCEIL(matDim, cuda_threads_per_block));
}


template<typename T>
__host__
void sqrtm (T * const host_data, const ulong matDim, const ulong block_size) {
	sqrtm(host_data, matDim, block_size, block_size);
}


#endif//___MSC__CORE__GPU__BLOCK_CUH___
