#ifndef ___UNBLOCKIFY___
#define ___UNBLOCKIFY___


#include "copy.h"

// std C++ headers
#include <algorithm>

using std::min;


typedef unsigned long ulong;



template<typename T>
void unblockify (
		const T * const blocks, 
		const ulong matrix_row_count,
		const ulong matrix_col_count,
		const ulong full_block_row_count,
		const ulong full_block_col_count,
		T * const matrix
) {
	const ulong matrix_block_col_count = matrix_col_count / full_block_col_count + (matrix_col_count % full_block_col_count != 0);
	const ulong matrix_block_row_count = matrix_row_count / full_block_row_count + (matrix_row_count % full_block_row_count != 0);

	// for every block
	#pragma omp parallel for collapse(2)
	for (ulong bc = 0; bc < matrix_block_col_count; ++bc) {
		for (ulong br = 0; br < matrix_block_row_count; ++br) {
			// columns
			const ulong block_col_limits[2] = {
				bc * full_block_col_count,
				min((bc + 1) * full_block_col_count, matrix_col_count) - 1
			};
			const ulong block_col_count = block_col_limits[1] - block_col_limits[0] + 1;
			const ulong matrix_block_col_first = block_col_limits[0] * matrix_row_count;
		
			// rows
			const ulong block_row_limits[2] = {
				br * full_block_row_count,
				min((br + 1) * full_block_row_count, matrix_row_count) - 1
			};
			const ulong block_row_count = block_row_limits[1] - block_row_limits[0] + 1;
			const ulong block_first = matrix_block_col_first + br * block_col_count * full_block_row_count;

			// for every column in the block
			#pragma omp parallel for
			for (ulong c = 0; c < block_col_count; ++c) {
				      T * const destiny = matrix + matrix_block_col_first + c * matrix_row_count + block_row_limits[0];
				const T * const source = blocks + block_first + c * block_row_count;

				copy(source, destiny, block_row_count);
			}
		}
	}
}



#endif//___UNBLOCKIFY___
