//! \file back_substitution.hpp
//! \brief Functions that implement the Back Substitution algorithm.
/////////////////////////////////////////////////////////////////////////
//
//  Copyright (C) 2013  Pedro F. A. Costa <dev@iampfac.com>
//  
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//  
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
/////////////////////////////////////////////////////////////////////////

#ifndef ___MSC__LINEAR_ALGEBRA__BACK_SUBSTITUTION_HPP___
#define ___MSC__LINEAR_ALGEBRA__BACK_SUBSTITUTION_HPP___

// types
typedef unsigned long ulong;


//! \brief Solves the linear system Ax=b, overwriting b with x.
//!
//! This function solves the linear system Ax=b, where A is upper
//! triangular matrix (only Back Substitution is required).
//!
//! This particular implementation overwrites the content of b with
//! the content of x.
//!
//! \param a The matrix A, assumed to be upper triangular.
//! \param b The vector b. When this function finishes, this array has
//!          become the vector x.
//! \param n The dimension of A, which is also the number of elements
//!          in b and x.
//!
template<typename T>
void back_substitution (const T * const a, T * const b, const ulong n) {
	for (ulong i = n - 1; i; --i) {
		b[i] /= a[i * n + i];

		for (ulong k = 0; k < i; ++k) {
			const ulong idx = i * n + k;// column-major
			b[k] -= b[i] * a[idx];
			// a[idx] = 0;// optional and destructive
		}
	}
}


#endif//___MSC__LINEAR_ALGEBRA__BACK_SUBSTITUTION_HPP___
