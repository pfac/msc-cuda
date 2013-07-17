Efficient Computation of the Matrix Square Root in CUDA C
=========================================================

This package implements **part** of the algorithm to compute the square root of a given matrix.

## Intro
Let *A* be a matrix. A square root of *A* is defined as any matrix *X* such that *A = X^2*. When it exists, *X* is not unique, but a **principal square root A^(1/2)** is. This matrix exists if and only if *A* has no real negative eigenvalues.

The Schur method of Björck and Hammarling is the most numerically stable method for computing the square root of a matrix. It reduces *A* to an upper quasi-triangular form *T* and solves a system of equations to obtain *U*, upper triangular, such that *T = U^2*. This is implemented in MATLAB as the `sqrtm` and `sqrtm_real` functions.

The focus of this project is to implement and analyse the core step of the algorithm, which computes *U*, using CUDA C to enable the use of NVIDIA GPUs.

## Caveats

At the moment, the only purpose of this package is as proof-of-concept and for performance analysis. As such, the application is not complete, and some assumptions are considered:

- *A* is assumed to be real. Complex matrices **are not** supported.
- *A* is assumed to be in upper triangular, and as such *T=A*. Full and quasi-triangular matrices **are not** supported.

## Task List

[x] Point method
[] Block method
[] Doxygen documentation
[] Unit tests


## Building

In order to build the main executable in this project, you'll need the following dependencies:

- **CMake 2.8.10 or later**;
- **Boost 1.34.0 or later**
- The **NVIDIA CUDA SDK** (version 5.0 is being used for development, previous versions were not tested);
- A C++ compiler compatible with the CUDA SDK (*i686-apple-darwin11-llvm-gcc-4.2* is being used for development);
- The **Armadillo** library (C++ linear algebra, used for the `arma2plain` conversion tools);

Running `make Release` in the top directory should be enough to build the project in release mode. After, the main executable can be found in `build/none/bin`.

### Advanced

The `Makefile` in the top directory creates a new directory `build/<BUILD_TYPE>`. By default, the build type `None` is used. See CMake documentation on build types for further details.

Executables are stored in `build/<BUILD_TYPE>/bin`.


## Documentation

*Pending*


## Tests

*Pending*


## References
- Nicholas J. Higham, *Functions of Matrices: Theory and Computation*
- Åke Bjórck and Sven Hammarling, *A Schur method for the square root of a matrix*
- Edvin Deadman, Nicholas J. Higham, and Rui Ralha, *Blocked Schur Algorithms for Computing the Matrix Square Root*


## About

MSc-Thesis by Pedro Costa, 2013. This project is being developed in collaboration with the Numerical Algorithms Group and funded by the Portuguese agency FCT, Fundação para a Ciência e Tecnologia, under the program UT Austin | Portugal

