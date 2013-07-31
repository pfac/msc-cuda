#ifndef ___MSC__CUDA__ERROR_HPP___
#define ___MSC__CUDA__ERROR_HPP___

// std C++ headers
#include <iostream>

// CUDA headers
#include <cuda_runtime.h>


// types
typedef unsigned long ulong;


// macros
#define HANDLE_ERROR(e) CUDA::handle_error(e, __FILE__, __LINE__)
#define HANDLE_LAST_ERROR() CUDA::handle_last_error(__FILE__, __LINE__)
#define endl '\n'


namespace CUDA {


	inline
	void handle_error (const cudaError_t error, const char * file, const ulong line) {
		if (error != cudaSuccess) {
			std::cerr << "\033[31;1m[CUDA ERROR]\033[0m  " << file << ':' << line << "  " << cudaGetErrorString(error) << endl;
			throw error;
			// exit(error);
		}
	}

	inline
	void handle_last_error (const char * file, const ulong line) {
		handle_error(cudaPeekAtLastError(), file, line);
		handle_error(cudaDeviceSynchronize(), file, line);
	}



}


#endif//___MSC__CUDA__ERROR_HPP___
