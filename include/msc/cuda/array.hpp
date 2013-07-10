#ifndef ___MSC__CUDA__ARRAY_HPP___
#define ___MSC__CUDA__ARRAY_HPP___

// project headers
#include <msc/cuda/error>


// macros
#define HANDLE_ERROR(e) handle_error(e, __FILE__, __LINE__)


namespace CUDA {



	template<typename T>
	class array {
		size_t data_size;
		T * device_data;
	public:
		array (const T * const host_data, const ulong count) {
			data_size = sizeof(T) * count;
			HANDLE_ERROR( cudaMalloc(&device_data, data_size) );
			HANDLE_ERROR( cudaMemcpy(device_data, host_data, data_size, cudaMemcpyDeviceToHost) );
		}

		~array () {
			HANDLE_ERROR( cudaFree(device_data) );
		}

		T * get_pointer() const { return device_data; }
	};



}


#endif//___MSC__CUDA__ARRAY_HPP___
