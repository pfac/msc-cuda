#ifndef ___MSC__CUDA__ARRAY_HPP___
#define ___MSC__CUDA__ARRAY_HPP___

// project headers
#include <msc/cuda/error>


namespace CUDA {



	template<typename T>
	class array {
		size_t data_size;
		T * device_data;
	public:
		array (const T * const host_data, const ulong count) {
			data_size = sizeof(T) * count;
			HANDLE_ERROR( cudaMalloc(&device_data, data_size) );
			HANDLE_ERROR( cudaMemcpy(device_data, host_data, data_size, cudaMemcpyHostToDevice) );
		}

		~array () {
			if (device_data)
				HANDLE_ERROR( cudaFree(device_data) );
		}


		//
		// memory
		//
		void to_host (T * const host_data) const {
			HANDLE_ERROR( cudaMemcpy(host_data, device_data, data_size, cudaMemcpyDeviceToHost) );
		}


		//
		// getters
		//
		T * get_pointer() const { return device_data; }
	};



}


#endif//___MSC__CUDA__ARRAY_HPP___
