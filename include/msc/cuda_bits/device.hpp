#ifndef ___MSC__CUDA__DEVICE_HPP___
#define ___MSC__CUDA__DEVICE_HPP___

// CUDA headers
#include <cuda_runtime.h>


namespace CUDA {



struct device : public cudaDeviceProp {
	bool supports_double_precision () const {
		return major > 1 || (major == 1 && minor >= 3);
	}
};



}


#endif//___MSC__CUDA__DEVICE_HPP___
