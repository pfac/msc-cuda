#ifndef ___MSC__CUDA__EXCEPTIONS__NO_DEVICE_HPP___
#define ___MSC__CUDA__EXCEPTIONS__NO_DEVICE_HPP___

// std C++ headers
#include <exception>


namespace CUDA { namespace exceptions {



class no_device : public std::exception {
	virtual
	const char * what() const throw() {
		return "No CUDA device found in the system";
	}
};



}}


#endif//___MSC__CUDA__EXCEPTIONS__NO_DEVICE_HPP___
