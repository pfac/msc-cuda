#ifndef ___MSC__CUDA__DEVICE_HPP___
#define ___MSC__CUDA__DEVICE_HPP___

// std C++ headers
#include <iostream>
#include <sstream>
#include <string>

// CUDA headers
#include <cuda_runtime.h>


// names
using std::ostream;
using std::string;
using std::stringstream;


// types
typedef unsigned long ulong;


// macros
#define endl '\n'


namespace CUDA {



struct device : public cudaDeviceProp {

	ulong cores_per_sm () const {
		const ulong version = (major << 4) + minor;

		struct {
			ulong version;
			ulong cores;
		} cores_per_sm[] = {
			{ 0x10,   8 }, // Tesla Generation (SM 1.0) G80 class
	        { 0x11,   8 }, // Tesla Generation (SM 1.1) G8x class
	        { 0x12,   8 }, // Tesla Generation (SM 1.2) G9x class
	        { 0x13,   8 }, // Tesla Generation (SM 1.3) GT200 class
	        { 0x20,  32 }, // Fermi Generation (SM 2.0) GF100 class
	        { 0x21,  48 }, // Fermi Generation (SM 2.1) GF10x class
	        { 0x30, 192 }, // Kepler Generation (SM 3.0) GK10x class
	        { 0x35, 192 }, // Kepler Generation (SM 3.5) GK11x class
		};

		ulong i;
		for (i = 0; i < 8 && version != cores_per_sm[i].version; ++i);

		return cores_per_sm[i].cores;
	}


	ulong cores_total () const {
		return cores_per_sm() * multiProcessorCount;
	}


	/* human readable */
	static
	string str_memory (const ulong bytes) {
		double total;
		char prefix;

		if (bytes > 1073741824) {
			total = bytes / 1073741824.0;
			prefix = 'G';
		} else if (bytes > 1048576) {
			total = bytes / 1048576.0;
			prefix = 'M';
		} else if (bytes > 1024) {
			total = bytes / 1024.0;
			prefix = 'K';
		} else {
			total = bytes;
			prefix = '\0';
		}

		stringstream ss;
		ss << total << ' ';
		if (prefix)
			ss << prefix;
		ss << 'B';

		return ss.str();
	}


	bool supports_double_precision () const {
		return major > 1 || (major == 1 && minor >= 3);
	}

	friend
	ostream& operator<< (ostream& out, const device& dev) {
		return out
			<< '[' << dev.name << ']' << endl
			<< "CUDA Capability: " << dev.major << '.' << dev.minor << endl
			<< "Global memory:   " << device::str_memory(dev.totalGlobalMem) << endl
			<< "Multiprocessors: " << dev.multiProcessorCount << endl
			<< "CUDA cores:      " << dev.cores_total() << '(' << dev.cores_per_sm() << " cores per SM)" << endl
			<< "Constant memory: " << device::str_memory(dev.totalConstMem) << endl
			<< "Shared memory:   " << device::str_memory(dev.sharedMemPerBlock) <<  " per block" << endl
			;
	}
};



}


#undef endl

#endif//___MSC__CUDA__DEVICE_HPP___
