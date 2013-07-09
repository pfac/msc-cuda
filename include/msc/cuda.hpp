#ifndef ___MSC__CUDA_HPP___
#define ___MSC__CUDA_HPP___

// project headers
#include <msc/cuda_bits/device>

// std C++ headers
#include <vector>


// names
using std::vector;


// types
typedef unsigned long ulong;


namespace CUDA {

	

	/** Queries the properties of the CUDA devices present in the system via CUDA Runtime API.
	 * Based on the deviceQuery CUDA sample.
	 *
	 * \param required If true, an exception will be thrown if no CUDA device is found.
	 */
	void query_devices (bool required);



	/** Checks whether the system has any CUDA device.
	 *
	 * \return True if there is at least one CUDA device in the system, False otherwise.
	 */
	bool has_devices ();



	/** Retrieves the information of a CUDA device.
	 *
	 * \param i The index, in the system, of the CUDA device to be retrieved.
	 * \return An instance of the device class, containing all the information required.
	 */
	const device& get_device (ulong i);



}


#endif//___MSC__CUDA_HPP___
