#define MAIN

// project headers
#include <msc/cuda/core>

// std C++ headers
#include <iostream>

// extra headers


// names
using std::cout;


// macros
#define endl '\n'


int main (int argc, char * argv[]) {

	CUDA::query_devices(true);
	cout << CUDA::get_device(0) << endl;

	return 0;
}
