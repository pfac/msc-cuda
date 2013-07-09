#define MAIN

// project headers
#include "main_options.h"

// stc C++ headers
#include <iostream>


// names
using std::clog;


// macros
#define endl '\n'


int main (int argc, char * argv[]) {
	parse_arguments(argc, argv);

	#ifndef NDEBUG
	clog << "Options parsed:" << endl
	     << "filename: \"" << filename << '"' << endl
	     ;
	#endif

	return 0;
}
