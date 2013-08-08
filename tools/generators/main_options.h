#pragma once
#ifndef ___MAIN_OPTIONS_HH___
#define ___MAIN_OPTIONS_HH___

// c++ std headers
#include <string>
	using std::string;


typedef unsigned long ulong;

#ifndef MAIN
#define SCOPE extern
#else
#define SCOPE
#endif

SCOPE bool control;
SCOPE ulong dimension;
SCOPE string output_filename;

/** Parse the program arguments given in the command line.
 * Supported options
 *   --help: Shows the usage text.
 * \param argc The number of arguments in the command line.
 * \param argv The command line arguments.
 */
void parse_arguments (int argc, char *argv[]);

#endif//___MAIN_OPTIONS_HH___
