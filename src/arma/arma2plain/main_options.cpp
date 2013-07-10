#include "main_options.h"

// c++ std headers
#include <iostream>
	using std::cout;
	using std::cerr;
#include <string>
	using std::string;
#include <exception>
	using std::exception;

// boost headers
#include <boost/program_options.hpp>
	using boost::program_options::options_description;
	using boost::program_options::positional_options_description;
	using boost::program_options::variables_map;
	using boost::program_options::command_line_parser;
	using boost::program_options::value;

#define endl '\n'

/** Parse the program arguments given in the command line.
 * Supported options
 *   --help: Shows the usage text.
 * \param argc The number of arguments in the command line.
 * \param argv The command line arguments.
 */
void parse_arguments (int argc, char *argv[]) {
	try {
		string smetrics;// list of metrics, to be parsed later

		options_description visible_options("Usage");
		visible_options.add_options()
			("help,h",  "Show the usage text.")
			("verbose", "Print a log of what is happening.")
		;

		options_description hidden_options;
		hidden_options.add_options()
			("input-file", value<string>(&filename), "")
		;

		options_description command_line_options;
		command_line_options.add(visible_options).add(hidden_options);

		positional_options_description pod;
		pod.add("input-file", -1);

		variables_map vm;
		store(command_line_parser(argc, argv).options(command_line_options).positional(pod).run(), vm);
		notify(vm);

		// --help prints usage information
		if (vm.count("help") > 0) {
			cout << visible_options << endl;
			exit(0);
		}

		verbose = vm.count("verbose") > 0;
	} catch (exception& e) {
		cerr << e.what() << endl;
		exit(-1);
	}
}
