// std C++ headers
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// names
using std::cin;
using std::cout;
using std::string;
using std::stringstream;
using std::vector;

// types
typedef unsigned long ulong;

// macros
#define endl '\n'

int main () {
	typedef vector<double> line_v;
	typedef vector<line_v> matrix_v;

	matrix_v mat;
	string stream_line;

	ulong m = 0;
	ulong n = 0;

	// dynamic read
	while (getline(cin, stream_line)) {
		double x;
		stringstream ss;

		ss << stream_line;

		line_v line;
		while (ss >> x)
			line.push_back(x);

		mat.push_back(line);

		++m;
		n = (n > line.size()) ? n : line.size();
	}

	double * frozen = new double[m * n];

	cout << m << ',' << n << endl;

	delete[] frozen;

	return 0;
}
