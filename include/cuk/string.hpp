#ifndef ___CUK__STRING_HPP___
#define ___CUK__STRING_HPP___


// #include <iostream>
// 	using std::clog;
// 	#define endl '\n'
#include <string>
#include <sstream>
	using std::stringstream;
#include <vector>
	using std::vector;



namespace cuk {



struct string
: public std::string
{
	string (const std::string& s)
	: std::string(s)
	{}



	bool is_int () const {
		const string& self = *this;

		const size_t l = self.size();

		size_t i;

		for (i = 0; i < l && isdigit( self[i] ); ++i);

		return i == l;
	}



	void split (const std::string& delim, vector<string>& tokens) const {
		const string& self = *this;

		const size_t m = self.size();
		const size_t n = delim.size();

		size_t t = 0;// token starting position

		// iterate over the string
		for (size_t i = 0; i < m; ++i) {

			// iterate over each of the delimiter chars in the string delim
			for (size_t j = 0; j < n; ++j) {

				// current string character is in the delimiter string?
				if ( self[i] == delim[j] ) {
					tokens.push_back( self.substr( t, i - t ) );
					t = i + 1;
					break;
				}
			}
		}

		if (t < m)
			tokens.push_back( self.substr( t, m - t) );
	}



	vector<string> split (const std::string& delim) const {
		const string& self = *this;
		vector<string> tokens;

		self.split( delim, tokens );

		return tokens;
	}



	vector<string> split (const char delim) const {
		const string& self = *this;
		stringstream ss;
		ss << delim;
		const std::string _delim = ss.str();
		return self.split( _delim );
	}
};



}



#endif//___CUK__STRING_HPP___