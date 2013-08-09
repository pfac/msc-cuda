#ifndef ___MSC__MATRIX_HPP___
#define ___MSC__MATRIX_HPP___

// std C++ headers
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>


// names
using std::cerr;
using std::ios_base;
using std::istream;
using std::ifstream;
using std::numeric_limits;
using std::ostream;
using std::string;
using std::stringstream;


// types
typedef unsigned long ulong;


// macros
#define endl '\n'


template<typename T>
class matrix {
	ulong r;
	ulong c;
	T * content;


	//
	// class methods
	//
	static
	string header();


	static
	T convert_naninf (const string& token) {
		T result;
		size_t l = token.size();

		if (l == 3 || l == 4) {
			const bool neg = (token[0] == '-');
			const bool pos = (token[0] == '+');

			const size_t offset = ((neg || pos) && l == 4);

			const string s = token.substr(offset, 3);
			
			if (s == "inf" || s == "Inf" || s == "INF")
				result = numeric_limits<T>::infinity();
			else if (s == "nan" || s == "Nan" || s == "NaN" || s == "NAN")
				result = numeric_limits<T>::quiet_NaN();
			else {
				stringstream ss(token);
				ss >> result;
			}

			if (neg)
				result = -result;

		} else {
			stringstream ss(token);
			ss >> result;
		}

		return result;
	}

	//
	// setters
	//
	void rows(const ulong r) { this->r = r; }
	void cols(const ulong c) { this->c = c; }

	//
	// utilities (read-only)
	//
	ulong linidx (const ulong i, const ulong j) const { return j * r + i; }


	void resize () {
		T * const content = new T[r * c];
		if (this->content) {
			memcpy(content, this->content, r * c * sizeof(T));
			delete[] this->content;
		}
		this->content = content;
	}


public:

	matrix (const string& filename) : r(0), c(0), content(NULL) {
		ifstream file;
		file.open(filename.c_str());
		file >> *this;
		file.close();
	}

	matrix (const ulong r, const ulong c) : r(r), c(c), content(NULL) { this->resize(); }


	//
	// getters
	//
	ulong rows() const { return r; }
	ulong cols() const { return c; }
	T * data_ptr() { return content; }


	//
	// utilities
	//
	void resize (const ulong r, const ulong c) {
		if (r == rows() && c == cols())
			return;

		rows(r);
		cols(c);
		resize();
	}


	//
	// operators
	//
	      T& operator() (const ulong i, const ulong j)       { return content[ linidx(i,j) ]; }
	const T& operator() (const ulong i, const ulong j) const { return content[ linidx(i,j) ]; }


	//
	// friends
	//
	friend
	istream& operator>> (istream& in, matrix<T>& mat) {
		string header;

		in >> header;
		in >> mat.r;
		in >> mat.c;

		if (header == "ARMA_MAT_TXT_FN008") {
			mat.resize();

			string token;
			stringstream ss;

			for (ulong i = 0; i < mat.r; ++i) {
				for (ulong j = 0; j < mat.c; ++j) {
					in >> token;

					ss.clear();
					ss.str(token);

					mat(i,j) = T(0);
					ss >> mat(i,j);

					if (ss.fail())
						mat(i,j) = matrix::convert_naninf(token);
				}
			}
		} else {
			cerr << "ERROR[stream >> matrix]: incorrect header \"" << header << '\"' << endl;
			cerr << "   expecting \"" << matrix<T>::header() << '\"' << endl;
		}

		return in;
	}


	friend
	ostream& operator<< (ostream& out, const matrix<T>& m) {
		ios_base::fmtflags state = out.flags();

		out << m.r << ' ' << m.c << endl;
		for (ulong i = 0; i < m.r; ++i) {
			for (ulong j = 0; j < m.c; ++j) {
				out.put(' ');
				out << m(i,j);
			}
			out.put(endl);
		}

		out.flags(state);
		return out;
	}
};


template<>
string matrix<float>::header () { return "ARMA_MAT_TXT_FN004"; }


template<>
string matrix<double>::header () { return "ARMA_MAT_TXT_FN008"; }


#undef endl


#endif//___MSC__MATRIX_HPP___
