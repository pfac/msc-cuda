#ifndef ___MSC__MATRIX_HPP___
#define ___MSC__MATRIX_HPP___

// std C++ headers
#include <fstream>
#include <sstream>


// names
using std::ios_base;
using std::istream;
using std::ifstream;
using std::ostream;
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
	// setters
	//
	void rows(const ulong r) { this->r = r; }
	void cols(const ulong c) { this->c = c; }

	//
	// utilities (read-only)
	//
	ulong linidx (const ulong i, const ulong j) const { return i * c + j; }


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
	istream& operator>> (istream& in, matrix<T>& m) {
		{// first line: dimensions
			string line;
			stringstream ss;
			ulong r, c;

			getline(in, line);
			ss << line;
			ss >> r;
			ss >> c;
			m.resize(r, c);
		}

		{// next R lines: content (C elements in each line)
			const ulong rows = m.rows();
			const ulong cols = m.cols();

			for (ulong i = 0; i < rows; ++i) {
				string line;
				stringstream ss;

				getline(in, line);
				ss << line;
				for (ulong j = 0; j < cols; ++j)
					ss >> m(i,j);
			}
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


#undef endl


#endif//___MSC__MATRIX_HPP___
