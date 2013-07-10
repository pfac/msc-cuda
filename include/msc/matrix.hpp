#ifndef ___MSC__MATRIX_HPP___
#define ___MSC__MATRIX_HPP___

// std C++ headers
#include <fstream>
#include <sstream>


// names
using std::istream;
using std::ifstream;
using std::ostream;
using std::stringstream;


// macros
#define endl '\n'


template<typename T>
class matrix {
	ulong r;
	ulong c;
	T * content;

	ulong linidx (const ulong i, const ulong j) const { return i * c + j; }
public:
	matrix (const string& filename) {
		ifstream file;
		file.open(filename.c_str());
		file >> *this;
	}


	//
	// getters
	//
	ulong rows() const { return r; }
	ulong cols() const { return c; }


	//
	// setters
	//
	void rows(const ulong r) { this->r = r; }
	void cols(const ulong c) { this->c = c; }


	//
	// utilities
	//
	void resize (const ulong r, const ulong c) {
		if (r == rows() && c == cols())
			return;

		T * const content = new T[r * c];
		if (content) {
			memcpy(content, this->content, r * c * sizeof(T));
			delete[] this->content;
		}

		rows(r);
		cols(c);
		this->content = content;
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
		for (ulong i = 0; i < m.r; ++i) {
			for (ulong j = 0; j < m.c; ++j) {
				out.put(' ');
				out << m(i,j);
			}
			out.put(endl);
		}

		return out;
	}
};


#endif//___MSC__MATRIX_HPP___
