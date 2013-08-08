#ifndef ___CUK__HW_METER_HPP___
#define ___CUK__HW_METER_HPP___

#include <string>
	using std::string;
#include <vector>
	using std::vector;


#include <papi.h>

#ifdef _OPENMP
	#include <omp.h>
#endif

namespace cuk {



class hw_meter {

	int _set;// the event set used by this meter

	unsigned _nvalues;// number of counters to measure

	long long int * _values;


public:

	static
	unsigned long get_thread_id () {
		#if _OPENMP
			return omp_get_thread_num();
		#else
			return 0;
		#endif
	}



	static
	void init () {
		#ifndef NDEBUG
			assert(PAPI_library_init(PAPI_VER_CURRENT) == PAPI_VER_CURRENT);
			#ifdef _OPENMP
				assert(PAPI_thread_init(get_thread_id) == PAPI_OK);
			#endif
		#else
			PAPI_library_init(PAPI_VER_CURRENT);
			#ifdef _OPENMP
				PAPI_thread_init(hw_meter::get_thread_id);
			#endif
		#endif
	}



	static
	int  create_event_set () {
		int set = PAPI_NULL;

		#ifndef NDEBUG
			assert( PAPI_create_eventset( &set ) == PAPI_OK );
		#else
			PAPI_create_eventset( &set );
		#endif

		return set;
	}



	static
	void cleanup_event_set (const int set) {
		#ifndef NDEBUG
			assert( PAPI_cleanup_eventset( set ) == PAPI_OK );
		#else
			PAPI_cleanup_eventset( set );
		#endif
	}



	static
	int destroy_event_set (int set) {
		cleanup_event_set( set );
		#ifndef NDEBUG
			assert( PAPI_destroy_eventset( &set ) == PAPI_OK );
		#else
			PAPI_destroy_eventset( &set );
		#endif

		return set;
	}



	static
	int event_code (const string& name) {
		init();

		int code;
		char * _name = strdup( name.c_str() );

		#ifndef NDEBUG
			assert( PAPI_event_name_to_code( _name, &code ) == PAPI_OK );
		#else
			PAPI_event_name_to_code( _name, &code );
		#endif

		free( _name );

		return code;
	}



	hw_meter (const vector<int>& events)
	: _nvalues( events.size() )
	{
		// initialize the PAPI library
		hw_meter::init();

		// create the event set
		_set = hw_meter::create_event_set();

		// alocate the space for the values
		_values = new long long int[ _nvalues ];

		// add the events
		vector<int>::const_iterator ci;
		for (ci = events.begin(); ci != events.end(); ++ci) {
			hw_meter::add_event( *ci );
		}
	}



	~hw_meter () {
		// delete the alocated memory
		delete[] _values;

		// delete the event set
		hw_meter::destroy_event_set( _set );
	}



	void add_event (const int event) {
		#ifndef NDEBUG
			assert( PAPI_add_event( _set, event ) == PAPI_OK );
		#else
			PAPI_add_event( _set, event );
		#endif
	}



	void start () {
		// clear the values
		for (unsigned i = 0; i < _nvalues; ++i)
			_values[i] = 0;
		// start measuring
		#ifndef NDEBUG
			assert( PAPI_start( _set ) == PAPI_OK );
		#else
			PAPI_start( _set );
		#endif
	}

	void stop () {
		// stop measuring
		#ifndef NDEBUG
			assert( PAPI_stop( _set, _values ) == PAPI_OK );
		#else
			PAPI_stop( _set, _values );
		#endif
	}

	void values (vector<long long int>& values) {
		values.resize( _nvalues );
		for (unsigned i = 0; i < _nvalues; ++i)
			values[i] = _values[i];
	}
};



}

#endif//___CUK__HW_METER_HPP___