#ifndef ___TK__APPLE__STOPWATCH_HPP___
#define ___TK__APPLE__STOPWATCH_HPP___

// Mac OS X headers
#include <mach/mach.h>
#include <mach/mach_time.h>

// std C++ headers
#include <iostream>

namespace cuk {

	/**
	 * This stopwatch does not measure overhead because it is not constant thorought the execution of an application. If the stopwatch start/stop overhead is an important issue, the developer is encouraged to start/stop the stopwatch a considerable amount of times in the code zone where it will be used. The minimum value obtained this way may be considered a "safe" value to represent the overhead.
	 */
	class stopwatch {
		uint64_t _begin;
		uint64_t _end;
		mach_timebase_info_data_t tb;
	public:
		stopwatch() {
			(void)mach_timebase_info(&tb);
		}

		/** Sets the time reference.
		 * This function starts the stopwatch in the sense that it records the current time stamp as reference for calculating elapsed times.
		 * Also changes the final time stamp in order to avoid negative intervals.
		 * Subsequent calls to this function effectively restart the stopwatch.
		 */
		inline
		void start () {
			_end = _begin = mach_absolute_time();
		}

		/** Sets the final time reference.
		 * Does not preform any calculation, elapsed times are only computed on a subsequent read.
		 * Consecutive calls to this function act as "lap", allowing to retrieve partial times.
		 */
		inline
		void stop () {
			_end = mach_absolute_time();
		}

		/** Retrieves the elapsed time in nanoseconds.
		 * Makes use of the references created by start/stop.
		 */
		inline
		double ns () const {
			uint64_t elapsed = _end - _begin;
			return (double)elapsed * (double)tb.numer / (double)tb.denom;
		}

		inline
		double us () const { return ns() * 1e-3; }

		inline
		double ms () const { return ns() * 1e-6; }

		inline
		double s () const { return ns() * 1e-9; }

		friend
		std::ostream& operator<< (std::ostream& out, const stopwatch& sw) {
			double elapsed = sw.ns();
			if (elapsed > 1e9)
				out << sw.s() << "s";
			else if (elapsed > 1e6)
				out << sw.ms() << "ms";
			else if (elapsed > 1e3)
				out << sw.us() << "us";
			else
				out << elapsed << "ns";
			return out;
		}
	};

}

#endif//___TK__APPLE__STOPWATCH_HPP___
