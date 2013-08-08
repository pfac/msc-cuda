#ifndef ___TK__POSIX__STOPWATCH_HPP___
#define ___TK__POSIX__STOPWATCH_HPP___

// std C++ headers
#include <iostream>

// std C headers
#include <ctime>

namespace cuk {

	struct Time : public timespec {
		Time () { zero(); }

		Time (const Time * const t) {
			this->tv_sec = t->tv_sec;
			this->tv_nsec = t->tv_nsec;
		}

		Time zero () {
			tv_sec = tv_nsec = 0;
			return *this;
		}

		double nsec () const { return tv_sec * 1e9 + tv_nsec; }

		const Time& operator-= (const Time& t) {
			if (t.tv_sec > tv_sec || (t.tv_sec == tv_sec && t.tv_nsec > tv_nsec))// set this as zero
				zero();
			else {
				tv_sec -= t.tv_sec;
				tv_nsec -= t.tv_nsec;
				if (tv_nsec < 0) {// take an extra second
					--tv_sec;
					tv_nsec += 1000000000;
				}
			}
			return *this;
		}

		Time operator- (const Time& t) const {
			Time result(this);
			result -= t;
			return result;
		}
	};

	/**
	 * This stopwatch does not measure overhead because it is not constant thorought the execution of an application. If the stopwatch start/stop overhead is an important issue, the developer is encouraged to start/stop the stopwatch a considerable amount of times in the code zone where it will be used. The minimum value obtained this way may be considered a "safe" value to represent the overhead.
	 */
	class stopwatch {
#if defined CLOCK_MONOTONIC_HR
		static const clockid_t clk_id = CLOCK_MONOTONIC_HR;
#elif defined CLOCK_MONOTONIC_RAW
		static const clockid_t clk_id = CLOCK_MONOTONIC_RAW;
#else
		static const clockid_t clk_id = CLOCK_MONOTONIC;
#endif
		bool _running;
		Time _begin;
		Time _end;
	public:
		/** Sets the time reference.
		 * This function starts the stopwatch in the sense that it records the current time stamp as reference for calculating elapsed times.
		 * Also changes the final time stamp in order to avoid negative intervals.
		 * Subsequent calls to this function effectively restart the stopwatch.
		 */
		inline
		void start () {
			_running = true;
			clock_gettime(clk_id, &_begin);
		}

		/** Sets the final time reference.
		 * Does not preform any calculation, elapsed times are only computed on a subsequent read.
		 * Consecutive calls to this function act as "lap", allowing to retrieve partial times.
		 */
		inline
		void stop () {
			clock_gettime(clk_id, &_end);
			_running = false;
		}

		/** Retrieves the elapsed time in nanoseconds.
		 * Makes use of the references created by start/stop.
		 */
		inline
		double ns () const {
			if (_running)
				return 0;
			Time elapsed = _end - _begin;
			return elapsed.nsec();
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

#endif//___TK__POSIX__STOPWATCH_HPP___
