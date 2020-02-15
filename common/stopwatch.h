#pragma once

#include <chrono>


class Stopwatch {
public:
	typedef std::chrono::duration<float> t_sec;
	Stopwatch() {
		reset();
	}
	void reset() {
		m_last_timepoint = m_clock.now();
	}
	double elapsed() {
		return (t_sec(m_clock.now() - m_last_timepoint)).count();
	}
protected:
	std::chrono::high_resolution_clock				m_clock;
	std::chrono::high_resolution_clock::time_point	m_last_timepoint;
};
