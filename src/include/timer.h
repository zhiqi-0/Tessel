#pragma once

#include <chrono>


class CpuTimer {

 public:
    void start() { t1 = std::chrono::high_resolution_clock::now(); }
    void stop() { t2 = std::chrono::high_resolution_clock::now(); }

    int elapsed() {
        // in millisecond
        return std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    }

 private:
    std::chrono::high_resolution_clock::time_point t1, t2;
};
