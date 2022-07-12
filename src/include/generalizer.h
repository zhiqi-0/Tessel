#pragma once

#include <composer.h>
#include <schedplan.h>

#include <vector>


class Generalizer {

 public:

    static GeneralSchedPlan tailHeadHeuristic(
        const SchedPlan& sched,
        const std::vector<float>& memory,
        const int nworkers
    );

    static SchedPlan& loosen_all(
        SchedPlan& sched,
        const std::vector<float>& memory,
        bool only_forward = true
    );

    static SchedPlan& loosen(
        SchedPlan& sched,
        Block* block,
        const std::vector<float>& memory,
        const int bound = -1
    );

};