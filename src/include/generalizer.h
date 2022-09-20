#pragma once

#include <composer.h>
#include <schedplan.h>

#include <vector>


class Generalizer {

 public:

    /**
     * @brief End2end search procedure
     * 
     */
    static GeneralSchedPlan searchDFS(
        Plans micros, const std::vector<float>& memory,
        int nworkers = 1, const long budget = -1
    );

    /**
     * @brief End2end search procedure
     * 
     */
    static GeneralSchedPlan searchBFS(
        Plans micros, const std::vector<float>& memory,
        int nworkers = 1, const long budget = -1
    );

    static GeneralSchedPlan tailHeadHeuristic(
        const SchedPlan& sched,
        const std::vector<float>& memory,
        const int steady_opt_step_upbound = -1,
        const int nworkers = 1, const int budget = -1
    );

    static GeneralSchedPlan tightenHeuristic(
        const SchedPlan& sched,
        const std::vector<float>& memory,
        const int steady_opt_step_upbound = -1,
        const int nworkers = 1
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