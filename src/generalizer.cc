
#include <generalizer.h>
#include <schedplan.h>


GeneralSchedPlan Generalizer::tailHeadHeuristic(
  const SchedPlan& sched,
  const std::vector<float>& memory,
  const int steady_opt_step_upbound,
  const int nworkers) {

    std::set<int> mids;
    for (auto blk : sched.allBlocks()) {
        mids.insert(blk->mid);
    }
    SchedPlan lsched(sched);
    lsched = Generalizer::loosen_all(lsched, memory, true);
    SchedPlan rsched = lsched.increaseMid(mids.size());
    // std::cout << "lsched:\n" << lsched << std::endl;
    // split to head-tail
    SchedPlan lhead = lsched.selectSteps(0, sched.nSteps() / 2);
    lhead.squeeze();
    SchedPlan ltail = lsched.selectSteps(sched.nSteps() / 2, sched.nSteps());
    ltail.squeeze();
    SchedPlan rhead = rsched.selectSteps(0, sched.nSteps() / 2);
    rhead.squeeze();
    SchedPlan rtail = rsched.selectSteps(sched.nSteps() / 2, sched.nSteps());
    rtail.squeeze();
    // step optimal compose
    Plans tail_heads = {ltail, rhead};
    // std::cout << tail_heads[0] << std::endl << tail_heads[1] << std::endl;
    // setup memory
    std::vector<float> steady_memory(sched.nDevs());
    for (int devid = 0; devid < sched.nDevs(); ++devid) {
        float curr_mem = lhead.currMemory(devid);
        steady_memory[devid] = memory[devid] - curr_mem;
    }
    Plans steadies = Composer::stepOptimalDFS(
        tail_heads, steady_memory, true,
        steady_opt_step_upbound, nworkers
    );

    GeneralSchedPlan gplan;
    if (steadies.size() > 0) {
        SchedPlan steady = steadies[0];
        gplan = GeneralSchedPlan(lhead, steady, rtail);
    }

    for (auto blk : rsched.allBlocks()) {
        gplan.addCreatedBlocks(blk);
    }
    return gplan;
}


GeneralSchedPlan Generalizer::tightenHeuristic(
  const SchedPlan& sched,
  const std::vector<float>& memory,
  const int steady_opt_step_upbound,
  const int nworkers) {

    std::set<int> mids;
    for (auto blk : sched.allBlocks()) {
        mids.insert(blk->mid);
    }
    SchedPlan lsched(sched);
    lsched = Generalizer::loosen_all(lsched, memory, true);
    SchedPlan rsched = lsched.increaseMid(mids.size());
    // std::cout << "lsched:\n" << lsched << std::endl;
    // split to head-tail
    SchedPlan lhead = lsched.selectSteps(0, sched.nSteps() / 2);
    lhead.squeeze();
    SchedPlan ltail = lsched.selectSteps(sched.nSteps() / 2, sched.nSteps());
    ltail.squeeze();
    SchedPlan rhead = rsched.selectSteps(0, sched.nSteps() / 2);
    rhead.squeeze();
    SchedPlan rtail = rsched.selectSteps(sched.nSteps() / 2, sched.nSteps());
    rtail.squeeze();
    // step optimal compose
    Plans tail_heads = {ltail, rhead};

    std::vector<float> steady_memory(sched.nDevs());
    for (int devid = 0; devid < sched.nDevs(); ++devid) {
        float curr_mem = lhead.currMemory(devid);
        steady_memory[devid] = memory[devid] - curr_mem;
    }
    SchedPlan steady = SchedPlan::concat(tail_heads);
    steady = Composer::tighten_all(steady, steady_memory);
    
    GeneralSchedPlan gplan(lhead, steady, rtail);
    for (auto blk : rsched.allBlocks()) {
        gplan.addCreatedBlocks(blk);
    }
    return gplan;
}


SchedPlan& Generalizer::loosen_all(SchedPlan& sched, const std::vector<float>& memory, bool only_forward) {
    for (int step = sched.nSteps() - 1; step >= 0; --step) {
        for (auto blk : sched.stepBlocks(step)) {
            if (!only_forward or (blk->btype == BlockType::Forward)) {
                sched = Generalizer::loosen(sched, blk, memory);
            }
        }
    }
    return sched;
}


SchedPlan& Generalizer::loosen(SchedPlan& sched, Block* blk, const std::vector<float>& memory, const int bound) {
    auto devids = sched.getDevice(blk);
    int step = sched.getStep(blk);
    int maxstep = (bound == -1) ? sched.nSteps() - 1 : bound;
    for (auto ablk : blk -> after) {
        if (sched.haveBlock(ablk)) {
            maxstep = std::min(maxstep, sched.getStep(ablk) - 1);
        }
    }
    if (maxstep <= step) return sched;

    int free_steps = 0;
    for (int t = maxstep; t > step; --t) {
        // check step slots
        bool have_block = false;
        for (int devid : devids) {
            if (sched.getBlock(devid, t) != nullptr) {
                have_block = true;
                break;
            }
        }
        if (have_block) {
            free_steps = 0;
            continue;
        }
        else {
            free_steps += 1;
            if (free_steps < blk->span) {
                continue;
            }
        }

        // check memory
        bool exceed_memory = false;
        sched.setPosition(blk, devids, t);
        for (int devid : devids) {
            if (sched.peakMemory(devid) > memory[devid]) {
                exceed_memory = true;
                break;
            }
        }
        if (exceed_memory) {
            sched.setPosition(blk, devids, step);
            free_steps -= 1;
            continue;
        }
        // sucess
        break;
    }
    return sched;
}
