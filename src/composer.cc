
#include <iostream>
#include <utility>
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cstdio>
#include <sstream>
#include <unordered_set>

#include <omp.h>

#include <composer.h>


template<typename T>
void _debug_print(const T& blocks) {
    for (Block* blk : blocks) {
        std::cout << *blk << " ";
    }
}

std::string encode(const Plans& micros, int step) {
    const int ndevs = micros.at(0).nDevs();
    std::vector<int> prob;
    prob.reserve(micros.size() + ndevs);
    for (std::size_t idx = 0; idx < micros.size(); ++idx) {
        prob.push_back(std::max(micros[idx].nSteps() - step, 0));
    }
    for (int devid = 0; devid < ndevs; ++devid) {
        prob.push_back(Composer::currMemory(micros, devid, 0, step));
    }
    std::stringstream ss;
    for (auto it = prob.begin(); it != prob.end(); ++it) {
        ss << *it;
    }
    return ss.str();
}


Block2Hash::Block2Hash(const std::vector<SchedPlan>& plans) {

    std::vector<Plans> same_groups;

    for (auto& plan : plans) {
        bool inserted = false;
        for (auto& group : same_groups) {
            if (Block2Hash::samePlan(plan, group[0])) {
                group.push_back(plan);
                inserted = true;
                break;
            }
        }
        if (!inserted) {
            same_groups.push_back(Plans({plan}));
        }
    }
    // std::cout << "Got " << same_groups.size() << " groups" << std::endl;
    for (auto& group : same_groups) {
        int ndevs = group[0].nDevs();
        int nsteps = group[0].nSteps();
        for (int step = 0; step < nsteps; ++step) {
            for (int devid = 0; devid < ndevs; ++devid) {
                for (auto& plan : group) {
                    Block* blk = plan.getBlock(devid, step);
                    if (blk != nullptr) {
                        this->_block2gid.emplace(blk, this->_uid);
                    }
                }
                this->_uid += 1;
            }
        }
    }
}


bool Block2Hash::samePlan(const SchedPlan& sched1, const SchedPlan sched2) {
    if (sched1.nDevs() != sched2.nDevs() or 
        sched1.nSteps() != sched2.nSteps() or
        sched1.allBlocks().size() != sched2.allBlocks().size()) {
        return false;
    }
    int nsteps = std::max(sched1.nSteps(), sched2.nSteps());
    int ndevs = std::max(sched1.nDevs(), sched2.nDevs());

    Block* blk1 = nullptr;
    Block* blk2 = nullptr;
    for (int step = 0; step < nsteps; ++step) {
        for (int devid = 0; devid < ndevs; ++devid) {
            blk1 = sched1.getBlock(step, devid);
            blk2 = sched2.getBlock(step, devid);
            if ((blk1 == nullptr and blk2 != nullptr) or
                (blk1 != nullptr and blk2 == nullptr)) {
                return false;
            }
        }
    }
    return true;
}


// ******************* Composer *******************

Plans Composer::stepOptimalBFS(std::vector<SchedPlan> micros, const std::vector<float>& memory,
                            bool silence, int rstep, int nworkers) {

    const int ndevs = micros.at(0).nDevs();
    
    // construct block mapping to micro index
    Block2Idx blk2idx;
    for (std::size_t i = 0; i < micros.size(); ++i) {
        for (auto blk : micros[i].allBlocks()) {
            blk2idx.emplace(blk, int(i));
        }
    }

    std::size_t total_status = 1;
    Block2Hash blk2hash(micros);

    std::vector<int> minsteps(ndevs, 0);
    for (auto& micro : micros) {
        for (int devid = 0; devid < ndevs; ++devid) {
            for (auto blk : micro.devBlocks(devid, 0)) {
                minsteps[devid] += blk->span;
            }
        }
    }

    // find a plan using heuristic
    SchedPlan concat = SchedPlan::concat(micros);
    concat = Composer::tighten_all(concat, memory);

    int opt_step_lbound = *(std::max_element(minsteps.begin(), minsteps.end()));
    int opt_step_rbound = rstep < 0 ? concat.nSteps() : std::min(concat.nSteps(), rstep);
    if (!silence) std::cout << "setting up bound of opt step to " << opt_step_rbound << std::endl;

    // ============== BFS search ==============
    
    // init curr status and next status
    std::vector<Plans> curr({micros});
    std::vector<Plans> next;
    std::vector<SchedPlan> schedules;

    if (concat.nSteps() <= opt_step_rbound) {
        schedules.push_back(concat);
    }

    // BFS steps
    int step = 0;
    // explain why <= instead of <: to avoid exit at begining due to arguement parse
    while (step < opt_step_rbound) {
        if (!silence) std::cout << "solving step " << step << ", candidates: " << curr.size() << std::endl;
        const int ncandidates = curr.size();

        std::vector<std::size_t> num_next(nworkers, 0);
        std::vector<int> rsteps(nworkers, opt_step_rbound);
        #pragma omp parallel num_threads(nworkers)
        {
            int tid = omp_get_thread_num();
            int start = curr.size() / nworkers * tid;
            int stop = (tid == nworkers - 1) ? ncandidates : start + ncandidates / nworkers;
            int rstep = opt_step_rbound;
            std::vector<Plans> local_next;
            std::vector<SchedPlan> local_schedules;
            for (int idx = start; idx < stop; idx++) {
                auto candidates_and_schedules = Composer::resolveStep(
                    curr[idx], memory, step,
                    blk2hash, blk2idx,
                    rstep, minsteps
                );
                local_next.insert(
                    local_next.end(),
                    candidates_and_schedules.first.begin(),
                    candidates_and_schedules.first.end()
                );

                // very important pruning techniques
                for (auto& ms : candidates_and_schedules.first) {
                    int opt_step_bound = step + 1;
                    for (auto& micro : ms) {
                        opt_step_bound += std::max(micro.nSteps() - step - 1, 0);
                    }
                    if (opt_step_bound < rstep) {
                        local_schedules.clear();
                        rstep = opt_step_bound;
                    }
                }

                for (auto& sched : candidates_and_schedules.second) {
                    if (sched.nSteps() < rstep) {
                        rstep = sched.nSteps();
                        local_schedules.clear();
                    }
                    if (sched.nSteps() == rstep) {
                        local_schedules.push_back(sched);
                    }
                }
            }
            num_next[tid] = local_next.size();
            rsteps[tid] = rstep;

            #pragma omp barrier

            #pragma omp single 
            {
                std::size_t total_next = 0;
                for (int num : num_next) {
                    total_next += num;
                }
                next.resize(total_next);
                for (int local_opt : rsteps) {
                    if (local_opt < opt_step_rbound) {
                        if (!silence) { printf("find fewer steps: %d\n", local_opt); }
                        opt_step_rbound = local_opt;
                        schedules.clear();
                    }
                }
            }

            #pragma omp barrier

            // insert next
            std::size_t local_start = 0;
            for (int idx = 0; idx < tid; ++idx) {
                local_start += num_next[idx];
            }
            std::copy(local_next.begin(), local_next.end(), next.begin() + local_start);

            #pragma omp critical
            {
                if (rstep == opt_step_rbound) {
                    schedules.insert(
                        schedules.end(), local_schedules.begin(), local_schedules.end()
                    );
                }
            }
        }

        total_status += next.size();
        curr.clear();
        std::swap(curr, next);
        ++step;
        if (opt_step_lbound == opt_step_rbound and schedules.size() > 0) {
            break;
        }
    }

    if (!silence) {
        std::cout << "search done on " << total_status << " cases. "
                  << "find " << schedules.size() << " tight step-optimal plans "
                  << "(step = " << opt_step_rbound << ")" << std::endl;
    }
    return schedules;
}


Plans Composer::stepOptimalDFS(Plans micros, const std::vector<float>& memory,
                               bool silence, int rstep, int nworkers) {
    const int ndevs = micros.at(0).nDevs();
    // remain_step_hash : current step
    std::unordered_map<std::string, int> explored;
    
    // construct block mapping to micro index
    Block2Idx blk2idx;
    for (std::size_t i = 0; i < micros.size(); ++i) {
        for (auto blk : micros[i].allBlocks()) {
            blk2idx.emplace(blk, int(i));
        }
    }

    std::size_t total_status = 1;
    Block2Hash blk2hash(micros);

    using Item = std::pair<int, std::vector<Plans>>;
    std::vector<Item> stack;
    Plans schedules;

    std::vector<int> minsteps(ndevs, 0);
    for (auto& micro : micros) {
        for (int devid = 0; devid < ndevs; ++devid) {
            minsteps[devid] += micro.devBlocks(devid, 0).size();
        }
    }

    // find a plan using heuristic
    SchedPlan concat = SchedPlan::concat(micros);
    concat = Composer::tighten_all(concat, memory);

    int opt_step = (rstep < 0) ? concat.nSteps() : std::min(concat.nSteps(), rstep);
    if (concat.nSteps() == opt_step) schedules.push_back(concat);
    int opt_step_lbound = *(std::max_element(minsteps.begin(), minsteps.end()));

    stack.emplace_back(0, std::vector<Plans>(1, micros));
    while (!stack.empty()) {
        Item& item = stack.back();
        int step = item.first;
        Plans micros = item.second.back();

        item.second.pop_back();
        if (item.second.empty()) {
            stack.pop_back();
        }

        // encode the rest problem as a searched problem
        // std::string prob = encode(micros, step);
        // if (explored.find(prob) != explored.end() and step < explored[prob]) {
        //     continue;
        // }
        // explored.insert_or_assign(prob, step);

        auto plans_scheds = Composer::resolveStep(
            micros, memory, step, blk2hash, blk2idx, opt_step, minsteps);

        if (plans_scheds.first.size() > 0) {
            stack.emplace_back(step+1, plans_scheds.first);
        }

        for (auto& sched : plans_scheds.second) {
            if (sched.nSteps() < opt_step) {
                if (!silence) std::cout << "\r ==========> find fewer steps " << sched.nSteps() << std::endl;
                opt_step = sched.nSteps();
                schedules.clear();
            }
            if (sched.nSteps() == opt_step) {
                schedules.push_back(sched);
            }
        }
        total_status += 1;
        if (total_status % 1000 == 0) {
            if (!silence) std::cout << "\rsearched " << total_status << "/? cases" << std::flush;
        }

        if (schedules.size() > 0 and opt_step == opt_step_lbound) {
            break;
        }
    }

    if (!silence) {
        std::cout << "\nsearch done on " << total_status << " cases. "
                  << "find " << schedules.size() << " tight step-optimal plans "
                  << "(step = " << opt_step << ")" << std::endl;
    }
    return schedules;
}


Plans Composer::stepOptimalBDFS(Plans micros, const std::vector<float>& memory,
                                bool silence, int opt_step_upbound, int nworkers) {
    const int ndevs = micros.at(0).nDevs();
    // upbound to handle
    const std::size_t breadth = nworkers * 128;
    // remain_step_hash : current step
    std::unordered_map<std::string, int> explored;
    
    // construct block mapping to micro index
    Block2Idx blk2idx;
    for (std::size_t i = 0; i < micros.size(); ++i) {
        for (auto blk : micros[i].allBlocks()) {
            blk2idx.emplace(blk, int(i));
        }
    }

    std::size_t total_status = 1;
    Block2Hash blk2hash(micros);

    using Item = std::pair<int, std::vector<Plans>>;
    std::vector<Item> stack;
    Plans schedules;

    std::vector<int> minsteps(ndevs, 0);
    for (auto& micro : micros) {
        for (int devid = 0; devid < ndevs; ++devid) {
            minsteps[devid] += micro.devBlocks(devid, 0).size();
        }
    }
    int opt_step = (opt_step_upbound > 0) ? opt_step_upbound : std::accumulate(minsteps.begin(), minsteps.end(), 0);
    int opt_step_lbound = *(std::max_element(minsteps.begin(), minsteps.end()));

    stack.emplace_back(0, std::vector<Plans>(1, micros));
    while (!stack.empty() and opt_step > opt_step_lbound) {
        Item& item = stack.back();
        int step = item.first;
        int nplans = std::min(item.second.size(), breadth);

        std::vector<Plans> plans;
        plans.insert(plans.end(), item.second.end()-nplans, item.second.end());
        
        // pop out the stack
        item.second.erase(item.second.end()-nplans, item.second.end());
        if (item.second.empty()) {
            stack.pop_back();
        }

        std::vector<Plans> candidates;

        #pragma omp parallel num_threads(nworkers)
        {
            int tid = omp_get_thread_num();
            int start = plans.size() / nworkers * tid;
            int stop = (tid == nworkers - 1) ? int(plans.size()) : start + plans.size() / nworkers;

            // encode the problem
            std::vector<std::string> probs(stop-start);
            for (int idx = 0; idx < stop-start; ++idx) {
                Plans& micros = plans.at(idx);
                probs[idx] = encode(micros, step);
            }

            // prune the same problem
            std::vector<bool> prunes(stop-start, false);
            // #pragma omp critical
            // {
            //     for (int idx = 0; idx < stop-start; ++idx) {
            //         if (explored.find(probs[idx]) != explored.end() and step < explored[probs[idx]]) {
            //             prunes[idx] = true;
            //         }
            //         else {
            //             explored.insert_or_assign(probs[idx], step);
            //         }
            //     }
            // }

            for (int idx = start; idx < stop; ++idx) {
                
                if (prunes[idx-start]) continue;
 
                Plans& micros = plans.at(idx);
                auto plans_scheds = Composer::resolveStep(
                    micros, memory, step, blk2hash, blk2idx,
                    opt_step, minsteps
                );

                #pragma omp critical
                {
                    if (plans_scheds.first.size() > 0) {
                        candidates.insert(
                            candidates.end(),
                            plans_scheds.first.begin(), plans_scheds.first.end()
                        );
                    }

                    for (auto& sched : plans_scheds.second) {
                        if (sched.nSteps() < opt_step) {
                            if (!silence) std::cout << "\r ==========> find fewer steps " << sched.nSteps() << std::endl;
                            opt_step = sched.nSteps();
                            schedules.clear();
                        }
                        if (sched.nSteps() == opt_step) {
                            schedules.push_back(sched);
                        }
                    }

                    total_status += 1;
                    if (total_status % 1000 == 0) {
                        if (!silence) std::cout << "\rsearched " << total_status << "/? cases" << std::flush;
                    }
                }
            }
        }

        if (candidates.size() > 0) {
            stack.emplace_back(step+1, candidates);
        }
    }

    if (!silence) {
        std::cout << "\nsearch done on " << total_status << " cases. "
                  << "find " << schedules.size() << " tight step-optimal plans "
                  << "(step = " << opt_step << ")" << std::endl;
    }
    return schedules;
}


std::pair<std::vector<Plans>, std::vector<SchedPlan>>
Composer::resolveStep(const Plans& micros, const std::vector<float>& memory,
                      int step, const Block2Hash& blk2hash, Block2Idx& blk2idx,
                      int rstep, const std::vector<int>& minsteps) {

    int ndevs = micros[0].nDevs();

    std::vector<Plans> next;
    std::vector<SchedPlan> schedules;

    auto conflicts = Composer::getConflict(micros, step, memory, blk2hash);
    Conflict can_keep = conflicts.first;
    Conflict to_shift = conflicts.second;
    // std::cout << "keepable conflict: " << can_keep << " | must move conflict: " << to_shift << std::endl;
    std::vector< std::set<Block*> > all_shifts = Composer::getShiftSpace(
        ndevs, micros, can_keep, to_shift, blk2idx
    );

    // std::cout << "prev:\n"; for (auto& micro : micros) std::cout << micro << std::endl;
    // std::cout << "memory constraints: "; for (float mem : memory) std::cout << mem << " "; std::cout << std::endl;
    // for (auto shifts : all_shifts) {std::cout << "shift(" << all_shifts.size() << "): "; for (auto blk : shifts) printf("%s(%d) ", blk->toStr().c_str(), blk->mid); std::cout << std::endl; }
    // { if (true) { int _x; std::cin >> _x; } }

    for (auto& shifts : all_shifts) {

        // copy for inplacement update
        Plans cmicros(micros);
        for (auto blk : shifts) {
            int idx = blk2idx[blk];
            cmicros[idx].shift(blk);
        }

        // dynamic pruning
        if (Composer::isDynSymm(cmicros, step)) {
            continue;
        }

        // std::cout << "prev:\n"; for (auto& micro : cmicros) std::cout << micro << std::endl;
        if (SchedPlan::stackable(cmicros, memory)) {
            SchedPlan sched = SchedPlan::stack(cmicros);
            rstep = std::min(sched.nSteps(), rstep);
            if (sched.nSteps() <= rstep) {
                schedules.push_back(sched);
            }
        }
        else {
            // pruning technique: discard plans that exceed rstep
            bool discard = false;
            // for (auto& micro : cmicros) {
            //     if (micro.nSteps() > rstep) {
            //         discard = true;
            //         break;
            //     }
            // }
            for (int devid = 0; devid < ndevs; ++devid) {
                int nbubbles = Composer::nBubbles(cmicros, devid, 0, step+1);
                if (rstep - minsteps[devid] < nbubbles) {
                    discard = true;
                    break;
                }
            }
            if (discard) continue;
            next.push_back(cmicros);
        }
    }
    // std::cout << "=====<" << std::endl;
    return std::make_pair(next, schedules);
}


std::vector<std::set<Block*>>
Composer::getShiftSpace(const int ndevice, const Plans& micros,
                const Conflict& can_keep, const Conflict& to_shift,
                const Block2Idx& blk2idx) {

    std::vector<std::set<Block*>> all_keeps;
    using Item = std::pair<std::set<Block*>, std::vector<bool>>;
    std::vector<Item> curr;
    std::vector<Item> next;
    curr.emplace_back(std::set<Block*>(), std::vector<bool>(ndevice, false));

    while (!curr.empty()) {
        for (auto& item : curr) {
            bool full = true;
            for (Block* blk : can_keep.allBlocks()) {
                bool keepable = true;
                for (int devid : can_keep.getDevice(blk)) {
                    if (item.second[devid]) {
                        keepable = false;
                        break;
                    }
                }
                if (keepable) {
                    std::set<Block*> blks(item.first);
                    std::vector<bool> devs(item.second);
                    blks.insert(blk);
                    for (int devid : can_keep.getDevice(blk)) {
                        devs[devid] = true;
                    }
                    next.emplace_back(blks, devs);
                    full = false;
                }
            }
            if (not full) continue;
            // add to keep candidates: check redundancy
            bool exist = false;
            for (auto& keep : all_keeps) {
                if (keep == item.first) {
                    exist = true;
                    break;
                }
            }
            if (exist) continue;
            all_keeps.push_back(item.first);
        }
        curr.clear();
        std::swap(curr, next);
    }

    // std::cout << "all keeps:" << std::endl; for (auto keep : all_keeps) { _debug_print<std::set<Block*>>(keep); std::cout << std::endl;}
    std::vector<std::set<Block*>> all_shifts;
    std::vector<Block*> must_shift_blks = to_shift.allBlocks();
    for (auto& keep : all_keeps) {
        std::set<Block*> shifts(must_shift_blks.begin(), must_shift_blks.end());
        for (Block* blk : can_keep.allBlocks()) {
            if (keep.find(blk) == keep.end()) {
                shifts.insert(blk);
            }
        }
        all_shifts.push_back(shifts);
    }
    return all_shifts;
}


std::pair<Conflict, Conflict>
Composer::getConflict(const Plans& micros, int step,
                      const std::vector<float>& memory,
                      const Block2Hash& blk2hash) {
    const int ndevs = micros.at(0).nDevs();
    Conflict can_keep(ndevs, step);
    Conflict to_shift(ndevs, step);

    // ===============  must shift: memory conflict ================
    std::vector<float> devmem(ndevs, 0.0);
    for (int devid = 0; devid < ndevs; ++devid) {
        for (auto& micro : micros) {
            devmem[devid] += micro.currMemory(devid, 0, step);
        }
    }

    for (std::size_t mid = 0; mid < micros.size(); ++mid) {
        for (auto blk : micros[mid].stepBlocks(step)) {
            if (!micros[mid].isTheStart(blk, step)) {
                continue;
            }
            std::vector<int> devs = micros[mid].getDevice(blk);
            bool need_shift = false;
            for (int dev : devs) {
                // pre-memory
                float min_peak_mem = devmem[dev] + blk->memory;
                if (min_peak_mem > memory[dev]) {
                    need_shift = true;
                    break;
                }
                // post-memory. TODO: prove this to be the real lower bound
                std::vector< std::pair<std::size_t, float> > remain_peak_mem;
                for (size_t rmid = 0; rmid < micros.size(); ++rmid) {
                    int start_step = (rmid == mid) ? step + 1 : step;
                    remain_peak_mem.emplace_back(
                        rmid, micros[rmid].peakMemory(dev, start_step)
                    );
                }
                std::sort(
                    remain_peak_mem.begin(), remain_peak_mem.end(),
                    [&](const std::pair<int,float>& a, const std::pair<int,float>& b) { return a.second < b.second; }
                );
                float curr_mem = 0.0;
                float peak_mem = -std::numeric_limits<float>::max();
                for (auto& mid_pmem : remain_peak_mem) {
                    std::size_t sort_mid = mid_pmem.first;
                    peak_mem = std::max(peak_mem, curr_mem + mid_pmem.second);
                    int start_step = (sort_mid == mid) ? step + 1 : step;
                    curr_mem += micros[sort_mid].currMemory(dev, start_step);
                }
                min_peak_mem = devmem[dev] + blk->memory + peak_mem;
                if (min_peak_mem > memory[dev]) {
                    need_shift = true;
                    break;
                }
            }

            if (need_shift) {
                for (auto devid : devs) {
                    to_shift.addBlock(blk, devid);
                }
            }
        }
    }

    // ===============  can keep: step conflict ================
    for (int devid = 0; devid < ndevs; ++devid) {
        std::unordered_map<int, Block*> keep_uids;
        std::unordered_map<Block*, std::vector<int>> blk_devs;

        int blk_cnt = 0;
        for (auto& micro : micros) {
            Block* blk = micro.getBlock(devid, step);
            if (blk != nullptr) {
                if (!to_shift.haveBlock(blk)) {
                    blk_cnt += 1;
                }
            }
        }
        if (blk_cnt <= 1) continue;
        // if the block at this step starts before, all other blocks
        // need to be shifted.
        bool to_shift_start = false;
        for (auto& micro : micros) {
            auto blk = micro.getBlock(devid, step);
            if (blk == nullptr) continue;
            if (!micro.isTheStart(blk, step)) {
                to_shift_start = true;
                break;
            }
        }
        for (auto& micro : micros) {
            auto blk = micro.getBlock(devid, step);
            if (blk == nullptr) continue;
            if (to_shift.haveBlock(blk) or !micro.isTheStart(blk, step)) continue;
            blk_devs.emplace(blk, micro.getDevice(blk));
            if (to_shift_start) {
                to_shift.addBlock(blk, blk_devs[blk]);
            }
            // static symmetry
            else {
                int uid = blk2hash.getUid(blk);
                if (keep_uids.find(uid) != keep_uids.end()) {
                    Block* kept = keep_uids[uid];
                    if (blk->mid < kept->mid) {
                        to_shift.addBlock(kept, blk_devs[kept]);
                        keep_uids[uid] = blk;
                    }
                    else {
                        to_shift.addBlock(blk, blk_devs[blk]);
                    }
                }
                else {
                    keep_uids.emplace(uid, blk);
                }
            }
        }
        for (auto& it : keep_uids) {
            can_keep.addBlock(it.second, blk_devs[it.second]);
        }
    }
    return std::make_pair(can_keep, to_shift);
}


bool Composer::isDynSymm(const Plans& micros, int step) {
    const int ndevs = micros.at(0).nDevs();
    // TODO: change implementation to block granularity.
    if (step == 0) return false;
    std::set<Block*> curr;
    std::vector<int> curr_mem(ndevs);
    for (auto& micro : micros) {
        for (auto blk : micro.stepBlocks(step)) {
            // if (!micro.isTheStart(blk, step)) return false;
            if (blk->span > 1) return false;
            curr.insert(blk);
            for (int devid : micro.getDevice(blk)) {
                curr_mem[devid] = blk->memory;
            }
        }
    }
    std::set<Block*> prev;
    std::vector<int> prev_mem(ndevs);
    for (auto& micro : micros) {
        for (auto blk : micro.stepBlocks(step-1)) {
            if (!micro.isTheStart(blk, step-1)) return false;
            prev.insert(blk);
            for (int devid : micro.getDevice(blk)) {
                prev_mem[devid] = blk->memory;
            }
        }
    }
    // check before
    for (Block* blk : curr) {
        for (Block* bblk : blk->before) {
            if (prev.find(bblk) != prev.end()) {
                return false;
            }
        }
    }
    // check after
    for (Block* blk : prev) {
        for (Block* ablk : blk->after) {
            if (curr.find(ablk) != curr.end()) {
                return false;
            }
        }
    }
    int prev_mid = std::numeric_limits<int>::max();
    int curr_mid = std::numeric_limits<int>::max();
    for (auto blk : prev) {
        prev_mid = std::min(prev_mid, blk->mid);
    }
    for (auto blk : curr) {
        curr_mid = std::min(curr_mid, blk->mid);
    }
    if (prev_mid > curr_mid) {
        for (int devid = 0; devid < ndevs; ++devid) {
            if (prev_mem[devid] < 0 and curr_mem[devid] > 0) {
                return false;
            }
        }
        return true;
    }
    return false;
}


float Composer::currMemory(const Plans& micros, int devid, int from_step, int to_step) {
    float curr_mem = 0.0;
    Block* blk = nullptr;
    int t = from_step;
    while (t < to_step) {
        for (auto& micro : micros) {
            blk = micro.getBlock(devid, t);
            if (blk == nullptr) {
                t += 1;
            }
            else {
                curr_mem += blk->memory;
                t += blk->span;
            }
        }
    }
    return curr_mem;
}


int Composer::nBubbles(const Plans& micros, int devid, int from_step, int to_step) {
    int nbubbles = 0;
    Block* blk = nullptr;
    int t = from_step;
    while (t < to_step) {
        bool is_bubble = true;
        for (auto& micro : micros) {
            blk = micro.getBlock(devid, t);
            if (blk != nullptr) {
                is_bubble = false;
                break;
            }
        }
        if (is_bubble) {
            nbubbles += 1;
            t += 1;
        }
        else {
            t += blk->span;
        }
    }
    return nbubbles;
}


SchedPlan& Composer::tighten_all(SchedPlan& sched, const std::vector<float>& memory) {
    for (int step = 0; step < sched.nSteps(); ++step) {
        for (auto blk : sched.stepBlocks(step)) {
            if (!sched.isTheStart(blk, step)) continue;
            sched = Composer::tighten(sched, blk, memory);
        }
    }
    return sched;
}


SchedPlan& Composer::tighten(SchedPlan& sched, Block* blk, const std::vector<float>& memory) {
    std::vector<int> devids = sched.getDevice(blk);
    int step = sched.getStep(blk);
    int minstep = 0;
    for (auto bblk : blk->before) {
        if (!sched.haveBlock(bblk)) continue;
        minstep = std::max(minstep, sched.getStep(bblk) + bblk->span);
    }
    if (minstep >= step) return sched;

    int free_steps = 0;
    for (int t = minstep; t < step + blk->span - 1; ++t) {
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

        bool exceed_memory = false;
        sched.setPosition(blk, devids, t-blk->span+1);
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
        // success
        break;
    }
    return sched;
}