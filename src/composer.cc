
#include <iostream>
#include <utility>
#include <vector>
#include <algorithm>
#include <limits>
#include <cstdio>

#include <omp.h>

#include <composer.h>

template<typename T>
void _debug_print(const T& blocks) {
    for (Block* blk : blocks) {
        std::cout << *blk << " ";
    }
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

Plans Composer::stepOptimal(std::vector<SchedPlan> micros, const std::vector<float>& memory,
                            bool prune_symm, bool silence, int opt_step_upbound, int nworkers) {

    
    // construct block mapping to micro index
    Block2Idx blk2idx;
    for (std::size_t i = 0; i < micros.size(); ++i) {
        for (auto blk : micros[i].allBlocks()) {
            blk2idx.emplace(blk, int(i));
        }
    }

    std::size_t total_status = 1;
    Block2Hash blk2hash(micros);

    int step = 0;
    // init opt_step upper boundary
    int opt_step = 0;
    for (auto& micro : micros) {
        opt_step += micro.nSteps();
    }
    opt_step = (opt_step_upbound == -1) ? opt_step : opt_step_upbound;

    // ============== BFS search ==============
    
    // init curr status and next status
    std::vector<Plans> curr({micros});
    std::vector<Plans> next;
    std::vector<SchedPlan> schedules;

    // BFS steps
    while (step < opt_step) {
        if (!silence) std::cout << "solving step " << step << ", candidates: " << curr.size() << std::endl;
        const int ncandidates = curr.size();

        // ==================== single thread version =================
        // for (int idx = 0; idx < ncandidates; ++idx) {
        //     auto candidates_and_schedules = Composer::resolveStep(curr[idx], memory, step, opt_step, blk2hash, blk2idx);
        //     next.insert(
        //         next.end(),
        //         candidates_and_schedules.first.begin(),
        //         candidates_and_schedules.first.end()
        //     );
        // 
        //     for (auto& sched : candidates_and_schedules.second) {
        //         if (sched.nSteps() < opt_step) {
        //             if (!silence) std::cout << "find fewer steps: " << sched.nSteps() << std::endl;
        //             opt_step = sched.nSteps();
        //             schedules.clear();
        //         }
        //         if (sched.nSteps() == opt_step) {
        //             schedules.push_back(sched);
        //         }
        //     }
        // }
    
        // ==================== openmp parallelized version =================
        std::vector<std::size_t> num_next(nworkers, 0);
        std::vector<int> local_opts(nworkers, opt_step);
        #pragma omp parallel num_threads(nworkers)
        {
            int tid = omp_get_thread_num();
            int start = curr.size() / nworkers * tid;
            int stop = (tid == nworkers - 1) ? ncandidates : start + ncandidates / nworkers;
            int local_opt_step = opt_step;
            std::vector<Plans> local_next;
            std::vector<SchedPlan> local_schedules;
            for (int idx = start; idx < stop; idx++) {
                auto candidates_and_schedules = Composer::resolveStep(
                    curr[idx], memory, step, local_opt_step, blk2hash, blk2idx);
                local_next.insert(
                    local_next.end(),
                    candidates_and_schedules.first.begin(),
                    candidates_and_schedules.first.end()
                );

                // very important pruning techniques
                for (auto& ms : candidates_and_schedules.first) {
                    int opt_step_bound = step + 1;
                    for (auto& micro : ms) {
                        if (micro.nSteps() > step) {
                            opt_step_bound += micro.nSteps() - step - 1;
                        }
                    }
                    if (opt_step_bound < local_opt_step) {
                        local_schedules.clear();
                        local_opt_step = opt_step_bound;
                    }
                    // if (opt_step_bound == local_opt_step) {
                    //     Plans stack_part;
                    //     Plans concat_part;
                    //     for (auto& micro : ms) {
                    //         stack_part.push_back(micro.selectSteps(0, step+1));
                    //         concat_part.push_back(micro.selectSteps(step+1, -1));
                    //     }
                    //     SchedPlan stacked = SchedPlan::stack(stack_part);
                    //     SchedPlan concated = SchedPlan::concat(concat_part);
                    //     Plans to_concat = {stacked, concated};
                    //     SchedPlan sched = SchedPlan::concat(to_concat);
                    //     std::cout << "--stacked\n" << stacked << std::endl;
                    //     std::cout << "--concated\n" << concated << std::endl;
                    //     std::cout << "--\n" << sched << std::endl;
                    //     local_schedules.push_back(sched);
                    // }
                }

                for (auto& sched : candidates_and_schedules.second) {
                    if (sched.nSteps() < local_opt_step) {
                        local_opt_step = sched.nSteps();
                        local_schedules.clear();
                    }
                    if (sched.nSteps() == local_opt_step) {
                        local_schedules.push_back(sched);
                    }
                }
            }
            num_next[tid] = local_next.size();
            local_opts[tid] = local_opt_step;

            #pragma omp barrier

            #pragma omp single 
            {
                std::size_t total_next = 0;
                for (int num : num_next) {
                    total_next += num;
                }
                next.resize(total_next);
                for (int local_opt : local_opts) {
                    if (local_opt < opt_step) {
                        if (!silence) { printf("find fewer steps: %d\n", local_opt); }
                        opt_step = local_opt;
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
                if (local_opt_step == opt_step) {
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
    }

    if (!silence) {
        std::cout << "search done on " << total_status << " cases. "
                  << "find " << schedules.size() << " tight step-optimal plans "
                  << "(step = " << opt_step << ")" << std::endl;
    }
    return schedules;
}


std::pair<std::vector<Plans>, std::vector<SchedPlan>>
Composer::resolveStep(const Plans& micros, const std::vector<float>& memory,
                      int step, int upper_opt_step,
                      const Block2Hash& blk2hash, Block2Idx& blk2idx) {

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
            upper_opt_step = std::min(sched.nSteps(), upper_opt_step);
            if (sched.nSteps() <= upper_opt_step) {
                schedules.push_back(sched);
            }
        }
        else {
            // pruning technique: discard plans that exceed opt_step
            bool discard = false;
            for (auto& micro : cmicros) {
                // ==========> change this to upper_opt_step - 1 will make faster
                if (micro.nSteps() > upper_opt_step) {
                    discard = true;
                    break;
                }
            }
            if (!discard) {
                next.push_back(cmicros);
            }
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
            if (!micro.isTheStart(blk, step)) return false;
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
