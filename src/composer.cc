
#include <iostream>
#include <utility>
#include <vector>
#include <algorithm>
#include <limits>
#include <cstdio>

#include <omp.h>

#include <composer.h>


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
                    local_opt_step = std::min(local_opt_step, opt_step_bound);
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
                        if (!silence) { printf("find fewer steps: %d\n", local_opt_step); }
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

            // #pragma omp critical
            // {
            //     //int _x; std::cin >> _x;
            //     next.insert(next.end(), local_next.begin(), local_next.end());
            //     if (local_opt_step < opt_step) {
            //         if (!silence) { printf("find fewer steps: %d\n", local_opt_step); }
            //         opt_step = local_opt_step;
            //         schedules.clear();
            //     }
            //     if (local_opt_step == opt_step) {
            //         schedules.insert(
            //             schedules.end(), local_schedules.begin(), local_schedules.end());
            //     }
            // }
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

    Conflict step_conflict = Composer::getStepConflict(
        micros, step, blk2hash
    );
    // std::cout << "step conflict: " << step_conflict << std::endl;
    Conflict mem_conflict = Composer::getMemConflict(
        micros, step, memory, blk2hash
    );
    // std::cout << "memory conflict: " << mem_conflict << std::endl;

    std::vector< std::set<Block*> > all_shifts = Composer::getShiftSpace(
        ndevs, micros, step_conflict, mem_conflict, blk2hash, blk2idx
    );

    // std::cout << "prev:\n"; for (auto& micro : micros) std::cout << micro << std::endl;
    // for (auto shifts : all_shifts) {std::cout << "shift: "; for (auto blk : shifts) printf("%s(%d) ", blk->toStr().c_str(), blk->mid); std::cout << std::endl; }
    // { int _x; std::cin >> _x; }

    for (auto& shifts : all_shifts) {

        // copy for inplacement update
        Plans cmicros(micros);
        for (auto blk : shifts) {
            int idx = blk2idx[blk];
            cmicros[idx].shift(blk);
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
Composer::getShiftSpace(const int ndevice,
                        const Plans& micros,
                        const Conflict& step_conflict,
                        const Conflict& mem_conflict,
                        const Block2Hash& blk2hash,
                        const Block2Idx& blk2idx) {

    // set up candidates by pruning out same blocks
    std::unordered_map<int, Block*> keep_uids;
    for (auto cblock : step_conflict.allBlocks()) {
        if (mem_conflict.haveBlock(cblock)) continue;
        int uid = blk2hash.getUid(cblock);
        if (keep_uids.find(uid) != keep_uids.end()) {
            if (keep_uids[uid]->mid > cblock->mid) {
                keep_uids[uid] = cblock;
            }
        }
        else {
            keep_uids.emplace(uid, cblock);
        }
    }
    std::set<Block*> keep_candidates;
    for (auto& it : keep_uids) {
        keep_candidates.insert(it.second);
    }
    // printf("candidates: "); for (Block* blk : keep_candidates) { printf("%s(%d) ", blk->toStr().c_str(), blk->mid); } std::cout << std::endl;

    std::vector<std::set<Block*>> all_keeps;
    using Item = std::pair<std::set<Block*>, std::vector<bool>>;
    std::vector<Item> curr;
    std::vector<Item> next;
    curr.emplace_back(std::set<Block*>(), std::vector<bool>(ndevice, false));

    while (!curr.empty()) {
        // std::cout << "next =====" << std::endl;
        for (auto& item : curr) {
            bool no_more_add = true;
            for (Block* blk : keep_candidates) {
                // std::cout << "here1 " << *blk << std::endl;
                if (item.first.find(blk) == item.first.end()) {
                    bool can_insert = true;
                    for (int devid : step_conflict.getDevice(blk)) {
                        if (item.second[devid]) {
                            can_insert = false;
                            break;
                        }
                    }
                    //std::cout << "can insert: " << can_insert << std::endl;
                    if (can_insert) {
                        std::set<Block*> blks(item.first);
                        std::vector<bool> devs(item.second);
                        blks.insert(blk);
                        for (int devid : step_conflict.getDevice(blk)) {
                            devs[devid] = true;
                        }
                        next.emplace_back(blks, devs);
                        no_more_add = false;
                    }
                }
            }
            if (no_more_add) {
                // check redundant
                // printf("adding all_keeps: "); for (auto blk : item.first) { printf("%s ", blk->toStr().c_str()); } printf("\n");
                bool exist = false;
                for (auto& keep : all_keeps) {
                    bool same = true;
                    for (auto& blk : item.first) {
                        if (keep.find(blk) == keep.end()) {
                            same = false;
                            break;
                        }
                    }
                    if (same) {
                            exist = true;
                            break;
                        }
                }
                if (!exist) {
                    // dynamic symmetric pruning technique
                    bool discard = false;
                    // we only consider block at step-1, if no block in step-1, this indicates
                    // there are blocks in kblock.before happen at step-1,
                    // otherwise kblock can be put on step-1.
                    for (auto& kblk : item.first) {
                        //printf("start\n");
                        std::set<Block*> exblks;
                        bool exchangable = true;
                        int micro_idx = blk2idx.at(kblk);
                        //printf("here1: micro_idx: %d\n", micro_idx);
                        int step = micros[micro_idx].getStep(kblk);
                        if (step == 0) break;
                        // before block 
                        for (auto& bblk : kblk->before) {
                            if (micros[micro_idx].haveBlock(bblk) && micros[micro_idx].getStep(bblk) == step-1) {
                                exchangable = false;
                                break;
                            }
                        }
                        //printf("here2\n");
                        if (!exchangable) continue;
                        // after block
                        for (int devid : micros[micro_idx].getDevice(kblk)) {
                            for (auto& micro : micros) {
                                Block* exblk = micro.getBlock(devid, step-1);
                                if (exblk == nullptr) continue;
                                exblks.insert(exblk);
                                // exchange will not change peak memory
                                if (exblk->memory * kblk->memory < 0) {
                                    exchangable = false;
                                    break;
                                }
                                // exchange will not break dependency 
                                for (auto& ablk : exblk->after) {
                                    if (micro.haveBlock(ablk) and micro.getStep(ablk) == step and item.first.find(ablk) != item.first.end()) {
                                        exchangable = false;
                                        break;
                                    }
                                }
                                break;
                            }
                        }
                        //printf("here3\n");
                        if (exchangable) {
                            for (auto& exblk : exblks) {
                                if (exblk->mid > kblk->mid) {
                                    discard = true;
                                    break;
                                }
                            }
                        }
                    }

                    if (!discard) {
                        // printf("....success!\n");
                        all_keeps.push_back(item.first);
                    }
                }
            }
        }
        curr.clear();
        std::swap(curr, next);
    }


    // for (auto& keep : all_keeps) { printf("keeps: "); for (auto& blk : keep) {std::cout << *blk << " ";} std::cout << std::endl; } 

    std::vector<std::set<Block*> > all_shifts;
    std::set<Block*> all_blks;
    auto step_cblks = step_conflict.allBlocks();
    auto mem_cblks = step_conflict.allBlocks();
    all_blks.insert(step_cblks.begin(), step_cblks.end());
    all_blks.insert(mem_cblks.begin(), mem_cblks.end());
    for (auto& keep : all_keeps) {
        std::set<Block*> shifts;
        std::set_difference(
            all_blks.begin(), all_blks.end(),
            keep.begin(), keep.end(),
            std::inserter(shifts, shifts.begin())
        );
        all_shifts.push_back(shifts);
    }
    return all_shifts;
}


Conflict Composer::getStepConflict(const std::vector<SchedPlan>& micros, int step,
                                   const Block2Hash& blk2hash) {
    const int ndevs = micros.at(0).nDevs();
    Conflict conflict(ndevs);
    for (int devid = 0; devid < ndevs; ++devid) {
        std::vector<Block*> blks;
        for (auto& micro : micros) {
            auto blk = micro.getBlock(devid, step);
            if (blk != nullptr) {
                blks.push_back(blk);
            }
        }
        if (blks.size() > 1) {
            for (auto blk : blks) {
                conflict.addBlock(blk, devid);
            }
        }
    }
    return conflict;
}


Conflict Composer::getMemConflict(const std::vector<SchedPlan>& micros, int step,
                                  const std::vector<float>& memory, const Block2Hash& blk2hash) {
    const int ndevs = micros.at(0).nDevs();
    Conflict conflict(ndevs);

    std::vector<float> devmem(ndevs, 0.0);
    for (int devid = 0; devid < ndevs; ++devid) {
        for (auto& micro : micros) {
            devmem[devid] += micro.currMemory(devid, 0, step);
        }
    }

    for (std::size_t mid = 0; mid < micros.size(); ++mid) {
        for (auto blk : micros[mid].stepBlocks(step)) {
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
                    conflict.addBlock(blk, devid);
                }
            }
        }
    }

    return conflict;
}