
#include <iostream>
#include <utility>
#include <vector>
#include <algorithm>
#include <limits>

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
    std::cout << "Got " << same_groups.size() << " groups" << std::endl;
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
                            bool prune_symm, bool silence, int nworkers) {

    
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

    // ============== BFS search ==============
    
    // init curr status and next status
    std::vector<Plans> curr({micros});
    std::vector<Plans> next;
    std::vector<SchedPlan> schedules;

    // BFS steps
    while (step < opt_step) {
        if (!silence) std::cout << "solving step " << step << ", candidates: " << curr.size() << std::endl;

        const int ncandidates = curr.size();
        for (int idx = 0; idx < ncandidates; ++idx) {
            auto candidates_and_schedules = Composer::resolveStep(curr[idx], memory, step, opt_step, blk2hash, blk2idx);
            next.insert(
                next.end(),
                candidates_and_schedules.first.begin(),
                candidates_and_schedules.first.end()
            );

            for (auto& sched : candidates_and_schedules.second) {
                if (sched.nSteps() < opt_step) {
                    if (!silence) std::cout << "find fewer steps: " << sched.nSteps() << std::endl;
                    opt_step = sched.nSteps();
                    schedules.clear();
                }
                if (sched.nSteps() == opt_step) {
                    schedules.push_back(sched);
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

    Conflict step_conflict = Composer::getStepConflict(
        micros, step, blk2hash
    );
    // std::cout << "step conflict: " << step_conflict << std::endl;
    Conflict mem_conflict = Composer::getMemConflict(
        micros, step, memory, blk2hash
    );
    // std::cout << "memory conflict: " << mem_conflict << std::endl;

    std::vector< std::set<Block*> > all_shifts = Composer::getShiftSpace(
        ndevs, step_conflict, mem_conflict, blk2hash
    );

    // std::cout << "prev:\n"; for (auto& micro : micros) std::cout << micro << std::endl;
    for (auto& shifts : all_shifts) {
        // std::cout << "shift: ";
        // for (auto blk : shifts) std::cout << *blk << " ";
        // std::cout << std::endl;
        // int _x; std::cin >> _x;

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
    return std::make_pair(next, schedules);
}


std::vector<std::set<Block*>>
Composer::getShiftSpace(const int ndevice,
                        const Conflict& step_conflict,
                        const Conflict& mem_conflict,
                        const Block2Hash& blk2hash) {

    // set up candidates
    std::unordered_map<int, std::vector<Block*>> keep_candidates(ndevice);
    for (auto cblock : step_conflict.allBlocks()) {
        if (!mem_conflict.haveBlock(cblock)) {
            for (auto devid : step_conflict.getDevice(cblock)) {
                keep_candidates[devid].push_back(cblock);
            }
        }
    }
    // for (auto& it : keep_candidates) {
    //     std::cout << "keep candidates: " << it.first << " : blocks: ";
    //     for (auto blk : it.second) {
    //         std::cout << " " << *blk;
    //     }
    //     std::cout << std::endl;
    // }

    std::vector< std::vector<Block*> > curr;  // current kept block
    std::vector< std::vector<Block*> > next;  // next kept block

    curr.push_back(std::vector<Block*>(ndevice, nullptr));

    for (int devid = 0; devid < ndevice; ++devid) {
        for (auto& keep_blks : curr) {
            if (keep_blks[devid] == nullptr) {
                std::vector<Block*> candidates;
                std::unordered_map<int, Block*> uid_blks;
                for (auto blk : keep_candidates[devid]) {
                    int uid = blk2hash.getUid(blk);
                    if (uid_blks.find(uid) != uid_blks.end()) {
                        if (blk->mid < uid_blks[uid]->mid) {
                            uid_blks[uid] = blk;
                        }
                    }
                    else {
                        uid_blks.emplace(uid, blk);
                    }
                }
                for (auto& it : uid_blks) {
                    candidates.push_back(it.second);
                }
                if (candidates.size() == 0) {
                    next.push_back(keep_blks);
                }
                else {
                    for (auto keep_blk : candidates) {
                        bool empty = true;
                        auto kdevs = step_conflict.getDevice(keep_blk);
                        for (int kdev : kdevs) {
                            if (keep_blks[kdev] != nullptr) {
                                empty = false;
                                break;
                            }
                        }
                        if (empty) {
                            std::vector<Block*> next_keep_blks(keep_blks);
                            for (int kdev : kdevs) {
                                next_keep_blks[kdev] = keep_blk;
                            }
                            next.push_back(next_keep_blks);
                        }
                    }
                }
            }
            else {
                next.push_back(keep_blks);
            }
        }
        curr.clear();
        std::swap(curr, next);
    }

    std::vector<std::set<Block*> > all_shifts;
    for (auto& keep_blks : curr) {
        std::set<Block*> shifts;
        for (int devid = 0; devid < ndevice; ++devid) {
            Block* kblock = keep_blks[devid];
            for (auto cblock : step_conflict.getDeviceBlocks(devid)) {
                if (cblock != nullptr and cblock != kblock) {
                    shifts.insert(cblock);
                }
            }
            for (auto cblock : mem_conflict.getDeviceBlocks(devid)) {
                shifts.insert(cblock);
            }
        }
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