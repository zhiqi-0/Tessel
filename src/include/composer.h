#pragma once

#include <iostream>
#include <unordered_map>
#include <utility>
#include <schedplan.h>
#include <vector>
#include <set>


using Plans = std::vector<SchedPlan>;
using Block2Idx = std::unordered_map<Block*, int>;


class Block2Hash {

 protected:

    int _uid;
    std::unordered_map<Block*, int> _block2gid;


 public:

    Block2Hash(): _uid(0) {};

    Block2Hash(const std::vector<SchedPlan>& plans);

    Block2Hash(const Block2Hash& blk2hash) = delete;

    inline int getUid(Block* blk) const { return _block2gid.find(blk)->second; };

    static bool samePlan(const SchedPlan& sched1, const SchedPlan sched2);

};


class Conflict {

 public:
    std::unordered_map< int, std::set<Block*> > device2blocks;
    std::unordered_map<Block*, std::vector<int> > block2device;

    Conflict() {};
    Conflict(int ndevs) {
        for (int devid = 0; devid < ndevs; ++devid) {
            device2blocks.emplace(devid, std::set<Block*>());
        }
    }

    void addBlock(Block* blk, int devid) {
        if (this->device2blocks.find(devid) == this->device2blocks.end()) {
            this->device2blocks.emplace(devid, std::set<Block*>({blk}));
        }
        else {
            this->device2blocks[devid].insert(blk);
        }
        if (this->block2device.find(blk) == this->block2device.end()) {
            this->block2device.emplace(blk, std::vector<int>());
        }
        this->block2device[blk].push_back(devid);
    }

    inline std::vector<int> getDevice(Block* blk) const { return this->block2device.find(blk)->second; }
    
    inline bool haveBlock(Block* blk) const { return this->block2device.find(blk) != this->block2device.end(); }

    std::vector<Block*> allBlocks() const {
        std::vector<Block*> ret;
        ret.reserve(block2device.size());
        for (auto it: this->block2device) {
            ret.push_back(it.first);
        }
        return ret;
    }

    const std::set<Block*> getDeviceBlocks(int devid) const {
        if (this->device2blocks.find(devid) == device2blocks.end()) {
            return std::set<Block*>();
        }
        return this->device2blocks.find(devid)->second;
    }

    friend std::ostream& operator<<(std::ostream& out, const Conflict& sched) {
        out << "Conflict: ";
        for (const auto it : sched.device2blocks) {
            if (it.second.size() == 0) {
                continue;
            }
            out << "dev" << it.first << "(";
            for (auto blk : it.second) {
                out << " " << blk->toStr();
            }
            out << " ) ";
        }
        return out;
    }
};


class Composer {

 public:

    static Plans stepOptimal(std::vector<SchedPlan> micros, const std::vector<float>& memory,
                             bool prune_symm = true, bool silence = false, int opt_step_upbound = -1,
                             int nworkers = 1);

    static std::pair<std::vector<Plans>, std::vector<SchedPlan>>
    resolveStep(const Plans& micros, const std::vector<float>& memory,
                int step, int upper_opt_step,
                const Block2Hash& blk2hash, Block2Idx& blk2idx);

    static std::vector<std::set<Block*>>
    getShiftSpace(const int ndevice,
                  const Conflict& step_conflict, const Conflict& mem_conflict,
                  const Block2Hash& blk2hash);

    static Conflict getStepConflict(const std::vector<SchedPlan>& micros, int step,
                                    const Block2Hash& blk2hash);

    static Conflict getMemConflict(const std::vector<SchedPlan>& micros, int step,
                                   const std::vector<float>& memory, const Block2Hash& blk2hash);

};