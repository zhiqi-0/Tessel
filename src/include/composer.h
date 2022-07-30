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
    int step;

    Conflict(): step(-1) {};
    Conflict(int ndevs, int step) : step(step) {
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

    void addBlock(Block* blk, std::vector<int> devs) {
        for (int devid : devs) {
            device2blocks.at(devid).insert(blk);
        }
        block2device.emplace(blk, devs);
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
                             bool silence = false, int opt_step_upbound = -1,
                             int nworkers = 1);

    static Plans stepOptimalDFS(Plans micros, const std::vector<float>& memory,
                                bool silence = false, int opt_step_upbound = -1,
                                int nworkers = 1);

    static Plans stepOptimalBDFS(Plans micros, const std::vector<float>& memory,
                                 bool silence = false, int opt_step_upbound = -1,
                                 int nworkers = 1);

    static std::pair<std::vector<Plans>, std::vector<SchedPlan>>
    resolveStep(const Plans& micros, const std::vector<float>& memory,
                int step, int upper_opt_step,
                const Block2Hash& blk2hash, Block2Idx& blk2idx);

    static std::vector<std::set<Block*>>
    getShiftSpace(const int ndevice, const Plans& micros,
                    const Conflict& can_keep, const Conflict& to_shift,
                    const Block2Idx& blk2idx);

    static std::pair<Conflict, Conflict>
    getConflict(const Plans& micros, int step,
                const std::vector<float>& memory,
                const Block2Hash& blk2hash);

    static bool isDynSymm(const Plans& micros, int step);

    static float currMemory(const Plans& micros, int devid, int from_step, int to_step);

};