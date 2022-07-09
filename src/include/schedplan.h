#pragma once

#include <vector>
#include <cmath>
#include <set>
#include <unordered_map>
#include <string>
#include <iostream>

#include <unique.h>


enum class BlockType { Forward, Bakward };


class Block {

    int _uid = -1;

 public:

    int mid;
    float memory;
    float span;
    BlockType btype;
    // dependency track
    std::set<Block*> before;
    std::set<Block*> after;
    
    Block(int mid, BlockType btype, float memory = 1, float span = 1) {
        this->mid = mid;
        this->btype = btype;
        this->span = span;
        if (btype == BlockType::Forward)
            this->memory = std::abs(memory);
        else
            this->memory = 0.0 - std::abs(memory);
        this->span = span;
        _uid = UniqueID::GetInstance()->GetUid();
    }

    Block(const Block& blk) {
        this->mid = blk.mid;
        this->btype = blk.btype;
        this->span = blk.span;
        this->memory = blk.memory;
        this->_uid = blk.uid();
        this->before = blk.before;
        this->after = blk.after;
    }

    inline bool operator == (const Block& blk) { return _uid == blk.uid(); }

    inline const int uid() const { return this->_uid; }

    static void addDependency(Block* predecessor, Block* successor) {
        predecessor->after.insert(successor);
        successor->before.insert(predecessor);
    }

    static void addDependencies(std::vector<Block*>& blocks) {
        for (std::size_t idx = 0; idx < blocks.size() - 1; ++idx) {
            Block::addDependency(blocks[idx], blocks[idx+1]);
        }
    }

    std::string toStr() {
        if (btype == BlockType::Forward)
            return std::string("f") + std::to_string(mid);
        else
            return std::string("b") + std::to_string(mid);
    }

};


class SchedPlan{

 protected:

    std::set<Block*> _blocks;
    std::unordered_map<Block*, std::vector<int>> _block_devices;
    std::unordered_map<Block*, int> _block_steps;
    std::vector< std::vector<Block*> > _plans;

    int _ndevs;
    int _maxsteps = 0;
    int _reserve_steps;

    void reserve(int nsteps) {
        /**
         * @brief reserve the step
         * 
         */
        if (nsteps > _reserve_steps) {
            for (int devid = 0; devid < _ndevs; ++devid) {
                _plans[devid].resize(nsteps, nullptr);
            }
            _reserve_steps = nsteps;
        }
    }

 public:
    
    SchedPlan(int ndevs, int nsteps): _ndevs(ndevs), _reserve_steps(nsteps) {
        _plans = std::vector<std::vector<Block*>>(ndevs);
        for (int devid = 0; devid < ndevs; ++devid) {
            _plans[devid].resize(nsteps, nullptr);
        }

    }

    SchedPlan(const SchedPlan& plan) {}

    // ***** Plan Modifier ********

    void setPosition(Block* blk, std::vector<int> devices, int step);

    void addBlock(Block* block, std::vector<int> devices, int step);

    void addBlock(Block* block, int device, int step);

    // ***** Plan Property ********

    inline int nDevs() { return this->_ndevs; }

    inline int nSteps() { return this->_maxsteps + 1; }

    float memory(int devid);

    float currMemory(int devid, int to_step = -1);

    float bubble_rate();

    // ***** Plan Block Access ********

    std::vector<int> getDevice(Block* blk);

    int getStep(Block* blk);

    inline Block* getBlock(const int devid, const int step) {
        if (step >= this->nSteps())
            return nullptr;
        else
            return _plans.at(devid).at(step);
    }

    const std::set<Block*> allBlocks() { return _blocks; }

    std::vector<Block*> stepBlocks(int step);

    std::vector<Block*> devBlocks(int devid, int start_step, int end_step = -1);

    inline bool haveBlock(Block* block) { return _blocks.find(block) != _blocks.end(); }

    // ***** Plan Selection and Creation ********

    SchedPlan selectSteps(int from_step, int to_step);

    SchedPlan selectBlocks(const std::set<Block*>& blocks);

    SchedPlan selectMicros(const std::set<int>& micro_ids);

    static SchedPlan concat(std::vector<SchedPlan>& plans);

    static bool stackable(std::vector<SchedPlan>& plans, const std::vector<float>& memory);

    static SchedPlan stack(std::vector<SchedPlan>& plans);

    // ***** Plan Primitive ********

    void shift(Block* blk);

    std::string toStr();

};

