#pragma once

#include <vector>
#include <cmath>
#include <set>
#include <unordered_map>
#include <string>
#include <iostream>
#include <ostream>

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

    std::string toStr() const {
        if (btype == BlockType::Forward)
            return std::string("f") + std::to_string(this->mid);
        else
            return std::string("b") + std::to_string(this->mid);
    }

    friend std::ostream& operator<<(std::ostream& out, const Block& block) {
        out << block.toStr();
        return out;
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
        if (nsteps > this->_reserve_steps) {
            for (int devid = 0; devid < _ndevs; ++devid) {
                this->_plans[devid].resize(nsteps, nullptr);
            }
            this->_reserve_steps = nsteps;
        }
    }

 public:
    
    SchedPlan(int ndevs = 1, int nsteps = 1);

    SchedPlan(const SchedPlan& plan);

    // ***** Plan Modifier ********

    void setPosition(Block* blk, std::vector<int> devices, int step);

    void addBlock(Block* block, std::vector<int> devices, int step);

    void addBlock(Block* block, int device, int step);

    // ***** Plan Property ********

    inline int nDevs() const { return this->_ndevs; }

    inline int nSteps() const { return this->_maxsteps + 1; }

    inline int nReserveSteps() const { return this->_reserve_steps; }

    float memory(int devid) const;

    float currMemory(int devid, int to_step = -1) const;

    float bubble_rate() const;

    // ***** Plan Block Access ********

    std::vector<int> getDevice(Block* blk) const;

    int getStep(Block* blk) const;

    inline Block* getBlock(const int devid, const int step) const {
        // std::cout << "access step: " << step << " devid: " << devid << " size: " << this->_plans[devid].size() << std::endl;
        return (devid >= this->_ndevs or step > this->_maxsteps) ? nullptr : this->_plans.at(devid).at(step);
    }

    const std::set<Block*> allBlocks() const { return _blocks; }

    std::vector<Block*> stepBlocks(int step) const;

    std::vector<Block*> devBlocks(int devid, int start_step, int end_step = -1) const;

    inline bool haveBlock(Block* block) const { return _blocks.find(block) != _blocks.end(); }

    // ***** Plan Selection and Creation ********

    SchedPlan selectSteps(int from_step, int to_step) const;

    SchedPlan selectBlocks(const std::set<Block*>& blocks) const;

    SchedPlan selectMicros(const std::set<int>& micro_ids) const;

    static SchedPlan concat(std::vector<SchedPlan>& plans);

    static bool stackable(std::vector<SchedPlan>& plans, const std::vector<float>& memory);

    static SchedPlan stack(std::vector<SchedPlan>& plans);

    // ***** Plan Primitive ********

    void shift(Block* blk);

    // ***** Console Visualize ********

    std::string toStr() const;

    friend std::ostream& operator<<(std::ostream& out, const SchedPlan& sched) {
        out << sched.toStr();
        return out;
    }

};

