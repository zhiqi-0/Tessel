#pragma once

#include <vector>
#include <cmath>
#include <set>
#include <unordered_map>
#include <string>
#include <iostream>
#include <ostream>

#include <unique.h>


enum class BlockType { Forward, Backward };


class Block {

    int _uid = -1;

 public:

    int mid;
    float memory;
    int span;
    BlockType btype;
    // dependency track
    std::set<Block*> before;
    std::set<Block*> after;
    
    Block(int mid, BlockType btype, float memory = 1, int span = 1) {
        this->mid = mid;
        this->btype = btype;
        this->span = span;
        if (btype == BlockType::Forward)
            this->memory = std::abs(memory);
        else
            this->memory = 0.0 - std::abs(memory);
        if (span < 1) {
            throw std::runtime_error("cannot set a block with span samller than 1\n");
        }
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


class SchedPlan {

 protected:

    // all blocks in the plan
    std::set<Block*> _blocks;
    // the devices of a block
    std::unordered_map<Block*, std::vector<int>> _block_devices;
    // the start step of block
    std::unordered_map<Block*, int> _block_steps;
    // the 2D plan of: row-device | col-step
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

    /**
     * @brief Reset the block position
     * 
     */
    void setPosition(Block* blk, std::vector<int> devices, int step);

    /**
     * @brief Add a block into the schedule plan
     * 
     */
    void addBlock(Block* block, std::vector<int> devices, int step);

    /**
     * @brief Add a block into the schedule plan
     * 
     */
    void addBlock(Block* block, int device, int step);

    /**
     * @brief Add a block sequence into the schedule plan
     * 
     */
    void addBlockSeq(const std::vector<Block*>& blocks, const std::vector< std::vector<int> >& devices);

    /**
     * @brief Add a block sequence into the schedule plan
     * 
     */
    void addBlockSeq(const std::vector<Block*>& blocks, const std::vector<int>& devices);

    /**
     * @brief Sequeeze the plan by moving empty head steps and empty tail steps
     * 
     */
    void squeeze();

    // ***** Plan Property ********

    inline int nDevs() const { return this->_ndevs; }

    inline int nSteps() const { return this->_maxsteps + 1; }

    inline int nReserveSteps() const { return this->_reserve_steps; }

    float peakMemory(int devid, int from_step = 0, int to_step = -1) const;

    /**
     * @brief Get current memory of device devid until to_step
     * @note the first block that its start step is before `start_step` but
     * its end step is after `start_step` will not be considered.
     * @note the last block that its start step is before `end_step` but
     * its end step is after `end_step` will be considered.
     */
    float currMemory(int devid, int from_step = 0, int to_step = -1) const;

    /**
     * @brief Get bubble rate of this plan
     * 
     */
    float bubble_rate(int from_step = 0, int to_step = -1) const;

    // ***** Plan Block Access ********

    /**
     * @brief Get devices of the block
     * 
     */
    std::vector<int> getDevice(Block* blk) const;

    /**
     * @brief Get block steps
     * 
     */
    int getStep(Block* blk) const;

    inline Block* getBlock(const int devid, const int step) const {
        // std::cout << "access step: " << step << " devid: " << devid << " size: " << this->_plans[devid].size() << std::endl;
        return (devid >= this->_ndevs or step > this->_maxsteps) ? nullptr : this->_plans.at(devid).at(step);
    }

    const std::set<Block*> allBlocks() const { return _blocks; }

    /**
     * @brief Get blocks of the step. The blocks include not-the-start blocks.
     * 
     * @param step 
     * @return std::vector<Block*> 
     */
    std::vector<Block*> stepBlocks(int step) const;

    /**
     * @brief Get blocks on a device
     * @note the first block that its start step is before `start_step` but
     * its end step is after `start_step` will not be considered.
     * @note the last block that its start step is before `end_step` but
     * its end step is after `end_step` will be considered.
     */
    std::vector<Block*> devBlocks(int devid, int start_step, int end_step = -1) const;

    /**
     * @brief Check whether the block is in the plan
     */
    inline bool haveBlock(Block* block) const { return _blocks.find(block) != _blocks.end(); }

    /**
     * @brief Check whether the block is starting at `step`.
     */
    inline bool isTheStart(Block* block, int step) const { return _block_steps.at(block) == step; }

    // ***** Plan Selection and Creation ********

    /**
     * @brief Create a schedule plan that only contains blocks in steps of
     * [from_step, to_step]. the from_step will be increased if it is empty.
     *
     * @note the first block that its starting step is before `from_step` but
     * its ending step is after `start_step` will not be considered.
     * @note the last block that its starting step is before `end_step` but
     * its ending step is after `end_step` will be considered.
     */
    SchedPlan selectSteps(int from_step, int to_step = -1) const;

    /**
     * @brief Create a schedule plan only containing the set of blocks.
     */
    SchedPlan selectBlocks(const std::set<Block*>& blocks) const;

    SchedPlan selectMicros(const std::set<int>& micro_ids) const;

    /**
     * @brief Increase micro batch id for each block in the micro.
     * The result will be in a new instance.
     * 
     * @warning this will allocate memory for blocks, user should
     * manually delete them after finished the use.
     */
    SchedPlan increaseMid(const int increase_mid) const;

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

    /**
     * @brief Export the plan to json file
     * 
     * @param filename: the json file name
     */

    void to_json(const std::string& filename) const;

};


class GeneralSchedPlan: public SchedPlan {

 public:

    GeneralSchedPlan(): SchedPlan(), _empty(true) {}

    GeneralSchedPlan(const SchedPlan& lhead, const SchedPlan& steady, const SchedPlan& rtail);

    GeneralSchedPlan(const GeneralSchedPlan&);

    inline const int getLBound() const { return _lbound; }
    inline const int getRBound() const { return _rbound; }
    inline std::set<Block*> getCreatedBlocks() const {return _created;}
    inline bool isEmpty() const { return _empty; }

    void addCreatedBlocks(Block* blk) { _created.insert(blk); }
    void destroyCreatedBlocks();

    float steady_bubble_rate() const { return bubble_rate(_lbound, _rbound); }

    std::string toStr() const;

    friend std::ostream& operator<<(std::ostream& out, const GeneralSchedPlan& sched) {
        out << sched.toStr();
        return out;
    }




 private:
    int _lbound;
    int _rbound;
    bool _empty;
    std::set<Block*> _created;
};
