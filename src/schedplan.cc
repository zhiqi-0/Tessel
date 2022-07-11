#include <schedplan.h>
#include <unique.h>
#include <iostream>

// ***** Plan Constructor ********

SchedPlan::SchedPlan(int ndevs, int nsteps): _ndevs(ndevs), _reserve_steps(nsteps) {
    this->_plans = std::vector<std::vector<Block*>>(ndevs);
    for (int devid = 0; devid < ndevs; ++devid) {
        this->_plans[devid].resize(nsteps, nullptr);
    }
}


SchedPlan::SchedPlan(const SchedPlan& plan) {
    std::cout << "copy constructor: " << plan.allBlocks().size() << std::endl;
    std::cout << plan << std::endl;
    this->_ndevs = plan.nDevs();
    this->_maxsteps = plan.nSteps() - 1;
    this->_reserve_steps = plan.nReserveSteps();
    this->_plans = std::vector< std::vector<Block*> >(this->_ndevs);
    for (int devid = 0; devid < this->_ndevs; ++devid) {
        this->_plans[devid].resize(this->_reserve_steps, nullptr);
    }
    for (auto blk : plan.allBlocks()) {
        auto device = plan.getDevice(blk);
        int step = plan.getStep(blk);
        this->addBlock(blk, device, step);
    }
    std::cout << "finished copy: \n" << *this << std::endl;
}


// ***** Plan Modifier ********

void SchedPlan::addBlock(Block* block, std::vector<int> devids, int step) {
    /**
     * @brief Get a block into the schedule plan
     * 
     */
    if (step >= this->_reserve_steps) this->reserve(step * 2);
    if (this->haveBlock(block)) {
        throw std::runtime_error("Try to double-add a block.");
    }
    for (const auto& devid : devids) {
        if (_plans.at(devid)[step] != nullptr) {
            throw std::runtime_error("Try to add a block with conflict postion on others.");
        }
        this->_plans.at(devid)[step] = block;
    }
    this->_blocks.insert(block);
    this->_block_devices.emplace(block, devids);
    this->_block_steps.emplace(block, step);
    this->_maxsteps = std::max(this->_maxsteps, step);
}


void SchedPlan::addBlock(Block* block, int device, int step) {
    /**
     * @brief Add a block into schedule plan
     * 
     */
    if (step >= this->_reserve_steps) this->reserve(step * 2);
    if (this->haveBlock(block)) {
        throw std::runtime_error("Try to double-add a block.");
    }
    if (this->getBlock(device, step) != nullptr) {
        throw std::runtime_error("Try to add a block with conflict postion on others.");
    }
    this->_plans[device][step] = block;
    this->_blocks.insert(block);
    this->_block_devices.emplace(block, std::vector<int>({device}));
    this->_block_steps.emplace(block, step);
    this->_maxsteps = std::max(this->_maxsteps, step);
}


void SchedPlan::setPosition(Block* block, std::vector<int> devices, int step) {
    /**
     * @brief Reset the block position
     * 
     */
    if (step >= this->_reserve_steps) this->reserve(step * 2);
    if (!this->haveBlock(block)) {
        throw std::runtime_error("Try to reset a block that not exists.");
    }
    // remove orignal device and step
    std::vector<int> odevices = this->getDevice(block);
    int ostep = this->getStep(block);
    for (auto devid : odevices) {
        this->_plans[devid][ostep] = nullptr;
    }
    // add new device and step
    for (auto devid : devices) {
        this->_plans[devid][step] = block;
    }
    this->_block_devices[block] = devices;
    this->_block_steps[block] = step;
    this->_maxsteps = std::max(this->_maxsteps, step);
}


// ***** Plan Property ********


float SchedPlan::peakMemory(int devid, int from_step, int to_step) const {
    float peak_mem = 0.0;
    float mem = 0.0;
    from_step = std::max(0, from_step);
    to_step = (to_step == -1 or to_step > this->nSteps()) ? nSteps() : to_step;
    for (int step = from_step; step < to_step; ++step) {
        Block* blk = this->_plans[devid][step];
        if (blk != nullptr) {
            mem += blk->memory;
            peak_mem = std::max(peak_mem, mem);
        }
    }
    return peak_mem;
}


float SchedPlan::currMemory(int devid, int from_step, int to_step) const {
    /**
     * @brief Get current memory of device devid until to_step
     * 
     */
    float mem = 0.0;
    from_step = std::max(0, from_step);
    to_step = (to_step == -1 or to_step > this->nSteps()) ? nSteps() : to_step;
    for (int step = from_step; step < to_step; ++step) {
        Block* blk = this->_plans.at(devid).at(step);
        if (blk != nullptr) {
            mem += blk->memory;
        }
    }
    return mem;
}


float SchedPlan::bubble_rate() const {
    /**
     * @brief get bubble rate of this plan
     * 
     */
    float nbubbles = 0;
    for (int devid = 0; devid < nDevs(); ++devid) {
        for (int step = 0; step < nSteps(); ++step) {
            if (_plans[devid][step] == nullptr) {
                nbubbles += 1;
            }
        }
    }
    return nbubbles / float(nDevs() * nSteps());
}


// ***** Plan Block Access ********


std::vector<int> SchedPlan::getDevice(Block* blk) const {
    /**
     * @brief Get block devices
     * 
     */
    if (!this->haveBlock(blk)) {
        throw std::runtime_error("block not exists");
    }
    return this->_block_devices.find(blk)->second;
}


int SchedPlan::getStep(Block* blk) const {
    /**
     * @brief Get block steps
     * 
     */
    if (!this->haveBlock(blk)) {
        throw std::runtime_error("block not exists");
    }
    return this->_block_steps.find(blk)->second;
}


std::vector<Block*> SchedPlan::stepBlocks(int step) const {
    /**
     * @brief Get blocks on a step
     * 
     */
    std::vector<Block*> blks;
    for (int devid = 0; devid < _ndevs; ++devid) {
        Block* blk = _plans[devid][step];
        if (blk != nullptr) {
            blks.push_back(blk);
        }
    }
    return blks;
}


std::vector<Block*> SchedPlan::devBlocks(int devid, int start_step, int end_step) const {
    /**
     * @brief Get blocks on a device
     * 
     */
    std::vector<Block*> blks;
    if (end_step == -1 or end_step > _maxsteps) {
        end_step = _maxsteps + 1;
    }
    for (int step = start_step; step < end_step; ++step) {
        Block* blk = _plans.at(devid)[step];
        if (blk != nullptr) {
            blks.push_back(blk);
        }
    }
    return blks;
}


// ***** Plan Selection and Creation ********


SchedPlan SchedPlan::selectSteps(int from_step, int to_step) const {
    SchedPlan sched(_ndevs, _reserve_steps);
    for (int step = from_step; step < to_step; ++step) {
        for (auto blk : stepBlocks(step)) {
            auto devids = getDevice(blk);
            int step = this->getStep(blk);
            sched.addBlock(blk, devids, step-from_step);
        }
    }
    return sched;
}


SchedPlan SchedPlan::selectBlocks(const std::set<Block*>& blocks) const {
    /**
     * @brief Create a schedule plan only containing the set of blocks.
     * 
     */
    SchedPlan sched(_ndevs, _reserve_steps);
    for (auto blk : blocks) {
        auto devids = this->getDevice(blk);
        int step = this->getStep(blk);
        sched.addBlock(blk, devids, step);
    }
    return sched;
}


SchedPlan SchedPlan::selectMicros(const std::set<int>& micro_ids) const {
    /**
     * @brief Create a schedule plan only containing blocks of certain micro ids.
     * 
     */
    SchedPlan sched(this->_ndevs, this->_reserve_steps);
    for (auto blk : _blocks) {
        if (micro_ids.find(blk->mid) != micro_ids.end()) {
            auto devids = this->getDevice(blk);
            int step = this->getStep(blk);
            sched.addBlock(blk, devids, step);
        }
    }
    return sched;
}


SchedPlan SchedPlan::concat(std::vector<SchedPlan>& plans) {
    /**
     * @brief Concatenate schedule plans
     * 
     */
    int nsteps = 0;
    for (auto& plan : plans) {
        nsteps += plan.nSteps();
    }
    int ndevs = plans[0].nDevs();
    SchedPlan sched(ndevs, nsteps);
    int ofst = 0;
    for (auto& plan: plans) {
        for (auto blk : plan.allBlocks()) {
            auto devices = plan.getDevice(blk);
            int step = plan.getStep(blk);
            sched.addBlock(blk, devices, step+ofst);
        }
        ofst += plan.nSteps();
    }
    return sched;
}


bool SchedPlan::stackable(std::vector<SchedPlan>& plans, const std::vector<float>& memory) {
    /**
     * @brief CHeck plans satisfy the stackable conditon, i.e., no conflicts and 
     * within memory constraints.
     * 
     */
    int nsteps = 0;
    for (auto& plan : plans) {
        nsteps = std::max(nsteps, plan.nSteps());
    }
    int ndevs = plans[0].nDevs();
    for (int devid = 0; devid < ndevs; ++devid) {
        for (int step = 0; step < nsteps; ++step) {
            bool have_block = false;
            for (auto& plan : plans) {
                if (step < plan.nSteps() and plan.getBlock(devid, step) != nullptr) {
                    if (have_block) {
                        return false;
                    }
                    have_block = true;
                }
            }
        }
    }
    return true;
}


SchedPlan SchedPlan::stack(std::vector<SchedPlan>& plans) {
    int nsteps = 0;
    for (auto& plan : plans) {
        nsteps = std::max(nsteps, plan.nSteps());
    }
    int ndevs = plans[0].nDevs();
    SchedPlan sched(ndevs, nsteps);
    for (auto& plan: plans) {
        for (auto blk : plan.allBlocks()) {
            auto devices = plan.getDevice(blk);
            int step = plan.getStep(blk);
            sched.addBlock(blk, devices, step);
        }
    }
    return sched;
}


// ***** Plan Primitive ********

void SchedPlan::shift(Block* blk) {
    /**
     * @brief The primitive: shift a block by pushing one step later
     * 
     */
    int step = this->getStep(blk);
    std::vector<int> devids = this->getDevice(blk);
    // shift next blocks
    for (auto ablk : blk->after) {
        if (this->haveBlock(ablk) and this->getStep(ablk) == step + 1) {
            this->shift(ablk);
        }
    }
    // shift later blocks
    for (auto devid : devids) {
        Block* lblk = this->getBlock(devid, step+1);
        if (lblk != nullptr) {
            this->shift(lblk);
        }
    }
    this->setPosition(blk, devids, step+1);
}


// ***** Plan String ********


std::string SchedPlan::toStr() const {
    std::string dscp;
    for (int devid = 0; devid < this->nDevs(); ++devid) {
        for (int step = 0; step < this->nSteps(); ++step) {
            Block* blk = this->getBlock(devid, step);
            if (blk == nullptr)
                dscp += "-- ";
            else
                dscp += blk->toStr() + " ";
        }
        dscp += "\n";
    }
    return dscp;
}