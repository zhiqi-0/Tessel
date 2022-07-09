/**
 * @file test.cc
 * @author your name (you@domain.com)
 * @brief Test for schedule implementation
 * @version 0.1
 * @date 2022-07-09
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <schedplan.h>
#include <vector>
#include <iostream>


SchedPlan test_plan_creation_vshape() {
    int ndevs = 4;
    SchedPlan micro(ndevs, ndevs);
    std::vector<Block*> fblocks(ndevs, nullptr);
    std::vector<Block*> bblocks(ndevs, nullptr);

    // test addBlock
    for (int devid = 0; devid < ndevs; ++devid) {
        fblocks[devid] = new Block(0,BlockType::Forward, 1.0, 1.0);
        micro.addBlock(fblocks[devid], devid, devid);
    }
    for (int devid = 0; devid < ndevs; ++devid) {
        bblocks[devid] = new Block(0,BlockType::Bakward, 1.0, 1.0);
        std::vector<int> devices = {devid};
        micro.addBlock(bblocks[devid], ndevs-1-devid, ndevs+devid);
    }

    // test add Dependency
    std::vector<Block*> blocks(fblocks);
    blocks.insert(blocks.end(), bblocks.begin(), bblocks.end());
    Block::addDependencies(blocks);

    if (micro.allBlocks().size() != 8) {
        throw std::runtime_error("Expected to 8");
    }

    return micro;
}


SchedPlan test_plan_creation_mshape() {
    /**
     *    f f            b b
     *    f  f         b   b
     *    f    f     b     b
     *    f      f b       b
     *
     */
    int ndevs = 4;
    SchedPlan micro(ndevs, ndevs);
    std::vector<Block*> fblocks(ndevs+1, nullptr);
    std::vector<Block*> bblocks(ndevs+1, nullptr);

    std::vector<int> devids(ndevs);
    for (int devid = 0; devid < ndevs; ++devid) {
        devids[devid] = devid;
    }

    fblocks[0] = new Block(0, BlockType::Forward, 1.0, 1.0);
    micro.addBlock(fblocks[0], devids, 0);
    for (int devid = 0; devid < ndevs; ++devid) {
        fblocks[devid+1] = new Block(0,BlockType::Forward, 1.0, 1.0);
        micro.addBlock(fblocks[devid+1], devid, devid+1);
    }

    for (int devid = 0; devid < ndevs; ++devid) {
        bblocks[devid] = new Block(0,BlockType::Bakward, 1.0, 1.0);
        std::vector<int> devices = {devid};
        micro.addBlock(bblocks[devid], ndevs-1-devid, ndevs+1+devid);
    }
    bblocks[ndevs] = new Block(0,BlockType::Bakward, 1.0, 1.0);
    micro.addBlock(bblocks[ndevs], devids, ndevs+1+ndevs);

    // test add Dependency
    std::vector<Block*> blocks(fblocks);
    blocks.insert(blocks.end(), bblocks.begin(), bblocks.end()); 
    Block::addDependencies(blocks);

    if (micro.allBlocks().size() != 10) {
        throw std::runtime_error("Expected to 8");
    }

    return micro;
}


int test_schedplan() {

    int ndevs = 4;
    int nmicros = 4;

    SchedPlan micro(ndevs, ndevs);
    std::vector<Block*> fblocks(ndevs, nullptr);
    std::vector<Block*> bblocks(ndevs, nullptr);

    // test add
    for (int devid = 0; devid < ndevs; ++devid) {
        fblocks[devid] = new Block(0,BlockType::Forward, 1.0, 1.0);
        micro.addBlock(fblocks[devid], devid, devid);
    }
    for (int devid = 0; devid < ndevs; ++devid) {
        bblocks[devid] = new Block(0,BlockType::Bakward, 1.0, 1.0);
        std::vector<int> devices = {devid};
        micro.addBlock(bblocks[devid], ndevs-1-devid, ndevs+devid);
    }
    std::cout << "test addBlock done." << std::endl;
    std::cout << micro.toStr() << std::endl;

    // test add dependency
    std::vector<Block*> blocks(fblocks);
    blocks.insert(blocks.end(), bblocks.begin(), bblocks.end());
    std::cout << "total blocks: " << blocks.size() << std::endl;
    Block::addDependencies(blocks);

    // test shift
    micro.shift(fblocks[0]);
    std::cout << micro.toStr() << std::endl;
    micro.shift(fblocks[1]);
    std::cout << micro.toStr() << std::endl;
    micro.shift(bblocks[2]);
    micro.shift(bblocks[3]);
    std::cout << micro.toStr() << std::endl;

    // test memory
    std::cout << micro.currMemory(0, 5) << " should be 1" << std::endl;
    std::cout << micro.currMemory(3, 5) << " shoud be 0" << std::endl;
    std::cout << micro.currMemory(3, 20) << " should be 0" << std::endl;
    for (int devid = 0; devid < ndevs; ++devid) {
        std::cout << "peak memory of device "
                  << devid << " : " << micro.memory(devid) << std::endl;
    }

    // test bubble rate
    std::cout << micro.bubble_rate() << " should be " << 1 - 2.0 / (8+4) << std::endl;
    
    return 0;
}



int main() {
    SchedPlan micro1 = test_plan_creation_vshape();
    std::cout << micro1 << std::endl;
    SchedPlan micro2 = test_plan_creation_mshape();
    std::cout << micro2 << std::endl;
    return 0;
}