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
    micro.shift(blocks[3]);
    std::cout << micro.toStr() << std::endl;
    
    return 0;
}

int main() {
    test_schedplan();
    return 0;
}