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

#include <string>


#define TEST(res, expect) \
    if (res != expect) \
        throw std::runtime_error(std::string("fail at line: ")+std::to_string(__LINE__));

SchedPlan test_plan_creation_vshape(int mid = 0) {
    int ndevs = 4;
    SchedPlan micro(ndevs, ndevs);
    std::vector<Block*> fblocks(ndevs, nullptr);
    std::vector<Block*> bblocks(ndevs, nullptr);

    // test addBlock
    for (int devid = 0; devid < ndevs; ++devid) {
        fblocks[devid] = new Block(mid ,BlockType::Forward, 1.0, 1.0);
        micro.addBlock(fblocks[devid], devid, devid);
    }
    for (int devid = 0; devid < ndevs; ++devid) {
        bblocks[devid] = new Block(mid ,BlockType::Bakward, 1.0, 1.0);
        std::vector<int> devices = {devid};
        micro.addBlock(bblocks[devid], ndevs-1-devid, ndevs+devid);
    }

    // test add Dependency
    std::vector<Block*> blocks(fblocks);
    blocks.insert(blocks.end(), bblocks.begin(), bblocks.end());
    Block::addDependencies(blocks);

    TEST(micro.allBlocks().size(), 8)

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

    TEST(micro.allBlocks().size(), 10)

    return micro;
}


int test_shift() {

    // test single device block shift
    SchedPlan sched = test_plan_creation_vshape();
    Block* fdev0 = sched.getBlock(0, 0);
    Block* fdev3 = sched.getBlock(3, 3);
    Block* bdev2 = sched.getBlock(2, 5);
    Block* bdev0 = sched.getBlock(0, 7);
    sched.shift(fdev0);
    sched.shift(fdev3);
    sched.shift(bdev2);
    sched.shift(bdev0);
    std::cout << sched << std::endl;

    // test multi device block shift
    SchedPlan msched = test_plan_creation_mshape();
    Block* mfadev0 = msched.getBlock(0, 0);
    Block* mfadev1 = msched.getBlock(1, 0);
    Block* mfdev1 = msched.getBlock(1, 2);
    msched.shift(mfadev0);
    msched.shift(mfadev1);
    msched.shift(mfdev1);
    std::cout << msched << std::endl;

    return 0;
}


int test_memory_bubble_rate() {
    SchedPlan msched = test_plan_creation_mshape();
    TEST(msched.memory(0), 2.0)
    TEST(msched.currMemory(0, 1), 1.0)
    TEST(msched.currMemory(0, 2), 2.0)
    TEST(msched.currMemory(0, 3), 2.0)
    TEST(msched.currMemory(0, 100), 0.0)
    TEST(msched.bubble_rate(), float(0.600))
    return 0;
}


int test_stack() {

    int nmicros = 4;
    int ndevs = 4;
    std::vector<float> memory(ndevs, float(ndevs));
    std::vector<SchedPlan> micros(nmicros);
    for (int mid = 0; mid < nmicros; ++mid) {
        micros[mid] = test_plan_creation_vshape();
    }

    for (int idx = 0; idx < nmicros; idx++) {
        auto blk = micros[idx].getBlock(0, 0);
        for (int t = 0; t < idx * 2; ++t) {
            micros[idx].shift(blk);
        }
        std::cout << micros[idx] << std::endl;
    }

    TEST(SchedPlan::stackable(micros, memory), true)
    SchedPlan sched = SchedPlan::stack(micros);
    std::cout << sched << std::endl;
    TEST(sched.nSteps(), 14)

}


void test_concat() {
    std::vector<SchedPlan> plans;
    plans.push_back(test_plan_creation_vshape());
    plans.push_back(test_plan_creation_mshape());
    SchedPlan csched = SchedPlan::concat(plans);
    std::cout << csched;

}



int main() {
    SchedPlan micro1 = test_plan_creation_vshape();
    std::cout << micro1 << std::endl;
    SchedPlan micro2 = test_plan_creation_mshape();
    std::cout << micro2 << std::endl;

    test_shift();
    test_memory_bubble_rate();
    test_stack();
    test_concat();

    return 0;
}