/**
 * @file cases.cc
 * @author your name (zhiqi.0@outlook.com)
 * @brief Case search with premise
 * @version 0.1
 * @date 2022-07-10
 * 
 * @copyright Copyright (c) 2022
 * 
 */


#include <vector>
#include <iostream>
#include <string>
#include <schedplan.h>
#include <composer.h>

#include <chrono>


class CpuTimer {

 public:
    void start() { t1 = std::chrono::high_resolution_clock::now(); }
    void stop() { t2 = std::chrono::high_resolution_clock::now(); }

    int elapsed() {
        // in millisecond
        return std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    }

 private:
    std::chrono::high_resolution_clock::time_point t1, t2;
};


std::vector<SchedPlan> premise_vshape(int ndevs, int nmicros) {
    std::vector<SchedPlan> micros;
    for (int mid = 0; mid < nmicros; ++mid) {
        SchedPlan micro(ndevs, ndevs);
        std::vector<Block*> fblocks(ndevs, nullptr);
        std::vector<Block*> bblocks(ndevs, nullptr);
        for (int devid = 0; devid < ndevs; ++devid) {
            fblocks[devid] = new Block(mid ,BlockType::Forward, 1.0, 1.0);
            micro.addBlock(fblocks[devid], devid, devid);
        }
        for (int devid = 0; devid < ndevs; ++devid) {
            bblocks[devid] = new Block(mid ,BlockType::Backward, 1.0, 1.0);
            micro.addBlock(bblocks[devid], ndevs-1-devid, ndevs+devid);
        }
        std::vector<Block*> blocks(fblocks);
        blocks.insert(blocks.end(), bblocks.begin(), bblocks.end());
        Block::addDependencies(blocks);
        micros.push_back(micro);
    }
    return micros;
}


std::vector<SchedPlan> premise_chimera(int ndevs, int nmicros) {
    /**
     * @brief chimera premise: bi-pipe
     *  f             b        f b
     *    f         b        f     b
     *      f     b        f         b
     *        f b        f             b
     */
    std::vector<SchedPlan> micros;
    if (nmicros % 2 != 0) {
        throw std::runtime_error("Expected nmicros % 2 == 0");
    }
    for (int mid = 0; mid < nmicros / 2; ++mid) {
        SchedPlan micro(ndevs, ndevs);
        std::vector<Block*> fblocks(ndevs, nullptr);
        std::vector<Block*> bblocks(ndevs, nullptr);
        for (int devid = 0; devid < ndevs; ++devid) {
            fblocks[devid] = new Block(mid ,BlockType::Forward, 1.0, 1.0);
            micro.addBlock(fblocks[devid], devid, devid);
        }
        for (int devid = 0; devid < ndevs; ++devid) {
            bblocks[devid] = new Block(mid ,BlockType::Backward, 1.0, 1.0);
            micro.addBlock(bblocks[devid], ndevs-1-devid, ndevs+devid);
        }
        std::vector<Block*> blocks(fblocks);
        blocks.insert(blocks.end(), bblocks.begin(), bblocks.end());
        Block::addDependencies(blocks);
        micros.push_back(micro);
    }
    for (int mid = 0; mid < nmicros / 2; ++mid) {
        SchedPlan micro(ndevs, ndevs);
        std::vector<Block*> fblocks(ndevs, nullptr);
        std::vector<Block*> bblocks(ndevs, nullptr);
        for (int devid = 0; devid < ndevs; ++devid) {
            fblocks[devid] = new Block(mid ,BlockType::Forward, 1.0, 1.0);
            micro.addBlock(fblocks[devid], ndevs-1-devid, devid);
        }
        for (int devid = 0; devid < ndevs; ++devid) {
            bblocks[devid] = new Block(mid ,BlockType::Backward, 1.0, 1.0);
            micro.addBlock(bblocks[devid], devid, ndevs+devid);
        }
        std::vector<Block*> blocks(fblocks);
        blocks.insert(blocks.end(), bblocks.begin(), bblocks.end());
        Block::addDependencies(blocks);
        micros.push_back(micro);
    }
    return micros;
}


std::vector<SchedPlan> premise_interleave(int ndevs, int nmicros) {
    /**
     * @brief interleave premise
     *
     * f f   f         b   b b
     * f   f f         b b   b
     * f     f f     b b     b
     * f     f   f b   b     b
     * 
     */
    std::vector<SchedPlan> micros;
    for (int mid = 0; mid < nmicros; ++mid) {
        SchedPlan micro(ndevs, ndevs);
        std::vector<Block*> fblocks(ndevs+2, nullptr);
        std::vector<Block*> bblocks(ndevs+2, nullptr);
        for (int step = 0; step < ndevs + 2; ++step) {
            if (step == 0 or step == ndevs / 2 + 1) {
                fblocks[step] = new Block(mid, BlockType::Forward, 1.0, 1.0);
                std::vector<int> all_device(ndevs);
                for (int devid = 0; devid < ndevs; ++devid) {
                    all_device[devid] = devid;
                }
                micro.addBlock(fblocks[step], all_device, step);
                bblocks[step] = new Block(mid, BlockType::Backward, 1.0, 1.0);
                micro.addBlock(bblocks[step], all_device, (ndevs+2)*2-1-step);
            }
            else {
                int dev = (step < ndevs + 1) ? step - 1 : step - 2;
                fblocks[step] = new Block(mid, BlockType::Forward, 1.0, 1.0);
                micro.addBlock(fblocks[step], dev, step);
                bblocks[step] = new Block(mid, BlockType::Backward, 1.0, 1.0);
                micro.addBlock(bblocks[step], dev, (ndevs+2)*2-1-step);
            }
        }
        std::vector<Block*> blocks(fblocks);
        blocks.insert(blocks.end(), bblocks.begin(), bblocks.end());
        Block::addDependencies(blocks);
        micros.push_back(micro);
    }
    return micros;
}


void search(int ndevs, int nmicros, float dev_memory) {

    CpuTimer timer;

    // std::vector<SchedPlan> micros = premise_vshape(ndevs, nmicros);
    // std::vector<SchedPlan> micros = premise_chimera(ndevs, nmicros);
    std::vector<SchedPlan> micros = premise_interleave(ndevs, nmicros);
    for (int mid = 0; mid < nmicros; ++mid) {
        std::cout << "Premise Micro ID# " << mid << ":\n";
        std::cout << micros[mid] << std::endl;
    }
    std::vector<float> memory(ndevs, dev_memory);

    // step optimal search
    timer.start();
    std::vector<SchedPlan> opt_plans = Composer::stepOptimal(micros, memory);
    timer.stop();
    std::cout << "step-optimal search time: "
              << float(timer.elapsed()) / 1000 
              << " seconds" << std::endl;

    if (opt_plans.size() == 0) {
        std::cout << "no solution" << std::endl;
    }
    else {
        std::cout << "one solution:\n" << opt_plans[0] << std::endl;
    }
}


int main() {

    search(4, 4, 10.0);
    return 0;
}