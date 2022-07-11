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
            bblocks[devid] = new Block(mid ,BlockType::Bakward, 1.0, 1.0);
            micro.addBlock(bblocks[devid], ndevs-1-devid, ndevs+devid);
        }
        std::vector<Block*> blocks(fblocks);
        blocks.insert(blocks.end(), bblocks.begin(), bblocks.end());
        Block::addDependencies(blocks);
        std::cout << micro << std::endl;
        micros.push_back(micro);
    }
    return micros;
}


void search(int ndevs, int nmicros, float dev_memory) {
    std::vector<SchedPlan> micros = premise_vshape(ndevs, nmicros);
    std::cout << micros.size();
    for (int mid = 0; mid < nmicros; ++mid) {
        std::cout << "Premise Micro ID# " << mid << ":\n";
        std::cout << micros[mid] << std::endl;
    }
    std::vector<float> memory(ndevs, dev_memory);

    // step optimal search
    std::vector<SchedPlan> opt_plans = Composer::stepOptimal(micros, memory);
    if (opt_plans.size() == 0) {
        std::cout << "no solution" << std::endl;
    }
    else {
        std::cout << "one solution:\n" << opt_plans[0] << std::endl;
    }
}


int main() {

    search(4, 4, 4.0);
    return 0;
}