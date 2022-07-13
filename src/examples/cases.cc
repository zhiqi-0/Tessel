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
#include <functional>
#include <chrono>

#include <schedplan.h>
#include <composer.h>
#include <generalizer.h>

#include <parser.h>


typedef Plans (PremiseFunc)(int, int);


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
                bblocks[ndevs+1-step] = new Block(mid, BlockType::Backward, 1.0, 1.0);
                micro.addBlock(bblocks[ndevs+1-step], all_device, (ndevs+2)*2-1-step);
            }
            else {
                int dev = (step < ndevs / 2 + 1) ? step - 1 : step - 2;
                fblocks[step] = new Block(mid, BlockType::Forward, 1.0, 1.0);
                micro.addBlock(fblocks[step], dev, step);
                bblocks[ndevs+1-step] = new Block(mid, BlockType::Backward, 1.0, 1.0);
                micro.addBlock(bblocks[ndevs+1-step], dev, (ndevs+2)*2-1-step);
            }
        }
        std::vector<Block*> blocks(fblocks);
        blocks.insert(blocks.end(), bblocks.begin(), bblocks.end());
        Block::addDependencies(blocks);
        micros.push_back(micro);
    }
    return micros;
}


void search(std::function<PremiseFunc> premise, int ndevs, int nmicros, float dev_memory, int nworkers) {

    CpuTimer timer;

    // std::vector<SchedPlan> micros = premise_vshape(ndevs, nmicros);
    // std::vector<SchedPlan> micros = premise_chimera(ndevs, nmicros);
    // std::vector<SchedPlan> micros = premise_interleave(ndevs, nmicros);
    std::vector<SchedPlan> micros = premise(ndevs, nmicros);

    for (int mid = 0; mid < nmicros; ++mid) {
        std::cout << "Premise Micro ID# " << mid << ":\n";
        std::cout << micros[mid] << std::endl;
    }
    std::vector<float> memory(ndevs, dev_memory);

    // step optimal search
    timer.start();
    std::vector<SchedPlan> opt_plans = Composer::stepOptimal(
        micros, memory, true, false, -1, nworkers
    );
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
    // for (int idx = 0; idx < opt_plans.size(); ++idx) { std::cout << "plan#" << idx << ":\n" << opt_plans[idx] << std::endl;}

    timer.start();

    GeneralSchedPlan best_plan;
    float min_bubble_rate = 1.0;
    float min_steady_step = opt_plans[0].nSteps() * 2;
    for (size_t idx = 0; idx < opt_plans.size(); ++idx) {
        // prune technique to early stop for other plan
        GeneralSchedPlan gsched = Generalizer::tailHeadHeuristic(
            opt_plans[idx], memory, min_steady_step - 1, nworkers
        );

        if ((idx+1) % 10 == 0) {
            std::cout << "searched " << idx + 1 << "/" << opt_plans.size() << "plans\n";
        }

        if (gsched.isEmpty()) {
            gsched.destroyCreatedBlocks();
            continue;
        }

        float bubble_rate = gsched.steady_bubble_rate();
        if (bubble_rate < min_bubble_rate) {
            min_bubble_rate = bubble_rate;
            min_steady_step = gsched.getRBound() - gsched.getLBound();
            std::cout << "find generalized plan with bubble rate: "
                      << min_bubble_rate << std::endl;
            std::cout << gsched << std::endl;
            best_plan.destroyCreatedBlocks();
            best_plan = gsched;
        }
        else {
            gsched.destroyCreatedBlocks();
        }
        if (min_bubble_rate == 0) {
            std::cout << "early stop as found 0-buuble plan\n";
            break;
        }
    }

    timer.stop();
    std::cout << "best bubble-rate generalized plan:\n" << best_plan << std::endl
              << "bubble rate: " << min_bubble_rate << std::endl
              << "Generalization time: " << float(timer.elapsed()) / 1000 << " seconds\n"; 
}


int main(int argc, const char* argv[]) {

    std::unordered_map<std::string, std::function<PremiseFunc> > premises;
    premises.emplace(
        std::string("vshape"), std::function<PremiseFunc>(premise_vshape)
    );
    premises.emplace(
        std::string("chimera"), std::function<PremiseFunc>(premise_chimera)
    );
    premises.emplace(
        std::string("interleave"), std::function<PremiseFunc>(premise_interleave)
    );

    CmdParser parser;
    parser.add<std::string>("--premise", "premise of micro-batch.");
    parser.add<int>("--ndevs", "number of devices.");
    parser.add<int>("--nmicros", "number of micro-batches.");
    parser.add<float>("--memory", "memory consumpition of each device.");
    parser.add<int>("--nworkers", "number of worker for omp threads");
    parser.setDefault<int>("--nworkers", 1);
    parser.parse(argc, argv);
    std::cout << parser << std::endl;

    std::string premise_str = parser.get<std::string>("premise");
    std::cout << "premise str: " << premise_str << std::endl;
    if (premises.find(premise_str) == premises.end()) {
        throw std::runtime_error("not find premise\n");
    }
    std::function<PremiseFunc> premise = premises[premise_str];
    search(
        premise,
        parser.get<int>("ndevs"),
        parser.get<int>("nmicros"),
        parser.get<float>("memory"),
        parser.get<int>("nworkers")
    );

    return 0;
}