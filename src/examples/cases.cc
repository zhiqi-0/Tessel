/**
 * @file cases.cc
 * @author your name (zhiqi.0@outlook.com)
 * @brief Case search with premise
 * @version 0.1
 * @date 2022-07-10
 * 
 * @copyright Copyright (c) 2022
 * 
 * ./build/cases --premise vshape --ndevs 4 --nmicros 4 --memory 4 --nworkers 1
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


class Premise {

 public:

    static std::vector<SchedPlan> vshape(int ndevs, int nmicros) {
        std::vector<SchedPlan> micros;
        for (int mid = 0; mid < nmicros; ++mid) {
            SchedPlan micro(ndevs, ndevs);
            std::vector<Block*> blocks(ndevs * 2, nullptr);
            std::vector<int> devs(ndevs * 2, -1);
            for (int devid = 0; devid < ndevs; ++devid) {
                blocks[devid] = new Block(mid, BlockType::Forward, 1.0, 1);
                devs[devid] = devid;
                blocks[2*ndevs-1-devid] = new Block(mid, BlockType::Backward, 1.0, 2);
                devs[2*ndevs-1-devid] = devid;
            }
            micro.addBlockSeq(blocks, devs);
            Block::addDependencies(blocks);
            micros.push_back(micro);
        }
        return micros;
    }

    static std::vector<SchedPlan> chimera(int ndevs, int nmicros) {
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
            std::vector<Block*> blocks(ndevs * 2, nullptr);
            std::vector<int> devs(ndevs * 2, -1);
            for (int devid = 0; devid < ndevs; ++devid) {
                blocks[devid] = new Block(mid, BlockType::Forward, 1.0, 1);
                devs[devid] = devid;
                blocks[2*ndevs-1-devid] = new Block(mid, BlockType::Backward, 1.0, 2);
                devs[2*ndevs-1-devid] = devid;
            }
            micro.addBlockSeq(blocks, devs);
            Block::addDependencies(blocks);
            micros.push_back(micro);
        }
        for (int mid = nmicros / 2; mid < nmicros; ++mid) {
            SchedPlan micro(ndevs, ndevs);
            std::vector<Block*> blocks(ndevs * 2, nullptr);
            std::vector<int> devs(ndevs * 2, -1);
            for (int idx = 0; idx < ndevs; ++idx) {
                blocks[idx] = new Block(mid, BlockType::Forward, 1.0, 1);
                devs[idx] = ndevs-1-idx;
                blocks[2*ndevs-1-idx] = new Block(mid, BlockType::Backward, 1.0, 2);
                devs[2*ndevs-1-idx] = ndevs-1-idx;
            }
            micro.addBlockSeq(blocks, devs);
            Block::addDependencies(blocks);
            micros.push_back(micro);
        }
        return micros;
    }

    static std::vector<SchedPlan> twotower(int ndevs, int nmicros){
        /**
         * @brief two tower model.
         * Replicated at last forward of each branch
         * 
         * f     f b     b
         *   f f     b b
         *   f f     b b
         * f     f b     b
         *
         */
        Plans micros;
        for (int mid = 0; mid < nmicros; ++mid) {
            SchedPlan micro(ndevs, ndevs * 3);
            std::vector<Block*> blocks_branch1(ndevs * 2, nullptr);
            std::vector<int> devs_branch1(ndevs * 2, -1);
            std::vector<Block*> blocks_branch2(ndevs * 2, nullptr);
            std::vector<int> devs_branch2(ndevs * 2, -1);
            for (int idx = 0; idx < ndevs; ++idx) {
                blocks_branch1[idx] = new Block(mid, BlockType::Forward, 1.0, 1);
                devs_branch1[idx] = idx;
                blocks_branch1[2*ndevs-1-idx] = new Block(mid, BlockType::Backward, 1.0, 2);
                devs_branch1[2*ndevs-1-idx] = idx;

                blocks_branch2[idx] = new Block(mid, BlockType::Forward, 1.0, 1);
                devs_branch2[idx] = ndevs-1-idx;
                blocks_branch2[2*ndevs-1-idx] = new Block(mid, BlockType::Backward, 1.0, 2);
                devs_branch2[2*ndevs-1-idx] = ndevs-1-idx;
            }
            micro.addBlockSeq(blocks_branch1, devs_branch1);
            micro.addBlockSeq(blocks_branch2, devs_branch2);
            Block::addDependencies(blocks_branch1);
            Block::addDependencies(blocks_branch2);
            Block::addDependency(
                *(blocks_branch2.end()-ndevs-2), *(blocks_branch1.end()-ndevs-1)
            );
            Block::addDependency(
                *(blocks_branch2.end()-ndevs+1), *(blocks_branch1.end()-ndevs+2)
            );
            Block::addDependency(
                *(blocks_branch1.end()-ndevs-2), *(blocks_branch2.end()-ndevs-1)
            );
            Block::addDependency(
                *(blocks_branch1.end()-ndevs+1), *(blocks_branch2.end()-ndevs+2)
            );
            micros.push_back(micro);
        }
        return micros;
    }

    static std::vector<SchedPlan> vlmodel(int ndevs, int nmicros){
        /**
         * @brief vision-language two tower model.
         * Replicated at last forward of each branch
         * 
         * f   f b   b
         *   f f b b 
         *   f f b b 
         * f   f b   b
         *
         */
        Plans micros;
        for (int mid = 0; mid < nmicros; ++mid) {
            SchedPlan micro(ndevs, ndevs * 3);
            
            // text encoder
            std::vector<Block*> te_blks(ndevs, nullptr);
            std::vector<int> te_devs(ndevs, -1);
            for (int idx = 0; idx < ndevs / 2; ++idx) {
                te_blks[idx] = new Block(mid, BlockType::Forward, 1.0, 1);
                te_devs[idx] = idx;
                te_blks[ndevs-1-idx] = new Block(mid, BlockType::Backward, 1.0, 2);
                te_devs[ndevs-1-idx] = idx;
            }

            // visual encoder
            std::vector<Block*> ve_blks(ndevs, nullptr);
            std::vector<int> ve_devs(ndevs, -1);
            for (int idx = 0; idx < ndevs / 2; ++idx) {
                ve_blks[idx] = new Block(mid, BlockType::Forward, 1.0, 1);
                ve_devs[idx] = ndevs - 1 - idx;
                ve_blks[ndevs-1-idx] = new Block(mid, BlockType::Backward, 1.0, 2);
                ve_devs[ndevs-1-idx] = ndevs - 1 - idx;
            }

            // cross-modal encoder
            Block* fce = new Block(mid, BlockType::Forward, 1.0, 1);
            Block* bce = new Block(mid, BlockType::Backward, 1.0, 2);
            std::vector<int> ce_devs(ndevs);
            for (int devid = 0; devid < ndevs; ++devid) {
                ce_devs[devid] = devid;
            }

            int step = 0;
            // add te / ve forward
            for (std::size_t idx = 0; idx < ve_blks.size() / 2; ++idx) {
                micro.addBlock(te_blks[idx], te_devs[idx], step);
                micro.addBlock(ve_blks[idx], ve_devs[idx], step);
                step += te_blks[idx]->span;
            }
            // ad ce forward + backward
            micro.addBlock(fce, ce_devs, step);
            step += fce->span;
            micro.addBlock(bce, ce_devs, step);
            step += bce->span;
            // add te / ve backward
            for (std::size_t idx = 0; idx < ve_blks.size() / 2; ++idx) {
                micro.addBlock(te_blks[ndevs / 2 + idx], te_devs[ndevs / 2 + idx], step);
                micro.addBlock(ve_blks[ndevs / 2 + idx], ve_devs[ndevs / 2 + idx], step);
                step += te_blks[ndevs / 2 + idx]->span;
            }

            // dependency
            std::vector<Block*> ve_fblks(ve_blks.begin(), ve_blks.begin() + ndevs / 2);
            std::vector<Block*> ve_bblks(ve_blks.begin() + ndevs / 2, ve_blks.end());
            std::vector<Block*> te_fblks(te_blks.begin(), te_blks.begin() + ndevs / 2);
            std::vector<Block*> te_bblks(te_blks.begin() + ndevs / 2, te_blks.end());
            
            Block::addDependencies(ve_fblks);
            Block::addDependencies(te_fblks);
            Block::addDependency(ve_blks[ndevs/2-1], fce);
            Block::addDependency(te_blks[ndevs/2-1], fce);
            Block::addDependency(fce, bce);
            Block::addDependency(bce, ve_blks[ndevs/2]);
            Block::addDependency(bce, te_blks[ndevs/2]);
            Block::addDependencies(ve_bblks);
            Block::addDependencies(te_bblks);
            micros.push_back(micro);
        }
        return micros;
    }

    static std::vector<SchedPlan> alphafold(int ndevs, int nmicros) {
        int repeat_forward = 3;
        Plans micros;
        for (int mid = 0; mid < nmicros; ++mid) {
            SchedPlan micro(ndevs, ndevs * (repeat_forward + 1));
            std::vector<Block*> blocks(ndevs * (repeat_forward + 1), nullptr);
            std::vector<int> devs(ndevs * (repeat_forward + 1), -1);
            // add forward blocks
            for (int rid = 0; rid < repeat_forward; ++rid) {
                for (int idx = 0; idx < ndevs; ++idx) {
                    int ofst = rid * ndevs;
                    float memory = rid == repeat_forward - 1 ? 1.0 : 0.0;
                    blocks[ofst + idx] = new Block(mid, BlockType::Forward, memory, 1);
                    devs[ofst + idx] = idx;
                }
            }
            // add backward blocks
            int ofst = repeat_forward * ndevs;
            for (int idx = 0; idx < ndevs; ++idx) {
                blocks[ofst + idx] = new Block(mid, BlockType::Backward, 1.0, 1);
                devs[ofst + idx] = ndevs - 1- idx;
            }
            micro.addBlockSeq(blocks, devs);
            Block::addDependencies(blocks);
            micros.push_back(micro);
        }
        return micros;
    }

    static std::vector<SchedPlan> interleave(int ndevs, int nmicros) {
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
            std::vector<Block*> blocks(ndevs*2, nullptr);
            std::vector<std::vector<int>> devs(ndevs*2);
            for (int devid = 0; devid < ndevs; ++devid) {
                blocks[devid] = new Block(mid, BlockType::Forward, 1.0, 1);
                devs[devid] = std::vector<int>({devid});
                blocks[2*ndevs-1-devid] = new Block(mid, BlockType::Backward, 1.0, 1);
                devs[2*ndevs-1-devid] = std::vector<int>({devid});
            }
            //
            std::vector<int> blk_dev(ndevs, -1);
            for (int idx = 0; idx < ndevs; ++idx) {
                blk_dev[idx] = idx;
            }
            Block* fblock_full = new Block(mid, BlockType::Forward, 1.0, 1);
            Block* bblock_full = new Block(mid, BlockType::Backward, 1.0, 1);
            blocks.insert(blocks.begin(), fblock_full);
            devs.insert(devs.begin(), blk_dev);
            blocks.insert(blocks.end(), bblock_full);
            devs.insert(devs.end(), blk_dev);
            //
            fblock_full = new Block(mid, BlockType::Forward, 1.0, 1);
            bblock_full = new Block(mid, BlockType::Backward, 1.0, 1);
            blocks.insert(blocks.begin()+1+ndevs/2, fblock_full);
            devs.insert(devs.begin()+1+ndevs/2, blk_dev);
            blocks.insert(blocks.begin()+2+ndevs*2-ndevs/2, bblock_full);
            devs.insert(devs.begin()+2+ndevs*2-ndevs/2, blk_dev);
            micro.addBlockSeq(blocks, devs);
            Block::addDependencies(blocks);
            micros.push_back(micro);
        }
        return micros;
    }

};


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
    micros[0].to_json("micro1.json");
    std::vector<float> memory(ndevs, dev_memory);

    // step optimal search
    timer.start();
    // std::vector<SchedPlan> opt_plans = Composer::stepOptimal(
    //     micros, memory, true, false, -1, nworkers
    // );
    std::vector<SchedPlan> opt_plans = Composer::stepOptimalBFS(
        micros, memory, false, -1, nworkers
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
    // for (auto& sched : opt_plans) { std::cout << sched << std::endl;};
    // return

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
            std::cout << "searched " << idx + 1 << "/" << opt_plans.size() << " plans\n";
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
            std::cout << "early stop as found 0-bubble plan\n";
            break;
        }
    }

    timer.stop();
    std::cout << "best bubble-rate generalized plan:\n" << best_plan << std::endl
              << "bubble rate: " << min_bubble_rate << std::endl
              << "Generalization time: " << float(timer.elapsed()) / 1000 << " seconds\n"; 

    best_plan.to_json("general_plan.json");
}


int main(int argc, const char* argv[]) {

    std::unordered_map<std::string, std::function<PremiseFunc> > premises;
    premises.emplace(
        std::string("vshape"), std::function<PremiseFunc>(Premise::vshape)
    );
    premises.emplace(
        std::string("chimera"), std::function<PremiseFunc>(Premise::chimera)
    );
    premises.emplace(
        std::string("interleave"), std::function<PremiseFunc>(Premise::interleave)
    );
    premises.emplace(
        std::string("twotower"), std::function<PremiseFunc>(Premise::twotower)
    );
    premises.emplace(
        std::string("alphafold"), std::function<PremiseFunc>(Premise::alphafold)
    );
    premises.emplace(
        std::string("vlmodel"), std::function<PremiseFunc>(Premise::vlmodel)
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