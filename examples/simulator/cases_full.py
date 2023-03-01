"""
PYTHONPATH=.:$PYTHONPATH python examples/cases.py \
    --premise vshape --ndevs 4 --nmicros 4 --memory 4
"""
from typing import List
import sys
import time
import argparse

from tetris.schedplan import SchedPlan, Block
from tetris.solver import StepOptimalSolver


FW='forward'
BW='backward'


class Premise:

    @staticmethod
    def vshape(ndevs: int, nmicros: int) -> List[SchedPlan]:
        """
        f             b
          f         b  
            f     b    
              f b      
        """
        scheds = []
        for mid in range(nmicros):
            sched = SchedPlan(ndevs)
            fblocks = [Block(mid, span=1, memory=1, btype=FW) for _ in range(ndevs)]
            fdevs = [[devid] for devid in range(ndevs)]
            bblocks = [Block(mid, span=1, memory=-1, btype=BW) for _ in range(ndevs)]
            bdevs = [[devid] for devid in range(ndevs)][::-1]
            blocks = fblocks + bblocks
            devs = fdevs + bdevs
            sched.add_block_seq(blocks, devs)
            scheds.append(sched)
        return scheds

    @staticmethod
    def chimera(ndevs: int, nmicros: int) -> SchedPlan:
        """
        f             b        f b
          f         b        f     b
            f     b        f         b
              f b        f             b
        """
        sched = SchedPlan(ndevs)
        assert nmicros % 2 == 0, "require microbatch# can be devided by 2"
        for mid in range(nmicros // 2): # V shape
            blocks = [None] * ndevs * 2
            devs = [None] * ndevs * 2
            for devid in range(ndevs):
                blocks[devid] = Block(mid, Block.BType.FW, span=1)
                devs[devid] = [devid]
                blocks[-1-devid] = Block(mid, Block.BType.BW, span=2)
                devs[-1-devid] = [devid]
            sched.add_block_seq(blocks, devs)
            sched.add_dependency(blocks)
        for mid in range(nmicros // 2, nmicros): # ^ shape
            blocks = [None] * ndevs * 2
            devs = [None] * ndevs * 2
            for devid in range(ndevs):
                blocks[devid] = Block(mid, Block.BType.FW, span=1)
                devs[devid] = [ndevs-1-devid]
                blocks[-1-devid] = Block(mid, Block.BType.BW, span=2)
                devs[-1-devid] = [ndevs-1-devid]
            sched.add_block_seq(blocks, devs)
            sched.add_dependency(blocks)
        return sched

    @staticmethod
    def interleave(ndevs: int, nmicros: int) -> SchedPlan:
        """
        f f   f         b   b b
        f   f f         b b   b
        f     f f     b b     b
        f     f   f b   b     b
        """
        sched = SchedPlan(ndevs)
        for mid in range(nmicros):
            fblocks = []
            bblocks = []
            fblocks = [Block(mid, Block.BType.FW, span=1) for _ in range(ndevs)]
            fdevs = [[devid] for devid in range(ndevs)]
            bblocks = [Block(mid, Block.BType.BW, span=2) for _ in range(ndevs)]
            bdevs = [[ndevs-1-devid] for devid in range(ndevs)]
            #
            fblocks.insert(ndevs // 2, Block(mid, Block.BType.FW, span=1))
            fdevs.insert(ndevs // 2, list(range(ndevs)))
            bblocks.insert(ndevs // 2, Block(mid, Block.BType.BW, span=2))
            bdevs.insert(ndevs // 2, list(range(ndevs)))
            # 
            fblocks.insert(0, Block(mid, Block.BType.FW, span=1))
            fdevs.insert(0, list(range(ndevs)))
            bblocks.insert(len(bblocks), Block(mid, Block.BType.BW, span=2))
            bdevs.insert(len(bblocks), list(range(ndevs)))

            blocks = fblocks + bblocks
            devs = fdevs + bdevs
            sched.add_block_seq(blocks, devs)
            sched.add_dependency(blocks)
        return sched
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='comm primitive')
    parser.add_argument('--premise', type=str,
                        choices=['vshape', 'interleave', 'mshape', 'chimera', 'alphafold', 'two_tower', 'custom'])
    parser.add_argument('--ndevs', type=int,
                        help='number of devices')
    parser.add_argument('--nmicros', type=int,
                        help='number of micro-batches')
    parser.add_argument('--memory', type=int,
                        help='memory limits')
    args = parser.parse_args()

    print('============== Scheduling Solver ================')
    print(args)
    sys.stdout.flush()

    premise = getattr(Premise, args.premise)
    micros = premise(args.ndevs, args.nmicros)
    memory = [args.memory] * args.ndevs

    # step-optimal search
    solver = StepOptimalSolver(args.ndevs)
    for micro in micros: 
        print('adding micro:')
        print(micro)
        solver.add_micro_plan(micro)
    tic = time.time()
    solver.time_optimal(memory)
    toc = time.time()
    print('search time for step optimal span: {:.2f} seconds'.format(toc-tic))

    print('find one solution:')
    for schedplan in solver.solutions():
        print(schedplan)
        break
