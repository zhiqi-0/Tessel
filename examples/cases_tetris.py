"""
PYTHONPATH=.:$PYTHONPATH python examples/cases_tetris.py \
    --premise vshape --ndevs 4 --nmicros 4 --memory 4
"""
from typing import List
import sys
import time
import argparse

from tetris.schedplan import SchedPlan, Block
from tetris.solver import StepOptimalSolver
from tetris.repetend import MicroPicker
from tetris.composer import Composer


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
            bblocks = [Block(mid, span=2, memory=-1, btype=BW) for _ in range(ndevs)]
            bdevs = [[devid] for devid in range(ndevs)][::-1]
            blocks = fblocks + bblocks
            devs = fdevs + bdevs
            sched.add_block_seq(blocks, devs)
            scheds.append(sched)
        return scheds

    @staticmethod
    def chimera(ndevs: int, nmicros: int) -> SchedPlan:
        """
        f     f b     b 
          f f     b b   
          f f     b b   
        f     f b     b 
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
        scheds = []
        for mid in range(nmicros):
            sched = SchedPlan(ndevs)
            # 
            fblocks = [Block(mid, span=1, memory=1, btype='forward') for _ in range(ndevs)]
            fdevs = [[devid] for devid in range(ndevs)]
            bblocks = [Block(mid, span=2, memory=-1, btype='backward') for _ in range(ndevs)]
            bdevs = [[ndevs-1-devid] for devid in range(ndevs)]
            #
            fblocks.insert(ndevs // 2, Block(mid, span=1, memory=1, btype='forward'))
            fdevs.insert(ndevs // 2, list(range(ndevs)))
            bblocks.insert(ndevs // 2, Block(mid, span=2, memory=-1, btype='backward'))
            bdevs.insert(ndevs // 2, list(range(ndevs)))
            # 
            fblocks.insert(0, Block(mid, span=1, memory=1, btype='forward'))
            fdevs.insert(0, list(range(ndevs)))
            bblocks.insert(len(bblocks), Block(mid, span=2, memory=-1, btype='backward'))
            bdevs.insert(len(bblocks), list(range(ndevs)))

            blocks = fblocks + bblocks
            devs = fdevs + bdevs
            sched.add_block_seq(blocks, devs)
            scheds.append(sched)
        return scheds
    

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

    print(f'Premise: {args.ndevs} devices, {args.nmicros} micro-batches')
    print(micros[0])

    Composer.compose(micros, memory)

    # nsteps = micros[0].nsteps
    # for block_and_devs in MicroPicker.pick(micros):
    #     solver = StepOptimalSolver(args.ndevs)
    #     # add block inside solver
    #     for block, devs in block_and_devs:
    #         solver.add_block(block, devs, 0)
    #     # setup dependency
    #     blocks = [block_devs[0] for block_devs in block_and_devs]
    #     for idx, blk1 in enumerate(blocks):
    #         for blk2 in blocks[idx+1:]:
    #             if blk2 in blk1.after:
    #                 solver.add_dependency([blk1, blk2])
    #     case_nsteps = solver.time_optimal(memory, nsteps)
    #     if case_nsteps is not None:
    #         nsteps = case_nsteps
    #         print('find one solution:')
    #         for schedplan in solver.solutions():
    #             print(schedplan)
    #             break
    #     else:
    #         print("Fail to find a smaller-step solution")
        
        # break


    # step-optimal search
    # solver = StepOptimalSolver(args.ndevs)
    # for micro in micros: 
    #     print('adding micro:')
    #     print(micro)
    #     solver.add_micro_plan(micro)
    # tic = time.time()
    # solver.time_optimal(memory)
    # toc = time.time()
    # print('search time for step optimal span: {:.2f} seconds'.format(toc-tic))
    # 
    # print('find one solution:')
    # for schedplan in solver.solutions():
    #     print(schedplan)
    #     break
