"""
PYTHONPATH=.:$PYTHONPATH python examples/cases_tetris.py \
    --premise vshape --ndevs 4 --nmicros 4 --memory 4 --save figures > figures/log
"""
from typing import List
import sys
import time
import argparse
import os

from tetris.schedplan import SchedPlan, Block
from tetris.composer import Composer
from tetris.draw import Painter


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
        sched = SchedPlan(ndevs)
        fblocks = [Block(0, span=1, memory=1, btype=FW) for _ in range(ndevs)]
        fdevs = [[devid] for devid in range(ndevs)]
        bblocks = [Block(0, span=2, memory=-1, btype=BW) for _ in range(ndevs)]
        bdevs = [[devid] for devid in range(ndevs)][::-1]
        blocks = fblocks + bblocks
        devs = fdevs + bdevs
        sched.add_block_seq(blocks, devs)

        scheds = [sched.copy(mid) for mid in range(nmicros)]
        return scheds

    @staticmethod
    def interlace(ndevs: int, nmicros: int) -> SchedPlan:
        """
        f f   f         b   b b
        f   f f         b b   b
        f     f f     b b     b
        f     f   f b   b     b
        """
        sched = SchedPlan(ndevs)
        # 
        fblocks = [Block(0, span=1, memory=1, btype=FW) for _ in range(ndevs)]
        fdevs = [[devid] for devid in range(ndevs)]
        bblocks = [Block(0, span=2, memory=-1, btype=BW) for _ in range(ndevs)]
        bdevs = [[ndevs-1-devid] for devid in range(ndevs)]
        #
        fblocks.insert(ndevs // 2, Block(0, span=1, memory=1, btype=FW))
        fdevs.insert(ndevs // 2, list(range(ndevs)))
        bblocks.insert(ndevs // 2, Block(0, span=2, memory=-1, btype=BW))
        bdevs.insert(ndevs // 2, list(range(ndevs)))
        #
        fblocks.insert(0, Block(0, span=1, memory=1, btype=FW))
        fdevs.insert(0, list(range(ndevs)))
        bblocks.insert(len(bblocks), Block(0, span=2, memory=-1, btype=BW))
        bdevs.insert(len(bblocks), list(range(ndevs)))

        blocks = fblocks + bblocks
        devs = fdevs + bdevs
        sched.add_block_seq(blocks, devs)

        scheds = [sched.copy(mid) for mid in range(nmicros)]
        return scheds
    
    @staticmethod
    def finetune(ndevs: int, nmicros: int) -> SchedPlan:
        """
        f f             b
        f   f         b  
        f     f     b    
        f       f b      
        """
        sched = SchedPlan(ndevs)

        fblocks = [Block(0, span=1, memory=1, btype=FW) for _ in range(ndevs)]
        fdevs = [[devid] for devid in range(ndevs)]
            
        fblocks.insert(0, Block(0, span=1, memory=0, btype=FW))
        fdevs.insert(0, list(range(ndevs)))
            
        bblocks = [Block(0, span=2, memory=-1, btype=BW) for _ in range(ndevs)]
        bdevs = [[devid] for devid in range(ndevs)][::-1]
            
        blocks = fblocks + bblocks
        devs = fdevs + bdevs
        sched.add_block_seq(blocks, devs)

        scheds = [sched.copy(mid) for mid in range(nmicros)]
        return scheds
    
    @staticmethod
    def yshape(ndevs: int, nmicros: int) -> SchedPlan:
        """
        f   f b   b
          f f b b
          f f b b
        f   f b   b

        f f         b b
            f f b b
            f f b b
        f f         b b
        """
        pass
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='comm primitive')
    parser.add_argument('--premise', type=str,
                        choices=['vshape', 'interlace', 'finetune'])
    parser.add_argument('--ndevs', type=int,
                        help='number of devices')
    parser.add_argument('--nmicros', type=int,
                        help='number of micro-batches')
    parser.add_argument('--memory', type=int,
                        help='memory limits')
    parser.add_argument('--save', type=str, default=None,
                        help='save searched schedule under a folder')
    args = parser.parse_args()

    print('============== Scheduling Solver ================')
    print(args)
    sys.stdout.flush()

    premise = getattr(Premise, args.premise)
    micros = premise(args.ndevs, args.nmicros)
    memory = [args.memory] * args.ndevs

    print(f'Premise: {args.ndevs} devices, {args.nmicros} micro-batches')
    print(micros[0])

    tic = time.time()
    schedules = Composer.compose(micros, memory)
    toc = time.time()

    print('\n================================================')
    print('search time: {:.2f} seconds'.format(toc-tic))

    for idx, schedule in enumerate(schedules):
        print(f'schedule-{idx}:\n{schedule}')

    if args.save is not None:
        now = time.strftime("%Y-%m-%d-%H-%M", time.gmtime())
        Painter.visualize(
            micros[0],
            os.path.join(args.save, f"{args.premise}-premise.{now}.png"))
        repetends = [sched.extract(sched.repetend[0], sched.repetend[1]) for sched in schedules]
        for idx, (schedule, repetend) in enumerate(zip(schedules, repetends)):
            Painter.visualize(
                schedule,
                os.path.join(args.save, f"{args.premise}-schedule-{idx}.{now}.png"))
            Painter.visualize(
                repetends[idx], 
                os.path.join(args.save, f"{args.premise}-repetend-{idx}.{now}.png"))
