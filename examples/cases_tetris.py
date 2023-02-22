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
    def interlace(ndevs: int, nmicros: int) -> SchedPlan:
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
                        choices=['vshape', 'interlace', 'mshape', 'chimera', 'alphafold', 'two_tower', 'custom'])
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

    if args.save is not None:
        for idx, schedule in enumerate(schedules):
            # now = datetime.datetime.now()
            now = time.strftime("%Y-%m-%d-%H-%M", time.gmtime())
            # now = f"{now.year}{now.month}{now.day}-{now.hour}{now.minute}"
            Painter.visualize(
                schedule,
                os.path.join(args.save, f"{args.premise}-{idx}.{now}.png"))
