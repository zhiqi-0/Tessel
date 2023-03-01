"""
mkdir -p figures

PYTHONPATH=.:$PYTHONPATH python examples/cases_tetris.py \
    --premise vshape --ndevs 4 --nmicros 4 --memory 4 --save figures \
    > figures/vshape-4dev-4micro-4mem.log

PYTHONPATH=.:$PYTHONPATH python examples/cases_tetris.py \
    --premise chimera --ndevs 4 --nmicros 3 --memory 4 --save figures \
    > figures/chimera-4dev-3micro-4mem.log

PYTHONPATH=.:$PYTHONPATH python examples/cases_tetris.py \
    --premise interlace_s2s --ndevs 4 --nmicros 4 --memory 10 --save figures \
    > figures/interlace_s2s-4dev-4micro-10mem.log

PYTHONPATH=.:$PYTHONPATH python examples/cases_tetris.py \
    --premise interlace_mlm --ndevs 4 --nmicros 4 --memory 10 --save figures \
    > figures/interlace_mlm-4dev-4micro-10mem.log

PYTHONPATH=.:$PYTHONPATH python examples/cases_tetris.py \
    --premise finetune --ndevs 4 --nmicros 4 --memory 10 --save figures \
    > figures/finetune-4dev-4micro-10mem.log
"""
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
    def vshape(ndevs: int) -> SchedPlan:
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
        return sched

    @staticmethod
    def chimera(ndevs: int) -> SchedPlan:
        """
        f     f b     b
          f f     b b  
          f f     b b  
        f     f b     b
        """
        sched = SchedPlan(ndevs)
        # v
        fblocks = [Block(0, span=1, memory=1, btype=FW) for _ in range(ndevs)]
        fdevs = [[devid] for devid in range(ndevs)]
        bblocks = [Block(0, span=1, memory=-1, btype=BW) for _ in range(ndevs)]
        bdevs = [[devid] for devid in range(ndevs)][::-1]
        vblocks = fblocks + bblocks
        vdevs = fdevs + bdevs
        # ^
        fblocks = [Block(0, span=1, memory=1, btype=FW) for _ in range(ndevs)]
        fdevs = [[devid] for devid in range(ndevs)][::-1]
        bblocks = [Block(0, span=1, memory=-1, btype=BW) for _ in range(ndevs)]
        bdevs = [[devid] for devid in range(ndevs)]
        rvblocks = fblocks + bblocks
        rvdevs = fdevs + bdevs

        sched.add_block_seq(vblocks, vdevs)
        sched.add_block_seq(rvblocks, rvdevs)
        return sched

    @staticmethod
    def interlace_s2s(ndevs: int) -> SchedPlan:
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
        return sched
    
    @staticmethod
    def interlace_mlm(ndevs: int) -> SchedPlan:
        """
        f f       f b       b b
        f   f     f b     b   b
        f     f   f b   b     b
        f       f f b b       b
        """
        sched = SchedPlan(ndevs)
        # 
        fblocks = [Block(0, span=1, memory=1, btype=FW) for _ in range(ndevs)]
        fdevs = [[devid] for devid in range(ndevs)]
        bblocks = [Block(0, span=2, memory=-1, btype=BW) for _ in range(ndevs)]
        bdevs = [[ndevs-1-devid] for devid in range(ndevs)]
        #
        fblocks.insert(len(fblocks), Block(0, span=1, memory=1, btype=FW))
        fdevs.insert(len(fdevs), list(range(ndevs)))
        bblocks.insert(0, Block(0, span=2, memory=-1, btype=BW))
        bdevs.insert(0, list(range(ndevs)))
        #
        fblocks.insert(0, Block(0, span=1, memory=1, btype=FW))
        fdevs.insert(0, list(range(ndevs)))
        bblocks.insert(len(bblocks), Block(0, span=2, memory=-1, btype=BW))
        bdevs.insert(len(bblocks), list(range(ndevs)))

        blocks = fblocks + bblocks
        devs = fdevs + bdevs
        sched.add_block_seq(blocks, devs)
        return sched
    
    @staticmethod
    def two_tower(ndevs: int) -> SchedPlan:
        """
        f   f b-b b-b
        f   f b-b b-b
        f   f b-b     b-b
          f f b-b b-b
        """
        sched = SchedPlan(ndevs)
        # full spmd on loss:
        loss = [Block(0, span=1, memory=1, btype=FW), Block(0, span=1, memory=-1, btype=BW)]
        loss_devs = [list(range(ndevs)), list(range(ndevs))]
        # spmd
        ndevs1 = ndevs // 2
        fbranch1 = [Block(0, span=1, memory=1, btype=FW)]
        fdevs = [list(range(ndevs1))]
        bbranch1 = [Block(0, span=2, memory=-1, btype=BW)]
        bdevs = [list(range(ndevs1))]
        branch1 = fbranch1 + [None] * (ndevs-ndevs1-1) + loss + bbranch1
        branch1_devs = fdevs + [None] * (ndevs-ndevs1-1) + loss_devs + bdevs

        # vshape
        ndevs2 = ndevs - ndevs1
        fblocks = [Block(0, span=1, memory=1, btype=FW) for _ in range(ndevs2)]
        fdevs = [[devid+ndevs1] for devid in range(ndevs2)]
        bblocks = [Block(0, span=2, memory=-1, btype=BW) for _ in range(ndevs2)]
        bdevs = [[devid+ndevs1] for devid in range(ndevs2)][::-1]
        branch2 = fblocks + loss + bblocks
        branch2_devs = fdevs + loss_devs + bdevs

        print(branch1)
        print(branch2)
        sched.add_block_seq(branch2, branch2_devs)
        sched.add_block_seq(branch1, branch1_devs)
        return sched

    @staticmethod
    def finetune(ndevs: int) -> SchedPlan:
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
        return sched
    
    @staticmethod
    def yshape(ndevs: int) -> SchedPlan:
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

    parser = argparse.ArgumentParser(description='Tetris Cases')
    parser.add_argument('--premise', type=str,
                        choices=['vshape', 'chimera', 'interlace_s2s', 'interlace_mlm', 'finetune', 'two_tower'])
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
    micro: SchedPlan = premise(args.ndevs)
    for gid, blk in enumerate(micro.chain_blocks()):
        blk.gid = gid

    micros = [micro.copy(mid) for mid in range(args.nmicros)]
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
        print(f'Unrolled schedule:\n{schedule.unroll(args.nmicros + 2)}')

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
