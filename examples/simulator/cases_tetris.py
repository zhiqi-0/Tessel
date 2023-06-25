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
    def vshape(ndevs: int, train: bool = True) -> SchedPlan:
        """
        f             b
          f         b  
            f     b    
              f b      
        """
        sched = SchedPlan(ndevs)
        fblocks = [Block(0, span=1, memory=1, btype=FW) for _ in range(ndevs)]
        fdevs = [[devid] for devid in range(ndevs)]
        if train:
            bblocks = [Block(0, span=3, memory=-1, btype=BW) for _ in range(ndevs)]
            bdevs = [[devid] for devid in range(ndevs)][::-1]
        else:
            bblocks, bdevs = [], []
        blocks = fblocks + bblocks
        devs = fdevs + bdevs
        sched.add_block_seq(blocks, devs)
        return sched

    @staticmethod
    def xshape(ndevs: int, train: bool = True) -> SchedPlan:
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
        if train:
            bblocks = [Block(0, span=3, memory=-1, btype=BW) for _ in range(ndevs)]
            bdevs = [[devid] for devid in range(ndevs)][::-1]
        else:
            bblocks, bdevs = [], []
        vblocks = fblocks + bblocks
        vdevs = fdevs + bdevs
        # ^
        fblocks = [Block(0, span=1, memory=1, btype=FW) for _ in range(ndevs)]
        fdevs = [[devid] for devid in range(ndevs)][::-1]
        if train:
            bblocks = [Block(0, span=3, memory=-1, btype=BW) for _ in range(ndevs)]
            bdevs = [[devid] for devid in range(ndevs)]
        else:
            bblocks, bdevs = [], []
        rvblocks = fblocks + bblocks
        rvdevs = fdevs + bdevs

        sched.add_block_seq(vblocks, vdevs)
        sched.add_block_seq(rvblocks, rvdevs)
        return sched

    @staticmethod
    def nnshape(ndevs: int, train: bool = True) -> SchedPlan:
        """
        f f   f         b   b b
        f   f f         b b   b
        f     f f     b b     b
        f     f   f b   b     b
        """
        sched = SchedPlan(ndevs)
        # v-shape
        fblocks = [Block(0, span=1, memory=1, btype=FW) for _ in range(ndevs)]
        fdevs = [[devid] for devid in range(ndevs)]
        bblocks, bdevs = [], []
        if train:
            bblocks = [Block(0, span=3, memory=-1, btype=BW) for _ in range(ndevs)]
            bdevs = [[ndevs-1-devid] for devid in range(ndevs)]
        # full shard 2
        fblocks.insert(ndevs // 2, Block(0, span=1, memory=0, btype=FW))
        fdevs.insert(ndevs // 2, list(range(ndevs)))
        if train:
            bblocks.insert(ndevs // 2, Block(0, span=1, memory=0, btype=BW))
            bdevs.insert(ndevs // 2, list(range(ndevs)))
        # full shard 1
        fblocks.insert(0, Block(0, span=1, memory=0, btype=FW))
        fdevs.insert(0, list(range(ndevs)))
        if train:
            bblocks.insert(len(bblocks), Block(0, span=1, memory=0, btype=BW))
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
    def mshape(ndevs: int, train: bool = True) -> SchedPlan:
        """
        f f             b b
        f   f         b   b
        f     f     b     b
        f       f b       b
        """
        sched = SchedPlan(ndevs)
        # 
        fblocks = [Block(0, span=1, memory=1, btype=FW) for _ in range(ndevs)]
        fdevs = [[devid] for devid in range(ndevs)]
        if train:
            bblocks = [Block(0, span=3, memory=-1, btype=BW) for _ in range(ndevs)]
            bdevs = [[ndevs-1-devid] for devid in range(ndevs)]
        else:
            bblocks, bdevs = [], []
        # fully shard
        fblocks.insert(0, Block(0, span=1, memory=0, btype=FW))
        fdevs.insert(0, list(range(ndevs)))
        if train:
            bblocks.insert(len(bblocks), Block(0, span=1, memory=0, btype=BW))
            bdevs.insert(len(bblocks), list(range(ndevs)))
    
        blocks = fblocks + bblocks
        devs = fdevs + bdevs
        sched.add_block_seq(blocks, devs)
        return sched

    @staticmethod
    def kshape(ndevs: int, train: bool = True) -> SchedPlan:
        """
        f   f b   b
          f f b b  
        f   f b   b
          f f b b
        """
        sched = SchedPlan(ndevs)
        # full spmd on loss:
        mm = [Block(0, span=1, memory=1, btype=FW)]
        mm_devs = [list(range(ndevs))]
        if train:
            mm += [Block(0, span=1, memory=-1, btype=BW)]
            mm_devs += [list(range(ndevs))]

        # vshape on branch 1
        ndevs1 = ndevs // 2
        fbranch1 = [Block(0, span=1, memory=1, btype=FW) for _ in range(ndevs1)]
        fdevs = [[devid] for devid in range(ndevs1)]
        bbranch1, bdevs1 = [], []
        if train:
            bbranch1 = [Block(0, span=3, memory=-1, btype=BW) for _ in range(ndevs1)]
            bdevs1 = [[devid] for devid in range(ndevs1)][::-1]
        branch1 = fbranch1 + mm + bbranch1
        branch1_devs = fdevs + mm_devs + bdevs1

        # vshape on branch 2
        ndevs2 = ndevs - ndevs1
        fbranch2 = [Block(0, span=1, memory=1, btype=FW) for _ in range(ndevs1)]
        fdevs = [[devid] for devid in range(ndevs1, ndevs1+ndevs2)]
        bbranch2, bdevs2 = [], []
        if train:
            bbranch2 = [Block(0, span=3, memory=-1, btype=BW) for _ in range(ndevs1)]
            bdevs2 = [[devid] for devid in range(ndevs1, ndevs1+ndevs2)][::-1]
        branch2 = fbranch2 + mm + bbranch2
        branch2_devs = fdevs + mm_devs + bdevs2
    
        sched.add_block_seq(branch1, branch1_devs)
        sched.add_block_seq(branch2, branch2_devs)
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
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Tetris Cases')
    parser.add_argument('--premise', type=str)
    parser.add_argument('--ndevs', type=int,
                        help='number of devices')
    parser.add_argument('--nmicros', type=int,
                        help='number of micro-batches')
    parser.add_argument('--inflight', type=int,
                        help='memory limits')
    parser.add_argument('--save', type=str, default=None,
                        help='save searched schedule under a folder')
    parser.add_argument('--infer', action='store_true', default=False,
                        help='search for inference schedule')
    args = parser.parse_args()

    print('============== Scheduling Solver ================')
    print(args)
    sys.stdout.flush()

    premise = getattr(Premise, args.premise)
    micro: SchedPlan = premise(args.ndevs, train=not args.infer)
    for gid, blk in enumerate(micro.chain_blocks()):
        blk.gid = gid

    nmicros = min(args.nmicros, args.inflight)
    print(f'Premise: {args.ndevs} devices, {nmicros} micro-batches')
    print(micro)

    micros = [micro.copy(mid) for mid in range(nmicros)]
    # for inference, we don't consider memory
    if args.infer:
        for micro in micros:
            for block in micro.all_blocks():
                block.memory = 0

    memory = [args.inflight] * args.ndevs

    tic = time.time()
    schedules = Composer.compose(micros, memory)
    toc = time.time()

    print('\n================================================')
    print('search time: {:.2f} seconds'.format(toc-tic))

    if len(schedules) == 0:
        print(f'no solution')
    for idx, schedule in enumerate(schedules):
        print(f'schedule-{idx}:\n{schedule}')
        print(f'Unrolled schedule:\n{schedule.unroll(nmicros + 2)}')

    # schedule.save('./xshape.tsched.4stages.json')

    if args.save is not None:
        now = time.strftime("%Y-%m-%d-%H-%M", time.gmtime())
        Painter.visualize(
            micros[0],
            os.path.join(args.save, f"{args.premise}-premise.{now}.png"))
        repetends = [sched.extract(sched.repetend[0], sched.repetend[1]) for sched in schedules]
        for idx, (schedule, repetend) in enumerate(zip(schedules, repetends)):
            schedule.save(os.path.join(args.save, f"{args.premise}-schedule-{idx}.{now}.json"))
            Painter.visualize(
                schedule,
                os.path.join(args.save, f"{args.premise}-schedule-{idx}.{now}.png"))
            Painter.visualize(
                repetends[idx], 
                os.path.join(args.save, f"{args.premise}-repetend-{idx}.{now}.png"))
