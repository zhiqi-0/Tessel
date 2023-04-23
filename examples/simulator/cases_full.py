"""
PYTHONPATH=.:$PYTHONPATH python examples/cases.py \
    --premise vshape --ndevs 4 --nmicros 4 --inflight 4
"""
from typing import List
import sys
import time
import argparse

from tetris.schedplan import SchedPlan, Block
from tetris.solver import StepOptimalSolver
from tetris.composer import Composer


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
        bblocks = [Block(0, span=3, memory=-1, btype=BW) for _ in range(ndevs)]
        bdevs = [[devid] for devid in range(ndevs)][::-1]
        blocks = fblocks + bblocks
        devs = fdevs + bdevs
        sched.add_block_seq(blocks, devs)
        return sched

    @staticmethod
    def xshape(ndevs: int) -> SchedPlan:
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
        bblocks = [Block(0, span=3, memory=-1, btype=BW) for _ in range(ndevs)]
        bdevs = [[devid] for devid in range(ndevs)][::-1]
        vblocks = fblocks + bblocks
        vdevs = fdevs + bdevs
        # ^
        fblocks = [Block(0, span=1, memory=1, btype=FW) for _ in range(ndevs)]
        fdevs = [[devid] for devid in range(ndevs)][::-1]
        bblocks = [Block(0, span=3, memory=-1, btype=BW) for _ in range(ndevs)]
        bdevs = [[devid] for devid in range(ndevs)]
        rvblocks = fblocks + bblocks
        rvdevs = fdevs + bdevs

        sched.add_block_seq(vblocks, vdevs)
        sched.add_block_seq(rvblocks, rvdevs)
        return sched

    @staticmethod
    def mshape(ndevs: int) -> SchedPlan:
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
        bblocks = [Block(0, span=3, memory=-1, btype=BW) for _ in range(ndevs)]
        bdevs = [[ndevs-1-devid] for devid in range(ndevs)]
        #
        fblocks.insert(0, Block(0, span=1, memory=0, btype=FW))
        fdevs.insert(0, list(range(ndevs)))
        bblocks.insert(len(bblocks), Block(0, span=1, memory=0, btype=BW))
        bdevs.insert(len(bblocks), list(range(ndevs)))
    
        blocks = fblocks + bblocks
        devs = fdevs + bdevs
        sched.add_block_seq(blocks, devs)
        return sched

    @staticmethod
    def yshape(ndevs: int) -> SchedPlan:
        """
        f   f
          f f
        f   f
          f f 
        """
        sched = SchedPlan(ndevs)
        # full spmd on loss:
        mm = [Block(0, span=1, memory=0, btype=FW)]
        mm_devs = [list(range(ndevs))]
        # vshape
        ndevs1 = ndevs // 2
        fbranch1 = [Block(0, span=1, memory=0, btype=FW) for _ in range(ndevs1)]
        fdevs = [[devid] for devid in range(ndevs1)]
        branch1 = fbranch1 + mm
        branch1_devs = fdevs + mm_devs
        # vshape
        ndevs2 = ndevs - ndevs1
        fbranch2 = [Block(0, span=1, memory=0, btype=FW) for _ in range(ndevs1)]
        fdevs = [[devid] for devid in range(ndevs1, ndevs1+ndevs2)]
        branch2 = fbranch2 + mm
        branch2_devs = fdevs + mm_devs
    
        sched.add_block_seq(branch1, branch1_devs)
        sched.add_block_seq(branch2, branch2_devs)
        return sched
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='comm primitive')
    parser.add_argument('--premise', type=str, required=True)
    parser.add_argument('--ndevs', type=int,
                        help='number of devices')
    parser.add_argument('--nmicros', type=int,
                        help='number of micro-batches')
    parser.add_argument('--inflight', type=int,
                        help='memory limits')
    args = parser.parse_args()

    print('============== Scheduling Solver ================')
    print(args)
    sys.stdout.flush()

    premise = getattr(Premise, args.premise)
    micro: SchedPlan = premise(args.ndevs)
    for gid, blk in enumerate(micro.chain_blocks()):
        blk.gid = gid

    micros = [micro.copy(mid) for mid in range(args.nmicros)]
    memory = [args.inflight] * args.ndevs

    print(f'Premise: {args.ndevs} devices, {args.nmicros} micro-batches')
    print(micros[0])

    all_blocks = []
    all_devices = []
    for micro in micros:
        for block in micro.all_blocks():
            all_blocks.append(block)
            all_devices.append(micro.device(block))

    tic = time.time()
    sched, nsteps = Composer.construct(
        all_blocks, all_devices, args.ndevs, memory,
        optimizer=StepOptimalSolver)
    span = time.time() - tic

    print(f'search time for optimal step {nsteps} span: {span:.2f} seconds')

    print('find one solution:')
    print(sched)
