import sys
import argparse
import math

from tessel.schedule.schedplan import SchedPlan, Block
from tessel.schedule.composer import Composer
from tessel.timer import CpuTimer


FW='forward'
BW='backward'


class Placement:

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
    def nnshape_eager(ndevs: int, train: bool = True) -> SchedPlan:
        """
        f f f             b b b
        f   f f         b b   b
        f   f   f     b   b   b
        f   f     f b     b   b
        """
        assert ndevs == 4
        sched = SchedPlan(ndevs)
        # v-shape
        fblocks = [Block(0, span=1, memory=1, btype=FW) for _ in range(ndevs)]
        fdevs = [[devid] for devid in range(ndevs)]
        bblocks, bdevs = [], []
        if train:
            bblocks = [Block(0, span=3, memory=-1, btype=BW) for _ in range(ndevs)]
            bdevs = [[ndevs-1-devid] for devid in range(ndevs)]
        # full shard 2
        fblocks.insert(1, Block(0, span=1, memory=0, btype=FW))
        fdevs.insert(1, list(range(ndevs)))
        if train:
            bblocks.insert(ndevs-1, Block(0, span=1, memory=0, btype=BW))
            bdevs.insert(ndevs-1, list(range(ndevs)))
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
    def mllm(ndevs: int, train: bool = True) -> SchedPlan:
        """
        f f       f b       b b
        f   f     f b     b   b
        f     f   f b   b     b
        f       f f b b       b
        """
        assert train, f"Only support training mode for now"
        sched = SchedPlan(ndevs)
        # 
        fblocks = [Block(0, span=1, memory=1, btype=FW) for _ in range(ndevs)]
        fdevs = [[devid] for devid in range(ndevs)]
        bblocks = [Block(0, span=3, memory=-1, btype=BW) for _ in range(ndevs)]
        bdevs = [[ndevs-1-devid] for devid in range(ndevs)]
        #
        fblocks.insert(len(fblocks), Block(0, span=1, memory=0, btype=FW))
        fdevs.insert(len(fdevs), list(range(ndevs)))
        bblocks.insert(0, Block(0, span=1, memory=0, btype=BW))
        bdevs.insert(0, list(range(ndevs)))
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
    def finetune(ndevs: int, train: bool = True) -> SchedPlan:
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
        
        if train:
            bblocks = [Block(0, span=2, memory=-1, btype=BW) for _ in range(ndevs)]
            bdevs = [[devid] for devid in range(ndevs)][::-1]
        else:
            bblocks, bdevs = [], []

        blocks = fblocks + bblocks
        devs = fdevs + bdevs
        sched.add_block_seq(blocks, devs)
        return sched

    @staticmethod
    def alphafold(ndevs: int, train: bool = True) -> SchedPlan:
        """
        f   f   f       b
         f   f   f     b
          f   f   f   b
           f   f   f b
        """
        sched = SchedPlan(ndevs)

        fblocks_1 = [Block(0, span=1, memory=0, btype=FW) for _ in range(ndevs)]
        fdevs_1 = [[devid] for devid in range(ndevs)]

        fblocks_2 = [Block(0, span=1, memory=0, btype=FW) for _ in range(ndevs)]
        fdevs_2 = [[devid] for devid in range(ndevs)]

        fblocks_3 = [Block(0, span=1, memory=1, btype=FW) for _ in range(ndevs)]
        fdevs_3 = [[devid] for devid in range(ndevs)]

        if train:
            bblocks = [Block(0, span=1, memory=-1, btype=BW) for _ in range(ndevs)]
            bdevs = [[devid] for devid in range(ndevs)][::-1]
        else:
            bblocks, bdevs = [], []

        blocks = fblocks_1 + fblocks_2 + fblocks_3 + bblocks
        devs = fdevs_1 + fdevs_2 + fdevs_3 + bdevs
        sched.add_block_seq(blocks, devs)
        return sched


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Tessel Cases')
    parser.add_argument('--placement', type=str)
    parser.add_argument('--ndevs', type=int,
                        help='number of devices')
    parser.add_argument('--nmicros', type=int, default=None,
                        help='number of micro-batches')
    parser.add_argument('--memory', type=int,
                        help='memory limits')
    parser.add_argument('--save', type=str, default=None,
                        help='save searched schedule under a folder')
    parser.add_argument('--infer', action='store_true', default=False,
                        help='search for inference schedule')
    parser.add_argument('--fast-search', action='store_true', default=False,
                        help='use fast search')
    args = parser.parse_args()

    print('============== Scheduling Solver ================')
    print(args)
    sys.stdout.flush()

    placement = getattr(Placement, args.placement)
    micro: SchedPlan = placement(args.ndevs, train=not args.infer)
    print(micro)

    # for inference, we don't consider memory
    if args.infer:
        for block in micro.all_blocks():
            block.memory = 0

    CpuTimer(enable=True).start('search e2e')
    if not args.fast_search:
        print('using compose_n for search')
        schedule = Composer.compose_n(micro, args.memory, args.nmicros)
    else:
        schedule = Composer.compose_fast(micro, args.memory)
    CpuTimer().stop('search e2e')

    print('\n' + '=' * 48)
    CpuTimer().print_all(times=1)

    if schedule is None:
        print(f'no solution')
    else:
        print(f'best schedule:\n{schedule}')
        nmicros = max(blk.mid for blk in schedule.all_blocks()) + 1
        print(f'Unrolled schedule:\n{schedule.unroll(nmicros + 2)}')
        print(f'peak memory of each device: {[schedule.peak_memory(devid) for devid in range(args.ndevs)]}')
        warmup_step = schedule.repetend[0]
        repetend_step = schedule.repetend[1] - schedule.repetend[0]
        cooldown_steps = schedule.nsteps - schedule.repetend[1]
        print(f'warmup nsteps: {warmup_step}, repetend nsteps: {repetend_step}, cooldown nsteps: {cooldown_steps}')

    if args.save is not None:
        schedule.save(args.save)
