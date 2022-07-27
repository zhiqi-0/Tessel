"""
A solver based solution for scheduling plan

python solver.py --premise vshape --nmicros 4 --ndevs 4 --memory 4
"""

from typing import List, Optional, Set, Dict
from enum import Enum
import time
import argparse

import z3


class Block:

    class BType(Enum):
        FW = 'forward'
        BW = 'backward'

    def __init__(self, mid: int, btype: BType, mem: int = 1, span: int = 1):
        self.mid = mid
        self.span = span
        self.memory = abs(mem) if btype == Block.BType.FW else 0-abs(mem)
        self.btype = btype

    def __repr__(self):
        return f'f{self.mid}' if self.btype == Block.BType.FW else f'b{self.mid}'


class SchedulePlan:

    def __init__(self, ndevs: int) -> None:
        
        self._blocks: Set[Block] = set()
        self._ndevs = ndevs
        self._block_devices: Dict[Block, List[int]] = {}
        self._block_steps: Dict[Block, z3.ArithRef] = {} # the start step
        self._nsteps: z3.ArithRef = None
        self._mem: List[z3.ArithRef] = [None] * ndevs
        self._solution: Optional[z3.z3.ModelRef] = None
        self._solver = z3.Solver()

    @property
    def nblocks(self) -> int:
        return len(self._blocks)

    @property
    def ndevs(self) -> int:
        return self._ndevs

    @property
    def nsteps(self) -> z3.ArithRef:
        return self._nsteps

    def add_block(self, block: Block, devs: List[int], step: int):
        self._block_devices[block] = devs
        start = z3.Int('blk' + str(len(self._block_devices)))
        end = start + block.span
        self._block_steps[block] = start
        self._solver.add(start >= step)
        # intra-device: no overlapping constraints
        for devid in devs:
            blocks = [blk for blk in self._blocks if devid in self.get_device(blk)]
            for blk in blocks:
                bstart = self.get_step(blk)
                bend = bstart + blk.span
                min_end = z3.If(bend < end, bend, end)
                max_start = z3.If(bstart > start, bstart, start)
                self._solver.add(min_end <= max_start)
        # set plan step
        if self._nsteps is None:
            self._nsteps = end
        else:
            self._nsteps = z3.If(end > self._nsteps, end, self._nsteps)
        self._blocks.add(block)

    def add_block_seq(self, blocks: List[Block], bdevs: List[List[int]]):
        step = 0
        for block, devs in zip(blocks, bdevs):
            self.add_block(block, devs, step)
            step += block.span

    def add_dependency(self, blocks: List[Block]):
        for idx in range(len(blocks) - 1):
            pre, post = blocks[idx], blocks[idx+1]
            pre_t, post_t = self.get_step(pre), self.get_step(post)
            self._solver.add(pre_t + pre.span <= post_t)

    def have_block(self, block: Block) -> bool:
        return block in self._blocks

    def init_peak_mem(self):
        """
        This can only be called after all blocks added
        """
        peak_mem_per_dev = []
        maxspan = sum(blk.span for blk in self._blocks)
        for devid in range(self.ndevs):
            peak = 0
            curr = 0
            blocks = [blk for blk in self._blocks if devid in self.get_device(blk)]
            for step in range(0, maxspan):
                mem = 0
                for block in blocks:
                    mem = z3.If(self.get_step(block) == step, block.memory, mem)
                curr = mem + curr
                peak = z3.If(curr > peak, curr, peak)
            peak_mem_per_dev.append(peak)
        self._mem = peak_mem_per_dev

    def get_device(self, block: Block) -> List[int]:
        assert block in self._blocks
        return self._block_devices[block]

    def get_step(self, block: Block) -> z3.ArithRef:
        assert block in self._blocks
        return self._block_steps[block]

    def step_optimal(self, memory: List[int]):
        """
        Find step optimal plans

        @param memory List[int]: the memory constraint of each device
        """
        print('memory constraints:', memory)
        self.init_peak_mem()
        for devid in range(self.ndevs):
            self._solver.add(self._mem[devid] <= memory[devid])
        # binary search
        opt_upper_step = sum(blk.span for blk in self._blocks) * 2
        opt_lower_step = 0
        opt_step = opt_upper_step
        while opt_lower_step != opt_upper_step:
            try_step = (opt_upper_step + opt_lower_step) // 2
            self._solver.push()
            self._solver.add(self._nsteps == try_step)
            if self._solver.check() == z3.sat:
                print(f'find sched plan of {try_step} steps')
                self._solution = self._solver.model()
                opt_upper_step = try_step
                opt_step = try_step
            else:
                print(f'fail to find sched plan of {try_step} steps')
                opt_lower_step = try_step + 1
            self._solver.pop()
        print(f'find step optimal sched plan {opt_step}')
        return opt_step

    def iter_step_optimal_plan(self):
        assert self._solution is not None, "Expected first call step_optimal"
        step = self._solution.eval(self._nsteps).as_long()
        self._solver.push()
        self._solver.add(self._nsteps == step)
        while self._solver.check() == z3.sat:
            solution: Dict[Block, int] = dict()
            model = self._solver.model()
            for blk, t in self._block_steps.items():
                solution[blk] = model.eval(t).as_long()
            yield solution
            block = [d() != model[d] for d in model]
            self._solver.add(z3.Or(block))
        self._solver.pop()
    

    def to_str(self, solution: Dict[Block, int]) -> str:
        dscp = ''
        nsteps = max(start + blk.span for blk, start in solution.items())
        for devid in range(self.ndevs):
            for step in range(0, nsteps):
                have_block = False
                for blk, start in solution.items():
                    if devid in self.get_device(blk) and step == start:
                        dscp += ' ' + '-'.join([repr(blk)] * blk.span)
                        step += blk.span
                        have_block = True
                        break
                if not have_block:
                    dscp += ' --'
            dscp += '\n'
        return dscp


class Premise:

    @staticmethod
    def vshape(ndevs: int, nmicros) -> SchedulePlan:
        """
        f             b
          f         b  
            f     b    
              f b      
        """
        sched = SchedulePlan(ndevs)
        for mid in range(nmicros):
            fblocks = [Block(mid, Block.BType.FW, span=1) for _ in range(ndevs)]
            fdevs = [[devid] for devid in range(ndevs)]
            bblocks = [Block(mid, Block.BType.BW, span=1) for _ in range(ndevs)]
            bdevs = [[devid] for devid in range(ndevs)][::-1]
            blocks = fblocks + bblocks
            devs = fdevs + bdevs
            sched.add_block_seq(blocks, devs)
            sched.add_dependency(blocks)
        return sched


    def chimera(ndevs: int, nmicros: int) -> SchedulePlan:
        """
        f             b        f b
          f         b        f     b
            f     b        f         b
              f b        f             b
        """
        sched = SchedulePlan(ndevs)
        assert nmicros % 2 == 0, "require microbatch# can be devided by 2"
        for mid in range(nmicros // 2): # V shape
            fblocks = [Block(mid, Block.BType.FW, f'f{mid}d{devid}', mem=1) for devid in range(ndevs)]
            bblocks = [Block(mid, Block.BType.BW, f'b{mid}d{devid}', mem=1) for devid in range(ndevs-1,-1,-1)]
            blocks = fblocks + bblocks
            for idx in range(ndevs * 2 - 1):
                Block.add_dependency(blocks[idx], blocks[idx+1])
            for devid in range(ndevs):
                sched.add_block(fblocks[devid], devid)
                sched.add_block(bblocks[ndevs-1-devid], devid)
        for mid in range(nmicros // 2): # ^ shape
            mid = mid + nmicros // 2
            fblocks = [Block(mid, Block.BType.FW, f'f{mid}d{devid}', mem=1) for devid in range(ndevs-1,-1,-1)]
            bblocks = [Block(mid, Block.BType.BW, f'b{mid}d{devid}', mem=1) for devid in range(ndevs)]
            blocks = fblocks + bblocks
            for idx in range(ndevs * 2 - 1):
                Block.add_dependency(blocks[idx], blocks[idx+1])
            for devid in range(ndevs):
                sched.add_block(fblocks[ndevs-1-devid], devid)
                sched.add_block(bblocks[devid], devid)
        return sched


    def interleave(ndevs: int, nmicros: int) -> SchedulePlan:
        """
        f f   f         b   b b
        f   f f         b b   b
        f     f f     b b     b
        f     f   f b   b     b
        """
        sched = SchedulePlan(ndevs)
        for mid in range(nmicros):
            fblocks = []
            bblocks = []
            for step in range(ndevs+2):
                if step in [0, ndevs // 2 + 1]:
                    fdevid = bdevid = tuple(range(ndevs))
                    fblock = Block(mid, Block.BType.FW, f'fe{step}{mid}devall', mem=1)
                    bblock = Block(mid, Block.BType.BW, f'be{step}{mid}devall', mem=1)
                else:
                    fdevid = bdevid = step - 1 if step < ndevs // 2 + 1 else step - 2
                    fblock = Block(mid, Block.BType.FW, f'f{mid}dev{fdevid}', mem=1)
                    bblock = Block(mid, Block.BType.BW, f'b{mid}dev{bdevid}', mem=1)
                fblocks.append(fblock)
                bblocks.append(bblock)
                sched.add_block(fblock, fdevid)
                sched.add_block(bblock, bdevid)
            blocks = fblocks + bblocks[::-1]
            for idx in range((ndevs + 2) * 2 - 1):
                Block.add_dependency(blocks[idx], blocks[idx+1])
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

    premise = getattr(Premise, args.premise)
    sched = premise(args.ndevs, args.nmicros)
    memory = [args.memory] * args.ndevs

    # step-optimal search
    tic = time.time()
    sched.step_optimal(memory)
    toc = time.time()
    print('search time for step optimal span: {:.2f} seconds'.format(toc-tic))

    print('find one solution:')
    for solution in sched.iter_step_optimal_plan():
        print(sched.to_str(solution))
        break
