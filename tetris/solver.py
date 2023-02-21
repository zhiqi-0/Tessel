"""
A solver based solution for scheduling plan

python solver.py --premise vshape --nmicros 4 --ndevs 4 --memory 4
"""

from typing import List, Optional, Set, Dict
import sys

from tetris.schedplan import SchedPlan, Block

import z3


class SolverBase:

    def __init__(self, ndevs: int):

        self._blocks: Set[Block] = set()
        self._ndevs = ndevs
        self._block_devices: Dict[Block, List[int]] = {}
        self._block_steps: Dict[Block, z3.ArithRef] = {} # the start step
        self._nsteps: z3.ArithRef = None
        self._mem: List[z3.ArithRef] = [None] * ndevs
        self._solution: Optional[z3.z3.ModelRef] = None
        self._solved = False
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
    
    @property
    def solved(self) -> bool:
        return True
    
    def device(self, block: Block) -> List[int]:
        assert block in self._blocks
        return self._block_devices[block]

    def step(self, block: Block) -> z3.ArithRef:
        assert block in self._blocks
        return self._block_steps[block]
    
    def add_block(self, block: Block, devs: List[int], step: int):
        self._solved, self._solution = False, None
        self._block_devices[block] = devs
        start = z3.Int('blk' + str(len(self._block_devices)))
        end = start + block.span
        self._block_steps[block] = start
        self._solver.add(start >= step)
        # intra-device: no overlapping constraints
        for devid in devs:
            blocks = [blk for blk in self._blocks if devid in self.device(blk)]
            for blk in blocks:
                bstart = self.step(blk)
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

    def add_dependency(self, blocks: List[Block]):
        self._solved, self._solution = False, None
        for idx in range(len(blocks) - 1):
            pre, post = blocks[idx], blocks[idx+1]
            pre_t, post_t = self.step(pre), self.step(post)
            self._solver.add(pre_t + pre.span <= post_t)

    def add_micro_plan(self, micro: SchedPlan):
        self._solved, self._solution = False, None
        for block in micro.all_blocks():
            step = micro.step(block)
            devs = micro.device(block)
            self.add_block(block, devs, step)
        for block in micro.all_blocks():
            for after_block in block.after:
                self.add_dependency([block, after_block])
    
    def init_peak_mem(self):
        """
        This can only be called after all blocks added
        """
        peak_mem_per_dev = []
        maxspan = sum(blk.span for blk in self._blocks)
        for devid in range(self.ndevs):
            peak = 0
            curr = 0
            blocks = [blk for blk in self._blocks if devid in self.device(blk)]
            for step in range(0, maxspan):
                mem = 0
                for block in blocks:
                    mem = z3.If(self.step(block) == step, block.memory, mem)
                curr = mem + curr
                peak = z3.If(curr > peak, curr, peak)
            peak_mem_per_dev.append(peak)
        self._mem = peak_mem_per_dev
    

class StepOptimalSolver(SolverBase):

    def __init__(self, ndevs: int) -> None:
        super().__init__(ndevs)

    def time_optimal(self, memory: List[int], time: Optional[int] = None) -> Optional[int]:
        """
        Find step optimal plans

        @param memory List[int]: the memory constraint of each device
        """
        self._solution = None
        print('memory constraints:', memory)
        self.init_peak_mem()
        for devid in range(self.ndevs):
            self._solver.add(self._mem[devid] <= memory[devid])
        # binary search
        opt_upper_step = sum(blk.span for blk in self._blocks) if time is None else time
        opt_lower_step = 0
        opt_step = opt_upper_step
        while opt_lower_step != opt_upper_step:
            try_step = (opt_upper_step + opt_lower_step) // 2
            self._solver.push()
            self._solver.add(self._nsteps == try_step)
            if self._solver.check() == z3.sat:
                print(f'find sched plan of {try_step} steps')
                sys.stdout.flush()
                self._solution = self._solver.model()
                opt_upper_step = try_step
                opt_step = try_step
            else:
                print(f'fail to find sched plan of {try_step} steps')
                sys.stdout.flush()
                opt_lower_step = try_step + 1
            self._solver.pop()
        if self._solution is not None:
            print(f'find step optimal sched plan {opt_step}')
        sys.stdout.flush()
        self._solved = True
        return opt_step if self._solution is not None else None
    
    def solutions(self) -> SchedPlan:
        """
        iterate all possible solutions given the time-optimal solutions

        @yield solution SchedPlan
        """
        assert self._solved, "Expected first call time_optimal"
        step = self._solution.eval(self._nsteps).as_long()
        self._solver.push()
        self._solver.add(self._nsteps == step)
        while self._solver.check() == z3.sat:
            solution: Dict[Block, int] = dict()
            model = self._solver.model()
            for blk, t in self._block_steps.items():
                solution[blk] = model.eval(t).as_long()
            # solution to schedule plan
            schedplan = SchedPlan(self.ndevs)
            for blk, t in self._block_steps.items():
                step = model.eval(t).as_long()
                schedplan.add_block(blk, self.device(blk), step)
            yield schedplan
            block = [d() != model[d] for d in model]
            self._solver.add(z3.Or(block))
        self._solver.pop()
