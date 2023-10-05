from typing import List, Tuple, Optional
from functools import partial
from tetris.schedplan import SchedPlan, Block

from tetris.repetend import MicroPicker
from tetris.solver import StepOptimalSolver, BubbleOptimalSolver, SolverBase

from tetris.timer import CpuTimer

class Composer:

    @staticmethod
    def compose(micros: List[SchedPlan], memory: List[int]) -> List[SchedPlan]:
        assert len(micros) > 0
        assert all(micro.ndevs == micros[0].ndevs for micro in micros)
        ndevs = micros[0].ndevs

        schedule_parts = [None, None, None]  # warmup / repetend / cooldown
        nbubbles = micros[0].nsteps + 1
        for warmup_blks, repetend_blks, cooldown_blks, devices in MicroPicker.pick(micros):
            warmup_devs = [devices[blk] for blk in warmup_blks]
            repetend_devs = [devices[blk] for blk in repetend_blks]
            cooldown_devs = [devices[blk] for blk in cooldown_blks]
            warmup_post_mem = Composer.memory(warmup_blks, warmup_devs, ndevs)
            repetend_post_mem = Composer.memory(warmup_blks + repetend_blks, warmup_devs + repetend_devs, ndevs)
            
            # step 1: construct a repetend
            repetend_pre_mem = [memory[devid] - warmup_post_mem[devid] for devid in range(ndevs)]
            CpuTimer().start('repetend')
            repetend, case_nbubbles = Composer.construct(
                repetend_blks, repetend_devs, ndevs, repetend_pre_mem, nbubbles, optimizer=BubbleOptimalSolver)
            # repetend, case_nbubbles = Composer.construct(
            #     repetend_blks, repetend_devs, ndevs, mem, nbubbles, optimizer=StepOptimalSolver)
            CpuTimer().stop('repetend')
            if repetend is None: continue
            
            # step 2: validate warmup
            CpuTimer().start('warmup')
            exist = Composer.satisfy(warmup_blks, warmup_devs, ndevs, memory) 
            CpuTimer().stop('warmup')
            if not exist: continue
            
            # step 3: validate cooldown
            cooldown_pre_mem = [memory[devid] - repetend_post_mem[devid] for devid in range(ndevs)]
            CpuTimer().start('cooldown')
            exist = Composer.satisfy(cooldown_blks, cooldown_devs, ndevs, cooldown_pre_mem)
            CpuTimer().stop('cooldown')
            if not exist: continue
            
            print(f'find a better solution solution: bubbles per repetend: {case_nbubbles}')

            assert case_nbubbles < nbubbles
            nbubbles = case_nbubbles
            schedule_parts = [
                (warmup_blks, warmup_devs, memory),
                repetend,
                (cooldown_blks, cooldown_devs, cooldown_pre_mem)
            ]

            if case_nbubbles == 0:
                print('> early stop as find 0-bubble plans')
                break
            print(f'> setting repetend maximal bubbles: {nbubbles}')

        repetend = schedule_parts[1]
        if repetend is None:  # no solution
            return []

        # search for warmup parts
        print(f'> constructing warmup and cooldown parts...')
        CpuTimer().start('warmup')
        warmup_blks, warmup_devs, warmup_mem = schedule_parts[0]
        warmup, _ = Composer.construct(warmup_blks, warmup_devs, ndevs, warmup_mem,
                                       optimizer=StepOptimalSolver)
        CpuTimer().stop('warmup')
        assert warmup is not None

        # search for cooldown parts
        CpuTimer().start('cooldown')
        cooldown_blks, cooldown_devs, cooldown_mem = schedule_parts[2]
        cooldown, _ = Composer.construct(cooldown_blks, cooldown_devs, ndevs, cooldown_mem,
                                         optimizer=StepOptimalSolver)
        CpuTimer().stop('cooldown')
        assert cooldown is not None

        schedule = SchedPlan.concat([warmup, repetend, cooldown])
        schedule.repetend = (warmup.nsteps, warmup.nsteps + repetend.nsteps)

        return [schedule]

    @staticmethod
    def construct(blocks: List[Block], devices: List[Tuple[int]],
                  ndevs: int, memory: Tuple[int],
                  upper: Optional[int]=None, optimizer: Optional[SolverBase] = StepOptimalSolver) -> Tuple[Optional[SchedPlan], Optional[int]]:
        solver = optimizer(ndevs)
        # step 1 add block inside solver
        for block, devs in zip(blocks, devices):
            solver.add_block(block, devs, 0)
        # step 2 setup dependency
        for blk1 in blocks:
            for blk2 in blocks:
                if blk2 in blk1.after:
                    # print(f'> add dependency: {blk1}-dev{devices[idx1]} -> {blk2}-dev{devices[idx2]}')
                    solver.add_dependency_constraints([blk1, blk2])
        # step 3 construct
        lowest = solver.solve(memory, upper, silence=True)
        if lowest is None:
            print(f"{optimizer.__name__}: Fail to find a solution given boundary constraints ( solution > {upper} (upper) )\n")
            return None, None
        for schedplan in solver.solutions():
            # assert schedplan.nsteps == nsteps
            return schedplan, lowest

    @staticmethod
    def satisfy(blocks: List[Block], devices: List[Tuple[int]],
                ndevs: int, memory: Tuple[int]) -> bool:
        """Check the existence of a schedule"""
        solver = StepOptimalSolver(ndevs)
        # step 1 add block inside solver
        for block, devs in zip(blocks, devices):
            solver.add_block(block, devs, 0)
        # step 2 setup dependency
        for blk1 in blocks:
            for blk2 in blocks:
                if blk2 in blk1.after:
                    # print(f'> add dependency: {blk1}-dev{devices[idx1]} -> {blk2}-dev{devices[idx2]}')
                    solver.add_dependency_constraints([blk1, blk2])
        return solver.satisfy(memory)

    @staticmethod
    def memory(blocks: List[Block], devices: List[Tuple[int]], ndevs: int) -> Tuple[int]:
        """
        Calculate memory after executing all blocks
        """
        memory: Tuple[int] = [0] * ndevs
        for block, device in zip(blocks, devices):
            for devid in device:
                memory[devid] += block.memory
        return memory
