from typing import List, Tuple, Optional, Dict
from tetris.schedplan import SchedPlan, Block

from tetris.repetend import MicroPicker
from tetris.solver import StepOptimalSolver, BubbleOptimalSolver, SolverBase


class Composer:

    @staticmethod
    def compose(micros: List[SchedPlan], memory: List[int]) -> List[SchedPlan]:
        assert len(micros) > 0
        assert all(micro.ndevs == micros[0].ndevs for micro in micros)
        ndevs = micros[0].ndevs

        schedules = []
        nbubbles = micros[0].nsteps + 1
        for warmup_blks, repetend_blks, cooldown_blks, devices in MicroPicker.pick(micros):
            warmup_devs = [devices[blk] for blk in warmup_blks]
            repetend_devs = [devices[blk] for blk in repetend_blks]
            cooldown_devs = [devices[blk] for blk in cooldown_blks]
            warmup_mem = Composer.memory(warmup_blks, warmup_devs)
            repetend_mem = Composer.memory(warmup_blks + repetend_blks, warmup_devs + repetend_devs)
            
            # step 1: construct a repetend
            mem = [memory[devid] - warmup_mem.get(devid, 0) for devid in range(ndevs)]
            repetend, case_nbubbles = Composer.construct(
                repetend_blks, repetend_devs, ndevs, mem, nbubbles, optimizer=BubbleOptimalSolver)
            # repetend, case_nbubbles = Composer.construct(
            #     repetend_blks, repetend_devs, ndevs, mem, nbubbles, optimizer=StepOptimalSolver)
            if repetend is None: continue
            
            # step 2: construct warmup
            warmup, _ = Composer.construct(warmup_blks, warmup_devs, ndevs, memory)
            if warmup is None: continue
            
            # step 3: construct cooldown
            mem = [memory[devid] - repetend_mem.get(devid, 0) for devid in range(ndevs)]
            cooldown, _ = Composer.construct(cooldown_blks, cooldown_devs, ndevs, mem)
            if cooldown is None: continue
            
            print('find one solution:')
            schedule = SchedPlan.concat([warmup, repetend, cooldown])
            schedule.repetend = (warmup.nsteps, warmup.nsteps + repetend.nsteps)
            print(schedule)

            if case_nbubbles < nbubbles:
                schedules = []
            nbubbles = case_nbubbles
            schedules.append(schedule)

            if case_nbubbles == 0:
                print('> early exit as find 0-bubble plans')
                return schedules
            print(f'> setting repetend maximal bubbles: {nbubbles}')

        return schedules

    @staticmethod
    def construct(blocks: List[Block], devices: List[Tuple[int]],
                  ndevs: int, memory: int,
                  upper: Optional[int]=None, optimizer: Optional[SolverBase] = StepOptimalSolver) -> Tuple[Optional[SchedPlan], Optional[int]]:
        solver = optimizer(ndevs)
        # step 1 add block inside solver
        for block, devs in zip(blocks, devices):
            solver.add_block(block, devs, 0)
        # step 2 setup dependency
        for idx1, blk1 in enumerate(blocks):
            for idx2, blk2 in enumerate(blocks):
                if blk2 in blk1.after:
                    # print(f'> add dependency: {blk1}-dev{devices[idx1]} -> {blk2}-dev{devices[idx2]}')
                    solver.add_dependency_constraints([blk1, blk2])
        # step 3 construct
        lowest = solver.solve(memory, upper)
        if lowest is None:
            print(f"{optimizer.__name__}: Fail to find a solution given boundary constraints ( solution > {upper} (upper) )\n")
            return None, None
        for schedplan in solver.solutions():
            # assert schedplan.nsteps == nsteps
            return schedplan, lowest

    @staticmethod
    def memory(blocks: List[Block], devices: List[Tuple[int]] ) -> Dict[int, int]:
        """
        Calculate memory after executing all blocks
        """
        memory: Dict[int, int] = {}
        for block, device in zip(blocks, devices):
            for devid in device:
                memory.setdefault(devid, 0)
                memory[devid] += block.memory
        return memory
