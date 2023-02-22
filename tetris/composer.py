from typing import List, Tuple, Optional, Dict
from tetris.schedplan import SchedPlan, Block

from tetris.repetend import MicroPicker
from tetris.solver import StepOptimalSolver


class Composer:

    @staticmethod
    def compose(micros: List[SchedPlan], memory: List[int]):
        assert len(micros) > 0
        assert all(micro.ndevs == micros[0].ndevs for micro in micros)
        ndevs = micros[0].ndevs

        nsteps = micros[0].nsteps
        for warmup_blks, repetend_blks, cooldown_blks, devices in MicroPicker.pick(micros):
            warmup_devs = [devices[blk] for blk in warmup_blks]
            repetend_devs = [devices[blk] for blk in repetend_blks]
            cooldown_devs = [devices[blk] for blk in cooldown_blks]
            warmup_mem = Composer.memory(warmup_blks, warmup_devs)
            repetend_mem = Composer.memory(warmup_blks + repetend_blks, warmup_devs + repetend_devs)
            
            # step 1: construct a repetend
            mem = [memory[devid] - warmup_mem.get(devid, 0) for devid in range(ndevs)]
            repetend = Composer.construct(repetend_blks, repetend_devs, ndevs, mem, nsteps)
            if repetend is None: continue
            # step 2: construct warmup
            # TODO: setup memory
            warmup = Composer.construct(warmup_blks, warmup_devs, ndevs, memory)
            if warmup is None: continue
            # step 3: construct cooldown
            mem = [memory[devid] - repetend_mem.get(devid, 0) for devid in range(ndevs)]
            cooldown = Composer.construct(cooldown_blks, cooldown_devs, ndevs, mem)
            if cooldown is None: continue
            
            print('find one solution:')
            schedule = SchedPlan.concat([warmup, repetend, cooldown])
            schedule.split_steps = [warmup.nsteps, warmup.nsteps + repetend.nsteps]
            print(schedule)

            # TODO: optimize this to identify real bubble rate
            nsteps = repetend.nsteps
            print(f'> setting repetend maximal nsteps: {nsteps}')
            

    @staticmethod
    def construct(blocks: List[Block], devices: List[Tuple[int]],
                  ndevs: int, memory: int,
                  upper_nsteps: Optional[int]=None) -> Optional[SchedPlan]:
        solver = StepOptimalSolver(ndevs)
        # step 1 add block inside solver
        for block, devs in zip(blocks, devices):
            solver.add_block(block, devs, 0)
        # step 2 setup dependency
        for idx1, blk1 in enumerate(blocks):
            for idx2, blk2 in enumerate(blocks):
                if blk2 in blk1.after:
                    # print(f'> add dependency: {blk1}-dev{devices[idx1]} -> {blk2}-dev{devices[idx2]}')
                    solver.add_dependency([blk1, blk2])
        # step 3 construct
        nsteps = solver.time_optimal(memory, upper_nsteps)
        if nsteps is None:
            print("Fail to find a solution given step constraints\n")
            return None
        for schedplan in solver.solutions():
            assert schedplan.nsteps == nsteps
            return schedplan

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
