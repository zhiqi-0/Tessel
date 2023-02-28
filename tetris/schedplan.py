from typing import Dict, Set, Tuple, List, Optional
import json

import more_itertools


StartEnd = Tuple[int, int]
Devices = Tuple[int, ...]


class Block:

    def __init__(self, mid: int, span: int, memory: float, btype: str, _gid=None):
        assert span > 0
        # micro-batch index
        self.mid: int = mid
        self.span = span
        self.memory = memory
        assert btype in ('forward', 'backward')
        self.btype = btype
        self.before = set()
        self.after = set()
        # sub-graph index
        self.gid: Optional[int] = _gid

    @staticmethod
    def make_dependency(prev, next):
        prev.after.add(next)
        next.before.add(prev)

    def __repr__(self):
        return f'f{self.mid}' if self.btype == 'forward' else f'b{self.mid}'


class SchedPlan:

    def __init__(self, ndevs: int) -> None:
        
        self._ndevs = ndevs
        self._nsteps = 0
        self._blocks: Set[Block] = set()
        self._block_devices: Dict[Block, Tuple[int]] = dict()
        self._block_steps: Dict[Block, int] = dict()
        self._step_blocks: Dict[int, List[Block]] = {0:[]}
        self._plans: List[List[Optional[Block]]] = [[] for _ in range(ndevs)]
        # repetend start step and end step
        self.repetend: Optional[StartEnd] = None

    @property
    def nsteps(self) -> int:
        return self._nsteps

    @property
    def ndevs(self) -> int:
        return self._ndevs

    def all_blocks(self) -> Set[Block]:
        return self._blocks
    
    def chain_blocks(self) -> List[Block]:
        """
        sort all blocks by step from early to later
        """
        blocks = []
        for step in range(self.nsteps):
            for block in self.blocks(step):
                blocks.append(block)
        assert len(blocks) == len(self._blocks)
        return blocks

    def add_block(self, block: Block, device: List[int], step: int):
        """Add a block into schedule plan. If the block is already inserted
        inside the scheduling plan, the block must have same step and device.
        """
        if block in self._blocks:
            assert self.step(block) == step and tuple(self.device(block)) == tuple(device), (
                f"Repeated adding a block but has different device and starting step setup:\n"
                f"Try to add   : {block}-{device} on step {step}\n"
                f"Already exist: {block}-{self.device(block)} on step {step}"
            )
            return
        maxstep = step + block.span
        if maxstep > self._nsteps:
            for devplan in self._plans:
                devplan += [None] * (maxstep - self._nsteps)
            for t in range(self._nsteps, maxstep):
                self._step_blocks.setdefault(t, [])
            self._nsteps = maxstep
        self._blocks.add(block)
        self._block_devices[block] = tuple(device)
        self._block_steps[block] = step
        self._step_blocks.setdefault(step, []).append(block)
        for devid in device:
            for t in range(step, step + block.span):
                assert self._plans[devid][t] is None, f"Conflict block {block}-{device} add on device {devid} at step {step}"
                self._plans[devid][t] = block

    def add_block_seq(self, blocks: List[Optional[Block]], devices: List[Optional[Devices]]):
        """
        Add a sequence of blocks into schedule plan

        The None in blocks indicates an empty step, which will not place block

        This assumes the blocks are dependent one after another.
        This will add blocks starting from time step 0.

        @param blocks List[Optional[Block]]
        @param devices List[Optional[Devices]]
        """
        assert len(blocks) == len(devices)
        step = 0
        for block, devs in zip(blocks, devices):
            if block is not None:
                self.add_block(block, devs, step)
            step += (block.span if block is not None else 1)
        blocks = [blk for blk in blocks if blk is not None]
        for blk1, blk2 in more_itertools.windowed(blocks, 2):
            Block.make_dependency(blk1, blk2)

    def blocks(self, step: int) -> List[Block]:
        return tuple(self._step_blocks[step])
    
    def step(self, block: Block) -> int:
        return self._block_steps[block]
    
    def device(self, block: Block) -> Tuple[int]:
        return self._block_devices[block]
    
    def extract(self, from_step: int, to_step: int):
        sched = SchedPlan(self.ndevs)
        for step in range(from_step, to_step):
            for block in self.blocks(step):
                sched.add_block(block, self.device(block), step-from_step)
        return sched

    def copy(self, mid_offset: Optional[int] = 0):
        """
        Copy the schedule plan and create the block with increased `mid_offset`
        """
        blks: Dict[Block, Block] = {}
        def new(block: Block):
            return blks.setdefault(
                block, Block(block.mid+mid_offset, block.span, block.memory, block.btype, block.gid))

        sched = SchedPlan(self.ndevs)
        for block in self._blocks:
            blk = new(block)
            sched.add_block(blk, self.device(block), self.step(block))
            # set dependency
            blk.before = set(new(bblock) for bblock in block.before)
            blk.after = set(new(ablock) for ablock in block.after)
        return sched

    def __repr__(self) -> str:
        dscp = ''
        for devid in range(self.ndevs):
            step = 0
            while step < self.nsteps:
                if self.repetend is not None and step in self.repetend:
                    dscp += ' |'
                have_block = False
                for blk in self.blocks(step):
                    if devid in self.device(blk):
                        dscp += ' ' + '-'.join([repr(blk)] * blk.span)
                        have_block = True
                        step += blk.span
                        break
                if not have_block:
                    dscp += ' --'
                    step += 1
            dscp += '\n'
        return dscp

    def save(filename: str):
        pass

    @staticmethod
    def load(filename: str):
        with open(filename, 'r') as f:
            plan = json.load(f)
        ndevs = plan['ndevs']
        schedplan = SchedPlan(ndevs)
        for block in plan['blocks']:
            # block attr
            mid = block['mid']
            span = block['span']
            memory = block['memory']
            btype = block['btype']
            gid = block.get('gid', None)
            # schedule plan position
            start = block['step']
            device: List[int] = block['device']
            schedplan.add_block(
                Block(mid, span, memory, btype, gid),
                device, start
            )
        return schedplan

    @staticmethod
    def concat(plans: List):
        cplan = SchedPlan(plans[0].ndevs)
        step_ofst = 0
        for plan in plans:
            for block in plan.all_blocks():
                cplan.add_block(block, plan.device(block), plan.step(block) + step_ofst)
            step_ofst += plan.nsteps
        return cplan

