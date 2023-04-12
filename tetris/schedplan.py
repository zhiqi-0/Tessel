from typing import Dict, Set, Tuple, List, Optional
import json
import numpy as np

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
    
    def unroll(self, nmicros: int):
        """Unroll repetend to `nmicros` microbatches

        @note the new blocks in unrolled schedule are not set
        with any dependency.
        
        @param nmicros int: numbe of microbatches
        @return unrolled_plan SchedPlan: a new unrolled schedule plan
        """
        assert self.repetend is not None
        rstart, rend = self.repetend
        # already existed number of microbatches
        mids = set(blk.mid for blk in self._blocks)
        assert len(mids) == max(mids) + 1, f"Microbatch index should be consecutive"
        nmids = len(mids)
        assert nmicros >= nmids, \
            f"Unroll to nmicros ({nmicros}) smaller than num-microbatches that are already in schedule ({nmids})"

        all_blocks: Dict[Tuple[int, int], Block] = {}
        for blk in self._blocks:
            assert blk.gid is not None
            all_blocks[(blk.gid, blk.mid)] = blk
    
        def get_block(gid: int, mid: int) -> Block:
            ref = all_blocks[(gid, 0)]
            return all_blocks.setdefault(
                (gid, mid), Block(mid, ref.span ,ref.memory, ref.btype, gid))

        in_repetend = lambda blk: blk in self._blocks and rstart <= self.step(blk) and self.step(blk) < rend

        # get repetend offset
        dev_span = []
        for devid in range(self.ndevs):
            blocks = [blk for blk in self._blocks if devid in self.device(blk)]
            blocks = [blk for blk in blocks if in_repetend(blk)]
            maxstep = max(self.step(blk) + blk.span for blk in blocks)
            minstep = min(self.step(blk) for blk in blocks)
            dev_span.append(maxstep - minstep)
        rspan = max(dev_span)

        rofst = 0
        for devid in range(self.ndevs):
            blocks = [blk for blk in self._blocks if devid in self.device(blk)]
            blocks = [blk for blk in blocks if in_repetend(blk)]
            for block in blocks:
                # after blocks
                ablocks = list(block.after)
                keys = [(blk.gid, blk.mid + 1) for blk in ablocks]
                ablocks = [get_block(*key) for key in keys]
                ablocks = [blk for blk in ablocks if in_repetend(blk)]
                astarts = [self.step(blk) for blk in ablocks]
                if len(astarts) != 0:
                    min_ofst = min(astarts) - self.step(block) - rspan
                    rofst = max(rofst, min_ofst)
                # before blocks
                bblocks = list(block.before)
                keys = [(blk.gid, blk.mid + 1) for blk in bblocks]
                bblocks = [get_block(*key) for key in keys]
                bblocks = [blk for blk in bblocks if in_repetend(blk)]
                bends = [self.step(blk) + blk.span for blk in bblocks]
                if len(bends) != 0:
                    min_ofst = max(bends) - self.step(block) - rspan
                    rofst = max(rofst, min_ofst)

        unrolled_plan = SchedPlan(self.ndevs)

        # warmup
        for step in range(rstart):
            blocks = self.blocks(step)
            for block in blocks:
                unrolled_plan.add_block(block, self.device(block), step)

        # steady
        rspan = max(dev_span)
        for mid_ofst in range(nmicros-nmids+1):
            for step in range(rstart, rend):
                for blk in self.blocks(step):
                    rblk = get_block(blk.gid, blk.mid + mid_ofst)
                    unrolled_plan.add_block(
                        rblk, self.device(blk),
                        step + (rspan + rofst) * mid_ofst
                    )

        # cooldown
        mid_ofst = nmicros - nmids
        for step in range(rend, self.nsteps):
            for blk in self.blocks(step):
                unrolled_plan.add_block(
                    get_block(blk.gid, blk.mid + mid_ofst), 
                    self.device(blk),
                    step + (rspan + rofst) * mid_ofst
                )
        
        unrolled_plan.repetend = (rstart, rend + (rspan + rofst) * mid_ofst)
        return unrolled_plan

    def copy(self, mid_offset: Optional[int] = 0):
        """Copy the schedule plan and create the block with 
        increased `mid_offset`
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

    def getstate(self) -> np.ndarray:
        """
        return state format: 2-D array of (M, N+2) shape,
        where M is number of microbatches, N is number of sub-graphs.
        
        (i, j) in (M, N) denotes the start time of block gid j of microbatch i 
        (*, N+1) and (*, N+2) denotes the start and end of the repetend, respectively.
        """
        nmicros = max(blk.mid for blk in self._blocks) + 1
        nstages = max(blk.gid for blk in self._blocks) + 1
        state = -np.ones((nmicros, nstages+2), dtype=int)
        for blk in self._blocks:
            step = self.step(blk)
            state[blk.mid, blk.gid] = step
        state[:,-2:] = self.repetend
        assert np.all(state >= 0)
        return state

    def loadstate(self, blocks: List[Block], devices: List[List[int]], state: np.ndarray):
        """Load the state from the state array"""
        getblock, getdevice = {}, {}
        for blk, devs in zip(blocks, devices):
            getblock[(blk.mid, blk.gid)] = blk
            getdevice[blk] = devs
        self.repetend = tuple(state[0,-2:])
        for mid in range(state.shape[0]):
            for gid in range(state.shape[1]-2):
                step = state[mid, gid]
                block = getblock[(mid, gid)]
                self.add_block(block, getdevice[block], step)

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

