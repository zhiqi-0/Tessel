from typing import Dict, Set, Tuple, List, Optional

import json
import numpy as np

import more_itertools

import argparse


class Block:

    def __init__(self, mid: int, span: int, memory: float, btype: str):
        assert span > 0
        self.mid = mid
        self.span = span
        self.memory = memory
        assert btype in ('forward', 'backward')
        self.btype = btype
        self.before = set()
        self.after = set()

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

    @property
    def nsteps(self) -> int:
        return self._nsteps

    @property
    def ndevs(self) -> int:
        return self._ndevs

    def all_blocks(self) -> Set[Block]:
        return self._blocks

    def add_block(self, block: Block, device: List[int], step: int):
        """
        Add a block into schedule plan
        """
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
                assert self._plans[devid][t] is None, f"Conflict block add on device {devid} at step {step}"
                self._plans[devid][t] = block

    def add_block_seq(self, blocks: List[Block], devices: List[Tuple[int]]):
        """
        Add a sequence of blocks into schedule plan

        This assumes the blocks are dependent one after another.
        This will add blocks starting from time step 0.
        """
        assert len(blocks) == len(devices)
        step = 0
        for block, devs in zip(blocks, devices):
            self.add_block(block, devs, step)
            step += block.span
        for blk1, blk2 in more_itertools.windowed(blocks, 2):
            Block.make_dependency(blk1, blk2)

    def blocks(self, step: int) -> List[Block]:
        return tuple(self._step_blocks[step])
    
    def step(self, block: Block) -> int:
        return self._block_steps[block]
    
    def device(self, block: Block) -> Tuple[int]:
        return self._block_devices[block]

    def __repr__(self) -> str:
        dscp = ''
        for devid in range(self.ndevs):
            step = 0
            while step < self.nsteps:
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
            # schedule plan position
            start = block['step']
            device: List[int] = block['device']
            schedplan.add_block(
                Block(mid, span, memory, btype),
                device, start
            )
        return schedplan

