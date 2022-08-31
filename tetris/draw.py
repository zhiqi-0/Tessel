"""
Example usage:

python -m tetris.draw --planfile plan.json --outfile plan.png
"""

from typing import Dict, Set, Tuple, List, Optional

import json
import numpy as np

import argparse


class Block:

    def __init__(self, mid: int, span: int, memory: float, btype: str):
        self.mid = mid
        self.span = span
        self.memory = memory
        assert btype in ('forward', 'backward')
        self.btype = btype


class SchedPlan:

    def __init__(self, ndevs: int, nsteps: int = 1) -> None:
        
        self._ndevs = ndevs
        self._maxsteps = 0
        self._blocks: Set[Block] = set()
        self._block_devices: Dict[Block, Tuple[int]] = dict()
        self._block_steps: Dict[Block, int] = dict()
        self._plans: List[List[Optional[Block]]] = [[] for _ in range(ndevs)]

    @property
    def nsteps(self) -> int:
        return self._maxsteps + 1

    @property
    def ndevs(self) -> int:
        return self._ndevs

    def add_block(self, block: Block, device: List[int], step: int):
        maxstep = step + block.span
        if maxstep > self._maxsteps:
            for devplan in self._plans:
                devplan += [None] * (maxstep - self._maxsteps)
            self._maxsteps = maxstep
        self._blocks.add(block)
        self._block_devices[block] = tuple(device)
        self._block_steps[block] = step
        for devid in device:
            for t in range(step, step + block.span):
                assert self._plans[devid][t] is None, f"Conflict block add on device {devid} at step {step}"
                self._plans[devid][t] = block

    def visualize(self, outfile: str):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from matplotlib.ticker import AutoMinorLocator
        plt.close('all')
        fig, ax = plt.subplots(figsize=(4 * self.nsteps // self.ndevs, 4))
        renderer = fig.canvas.get_renderer()

        # xaxis
        ax.set_xlim((0, self.nsteps-1))
        plt.xticks(
            ticks=np.arange(0.5, self.nsteps-0.5, 1.0, dtype=float),
            labels=np.arange(1, self.nsteps, 1, dtype=int)
        )
        minor_locator = AutoMinorLocator(2)
        plt.gca().xaxis.set_minor_locator(minor_locator)
        ax.xaxis.grid(which='minor', linestyle='--')
        # yaxis
        ax.set_ylim((0.5, self.ndevs+0.5))
        plt.yticks(np.arange(1, self.ndevs+1, 1, dtype=int))
        ax.invert_yaxis()

        fontsize = [40]
        txts = list()
        def draw_block(block: Block, position: Tuple[Tuple[int], int], fontsize):
            color = '#4472C4' if block.btype == "forward" else '#ED7D31'
            devs, step = position
            for dev in devs:
                rec = Rectangle((step, dev+0.5), block.span, 1, color=color, ec='black', lw=1.5)
                ax.add_artist(rec)
                rx, ry = rec.get_xy()
                cx = rx + rec.get_width() / 2.0
                cy = ry + rec.get_height() / 2.0
                anno = str(block.mid)
                txt = ax.text(x=cx, y=cy, s=anno, fontsize=40, ha='center', va='center', color='w')
                rbox = rec.get_window_extent(renderer)
                for fs in range(fontsize[0], 1, -2):
                    txt.set_fontsize(fs)
                    tbox = txt.get_window_extent(renderer)
                    if tbox.x0 > rbox.x0 and tbox.x1 < rbox.x1 and tbox.y0 > rbox.y0 and tbox.y1 < rbox.y1:
                        break
                fontsize[0] = min(fontsize[0], fs)
                txts.append(txt)

        for block in self._blocks:
            devs = self._block_devices[block]
            step = self._block_steps[block]
            draw_block(block, (devs, step), fontsize)

        # set fontsize to same
        fontsize = fontsize[0]
        for txt in txts:
            txt.set_fontsize(fontsize)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)
        plt.xlabel('Time Step', fontsize=fontsize)
        plt.ylabel('Device', fontsize=fontsize)
        plt.tight_layout()
        if outfile:
            plt.savefig(outfile)
        else:
            plt.show()

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='visualization')
    parser.add_argument('--planfile', type=str,
                        help='json file plan location')
    parser.add_argument('--outfile', type=str,
                        help='output file name')
    args = parser.parse_args()

    sched = SchedPlan.load(args.planfile)
    sched.visualize(args.outfile)
