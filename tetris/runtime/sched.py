from typing import Union, List, Dict, Callable, Optional
import time
import os

from cube.ir.operator import IRFwOperation, IRDataOperation
from cube.graph.graph import IRGraph, IRSegment
from cube.graph.schedule.schedplan import SchedulePlan as CSched

from tetris.schedplan import SchedPlan as TSched
from tetris.schedplan import Block as TBlock
from tetris.runtime.utils import annotate_structure, replica
from tetris.composer import Composer
from tetris.draw import Painter


def schedule(graph: IRGraph, tsched: Union[str, TSched], num_microbatches: int, blk2seg: Dict[TBlock, IRSegment]) -> CSched:
    """
    Translate a searched schedplan of Tetris into Cube SchedulePlan runtime

    @param graph IRGraph: staged IRGraph
    @param schedplan Union[TSched, str]: Tetris SchedPlan instance or file (saved in json format)
    
    @return schedplan CSched
    """
    tsched: TSched = tsched if isinstance(tsched, TSched) else TSched.load(tsched)
    # unroll the plan
    tsched = tsched.unroll(num_microbatches)
    csched = CSched(graph, num_microbatches)
    for step in range(tsched.nsteps):
        tblocks = tsched.blocks(step)
        for tblock in tblocks:
            csched.add_segment(blk2seg[tblock.gid], tblock.mid, step, tblock.span)
    csched.finish()
    return csched


def tsched(graph: IRGraph, resource,
           num_microbatches: int,
           premise: Callable[[IRGraph, int], TSched],
           max_inflight_blks: List[int],
           save_dir: Optional[str] = None) -> IRGraph:
    """
    Compile policy in cube.

    @param graph IRGraph
    @param resource EnvResource
    @param num_microbatches int
    @param premise Callable[[IRGraph], IRGraph, int]:
        Premise function to partition the whole graph into multiple sub-graphs
    @param max_inflight_blks List[int]: maximal inflight blocks of each device
    @param tsched TSched: searched schedule plan

    @return graph IRGraph
    """
    assert len(max_inflight_blks) == resource.ngpus, f"Only support for same device number"
    memory = resource.gpus[0].memory
    print(f'> memory limit: {memory} bytes')
    micro: TSched = premise(graph, resource.ngpus, memory)
    assert not any(isinstance(node, IRFwOperation) for node in graph.nodes()), \
        "Premise should call graph.blocking() or graph.staging()"
    
    # replicate dataloader to all devices
    dls = graph.select(ntype=IRDataOperation)
    assert len(dls) == 1, f"Only consider one dataloader in IRGraph"
    dl = dls[0]
    dl_devices = set()
    for segment in graph.select(ntype=IRSegment, flatten=False):
        if graph.depends(dl, segment):
            dl_devices.update(segment.device)
    dl_devices = sorted(dl_devices)
    replica(graph, dl, dl_devices)

    # print(graph.extra_repr())

    # assign block sub-graph index
    for idx, block in enumerate(micro.chain_blocks()):
        block.gid = idx
    segments = graph.select(ntype=IRSegment, flatten=False)
    block2seg: Dict[TBlock, IRSegment] = {}
    for block, segment in zip(micro.chain_blocks(), segments):
        block2seg[block.gid] = segment
    
    # search
    # for nmicros in range(num_microbatches):
    nmicros = micro.ndevs
    # nmicros = resource.ngpus
    micros: List[TSched] = [micro.copy(mid) for mid in range(nmicros)]
    # compose
    schedplans = Composer.compose(micros, max_inflight_blks)
    assert len(schedplans) > 0, f"No schedule solution"
    tsched = schedplans[0]

    csched = schedule(graph, tsched, num_microbatches, block2seg)
    # print(f'> {csched}')

    if save_dir is not None:
        now = time.strftime("%Y-%m-%d-%H-%M", time.gmtime())
        Painter.visualize(
            micros[0],
            os.path.join(save_dir, f"premise.{now}.png"))
        Painter.visualize(
            tsched,
            os.path.join(save_dir, f"schedule.{now}.png"))
        Painter.visualize(
            tsched.extract(tsched.repetend[0], tsched.repetend[1]), 
            os.path.join(save_dir, f"repetend.{now}.png"))

    return graph
