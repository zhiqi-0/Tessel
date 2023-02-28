from typing import Union, List, Dict, Callable, Optional
import time
import os

from cube.ir.operator import IRFwOperation, IRDataOperation
from cube.graph.graph import IRGraph, IRSegment
from cube.graph.schedule.schedplan import SchedulePlan as CSched

from tetris.schedplan import SchedPlan as TSched
from tetris.schedplan import Block as TBlock
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
    mids = set(tblk.mid for tblk in tsched.all_blocks())
    assert len(mids) == max(mids) + 1, f"Microbatch index should be consecutive"
    nmicros = max(mids) + 1

    csched = CSched(graph, num_microbatches)

    tsched.repetend = (tsched.nsteps, tsched.nsteps) if tsched.repetend is None else tsched.repetend
    rstart, rend = tsched.repetend

    # warmup
    for step in range(rstart):
        tblocks = tsched.blocks(step)
        for tblock in tblocks:
            csched.add_segment(blk2seg[tblock.gid], tblock.mid, step)

    # steady
    rspan = rend - rstart
    for ofst in range(num_microbatches-nmicros+1):
        for step in range(rstart, rend):
            tblocks = tsched.blocks(step)
            for tblock in tblocks:
                # print(f'adding segment {tblock}-{tsched.device(tblock)}, mid: {tblock.mid + ofst} to step {step + rspan * ofst}')
                endtime = step + tblock.span - 1
                csched.add_segment(
                    blk2seg[tblock.gid], tblock.mid + ofst, endtime + rspan * ofst)
    
    # cooldown
    ofst = num_microbatches - nmicros
    for step in range(rend, tsched.nsteps):
        tblocks = tsched.blocks(step)
        for tblock in tblocks:
            csched.add_segment(blk2seg[tblock.gid], tblock.mid + ofst, step + rspan * ofst)

    csched.finish()

    return csched


def policy(graph: IRGraph, resource,
           num_microbatches: int,
           premise: Callable[[IRGraph, int], TSched],
           memory_limits: List[int],
           save_dir: Optional[str] = None) -> IRGraph:
    """
    Compile policy in cube.

    @param graph IRGraph
    @param resource
    @param num_microbatches int
    @param premise Callable[[IRGraph], IRGraph]:
        Premise function to partition the whole graph into multiple sub-graphs
    @param tsched TSched: searched schedule plan

    @return graph IRGraph
    """
    assert len(memory_limits) == resource.ngpus, f"Only support for same device number"
    micro = premise(graph, resource.ngpus)
    assert not any(isinstance(node, IRFwOperation) for node in graph.nodes()), \
        "Premise should call graph.blocking() or graph.staging()"
    
    # replicate dataloader to all devices
    for dl in graph.select(ntype=IRDataOperation):
        if len(dl.device) > 0: continue 
        dls = graph.replicate(dl, resource.ngpus)
        for devid, dl in enumerate(dls):
            graph.assign(dl, devid)

    print(graph.extra_repr())

    # assign block sub-graph index
    for idx, block in enumerate(micro.chain_blocks()):
        block.gid = idx
    segments = graph.select(ntype=IRSegment, flatten=False)
    block2seg: Dict[TBlock, IRSegment] = {}
    for block, segment in zip(micro.chain_blocks(), segments):
        block2seg[block.gid] = segment
    
    # search
    # for nmicros in range(num_microbatches):
    nmicros = resource.ngpus
    micros: List[TSched] = [micro.copy(mid) for mid in range(nmicros)]
    # compose
    schedplans = Composer.compose(micros, memory_limits)
    assert len(schedplans) > 0, f"No schedule solution"
    tsched = schedplans[0]

    csched = schedule(graph, tsched, num_microbatches, block2seg)
    print(csched)

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
