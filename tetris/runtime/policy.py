from typing import Union, List, Dict, Callable, Tuple

from cube.ir.operator import IRFwOperation
from cube.graph.graph import IRGraph, IRSegment
from cube.graph.schedule.schedplan import SchedulePlan as CSched

from tetris.schedplan import SchedPlan as TSched
from tetris.schedplan import Block as TBlock


def schedule(graph: IRGraph, tsched: Union[str, TSched], num_microbatches: int, blk2seg: Dict[TBlock, IRSegment]) -> CSched:
    """
    Translate a searched schedplan of Tetris into Cube SchedulePlan runtime

    @param graph IRGraph: staged IRGraph
    @param schedplan Union[TSched, str]: Tetris SchedPlan instance or file (saved in json format)
    
    @return schedplan CSched
    """
    tsched: TSched = tsched if isinstance(tsched, TSched) else TSched.load(tsched)
    nmicros = set(tblk.mid for tblk in tsched.all_blocks())
    assert len(nmicros) == max(nmicros) + 1, f"Microbatch index should be consecutive"
    nmicros = len(nmicros)

    segments: List[IRSegment] = graph.select(ntype=IRSegment, flatten=False)
    assert len(tsched.all_blocks()) == len(segments)

    csched = CSched(graph, num_microbatches)

    tsched.repetend = (tsched.nsteps, tsched.nsteps) if tsched.repetend is None else tsched.repetend
    rstart, rend = tsched.repetend

    # warmup
    for step in range(rstart):
        tblocks = tsched.blocks(step)
        for tblock in tblocks:
            csched.add_segment(blk2seg[tblock], tblock.mid, step)

    # steady
    rspan = rend - rstart
    for ofst in range(num_microbatches - nmicros):
        for step in range(rstart, rend):
            tblocks = tsched.blocks(step)
            for tblock in tblocks:
                csched.add_segment(
                    blk2seg[tblock], tblock.mid + ofst, step + rspan * ofst)
    
    # cooldown
    ofst = num_microbatches - nmicros
    for step in range(rend, tsched.nsteps):
        tblocks = tsched.blocks(step)
        for tblock in tblocks:
            csched.add_segment(blk2seg[tblock], tblock.mid + ofst, step + rspan * ofst)

    return csched


def policy(graph: IRGraph, resource,
           num_microbatches: int,
           premise: Callable[[IRGraph], Tuple[IRGraph, TSched]],
           tsched: Union[TSched, str]) -> IRGraph:
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
    graph, micro = premise(graph)
    assert resource.ngpus == micro.ndevs, f"Only support for same device number"
    assert not any(isinstance(node, IRFwOperation) for node in graph.nodes()), \
        "Premise should call graph.blocking() or graph.staging()"
    
    segments = graph.select(ntype=IRSegment, flatten=False)
    block2seg: Dict[TBlock, IRSegment] = {}
    for block, segment in zip(micro.chain_blocks(), segments):
        block2seg[block] = segment

    csched = schedule(graph, tsched, num_microbatches, block2seg)
    return graph
