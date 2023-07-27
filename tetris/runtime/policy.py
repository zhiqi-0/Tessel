from typing import Callable, List, Optional, Union, Dict
import os

from cube.graph.schedule.predefined import PredefinedSched

from cube.runtime.device import DeviceGroup
from cube.ir.operator import IRFwOperation, IRDataOperation
from cube.graph.graph import IRGraph, IRSegment
from cube.graph import IRGraph
from cube.graph.segment import IRSegment
from cube.graph.schedule.schedplan import SchedulePlan as CSched

from tetris.schedplan import SchedPlan as TSched
from tetris.schedplan import Block as TBlock
from tetris.runtime.utils import replica
from tetris.composer import Composer


def PAS1F1B(graph: IRGraph, resource, premise: Callable,
            nmicros: int, sched: str = '1f1b') -> IRGraph:

    memory = resource.gpus[0].memory
    print(f'> memory limit: {memory} bytes')
    premise(graph, resource.ngpus, memory)

    segments = graph.select(ntype=IRSegment, flatten=False)
    fsegments: List[IRSegment] = [seg for seg in segments if seg.isfw()]
    
    assert all(len(seg.device) < resource.ngpus for seg in fsegments)

    if sched == '1f1b':
        PredefinedSched.sched_1f1b(graph, nmicros, len(fsegments))
    elif sched == 'gpipe':
        PredefinedSched.sched_gpipe(graph, nmicros, len(fsegments))
    else:
        raise RuntimeError
    return graph


def PAS1F1BPlus(graph: IRGraph, resource, premise: Callable,
                nmicros: int) -> IRGraph:
    
    memory = resource.gpus[0].memory
    print(f'> memory limit: {memory} bytes')
    premise(graph, resource.ngpus, memory)

    segments = graph.select(ntype=IRSegment, flatten=False)
    fsegments: List[IRSegment] = [seg for seg in segments if seg.isfw()]
    nstages = len([seg for seg in fsegments if len(seg.device) < resource.ngpus])
    PredefinedSched.sched_1f1b_plus(graph, nmicros, nstages)
    return graph


def PASChimera(graph: IRGraph, resource, premise: Callable,
               nmicros: int) -> IRGraph:
    """Chimera Direct policy"""
    memory = resource.gpus[0].memory
    print(f'> memory limit: {memory} bytes')
    premise(graph, resource.ngpus, memory)

    segments = graph.select(ntype=IRSegment, flatten=False)
    fsegments: List[IRSegment] = [seg for seg in segments if seg.isfw()]
    assert len(fsegments) == 4

    PredefinedSched.sched_chimera_direct(graph, nmicros, len(fsegments))
    return graph


def _schedule(graph: IRGraph, tsched: Union[str, TSched], nmicros: int, blk2seg: Dict[TBlock, IRSegment]) -> CSched:
    """
    Translate a searched schedplan of Tetris into Cube SchedulePlan runtime

    @param graph IRGraph: staged IRGraph
    @param schedplan Union[TSched, str]: Tetris SchedPlan instance or file (saved in json format)
    
    @return schedplan CSched
    """
    tsched: TSched = tsched if isinstance(tsched, TSched) else TSched.load(tsched)
    # unroll the plan
    tsched = tsched.unroll(nmicros)
    csched = CSched(graph, nmicros)
    for step in range(tsched.nsteps):
        tblocks = tsched.blocks(step)
        for tblock in tblocks:
            csched.add_segment(blk2seg[tblock.gid], tblock.mid, step, tblock.span)
    csched.finish()
    return csched


def PASTetris(graph: IRGraph, resource,
              nmicros: int,
              premise: Callable[[IRGraph, int], TSched],
              max_inflight_blks: List[int],
              load_plan: Optional[str] = None,
              save_dir: Optional[str] = None) -> IRGraph:
    """
    Compile policy in cube.

    @param graph IRGraph
    @param resource EnvResource
    @param nmicros int
    @param premise Callable[[IRGraph], IRGraph, int]:
        Premise function to partition the whole graph into multiple sub-graphs
    @param max_inflight_blks List[int]: maximal inflight blocks of each device
    @param tsched TSched: searched schedule plan

    @return graph IRGraph
    """
    import torch, time

    assert len(max_inflight_blks) == resource.ngpus, f"Only support for same device number"
    memory = resource.gpus[0].memory
    print(f'> memory limit: {memory} bytes')
    micro: TSched = premise(graph, resource.ngpus, memory)
    assert not any(isinstance(node, IRFwOperation) for node in graph.nodes()), \
        "Premise should call graph.blocking() or graph.staging()"
    
    # replicate dataloader to all devices
    dls = graph.select(ntype=IRDataOperation)
    if len(dls) == 1 and len(dls[0].device) == 0:
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
    # for nmicros in range(nmicros):
    repetend_nmicros = micro.ndevs + 2
    # nmicros = resource.ngpus
    micros: List[TSched] = [micro.copy(mid) for mid in range(repetend_nmicros)]

    # single-node compose
    # schedplans = Composer.compose(micros, max_inflight_blks)
    # assert len(schedplans) > 0, f"No schedule solution"
    # tsched = schedplans[0]

    if load_plan is not None:
        print(f'> loading schedule plan from {load_plan}')
        tsched = TSched.load(load_plan)

    else:
        # due to non-deterministic behavior of z3-solver across nodes,
        # we follow a same plan from rank 0
        if DeviceGroup().rank == 0:
            schedplans = Composer.compose(micros, max_inflight_blks)
            assert len(schedplans) > 0, f"No schedule solution"
            tsched = schedplans[0]
            print(f'> saving searched plan in tsched.json...')
            tsched.save('tsched.json')
            state = tsched.getstate()
            # send schedule plan
            for rank in range(8, DeviceGroup().world_size, 8):
                torch.distributed.send(torch.tensor(state, dtype=torch.int).cuda(), rank)
        else:
            print('> waiting compose result from global rank 0...')
            blocks, devices = [], []
            for micro in micros:
                for blk in micro.all_blocks():
                    blocks.append(blk)
                    devices.append(micro.device(blk))
            state = torch.empty((repetend_nmicros, len(micro.all_blocks())+2), dtype=torch.int).cuda()
            torch.distributed.recv(state, 0)
            torch.cuda.synchronize()
            state = state.cpu().numpy()
            tsched = TSched(micro.ndevs)
            tsched.loadstate(blocks, devices, state)
    print(f'> get composed schedule:\n{tsched}')

    csched = _schedule(graph, tsched, nmicros, block2seg)
    # print(f'> {csched}')

    if save_dir is not None:
        from tetris.draw import Painter
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