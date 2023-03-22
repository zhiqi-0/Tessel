"""
Piper policy

https://openreview.net/attachment?id=-U9I0f2S7W&name=supplementary_material

The implementation is a little bit adapted to fit with cube's view
"""
from typing import List, Callable
from functools import partial

from cube.ir.operator import IRFwOperation, IRDataOperation
from cube.graph import IRGraph
from cube.graph.segment import IRSegment
from cube.graph.schedule.predefined import PredefinedSched

from tetris.runtime.estimator import Estimator
from tetris.runtime.utils import replica, annotate_structure
from tetris.runtime.division import TPS, layer_division


def Piper(graph: IRGraph, resource, nmicros: int,
          recompute: bool, tp_sprog: Callable, db_cache):
    """
    Piper policy

    @param graph IRGraph
    @param resource EnvResource
    @param nmicros int: number of microbatches
    @param recompute bool: whether perform recompute
    @param tp_sprog Callable: sProgram of tensor parallelism.
        Takes graph, segment, tp_devs

    @return graph IRGraph
    """
    annotate_structure(graph)
    nodes = tuple(graph.select(ntype=IRFwOperation))

    dl: IRDataOperation = graph.select(ntype=IRDataOperation)[0]
    mbs: int = dl.output(0).shape[dl.get_batch_dims()[0]]

    mem_limit = resource.gpus[0].memory
    print(f'> search [constraints]: device limitied memory: {mem_limit}')

    estimator = Estimator(db_cache)
    tps = partial(TPS, recompute=recompute, 
                  estimator=estimator, mem_limit=mem_limit)
    print(f'> search [initialize]: profiling model...')
    latency, memory  = estimator(nodes, train=True)
    print(f'> search [estimation]: single device latency: {latency} ms, memory: {memory/1024/1024/1024} GB')
    # save profiled database
    print(f'> search [dump]: saving profiled database...')
    estimator.save()

    min_cost, best_config = layer_division(nodes, resource.ngpus, tps, mbs)

    # ======================= instantiate plan ====================

    fstages = [stage_nodes for stage_nodes, dp, tp in best_config]
    graph.staging([snodes[0] for snodes in fstages])

    segments = graph.select(ntype=IRSegment, flatten=False)
    fsegments: List[IRSegment] = [seg for seg in segments if seg.isfw()]
    assert len(fsegments) == len(best_config), f"Expected {len(best_config)} stages in plan, but got {len(fsegments)}"

    devices = list(range(resource.ngpus))
    for sid, segment in enumerate(fsegments):
        _, dp, tp = best_config[sid]
        stage_devices, devices = devices[:dp*tp], devices[dp*tp:]
        assert len(stage_devices) == dp * tp
        # apply tensor parallelism
        print(f'> applying {tp}-way tensor parallelism')
        tp_sprog(graph, segment, stage_devices[:tp])
        # apply data parallelism
        if dp == 1: continue
        else: assert False

    assert len(devices) == 0, f'not all devices are used (remaining {len(devices)})'

    dls = graph.select(ntype=IRDataOperation)
    assert len(dls) == 1, f"tp_sprog is not allowed to partition/replicate dataloader"
    replica(graph, dls[0], fsegments[0].device)

    # print(graph.extra_repr())
    PredefinedSched.sched_1f1b(graph, nmicros, len(fsegments))
    return graph
