"""
Piper policy

https://openreview.net/attachment?id=-U9I0f2S7W&name=supplementary_material

The implementation is a little bit adapted to fit with cube's view
"""
from typing import List, Callable, Tuple, Dict
from functools import partial
import time

from cube.ir.operator import IRFwOperation
from cube.graph import IRGraph
from cube.graph.function.anchor import IRGraphAnchor

from tetris.runtime.estimator import Estimator


def TPS(nodes: List[IRFwOperation], t: int, d: int, inflight: int,
        recompute: bool, estimator: Callable, mem_limit) -> float:
    """
    Get timer per sample (latency) of executing the nodes
    
    @param t int: tensor parallelism size
    @param d int: data parallelism size
    @param inflight int: in-flight micro-batch numbers
    @param estimator Callable: take (nodes, t, d) return cost and memory

    @return latency float: cost of executing these nodes
    """
    assert inflight > 0
    latency, memory = estimator(nodes, train=True)
    # latency, memory = 1000, 1000  # Fake for debug
    # TODO: need efficiency factor
    latency = latency / (t * d)
    if not recompute:
        memory = memory * inflight
    return 1e12 if memory > mem_limit else latency


def DP(nodes: Tuple[IRFwOperation], k: int, s: int, tps: Callable,
       _cost : Dict = None, _config : Dict =None) -> Dict:
    """
    
    cost[D][k][s] = min_{D' \in D} min_{t, d where t*d<=k} max( 
        TPS(D\D',t,d,s), cost[D'][k-d*t][s-1] )
    
    D: subgraph
    K: number of devices
    t: tensor parallelism size
    d: data parallelism size
    s: number of pipeline stages

    """
    nodes = nodes if isinstance(nodes, tuple) else tuple(nodes)

    # initialize: dp[((), k, s)] = 0 for every k and s
    _cost = dict() if _cost is None else _cost
    _config = dict() if _config is None else _config

    # dp tatble border
    if len(nodes) == 0 or s == 0 or k == 0:
        _cost[(nodes, k, s)] = 0
        _config[(nodes, k, s)] = []
        return _cost, _config
    
    if k < s: # illegal space
        _cost[(nodes, k, s)] = 1e12
        _config[(nodes, k, s)] = []
        return _cost, _config
    
    is_or_power2 = lambda n: (n & (n-1) == 0) and n != 0
    if not is_or_power2(k):  # illegal space
        _cost[(nodes, k, s)] = 1e12
        _config[(nodes, k, s)] = []
        return _cost, _config


    # split point
    anchors = [node for node in nodes if isinstance(node, IRGraphAnchor)]
    assert len(anchors) >= s - 1, f"require more partition positions but got: {len(anchors)} anchors for {s} stages"
    if len(anchors) == 0:
        assert s == 1
        anchors += [None]
    # construct dynamic programming table
    print(f'enter searching config nodes# {len(nodes)}, k={k}, s={s}, anchors# {len(anchors)}')
    min_val = None
    for anchor in anchors:
        if s == 1:
            sub1, sub2 = nodes, ()
        else:
            idx = nodes.index(anchor)
            sub1, sub2 = nodes[:idx], nodes[idx:]
        for d in range(1, k + 1):
            for t in range(1, k // d + 1):
                print(f'check searching config d={d}, t={t}, k={k}, s={s}')
                # sub1 cost: s is also the in-flight microbatch number
                sub1_cost = tps(sub1, t, d, s)
                # sub2 cost
                DP(sub2, k-d*t, s-1, tps, _cost, _config)
                sub2_cost = _cost[(sub2, k-d*t, s-1)]
                # pipeline cost
                cost = max(sub1_cost, sub2_cost)
                config = [(sub1, d, t)] + _config[(sub2, k-d*t, s-1)]
                # update
                if min_val is None or cost < min_val:
                    min_val = cost
                    _config[(nodes, k, s)] = config

    print(f'finish searching config nodes# {len(nodes)}, k={k}, s={s}')
    assert min_val is not None
    _cost[(nodes, k, s)] = min_val
    return _cost, _config


def Piper(graph: IRGraph, resource, recompute: bool):

    estimator = Estimator()


    mem_limit = resource.gpus[0].memory
    print(f'> search [constraints]: device limitied memory: {mem_limit}')


    tps = partial(TPS, recompute=recompute, 
                  estimator=estimator, mem_limit=mem_limit)

    nodes = tuple(graph.select(ntype=IRFwOperation))

    print(f'> search [initialize]: profiling model...')
    _ = estimator(nodes, train=True)

    # save profiled database
    print(f'> search [dump]: saving profiled database...')
    estimator.save()

    tic = time.time()
    # apply dp algorithm
    print(f'> search [initialize]: initializing dp tables...')
    cost, config = None, None
    for nstages in range(1, resource.ngpus+1):
        cost, config = DP(nodes, resource.ngpus, nstages, tps, cost, config)
    
    # get best result
    print(f'> search [search]: search using DP algorithm...')
    min_cost, best_config = None, None
    for nstages in range(1, resource.ngpus+1):
        tcost = cost[(nodes, resource.ngpus, nstages)]
        if min_cost is None or tcost < min_cost:
            min_cost = tcost
            best_config = config[(nodes, resource.ngpus, nstages)]

    print(f'> search [result]: minimal latency {min_cost}')
    print(f'> search [result]: best config:')
    for sid, stage_nodes_dp_tp in enumerate(best_config):
        stage_nodes, dp, tp = stage_nodes_dp_tp
        stage_anchors = tuple(n for n in stage_nodes if isinstance(n, IRGraphAnchor))
        print(f'Stage: {sid} ({len(stage_anchors)} layers | dp={dp} | tp={tp})')
        for anchor in stage_anchors:
            print(anchor)
    
    toc = time.time()
    span = toc - tic
    print(f'> search [finish]: searching time: {span} s')
