"""
Piper policy

https://openreview.net/attachment?id=-U9I0f2S7W&name=supplementary_material

The implementation is a little bit adapted to fit with cube's view
"""
from typing import List, Callable, Tuple, Dict, Optional
import time

from cube.ir.operator import IRFwOperation
from cube.graph.function.anchor import IRGraphAnchor


def TPS(nodes: List[IRFwOperation], t: int, d: int, inflight: int,
        recompute: bool, estimator: Callable, mem_limit: Optional[int] = None) -> Optional[float]:
    """
    Get timer per sample (latency) of executing the nodes
    
    @param t int: tensor parallelism size
    @param d int: data parallelism size
    @param inflight int: in-flight micro-batch numbers
    @param estimator Callable: take (nodes, t, d) return cost and memory

    @return latency float: cost of executing these nodes
    """
    assert inflight > 0
    # return 1000, 1000  # Fake for debug
    latency, memory = estimator(nodes, train=True)
    # TODO: precise efficiency factor
    efficiency = 1.0 + 0.1 * (t-1)
    latency = latency / (t * d)
    latency *= efficiency
    if not recompute:
        memory = memory * inflight
    if isinstance(mem_limit, int) and memory > mem_limit:
        return None
    else:
        return latency


def iter_subgraph(nodes: Tuple[IRFwOperation], s: int):
    """
    Iterate sub-graphs of the nodes

    @param nodes Tuple[IRFwOperation]
    @param s int: number of stages

    @return (sub_graph1, sub_graph2) Tuple[Tuple[IRFwOp], Tuple[IRFwOp]]
    """
    assert s > 0
    if s > 1:
        # don't consider the head and tail to be anchor
        anchors = tuple(n for n in nodes[1:-1] if isinstance(n, IRGraphAnchor))
        assert len(anchors) >= s - 1
        for idx, anchor in enumerate(anchors):
            remain_anchor = len(anchors) - (idx + 1)
            # sub-problem of iter(sub_graph2, s-1) must iterable
            if remain_anchor < s - 2: continue
            nidx = nodes.index(anchor)
            sub_graph1, sub_graph2 = nodes[:nidx], nodes[nidx:]
            yield sub_graph1, sub_graph2
    else:
        # s == 1, take all
        yield nodes, ()


def DP(nodes: Tuple[IRFwOperation], k: int, s: int, tps: Callable,
       mbs: int, max_d: Optional[int] = None, max_t: Optional[int] = None,
       _cost : Dict = None, _config : Dict =None) -> Tuple[Dict, Dict]:
    """
    DP algorithm to search for balanced pipeline stage divisions by considering
    tensor parallelism and pipeline parallelism.
    
    cost[D][k][s] = min_{D' \in D} min_{t, d where t*d<=k} max( 
        TPS(D\D',t,d,s), cost[D'][k-d*t][s-1] )
    
    D: subgraph
    K: number of devices
    t: tensor parallelism size
    d: data parallelism size
    s: number of pipeline stages

    @param nodes Tuple[IRFwOperation]: sub-graph
    @param k int: number of devices
    @param s: number of pipeline stages
    @param tps: estimator 
        which takes nodes, tensor parallelism size, data parallelism size
        and in-flight number of microbatches, and outputs of 
        cost (latency in ms) and memory conumption (GB)
    @param mbs: micro-batch size
    @param max_d int: maximal data parallelism size constraint
    @param max_t int: maximal tensor parallelism size constraint

    @return costs Dict[( (IRCell,), k, s ), latency]
    @return config Dict[( (IRCell,), k, s ), [(IRCell,),] ]
    """
    nodes = nodes if isinstance(nodes, tuple) else tuple(nodes)
    key = (nodes, k, s)

    # initialize: dp[((), k, s)] = 0 for every k and s
    _cost = dict() if _cost is None else _cost
    _config = dict() if _config is None else _config
    max_d = k if max_d is None else max_d
    max_t = k if max_t is None else max_t
    if key in _cost: return _cost, _config

    # dp tatble boundary
    if len(nodes) == 0:
        _cost[key], _config[key] = 0, []
        return _cost, _config
    
    assert not (k == 0 or s == 0), \
        f"Illegal configuration: nodes: {len(nodes)} k={k}, s={s}: device number (k) cannot be smaller than pipeline stages (s)"
    assert k >= s, f"Expected k >= s but got k={k}, s={s}"

    # True for 1,2,4,8,16,...
    is_of_power2 = lambda n: (n & (n-1) == 0) and n != 0

    # construct dynamic programming table
    min_val = None  # None means no solution
    for sub1, sub2 in iter_subgraph(nodes, s):
        for d in range(1, min(k + 1, max_d + 1)):
            if mbs % d != 0: continue
            for t in range(1, min(k // d + 1, max_t + 1)):
                # only search for gpu# of power of 2
                if not is_of_power2(t * d): continue
                # guarantee sub-problem searchable
                if k - d * t < s - 1: continue
                # sub1 cost: s is also the in-flight microbatch number
                sub1_cost = tps(sub1, t, d, s)
                if sub1_cost is None: continue
                # sub2 cost
                DP(sub2, k-d*t, s-1, tps, mbs, max_d, max_t, _cost, _config)
                sub2_cost = _cost[(sub2, k-d*t, s-1)]
                if sub2_cost is None: continue
                # pipeline cost
                cost = max(sub1_cost, sub2_cost)
                config = [(sub1, d, t)] + _config[(sub2, k-d*t, s-1)]
                # update
                if min_val is None or cost < min_val:
                    min_val = cost
                    _config[(nodes, k, s)] = config

    _cost[key] = min_val
    return _cost, _config


def layer_division(nodes: Tuple[IRFwOperation], ndevs: int, tps: Callable, mbs: int, 
                   max_d: Optional[int]=None, max_t: Optional[int]=None):
    """
    DP algorithm to search for balanced pipeline stage divisions by considering
    tensor parallelism and pipeline parallelism.

    @param nodes List[IRFwOperation]: graph
    @param ndevs int: number of devices
    @param tps Callable: estimator 
        which takes nodes, tensor parallelism size, data parallelism size
        and in-flight number of microbatches, and outputs of 
        cost (latency in ms) and memory conumption (GB)
    @param mbs: micro-batch size
    @param max_d int: maximal data parallelism size constraint
    @param max_t int: maximal tensor parallelism size constraint

    @return min_cost: optimal latency of executing a microbatch
    @return best_config List[Tuple[SubGraph, int, int]]:
        [ ( (IRCell,), dp_size, tp_size ), ... ]
    """
    nodes = tuple(nodes)
    print(f'> search [search]: constructing dp tables...')
    tic = time.time()
    max_d = mbs if max_d is None else mbs
    max_d = min(max_d, mbs, ndevs)
    max_t = ndevs if max_t is None else max_t
    max_t = min(max_t, ndevs)
    cost, config = None, None
    for nstages in range(1, ndevs+1):
        cost, config = DP(nodes, ndevs, nstages, tps, mbs, 
                          max_d, max_t, cost, config)
    print(f'> search [search]: getting optimal results...')
    min_cost, best_config = None, None
    for nstages in range(1, ndevs+1):
        tcost = cost[(nodes, ndevs, nstages)]
        if tcost is None: continue
        if min_cost is None or tcost < min_cost:
            min_cost = tcost
            best_config = config[(nodes, ndevs, nstages)]
    toc = time.time()
    span = toc - tic
    print(f'> search [finish]: searching time: {span} s')
    print(f'> search [result]: minimal latency per microbatch {min_cost} ms')
    print(f'> search [result]: division plan:')
    for sidx, (stage_nodes, dp, tp) in enumerate(best_config):
        nlayers = len([n for n in stage_nodes if isinstance(n, IRGraphAnchor)])
        print(f'    stage-{sidx}: layers: {nlayers} | dp = {dp}, tp = {tp}')
    return min_cost, best_config
