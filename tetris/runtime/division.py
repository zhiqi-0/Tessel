"""
Piper policy

https://openreview.net/attachment?id=-U9I0f2S7W&name=supplementary_material

The implementation is a little bit adapted to fit with cube's view
"""
from typing import List, Callable, Tuple, Dict, Optional
import time
import os

from cube.ir.cten import IRTensor
from cube.ir.operator import IRFwOperation
from cube.graph.function.anchor import IRGraphAnchor

from tetris.runtime.layer_op import IRLayerOp, cluster_to_layer_ops


PARAM_LIMIT = os.environ.get('PARAM_LIMIT', None)  # in GB


def TPS(nodes: List[IRLayerOp], d: int, t: int, inflight: int,
        recompute: bool, estimator: Callable, mem_limit: int) -> Optional[float]:
    """
    Get timer per sample (latency) of executing the nodes
    
    @param d int: data parallelism size
    @param t int: tensor parallelism size
    @param inflight int: in-flight micro-batch numbers
    @param estimator Callable: take (nodes, t, d) return cost and memory

    @return latency Optional[float]: cost of executing these nodes
    @return memory int: estimated memory in bytes
    """
    # return 1000, 1000  # Fake for debug
    # TODO: precise efficiency factor and memory preservation
    tp_mem_efficiency = 1.0 + 0.10 * (t-1)
    tp_com_efficienty = 1.0 + 0.10 * (t-1)
    mem_preserve = 2 * 1024 * 1024 * 1024 # 2GB
    assert t > 0 and d > 0 and inflight > 0

    mem_limit -= mem_preserve

    total_act_memory = 0
    total_latency = 0
    for layer_op in nodes:
        latency, act_memory = estimator(layer_op.nodes, train=True)
        # activation memory
        act_memory = act_memory / (t * d) * tp_mem_efficiency
        # recompute granularity: per stage
        # inflight = 1 if recompute else inflight
        # total_act_memory += act_memory * inflight

        # recompute granularity: per layer
        total_act_memory = max(total_act_memory, act_memory) if recompute \
            else total_act_memory + act_memory * inflight

        # latency
        if recompute:
            latency = latency / 3 * 4  # suppose forward:backward=1:2
        latency = latency / (t * d) * tp_com_efficienty
        total_latency += latency

    # parameter size
    param_size = 0
    for layer_op in nodes:
        for node in layer_op.nodes:
            for tensor in node.inputs():
                if isinstance(tensor, IRTensor) and tensor.is_attr():
                    # too large weight will bring memory fragment
                    factor = 1 if tensor.byte_size() // t <= 1.5 * 1024 * 1024 * 1024 else 1.5
                    param_size += tensor.byte_size() * factor
    # consider gradient and adam optimizer (totally 3x param size)
    param_size = param_size * 4 / t
    total_memory = param_size + total_act_memory
    if PARAM_LIMIT is not None:
        if param_size >= int(PARAM_LIMIT) * 1024 * 1024 * 1024:
            return None, total_memory
    return total_latency if total_memory < mem_limit else None, total_memory


def iter_subgraph(nodes: Tuple[IRLayerOp], s: int):
    """
    Iterate sub-graphs of the nodes

    @param nodes Tuple[IRFwOperation]
    @param s int: number of stages

    @return (sub_graph1, sub_graph2) Tuple[Tuple[IRFwOp], Tuple[IRFwOp]]
    """
    assert s > 0
    if s > 1:
        # don't consider the head and tail to be anchor
        assert len(nodes) >= s - 1
        for idx in range(len(nodes)):
            remain_nodes = len(nodes) - (idx + 1)
            # sub-problem of iter(sub_graph2, s-1) must iterable
            if remain_nodes < s - 2: continue
            sub_graph1, sub_graph2 = nodes[:idx+1], nodes[idx+1:]
            yield sub_graph1, sub_graph2
    else:
        # s == 1, take all
        yield nodes, ()


def DP(nodes: Tuple[IRLayerOp], k: int, s: int, tps: Callable,
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
                # constraints: all devices must be used
                if s == 1 and d * t != k: continue
                # only search for gpu# of power of 2
                if not is_of_power2(t * d): continue
                # guarantee sub-problem searchable
                if k - d * t < s - 1: continue
                # sub1 cost: s is also the in-flight microbatch number
                sub1_cost, _ = tps(sub1, d, t, s)
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
                   max_d: Optional[int]=None, max_t: Optional[int]=None, max_p: Optional[int]=None):
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
    nodes: List[IRLayerOp] = cluster_to_layer_ops(nodes)
    nodes = tuple(nodes)
    print(f'> search [search]: constructing dp tables ({len(nodes)} layer ops)...')
    tic = time.time()
    max_d = mbs if max_d is None else mbs
    max_d = min(max_d, mbs, ndevs)
    max_t = ndevs if max_t is None else max_t
    max_t = min(max_t, ndevs)
    max_p = ndevs if max_p is None else min(max_p, ndevs)
    cost, config = None, None
    for nstages in range(1, max_p+1):
        cost, config = DP(nodes, ndevs, nstages, tps, mbs, 
                          max_d, max_t, cost, config)
    print(f'> search [search]: getting optimal results...')
    min_cost, best_config = None, None
    for nstages in range(1, max_p+1):
        tcost = cost[(nodes, ndevs, nstages)]
        if tcost is None: continue
        if min_cost is None or tcost < min_cost:
            min_cost = tcost
            best_config = config[(nodes, ndevs, nstages)]
    assert best_config is not None, f"no solution"
    toc = time.time()
    span = toc - tic
    print(f'> search [finish]: searching time: {span} s')
    print(f'> search [result]: minimal latency per microbatch {min_cost} ms')
    print(f'> search [result]: division plan:')
    for sidx, (layer_ops, dp, tp) in enumerate(best_config):
        # tp = 1 if sidx == 1 else tp
        nlayers = len(layer_ops)
        est_latency, est_mem = tps(layer_ops, dp, tp, len(best_config) - sidx)
        est_latency, est_mem = round(est_latency, 2), round(est_mem / 1024 / 1024 / 1024, 2)
        print(f'    stage-{sidx}: layers: {nlayers} | dp = {dp}, tp = {tp} | est latency: {est_latency} ms, est memory: {est_mem} GB')
    # unpack to IRFwOperations
    best_config_fwops = []
    for layer_ops, dp, tp in best_config:
        fwops = []
        for layer_op in layer_ops:
            fwops += layer_op.nodes
        best_config_fwops.append((fwops, dp, tp))
    return min_cost, best_config_fwops
