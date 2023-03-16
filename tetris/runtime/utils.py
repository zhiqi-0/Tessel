"""
Premise utils
"""
from typing import List, Tuple
import numpy as np

from cube.graph.graph import IRGraph
from cube.graph.function import IRGraphAnchor
from cube.ir.operator import IRFwOperation
from cube.ir.cten import IRCell
from cube.graph.function.dimops import IRDimops

import more_itertools


def replica(graph: IRGraph, node: IRCell, devs: List[int]) -> List[IRDimops]:
    """Replicate a node"""
    sub_nodes = [node] if len(devs) == 1 else graph.replicate(node, len(devs))
    for devid, sub_node in zip(devs, sub_nodes):
        graph.assign(sub_node, devid)
    return sub_nodes


def tp(graph: IRGraph, node: IRDimops, devs: List[int], **configs) -> List[IRDimops]:
    """Tensor parallelism on a node"""
    sub_nodes = [node] if len(devs) == 1 \
        else graph.partition(node, node.algorithms('dim'), **configs)
    for devid, sub_node in zip(devs, sub_nodes):
        graph.assign(sub_node, devid)
    return sub_nodes


def annotate_structure(graph: IRGraph) -> List[Tuple[IRFwOperation]]:
    """Annotate graph stucture in generated code"""
    anchors = graph.select(ntype=IRGraphAnchor)
    for idx, anchor in enumerate(anchors):
        nidx = graph.index(anchor)
        graph.node(nidx + 1).comment = f'===> split position {idx}'
    fnodes = graph.select(ntype=IRFwOperation)
    subgraphs = more_itertools.split_when(fnodes, lambda n: isinstance(n, IRGraphAnchor))
    return subgraphs


def layer_division_rules(nstages: int, block_comp_cost: List[float],
                         adaptive: bool = True,
                         limits: Tuple[int] = None) -> List[Tuple[int, int]]:
    """
    Layer division

    @param nstages int: number of stages
    @param adaptive bool: reallocate pipeline
    @param limits: Maximal layer number constraints due to memory consumption

    @return 
    """
    nlayers = len(block_comp_cost)
    if nstages == 1: return [(0, nlayers),]
    divisions: List[Tuple[int, int]] = []
    start, end = 0, 1
    # uniform partition
    if not adaptive:
        for sid in range(nstages):
            addone = 1 if sid < len(block_comp_cost) % nstages else 0
            end = start + len(block_comp_cost) // nstages + addone
            divisions.append((start, end))
            start = end
        assert end == len(block_comp_cost)
    # adaptive partition
    else:
        # ====================== Rule ========================
        # maxmial layer number constraints due to memory consumption
        limits = [None] * nstages if limits is None else limits
        # ====================================================
        remain_time = sum(block_comp_cost)
        start, end = 0, 1
        for sid in range(nstages):
            budget = remain_time / (nstages - sid)
            accum = block_comp_cost[start]
            while end < len(block_comp_cost):
                if limits[sid] is not None and (end - start) == limits[sid]:
                    break
                if budget - accum < 0.5 * block_comp_cost[end]:
                    break
                accum += block_comp_cost[end]
                end += 1
            remain_time -= accum
            if sid == nstages - 1:
                end = len(block_comp_cost)
            divisions.append((start, end))
            start, end = end, end + 1
    return divisions