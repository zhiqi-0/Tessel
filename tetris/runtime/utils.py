"""
Premise utils
"""
from typing import List, Tuple

from cube.graph.graph import IRGraph
from cube.ir.cten import IRCell
from cube.graph.function.dimops import IRDimops


def replica(graph: IRGraph, node: IRCell, devs: List[int]) -> List[IRDimops]:
    """Replicate a node"""
    sub_nodes = [node] if len(devs) == 1 else graph.replicate(node, len(devs))
    for devid, sub_node in zip(devs, sub_nodes):
        graph.assign(sub_node, devid)
    return sub_nodes


def tensor_parallelism(graph: IRGraph, node: IRDimops, idx: int, dim: int, devices: Tuple[int]) -> List[IRDimops]:
    """Tensor parallelism on a node"""
    sub_nodes = [node] if len(devices) == 1 \
        else graph.partition(node, node.algorithms('dim'), idx=idx, dim=dim, num=len(devices))
    for devid, sub_node in zip(devices, sub_nodes):
        graph.assign(sub_node, devid)
    return sub_nodes
