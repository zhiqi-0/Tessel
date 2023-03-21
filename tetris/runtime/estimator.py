from typing import Union, Tuple, List, Set
import sys
import os

from cube.ir.operator import IRFwOperation
from cube.graph.segment import IRSegment
from cube.graph.function import IRGraphAnchor
from cube.graph.function.dimops import IRDimops, DimAnno
from tetris.runtime.profiler import ProfileDataBase


def get_partition_space(node: IRDimops) -> List[Tuple[int, int]]:
    """
    Get partition space of an IRDimops node

    @param node IRDimops
    @return space List[Tuple[int, int, int]]: tuple of configs: (idx, dim)
    """
    if not isinstance(node, IRDimops):
        return []
    visited : Set[str] = set()
    configs = []
    eshapes = node.anno.inputs() + node.anno.outputs()
    for idx, eshape in enumerate(eshapes):
        for dim, edim in enumerate(eshape.dims):
            for identifier, reduce in zip(edim.identifiers, edim.reduces):
                if identifier in visited: continue
                visited.add(identifier)
                if identifier == '1' or node.anno.getlen(identifier) == 1: continue
                if reduce == DimAnno.ReduceType.Freeze: break
                configs.append((idx, dim))
                break
    return configs


class Estimator:
    """
    Estimator to measture the computation / memory cost of a subgraph
    """

    def __init__(self, cache='./profile_database.json'):

        self.cache_file = cache
        reload = cache if os.path.exists(cache) else None
        self.database = ProfileDataBase(reload)


    def __call__(self, nodes_or_segment: Union[Tuple[IRFwOperation], IRSegment], 
                 train: bool=False):
        """
        Profile the computation cost of a subgraph

        @param nodes_or_segment Tuple[IRFwOperation] | IRSegment

        @return latency float: latency in ms
        @return memory int: memory in bytes
        """
        nodes = nodes_or_segment.nodes() if isinstance(nodes_or_segment, IRSegment) else nodes_or_segment
        memory, latency = 0.0, 0.0
        for node in nodes:
            if node.name == 'multiref' or isinstance(node, IRGraphAnchor):
                continue
            trials = [None,] + get_partition_space(node)
            for config in trials:
                if config is None:
                    num = 1
                    infer_span, infer_mem, train_span, train_mem = self.database.profile(node)
                else:
                    algo = node.algorithms('dim')
                    idx, dim = config
                    dimlen = node.input(idx).shape[dim]
                    num = 2
                    while num < dimlen:
                        if dimlen % num != 0:
                            dim *= 2
                            continue
                        if not algo.satisfy(idx=idx, dim=dim, num=num): break
                        print(f'> ... try node {node.name} with idx={idx}, dim={dim}, num={num} ')
                        sub_node = algo.instantiate(idx=idx, dim=dim, num=num)[0]
                        infer_span, infer_mem, train_span, train_mem = self.database.profile(sub_node)
                        if isinstance(train_span, float): break
                        num *= 2
                if isinstance(train_span, float):
                    break
            assert isinstance(train_span, float), f"Failed to profile: {node}"
            self.database.insert(node, infer_span * num, infer_mem * num, 
                                 train_span * num, train_mem * num)
            memory += train_mem * num
            latency += train_span * num
        return latency, memory

    def save(self):
        self.database.dump(self.cache_file, override=True)
