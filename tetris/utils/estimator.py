from typing import Union, Tuple, List, Set
import os

from cube.ir.cten import IRTensor
from cube.ir.operator import IRFwOperation
from cube.graph.segment import IRSegment
from cube.graph.function import IRGraphAnchor
from cube.graph.function.dimops import IRDimops, DimAnno
from .profiler import ProfileDataBase


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
        if idx < len(node.inputs()):
            if not isinstance(node.input(idx), IRTensor): continue
        for dim, edim in enumerate(eshape.dims):
            for identifier, reduce in zip(edim.identifiers, edim.reduces):
                if identifier in visited: continue
                visited.add(identifier)
                if identifier == '1' or node.anno.getlen(identifier) == 1: continue
                if reduce == DimAnno.ReduceType.Freeze: break
                dimlen = node.anno.getlen(identifier)
                algo = node.algorithms('dim')
                num = 2
                while num < min(16, dimlen) + 1:
                    if dimlen % num != 0:
                        num *= 2
                        continue
                    if not algo.satisfy(idx=idx, dim=dim, num=num): break
                    configs.append((idx, dim, num))
                    num *= 2
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

    def profile(self, node: IRFwOperation, train: bool) -> Tuple[float, int]:
        if node.name == 'multiref' or isinstance(node, IRGraphAnchor):
            return 0.0, 0, 0.0, 0
        trials = [None,] + get_partition_space(node)
        trials = Estimator.special_rules(node, trials)
        for config in trials:
            if config is None:
                num = 1
                infer_span, infer_mem, train_span, train_mem = self.database.profile(node, train)
            else:
                idx, dim, num = config
                print(f'> ... try node {node.name} with idx={idx}, dim={dim}, num={num}')
                sub_node = node.algorithms('dim').instantiate(idx=idx, dim=dim, num=num)[0]
                infer_span, infer_mem, train_span, train_mem = self.database.profile(sub_node, train)
                if isinstance(train_span, float): break
            if isinstance(train_span, float): break
        assert isinstance(train_span, float), f"Failed to profile: {node}"
        infer_span, infer_mem = infer_span * num, infer_mem * num
        train_span, train_mem = train_span * num, train_mem * num
        self.database.insert(node, infer_span, infer_mem, train_span, train_mem)
        return infer_span, infer_mem, train_span, train_mem


    def __call__(self, nodes_or_segment: Union[Tuple[IRFwOperation], IRSegment], 
                 train: bool = False):
        """
        Profile the computation cost of a subgraph

        @param nodes_or_segment Tuple[IRFwOperation] | IRSegment

        @return latency float: latency in ms
        @return memory int: memory in bytes
        """
        nodes = nodes_or_segment.nodes() if isinstance(nodes_or_segment, IRSegment) else nodes_or_segment
        memory, latency = 0.0, 0.0
        for node in nodes:
            if self.database.exist(node):
                infer_span, infer_mem, train_span, train_mem = self.database.query(node)
            else:
                infer_span, infer_mem, train_span, train_mem = self.profile(node, train)
            if train:
                memory += train_mem
                latency += train_span
            else:
                memory = max(memory, infer_mem)
                latency += infer_span
        return latency, memory

    def save(self):
        self.database.dump(self.cache_file, override=True)

    def special_rules(node, trials):
        # if node.name == 'embedding':  # for GPT
        #     trials = [(1, 0, 4),]
        # if node.name == 'window_attn':  # for Swin
        #     trials = [(1, 0, 4),]
        return trials
