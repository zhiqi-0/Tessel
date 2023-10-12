from typing import Tuple
import warnings
import more_itertools as mitr
import itertools

from cube.graph import IRGraph
from cube.graph.segment import IRSegment
from cube.graph.function.anchor import IRGraphAnchor
from cube.graph.function.dimops import IRDimops
from cube.ir.operator import IRFwOperation

from tetris.runtime.utils import tensor_parallelism, replica
from tetris.config import TetrisConfig
from tetris.runtime.core import staged_spmd, instantiate
from tetris.placement.block import blocking
from tetris.placement.stage import ParallelSpec


def tp_func(graph, fnode, devices: Tuple[int]):
    if fnode.name == 'embedding' and fnode.input(1).shape[0] > 10240:
        sub_nodes = tensor_parallelism(graph, fnode, idx=1, dim=0, devices=devices)
    elif fnode.name == 'self_attention' or fnode.name == 'feedforward':
        sub_nodes = tensor_parallelism(graph, fnode, idx=1, dim=0, devices=devices)
    elif fnode.name == 'cross_attention':
        sub_nodes = tensor_parallelism(graph, fnode, idx=2, dim=0, devices=devices)
    elif fnode.name == 'linear':  # the last embedding linear
        sub_nodes = tensor_parallelism(graph, fnode, idx=1, dim=0, devices=devices)
    else:
        sub_nodes = replica(graph, fnode, devices)
    return sub_nodes


def vshape(graph: IRGraph,
           ngpus: int,
           mbs: int,
           mem_limit: int,
           config: TetrisConfig):
    
    fnodes = graph.select(ntype=IRFwOperation)
    blocks = blocking(fnodes, config.max_layer_num)
    config.max_tp_size = ngpus - 1
    config.max_dp_size = 1
    spec: ParallelSpec = staged_spmd(blocks, ngpus, mem_limit, config)
    print(spec)

    graph.staging([stage.blocks[0].nodes[0] for stage in spec.stages])
    segments = graph.select(ntype=IRSegment, flatten=False)
    fsegments = [seg for seg in segments if seg.isfw()]

    instantiate(graph, ngpus, fsegments, spec, tp_func)
    return graph


def xshape(graph: IRGraph,
           ngpus: int,
           mbs: int,
           mem_limit: int,
           config: TetrisConfig):

    assert ngpus % 4 == 0
    tp_size = ngpus // 4

    fnodes = graph.select(ntype=IRFwOperation)
    blocks = blocking(fnodes, config.max_layer_num)

    config.max_tp_size = 1
    config.max_dp_size = 1
    spec: ParallelSpec = staged_spmd(
        blocks, ngpus, mem_limit * tp_size, config)

    for stage in spec.stages:
        stage.tp_size = tp_size

    graph.staging([stage.blocks[0].nodes[0] for stage in spec.stages])
    segments = graph.select(ntype=IRSegment, flatten=False)
    fsegments = [seg for seg in segments if seg.isfw()]

    for sid, segment in enumerate(fsegments):
        dp_devs = {0: [0, 3], 1: [1, 2], 2: [2, 1], 3: [3, 0]}[sid]
        mb1_devs = [dp_devs[0] * tp_size + i for i in range(tp_size)]
        mb2_devs = [dp_devs[1] * tp_size + i for i in range(tp_size)]
        for node in segment.select(ntype=IRFwOperation):
            # 2-way data parallelism
            if isinstance(node, IRDimops):
                algo = node.algorithms('dim')
                dim = node.input(0).shape.index(mbs)
                mb1, mb2 = graph.partition(node, algo, idx=0, dim=dim, num=2)
            else:
                warnings.warn(
                    f'node: {node}\ncannot split node into two micro-batches, use replicate instead.',
                    category=RuntimeWarning, stacklevel=0)
                mb1, mb2 = graph.replicate(node, times=2)
            # tensor parallelism
            tp_func(graph, mb1, mb1_devs)
            tp_func(graph, mb2, mb2_devs)
    return graph


def nnshape(graph: IRGraph,
           ngpus: int,
           mbs: int,
           mem_limit: int,
           config: TetrisConfig) -> IRGraph:

    fnodes = graph.select(ntype=IRFwOperation)
    layers = list(mitr.split_before(fnodes, lambda n : isinstance(n, IRGraphAnchor)))
    nlayers = len(layers)
    # get embedding layers
    embed2 = layers.pop(nlayers // 2)
    embed1 = layers.pop(0)
    # pipeline 
    encoders = layers[:len(layers) // 2]
    decoders = layers[len(layers) // 2:]

    transformers = list(itertools.chain(*(encoders+decoders)))
    blocks = blocking(transformers, config.max_layer_num)
    config.max_dp_size = 1
    config.min_pp_size = 4
    spec: ParallelSpec = staged_spmd(blocks, ngpus, mem_limit, config)
    print(spec)

    fstages = [embed1] + [spec.stages[0].blocks[0].nodes] + [embed2] + \
              [s.blocks[0].nodes for s in spec.stages[1:]]
    # re-schedule nodes
    graph.order(embed2, spec.stages[1].blocks[0].nodes[0:2])
    graph.staging(tuple(stage[0] for stage in fstages))
    segments = graph.select(ntype=IRSegment, flatten=False)
    fsegments = [seg for seg in segments if seg.isfw()]

    # full tensor parallelism
    devices = list(range(ngpus))
    for node in fsegments[0].nodes():
        tp_func(graph, node, devices)
    for node in fsegments[2].nodes():
        tp_func(graph, node, devices)

    # vshape
    xcoders = fsegments[1:2] + fsegments[3:]
    instantiate(graph, ngpus, xcoders, spec, tp_func)
    return graph
