from typing import Tuple, Callable, Dict

from cube.graph import IRGraph
from cube.graph.segment import IRSegment
from cube.graph.function.anchor import IRGraphAnchor
from cube.ir.operator import IRFwOperation

from tetris.runtime.utils import tensor_parallelism, replica
from tetris.config import TetrisConfig
from tetris.runtime.core import staged_spmd, instantiate
from tetris.placement.block import blocking
from tetris.placement.stage import ParallelSpec


def tp_func(graph, fnode, devices: Tuple[int]):
    if fnode.name == 'self_attention' or fnode.name == 'feedforward':
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
    config.max_dp_size = 1
    config.min_pp_size = ngpus
    spec: ParallelSpec = staged_spmd(blocks, ngpus, mem_limit, config)
    print(spec)

    graph.blocking([stage.blocks[0].nodes[0] for stage in spec.stages])
    segments = graph.select(ntype=IRSegment, flatten=False)
    fsegments = [seg for seg in segments if seg.isfw()]

    instantiate(graph, ngpus, fsegments, spec, tp_func)
    return graph


def kshape(graph: IRGraph,
           ngpus: int,
           mbs: int,
           mem_limit: int,
           config: TetrisConfig):

    nodes = tuple(graph.select(ntype=IRFwOperation))
    text_anchor = graph.select(name='text')[0]
    text_anchor_idx = nodes.index(text_anchor)
    mm_anchor = graph.select(name='mm')[0]
    mm_anchor_idx = nodes.index(mm_anchor)

    img_nodes = nodes[:text_anchor_idx]
    txt_nodes = nodes[text_anchor_idx:mm_anchor_idx]
    mm_nodes = nodes[mm_anchor_idx:]

    config.min_pp_size = ngpus // 2
    assert ngpus // 2 > 0

    im_blocks = blocking(img_nodes, config.max_layer_num // 2)
    im_spec: ParallelSpec = staged_spmd(im_blocks, ngpus // 2, mem_limit, config)
    im_branch = [stage.blocks[0].nodes for stage in im_spec.stages]

    tx_blocks = blocking(txt_nodes, config.max_layer_num // 2)
    tx_spec: ParallelSpec = staged_spmd(tx_blocks, ngpus // 2, mem_limit, config)
    tx_branch = [stage.blocks[0].nodes for stage in tx_spec.stages]

    graph.blocking([ns[0] for ns in im_branch] + \
                   [ns[0] for ns in tx_branch] + \
                   [mm_nodes[0]])

    fsegments = graph.select(ntype=IRSegment, flatten=False)
    im_branch = fsegments[:ngpus // 2]
    tx_branch = fsegments[ngpus // 2: ngpus]
    mm_branch = fsegments[-1]
    for sid, segment in enumerate(im_branch):
        for node in segment.nodes():
            graph.assign(node, sid)
    for sid, segment in enumerate(tx_branch):
        for node in segment.nodes():
            graph.assign(node, sid + ngpus // 2)
    for node in mm_branch.nodes():
        if node.name == 'multiref' or isinstance(node, IRGraphAnchor):
            continue
        tp_func(graph, node, list(range(ngpus)))
    
    return graph


def PASTetris(graph: IRGraph,
              resource,
              mbs: int,
              nmicros: int,
              premise: Callable,
              config: TetrisConfig,
              load_sched: str) -> IRGraph:
    """policy entry for tetris.

    Args:
        graph (IRGraph)
        resource (EnvResource)
        mbs (int): micro-batch size
        nmicros (int): number of micro-batches
        premise (Callable): function to determine the graph.
            It takes inputs of (graph, num_devices, mem_limit, config)

    Returns:
        IRGraph
    """
    from tetris.runtime.policy import (_recompute, IRDataOperation,
        _create_tblocks, TBlock, _schedule, TSched)
    config = TetrisConfig() if config is None else config
    config.max_dp_size = mbs if config.max_dp_size is None \
        else min(mbs, config.max_dp_size)

    if config.recompute:
        _recompute(graph)
    
    mem_limit = resource.gpus[0].memory - 2 * 1024 * 1024 * 1024  # reserve 2GB
    print(f'> memory limit: {mem_limit} bytes')

    premise(graph, resource.ngpus, mbs, mem_limit, config)
    assert not any(isinstance(node, IRFwOperation) for node in graph.nodes()), \
        "Premise should call graph.blocking() or graph.staging()"

    # replicate dataloader to all devices
    for dl in graph.select(ntype=IRDataOperation):
        if len(dl.device) == 0:
            replica(graph, dl, list(range(resource.ngpus)))

    # assign block sub-graph index
    blocks = _create_tblocks(graph)
    segments = graph.select(ntype=IRSegment, flatten=False)
    # NOTE: ===> adapt to change mapping ordering
    segments = [segments[0], segments[2], segments[1], segments[3], segments[-1]]
    block2seg: Dict[TBlock, IRSegment] = {}
    for block, segment in zip(blocks, segments):
        block2seg[block.gid] = segment

    print(f'> loading schedule plan from {load_sched}')
    tsched = TSched.load(load_sched)

    print(f'> get composed schedule:\n{tsched}')
    csched = _schedule(graph, tsched, nmicros, block2seg)
    return graph