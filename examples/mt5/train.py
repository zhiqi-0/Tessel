
from typing import List
import torch
from functools import partial
import warnings
import itertools

from examples.mt5.model import Config, mT5, mT5DataLoader

import cube
from cube.profiler.timer import CudaTimer, print_each_rank
from cube.profiler.memory import memory_summary

from cube.graph import IRGraph
from cube.graph.segment import IRSegment
from cube.graph.function.anchor import IRGraphAnchor
from cube.ir.operator import IRFwOperation, IRDataOperation
from cube.runtime.device import DeviceGroup
from cube.graph.function.dimops import IRDimops

from tetris.runtime.utils import tp, replica, annotate_structure
from tetris.runtime.division import TPS, layer_division
from tetris.runtime.policy import PAS1F1B, PAS1F1BPlus, PASChimera, PASTetris
from tetris.runtime.estimator import Estimator
from tetris.schedplan import SchedPlan as TSched
from tetris.schedplan import Block as TBlock
from tetris.runtime.flags import SearchFlag

import argparse

parser = argparse.ArgumentParser(description='mT5 Train')
parser.add_argument('--fp16', action='store_true', default=False)
parser.add_argument('--mbs', type=int, default=1, help='micro-batch size')
parser.add_argument('--gbs', type=int, default=256, help='global batch size')
# arch
parser.add_argument('--layers', type=int, required=True, help='number of encoder / decoder layers')
parser.add_argument('--hidden', type=int, required=True)
parser.add_argument('--heads', type=int, required=True)
parser.add_argument('--seqlen', type=int, required=True)
parser.add_argument('--vocab', type=int, required=True)
# policy
parser.add_argument('--premise', type=str,
                    choices=['1f1b', 'nnshape', 'gpipe', 'tp', 'chimera', 'nnshape_eager', '1f1b+'],
                    help='premise shape')
parser.add_argument('--max-pp', type=int, default=32,
                    help='max number of pipeline stages')
parser.add_argument('--max-tp', type=int, default=32,
                    help='max size of tensor paralllelism')
parser.add_argument('--recompute', action='store_true', default=False)
# log save
parser.add_argument('--save', type=str, default=None,
                    help='folder for save searched results.')
parser.add_argument('--load-tsched', type=str, default=None,
                    help='load searched tetris schedule from file')
parser.add_argument('--db-cache', type=str, default='mt5_db.json',
                    help='profiled database save file')
args = parser.parse_args()

cube.init()
print_each_rank(str(args), rank_only=0)
print_each_rank(str(SearchFlag()), rank_only=0)

if SearchFlag.mem_limit is not None:
    fraction = SearchFlag.mem_limit * 1024 * 1024 * 1024 / torch.cuda.get_device_properties(0).total_memory
    print_each_rank(f'> setting memory fraction: {fraction}')
    torch.cuda.set_per_process_memory_fraction(fraction)


# ========================= parallelisms =================================

def stage_tp(graph: IRGraph, nodes: List[IRFwOperation], devs: List[int]):
    """Tensor parallelsim a stage"""
    ndevs = len(devs)
    for fnode in nodes:
        if fnode.name == 'multiref' or isinstance(fnode, IRGraphAnchor): continue
        if fnode.name == 'embedding' and fnode.input(1).shape[0] > 10240:
            tp(graph, fnode, devs, idx=1, dim=0, num=ndevs)
        elif fnode.name in ('self_attention', 'feedforward'):
            tp(graph, fnode, devs, idx=1, dim=0, num=ndevs)
        elif fnode.name == 'cross_attention':
            tp(graph, fnode, devs, idx=2, dim=0, num=ndevs)
        elif fnode.name == 'linear':  # the last embedding linear
            tp(graph, fnode, devs, idx=1, dim=0, num=ndevs)
        else:
            replica(graph, fnode, devs)


def stage_dp(graph: IRGraph, nodes: List[IRFwOperation], devs: List[int]):
    """data parallelism a stage"""
    assert args.mbs % len(devs) == 0
    ndevs = len(devs)
    for fnode in nodes:
        if fnode.name == 'multiref' or isinstance(fnode, IRGraphAnchor): continue
        try:
            dim = fnode.input(0).shape.index(args.mbs)
            tp(graph, fnode, devs, idx=0, dim=dim, num=ndevs)
        except:
            warnings.warn(f'fail to partition node: {fnode.name} using data parallelism',
                          category=RuntimeWarning, stacklevel=0)
            replica(graph, fnode, devs)

def full_tp(graph: IRGraph, resource):

    transformers = annotate_structure(graph)
    if args.recompute:
        for transformer in transformers:
            graph.recompute(transformer)

    devs = list(range(resource.ngpus))
    stage_tp(graph, graph.select(ntype=IRFwOperation), devs)
    for dl in graph.select(ntype=IRDataOperation):
        replica(graph, dl, devs)
    return graph

# ========================= parallelisms =================================

def premise_vshape(graph: IRGraph, ndevs: int, mem_limit: int):
    """VShape policy"""
    transformers = annotate_structure(graph)
    if args.recompute:
        for transformer in transformers:
            graph.recompute(transformer)
    nodes = tuple(graph.select(ntype=IRFwOperation))

    dl: IRDataOperation = graph.select(ntype=IRDataOperation)[0]
    mbs: int = dl.output(0).shape[dl.get_batch_dims()[0]]

    mem_limit = mem_limit if SearchFlag.mem_limit is None else SearchFlag.mem_limit * 1024 * 1024 * 1024
    print(f'> search [constraints]: device limitied memory: {mem_limit}')

    estimator = Estimator(args.db_cache)
    tps = partial(TPS, recompute=args.recompute, 
                  estimator=estimator, mem_limit=mem_limit)
    print(f'> search [initialize]: profiling model...')
    latency, memory  = estimator(nodes, train=True)
    print(f'> search [estimation]: single device latency: {latency} ms, memory: {memory/1024/1024/1024} GB')
    # save profiled database
    print(f'> search [dump]: saving profiled database...')
    estimator.save()

    min_cost, best_config = layer_division(
        nodes, ndevs, tps, mbs,
        max_t=min(ndevs-1, args.max_tp),
        max_d=1,
        max_p=args.max_pp)

    # ======================= instantiate plan ====================

    fstages = [stage_nodes for stage_nodes, dp, tp in best_config]
    graph.staging([snodes[0] for snodes in fstages])

    segments = graph.select(ntype=IRSegment, flatten=False)
    fsegments: List[IRSegment] = [seg for seg in segments if seg.isfw()]
    assert len(fsegments) == len(best_config), f"Expected {len(best_config)} stages in plan, but got {len(fsegments)}"

    devices = list(range(ndevs))
    for sid, segment in enumerate(fsegments):
        _, dp, tp = best_config[sid]
        stage_devices, devices = devices[:dp*tp], devices[dp*tp:]
        assert len(stage_devices) == dp * tp
        # apply tensor parallelism
        print(f'> applying {tp}-way tensor parallelism')
        stage_tp(graph, segment.select(ntype=IRFwOperation), stage_devices[:tp])
        # apply data parallelism
        if dp == 1: continue
        else: assert False

    assert len(devices) == 0, f'not all devices are used (remaining {len(devices)})'

    replica(graph, dl, list(range(ndevs)))


def premise_xshape(graph: IRGraph, ndevs: int, mem_limit: int):
    """Chimera Direct policy"""
    transformers = annotate_structure(graph)
    if args.recompute:
        for transformer in transformers:
            graph.recompute(transformer)
    nodes = tuple(graph.select(ntype=IRFwOperation))

    dl: IRDataOperation = graph.select(ntype=IRDataOperation)[0]
    mbs: int = dl.output(0).shape[dl.get_batch_dims()[0]]

    mem_limit = mem_limit if SearchFlag.mem_limit is None else SearchFlag.mem_limit * 1024 * 1024 * 1024
    print(f'> search [constraints]: device limitied memory: {mem_limit}')

    # for chimera, we constrain the tensor parallelism size of each pipeline to be same
    # due to its special scheduling polies.
    assert ndevs % 4 == 0
    tp_size = ndevs // 4
    mem_limit = mem_limit * tp_size

    estimator = Estimator(args.db_cache)
    tps = partial(TPS, recompute=args.recompute, 
                  estimator=estimator, mem_limit=mem_limit)
    print(f'> search [initialize]: profiling model...')
    latency, memory  = estimator(nodes, train=True)
    print(f'> search [estimation]: single device latency: {latency} ms, memory: {memory/1024/1024/1024} GB')
    # save profiled database
    print(f'> search [dump]: saving profiled database...')
    estimator.save()

    min_cost, best_config = layer_division(
        nodes, 4, tps, mbs, max_t=1, max_d=1)

    # ======================= instantiate plan ====================

    fstages = [stage_nodes for stage_nodes, dp, tp in best_config]
    graph.staging([snodes[0] for snodes in fstages])

    segments = graph.select(ntype=IRSegment, flatten=False)
    fsegments: List[IRSegment] = [seg for seg in segments if seg.isfw()]
    assert len(fsegments) == len(best_config), f"Expected {len(best_config)} stages in plan, but got {len(fsegments)}"

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
                if not isinstance(node, IRGraphAnchor):
                    warnings.warn(
                        f'node: {node}\ncannot split node into two micro-batches, use replicate instead.',
                        category=RuntimeWarning, stacklevel=0)
                mb1, mb2 = graph.replicate(node, times=2)
            # tensor parallelism
            stage_tp(graph, [mb1], mb1_devs)
            stage_tp(graph, [mb2], mb2_devs)
    print(f'> micro batch number: {args.gbs // args.mbs}')
    replica(graph, dl, graph.device)


def premise_nnshape(graph: IRGraph, ndevs: int, mem_limit: int):
    """NNShape schedule"""
    transformers = annotate_structure(graph)
    if args.recompute:
        for transformer in transformers:
            graph.recompute(transformer)

    nlayers = len(transformers)
    # get embedding layers
    embed2 = transformers.pop(nlayers // 2)
    embed1 = transformers.pop(0)
    # pipeline 
    encoders = transformers[:len(transformers) // 2]
    decoders = transformers[len(transformers) // 2:]

    estimator = Estimator(args.db_cache)
    tps = partial(TPS, recompute=args.recompute, estimator=estimator, mem_limit=mem_limit)
    print(f'> search [initialize]: profiling model...')
    latency, memory  = estimator(tuple(graph.select(ntype=IRFwOperation)), train=True)
    print(f'> search [estimation]: single device latency: {latency} ms, memory: {memory/1024/1024/1024} GB')
    # save profiled database
    print(f'> search [dump]: saving profiled database...')
    estimator.save()
    
    # policy search for balanced pipeline stages
    encoder_min_cost, encoder_config = layer_division(
        list(itertools.chain(*encoders)), ndevs // 2, tps, args.mbs, max_d=1, max_t=ndevs // 4, max_p=2)
    decoder_min_cost, decoder_config = layer_division(
        list(itertools.chain(*decoders)), ndevs // 2, tps, args.mbs, max_d=1, max_t=ndevs // 4, max_p=2)

    fstages = [embed1] + [snodes for snodes, _, _ in encoder_config] + \
              [embed2] + [snodes for snodes, _, _ in decoder_config]
    graph.staging(tuple(stage[0] for stage in fstages))
    segments = graph.select(ntype=IRSegment, flatten=False)
    fsegments = [seg for seg in segments if seg.isfw()]

    # full tensor parallelism on embedding layers
    embed_tp_devs = list(range(ndevs))
    stage_tp(graph, fsegments[0].select(ntype=IRFwOperation), embed_tp_devs)
    stage_tp(graph, fsegments[len(fstages)//2].select(ntype=IRFwOperation), embed_tp_devs)

    curr_devs = 0
    # sub tensor parallelism on pipeline stages
    encoder_stages = fsegments[1:len(fsegments)//2]
    for segment, (_, dp, tp) in zip(encoder_stages, encoder_config):
        stage_ndevs = dp * tp
        if stage_ndevs > 1:
            stage_tp(graph, segment.select(ntype=IRFwOperation),
                     list(range(curr_devs, curr_devs + stage_ndevs)))
        else:
            graph.assign(segment, curr_devs)
        curr_devs += stage_ndevs
    decoder_stages = fsegments[len(fsegments)//2+1:]
    for segment, (_, dp, tp) in zip(decoder_stages, decoder_config):
        stage_ndevs = dp * tp
        if stage_ndevs > 1:
            stage_tp(graph, segment.select(ntype=IRFwOperation),
                     list(range(curr_devs, curr_devs + stage_ndevs)))
        else:
            graph.assign(segment, curr_devs)
        curr_devs += stage_ndevs
    assert curr_devs == ndevs

    dl = graph.select(ntype=IRDataOperation)[0]
    replica(graph, dl, embed_tp_devs)

    FW, BW = 'forward', 'backward'
    sched = TSched(ndevs)
    # v-shape
    fblocks = [TBlock(0, span=1, memory=1, btype=FW) for _ in range(ndevs)]
    fdevs = [[devid] for devid in range(ndevs)]
    bblocks, bdevs = [], []
    bblocks = [TBlock(0, span=3, memory=-1, btype=BW) for _ in range(ndevs)]
    bdevs = [[ndevs-1-devid] for devid in range(ndevs)]
    # full shard 2
    fblocks.insert(ndevs // 2, TBlock(0, span=1, memory=0, btype=FW))
    fdevs.insert(ndevs // 2, list(range(ndevs)))
    bblocks.insert(ndevs // 2, TBlock(0, span=1, memory=0, btype=BW))
    bdevs.insert(ndevs // 2, list(range(ndevs)))
    # full shard 1
    fblocks.insert(0, TBlock(0, span=1, memory=0, btype=FW))
    fdevs.insert(0, list(range(ndevs)))
    bblocks.insert(len(bblocks), TBlock(0, span=1, memory=0, btype=BW))
    bdevs.insert(len(bblocks), list(range(ndevs)))

    blocks = fblocks + bblocks
    devs = fdevs + bdevs
    sched.add_block_seq(blocks, devs)
    return sched


def premise_nnshape_eager(graph: IRGraph, ndevs: int, mem_limit: int):
    """NNShape schedule"""
    transformers = annotate_structure(graph)
    if args.recompute:
        for transformer in transformers:
            graph.recompute(transformer)

    nlayers = len(transformers)
    # get embedding layers
    embed2 = transformers.pop(nlayers // 2)
    embed1 = transformers.pop(0)
    # pipeline 
    encoders = transformers[:len(transformers) // 2]
    decoders = transformers[len(transformers) // 2:]

    estimator = Estimator(args.db_cache)
    tps = partial(TPS, recompute=args.recompute, estimator=estimator, mem_limit=mem_limit)
    print(f'> search [initialize]: profiling model...')
    latency, memory  = estimator(tuple(graph.select(ntype=IRFwOperation)), train=True)
    print(f'> search [estimation]: single device latency: {latency} ms, memory: {memory/1024/1024/1024} GB')
    # save profiled database
    print(f'> search [dump]: saving profiled database...')
    estimator.save()
    
    # policy search for balanced pipeline stages
    min_cost, config = layer_division(
        list(itertools.chain(*(encoders+decoders))), ndevs, tps, args.mbs,
        max_d=1, max_t=ndevs//4, max_p=4
    )
    fstages = [embed1] + [config[0][0]] + [embed2] + \
              [snodes for snodes, _, _ in config[1:]]
    # re-schedule nodes
    graph.order(embed2, config[1][0][0:2])
    graph.staging(tuple(stage[0] for stage in fstages))
    segments = graph.select(ntype=IRSegment, flatten=False)
    fsegments = [seg for seg in segments if seg.isfw()]

    # full tensor parallelism on embedding layers
    embed_tp_devs = list(range(ndevs))
    stage_tp(graph, fsegments[0].select(ntype=IRFwOperation), embed_tp_devs)
    stage_tp(graph, fsegments[2].select(ntype=IRFwOperation), embed_tp_devs)

    curr_devs = 0
    # sub tensor parallelism on pipeline stages
    xcoders = fsegments[1:2] + fsegments[3:]
    for segment, (_, dp, tp) in zip(xcoders, config):
        stage_ndevs = dp * tp
        if stage_ndevs > 1:
            if args.mbs % (ndevs // 4) == 0:
                stage_dp(graph, segment.select(ntype=IRFwOperation),
                         list(range(curr_devs, curr_devs + stage_ndevs)))
            else:
                stage_tp(graph, segment.select(ntype=IRFwOperation),
                         list(range(curr_devs, curr_devs + stage_ndevs)))
        else:
            graph.assign(segment, curr_devs)
        curr_devs += stage_ndevs
    assert curr_devs == ndevs

    dl = graph.select(ntype=IRDataOperation)[0]
    replica(graph, dl, embed_tp_devs)

    FW, BW = 'forward', 'backward'
    sched = TSched(ndevs)
    # v-shape
    fblocks = [TBlock(0, span=1, memory=1, btype=FW) for _ in range(ndevs)]
    fdevs = [[devid] for devid in range(ndevs)]
    bblocks, bdevs = [], []
    bblocks = [TBlock(0, span=3, memory=-1, btype=BW) for _ in range(ndevs)]
    bdevs = [[ndevs-1-devid] for devid in range(ndevs)]
    # full shard 2
    fblocks.insert(1, TBlock(0, span=1, memory=0, btype=FW))
    fdevs.insert(1, list(range(ndevs)))
    bblocks.insert(ndevs-1, TBlock(0, span=1, memory=0, btype=BW))
    bdevs.insert(ndevs-1, list(range(ndevs)))
    # full shard 1
    fblocks.insert(0, TBlock(0, span=1, memory=0, btype=FW))
    fdevs.insert(0, list(range(ndevs)))
    bblocks.insert(len(bblocks), TBlock(0, span=1, memory=0, btype=BW))
    bdevs.insert(len(bblocks), list(range(ndevs)))

    blocks = fblocks + bblocks
    devs = fdevs + bdevs
    sched.add_block_seq(blocks, devs)
    return sched


def train():

    if args.premise == '1f1b':
        runtime_policy = partial(PAS1F1B,
                                 premise=premise_vshape,
                                 nmicros=args.gbs//args.mbs,
                                 sched='1f1b')
    elif args.premise == 'gpipe':
        runtime_policy = partial(PAS1F1B,
                                 premise=premise_vshape,
                                 nmicros=args.gbs//args.mbs,
                                 sched='gpipe')
    elif args.premise == '1f1b+':
        runtime_policy = partial(PAS1F1BPlus,
                                 premise=premise_nnshape_eager,
                                 nmicros=args.gbs//args.mbs)
    elif args.premise == 'chimera':
        args.mbs = 2 * args.mbs # double for chimera execution
        runtime_policy = partial(PASChimera,
                                 premise=premise_xshape,
                                 nmicros=args.gbs//args.mbs)
    elif args.premise == 'nnshape':
        runtime_policy = partial(PASTetris,
                                 premise=premise_nnshape,
                                 nmicros=args.gbs//args.mbs,
                                 max_inflight_blks = [10] * DeviceGroup().world_size,
                                 load_plan=args.load_tsched,
                                 save_dir=args.save)
        assert 'eager' not in args.load_tsched
    elif args.premise == 'nnshape_eager':
        runtime_policy = partial(PASTetris,
                                 premise=premise_nnshape_eager,
                                 nmicros=args.gbs//args.mbs,
                                 max_inflight_blks = [10] * DeviceGroup().world_size,
                                 load_plan=args.load_tsched,
                                 save_dir=args.save)
        assert 'eager' in args.load_tsched
    elif args.premise == 'tp':
        runtime_policy = full_tp
    else:
        raise KeyError

    # setup model arg
    cfg = Config(
        args.vocab,
        args.hidden,
        args.hidden // args.heads,
        args.hidden * 4,
        args.layers,
        args.heads,
        args.seqlen)
    assert args.hidden % args.heads == 0

    print_each_rank(f"{cfg}", rank_only=0)

    if DeviceGroup().local_rank == 0:
        model = mT5(cfg)
        model = model.half() if args.fp16 else model
    else:
        model = None
    dataloader = mT5DataLoader(args.mbs, cfg)

    if torch.distributed.get_rank() == 0:
        nparams = 0
        for param in model.parameters():
            nparams += param.nelement()
        print(f'full model parameter: {nparams}')

    model = cube.SemanticModel(model)
    @cube.compile(model, dataloader, PAS=runtime_policy, override=True, load_content=False, 
                  comm_cost_fn=lambda x: 1)
    def train_iter(model, dataloader):
        datas = next(dataloader)
        loss = model(*datas)
        loss.backward()
        # return loss
    model: torch.nn.Module = model.get_gen_module()

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999))

    torch.distributed.barrier()
    print_each_rank('model weight consumpition:')
    memory_summary()
    nparams = 0
    for param in model.parameters():
        nparams += param.nelement()
    print_each_rank(f'loaded model parameter: {nparams}')

    CudaTimer(enable=False).warmup()
    iter_num, warmup = 3, 2
    for step in range(iter_num):

        if step == warmup:
            CudaTimer(enable=True, predefined=False).start('e2e')

        # training
        train_iter(model, dataloader)
        optimizer.step()
        optimizer.zero_grad()

        if step == 0:
            print_each_rank('passed first iteration')
        if (step + 1) % 2 == 0:
            print_each_rank(f'iter [{step + 1}/{iter_num}]', rank_only=0)

    CudaTimer().stop('e2e')
    print_each_rank('e2e time (ms) per iteration: {} ms'.format(
        CudaTimer().duration(iter_num-warmup, field_name='e2e')))
    CudaTimer().print_all(times=iter_num-warmup)
    memory_summary()


if __name__ == '__main__':

    train()
