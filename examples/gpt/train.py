
from typing import List, Callable
import torch
from functools import partial
import warnings

from examples.gpt.model import Config, GPT, GPTDataLoader

import cube
from cube.profiler.timer import CudaTimer, print_each_rank
from cube.profiler.memory import memory_summary
from cube.graph.schedule.predefined import PredefinedSched

from cube.graph import IRGraph
from cube.graph.segment import IRSegment
from cube.graph.function.anchor import IRGraphAnchor
from cube.graph.function.dimops import IRDimops
from cube.ir.operator import IRFwOperation, IRDataOperation
from cube.runtime.device import DeviceGroup

from tetris.runtime.utils import tp, replica, annotate_structure
from tetris.runtime.division import TPS, layer_division
from tetris.runtime.sched import tsched
from tetris.runtime.estimator import Estimator
from tetris.schedplan import SchedPlan as TSched
from tetris.schedplan import Block as TBlock
from tetris.runtime.flags import SearchFlag

import argparse

parser = argparse.ArgumentParser(description='SwinTransformer Train')
parser.add_argument('--fp16', action='store_true', default=False)
parser.add_argument('--mbs', type=int, default=1, help='micro-batch size')
parser.add_argument('--gbs', type=int, default=256, help='global batch size')
# arch
parser.add_argument('--layers', type=int, required=True)
parser.add_argument('--hidden', type=int, required=True)
parser.add_argument('--heads', type=int, required=True)
parser.add_argument('--seqlen', type=int, required=True)
parser.add_argument('--vocab', type=int, required=True)
# policy
parser.add_argument('--premise', type=str, choices=['1f1b', 'mshape', 'gpipe', 'tp', 'chimera'],
                    help='premise shape')
parser.add_argument('--recompute', action='store_true', default=False)
# log save
parser.add_argument('--save', type=str, default=None,
                    help='folder for save searched results.')
parser.add_argument('--load-tsched', type=str, default=None,
                    help='load searched tetris schedule from file')
parser.add_argument('--db-cache', type=str, default='gpt_db.json',
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

def stage_tp(graph: IRGraph, nodes: List[IRFwOperation], devs: List[int]) -> IRSegment:
    """Tensor parallelsim a stage"""
    ndevs = len(devs)
    for fnode in nodes:
        if fnode.name == 'multiref' or isinstance(fnode, IRGraphAnchor): continue
        if fnode.name == 'embedding' and fnode.input(1).shape[0] > 10240:
            tp(graph, fnode, devs, idx=1, dim=0, num=ndevs)
            # if args.premise == 'mshape':
            #     if len(devs) > 8:
            #         assert len(devs) % 2 == 0
            #         pdevs = len(devs) // 2
            #         embeds = graph.replicate(fnode, times=2)
            #         tp(graph, embeds[0], devs[:pdevs], idx=1, dim=0, num=pdevs)
            #         tp(graph, embeds[1], devs[pdevs:], idx=1, dim=0, num=pdevs)
            # else:
            #     tp(graph, fnode, devs, idx=1, dim=0, num=ndevs)
        elif fnode.name == 'self_attention' or fnode.name == 'feedforward':
            tp(graph, fnode, devs, idx=1, dim=0, num=ndevs)
        elif fnode.name == 'linear':  # the last embedding linear
            tp(graph, fnode, devs, idx=1, dim=0, num=ndevs)
        else:
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

def PASPredefined(graph: IRGraph, resource, sched: str):
    """VShape policy"""
    transformers = annotate_structure(graph)
    if args.recompute:
        for transformer in transformers:
            graph.recompute(transformer)
    nodes = tuple(graph.select(ntype=IRFwOperation))

    dl: IRDataOperation = graph.select(ntype=IRDataOperation)[0]
    mbs: int = dl.output(0).shape[dl.get_batch_dims()[0]]

    mem_limit = resource.gpus[0].memory if SearchFlag.mem_limit is None else SearchFlag.mem_limit * 1024 * 1024 * 1024
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
        nodes, resource.ngpus, tps, mbs, max_t=resource.ngpus-1)

    # ======================= instantiate plan ====================

    fstages = [stage_nodes for stage_nodes, dp, tp in best_config]
    graph.staging([snodes[0] for snodes in fstages])

    segments = graph.select(ntype=IRSegment, flatten=False)
    fsegments: List[IRSegment] = [seg for seg in segments if seg.isfw()]
    assert len(fsegments) == len(best_config), f"Expected {len(best_config)} stages in plan, but got {len(fsegments)}"

    devices = list(range(resource.ngpus))
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

    replica(graph, dl, fsegments[0].device)

    if sched == 'vshape':
        PredefinedSched.sched_1f1b(graph, args.gbs // args.mbs, len(fsegments))
    elif sched == 'gpipe':
        PredefinedSched.sched_gpipe(graph, args.gbs // args.mbs, len(fsegments))
    else:
        raise RuntimeError
    return graph


def PASChimera(graph: IRGraph, resource):
    """Chimera Direct policy"""
    transformers = annotate_structure(graph)
    if args.recompute:
        for transformer in transformers:
            graph.recompute(transformer)
    nodes = tuple(graph.select(ntype=IRFwOperation))

    dl: IRDataOperation = graph.select(ntype=IRDataOperation)[0]
    mbs: int = dl.output(0).shape[dl.get_batch_dims()[0]]

    mem_limit = resource.gpus[0].memory if SearchFlag.mem_limit is None else SearchFlag.mem_limit * 1024 * 1024 * 1024
    print(f'> search [constraints]: device limitied memory: {mem_limit}')

    # for chimera, we constrain the tensor parallelism size of each pipeline to be same
    # due to its special scheduling polies.
    assert resource.ngpus % 4 == 0
    tp_size = resource.ngpus // 4
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
                warnings.warn(
                    f'node: {node}\ncannot split node into two micro-batches, use replicate instead.',
                    category=RuntimeWarning, stacklevel=0)
                mb1, mb2 = graph.replicate(node, times=2)
            # tensor parallelism
            stage_tp(graph, [mb1], mb1_devs)
            stage_tp(graph, [mb2], mb2_devs)

    replica(graph, dl, fsegments[0].device)
    PredefinedSched.sched_chimera_direct(graph, args.gbs // args.mbs, len(fsegments))
    return graph


def premise_mshape(graph: IRGraph, ndevs: int, mem_limit: int):
    """MShape schedule"""
    transformers = annotate_structure(graph)
    if args.recompute:
        for transformer in transformers:
            graph.recompute(transformer)
    
    nlayers_to_tp = 1
    full_tps, sub_tps = [], []
    for idx, layer_nodes in enumerate(transformers):
        if idx < nlayers_to_tp:
            full_tps += layer_nodes
        else:
            sub_tps += layer_nodes

    estimator = Estimator(args.db_cache)
    tps = partial(TPS, recompute=args.recompute, estimator=estimator, mem_limit=mem_limit)
    min_cost, best_config = layer_division(
        sub_tps, ndevs, tps, args.mbs, max_d=1, max_p=4)

    fstages = [full_tps] + [stage_nodes for stage_nodes, _, _ in best_config]
    graph.staging(tuple(stages[0] for stages in fstages))

    segments = graph.select(ntype=IRSegment, flatten=False)
    fsegments = [seg for seg in segments if seg.isfw()]

    tp_devs = list(range(ndevs))
    stage_tp(graph, fsegments[0].select(ntype=IRFwOperation), tp_devs)

    curr_devs = 0
    for segment, (_, dp, tp) in zip(fsegments[1:], best_config):
        stage_ndevs = dp * tp
        if stage_ndevs > 1:
            stage_tp(graph, segment.select(ntype=IRFwOperation), 
                     list(range(curr_devs, curr_devs + stage_ndevs)))
        else:
            graph.assign(segment, curr_devs)
        curr_devs += stage_ndevs

    FW, BW = 'forward', 'backward'
    ndevs = len(fsegments) - 1
    sched = TSched(ndevs)
    # 
    fblocks = [TBlock(0, span=1, memory=1, btype=FW) for _ in range(ndevs)]
    fdevs = [[devid] for devid in range(ndevs)]
    bblocks = [TBlock(0, span=3 if args.recompute else 2, memory=-1, btype=BW) for _ in range(ndevs)]
    bdevs = [[ndevs-1-devid] for devid in range(ndevs)]
    #
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
        runtime_policy = partial(PASPredefined, sched='vshape')
    elif args.premise == 'gpipe':
        runtime_policy = partial(PASPredefined, sched='gpipe')
    elif args.premise == 'chimera':
        runtime_policy = PASChimera
        args.mbs = 2 if args.mbs == 1 else args.mbs # double for chimera execution
    elif args.premise == 'tp':
        runtime_policy = full_tp
    else:
        runtime_policy = partial(tsched,
                                 num_microbatches = args.gbs//args.mbs,
                                 premise=premise_mshape,
                                 max_inflight_blks = [10] * DeviceGroup().world_size,
                                 load_plan=args.load_tsched,
                                 save_dir=args.save)

    # setup model arg
    cfg = Config(
        args.hidden, args.layers, args.heads, args.hidden,
        args.hidden * 4, args.vocab, args.seqlen)
    assert args.hidden % args.heads == 0

    print_each_rank(f"{cfg}", rank_only=0)

    if DeviceGroup().local_rank == 0:
        model = GPT(cfg)
        model = model.half() if args.fp16 else model
    else:
        model = None
    dataloader = GPTDataLoader(args.mbs, cfg)

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
    print_each_rank(f'model parameter: {nparams}')

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
