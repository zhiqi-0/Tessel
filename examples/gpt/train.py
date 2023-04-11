"""
PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun --nproc_per_node=1 --nnodes=1 \
    examples/gpt/train.py \
        --fp16 --mbs 1 --gbs 1 --premise piper \
        --layers 16 --hidden 2560 --heads 16 --seqlen 2048 --vocab 500000 \
        --db-cache gpt_V100_db.json
"""

from typing import List
import torch
from functools import partial

from examples.gpt.model import Config, GPT, GPTDataLoader

import cube
from cube.profiler.timer import CudaTimer, print_each_rank
from cube.profiler.memory import memory_summary
from cube.codegen.schedule.schedule import ScheduleCodeGen

from cube.graph import IRGraph
from cube.graph.segment import IRSegment
from cube.graph.function.anchor import IRGraphAnchor
from cube.ir.operator import IRFwOperation
from cube.runtime.device import DeviceGroup

from tetris.runtime.utils import tp, replica, annotate_structure
from tetris.runtime.division import TPS, layer_division
from tetris.runtime.sched import tsched
from tetris.runtime.estimator import Estimator
from tetris.schedplan import SchedPlan as TSched
from tetris.schedplan import Block as TBlock
from tetris.runtime.piper import Piper

import argparse
import os

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
parser.add_argument('--premise', type=str, choices=['vshape', 'mshape', 'piper'],
                    help='premise shape')
parser.add_argument('--recompute', action='store_true', default=False)
# log save
parser.add_argument('--save', type=str, default=None,
                    help='folder for save searched results.')
parser.add_argument('--db-cache', type=str, default='gpt_db.json',
                    help='profiled database save file')
args = parser.parse_args()

cube.init()
print_each_rank(str(args), rank_only=0)


MEM_LIMIT = os.environ.get('MEM_LIMIT', None)
if MEM_LIMIT is not None:
    fraction = int(MEM_LIMIT) * 1024 * 1024 * 1024 / torch.cuda.get_device_properties(0).total_memory
    print_each_rank(f'> setting memory fraction: {fraction}')
    torch.cuda.set_per_process_memory_fraction(fraction)


# ========================= parallelisms =================================

def stage_tp(graph: IRGraph, segment: IRSegment, devs: List[int]) -> IRSegment:
    """Tensor parallelsim a stage"""
    ndevs = len(devs)
    for fnode in segment.select(ntype=IRFwOperation):
        if fnode.name == 'multiref' or isinstance(fnode, IRGraphAnchor): continue
        if fnode.name == 'embedding':
            tp(graph, fnode, devs, idx=1, dim=0, num=ndevs)
        elif fnode.name == 'self_attention' or fnode.name == 'feedforward':
            tp(graph, fnode, devs, idx=1, dim=0, num=ndevs)
        elif fnode.name == 'linear':  # the last embedding linear
            tp(graph, fnode, devs, idx=1, dim=0, num=ndevs)
        else:
            replica(graph, fnode, devs)
    return segment

# ========================= parallelisms =================================

def premise_vshape(graph: IRGraph, ndevs: int, mem_limit: int):
    """1F1B schedule"""
    transformers = annotate_structure(graph)

    global stage_comp_cost
    FW, BW = 'forward', 'backward'

    ScheduleCodeGen.recompute = True
    # if args.recompute:
    #     for transformer in transformers:
    #         graph.recompute(transformer)

    pp_size = ndevs

    fnodes = graph.select(ntype=IRFwOperation)
    estimator = Estimator(args.db_cache)
    tps = partial(TPS, recompute=args.recompute, estimator=estimator, mem_limit=mem_limit)
    min_cost, best_config = layer_division(fnodes, ndevs, tps, args.mbs, max_d=1, max_t=1)

    fstages = [stage_nodes for stage_nodes, _, _ in best_config]
    graph.staging(tuple(stages[0] for stages in fstages))
    
    segments = graph.select(ntype=IRSegment, flatten=False)
    fsegments = [seg for seg in segments if seg.isfw()]
    for devid, segment in enumerate(fsegments):
        graph.assign(segment, devid)

    sched = TSched(pp_size)
    fblocks = [TBlock(0, span=1, memory=1, btype=FW) for _ in range(pp_size)]
    fdevs = [[devid] for devid in range(pp_size)]
    bblocks = [TBlock(0, span=2, memory=-1, btype=BW) for _ in range(pp_size)]
    bdevs = [[devid] for devid in range(pp_size)][::-1]
    blocks = fblocks + bblocks
    devs = fdevs + bdevs
    sched.add_block_seq(blocks, devs)

    # print(graph.extra_repr())
    return sched


def premise_mshape(graph: IRGraph, ndevs: int, mem_limit: int):
    """MShape schedule"""
    transformers = annotate_structure(graph)
    FW, BW = 'forward', 'backward'

    ScheduleCodeGen.recompute = True
    # if args.recompute:
    #     for transformer in transformers:
    #         graph.recompute(transformer)
    
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
    stage_tp(graph, fsegments[0], tp_devs)

    curr_devs = 0
    for segment, (_, dp, tp) in zip(fsegments[1:], best_config):
        stage_ndevs = dp * tp
        if stage_ndevs > 1:
            stage_tp(graph, segment, 
                     list(range(curr_devs, curr_devs + stage_ndevs)))
        else:
            graph.assign(segment, curr_devs)
        curr_devs += stage_ndevs

    ndevs = len(fsegments) - 1
    sched = TSched(ndevs)
    # 
    fblocks = [TBlock(0, span=1, memory=1, btype=FW) for _ in range(ndevs)]
    fdevs = [[devid] for devid in range(ndevs)]
    bblocks = [TBlock(0, span=1, memory=-1, btype=BW) for _ in range(ndevs)]
    bdevs = [[ndevs-1-devid] for devid in range(ndevs)]
    #
    fblocks.insert(0, TBlock(0, span=1, memory=1, btype=FW))
    fdevs.insert(0, list(range(ndevs)))
    bblocks.insert(len(bblocks), TBlock(0, span=1, memory=-1, btype=BW))
    bdevs.insert(len(bblocks), list(range(ndevs)))

    blocks = fblocks + bblocks
    devs = fdevs + bdevs
    sched.add_block_seq(blocks, devs)
    return sched


def train():

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

    if args.premise == 'piper':
        runtime_policy = partial(Piper, nmicros=args.gbs//args.mbs,
                                 recompute=args.recompute, tp_sprog=stage_tp, db_cache=args.db_cache)
    else:
        runtime_policy = partial(tsched,
                                 num_microbatches = args.gbs//args.mbs,
                                 premise=premise_vshape if args.premise == 'vshape' else premise_mshape,
                                 max_inflight_blks = [8] * DeviceGroup().world_size,
                                 save_dir=args.save)

    model = cube.SemanticModel(model)
    @cube.compile(model, dataloader, PAS=runtime_policy, override=True, load_content=False)
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
