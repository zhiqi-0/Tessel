"""
OMP_NUM_THREADS=4 torchrun --nproc_per_node=1 --nnodes=1 \
    examples/swin/train.py \
        --fp16 --mbs 1 --gbs 1 --dp 1 --premise vshape \
        --layers 2 --hidden 512 --heads 16
"""

from typing import List
import torch
import math
from functools import partial

from examples.swin.blocks.attention import init_relative_position_index
from examples.swin.model import Config, SwinTransformer, ImageDataLoader

import cube
from cube.profiler.timer import CudaTimer, print_each_rank
from cube.profiler.memory import memory_summary

from cube.graph import IRGraph
from cube.graph.segment import IRSegment
from cube.graph.function.anchor import IRGraphAnchor
from cube.ir.operator import IRDataOperation, IRFwOperation
from cube.runtime.device import DeviceGroup

from tetris.runtime.utils import tp, replica, annotate_structure, MemoryProfiler
from tetris.runtime.division import TPS, layer_division
from tetris.runtime.sched import tsched
from tetris.runtime.estimator import Estimator
from tetris.schedplan import SchedPlan as TSched
from tetris.schedplan import Block as TBlock
from tetris.runtime.piper import Piper
from tetris.runtime.flags import SearchFlag

import argparse

parser = argparse.ArgumentParser(description='SwinTransformer Train')
parser.add_argument('--fp16', action='store_true', default=False,
                    help='use fp16 for the training')
parser.add_argument('--mbs', type=int, default=1,
                    help='micro-batch size, premise total batch size')
parser.add_argument('--gbs', type=int, default=256,
                    help='global batch size')
# model arch
parser.add_argument('--layers', type=int, required=True, help="the third stage layer number")
parser.add_argument('--hidden', type=int, required=True, help="the first stage embedding dimension")
parser.add_argument('--heads', type=int, required=True, help="the first stage number of head")
# input size: (640, 40) or 1536, 48
parser.add_argument('--resolution', type=int, default=1536, help='image resolution')
parser.add_argument('--window-size', type=int, default=48, help='window size')

# policy
parser.add_argument('--premise', type=str, choices=['vshape', 'mshape', 'piper', 'tp'],
                    help='premise shape')
parser.add_argument('--recompute', action='store_true', default=False,
                    help='enable recompute for each layer')

# log save
parser.add_argument('--save', type=str, default=None, 
                    help='folder for save searched results.')
parser.add_argument('--load-tsched', type=str, default=None,
                    help='load searched tetris schedule from file')
parser.add_argument('--db-cache', type=str, default='db.json',
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

def stage_tp(graph: IRGraph, segment: IRSegment, devs: List[int]) -> IRSegment:
    """Tensor parallelsim a stage"""
    ndevs = len(devs)
    for fnode in segment.select(ntype=IRFwOperation):
        if fnode.name == 'multiref' or isinstance(fnode, IRGraphAnchor): continue
        if fnode.name == 'window_attn' or fnode.name == 'feedforward':
            tp(graph, fnode, devs, idx=1, dim=0, num=ndevs)
        else:
            replica(graph, fnode, devs)
    return segment

# ========================= parallelisms =================================


def premise_mshape(graph: IRGraph, ndevs: int, mem_limit: int):
    """MShape schedule"""
    transformers = annotate_structure(graph)
    if args.recompute:
        for transformer in transformers:
            graph.recompute(transformer)
    
    nlayers_to_tp = 5
    full_tps, sub_tps = [], []
    for idx, layer_nodes in enumerate(transformers):
        if idx < nlayers_to_tp:
            full_tps += layer_nodes
        else:
            sub_tps += layer_nodes

    estimator = Estimator(args.db_cache)
    tps = partial(TPS, recompute=args.recompute, estimator=estimator, mem_limit=mem_limit)
    min_cost, best_config = layer_division(
        sub_tps, ndevs, tps, args.mbs, max_d=1, max_p=4, max_t=1)

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

    FW, BW = 'forward', 'backward'
    ndevs = len(fsegments) - 1
    sched = TSched(ndevs)
    # 
    fblocks = [TBlock(0, span=1, memory=1, btype=FW) for _ in range(ndevs)]
    fdevs = [[devid] for devid in range(ndevs)]
    bblocks = [TBlock(0, span=3 if args.recompute else 2, memory=-1, btype=BW) for _ in range(ndevs)]
    bdevs = [[ndevs-1-devid] for devid in range(ndevs)]
    # fully shard
    fblocks.insert(0, TBlock(0, span=1, memory=1, btype=FW))
    fdevs.insert(0, list(range(ndevs)))
    bblocks.insert(len(bblocks), TBlock(0, span=2, memory=-1, btype=BW))
    bdevs.insert(len(bblocks), list(range(ndevs)))

    blocks = fblocks + bblocks
    devs = fdevs + bdevs
    sched.add_block_seq(blocks, devs)
    return sched


def full_tp(graph: IRGraph, resource):

    transformers = annotate_structure(graph)
    if args.recompute:
        for transformer in transformers:
            graph.recompute(transformer)

    devs = list(range(resource.ngpus))
    stage_tp(graph, graph, devs)
    for dl in graph.select(ntype=IRDataOperation):
        replica(graph, dl, devs)
    return graph


def train():

    batch_size = args.mbs
    load_content: bool = False

    # setup model arg
    cfg = Config()
    cfg.embed_dim = args.hidden
    cfg.depths = [2, 2, args.layers, 2]
    cfg.num_heads = [args.heads * (2 ** i) for i in range(4)]
    cfg.img_size = args.resolution
    cfg.window_size = args.window_size
    cfg.num_classes = 1024  # for partitionable
    assert args.hidden % args.heads == 0
    assert (args.resolution, args.window_size) == (1536, 48) or \
           (args.resolution, args.window_size) == (640, 40)

    print_each_rank(f"model arch: layers={cfg.depths}, heads={cfg.num_heads}, hidden={cfg.embed_dim}", rank_only=0)

    if DeviceGroup().local_rank == 0:
        model = SwinTransformer(cfg)
        model = model.half() if args.fp16 else model
    else:
        model = None

    dtype = torch.float16 if args.fp16 else torch.float32
    dataloader = ImageDataLoader(batch_size, cfg.img_size, cfg.num_classes, dtype=dtype)

    if args.premise == 'piper':
        runtime_policy = partial(Piper, nmicros=args.gbs//args.mbs,
                                 recompute=args.recompute, tp_sprog=stage_tp, db_cache=args.db_cache)
    elif args.premise == 'tp':
        runtime_policy = full_tp
    elif args.premise == 'mshape':
        runtime_policy = partial(tsched,
                                 num_microbatches = args.gbs//args.mbs,
                                 premise=premise_mshape if args.premise == 'vshape' else premise_mshape,
                                 max_inflight_blks = [10] * DeviceGroup().world_size,
                                 load_plan=args.load_tsched,
                                 save_dir=args.save)
    else:
        raise RuntimeError(f"not Supported for {args.premise}")

    model = cube.SemanticModel(model)
    @cube.compile(model, dataloader, PAS=runtime_policy, override=True, 
                  load_content=load_content, comm_cost_fn=lambda x: 1)
    def train_iter(model, dataloader):
        imgs = next(dataloader)
        loss = model(imgs)
        loss.backward()
        # return loss
    model: torch.nn.Module = model.get_gen_module()

    if not load_content:
        for name, buffer in model.named_buffers():
            if 'rp_index' in name:
                window_size = int(math.sqrt(buffer.size(0)))
                buffer.copy_(init_relative_position_index(window_size).cuda())

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
    if DeviceGroup().local_rank == 0:
        print(torch.cuda.memory_summary())


if __name__ == '__main__':

    train()
