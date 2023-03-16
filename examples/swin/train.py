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
from cube.profiler.memory import memory_summary, model_summary

from cube.graph import IRGraph
from cube.graph.segment import IRSegment
from cube.graph.function.anchor import IRGraphAnchor
from cube.ir.cten import IRCell
from cube.ir.operator import IRDataOperation, IRFwOperation
from cube.runtime.device import DeviceGroup

from tetris.runtime.utils import layer_division_rules, tp, replica, annotate_structure
from tetris.runtime.policy import policy
from tetris.schedplan import SchedPlan as TSched
from tetris.schedplan import Block as TBlock
from tetris.runtime.piper import Piper

import argparse

parser = argparse.ArgumentParser(description='SwinTransformer Train')
parser.add_argument('--premise', type=str, choices=['vshape', 'mshape', 'piper'],
                    help='premise shape')
parser.add_argument('--fp16', action='store_true', default=False,
                    help='use fp16 for the training')
parser.add_argument('--mbs', type=int, default=1,
                    help='micro-batch size, premise total batch size')
parser.add_argument('--gbs', type=int, default=256,
                    help='global batch size')
# input size: (640, 40) or 1536, 48
parser.add_argument('--resolution', type=int, default=640,
                    help='image resolution')
parser.add_argument('--window-size', type=int, default=40,
                    help='window size')
parser.add_argument('--dp', type=int, default=1, 
                    help='data parallel size')
parser.add_argument('--recompute', action='store_true', default=False,
                    help='enable recompute for each layer')
parser.add_argument('--stage-balance', action='store_true', default=False,
                    help='profile each layer and try to get a balanced pipeline stage within memory constraints')
# model arch
parser.add_argument('--layers', type=int, required=True,
                    help="the third stage layer number")
parser.add_argument('--hidden', type=int, required=True,
                    help="the first stage embedding dimension")
parser.add_argument('--heads', type=int, required=True,
                    help="the first stage number of head")
# log save
parser.add_argument('--save', type=str, default=None,
                    help='folder for save searched results.')

args = parser.parse_args()

cube.init()
print_each_rank(str(args), rank_only=0)


# ========================= parallelisms =================================

def stage_tp(graph: IRGraph, segment: IRSegment, devs: List[int]) -> IRSegment:
    """Tensor parallelsim a stage"""
    ndevs = len(devs)
    for fnode in segment.select(ntype=IRFwOperation):
        if fnode.name == 'multiref' or isinstance(fnode, IRGraphAnchor): continue
        if fnode.name == 'window_attn' or fnode.name == 'feedforward':
            tp(graph, fnode, devs, idx=1, dim=0, num=ndevs)
        elif fnode.name == 'linear':  # the last embedding linear
            tp(graph, fnode, devs, idx=1, dim=0, num=ndevs)
        else:
            replica(graph, fnode, devs)
    return segment

# ========================= parallelisms =================================

def PASingle(graph: IRGraph, resource):
    """
    Debugging policy
    """
    for node in graph.select(ntype=(IRFwOperation, IRDataOperation)):
        graph.assign(node, 0)
    return graph


# V100
stage_comp_cost = {
    192 : [109.93]  * 2 + [60.34]  * 2 + [43.18]  * args.layers + [27.51] * 2, # FAKE
    256 : [109.93]  * 2 + [60.34]  * 2 + [43.18]  * args.layers + [27.51] * 2,
    384 : [255.10]  * 2 + [139.92] * 2 + [90.98 ] * args.layers + [63.78 ] * 2, # FAKE
    512 : [255.10]  * 2 + [139.92] * 2 + [90.98 ] * args.layers + [63.78 ] * 2,
    768 : [1486.10] * 2 + [795.47] * 2 + [618.98] * args.layers + [637.95] * 2,
    1024: [1170.26] * 2 + [615.17] * 2 + [452.65] * args.layers + [439.29] * 2,
    1536: [1009.80] * 2 + [521.61] * 2 + [315.63] * args.layers + [302.63] * 2,
}


def premise_vshape(graph: IRGraph, ndevs: int):
    """1F1B schedule"""
    transformers = annotate_structure(graph)

    global stage_comp_cost
    FW, BW = 'forward', 'backward'

    if args.recompute:
        for transformer in transformers:
            graph.recompute(transformer)

    pp_size = ndevs
    layer_range_per_stage = layer_division_rules(
        pp_size,
        stage_comp_cost[args.hidden], 
        limits=[1, 1, None, None]
    )
    print(f'pipeline ndevs: {pp_size}, layer division: {layer_range_per_stage}')
    fstages = [[] for _ in range(pp_size)]
    for lid, fnodes in enumerate(transformers):
        find_stage = False
        for sid, (start, end) in enumerate(layer_range_per_stage):
            if start <= lid and lid < end:
                find_stage = True
                break
        assert find_stage
        fstages[sid] += fnodes
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


def premise_mshape(graph: IRGraph, ndevs: int):
    """MShape schedule"""
    transformers = annotate_structure(graph)

    global stage_comp_cost
    FW, BW = 'forward', 'backward'
    if args.recompute:
        for transformer in transformers:
            graph.recompute(transformer)
    
    comp_costs = stage_comp_cost[args.hidden]
    layer_range_per_stage = layer_division_rules(ndevs, comp_costs[2:])
    nlayers_to_tp = 2
    layer_range_per_stage = [(s+nlayers_to_tp, e+nlayers_to_tp) for s, e in layer_range_per_stage]
    layer_range_per_stage = [(0, nlayers_to_tp),] + layer_range_per_stage
    print(f'layer division: {layer_range_per_stage}')
    fstages = [[] for _ in range(len(layer_range_per_stage))]
    for lid, fnodes in enumerate(transformers):
        find_stage = False
        for sid, (start, end) in enumerate(layer_range_per_stage):
            if start <= lid and lid < end:
                find_stage = True
                break
        assert find_stage
        fstages[sid] += fnodes
    graph.staging(tuple(stages[0] for stages in fstages))

    segments = graph.select(ntype=IRSegment, flatten=False)
    fsegments = [seg for seg in segments if seg.isfw()]

    tp_devs = list(range(ndevs))
    for fnode in fsegments[0].select(ntype=IRFwOperation):
        if fnode.name == 'multiref' or isinstance(fnode, IRGraphAnchor): continue
        if fnode.name == 'window_attn' or fnode.name == 'feedforward':
            subnodes = _tp(graph, fnode, tp_devs, idx=1, dim=0, num=ndevs)
        elif fnode.name == 'linear':  # the last embeding linear
            subnodes = _tp(graph, fnode, tp_devs, idx=1, dim=0, num=ndevs)
        else:
            subnodes = _replica(graph, fnode, tp_devs)
        for devid, subnode in enumerate(subnodes):
            graph.assign(subnode, devid)
    
    for devid, segment in enumerate(fsegments[1:]):
        graph.assign(segment, devid)

    sched = TSched(ndevs)
    # 
    fblocks = [TBlock(0, span=1, memory=1, btype=FW) for _ in range(ndevs)]
    fdevs = [[devid] for devid in range(ndevs)]
    bblocks = [TBlock(0, span=2, memory=-1, btype=BW) for _ in range(ndevs)]
    bdevs = [[ndevs-1-devid] for devid in range(ndevs)]
    #
    fblocks.insert(0, TBlock(0, span=1, memory=1, btype=FW))
    fdevs.insert(0, list(range(ndevs)))
    bblocks.insert(len(bblocks), TBlock(0, span=2, memory=-1, btype=BW))
    bdevs.insert(len(bblocks), list(range(ndevs)))

    blocks = fblocks + bblocks
    devs = fdevs + bdevs
    sched.add_block_seq(blocks, devs)
    return sched


def train():

    batch_size = args.mbs
    load_content: bool = False

    # setup model arg
    cfg = Config()
    cfg.embed_dim = args.hidden
    cfg.depths = [2, 2, args.layers, 2]
    cfg.num_heads = [args.heads * (2 ** i) for i in range(4)]
    assert args.hidden % args.heads == 0

    cfg.img_size = args.resolution
    cfg.window_size = args.window_size
    assert (args.resolution, args.window_size) == (1536, 48) or \
           (args.resolution, args.window_size) == (640, 40)

    cfg.num_classes = 1024 # for partitionable

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
                                 recompute=args.recompute, tp_sprog=stage_tp)
    else:
        runtime_policy = partial(policy,
                                 num_microbatches = args.gbs//args.mbs,
                                 premise=premise_vshape if args.premise == 'vshape' else premise_mshape,
                                 memory_limits = [8] * DeviceGroup().world_size,
                                 save_dir=args.save)
    # runtime_policy = PASingle

    model = cube.SemanticModel(model)
    @cube.compile(model, dataloader, PAS=runtime_policy, override=True, load_content=load_content)
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


if __name__ == '__main__':

    train()
