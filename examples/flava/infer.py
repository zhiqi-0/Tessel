"""
PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun --nproc_per_node=1 \
    examples/flava/infer.py \
        --fp16 --layers 12 --hidden 768 --heads 12 \
        --premise 1f1b --mbs 1 --gbs 8
"""

from functools import partial
from typing import List
import torch

import cube
from cube.runtime.device import DeviceGroup
from cube.profiler.memory import memory_summary
from cube.profiler.timer import CudaTimer, print_each_rank
from cube.graph import IRGraph
from cube.graph.segment import IRSegment
from cube.graph.function.anchor import IRGraphAnchor
from cube.graph.schedule.predefined import PredefinedSched
from cube.ir.operator import IRFwOperation, IRDataOperation

from tetris.runtime.estimator import Estimator
from tetris.runtime.division import TPS, layer_division
from tetris.runtime.utils import tp, replica, annotate_structure
from tetris.runtime.flags import SearchFlag
from tetris.schedplan import SchedPlan as TSched
from tetris.schedplan import Block as TBlock
from tetris.runtime.sched import schedule

from examples.flava.model import FLAVAModel, Config, ImageTextDataLoader

import argparse
parser = argparse.ArgumentParser(description='Flava Inference')
parser.add_argument('--fp16', action='store_true', default=False)
parser.add_argument('--mbs', type=int, default=1, help='micro-batch size')
parser.add_argument('--gbs', type=int, default=8, help='global batch size')

parser.add_argument('--layers', type=int, required=True)
parser.add_argument('--hidden', type=int, required=True)
parser.add_argument('--heads', type=int, required=True)

# policy
parser.add_argument('--premise', type=str, choices=['1f1b', 'yshape', 'tp'],
                    help='premise shape')
# log save
parser.add_argument('--save', type=str, default=None,
                    help='folder for save searched results.')
parser.add_argument('--load-tsched', type=str, default=None,
                    help='load searched tetris schedule from file')
parser.add_argument('--db-cache', type=str, default='flava_db.json',
                    help='profiled database save file')
args = parser.parse_args()

cube.init()
print_each_rank(str(args), rank_only=0)


def PASDebug(graph, resource):
    for node in graph.select(ntype=(IRFwOperation, IRDataOperation)):
        graph.assign(node, 0)
    return graph


def stage_tp(graph: IRGraph, segment: IRSegment, devs: List[int]):
    ndevs = len(devs)
    for fnode in segment.select(ntype=IRFwOperation):
        if fnode.name == 'multiref' or isinstance(fnode, IRGraphAnchor): continue
        if fnode.name == 'self_attention' or fnode.name == 'feedforward':
            tp(graph, fnode, devs, idx=1, dim=0, num=ndevs)
        else:
            replica(graph, fnode, devs)


def PASTP(graph: IRGraph, resource):
    devices = list(range(resource.ngpus))
    dl: IRDataOperation = graph.select(ntype=IRDataOperation)[0]
    replica(graph, dl, devices)
    stage_tp(graph, graph, devices)
    return graph


def PAS1F1B(graph: IRGraph, resource):
    annotate_structure(graph)
    nodes = tuple(graph.select(ntype=IRFwOperation))

    dl: IRDataOperation = graph.select(ntype=IRDataOperation)[0]
    mbs: int = dl.output(0).shape[dl.get_batch_dims()[0]]

    estimator = Estimator(args.db_cache)
    estimator_fn = partial(estimator, train=False)

    mem_limit = resource.gpus[0].memory if SearchFlag.mem_limit is None else SearchFlag.mem_limit * 1024 * 1024 * 1024
    print(f'> search [constraints]: device limitied memory: {mem_limit}')
    tps = partial(TPS, recompute=False, 
                  estimator=estimator_fn, mem_limit=mem_limit, train=False)
    print(f'> search [initialize]: profiling model...')
    latency, memory = estimator_fn(nodes)
    print(f'> search [estimation]: single device latency: {latency} ms, memory: {memory/1024/1024/1024} GB')
    # save profiled database
    print(f'> search [dump]: saving profiled database...')
    estimator.save()

    # decrease search space
    anchors = [n for n in nodes if isinstance(n, IRGraphAnchor)]
    if len(anchors) > 48:
        nodes = list(nodes)
        for anchor in anchors[::2]:
            nodes.remove(anchor)
        nodes = tuple(nodes)
    min_cost, best_config = layer_division(
        nodes, resource.ngpus, tps, mbs, max_t=1, max_d=1)
    
    fstages = [stage_nodes for stage_nodes, dp, tp in best_config]
    graph.staging([snodes[0] for snodes in fstages])
    segments = graph.select(ntype=IRSegment, flatten=False)
    fsegments: List[IRSegment] = [seg for seg in segments if seg.isfw()]
    assert len(fsegments) == len(best_config), f"Expected {len(best_config)} stages in plan, but got {len(fsegments)}"

    replica(graph, dl, list(range(resource.ngpus)))
    for sid, segment in enumerate(fsegments):
        for node in segment.nodes():
            graph.assign(node, sid)
        
    PredefinedSched.sched_infer_pipe(graph, args.gbs // args.mbs, len(fsegments))
    return graph


def PASYShape(graph: IRGraph, resource):
    annotate_structure(graph)
    nodes = tuple(graph.select(ntype=IRFwOperation))

    dl: IRDataOperation = graph.select(ntype=IRDataOperation)[0]
    mbs: int = dl.output(0).shape[dl.get_batch_dims()[0]]

    estimator = Estimator(args.db_cache)
    estimator_fn = partial(estimator, train=False)

    mem_limit = resource.gpus[0].memory if SearchFlag.mem_limit is None else SearchFlag.mem_limit * 1024 * 1024 * 1024
    print(f'> search [constraints]: device limitied memory: {mem_limit}')
    tps = partial(TPS, recompute=False, 
                  estimator=estimator_fn, mem_limit=mem_limit, train=False)
    print(f'> search [initialize]: profiling model...')
    latency, memory = estimator_fn(nodes)
    print(f'> search [estimation]: single device latency: {latency} ms, memory: {memory/1024/1024/1024} GB')
    print(f'> search [dump]: saving profiled database...')
    estimator.save()

    text_anchor = graph.select(name='text')[0]
    text_anchor_idx = nodes.index(text_anchor)
    mm_anchor = graph.select(name='mm')[0]
    mm_anchor_idx = nodes.index(mm_anchor)

    img_nodes = nodes[:text_anchor_idx]
    img_latency, _ = estimator_fn(img_nodes)
    txt_nodes = nodes[text_anchor_idx:mm_anchor_idx]
    txt_latency, _ = estimator_fn(txt_nodes)
    mm_nodes = nodes[mm_anchor_idx:]
    mm_latency, _ = estimator_fn(mm_nodes)
    print(f'> search [estimation]: image latency: {img_latency} ms, text latency: {txt_latency} ms, mm latency: {mm_latency} ms')
    
    min_cost, best_config = layer_division(
        img_nodes, resource.ngpus // 2, tps, mbs, max_t=1, max_d=1)
    img_branch = [stage_nodes for stage_nodes, dp, tp in best_config]

    min_cost, best_config = layer_division(
        txt_nodes, resource.ngpus // 2, tps, mbs, max_t=1, max_d=1)
    txt_branch = [stage_nodes for stage_nodes, dp, tp in best_config]

    graph.blocking([ns[0] for ns in img_branch] + \
                   [ns[0] for ns in txt_branch] + \
                   [mm_nodes[0]])
    
    fsegments = graph.select(ntype=IRSegment, flatten=False)
    img_branch = fsegments[:resource.ngpus // 2]
    txt_branch = fsegments[resource.ngpus // 2: resource.ngpus]
    mm_branch = fsegments[-1]

    for sid, segment in enumerate(img_branch):
        for node in segment.nodes():
            graph.assign(node, sid)
    for sid, segment in enumerate(txt_branch):
        for node in segment.nodes():
            graph.assign(node, sid + resource.ngpus // 2)
    stage_tp(graph, mm_branch, list(range(resource.ngpus)))
    replica(graph, dl, list(range(resource.ngpus)))

    # print(graph.extra_repr())
    if args.gbs // args.mbs == 1:
        return graph

    tsched = TSched.load(args.load_tsched)
    print(f'>>> loaded tschedule: \n{tsched}')

    # setup blocks
    """
    0   4
      2 4
    1   4
      3 4 
    """
    block2seg = {
        0: img_branch[0],
        1: txt_branch[0],
        2: img_branch[1],
        3: txt_branch[1],
        4: mm_branch
    }
    schedule(graph, tsched, args.gbs // args.mbs, block2seg)
    return graph


def inference():

    assert args.hidden % args.heads == 0, 'hidden must be divisible by heads'
    cfg = Config(
        hidden_size=args.hidden,
        num_heads=args.heads,
        num_layers=args.layers
    )
    if DeviceGroup().local_rank == 0:
        model = FLAVAModel(cfg) # .cuda()
        model = model.half() if args.fp16 else model
        model = model.eval()
    else:
        model = None

    dataloader = ImageTextDataLoader(
        batch_size=args.mbs,
        dtype=torch.float16 if args.fp16 else torch.float32,
        cfg=cfg
    )

    runtime_policy = None
    if args.premise == '1f1b':
        runtime_policy = PAS1F1B
    elif args.premise == 'yshape':
        runtime_policy = PASYShape
    elif args.premise == 'tp':
        runtime_policy = PASTP

    model = cube.SemanticModel(model)
    @cube.compile(model, dataloader, PAS=runtime_policy, load_content=False)
    def serve(model, dataloader):
        image, text = next(dataloader)
        logits = model(image, text)
    
    model = cube.load_model(load_content=False)

    torch.distributed.barrier()
    print_each_rank('model weight consumption:')
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

        with torch.no_grad():
            serve(model, dataloader)

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

    inference()