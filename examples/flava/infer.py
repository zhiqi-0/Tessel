"""
PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun --nproc_per_node=4 \
    examples/flava/infer.py \
        --fp16 --layers 12 --hidden 768 --heads 12 \
        --premise 1f1b --mbs 1 --gbs 8 \
        --load-tsched flava.yshape.tsched.4stages.json
"""

from functools import partial
import torch

from examples.flava.model import FLAVAModel, Config, ImageTextDataLoader
from examples.flava.placement import vshape, kshape, tp_func, PASTetris

import cube
from cube.profiler.timer import CudaTimer, print_each_rank
from cube.profiler.memory import memory_summary
from cube.runtime.device import DeviceGroup

from tetris.runtime.policy import PAS1F1B, PASFullTP
from tetris.config import build_config, build_parser

import argparse

parser = argparse.ArgumentParser(parents=[build_parser()], description='Flava Inference')
parser.add_argument('--fp16', action='store_true', default=False)
parser.add_argument('--mbs', type=int, default=1, help='micro-batch size')
parser.add_argument('--gbs', type=int, default=8, help='global batch size')

parser.add_argument('--layers', type=int, required=True)
parser.add_argument('--hidden', type=int, required=True)
parser.add_argument('--heads', type=int, required=True)

# policy
parser.add_argument('--premise', type=str,
                    choices=['1f1b', 'tetris', 'tp'],
                    help='premise shape')
# log save
parser.add_argument('--load-tsched', type=str, default=None,
                    help='load searched tetris schedule from file')
args = parser.parse_args()

cube.init()
print_each_rank(str(args), rank_only=0)


def inference():

    config = build_config(parser)
    print_each_rank(config, rank_only=0)

    if args.premise == '1f1b':
        runtime_policy = partial(PAS1F1B,
                                 mbs=args.mbs,
                                 nmicros=args.gbs//args.mbs,
                                 premise=vshape,
                                 config=config,
                                 sched='1f1b')
    elif args.premise == 'tetris':
        runtime_policy = partial(PASTetris,
                                 mbs=args.mbs,
                                 nmicros=args.gbs//args.mbs,
                                 premise=kshape,
                                 config=config,
                                 load_sched=args.load_tsched)
    elif args.premise == 'tp':
        runtime_policy = partial(PASFullTP,
                                 mbs=args.mbs,
                                 nmicros=args.gbs//args.mbs,
                                 tp_func=tp_func,
                                 config=config)
    else:
        raise KeyError

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