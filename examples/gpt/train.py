
import torch
from functools import partial

from examples.gpt.model import Config, GPT, get_gpt_dummy_dataloader
from examples.gpt.placement import vshape, xshape, mshape, tp_func

import cube
from cube.profiler.timer import CudaTimer, print_each_rank
from cube.profiler.memory import memory_summary
from cube.runtime.device import DeviceGroup

from tessel.runtime.policy import PAS1F1B, PAS1F1BPlus, PASChimera, PASTessel, PASFullTP
from tessel.runtime.config import build_config, build_parser

import argparse

parser = argparse.ArgumentParser(parents=[build_parser()], description='GPT Train')
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
parser.add_argument('--premise', type=str, choices=['1f1b', 'tessel', 'gpipe', 'tp', 'chimera', '1f1b+'],
                    help='premise shape')
# log save
parser.add_argument('--save', type=str, default=None,
                    help='folder for save searched results.')
parser.add_argument('--load-tsched', type=str, default=None,
                    help='load searched tessel schedule from file')
args = parser.parse_args()

cube.init()
print_each_rank(str(args), rank_only=0)


def train():

    config = build_config(parser)
    print_each_rank(config, rank_only=0)

    if args.premise == '1f1b':
        runtime_policy = partial(PAS1F1B,
                                 mbs=args.mbs,
                                 nmicros=args.gbs//args.mbs,
                                 premise=vshape,
                                 config=config,
                                 sched='1f1b')
    elif args.premise == 'gpipe':
        runtime_policy = partial(PAS1F1B,
                                 mbs=args.mbs,
                                 nmicros=args.gbs//args.mbs,
                                 premise=vshape,
                                 config=config,
                                 sched='gpipe')
    elif args.premise == '1f1b+':
        runtime_policy = partial(PAS1F1BPlus,
                                 mbs=args.mbs,
                                 nmicros=args.gbs//args.mbs,
                                 premise=mshape,
                                 config=config)
    elif args.premise == 'chimera':
        args.mbs = args.mbs * 2 # double for chimera execution
        runtime_policy = partial(PASChimera,
                                 mbs=args.mbs,
                                 nmicros=args.gbs//args.mbs,
                                 premise=xshape,
                                 config=config)
    elif args.premise == 'tessel':
        runtime_policy = partial(PASTessel,
                                 mbs=args.mbs,
                                 nmicros=args.gbs//args.mbs,
                                 premise=mshape,
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
    dataloader = get_gpt_dummy_dataloader(args.mbs, cfg)

    if torch.distributed.get_rank() == 0:
        nparams = 0
        for param in model.parameters():
            nparams += param.nelement()
        print(f'full model parameter: {nparams}')

    @cube.compile(model, dataloader, PAS=runtime_policy, override=True, load_content=False, 
                  comm_cost_fn=lambda x: 1)
    def train_iter(model, dataloader):
        datas = next(dataloader)
        loss = model(*datas)
        loss.backward()
        # return loss
    model: torch.nn.Module = cube.load_model(load_content=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999))

    torch.distributed.barrier()
    print_each_rank('model weight consumpition:')
    memory_summary()
    nparams = 0
    for param in model.parameters():
        nparams += param.nelement()
    print_each_rank(f'model parameter: {nparams}')

    CudaTimer(enable=False).warmup()
    dataloader = iter(dataloader)
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
