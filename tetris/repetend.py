
from typing import List, Tuple, Dict
from tetris.schedplan import SchedPlan, Block


Devices = Tuple[int]


class MicroPicker:

    @staticmethod
    def iter_chain(nblocks: int, num_microbatches: int, _res = None):
        if _res is None:
            _res, nblocks = [num_microbatches - 1], nblocks-1
        if nblocks == 0:
            yield _res
        else:
            max_mid = min(num_microbatches - 1, _res[-1]) \
                if len(_res) > 0 else num_microbatches - 1
            # for m in range(max_mid, -1, -1):
            #     _res.append(m)
            #     for res in MicroPicker.iter_chain(nblocks-1, num_microbatches, _res):
            #         yield res
            #     _res.pop(-1)
            mids = [max_mid, max_mid - 1] if max_mid - 1 > 0 else [max_mid]
            if max_mid - 1 >= 0 and nblocks == max_mid + 1:
                mids = [max_mid - 1]
            for m in mids:
                _res.append(m)
                for res in MicroPicker.iter_chain(nblocks-1, num_microbatches, _res):
                    yield res
                _res.pop(-1)
    
    @staticmethod
    def pick(micros: List[SchedPlan]) -> Tuple[List[Block], List[Block], List[Block], Dict[Block, Devices]]:
        """
        
        @yield warmup blocks, repetend blocks, cooldown blocks, block2device mapping
        """
        nmicros = len(micros)
        # collect device mapping
        block2device = {}
        for micro in micros:
            for block in micro.all_blocks():
                block2device[block] = micro.device(block)

        ref = micros[0]
        blocks = ref.chain_blocks()
    
        # TODO: support multi-branch
        for mids in MicroPicker.iter_chain(len(blocks), nmicros):
            warmup, repetend, cooldown = [], [], []
            print(f'assigning mids: {mids}')
            # collect repetend blocks
            for idx, (mid, block) in enumerate(zip(mids, blocks)):
                blk = micros[mid].chain_blocks()[idx]
                repetend.append(blk)
            # collect warmup and cooldown blocks
            for mid, micro in enumerate(micros):
                for block in micro.all_blocks():
                    if block in repetend: continue
                    idx = micro.chain_blocks().index(block)
                    if idx < mids.index(mid):
                        warmup.append(block)
                    else:
                        cooldown.append(block)
            print(f'warmup: {warmup}\nrepetend: {repetend}\ncooldown: {cooldown}')
            yield warmup, repetend, cooldown, block2device
