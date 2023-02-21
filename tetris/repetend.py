
from typing import List, Tuple
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
    def pick(micros: List[SchedPlan]) -> List[Tuple[Block, Devices]]:
        """
        
        @yield tuple of (block, devices)
        """

        nmicros = len(micros)
        nblocks = len(micros[0].all_blocks())

        print(nblocks, nmicros)
        ref = micros[0]

        # TODO: support multi-branch
        for mids in MicroPicker.iter_chain(nblocks, nmicros):
            block_and_devs = []
            print(f'assigning mids: {mids}')
            bidx = 0
            for step in range(ref.nsteps):
                ref_blocks = ref.blocks(step)
                if len(ref_blocks) == 0: continue
                assert len(ref_blocks) == 1
                mid = mids[bidx]
                blk = micros[mid].blocks(step)[0]
                block_and_devs.append((blk, micros[mid].device(blk)))
                bidx += 1
            # print(blocks)
            yield block_and_devs
