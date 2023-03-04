"""
Premise utils
"""
from typing import List, Tuple


def layer_division_rules(nstages: int, block_comp_cost: List[float],
                         adaptive: bool = True) -> List[Tuple[int, int]]:
    """
    Layer division

    @param nstages int: number of stages
    @param adaptive bool: reallocate pipeline

    @return 
    """
    nlayers = len(block_comp_cost)
    if nstages == 1: return [(0, nlayers),]
    divisions: List[Tuple[int, int]] = []
    start, end = 0, 1
    # uniform partition
    if not adaptive:
        for sid in range(nstages):
            addone = 1 if sid < len(block_comp_cost) % nstages else 0
            end = start + len(block_comp_cost) // nstages + addone
            divisions.append((start, end))
            start = end
        assert end == len(block_comp_cost)
    # adaptive partition
    else:
        # ====================== Rule ========================
        # maxmial layer number constraints due to memory consumption
        limits = [None] * nstages
        # ====================================================
        remain_time = sum(block_comp_cost)
        start, end = 0, 1
        for sid in range(nstages):
            budget = remain_time / (nstages - sid)
            accum = block_comp_cost[start]
            while end < len(block_comp_cost):
                if limits[sid] is not None and (end - start) == limits[sid]:
                    break
                if budget - accum < 0.5 * block_comp_cost[end]:
                    break
                accum += block_comp_cost[end]
                end += 1
            remain_time -= accum
            if sid == nstages - 1:
                end = len(block_comp_cost)
            divisions.append((start, end))
            start, end = end, end + 1
    return divisions