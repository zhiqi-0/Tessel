import os
from dataclasses import dataclass


def to_float(s: str) -> int:
    s = os.environ.get(s, None)
    if s is None: return None
    return float(s)


@dataclass
class SearchFlag:

    param_limit = to_float('PARAM_LIMIT')  # parameter limit in GB
    act_limit = to_float('ACT_LIMIT')  # activation limit in GB
    mem_limit = to_float('MEM_LIMIT')  # in GB

    def __repr__(cls):
        return f'SearchFlag(param_limit={cls.param_limit}, act_limit={cls.act_limit}, mem_limit={cls.mem_limit})'
