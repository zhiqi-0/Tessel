"""
Compile with searched scheduling plans
"""
from typing import Callable

import cube
from cube.graph.schedule.schedplan import SchedulePlan
from cube.graph import IRGraph
from cube.graph.segment import IRSegment

from tetris.schedplan import SchedPlan as TPlan


def policy(graph: IRGraph, resource, premise_fn: Callable, schedplan: TPlan):
    """
    Cube sProgram to generate schedule plan
    
    @param graph IRGraph: model graph
    @param resource Resource: environment
    @param premise_fn Callable: the funtion to change graph into premise shape
    @param schedplan TPlan: searched Tetris plan

    @return graph IRGraph: transformed graph
    """
    num_microbatches = 16
    
    # staging and assign graph
    graph = premise_fn(graph)

    segments = graph.select(ntype=IRSegment)
    fsegments = [seg for seg in segments if seg.isfw()]
    assert all(len(seg.device) > 0 for seg in fsegments), \
        f"Expected each segment is assigned to devices"
    
    SchedulePlan(graph, num_microbatches)
    for step in range(schedplan.nsteps):
        pass