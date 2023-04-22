from typing import List, Dict, Optional
from cube.ir.cten import IRCell, IRTensor
from cube.ir.operator import IRFwOperation
from cube.graph.graph import IRGraph
from cube.graph.function.anchor import IRGraphAnchor


class IRLayerOp(IRCell):

    def __init__(self, nodes: List[IRCell], layer_id: int = None):
        super().__init__('layer_op', 'layer_op', 0, 0, init_outputs=False)
        self.nodes = nodes
        self.layer_id : int = layer_id

        self.param_size: int = 0  # in bytes
        for node in nodes:
            tsrs = (tsr for tsr in node.inputs() if isinstance(tsr, IRTensor))
            for tensor in tsrs:
                self.param_size += tensor.byte_size()


def cluster_to_layer_ops(nodes: List[IRFwOperation]) -> List[IRLayerOp]:
    layer_ops: List[IRLayerOp] = []
    ops = []
    for node in nodes:
        if isinstance(node, IRGraphAnchor):
            if len(ops) != 0:
                layer_ops.append(IRLayerOp(ops, layer_id=len(layer_ops)))
            ops = [node]
        elif isinstance(node, IRFwOperation):
            ops.append(node)
    if len(ops) != 0:
        layer_ops.append(IRLayerOp(ops, layer_id=len(layer_ops)))
    return layer_ops
