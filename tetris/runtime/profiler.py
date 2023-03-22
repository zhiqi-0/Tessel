from typing import Callable, Tuple, Union, Optional, Dict, NewType, List
import torch
import time
import os
import json

import cube
from cube.ir.cten import IRTensor, IRObject, IRCell
from cube.ir.operator import IRFwOperation
from cube.graph.parser.dtype import IRDType2TorchDType
from cube.graph.parser.register import CustomizedOps


Shapes = NewType('Shapes', Tuple[Tuple[int]])
DTypes = NewType('DTypes', Tuple[torch.dtype])
ShapesDTypes = NewType('ShapesDTypes', Tuple[Shapes, DTypes])
NameOrFunc = Union[str, Callable]


_train_module_ref: torch.nn.Module = torch.nn.Module().train()
_eval_module_ref: torch.nn.Module = torch.nn.Module().eval()


class CompProfiler:

    @staticmethod
    def profile(node: IRCell,
                warmup_sec: float = 2, prof_times: int = 50) -> Tuple[float, float, int, Tuple[int]]:
        """
        Profile a function

        @param func Callable: the callable function, e.g., torch.nn.functional.linear
        @param shapes Tuple[Tuple[int]]: the shapes of each input tensor
        @param dtypes Optional[Tuple[torch.dtype]]: the dtype of each input tensor. Default will use torch.float32
        @param warmup_sec float: warmup seconds
        @param prof_times int: profile times
        @param kwargs Dict: other keyword argument for func call.

        @return fw_span float: the time in milliseconds for forward time
        @return bw_span float: the time in milliseconds for backward time
        @return infer_mem int: the peak memory in bytes after inference of the function
        @return train_mem_info Tuple[int]: byte sizes of tensors saved for backward
        """
        torch.cuda.empty_cache()
        print(f'current GPU memory: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024} GB')

        func: Callable = CompProfiler.get_func(node)
        args, kwargs = CompProfiler.get_inputs(node, train=True)
    
        # prepare gradients
        with torch.no_grad():
            outputs = func(*args, **kwargs)
        outputs = (outputs,) if torch.is_tensor(outputs) else outputs
        assert all(torch.is_tensor(otensor) for otensor in outputs), \
            f"{func.__name__}: require all the outputs to be tensors"
        grads = tuple(torch.zeros_like(otensor) for otensor in outputs)
        del outputs

        def run_step(func, tensors, kwargs, backward: bool):
            if not backward:
                with torch.no_grad():
                    outputs = func(*tensors, **kwargs)
            else:
                outputs = func(*tensors, **kwargs)
                torch.autograd.backward(outputs, grads)

        # ================ measure training peak memory ====================
        # inference
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        mtic = torch.cuda.max_memory_allocated()  # in bytes
        run_step(func, args, kwargs, backward=False)
        mtoc = torch.cuda.max_memory_allocated()
        infer_memory = mtoc - mtic

        # training
        train_memory, used_tensor = 0, set()
        def pack_hook(x):
            nonlocal train_memory, used_tensor
            if x.storage().data_ptr() not in used_tensor:
                used_tensor.add(x.storage().data_ptr())
                byte_size = x.element_size()
                for dim in list(x.size()):
                    byte_size = byte_size * dim
                train_memory += byte_size
            return x
        def unpack_hook(x): return x

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            run_step(func, args, kwargs, backward=True)

        # for ptr in used_tensor:
        #     torch.cuda.caching_allocator_delete(ptr)
        del used_tensor

        # ===================================================================

        # warmup
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        tic = time.time()
        while time.time() - tic < warmup_sec:
            run_step(func, args, kwargs, backward=True)
            torch.cuda.synchronize()

        def profile(backward: bool):
            torch.cuda.synchronize()
            tic = time.perf_counter()
            for _ in range(prof_times):
                run_step(func, args, kwargs, backward=backward)
            torch.cuda.synchronize()
            toc = time.perf_counter()
            return (toc - tic) / prof_times * 1000  # in milliseconds

        infer_span = profile(backward=False)
        train_span = profile(backward=True)
        
        return infer_span, infer_memory, train_span, train_memory

    @staticmethod
    def get_inputs(node: IRFwOperation, train: bool) -> Tuple[List, Dict]:
        # create data
        def dummy_torch_tensor(tensor: IRTensor):
            """Generate dummy input tenosrs"""
            dtype = IRDType2TorchDType.map(tensor.dtype)
            constructor = torch.zeros if dtype in (torch.int64, torch.int32, torch.bool) else torch.rand
            return constructor(tuple(tensor.shape), dtype=dtype, device=torch.cuda.current_device(), requires_grad=tensor.requires_grad)

        args = [dummy_torch_tensor(t) if isinstance(t, IRTensor) else t for t in node.inputs()]
        # replace kwargs starting with 'self.xxx'
        kwargs = {}
        for name, value in node.kwargs.items():
            if isinstance(value, str) and value.startswith('self.'):
                value = getattr(_train_module_ref, value[5:]) if train else getattr(_eval_module_ref, value[5:])
            kwargs[name] = value
        
        return args, kwargs

    @staticmethod
    def get_func(node: IRFwOperation) -> Callable:
        """
        Get function call
        """
        assert isinstance(node, IRFwOperation), f"Only support profiling forward operation but got {type(node)}"

        def get_dep_names(sign: str):
            ret = []
            code_impl = CustomizedOps.kOpCodeDef[sign]
            for code_line in code_impl.split('\n'):
                idx = code_line.find('# call: ')
                if idx != -1:
                    dep_name = code_line[idx + 8:]
                    assert dep_name in CustomizedOps.kOpCodeDef, dep_name
                    ret = ret + get_dep_names(dep_name)
                    ret.append(dep_name)
            return ret

        if node.signature in CustomizedOps.kOpCodeDef:
            dep_code_impl = ''
            for dep_name in get_dep_names(node.signature):
                dep_code_impl = dep_code_impl + CustomizedOps.kOpCodeDef[dep_name]
            code_impl: str = CustomizedOps.kOpCodeDef[node.signature]
            def_end = code_impl.find(':\n')
            assert def_end >= 0
            prev_code_lines = code_impl[:def_end+2]
            succ_code_lines = code_impl[def_end+2:]
            for line in dep_code_impl.split('\n'):
                prev_code_lines = prev_code_lines + '    ' + line + '\n'
            code_impl = prev_code_lines + succ_code_lines
            local = {}
            exec(code_impl, globals(), local)
            fn = list(local.values())[0]
        else:
            fn = eval(node.signature)
        return fn


class ProfileDataBase:

    def __init__(self, filename: Optional[str] = None) -> None:
        """!
        Create a database for profiling result
        """

        self._data: Dict[str, Dict[str, Tuple[float, float, int]]] = dict()
        if filename is not None:
            self.load(filename)

    def profile(self, node: IRFwOperation, device: Optional[int] = None):
        """
        Profile a forward node in IRGraph on a specific device (default current device)
        
        @param node IRFwOperation: node of IRGraph
        @param device int: the device that the node will execute on
        
        @return infer_span float: inference time in milliseconds
        @return infer_memory int: inference peak memory in bytes
        @return train_span flaot: train time in milliseconds
        @return train_memory int: train peak memory in bytes
        @return param_memory int: trained parameters
        """
        if self.exist(node):
            return self.query(node)

        if isinstance(device, int):
            orig_device = torch.cuda.current_device()
            torch.cuda.set_device(device)

        color, default = '\033[31m', '\033[0m'

        #FIXME: OOM will increase cuda allocated memory
        try:
            infer_span, infer_memory, train_span, train_memory = CompProfiler.profile(node)
            # log to database
            self.insert(node, infer_span, infer_memory, train_span, train_memory)
        except Exception as e:
            err = f'{color}profil error:\n {str(e)}{default}'
            print(err)
            infer_span, infer_memory, train_span, train_memory = e, e, e, e
        
        shapes = tuple(t.shape if isinstance(t, IRTensor) else None for t in node.inputs())
        dtypes = tuple(IRDType2TorchDType.map(t.dtype) if isinstance(t, IRTensor) else None for t in node.inputs())
        error = f'{color}None{default}'
        print(
            f"profiled {node.signature} | shapes: {shapes} | dtypes: {dtypes} => "
            f"infer: {round(infer_span, 2) if isinstance(infer_span, float) else error} ms | "
            f"{infer_memory if isinstance(infer_memory, int) else None} bytes ; "
            f"train: {round(train_span, 2) if isinstance(train_span, float) else error} ms | "
            f"{train_memory if isinstance(train_memory, int) else error} bytes")

        if isinstance(device, int):
            torch.cuda.set_device(orig_device)
        return infer_span, infer_memory, train_span, train_memory

    def insert(self, node: IRCell, infer_span: float, infer_memory: int,
               train_span: float, train_memory: int):
        """
        log (reset) the span of a node with key

        @param node IRCell
        @return infer_span float: inference time in milliseconds
        @return infer_memory int: inference peak memory in bytes
        @return train_span flaot: train time in milliseconds
        @return train_memory int: train peak memory in bytes
        """
        name = node.signature
        key = self._serialize(node)
        assert isinstance(name, str) and isinstance(key, str)
        if name not in self._data:
            self._data[name] = dict()
        infer_span = infer_span if isinstance(infer_span, float) else None
        infer_memory = infer_memory if isinstance(infer_memory, int) else None
        train_span = train_span if isinstance(train_span, float) else None
        train_memory = train_memory if isinstance(train_memory, int) else None
        self._data[name][key] = (infer_span, infer_memory, train_span, train_memory)

    def exist(self, node: IRFwOperation) -> bool:
        """
        Check if the node has the performance recorded in the database

        @param node IRFwOperation: forward operation

        @return exist bool: True if the performance is recorded, else False
        """
        key = self._serialize(node)
        if node.signature not in self._data:
            return False
        if key not in self._data[node.signature]:
            return False
        return True

    def query(self, node: IRFwOperation) -> Tuple[Tuple[int], Tuple[int], float, float, int, Tuple[int]]:
        """!
        Get the performance number of a node in IRGraph

        @param node IRFwOperation: node in IRGraph

        @return in_mem_info Tuple[int]: byte sizes of input tensors
        @return param_mem_info Tuple[int]: byte sizes of param tensors
        @return fw_span float: the forward span time in milliseconds
        @return bw_span float: the backward span time in milliseconds
        @return infer_memory int: the peak memory in bytes after inference of the function
        @return train_mem_info Tuple[int]: byte sizes of tensors saved for backward
        """
        key = self._serialize(node)
        if node.signature not in self._data:
            return None
        if key not in self._data[node.signature]:
            return None
        return self._data[node.signature][key]

    def query_func(self, signature, shapes, dtypes) -> Tuple[Tuple[int], Tuple[int], float, float, int, Tuple[int]]:
        """
        Get performance number of given name (signature), shapes and dtypes
        
        @param signature str: function signature
        @param shapes Tuple[Tuple[int]]: the shape of each input tensor
        @param dtypes Tuple[torch.dtype]: the dtype of each tensor

        @return in_mem_info Tuple[int]: byte sizes of input tensors
        @return param_mem_info Tuple[int]: byte sizes of param tensors
        @return fw_span float: the forward span time in milliseconds
        @return bw_span float: the backward span time in milliseconds
        @return infer_memory int: the peak memory in bytes after inference of the function
        @return train_mem_info Tuple[int]: byte sizes of tensors saved for backward
        """
        key = self._serialize(shapes, dtypes)
        if signature not in self._data:
            return None
        if key not in self._data[signature]:
            return None
        return self._data[signature][key]

    def query_args(self, signature: str) -> Tuple[List[Shapes], List[DTypes]]:
        """
        Get the recorded shapes and dtypes of 
        """
        item_shapes, item_dtypes = [], []
        if signature not in self._data:
            return item_shapes, item_dtypes
        for shapes_dtypes_str in self._data[torch.signature].keys():
            shapes, dtypes = self._deserialize(shapes_dtypes_str)
            item_shapes.append(shapes)
            item_dtypes.append(dtypes)
        return item_shapes, item_dtypes

    def _serialize(self, node: IRFwOperation) -> str:
        """
        Serialize the shapes, dtypes and kwargs into a string

        e.g.,
            shapes: ((1024,), (1024,1024))
            dtypes: (torch.float32, torch.float32)
        => ((1024,), (1024,1024)) : (torch.float32, torch.float32)

        @param shapes Tuple[Tuple[int]]: the shape of each tensor
        @param dtypes Tuple[torch.dtype]: the dtype of each tensor

        @return key str: the serialized string
        """
        shapes, dtypes = [], []
        for t in node.inputs():
            if isinstance(t, IRTensor):
                shapes.append(t.shape)
                dtypes.append(IRDType2TorchDType.map(t.dtype))
            elif isinstance(t, IRObject):
                raise RuntimeError('IRObject has not been supported in _serialize')
            else:
                shapes.append(None)
                dtypes.append(type(t))
        shapes = str(tuple(shapes))
        dtypes= str(tuple(dtypes))
        return shapes + ' : ' + dtypes

    def _deserialize(self, key: str) -> ShapesDTypes:
        """
        De-serialize the key string to shapes and dtypes

        e.g., (1024,)-(1024,1024)=torch.float32-torch.float32
        =>  shapes: ((1024,), (1024,1024))
            dtypes: (torch.float32, torch.float32)

        @param key str: the serialized string
        @return shapes_and_dtypes ShapesDTypes: shapes and dtypes
        """
        shapes, dtypes = key.split(' : ')
        shapes = eval(shapes)
        dtypes = eval(dtypes)
        # shapes = tuple(eval(shape) for shape in shapes.split('-'))
        # dtypes = tuple(eval(dtype) for dtype in dtypes.split('-'))
        return shapes, dtypes

    def dump(self, file: str, override=False):
        """!
        dump the profiled data into json format

        @param file str: the file name
        @param override bool: True if the existed can be overrided else False
        """
        if os.path.exists(file):
            assert override, f"File {file} exists. Set override = True to force dump."
        with open(file, 'w') as f:
            json.dump(self._data, f)

    def load(self, file: str):
        """!
        load the profiled data into data base. The original existed one will be
        overrided by the loaded data.

        @param file str: the file name
        """
        with open(file, 'r') as f:
            self._data = json.load(f)

    def __repr__(self) -> str:
        data = []
        for signature in self._data:
            for key in self._data[signature]:
                shapes, dtypes = self._deserialize(key)
                in_mem_info, param_mem_info, fw_span, bw_span, infer_mem, train_mem = self._data[signature][key]
                data.append(f'{signature}: shapes={shapes}, dtypes={dtypes}, in mem {in_mem_info} bytes, param mem {param_mem_info} bytes, fw span: {fw_span} ms, bw span: {bw_span} ms, infer mem {infer_mem} bytes, train mem {train_mem} bytes')
        data = '\n'.join(data)
        return data
