"""CUDA rank setup and solver entry guards for multi-GPU execution."""

from datetime import timedelta
import os

import taichi as ti

from src.utils import GlobalVariable
from src.utils.errors import ConfigurationError


def init_distributed_cuda(**taichi_options):
    launch_error = "distributed initialization requires torchrun with at least two ranks"
    try:
        world_size = int(os.environ["WORLD_SIZE"])
        local_size = int(os.environ["LOCAL_WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    except (KeyError, ValueError) as exc:
        raise ConfigurationError(launch_error) from exc
    if world_size < 2 or not 0 <= local_rank < local_size <= world_size:
        raise ConfigurationError(launch_error)

    try:
        import torch
        import torch.distributed as dist
    except ImportError as exc:
        raise ConfigurationError("distributed initialization requires PyTorch with CUDA and NCCL") from exc
    if not dist.is_available() or not dist.is_nccl_available():
        raise ConfigurationError("distributed initialization requires PyTorch with CUDA and NCCL")

    initialized = dist.is_initialized()
    if initialized:
        if dist.get_backend() != "nccl" or dist.get_world_size() != world_size:
            raise ConfigurationError("the existing process group must use NCCL and match WORLD_SIZE")
    else:
        if torch.cuda.is_initialized():
            raise ConfigurationError("call gt.init(distributed=True) before initializing PyTorch CUDA")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        devices = [str(i) for i in range(local_size)]
        if visible is not None:
            devices = [device.strip() for device in visible.split(",")[:local_size]]
        if len(set(devices)) != local_size or not all(devices):
            raise ConfigurationError("distributed initialization requires one distinct CUDA GPU per local rank")
        os.environ["CUDA_VISIBLE_DEVICES"] = devices[local_rank]

    if torch.cuda.device_count() != 1:
        raise ConfigurationError("distributed initialization requires exactly one visible CUDA GPU per rank")
    torch.cuda.set_device(0)
    ti.init(arch=ti.cuda, enable_fallback=False, **taichi_options)
    if ti.lang.impl.current_cfg().arch != ti.cuda:
        raise ConfigurationError("distributed initialization requires the Taichi CUDA backend")
    if not initialized:
        dist.init_process_group("nccl", timeout=timedelta(minutes=20))


def require_single_rank(solver_name):
    if GlobalVariable.DISTRIBUTED:
        raise ConfigurationError(f"{solver_name} does not support distributed execution")
