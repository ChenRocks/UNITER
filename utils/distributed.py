"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

distributed API using Horovod
Modified from OpenNMT's native pytorch distributed utils
(https://github.com/OpenNMT/OpenNMT-py)
"""
import math
import pickle

import torch
from horovod import torch as hvd


def all_reduce_and_rescale_tensors(tensors, rescale_denom):
    """All-reduce and rescale tensors at once (as a flattened tensor)

    Args:
        tensors: list of Tensors to all-reduce
        rescale_denom: denominator for rescaling summed Tensors
    """
    # buffer size in bytes, determine equiv. # of elements based on data type
    sz = sum(t.numel() for t in tensors)
    buffer_t = tensors[0].new(sz).zero_()

    # copy tensors into buffer_t
    offset = 0
    for t in tensors:
        numel = t.numel()
        buffer_t[offset:offset+numel].copy_(t.view(-1))
        offset += numel

    # all-reduce and rescale
    hvd.allreduce_(buffer_t[:offset])
    buffer_t.div_(rescale_denom)

    # copy all-reduced buffer back into tensors
    offset = 0
    for t in tensors:
        numel = t.numel()
        t.view(-1).copy_(buffer_t[offset:offset+numel])
        offset += numel


def all_reduce_and_rescale_tensors_chunked(tensors, rescale_denom,
                                           buffer_size=10485760):
    """All-reduce and rescale tensors in chunks of the specified size.

    Args:
        tensors: list of Tensors to all-reduce
        rescale_denom: denominator for rescaling summed Tensors
        buffer_size: all-reduce chunk size in bytes
    """
    # buffer size in bytes, determine equiv. # of elements based on data type
    buffer_t = tensors[0].new(
        math.ceil(buffer_size / tensors[0].element_size())).zero_()
    buffer = []

    def all_reduce_buffer():
        # copy tensors into buffer_t
        offset = 0
        for t in buffer:
            numel = t.numel()
            buffer_t[offset:offset+numel].copy_(t.view(-1))
            offset += numel

        # all-reduce and rescale
        hvd.allreduce_(buffer_t[:offset])
        buffer_t.div_(rescale_denom)

        # copy all-reduced buffer back into tensors
        offset = 0
        for t in buffer:
            numel = t.numel()
            t.view(-1).copy_(buffer_t[offset:offset+numel])
            offset += numel

    filled = 0
    for t in tensors:
        sz = t.numel() * t.element_size()
        if sz > buffer_size:
            # tensor is bigger than buffer, all-reduce and rescale directly
            hvd.allreduce_(t)
            t.div_(rescale_denom)
        elif filled + sz > buffer_size:
            # buffer is full, all-reduce and replace buffer with grad
            all_reduce_buffer()
            buffer = [t]
            filled = sz
        else:
            # add tensor to buffer
            buffer.append(t)
            filled += sz

    if len(buffer) > 0:
        all_reduce_buffer()


def broadcast_tensors(tensors, root_rank, buffer_size=10485760):
    """broadcast tensors in chunks of the specified size.

    Args:
        tensors: list of Tensors to broadcast
        root_rank: rank to broadcast
        buffer_size: broadcast chunk size in bytes
    """
    # buffer size in bytes, determine equiv. # of elements based on data type
    buffer_t = tensors[0].new(
        math.ceil(buffer_size / tensors[0].element_size())).zero_()
    buffer = []

    def broadcast_buffer():
        # copy tensors into buffer_t
        offset = 0
        for t in buffer:
            numel = t.numel()
            buffer_t[offset:offset+numel].copy_(t.view(-1))
            offset += numel

        # broadcast
        hvd.broadcast_(buffer_t[:offset], root_rank)

        # copy all-reduced buffer back into tensors
        offset = 0
        for t in buffer:
            numel = t.numel()
            t.view(-1).copy_(buffer_t[offset:offset+numel])
            offset += numel

    filled = 0
    for t in tensors:
        sz = t.numel() * t.element_size()
        if sz > buffer_size:
            # tensor is bigger than buffer, broadcast directly
            hvd.broadcast_(t, root_rank)
        elif filled + sz > buffer_size:
            # buffer is full, broadcast and replace buffer with tensor
            broadcast_buffer()
            buffer = [t]
            filled = sz
        else:
            # add tensor to buffer
            buffer.append(t)
            filled += sz

    if len(buffer) > 0:
        broadcast_buffer()


def _encode(enc, max_size, use_max_size=False):
    enc_size = len(enc)
    enc_byte = max(math.floor(math.log(max_size, 256)+1), 1)
    if use_max_size:
        # this is used for broadcasting
        buffer_ = torch.cuda.ByteTensor(max_size+enc_byte)
    else:
        buffer_ = torch.cuda.ByteTensor(enc_size+enc_byte)
    remainder = enc_size
    for i in range(enc_byte):
        base = 256 ** (enc_byte-i-1)
        buffer_[i] = remainder // base
        remainder %= base
    buffer_[enc_byte:enc_byte+enc_size] = torch.ByteTensor(list(enc))
    return buffer_, enc_byte


def _decode(buffer_, enc_byte):
    size = sum(256 ** (enc_byte-i-1) * buffer_[i].item()
               for i in range(enc_byte))
    bytes_list = bytes(buffer_[enc_byte:enc_byte+size].tolist())
    shift = size + enc_byte
    return bytes_list, shift


_BUFFER_SIZE = 4096


def all_gather_list(data):
    """Gathers arbitrary data from all nodes into a list."""
    enc = pickle.dumps(data)

    enc_size = len(enc)
    max_size = hvd.allgather(torch.tensor([enc_size]).cuda()).max().item()
    in_buffer, enc_byte = _encode(enc, max_size)

    out_buffer = hvd.allgather(in_buffer[:enc_byte+enc_size])

    results = []
    for _ in range(hvd.size()):
        bytes_list, shift = _decode(out_buffer, enc_byte)
        out_buffer = out_buffer[shift:]
        result = pickle.loads(bytes_list)
        results.append(result)
    return results


def any_broadcast(data, root_rank):
    """broadcast arbitrary data from root_rank to all nodes."""
    enc = pickle.dumps(data)

    max_size = hvd.allgather(torch.tensor([len(enc)]).cuda()).max().item()
    buffer_, enc_byte = _encode(enc, max_size, use_max_size=True)

    hvd.broadcast_(buffer_, root_rank)

    bytes_list, _ = _decode(buffer_, enc_byte)
    result = pickle.loads(bytes_list)
    return result
