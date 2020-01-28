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
        buffer_size: all-reduce chunk size in bytes
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


def all_gather_list(data, max_size=4096):
    """Gathers arbitrary data from all nodes into a list."""
    world_size = hvd.size()
    if not hasattr(all_gather_list, '_in_buffer') or \
            max_size != all_gather_list._in_buffer.size():
        all_gather_list._in_buffer = torch.cuda.ByteTensor(max_size)
    in_buffer = all_gather_list._in_buffer

    enc = pickle.dumps(data)
    enc_size = len(enc)
    if enc_size + 2 > max_size:
        raise ValueError(
            'encoded data exceeds max_size: {}'.format(enc_size + 2))
    assert max_size < 255*256
    in_buffer[0] = enc_size // 255  # this encoding works for max_size < 65k
    in_buffer[1] = enc_size % 255
    in_buffer[2:enc_size+2] = torch.ByteTensor(list(enc))

    # FIXME cannot create buffer
    out = hvd.allgather(in_buffer.cuda())

    results = []
    for i in range(0, max_size*world_size, max_size):
        out_buffer = out[i:i+max_size]
        size = (255 * out_buffer[0].item()) + out_buffer[1].item()

        bytes_list = bytes(out_buffer[2:size+2].tolist())
        result = pickle.loads(bytes_list)
        results.append(result)
    return results


def any_broadcast(data, root_rank, max_size=4096):
    """broadcast arbitrary data from root_rank to all nodes."""
    if not hasattr(any_broadcast, '_in_buffer') or \
            max_size != any_broadcast._in_buffer.size():
        any_broadcast._buffer = torch.cuda.ByteTensor(max_size)
    buffer_ = any_broadcast._buffer

    enc = pickle.dumps(data)
    enc_size = len(enc)
    if enc_size + 2 > max_size:
        raise ValueError(
            'encoded data exceeds max_size: {}'.format(enc_size + 2))
    assert max_size < 255*256
    buffer_[0] = enc_size // 255  # this encoding works for max_size < 65k
    buffer_[1] = enc_size % 255
    buffer_[2:enc_size+2] = torch.ByteTensor(list(enc))

    hvd.broadcast_(buffer_, root_rank)

    size = (255 * buffer_[0].item()) + buffer_[1].item()

    bytes_list = bytes(buffer_[2:size+2].tolist())
    result = pickle.loads(bytes_list)
    return result
