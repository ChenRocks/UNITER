"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

sampler for length bucketing (batch by tokens)
"""
import random

from torch.utils.data import Sampler
from cytoolz import partition_all


class TokenBucketSampler(Sampler):
    def __init__(self, lens, bucket_size, batch_size,
                 droplast=False, size_multiple=8):
        self._lens = lens
        self._max_tok = batch_size
        self._bucket_size = bucket_size
        self._droplast = droplast
        self._size_mul = size_multiple

    def _create_ids(self):
        return list(range(len(self._lens)))

    def _sort_fn(self, i):
        return self._lens[i]

    def __iter__(self):
        ids = self._create_ids()
        random.shuffle(ids)
        buckets = [sorted(ids[i:i+self._bucket_size],
                          key=self._sort_fn, reverse=True)
                   for i in range(0, len(ids), self._bucket_size)]
        # fill batches until max_token (include padding)
        batches = []
        for bucket in buckets:
            max_len = 0
            batch_indices = []
            for indices in partition_all(self._size_mul, bucket):
                max_len = max(max_len, max(self._lens[i] for i in indices))
                if (max_len * (len(batch_indices) + self._size_mul)
                        > self._max_tok):
                    if not batch_indices:
                        raise ValueError(
                            "max_tokens too small / max_seq_len too long")
                    assert len(batch_indices) % self._size_mul == 0
                    batches.append(batch_indices)
                    batch_indices = list(indices)
                else:
                    batch_indices.extend(indices)
            if not self._droplast and batch_indices:
                batches.append(batch_indices)
        random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        raise ValueError("NOT supported. "
                         "This has some randomness across epochs")
