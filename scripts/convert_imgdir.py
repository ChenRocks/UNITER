"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

convert image npz to LMDB
"""
import argparse
import glob
import io
import json
import multiprocessing as mp
import os
from os.path import basename, exists

from cytoolz import curry
import numpy as np
from tqdm import tqdm
import lmdb

import msgpack
import msgpack_numpy
msgpack_numpy.patch()


def _compute_nbb(img_dump, conf_th, max_bb, min_bb, num_bb):
    num_bb = max(min_bb, (img_dump['conf'] > conf_th).sum())
    num_bb = min(max_bb, num_bb)
    return int(num_bb)


@curry
def load_npz(conf_th, max_bb, min_bb, num_bb, fname, keep_all=False):
    try:
        img_dump = np.load(fname, allow_pickle=True)
        if keep_all:
            nbb = None
        else:
            nbb = _compute_nbb(img_dump, conf_th, max_bb, min_bb, num_bb)
        dump = {}
        for key, arr in img_dump.items():
            if arr.dtype == np.float32:
                arr = arr.astype(np.float16)
            if arr.ndim == 2:
                dump[key] = arr[:nbb, :]
            elif arr.ndim == 1:
                dump[key] = arr[:nbb]
            else:
                raise ValueError('wrong ndim')
    except Exception as e:
        # corrupted file
        print(f'corrupted file {fname}', e)
        dump = {}
        nbb = 0

    name = basename(fname)
    return name, dump, nbb


def dumps_npz(dump, compress=False):
    with io.BytesIO() as writer:
        if compress:
            np.savez_compressed(writer, **dump, allow_pickle=True)
        else:
            np.savez(writer, **dump, allow_pickle=True)
        return writer.getvalue()


def dumps_msgpack(dump):
    return msgpack.dumps(dump, use_bin_type=True)


def main(opts):
    if opts.img_dir[-1] == '/':
        opts.img_dir = opts.img_dir[:-1]
    split = basename(opts.img_dir)
    if opts.keep_all:
        db_name = 'all'
    else:
        if opts.conf_th == -1:
            db_name = f'feat_numbb{opts.num_bb}'
        else:
            db_name = (f'feat_th{opts.conf_th}_max{opts.max_bb}'
                       f'_min{opts.min_bb}')
    if opts.compress:
        db_name += '_compressed'
    if not exists(f'{opts.output}/{split}'):
        os.makedirs(f'{opts.output}/{split}')
    env = lmdb.open(f'{opts.output}/{split}/{db_name}', map_size=1024**4)
    txn = env.begin(write=True)
    files = glob.glob(f'{opts.img_dir}/*.npz')
    load = load_npz(opts.conf_th, opts.max_bb, opts.min_bb, opts.num_bb,
                    keep_all=opts.keep_all)
    name2nbb = {}
    with mp.Pool(opts.nproc) as pool, tqdm(total=len(files)) as pbar:
        for i, (fname, features, nbb) in enumerate(
                pool.imap_unordered(load, files, chunksize=128)):
            if not features:
                continue  # corrupted feature
            if opts.compress:
                dump = dumps_npz(features, compress=True)
            else:
                dump = dumps_msgpack(features)
            txn.put(key=fname.encode('utf-8'), value=dump)
            if i % 1000 == 0:
                txn.commit()
                txn = env.begin(write=True)
            name2nbb[fname] = nbb
            pbar.update(1)
        txn.put(key=b'__keys__',
                value=json.dumps(list(name2nbb.keys())).encode('utf-8'))
        txn.commit()
        env.close()
    if opts.conf_th != -1 and not opts.keep_all:
        with open(f'{opts.output}/{split}/'
                  f'nbb_th{opts.conf_th}_'
                  f'max{opts.max_bb}_min{opts.min_bb}.json', 'w') as f:
            json.dump(name2nbb, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", default=None, type=str,
                        help="The input images.")
    parser.add_argument("--output", default=None, type=str,
                        help="output lmdb")
    parser.add_argument('--nproc', type=int, default=8,
                        help='number of cores used')
    parser.add_argument('--compress', action='store_true',
                        help='compress the tensors')
    parser.add_argument('--keep_all', action='store_true',
                        help='keep all features, overrides all following args')
    parser.add_argument('--conf_th', type=float, default=0.2,
                        help='threshold for dynamic bounding boxes '
                             '(-1 for fixed)')
    parser.add_argument('--max_bb', type=int, default=100,
                        help='max number of bounding boxes')
    parser.add_argument('--min_bb', type=int, default=10,
                        help='min number of bounding boxes')
    parser.add_argument('--num_bb', type=int, default=100,
                        help='number of bounding boxes (fixed)')
    args = parser.parse_args()
    main(args)
