"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

run inference for Image Text Retrieval
"""
import argparse
import json
import os
from os.path import exists
import pickle
from time import time

import torch
from torch.utils.data import DataLoader

from apex import amp
from horovod import torch as hvd

from data import (PrefetchLoader,
                  DetectFeatLmdb, TxtTokLmdb, ItmEvalDataset, itm_eval_collate)
from model.itm import UniterForImageTextRetrieval

from utils.logger import LOGGER
from utils.distributed import all_gather_list
from utils.misc import Struct
from utils.const import IMG_DIM
from utils.itm_eval import inference, itm_eval


def main(opts):
    hvd.init()
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    rank = hvd.rank()
    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, hvd.rank(), opts.fp16))

    if opts.train_config is not None:
        train_opts = Struct(json.load(open(opts.train_config)))
        opts.conf_th = train_opts.conf_th
        opts.max_bb = train_opts.max_bb
        opts.min_bb = train_opts.min_bb
        opts.num_bb = train_opts.num_bb

    # load DBs and image dirs
    eval_img_db = DetectFeatLmdb(opts.img_db,
                                 opts.conf_th, opts.max_bb,
                                 opts.min_bb, opts.num_bb,
                                 opts.compressed_db)
    eval_txt_db = TxtTokLmdb(opts.txt_db, -1)
    eval_dataset = ItmEvalDataset(eval_txt_db, eval_img_db, opts.batch_size)

    # Prepare model
    checkpoint = torch.load(opts.checkpoint)
    model = UniterForImageTextRetrieval.from_pretrained(
        opts.model_config, checkpoint, img_dim=IMG_DIM)
    if 'rank_output' not in checkpoint:
        model.init_output()  # zero shot setting

    model.to(device)
    model = amp.initialize(model, enabled=opts.fp16, opt_level='O2')

    eval_dataloader = DataLoader(eval_dataset, batch_size=1,
                                 num_workers=opts.n_workers,
                                 pin_memory=opts.pin_mem,
                                 collate_fn=itm_eval_collate)
    eval_dataloader = PrefetchLoader(eval_dataloader)

    eval_log, results = evaluate(model, eval_dataloader)
    if hvd.rank() == 0:
        if not exists(opts.output_dir) and rank == 0:
            os.makedirs(opts.output_dir)
        with open(f'{opts.output_dir}/config.json', 'w') as f:
            json.dump(vars(opts), f)
        with open(f'{opts.output_dir}/results.bin', 'wb') as f:
            pickle.dump(results, f)
        with open(f'{opts.output_dir}/scores.json', 'w') as f:
            json.dump(eval_log, f)
        LOGGER.info(f'evaluation finished')
        LOGGER.info(
            f"======================== Results =========================\n"
            f"image retrieval R1: {eval_log['img_r1']*100:.2f},\n"
            f"image retrieval R5: {eval_log['img_r5']*100:.2f},\n"
            f"image retrieval R10: {eval_log['img_r10']*100:.2f}\n"
            f"text retrieval R1: {eval_log['txt_r1']*100:.2f},\n"
            f"text retrieval R5: {eval_log['txt_r5']*100:.2f},\n"
            f"text retrieval R10: {eval_log['txt_r10']*100:.2f}")
        LOGGER.info("========================================================")


@torch.no_grad()
def evaluate(model, eval_loader):
    model.eval()
    st = time()
    LOGGER.info("start running Image/Text Retrieval evaluation ...")
    score_matrix = inference(model, eval_loader)
    dset = eval_loader.dataset
    all_score = hvd.allgather(score_matrix)
    all_txt_ids = [i for ids in all_gather_list(dset.ids)
                   for i in ids]
    all_img_ids = dset.all_img_ids
    assert all_score.size() == (len(all_txt_ids), len(all_img_ids))
    if hvd.rank() != 0:
        return {}, tuple()
    # NOTE: only use rank0 to compute final scores
    eval_log = itm_eval(all_score, all_txt_ids, all_img_ids,
                        dset.txt2img, dset.img2txts)

    results = (all_score, all_txt_ids, all_img_ids)
    tot_time = time()-st
    LOGGER.info(f"evaluation finished in {int(tot_time)} seconds, ")
    return eval_log, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--txt_db", default=None, type=str,
                        help="The input train corpus. (LMDB)")
    parser.add_argument("--img_db", default=None, type=str,
                        help="The input train images.")
    parser.add_argument("--checkpoint", default=None, type=str,
                        help="model checkpoint binary")
    parser.add_argument("--model_config", default=None, type=str,
                        help="model config json")
    parser.add_argument(
        "--output_dir", default=None, type=str,
        help="The output directory where the inference results will be "
             "written.")

    # optional parameters
    parser.add_argument("--train_config", default=None, type=str,
                        help="hps.json from training (for prepro hps)")
    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')
    parser.add_argument('--conf_th', type=float, default=0.2,
                        help='threshold for dynamic bounding boxes '
                             '(-1 for fixed)')
    parser.add_argument('--max_bb', type=int, default=100,
                        help='max number of bounding boxes')
    parser.add_argument('--min_bb', type=int, default=10,
                        help='min number of bounding boxes')
    parser.add_argument('--num_bb', type=int, default=36,
                        help='static number of bounding boxes')
    parser.add_argument("--batch_size", default=400, type=int,
                        help="number of tokens in a batch")

    # device parameters
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true',
                        help="pin memory")

    args = parser.parse_args()

    main(args)
