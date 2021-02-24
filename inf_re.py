"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

run inference of VQA for submission
"""
import argparse
import json
import os
from os.path import exists
from time import time

import torch
from torch.utils.data import DataLoader

from apex import amp
from horovod import torch as hvd
from cytoolz import concat

from data import (PrefetchLoader, DetectFeatLmdb, ReTxtTokLmdb,
                  ReEvalDataset, re_eval_collate)
from data.sampler import DistributedSampler
from model.re import UniterForReferringExpressionComprehension

from utils.logger import LOGGER
from utils.distributed import all_gather_list
from utils.misc import Struct
from utils.const import IMG_DIM


def write_to_tmp(txt, tmp_file):
    if tmp_file:
        f = open(tmp_file, "a")
        f.write(txt)


def main(opts):
    hvd.init()
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    rank = hvd.rank()
    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, hvd.rank(), opts.fp16))

    hps_file = f'{opts.output_dir}/log/hps.json'
    model_opts = json.load(open(hps_file))
    if 'mlp' not in model_opts:
        model_opts['mlp'] = 1
    model_opts = Struct(model_opts)
    # Prepare model
    if exists(opts.checkpoint):
        ckpt_file = opts.checkpoint
    else:
        ckpt_file = f'{opts.output_dir}/ckpt/model_epoch_{opts.checkpoint}.pt'
    checkpoint = torch.load(ckpt_file)
    model = UniterForReferringExpressionComprehension.from_pretrained(
        f'{opts.output_dir}/log/model.json', checkpoint,
        img_dim=IMG_DIM, mlp=model_opts.mlp)
    model.to(device)
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    if opts.fp16:
        model = amp.initialize(model, enabled=True, opt_level='O2')

    # load DBs and image dirs
    img_db_type = "gt" if "coco_gt" in opts.img_db else "det"
    conf_th = -1 if img_db_type == "gt" else model_opts.conf_th
    num_bb = 100 if img_db_type == "gt" else model_opts.num_bb
    eval_img_db = DetectFeatLmdb(opts.img_db,
                                 conf_th, model_opts.max_bb,
                                 model_opts.min_bb, num_bb,
                                 opts.compressed_db)

    # Prepro txt_dbs
    txt_dbs = opts.txt_db.split(':')
    for txt_db in txt_dbs:
        print(f'Evaluating {txt_db}')
        eval_txt_db = ReTxtTokLmdb(txt_db, -1)
        eval_dataset = ReEvalDataset(
            eval_txt_db, eval_img_db, use_gt_feat=img_db_type == "gt")

        sampler = DistributedSampler(eval_dataset, num_replicas=n_gpu,
                                     rank=rank, shuffle=False)
        eval_dataloader = DataLoader(eval_dataset,
                                     sampler=sampler,
                                     batch_size=opts.batch_size,
                                     num_workers=opts.n_workers,
                                     pin_memory=opts.pin_mem,
                                     collate_fn=re_eval_collate)
        eval_dataloader = PrefetchLoader(eval_dataloader)

        # evaluate
        val_log, results = evaluate(model, eval_dataloader)

        result_dir = f'{opts.output_dir}/results_test'
        if not exists(result_dir) and rank == 0:
            os.makedirs(result_dir)
        write_to_tmp(
            f"{txt_db.split('_')[1].split('.')[0]}-acc({img_db_type}): {results['acc']*100:.2f}% ",
            args.tmp_file)

        all_results = list(concat(all_gather_list(results)))

        if hvd.rank() == 0:
            db_split = txt_db.split('/')[-1].split('.')[0]  # refcoco+_val
            img_dir = opts.img_db.split('/')[-1]  # re_coco_gt
            with open(f'{result_dir}/'
                    f'results_{opts.checkpoint}_{db_split}_on_{img_dir}_all.json', 'w') as f:
                json.dump(all_results, f)
        # print
        print(f'{opts.output_dir}/results_test')

    write_to_tmp(f'\n', args.tmp_file)


@torch.no_grad()
def evaluate(model, eval_loader):
    LOGGER.info("start running evaluation...")
    model.eval()
    tot_score = 0
    n_ex = 0
    st = time()
    predictions = []
    for i, batch in enumerate(eval_loader):
        (tgt_box_list, obj_boxes_list, sent_ids) = (
            batch['tgt_box'], batch['obj_boxes'], batch['sent_ids'])
        # scores (n, max_num_bb)
        scores = model(batch, compute_loss=False)
        ixs = torch.argmax(scores, 1).cpu().detach().numpy()  # (n, )

        # pred_boxes
        for ix, obj_boxes, tgt_box, sent_id in \
                zip(ixs, obj_boxes_list, tgt_box_list, sent_ids):
            pred_box = obj_boxes[ix]
            predictions.append({'sent_id': int(sent_id),
                                'pred_box': pred_box.tolist(),
                                'tgt_box': tgt_box.tolist()})
            if eval_loader.loader.dataset.computeIoU(pred_box, tgt_box) > .5:
                tot_score += 1
            n_ex += 1
        if i % 100 == 0 and hvd.rank() == 0:
            n_results = len(predictions)
            n_results *= hvd.size()   # an approximation to avoid hangs
            LOGGER.info(f'{n_results}/{len(eval_loader.dataset)} '
                        'answers predicted')
    n_ex = sum(all_gather_list(n_ex))
    tot_time = time()-st
    tot_score = sum(all_gather_list(tot_score))
    val_acc = tot_score / n_ex
    val_log = {'valid/acc': val_acc, 'valid/ex_per_s': n_ex/tot_time}
    model.train()
    LOGGER.info(f"validation ({n_ex} sents) finished in"
                f" {int(tot_time)} seconds"
                f", accuracy: {val_acc*100:.2f}%")
    # summarizae
    results = {'acc': val_acc, 'predictions': predictions}
    return val_log, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--txt_db",
                        default=None, type=str,
                        help="The input train corpus. (LMDB)")
    parser.add_argument("--img_db",
                        default=None, type=str,
                        help="The input train images.")
    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')
    parser.add_argument("--checkpoint",
                        default=None, type=str,
                        help="can be the path to binary or int number (step)")
    parser.add_argument("--batch_size",
                        default=256, type=int,
                        help="number of sentences per batch")

    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory of the training command")

    # device parameters
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true',
                        help="pin memory")

    # Write simple results to some tmp file
    parser.add_argument('--tmp_file', type=str, default=None,
                        help="write results to tmp file")

    args = parser.parse_args()

    main(args)
