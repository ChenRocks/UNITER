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
import numpy as np
from cytoolz import concat

from data import (TokenBucketSampler, PrefetchLoader,
                  DetectFeatLmdb, TxtTokLmdb, VqaEvalDataset, vqa_eval_collate)
from model.vqa import UniterForVisualQuestionAnswering

from utils.logger import LOGGER
from utils.distributed import all_gather_list
from utils.misc import Struct
from utils.const import BUCKET_SIZE, IMG_DIM


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
    model_opts = Struct(json.load(open(hps_file)))

    # train_examples = None
    ans2label_file = f'{opts.output_dir}/ckpt/ans2label.json'
    ans2label = json.load(open(ans2label_file))
    label2ans = {label: ans for ans, label in ans2label.items()}

    # load DBs and image dirs
    eval_img_db = DetectFeatLmdb(opts.img_db,
                                 model_opts.conf_th, model_opts.max_bb,
                                 model_opts.min_bb, model_opts.num_bb,
                                 opts.compressed_db)
    eval_txt_db = TxtTokLmdb(opts.txt_db, -1)
    eval_dataset = VqaEvalDataset(len(ans2label), eval_txt_db, eval_img_db)

    # Prepare model
    if exists(opts.checkpoint):
        ckpt_file = opts.checkpoint
    else:
        ckpt_file = f'{opts.output_dir}/ckpt/model_step_{opts.checkpoint}.pt'
    checkpoint = torch.load(ckpt_file)
    model = UniterForVisualQuestionAnswering.from_pretrained(
        f'{opts.output_dir}/log/model.json', checkpoint,
        img_dim=IMG_DIM, num_answer=len(ans2label))
    model.to(device)
    if opts.fp16:
        model = amp.initialize(model, enabled=True, opt_level='O2')

    sampler = TokenBucketSampler(eval_dataset.lens, bucket_size=BUCKET_SIZE,
                                 batch_size=opts.batch_size, droplast=False)
    eval_dataloader = DataLoader(eval_dataset,
                                 batch_sampler=sampler,
                                 num_workers=opts.n_workers,
                                 pin_memory=opts.pin_mem,
                                 collate_fn=vqa_eval_collate)
    eval_dataloader = PrefetchLoader(eval_dataloader)

    val_log, results, logits = evaluate(model, eval_dataloader, label2ans,
                                        opts.save_logits)
    result_dir = f'{opts.output_dir}/results_test'
    if not exists(result_dir) and rank == 0:
        os.makedirs(result_dir)

    all_results = list(concat(all_gather_list(results)))
    if opts.save_logits:
        all_logits = {}
        for id2logit in all_gather_list(logits):
            all_logits.update(id2logit)
    if hvd.rank() == 0:
        with open(f'{result_dir}/'
                  f'results_{opts.checkpoint}_all.json', 'w') as f:
            json.dump(all_results, f)
        if opts.save_logits:
            np.savez(f'{result_dir}/logits_{opts.checkpoint}_all.npz',
                     **all_logits)


@torch.no_grad()
def evaluate(model, eval_loader, label2ans, save_logits=False):
    LOGGER.info("start running evaluation...")
    model.eval()
    n_ex = 0
    st = time()
    results = []
    logits = {}
    for i, batch in enumerate(eval_loader):
        qids = batch['qids']
        scores = model(batch, compute_loss=False)
        answers = [label2ans[i]
                   for i in scores.max(dim=-1, keepdim=False
                                       )[1].cpu().tolist()]
        for qid, answer in zip(qids, answers):
            results.append({'answer': answer, 'question_id': int(qid)})
        if save_logits:
            scores = scores.cpu()
            for i, qid in enumerate(qids):
                logits[qid] = scores[i].half().numpy()
        if i % 100 == 0 and hvd.rank() == 0:
            n_results = len(results)
            n_results *= hvd.size()   # an approximation to avoid hangs
            LOGGER.info(f'{n_results}/{len(eval_loader.dataset)} '
                        'answers predicted')
        n_ex += len(qids)
    n_ex = sum(all_gather_list(n_ex))
    tot_time = time()-st
    val_log = {'valid/ex_per_s': n_ex/tot_time}
    model.train()
    LOGGER.info(f"evaluation finished in {int(tot_time)} seconds "
                f"at {int(n_ex/tot_time)} examples per second")
    return val_log, results, logits


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1]  # argmax
    one_hots = torch.zeros(*labels.size(), device=labels.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


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
                        default=8192, type=int,
                        help="number of tokens in a batch")

    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory of the training command")

    parser.add_argument("--save_logits", action='store_true',
                        help="Whether to save logits (for making ensemble)")

    # Prepro parameters

    # device parameters
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true',
                        help="pin memory")

    args = parser.parse_args()

    main(args)
