"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

run inference of VCR for submission
"""
import argparse
import json
import os
from os.path import exists
import pandas as pd
from time import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from apex import amp
from horovod import torch as hvd
import numpy as np
from torch.utils.data.distributed import DistributedSampler
from data import (PrefetchLoader,
                  DetectFeatLmdb, VcrTxtTokLmdb, VcrEvalDataset,
                  vcr_eval_collate)
from model.vcr import UniterForVisualCommonsenseReasoning
from utils.logger import LOGGER
from utils.distributed import all_gather_list
from utils.misc import NoOp, Struct
from utils.const import IMG_DIM
from tqdm import tqdm
NUM_SPECIAL_TOKENS = 81


def load_img_feat(dir_list, opts):
    dir_ = dir_list.split(";")
    assert len(dir_) <= 2, "More than two img_dirs found"
    img_db_gt, img_db = None, None
    gt_db_path, db_path = "", ""
    for d in dir_:
        if "gt" in d:
            gt_db_path = d
        else:
            db_path = d
    if gt_db_path != "":
        img_db_gt = DetectFeatLmdb(
            gt_db_path, -1, opts.max_bb, opts.min_bb, 100,
            opts.compressed_db)
    if db_path != "":
        img_db = DetectFeatLmdb(
            db_path, opts.conf_th,
            opts.max_bb, opts.min_bb, opts.num_bb,
            opts.compressed_db)
    return img_db, img_db_gt


def save_for_submission(pred_file):
    with open(os.path.join(pred_file), "r") as f:
        data = json.load(f)
        probs_grp = []
        ids_grp = []
        ordered_data = sorted(data.items(),
                              key=lambda item: int(item[0].split("-")[1]))
        for annot_id, scores in ordered_data:
            ids_grp.append(annot_id)
            probs_grp.append(np.array(scores).reshape(1, 5, 4))

    # Double check the IDs are in the same order for everything
    # assert [x == ids_grp[0] for x in ids_grp]

    probs_grp = np.stack(probs_grp, 1)
    # essentially probs_grp is a [num_ex, 5, 4] array of probabilities.
    # The 5 'groups' are
    # [answer, rationale_conditioned_on_a0, rationale_conditioned_on_a1,
    #          rationale_conditioned_on_a2, rationale_conditioned_on_a3].
    # We will flatten this to a CSV file so it's easy to submit.
    group_names = ['answer'] + [f'rationale_conditioned_on_a{i}' 
                                for i in range(4)]
    probs_df = pd.DataFrame(data=probs_grp.reshape((-1, 20)),
                            columns=[f'{group_name}_{i}'
                            for group_name in group_names for i in range(4)])
    probs_df['annot_id'] = ids_grp
    probs_df = probs_df.set_index('annot_id', drop=True)
    return probs_df


def main(opts):
    hvd.init()
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    rank = hvd.rank()
    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, hvd.rank(), opts.fp16))
    if rank != 0:
        LOGGER.disabled = True

    hps_file = f'{opts.output_dir}/log/hps.json'
    model_opts = Struct(json.load(open(hps_file)))

    assert opts.split in opts.img_db and opts.split in opts.txt_db
    # load DBs and image dirs
    eval_img_db, eval_img_db_gt = load_img_feat(opts.img_db, model_opts)
    eval_txt_db = VcrTxtTokLmdb(opts.txt_db, -1)
    eval_dataset = VcrEvalDataset(
        "test", eval_txt_db, img_db=eval_img_db,
        img_db_gt=eval_img_db_gt)

    # Prepare model
    model = UniterForVisualCommonsenseReasoning.from_pretrained(
        f'{opts.output_dir}/log/model.json', state_dict={},
        img_dim=IMG_DIM)
    model.init_type_embedding()
    model.init_word_embedding(NUM_SPECIAL_TOKENS)
    if exists(opts.checkpoint):
        ckpt_file = opts.checkpoint
    else:
        ckpt_file = f'{opts.output_dir}/ckpt/model_step_{opts.checkpoint}.pt'
    checkpoint = torch.load(ckpt_file)
    state_dict = checkpoint.get('model_state', checkpoint)
    matched_state_dict = {}
    unexpected_keys = set()
    missing_keys = set()
    for name, param in model.named_parameters():
        missing_keys.add(name)
    for key, data in state_dict.items():
        if key in missing_keys:
            matched_state_dict[key] = data
            missing_keys.remove(key)
        else:
            unexpected_keys.add(key)
    LOGGER.info(f"Unexpected_keys: {list(unexpected_keys)}")
    LOGGER.info(f"Missing_keys: {list(missing_keys)}")
    model.load_state_dict(matched_state_dict, strict=False)
    model.to(device)
    if opts.fp16:
        model = amp.initialize(model, enabled=True, opt_level='O2')

    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size=opts.batch_size,
                                 num_workers=opts.n_workers,
                                 pin_memory=opts.pin_mem,
                                 shuffle=False,
                                 collate_fn=vcr_eval_collate)
    eval_dataloader = PrefetchLoader(eval_dataloader)

    _, results = evaluate(model, eval_dataloader)
    result_dir = f'{opts.output_dir}/results_{opts.split}'
    if not exists(result_dir) and rank == 0:
        os.makedirs(result_dir)

    all_results = {}
    for id2res in all_gather_list(results):
        all_results.update(id2res)
    if hvd.rank() == 0:
        with open(f'{result_dir}/'
                  f'results_{opts.checkpoint}_all.json', 'w') as f:
            json.dump(all_results, f)
        probs_df = save_for_submission(
            f'{result_dir}/results_{opts.checkpoint}_all.json')
        probs_df.to_csv(f'{result_dir}/results_{opts.checkpoint}_all.csv')


@torch.no_grad()
def evaluate(model, eval_loader):
    model.eval()
    LOGGER.info("start running evaluation ...")
    if hvd.rank() == 0:
        val_pbar = tqdm(total=len(eval_loader))
    else:
        val_pbar = NoOp()
    val_qa_loss, val_qar_loss = 0, 0
    tot_qa_score, tot_qar_score, tot_score = 0, 0, 0
    n_ex = 0
    st = time()
    results = {}
    for i, batch in enumerate(eval_loader):
        qids = batch['qids']
        qa_targets, qar_targets = batch['qa_targets'], batch['qar_targets']
        scores = model(batch, compute_loss=False)
        scores = scores.view(len(qids), -1)
        if torch.max(qa_targets) > -1:
            vcr_qa_loss = F.cross_entropy(
                scores[:, :4], qa_targets.squeeze(-1), reduction="sum")
            if scores.shape[1] > 8:
                qar_scores = []
                for batch_id in range(scores.shape[0]):
                    answer_ind = qa_targets[batch_id].item()
                    qar_index = [4+answer_ind*4+i
                                 for i in range(4)]
                    qar_scores.append(scores[batch_id, qar_index])
                qar_scores = torch.stack(qar_scores, dim=0)
            else:
                qar_scores = scores[:, 4:]
            vcr_qar_loss = F.cross_entropy(
                qar_scores, qar_targets.squeeze(-1), reduction="sum")
            val_qa_loss += vcr_qa_loss.item()
            val_qar_loss += vcr_qar_loss.item()

            curr_qa_score, curr_qar_score, curr_score = compute_accuracies(
                scores[:, :4], qa_targets, qar_scores, qar_targets)
            tot_qar_score += curr_qar_score
            tot_qa_score += curr_qa_score
            tot_score += curr_score
        for qid, score in zip(qids, scores):
            results[qid] = score.cpu().tolist()
        n_ex += len(qids)
        val_pbar.update(1)
    val_qa_loss = sum(all_gather_list(val_qa_loss))
    val_qar_loss = sum(all_gather_list(val_qar_loss))
    tot_qa_score = sum(all_gather_list(tot_qa_score))
    tot_qar_score = sum(all_gather_list(tot_qar_score))
    tot_score = sum(all_gather_list(tot_score))
    n_ex = sum(all_gather_list(n_ex))
    tot_time = time()-st
    val_qa_loss /= n_ex
    val_qar_loss /= n_ex
    val_qa_acc = tot_qa_score / n_ex
    val_qar_acc = tot_qar_score / n_ex
    val_acc = tot_score / n_ex
    val_log = {'valid/ex_per_s': n_ex/tot_time,
               'valid/vcr_qa_loss': val_qa_loss,
               'valid/vcr_qar_loss': val_qar_loss,
               'valid/acc_qa': val_qa_acc,
               'valid/acc_qar': val_qar_acc,
               'valid/acc': val_acc}
    model.train()
    LOGGER.info(f"evaluation finished in {int(tot_time)} seconds, "
                f"score_qa: {val_qa_acc*100:.2f} "
                f"score_qar: {val_qar_acc*100:.2f} "
                f"score: {val_acc*100:.2f} ")
    return val_log, results


def compute_accuracies(out_qa, labels_qa, out_qar, labels_qar):
    outputs_qa = out_qa.max(dim=-1)[1]
    outputs_qar = out_qar.max(dim=-1)[1]
    matched_qa = outputs_qa.squeeze() == labels_qa.squeeze()
    matched_qar = outputs_qar.squeeze() == labels_qar.squeeze()
    matched_joined = matched_qa & matched_qar
    n_correct_qa = matched_qa.sum().item()
    n_correct_qar = matched_qar.sum().item()
    n_correct_joined = matched_joined.sum().item()
    return n_correct_qa, n_correct_qar, n_correct_joined


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--txt_db",
                        default="/txt/vcr_val.db/", type=str,
                        help="The input train corpus. (LMDB)")
    parser.add_argument("--img_db",
                        default="/img/vcr_gt_val/;/img/vcr_val/", type=str,
                        help="The input train images.")
    parser.add_argument("--split",
                        default="val", type=str,
                        help="The input split")
    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')
    parser.add_argument("--checkpoint",
                        default=None, type=str,
                        help="can be the path to binary or int number (step)")
    parser.add_argument("--batch_size",
                        default=10, type=int,
                        help="number of examples in a batch")

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

    args = parser.parse_args()

    main(args)
