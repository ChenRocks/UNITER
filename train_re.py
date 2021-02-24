"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

UNITER finetuning for RE
"""
import argparse
import json
import os
from os.path import exists, join
from time import time

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.optim import Adam, Adamax

from apex import amp
from horovod import torch as hvd

from tqdm import tqdm

from data import (PrefetchLoader, DetectFeatLmdb,
                  ReTxtTokLmdb, ReDataset, ReEvalDataset,
                  re_collate, re_eval_collate)
from data.sampler import DistributedSampler
from model.re import UniterForReferringExpressionComprehension
from optim import AdamW, get_lr_sched

from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.distributed import (
    all_gather_list, all_reduce_and_rescale_tensors,
    broadcast_tensors)
from utils.save import ModelSaver, save_training_meta
from utils.misc import (
    NoOp, parse_with_config, set_dropout, set_random_seed)
from utils.const import IMG_DIM


def create_dataloader(img_path, txt_path, batch_size, is_train,
                      dset_cls, collate_fn, opts):
    img_db_type = "gt" if "coco_gt" in img_path else "det"
    conf_th = -1 if img_db_type == "gt" else opts.conf_th
    num_bb = 100 if img_db_type == "gt" else opts.num_bb
    img_db = DetectFeatLmdb(img_path, conf_th, opts.max_bb, opts.min_bb,
                            num_bb, opts.compressed_db)
    txt_db = ReTxtTokLmdb(txt_path, opts.max_txt_len if is_train else -1)
    if is_train:
        dset = dset_cls(txt_db, img_db)
    else:
        dset = dset_cls(txt_db, img_db, use_gt_feat=img_db_type == "gt")
    batch_size = (opts.train_batch_size if is_train
                  else opts.val_batch_size)
    sampler = DistributedSampler(dset, num_replicas=hvd.size(),
                                 rank=hvd.rank(), shuffle=False)
    dataloader = DataLoader(dset, sampler=sampler,
                            batch_size=batch_size,
                            num_workers=opts.n_workers,
                            pin_memory=opts.pin_mem, collate_fn=collate_fn)
    dataloader = PrefetchLoader(dataloader)
    return dataloader


def build_optimizer(model, opts):
    """ Re linear may get larger learning rate """
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    param_optimizer = [(n, p) for n, p in model.named_parameters()
                       if 're_output' not in n]
    param_top = [(n, p) for n, p in model.named_parameters()
                 if 're_output' in n]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_top
                    if not any(nd in n for nd in no_decay)],
         'lr': opts.learning_rate,
         'weight_decay': opts.weight_decay},
        {'params': [p for n, p in param_top
                    if any(nd in n for nd in no_decay)],
         'lr': opts.learning_rate,
         'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': opts.weight_decay},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # currently Adam only
    if opts.optim == 'adam':
        OptimCls = Adam
    elif opts.optim == 'adamax':
        OptimCls = Adamax
    elif opts.optim == 'adamw':
        OptimCls = AdamW
    else:
        raise ValueError('invalid optimizer')
    optimizer = OptimCls(optimizer_grouped_parameters,
                         lr=opts.learning_rate, betas=opts.betas)
    return optimizer


def main(opts):
    hvd.init()
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    rank = hvd.rank()
    opts.rank = rank
    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, hvd.rank(), opts.fp16))

    if opts.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, "
                         "should be >= 1".format(
                            opts.gradient_accumulation_steps))

    set_random_seed(opts.seed)

    # train_examples = None
    LOGGER.info(f"Loading Train Dataset {opts.train_txt_db}, "
                f"{opts.train_img_db}")
    train_dataloader = create_dataloader(opts.train_img_db, opts.train_txt_db,
                                         opts.train_batch_size, True,
                                         ReDataset, re_collate, opts)
    val_dataloader = create_dataloader(opts.val_img_db, opts.val_txt_db,
                                       opts.val_batch_size, False,
                                       ReEvalDataset, re_eval_collate, opts)

    # Prepare model
    if opts.checkpoint:
        checkpoint = torch.load(opts.checkpoint)
    else:
        checkpoint = {}

    all_dbs = [opts.train_txt_db, opts.val_txt_db]
    toker = json.load(open(f'{all_dbs[0]}/meta.json'))['toker']
    assert all(toker == json.load(open(f'{db}/meta.json'))['toker']
               for db in all_dbs)
    model = UniterForReferringExpressionComprehension.from_pretrained(
        opts.model_config, checkpoint,
        img_dim=IMG_DIM, loss=opts.train_loss,
        margin=opts.margin,
        hard_ratio=opts.hard_ratio, mlp=opts.mlp,)
    model.to(device)
    model.train()
    # make sure every process has same model parameters in the beginning
    broadcast_tensors([p.data for p in model.parameters()], 0)
    set_dropout(model, opts.dropout)

    optimizer = build_optimizer(model, opts)

    #  Apex
    model, optimizer = amp.initialize(
        model, optimizer, enabled=opts.fp16, opt_level='O2')

    global_step = 0
    if rank == 0:
        save_training_meta(opts)
        TB_LOGGER.create(join(opts.output_dir, 'log'))
        pbar = tqdm(total=opts.num_train_steps)
        model_saver = ModelSaver(join(opts.output_dir, 'ckpt'), 'model_epoch')
        os.makedirs(join(opts.output_dir, 'results'))  # store RE predictions
        add_log_to_file(join(opts.output_dir, 'log', 'log.txt'))
    else:
        LOGGER.disabled = True
        pbar = NoOp()
        model_saver = NoOp()

    LOGGER.info(f"***** Running training with {n_gpu} GPUs *****")
    LOGGER.info("  Num examples = %d", len(train_dataloader.dataset))
    LOGGER.info("  Batch size = %d", opts.train_batch_size)
    LOGGER.info("  Accumulate steps = %d", opts.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", opts.num_train_steps)

    running_loss = RunningMeter('loss')
    model.train()
    n_examples = 0
    n_epoch = 0
    best_val_acc, best_epoch = None, None
    start = time()
    # quick hack for amp delay_unscale bug
    optimizer.zero_grad()
    if global_step == 0:
        optimizer.step()

    while True:
        for step, batch in enumerate(train_dataloader):
            if global_step >= opts.num_train_steps:
                break

            n_examples += batch['input_ids'].size(0)

            loss = model(batch, compute_loss=True)
            loss = loss.sum()  # sum over vectorized loss TODO: investigate
            delay_unscale = (step+1) % opts.gradient_accumulation_steps != 0
            with amp.scale_loss(
                    loss, optimizer, delay_unscale=delay_unscale
                    ) as scaled_loss:
                scaled_loss.backward()
                if not delay_unscale:
                    # gather gradients from every processes
                    # do this before unscaling to make sure every process uses
                    # the same gradient scale
                    grads = [p.grad.data for p in model.parameters()
                             if p.requires_grad and p.grad is not None]
                    all_reduce_and_rescale_tensors(grads, float(1))

            running_loss(loss.item())

            if (step + 1) % opts.gradient_accumulation_steps == 0:
                global_step += 1

                # learning rate scheduling
                lr_this_step = get_lr_sched(global_step, opts)
                for i, param_group in enumerate(optimizer.param_groups):
                    if i == 0 or i == 1:
                        param_group['lr'] = lr_this_step * opts.lr_mul
                    elif i == 2 or i == 3:
                        param_group['lr'] = lr_this_step
                    else:
                        raise ValueError()
                TB_LOGGER.add_scalar('lr', lr_this_step, global_step)

                # log loss
                # NOTE: not gathered across GPUs for efficiency
                TB_LOGGER.add_scalar('loss', running_loss.val, global_step)
                TB_LOGGER.step()

                # update model params
                if opts.grad_norm != -1:
                    grad_norm = clip_grad_norm_(amp.master_params(optimizer),
                                                opts.grad_norm)
                    TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
                optimizer.step()
                optimizer.zero_grad()
                pbar.update(1)

                if global_step % 100 == 0:
                    # monitor training throughput
                    LOGGER.info(f'============Step {global_step}=============')
                    tot_ex = sum(all_gather_list(n_examples))
                    ex_per_sec = int(tot_ex / (time()-start))
                    LOGGER.info(f'{tot_ex} examples trained at '
                                f'{ex_per_sec} ex/s')
                    TB_LOGGER.add_scalar('perf/ex_per_s',
                                         ex_per_sec, global_step)
                    LOGGER.info('===========================================')

        # evaluate after each epoch
        val_log, _ = validate(model, val_dataloader)
        TB_LOGGER.log_scaler_dict(val_log)

        # save model
        n_epoch += 1
        model_saver.save(model, n_epoch)
        LOGGER.info(f"finished {n_epoch} epochs")

        # save best model
        if best_val_acc is None or val_log['valid/acc'] > best_val_acc:
            best_val_acc = val_log['valid/acc']
            best_epoch = n_epoch
            model_saver.save(model, 'best')

        # shuffle training data for the next epoch
        train_dataloader.loader.dataset.shuffle()

        # is training finished?
        if global_step >= opts.num_train_steps:
            break

    val_log, results = validate(model, val_dataloader)
    with open(f'{opts.output_dir}/results/'
              f'results_{global_step}_'
              f'rank{rank}_final.json', 'w') as f:
        json.dump(results, f)
    TB_LOGGER.log_scaler_dict(val_log)
    model_saver.save(model, f'{global_step}_final')

    # print best model
    LOGGER.info(
        f'best_val_acc = {best_val_acc*100:.2f}% at epoch {best_epoch}.')


@torch.no_grad()
def validate(model, val_dataloader):
    LOGGER.info("start running evaluation.")
    model.eval()
    tot_score = 0
    n_ex = 0
    st = time()
    predictions = {}
    for i, batch in enumerate(val_dataloader):
        # inputs
        (tgt_box_list, obj_boxes_list, sent_ids) = (
            batch['tgt_box'], batch['obj_boxes'], batch['sent_ids'])
        # scores (n, max_num_bb)
        scores = model(batch, compute_loss=False)
        ixs = torch.argmax(scores, 1).cpu().detach().numpy()  # (n, )

        # pred_boxes
        for ix, obj_boxes, tgt_box, sent_id in \
                zip(ixs, obj_boxes_list, tgt_box_list, sent_ids):
            pred_box = obj_boxes[ix]
            predictions[int(sent_id)] = {
                'pred_box': pred_box.tolist(),
                'tgt_box': tgt_box.tolist()}
            if val_dataloader.loader.dataset.computeIoU(
                    pred_box, tgt_box) > .5:
                tot_score += 1
            n_ex += 1

    tot_time = time()-st
    tot_score = sum(all_gather_list(tot_score))
    n_ex = sum(all_gather_list(n_ex))
    val_acc = tot_score / n_ex
    val_log = {'valid/acc': val_acc, 'valid/ex_per_s': n_ex/tot_time}
    model.train()
    LOGGER.info(
        f"validation ({n_ex} sents) finished in {int(tot_time)} seconds"
        f", accuracy: {val_acc*100:.2f}%")
    return val_log, predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--train_txt_db",
                        default=None, type=str,
                        help="The input train corpus. (LMDB)")
    parser.add_argument("--train_img_db",
                        default=None, type=str,
                        help="The input train images.")
    parser.add_argument("--val_txt_db",
                        default=None, type=str,
                        help="The input validation corpus. (LMDB)")
    parser.add_argument("--val_img_db",
                        default=None, type=str,
                        help="The input validation images.")
    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')
    parser.add_argument("--model_config",
                        default=None, type=str,
                        help="json file for model architecture")
    parser.add_argument("--checkpoint",
                        default=None, type=str,
                        help="pretrained model (can take 'google-bert') ")
    parser.add_argument("--mlp", default=1, type=int,
                        help="number of MLP layers for RE output")

    parser.add_argument(
        "--output_dir", default=None, type=str,
        help="The output directory where the model checkpoints will be "
             "written.")

    # Prepro parameters
    parser.add_argument('--max_txt_len', type=int, default=60,
                        help='max number of tokens in text (BERT BPE)')
    parser.add_argument('--conf_th', type=float, default=0.2,
                        help='threshold for dynamic bounding boxes '
                             '(-1 for fixed)')
    parser.add_argument('--max_bb', type=int, default=100,
                        help='max number of bounding boxes')
    parser.add_argument('--min_bb', type=int, default=10,
                        help='min number of bounding boxes')
    parser.add_argument('--num_bb', type=int, default=36,
                        help='static number of bounding boxes')

    # training parameters
    parser.add_argument("--train_batch_size",
                        default=128, type=int,
                        help="Total batch size for training. "
                             "(batch by examples)")
    parser.add_argument("--val_batch_size",
                        default=256, type=int,
                        help="Total batch size for validation. "
                             "(batch by examples)")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=16,
                        help="Number of updates steps to accumualte before "
                             "performing a backward/update pass.")
    parser.add_argument("--train_loss",
                        default="cls", type=str,
                        choices=['cls', 'rank'],
                        help="loss to used during training")
    parser.add_argument("--margin",
                        default=0.2, type=float,
                        help="margin of ranking loss")
    parser.add_argument("--hard_ratio",
                        default=0.3, type=float,
                        help="sampling ratio of hard negatives")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_steps",
                        default=32000,
                        type=int,
                        help="Total number of training updates to perform.")
    parser.add_argument("--optim", default='adam',
                        choices=['adam', 'adamax', 'adamw'],
                        help="optimizer")
    parser.add_argument("--betas", default=[0.9, 0.98], nargs='+', type=float,
                        help="beta for adam optimizer")
    parser.add_argument("--decay", default='linear',
                        choices=['linear', 'invsqrt', 'constant'],
                        help="learning rate decay method")
    parser.add_argument("--dropout",
                        default=0.1,
                        type=float,
                        help="tune dropout regularization")
    parser.add_argument("--weight_decay",
                        default=0.0,
                        type=float,
                        help="weight decay (L2) regularization")
    parser.add_argument("--grad_norm",
                        default=0.25,
                        type=float,
                        help="gradient clipping (-1 for no clipping)")
    parser.add_argument("--warmup_steps",
                        default=4000,
                        type=int,
                        help="Number of training steps to perform linear "
                             "learning rate warmup for. (invsqrt decay)")

    # device parameters
    parser.add_argument('--seed',
                        type=int,
                        default=24,
                        help="random seed for initialization")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true',
                        help="pin memory")

    # can use config files
    parser.add_argument('--config', help='JSON config files')

    args = parse_with_config(parser)

    if exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not "
                         "empty.".format(args.output_dir))

    if args.conf_th == -1:
        assert args.max_bb + args.max_txt_len + 2 <= 512
    else:
        assert args.num_bb + args.max_txt_len + 2 <= 512

    # options safe guard
    main(args)
