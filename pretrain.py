"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

UNITER pre-training
"""
import argparse
from collections import defaultdict
import json
import math
import os
from os.path import exists, join
from time import time

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from apex import amp
from horovod import torch as hvd

from tqdm import tqdm

from data import (TokenBucketSampler, TokenBucketSamplerForItm,
                  MetaLoader, PrefetchLoader,
                  TxtTokLmdb, ImageLmdbGroup, ConcatDatasetWithLens,
                  MlmDataset, MrfrDataset, MrcDataset,
                  mlm_collate, mrfr_collate, mrc_collate,
                  ItmDataset, itm_collate, itm_ot_collate)

from model.pretrain import UniterForPretraining
from optim import get_lr_sched
from optim.misc import build_optimizer

from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.distributed import (all_reduce_and_rescale_tensors, all_gather_list,
                               broadcast_tensors)
from utils.save import ModelSaver, save_training_meta
from utils.misc import NoOp, parse_with_config, set_dropout, set_random_seed
from utils.const import IMG_DIM, IMG_LABEL_DIM, BUCKET_SIZE


def build_dataloader(dataset, collate_fn, is_train, opts):
    if is_train:
        batch_size = opts.train_batch_size
    else:
        batch_size = opts.val_batch_size
    sampler = TokenBucketSampler(dataset.lens, bucket_size=BUCKET_SIZE,
                                 batch_size=batch_size, droplast=is_train)
    loader = DataLoader(dataset, batch_sampler=sampler,
                        num_workers=opts.n_workers, pin_memory=opts.pin_mem,
                        collate_fn=collate_fn)
    return loader


def build_dataloader_itm(dataset, collate_fn, is_train, opts):
    if is_train:
        batch_size = opts.train_batch_size
    else:
        batch_size = opts.val_batch_size
    sampler = TokenBucketSamplerForItm(
        dataset, bucket_size=BUCKET_SIZE,
        batch_size=batch_size, droplast=is_train)
    loader = DataLoader(dataset, batch_sampler=sampler,
                        num_workers=opts.n_workers, pin_memory=opts.pin_mem,
                        collate_fn=collate_fn)
    return loader


def build_mlm_dataset(txt_db, img_db, is_train, opts):
    if is_train:
        collate_fn = mlm_collate
        datasets = [MlmDataset(t, i) for t, i in zip(txt_db, img_db)]
        dataset = ConcatDatasetWithLens(datasets)
    else:
        collate_fn = mlm_collate
        dataset = MlmDataset(txt_db, img_db)

    return dataset, collate_fn


def build_mrfr_dataset(txt_db, img_db, is_train, opts):
    if is_train:
        datasets = [MrfrDataset(opts.mrm_prob, t, i)
                    for t, i in zip(txt_db, img_db)]
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = MrfrDataset(opts.mrm_prob, txt_db, img_db)

    return dataset, mrfr_collate


def build_mrc_dataset(txt_db, img_db, is_train, opts):
    if is_train:
        datasets = [MrcDataset(opts.mrm_prob, t, i)
                    for t, i in zip(txt_db, img_db)]
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = MrcDataset(opts.mrm_prob, txt_db, img_db)

    return dataset, mrc_collate


def build_itm_dataset(txt_db, img_db, is_train, opts):
    if is_train:
        datasets = [ItmDataset(t, i, opts.itm_neg_prob)
                    for t, i in zip(txt_db, img_db)]
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = ItmDataset(txt_db, img_db, opts.itm_neg_prob)
    collate_fn = itm_ot_collate if opts.itm_ot_lambda > 0 else itm_collate
    return dataset, collate_fn


def create_dataloaders(datasets, is_train, opts, all_img_dbs=None):
    if all_img_dbs is None:
        all_img_dbs = ImageLmdbGroup(opts.conf_th, opts.max_bb, opts.min_bb,
                                     opts.num_bb, opts.compressed_db)
    dataloaders = {}
    for dset in datasets:
        if is_train:
            assert len(dset['db']) == len(dset['img'])
            assert len(dset['tasks']) == len(dset['mix_ratio'])
            img_db = [all_img_dbs[path] for path in dset['img']]
        else:
            assert len(dset['db']) == len(dset['img']) == 1
            img_db = all_img_dbs[dset['img'][0]]

        for i, t in enumerate(dset['tasks']):
            task = f'{t}_{dset["name"]}'

            if is_train:
                LOGGER.info(f"Loading {task} train dataset "
                            f"{dset['db']}, {[img.img_dir for img in img_db]}")
                txt_db = [TxtTokLmdb(path, opts.max_txt_len)
                          for path in dset['db']]
            else:
                LOGGER.info(f"Loading {task} validation dataset, "
                            f"{dset['db']}, {img_db.img_dir}")
                txt_db = TxtTokLmdb(dset['db'][0], -1)

            if task.startswith('mlm'):
                dataset = build_mlm_dataset(txt_db, img_db, is_train, opts)
            elif task.startswith('mrfr'):
                dataset = build_mrfr_dataset(txt_db, img_db, is_train, opts)
            elif task.startswith('mrc'):
                dataset = build_mrc_dataset(txt_db, img_db, is_train, opts)
            elif task.startswith('itm'):
                dataset = build_itm_dataset(txt_db, img_db, is_train, opts)
            else:
                raise ValueError(f'Undefined task {task}')

            LOGGER.info(f"{len(dataset[0])*hvd.size()} samples loaded")
            if task.startswith('itm'):
                # itm handles distributed training in dset not sampler
                loader = build_dataloader_itm(*dataset, is_train, opts)
            else:
                loader = build_dataloader(*dataset, is_train, opts)
            if is_train:
                ratio = dset['mix_ratio'][i]
                dataloaders[task] = (loader, ratio)
            else:
                dataloaders[task] = PrefetchLoader(loader)
    return dataloaders, all_img_dbs


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

    if rank == 0:
        save_training_meta(opts)
        TB_LOGGER.create(join(opts.output_dir, 'log'))
        pbar = tqdm(total=opts.num_train_steps)
        model_saver = ModelSaver(join(args.output_dir, 'ckpt'))
        add_log_to_file(join(opts.output_dir, 'log', 'log.txt'))
    else:
        LOGGER.disabled = True
        pbar = NoOp()
        model_saver = NoOp()

    all_dbs = [db for datasets in [opts.train_datasets, opts.val_datasets]
               for dset in datasets for db in dset['db']]

    tokenizer = json.load(open(f'{all_dbs[0]}/meta.json'))['bert']
    assert all(tokenizer == json.load(open(f'{db}/meta.json'))['bert']
               for db in all_dbs)

    # build data loaders
    train_dataloaders, all_img_dbs = create_dataloaders(
        opts.train_datasets, True, opts)
    val_dataloaders, _ = create_dataloaders(
        opts.val_datasets, False, opts, all_img_dbs)
    meta_loader = MetaLoader(train_dataloaders,
                             accum_steps=opts.gradient_accumulation_steps,
                             distributed=n_gpu > 1)
    meta_loader = PrefetchLoader(meta_loader)

    # Prepare model
    if opts.checkpoint:
        checkpoint = torch.load(opts.checkpoint)
    else:
        checkpoint = {}
    model = UniterForPretraining.from_pretrained(
        opts.model_config, checkpoint,
        img_dim=IMG_DIM, img_label_dim=IMG_LABEL_DIM)
    model.to(device)
    model.train()
    # make sure every process has same model parameters in the beginning
    broadcast_tensors([p.data for p in model.parameters()], 0)
    set_dropout(model, opts.dropout)

    # Prepare optimizer
    optimizer = build_optimizer(model, opts)
    task2scaler = {t: i for i, t in enumerate(train_dataloaders.keys())}
    model, optimizer = amp.initialize(model, optimizer,
                                      num_losses=len(task2scaler),
                                      enabled=opts.fp16, opt_level='O2')

    global_step = 0
    LOGGER.info(f"***** Running training with {n_gpu} GPUs *****")
    LOGGER.info("  Batch size = %d", opts.train_batch_size)
    LOGGER.info("  Accumulate steps = %d", opts.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", opts.num_train_steps)

    # to compute training statistics
    task2loss = {task: RunningMeter(f'loss/{task}')
                 for task in train_dataloaders.keys()}
    # ITM w/ OT
    if opts.itm_ot_lambda > 0:
        for task in train_dataloaders.keys():
            if task.startswith('itm'):
                task2loss[f'{task}_xe'] = RunningMeter(f'loss/{task}_xe')
                task2loss[f'{task}_ot'] = RunningMeter(f'loss/{task}_ot')
                task2loss[f'{task}_ot_pos'] = RunningMeter(
                    f'loss/{task}_ot_pos')
                task2loss[f'{task}_ot_neg'] = RunningMeter(
                    f'loss/{task}_ot_neg')

    n_examples = defaultdict(int)
    n_in_units = defaultdict(int)
    n_loss_units = defaultdict(int)
    grad_norm = 0

    start = time()
    # quick hack for amp delay_unscale bug
    optimizer.zero_grad()
    optimizer.step()
    for step, (name, batch) in enumerate(meta_loader):
        # forward pass
        n_examples[name] += batch['input_ids'].size(0)
        n_in_units[name] += (batch['attn_masks'] == 1).sum().item()
        task = name.split('_')[0]
        loss = model(batch, task=task, compute_loss=True)
        if task.startswith('itm'):
            # OT
            itm_loss, ot_loss = loss
            n_loss_units[name] += itm_loss.size(0)
            itm_loss = itm_loss.mean()
            if ot_loss is not None:
                ot_pos, ot_neg = ot_loss
                ot_loss = (ot_pos.sum() - ot_neg.sum()
                           ) / (ot_pos.size(0) + ot_neg.size(0))

                # NOTE: be ware of empty tensor
                ot_pos = ot_pos.mean().item()
                if not math.isnan(ot_pos):
                    task2loss[f'{name}_ot_pos'](ot_pos)
                ot_neg = ot_neg.mean().item()
                if not math.isnan(ot_neg):
                    task2loss[f'{name}_ot_neg'](ot_neg)

                loss = itm_loss + opts.itm_ot_lambda * ot_loss
                task2loss[f'{name}_xe'](itm_loss.item())
                task2loss[f'{name}_ot'](ot_loss.item())
            else:
                loss = itm_loss
        else:
            n_loss_units[name] += loss.size(0)
            loss = loss.mean()  # loss is not normalized in model

        # backward pass
        delay_unscale = (step+1) % opts.gradient_accumulation_steps != 0
        with amp.scale_loss(loss, optimizer, delay_unscale=delay_unscale,
                            loss_id=task2scaler[name]) as scaled_loss:
            scaled_loss.backward()
            if not delay_unscale:
                # gather gradients from every processes
                # do this before unscaling to make sure every process uses
                # the same gradient scale
                grads = [p.grad.data for p in model.parameters()
                         if p.requires_grad and p.grad is not None]
                all_reduce_and_rescale_tensors(grads, float(1))
        task2loss[name](loss.item())

        # optimizer update and logging
        if (step + 1) % opts.gradient_accumulation_steps == 0:
            global_step += 1

            # learning rate scheduling
            lr_this_step = get_lr_sched(global_step, opts)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            TB_LOGGER.add_scalar('lr', lr_this_step, global_step)

            # log loss
            # NOTE: not gathered across GPUs for efficiency
            TB_LOGGER.log_scaler_dict({l.name: l.val
                                       for l in task2loss.values()
                                       if l.val is not None})
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
                LOGGER.info(f'==============Step {global_step}===============')
                for t in train_dataloaders.keys():
                    assert all(tt == t for tt in all_gather_list(t))
                    tot_ex = sum(all_gather_list(n_examples[t]))
                    ex_per_sec = int(tot_ex / (time()-start))
                    tot_in = sum(all_gather_list(n_in_units[t]))
                    in_per_sec = int(tot_in / (time()-start))
                    tot_l = sum(all_gather_list(n_loss_units[t]))
                    l_per_sec = int(tot_l / (time()-start))
                    LOGGER.info(f'{t}: {tot_ex} examples trained at '
                                f'{ex_per_sec} ex/s')
                    TB_LOGGER.add_scalar(f'perf/{t}_ex_per_s', ex_per_sec,
                                         global_step)
                    TB_LOGGER.add_scalar(f'perf/{t}_in_per_s', in_per_sec,
                                         global_step)
                    TB_LOGGER.add_scalar(f'perf/{t}_loss_per_s', l_per_sec,
                                         global_step)
                LOGGER.info(f'===============================================')

            if global_step % opts.valid_steps == 0:
                LOGGER.info(f'Step {global_step}: start validation')
                validate(model, val_dataloaders)
                model_saver.save(model, global_step)
        if global_step >= opts.num_train_steps:
            break
    if global_step % opts.valid_steps != 0:
        LOGGER.info(f'Step {global_step}: start validation')
        validate(model, val_dataloaders)
        model_saver.save(model, global_step)


def validate(model, val_dataloaders):
    model.eval()
    for task, loader in val_dataloaders.items():
        LOGGER.info(f"validate on {task} task")
        if task.startswith('mlm'):
            val_log = validate_mlm(model, loader)
        elif task.startswith('mrfr'):
            val_log = validate_mrfr(model, loader)
        elif task.startswith('mrc'):
            val_log = validate_mrc(model, loader, task)
        elif task.startswith('itm'):
            val_log = validate_itm(model, loader)
        else:
            raise ValueError(f'Undefined task {task}')
        val_log = {f'{task}_{k}': v for k, v in val_log.items()}
        TB_LOGGER.log_scaler_dict(
            {f'valid_{task}/{k}': v for k, v in val_log.items()})
    model.train()


@torch.no_grad()
def validate_mlm(model, val_loader):
    LOGGER.info("start running MLM validation...")
    val_loss = 0
    n_correct = 0
    n_word = 0
    st = time()
    for i, batch in enumerate(val_loader):
        scores = model(batch, task='mlm', compute_loss=False)
        labels = batch['txt_labels']
        labels = labels[labels != -1]
        loss = F.cross_entropy(scores, labels, reduction='sum')
        val_loss += loss.item()
        n_correct += (scores.max(dim=-1)[1] == labels).sum().item()
        n_word += labels.numel()
    val_loss = sum(all_gather_list(val_loss))
    n_correct = sum(all_gather_list(n_correct))
    n_word = sum(all_gather_list(n_word))
    tot_time = time()-st
    val_loss /= n_word
    acc = n_correct / n_word
    val_log = {'loss': val_loss,
               'acc': acc,
               'tok_per_s': n_word/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"acc: {acc*100:.2f}")
    return val_log


def accuracy_count(out, labels):
    outputs = out.max(dim=-1)[1]
    mask = labels != -1
    n_correct = (outputs == labels).masked_select(mask).sum().item()
    return n_correct


@torch.no_grad()
def validate_mrfr(model, val_loader):
    LOGGER.info("start running MRFR validation...")
    val_loss = 0
    n_feat = 0
    st = time()
    for i, batch in enumerate(val_loader):
        loss = model(batch, task='mrfr', compute_loss=True)
        val_loss += loss.sum().item() / IMG_DIM
        n_feat += batch['img_mask_tgt'].sum().item()
    val_loss = sum(all_gather_list(val_loss))
    n_feat = sum(all_gather_list(n_feat))
    tot_time = time()-st
    val_loss /= n_feat
    val_log = {'loss': val_loss,
               'feat_per_s': n_feat/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"loss: {val_loss:.2f}")
    return val_log


@torch.no_grad()
def validate_mrc(model, val_loader, task):
    LOGGER.info("start running MRC validation...")
    val_loss = 0
    n_feat = 0
    st = time()
    tot_score = 0
    for i, batch in enumerate(val_loader):
        prediction_soft_label = model(
            batch, task=task, compute_loss=False)
        if "kl" in task:
            prediction_soft_label = F.log_softmax(
                prediction_soft_label, dim=-1)
            label_targets = batch['label_targets']
            loss = F.kl_div(
                prediction_soft_label, label_targets, reduction='sum')
            tot_score += compute_accuracy_for_soft_targets(
                prediction_soft_label, label_targets)
        else:
            # background class should not be the target
            cls_label_targets = label_targets[:, 1:].max(dim=-1)[1] + 1
            loss = F.cross_entropy(
                prediction_soft_label, cls_label_targets,
                ignore_index=0, reduction='sum')
            tot_score += compute_accuracy_for_soft_targets(
                prediction_soft_label[:, 1:], label_targets[:, 1:])
        val_loss += loss.item()
        n_feat += batch['img_mask_tgt'].sum().item()
    val_loss = sum(all_gather_list(val_loss))
    tot_score = sum(all_gather_list(tot_score))
    n_feat = sum(all_gather_list(n_feat))
    tot_time = time()-st
    val_loss /= n_feat
    val_acc = tot_score / n_feat
    val_log = {'loss': val_loss,
               'acc': val_acc,
               'feat_per_s': n_feat/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"score: {val_acc*100:.2f}")
    return val_log


def compute_accuracy_for_soft_targets(out, labels):
    outputs = out.max(dim=-1)[1]
    labels = labels.max(dim=-1)[1]  # argmax
    n_correct = (outputs == labels).sum().item()
    return n_correct


@torch.no_grad()
def validate_itm(model, val_loader):
    LOGGER.info("start running ITM validation...")
    val_loss = 0
    tot_ot_loss = 0
    tot_ot_pos = 0
    tot_ot_neg = 0
    tot_score = 0
    n_ex = 0
    st = time()
    for i, batch in enumerate(val_loader):
        scores, ot_loss = model(batch, task='itm', compute_loss=False)
        if ot_loss is not None:
            if isinstance(ot_loss, tuple):
                ot_pos, ot_neg = ot_loss
                ot_pos = ot_pos.sum().item()
                ot_neg = ot_neg.sum().item()
                tot_ot_pos += ot_pos
                tot_ot_neg += ot_neg
                tot_ot_loss += ot_pos - ot_neg
            else:
                tot_ot_loss += ot_loss.sum().item()
        targets = batch['targets']
        loss = F.cross_entropy(scores, targets, reduction='sum')
        val_loss += loss.item()

        tot_score += (scores.max(dim=-1)[1] == targets).sum().item()
        n_ex += len(targets)
    val_loss = sum(all_gather_list(val_loss))
    tot_score = sum(all_gather_list(tot_score))
    n_ex = sum(all_gather_list(n_ex))
    tot_time = time()-st
    val_loss /= n_ex
    val_acc = tot_score / n_ex
    val_log = {'valid/loss': val_loss,
               'valid/acc': val_acc,
               'valid/ex_per_s': n_ex/tot_time}

    if ot_loss is not None:
        tot_ot_loss = sum(all_gather_list(tot_ot_loss))
        tot_ot_pos = sum(all_gather_list(tot_ot_pos))
        tot_ot_neg = sum(all_gather_list(tot_ot_neg))
        val_log['valid/ot_loss'] = tot_ot_loss / n_ex
        val_log['valid/ot_pos'] = tot_ot_pos / n_ex
        val_log['valid/ot_neg'] = tot_ot_neg / n_ex

    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"score: {val_acc*100:.2f}")
    return val_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    # NOTE: train tasks and val tasks cannot take command line arguments
    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')

    parser.add_argument("--model_config", type=str,
                        help="path to model structure config json")
    parser.add_argument("--checkpoint", default=None, type=str,
                        help="path to model checkpoint (*.pt)")

    parser.add_argument(
        "--output_dir", default=None, type=str,
        help="The output directory where the model checkpoints will be "
             "written.")

    parser.add_argument('--mrm_prob', default=0.15, type=float,
                        help='probability to mask in MRM training')
    parser.add_argument('--itm_neg_prob', default=0.5, type=float,
                        help='probability to make negative examples'
                             'in ITM training')
    parser.add_argument('--itm_ot_lambda', default=0.0, type=float,
                        help='weight of OT (optimal transport) loss (WRA)')

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
    parser.add_argument("--train_batch_size", default=4096, type=int,
                        help="Total batch size for training. "
                             "(batch by tokens)")
    parser.add_argument("--val_batch_size", default=4096, type=int,
                        help="Total batch size for validation. "
                             "(batch by tokens)")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
                        help="Number of updates steps to accumualte before "
                             "performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--valid_steps", default=1000, type=int,
                        help="Run validation every X steps")
    parser.add_argument("--num_train_steps", default=100000, type=int,
                        help="Total number of training updates to perform.")
    parser.add_argument("--optim", default='adamw',
                        choices=['adam', 'adamax', 'adamw'],
                        help="optimizer")
    parser.add_argument("--betas", default=[0.9, 0.98], nargs='+',
                        help="beta for adam optimizer")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="tune dropout regularization")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="weight decay (L2) regularization")
    parser.add_argument("--grad_norm", default=2.0, type=float,
                        help="gradient clipping (-1 for no clipping)")
    parser.add_argument("--warmup_steps", default=10000, type=int,
                        help="Number of training steps to perform linear "
                             "learning rate warmup for.")

    # device parameters
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true', help="pin memory")

    # can use config files
    parser.add_argument('--config', required=True, help='JSON config files')

    args = parse_with_config(parser)

    if exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not "
                         "empty.".format(args.output_dir))

    # options safe guard
    if args.conf_th == -1:
        assert args.max_bb + args.max_txt_len + 2 <= 512
    else:
        assert args.num_bb + args.max_txt_len + 2 <= 512

    main(args)
