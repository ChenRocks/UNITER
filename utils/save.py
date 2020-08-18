"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

saving utilities
"""
import json
import os
from os.path import abspath, dirname, exists, join
import subprocess

import torch

from utils.logger import LOGGER


def save_training_meta(args):
    if args.rank > 0:
        return

    if not exists(args.output_dir):
        os.makedirs(join(args.output_dir, 'log'))
        os.makedirs(join(args.output_dir, 'ckpt'))

    with open(join(args.output_dir, 'log', 'hps.json'), 'w') as writer:
        json.dump(vars(args), writer, indent=4)
    model_config = json.load(open(args.model_config))
    with open(join(args.output_dir, 'log', 'model.json'), 'w') as writer:
        json.dump(model_config, writer, indent=4)
    # git info
    try:
        LOGGER.info("Waiting on git info....")
        c = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                           timeout=10, stdout=subprocess.PIPE)
        git_branch_name = c.stdout.decode().strip()
        LOGGER.info("Git branch: %s", git_branch_name)
        c = subprocess.run(["git", "rev-parse", "HEAD"],
                           timeout=10, stdout=subprocess.PIPE)
        git_sha = c.stdout.decode().strip()
        LOGGER.info("Git SHA: %s", git_sha)
        git_dir = abspath(dirname(__file__))
        git_status = subprocess.check_output(
            ['git', 'status', '--short'],
            cwd=git_dir, universal_newlines=True).strip()
        with open(join(args.output_dir, 'log', 'git_info.json'),
                  'w') as writer:
            json.dump({'branch': git_branch_name,
                       'is_dirty': bool(git_status),
                       'status': git_status,
                       'sha': git_sha},
                      writer, indent=4)
    except subprocess.TimeoutExpired as e:
        LOGGER.exception(e)
        LOGGER.warn("Git info not found. Moving right along...")


class ModelSaver(object):
    def __init__(self, output_dir, prefix='model_step', suffix='pt'):
        self.output_dir = output_dir
        self.prefix = prefix
        self.suffix = suffix

    def save(self, model, step, optimizer=None):
        output_model_file = join(self.output_dir,
                                 f"{self.prefix}_{step}.{self.suffix}")
        state_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                      for k, v in model.state_dict().items()}
        torch.save(state_dict, output_model_file)
        if optimizer is not None:
            dump = {'step': step, 'optimizer': optimizer.state_dict()}
            if hasattr(optimizer, '_amp_stash'):
                pass  # TODO fp16 optimizer
            torch.save(dump, f'{self.output_dir}/train_state_{step}.pt')
