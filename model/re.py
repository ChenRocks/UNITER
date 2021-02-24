"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Uniter for RE model
"""
from collections import defaultdict

import torch
from torch import nn
import random
import numpy as np
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

from .layer import GELU
from .model import UniterPreTrainedModel, UniterModel


class UniterForReferringExpressionComprehension(UniterPreTrainedModel):
    """ Finetune UNITER for RE
    """
    def __init__(self, config, img_dim, loss="cls",
                 margin=0.2, hard_ratio=0.3, mlp=1):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim)
        if mlp == 1:
            self.re_output = nn.Linear(config.hidden_size, 1)
        elif mlp == 2:
            self.re_output = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                GELU(),
                LayerNorm(config.hidden_size, eps=1e-12),
                nn.Linear(config.hidden_size, 1)
            )
        else:
            raise ValueError("MLP restricted to be 1 or 2 layers.")
        self.loss = loss
        assert self.loss in ['cls', 'rank']
        if self.loss == 'rank':
            self.margin = margin
            self.hard_ratio = hard_ratio
        else:
            self.crit = nn.CrossEntropyLoss(reduction='none')

        self.apply(self.init_weights)

    def forward(self, batch, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attn_masks = batch['attn_masks']
        gather_index = batch['gather_index']
        obj_masks = batch['obj_masks']

        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, gather_index,
                                      output_all_encoded_layers=False)
        # get only the region part
        txt_lens, num_bbs = batch["txt_lens"], batch["num_bbs"]
        sequence_output = self._get_image_hidden(
            sequence_output, txt_lens, num_bbs)

        # re score (n, max_num_bb)
        scores = self.re_output(sequence_output).squeeze(2)
        scores = scores.masked_fill(obj_masks, -1e4)  # mask out non-objects

        if compute_loss:
            targets = batch["targets"]
            if self.loss == 'cls':
                ce_loss = self.crit(scores, targets.squeeze(-1))  # (n, ) as no reduction
                return ce_loss
            else:
                # ranking
                _n = len(num_bbs)
                # positive (target)
                pos_ix = targets
                pos_sc = scores.gather(1, pos_ix.view(_n, 1))  # (n, 1)
                pos_sc = torch.sigmoid(pos_sc).view(-1)  # (n, ) sc[0, 1]
                # negative
                neg_ix = self.sample_neg_ix(scores, targets, num_bbs)
                neg_sc = scores.gather(1, neg_ix.view(_n, 1))  # (n, 1)
                neg_sc = torch.sigmoid(neg_sc).view(-1)  # (n, ) sc[0, 1]
                # ranking
                mm_loss = torch.clamp(
                    self.margin + neg_sc - pos_sc, 0)  # (n, )
                return mm_loss
        else:
            # (n, max_num_bb)
            return scores

    def sample_neg_ix(self, scores, targets, num_bbs):
        """
        Inputs:
        :scores    (n, max_num_bb)
        :targets   (n, )
        :num_bbs   list of [num_bb]
        return:
        :neg_ix    (n, ) easy/hard negative (!= target)
        """
        neg_ix = []
        cand_ixs = torch.argsort(
            scores, dim=-1, descending=True)  # (n, num_bb)
        for i in range(len(num_bbs)):
            num_bb = num_bbs[i]
            if np.random.uniform(0, 1, 1) < self.hard_ratio:
                # sample hard negative, w/ highest score
                for ix in cand_ixs[i].tolist():
                    if ix != targets[i]:
                        assert ix < num_bb, f'ix={ix}, num_bb={num_bb}'
                        neg_ix.append(ix)
                        break
            else:
                # sample easy negative, i.e., random one
                ix = random.randint(0, num_bb-1)  # [0, num_bb-1]
                while ix == targets[i]:
                    ix = random.randint(0, num_bb-1)
                neg_ix.append(ix)
        neg_ix = torch.tensor(neg_ix).type(targets.type())
        assert neg_ix.numel() == targets.numel()
        return neg_ix

    def _get_image_hidden(self, sequence_output, txt_lens, num_bbs):
        """
        Extracting the img_hidden part from sequence_output.
        Inputs:
        - sequence_output: (n, txt_len+num_bb, hid_size)
        - txt_lens       : [txt_len]
        - num_bbs        : [num_bb]
        Output:
        - img_hidden     : (n, max_num_bb, hid_size)
        """
        outputs = []
        max_bb = max(num_bbs)
        hid_size = sequence_output.size(-1)
        for seq_out, len_, nbb in zip(sequence_output.split(1, dim=0),
                                      txt_lens, num_bbs):
            img_hid = seq_out[:, len_:len_+nbb, :]
            if nbb < max_bb:
                img_hid = torch.cat(
                        [img_hid, self._get_pad(
                            img_hid, max_bb-nbb, hid_size)],
                        dim=1)
            outputs.append(img_hid)

        img_hidden = torch.cat(outputs, dim=0)
        return img_hidden

    def _get_pad(self, t, len_, hidden_size):
        pad = torch.zeros(1, len_, hidden_size, dtype=t.dtype, device=t.device)
        return pad
