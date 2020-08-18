"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

MRM Datasets
"""
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from .data import DetectFeatTxtTokDataset, pad_tensors, get_gather_index


def _get_img_mask(mask_prob, num_bb):
    img_mask = [random.random() < mask_prob for _ in range(num_bb)]
    if not any(img_mask):
        # at least mask 1
        img_mask[random.choice(range(num_bb))] = True
    img_mask = torch.tensor(img_mask)
    return img_mask


def _get_img_tgt_mask(img_mask, txt_len):
    z = torch.zeros(txt_len, dtype=torch.uint8)
    img_mask_tgt = torch.cat([z, img_mask], dim=0)
    return img_mask_tgt


def _get_feat_target(img_feat, img_masks):
    img_masks_ext = img_masks.unsqueeze(-1).expand_as(img_feat)  # (n, m, d)
    feat_dim = img_feat.size(-1)
    feat_targets = img_feat[img_masks_ext].contiguous().view(
        -1, feat_dim)  # (s, d)
    return feat_targets


def _mask_img_feat(img_feat, img_masks):
    img_masks_ext = img_masks.unsqueeze(-1).expand_as(img_feat)
    img_feat_masked = img_feat.data.masked_fill(img_masks_ext, 0)
    return img_feat_masked


class MrfrDataset(DetectFeatTxtTokDataset):
    def __init__(self, mask_prob, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_prob = mask_prob

    def __getitem__(self, i):
        """
        Return:
        - input_ids    : (L, ), i.e., [cls, wd, wd, ..., sep, 0, 0], 0s padded
        - img_feat     : (num_bb, d)
        - img_pos_feat : (num_bb, 7)
        - attn_masks   : (L + num_bb, ), ie., [1, 1, ..., 0, 0, 1, 1]
        - img_mask     : (num_bb, ) between {0, 1}
        """
        example = super().__getitem__(i)
        # text input
        input_ids = example['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)

        # image input features
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'])
        img_mask = _get_img_mask(self.mask_prob, num_bb)
        img_mask_tgt = _get_img_tgt_mask(img_mask, len(input_ids))

        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        return (input_ids, img_feat, img_pos_feat,
                attn_masks, img_mask, img_mask_tgt)


def mrfr_collate(inputs):
    """
    Return:
    - input_ids    : (n, max_L), i.e., [cls, wd, wd, ..., sep, 0, 0], 0s padded
    - position_ids : (n, max_L)
    - txt_lens     : list of [input_len]
    - img_feat     : (n, max_num_bb, d)
    - img_pos_feat : (n, max_num_bb, 7)
    - num_bbs      : list of [num_bb]
    - attn_masks   : (n, max_{L + num_bb}), ie., [1, 1, ..., 0, 0, 1, 1]
    - img_masks    : (n, max_num_bb) between {0, 1}
    """
    (input_ids, img_feats, img_pos_feats, attn_masks, img_masks, img_mask_tgts,
     ) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    # mask features
    img_masks = pad_sequence(img_masks, batch_first=True, padding_value=0)
    feat_targets = _get_feat_target(img_feat, img_masks)
    img_feat = _mask_img_feat(img_feat, img_masks)
    img_mask_tgt = pad_sequence(img_mask_tgts,
                                batch_first=True, padding_value=0)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'feat_targets': feat_targets,
             'img_masks': img_masks,
             'img_mask_tgt': img_mask_tgt}
    return batch


def _get_targets(img_masks, img_soft_label):
    soft_label_dim = img_soft_label.size(-1)
    img_masks_ext_for_label = img_masks.unsqueeze(-1).expand_as(img_soft_label)
    label_targets = img_soft_label[img_masks_ext_for_label].contiguous().view(
        -1, soft_label_dim)
    return label_targets


class MrcDataset(DetectFeatTxtTokDataset):
    def __init__(self, mask_prob, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_prob = mask_prob

    def _get_img_feat(self, fname):
        img_dump = self.img_db.get_dump(fname)
        num_bb = self.img_db.name2nbb[fname]
        img_feat = torch.tensor(img_dump['features'])
        bb = torch.tensor(img_dump['norm_bb'])
        img_bb = torch.cat([bb, bb[:, 4:5]*bb[:, 5:]], dim=-1)
        img_soft_label = torch.tensor(img_dump['soft_labels'])
        return img_feat, img_bb, img_soft_label, num_bb

    def __getitem__(self, i):
        example = super().__getitem__(i)
        img_feat, img_pos_feat, img_soft_labels, num_bb = self._get_img_feat(
            example['img_fname'])

        # image input features
        img_mask = _get_img_mask(self.mask_prob, num_bb)

        # text input
        input_ids = example['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)
        img_mask_tgt = _get_img_tgt_mask(img_mask, len(input_ids))

        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        return (input_ids, img_feat, img_pos_feat,
                img_soft_labels, attn_masks, img_mask, img_mask_tgt)


def mrc_collate(inputs):
    (input_ids, img_feats, img_pos_feats, img_soft_labels,
     attn_masks, img_masks, img_mask_tgts) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]
    num_bbs = [f.size(0) for f in img_feats]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)
    img_soft_label = pad_tensors(img_soft_labels, num_bbs)
    img_masks = pad_sequence(img_masks, batch_first=True, padding_value=0)
    label_targets = _get_targets(img_masks, img_soft_label)

    img_feat = _mask_img_feat(img_feat, img_masks)
    img_mask_tgt = pad_sequence(img_mask_tgts,
                                batch_first=True, padding_value=0)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'img_masks': img_masks,
             'img_mask_tgt': img_mask_tgt,
             'label_targets': label_targets}
    return batch
