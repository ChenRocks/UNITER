"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Referring Expression dataset
"""
import random
import numpy as np
import json
import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from .data import (DetectFeatTxtTokDataset, TxtTokLmdb, DetectFeatLmdb,
                   TxtLmdb, pad_tensors, get_gather_index)


class ReTxtTokLmdb(TxtTokLmdb):
    def __init__(self, db_dir, max_txt_len=120):
        # load refs = [{ref_id, sent_ids, ann_id, image_id, sentences, split}]
        refs = json.load(open(f'{db_dir}/refs.json', 'r'))
        self.ref_ids = [ref['ref_id'] for ref in refs]
        self.Refs = {ref['ref_id']: ref for ref in refs}

        # load annotations = [{id, area, bbox, image_id, category_id}]
        anns = json.load(open(f'{db_dir}/annotations.json', 'r'))
        self.Anns = {ann['id']: ann for ann in anns}

        # load categories = [{id, name, supercategory}]
        categories = json.load(open(f'{db_dir}/categories.json', 'r'))
        self.Cats = {cat['id']: cat['name'] for cat in categories}

        # load images = [{id, file_name, ann_ids, height, width}]
        images = json.load(open(f'{db_dir}/images.json', 'r'))
        self.Images = {img['id']: img for img in images}

        if max_txt_len == -1:
            self.id2len = json.load(open(f'{db_dir}/id2len.json'))
        else:
            self.id2len = {
                id_: len_
                for id_, len_ in json.load(open(f'{db_dir}/id2len.json')
                                           ).items()
                if len_ <= max_txt_len
            }
        self.max_txt_len = max_txt_len
        # self.sent_ids = self._get_sent_ids()

        self.db_dir = db_dir
        self.db = TxtLmdb(db_dir, readonly=True)
        meta = json.load(open(f'{db_dir}/meta.json', 'r'))
        self.cls_ = meta['CLS']
        self.sep = meta['SEP']
        self.mask = meta['MASK']
        self.v_range = meta['v_range']

    def _get_sent_ids(self):
        sent_ids = []
        for ref_id in self.ref_ids:
            for sent_id in self.Refs[ref_id]['sent_ids']:
                sent_len = self.id2len[str(sent_id)]
                if self.max_txt_len == -1 or sent_len < self.max_txt_len:
                    sent_ids.append(str(sent_id))
        return sent_ids

    def shuffle(self):
        # we shuffle ref_ids and make sent_ids according to ref_ids
        random.shuffle(self.ref_ids)
        self.sent_ids = self._get_sent_ids()

    def __getitem__(self, id_):
        # sent_id = self.sent_ids[i]
        txt_dump = self.db[id_]
        return txt_dump


class ReDetectFeatTxtTokDataset(DetectFeatTxtTokDataset):
    def __init__(self, txt_db, img_db):
        assert isinstance(txt_db, ReTxtTokLmdb)
        assert isinstance(img_db, DetectFeatLmdb)
        self.txt_db = txt_db
        self.img_db = img_db
        self.ids = self.txt_db._get_sent_ids()

    def __getitem__(self, i):
        id_ = self.ids[i]
        example = self.txt_db[id_]
        return example

    def shuffle(self):
        self.txt_db.shuffle()


class ReDataset(ReDetectFeatTxtTokDataset):
    def __getitem__(self, i):
        """
        Return:
        :input_ids     : (L, ), i.e., [cls, wd, wd, ..., sep, 0, 0]
        :position_ids  : range(L)
        :img_feat      : (num_bb, d)
        :img_pos_feat  : (num_bb, 7)
        :attn_masks    : (L+num_bb, ), i.e., [1, 1, ..., 0, 0, 1, 1]
        :obj_masks     : (num_bb, ) all 0's
        :target        : (1, )
        """
        # {sent_id, sent, ref_id, ann_id, image_id, bbox, input_ids}
        example = super().__getitem__(i)
        image_id = example['image_id']
        fname = f'visual_grounding_coco_gt_{int(image_id):012}.npz'
        img_feat, img_pos_feat, num_bb = self._get_img_feat(fname)

        # text input
        input_ids = example['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)
        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        # target bbox
        img = self.txt_db.Images[image_id]
        assert len(img['ann_ids']) == num_bb, \
            'Please use visual_grounding_coco_gt'
        target = img['ann_ids'].index(example['ann_id'])
        target = torch.tensor([target])

        # obj_masks, to be padded with 1, for masking out non-object prob.
        obj_masks = torch.tensor([0]*len(img['ann_ids']), dtype=torch.uint8)

        return input_ids, img_feat, img_pos_feat, attn_masks, obj_masks, target


def re_collate(inputs):
    """
    Return:
    :input_ids     : (n, max_L) padded with 0
    :position_ids  : (n, max_L) padded with 0
    :txt_lens      : list of [txt_len]
    :img_feat      : (n, max_num_bb, feat_dim)
    :img_pos_feat  : (n, max_num_bb, 7)
    :num_bbs       : list of [num_bb]
    :attn_masks    : (n, max_{L+num_bb}) padded with 0
    :obj_masks     : (n, max_num_bb) padded with 1
    :targets       : (n, )
    """
    (input_ids, img_feats, img_pos_feats, attn_masks, obj_masks, targets
     ) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    targets = torch.stack(targets, dim=0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)
    obj_masks = pad_sequence(
        obj_masks, batch_first=True, padding_value=1)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    return {'input_ids': input_ids,
            'position_ids': position_ids,
            'img_feat': img_feat,
            'img_pos_feat': img_pos_feat,
            'obj_masks': obj_masks,
            'attn_masks': attn_masks,
            'gather_index': gather_index,
            'targets': targets,
            'txt_lens': txt_lens,
            'num_bbs': num_bbs}


class ReEvalDataset(ReDetectFeatTxtTokDataset):
    def __init__(self, txt_db, img_db, use_gt_feat=True):
        super().__init__(txt_db, img_db)
        self.use_gt_feat = use_gt_feat

    def __getitem__(self, i):
        """
        Return:
        :input_ids     : (L, ), i.e., [cls, wd, wd, ..., sep, 0, 0]
        :position_ids  : range(L)
        :img_feat      : (num_bb, d)
        :img_pos_feat  : (num_bb, 7)
        :attn_masks    : (L+num_bb, ), i.e., [1, 1, ..., 0, 0, 1, 1]
        :obj_masks     : (num_bb, ) all 0's
        :tgt_box       : ndarray (4, ) xywh
        :obj_boxes     : ndarray (num_bb, 4) xywh
        :sent_id
        """
        # {sent_id, sent, ref_id, ann_id, image_id, bbox, input_ids}
        sent_id = self.ids[i]
        example = super().__getitem__(i)
        image_id = example['image_id']
        if self.use_gt_feat:
            fname = f'visual_grounding_coco_gt_{int(image_id):012}.npz'
        else:
            fname = f'visual_grounding_det_coco_{int(image_id):012}.npz'
        img_feat, img_pos_feat, num_bb = self._get_img_feat(fname)

        # image info
        img = self.txt_db.Images[image_id]
        im_width, im_height = img['width'], img['height']

        # object boxes, img_pos_feat (xyxywha) -> xywh
        obj_boxes = np.stack([img_pos_feat[:, 0]*im_width,
                              img_pos_feat[:, 1]*im_height,
                              img_pos_feat[:, 4]*im_width,
                              img_pos_feat[:, 5]*im_height], axis=1)
        obj_masks = torch.tensor([0]*num_bb, dtype=torch.uint8)

        # target box
        tgt_box = np.array(example['bbox'])  # xywh

        # text input
        input_ids = example['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)
        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        return (input_ids, img_feat, img_pos_feat, attn_masks, obj_masks,
                tgt_box, obj_boxes, sent_id)

    # IoU function
    def computeIoU(self, box1, box2):
        # each box is of [x1, y1, w, h]
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[0]+box1[2]-1, box2[0]+box2[2]-1)
        inter_y2 = min(box1[1]+box1[3]-1, box2[1]+box2[3]-1)

        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
        else:
            inter = 0
        union = box1[2]*box1[3] + box2[2]*box2[3] - inter
        return float(inter)/union


def re_eval_collate(inputs):
    """
    Return:
    :input_ids     : (n, max_L)
    :position_ids  : (n, max_L)
    :txt_lens      : list of [txt_len]
    :img_feat      : (n, max_num_bb, d)
    :img_pos_feat  : (n, max_num_bb, 7)
    :num_bbs       : list of [num_bb]
    :attn_masks    : (n, max{L+num_bb})
    :obj_masks     : (n, max_num_bb)
    :tgt_box       : list of n [xywh]
    :obj_boxes     : list of n [[xywh, xywh, ...]]
    :sent_ids      : list of n [sent_id]
    """
    (input_ids, img_feats, img_pos_feats, attn_masks, obj_masks,
     tgt_box, obj_boxes, sent_ids) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)
    obj_masks = pad_sequence(
        obj_masks, batch_first=True, padding_value=1)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    return {'input_ids': input_ids,
            'position_ids': position_ids,
            'img_feat': img_feat,
            'img_pos_feat': img_pos_feat,
            'obj_masks': obj_masks,
            'attn_masks': attn_masks,
            'gather_index': gather_index,
            'tgt_box': tgt_box,
            'obj_boxes': obj_boxes,
            'sent_ids': sent_ids,
            'txt_lens': txt_lens,
            'num_bbs': num_bbs}
