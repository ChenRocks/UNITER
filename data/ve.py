"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Visual entailment dataset
# NOTE: basically reuse VQA dataset
"""
from .vqa import VqaDataset, VqaEvalDataset, vqa_collate, vqa_eval_collate


class VeDataset(VqaDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(3, *args, **kwargs)


class VeEvalDataset(VqaEvalDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(3, *args, **kwargs)


ve_collate = vqa_collate
ve_eval_collate = vqa_eval_collate
