"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

UNITER for VE model
"""
from .vqa import UniterForVisualQuestionAnswering


class UniterForVisualEntailment(UniterForVisualQuestionAnswering):
    """ Finetune UNITER for VE
    """
    def __init__(self, config, img_dim):
        super().__init__(config, img_dim, 3)
