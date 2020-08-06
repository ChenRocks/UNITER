"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Uniter for NLVR2 model
"""
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F

from .model import UniterPreTrainedModel, UniterModel
from .attention import MultiheadAttention


class UniterForNlvr2Paired(UniterPreTrainedModel):
    """ Finetune UNITER for NLVR2 (paired format)
    """
    def __init__(self, config, img_dim):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim)
        self.nlvr2_output = nn.Linear(config.hidden_size*2, 2)
        self.apply(self.init_weights)

    def init_type_embedding(self):
        new_emb = nn.Embedding(3, self.uniter.config.hidden_size)
        new_emb.apply(self.init_weights)
        for i in [0, 1]:
            emb = self.uniter.embeddings.token_type_embeddings\
                .weight.data[i, :]
            new_emb.weight.data[i, :].copy_(emb)
        new_emb.weight.data[2, :].copy_(emb)
        self.uniter.embeddings.token_type_embeddings = new_emb

    def forward(self, batch, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attn_masks = batch['attn_masks']
        gather_index = batch['gather_index']
        img_type_ids = batch['img_type_ids']
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, gather_index,
                                      output_all_encoded_layers=False,
                                      img_type_ids=img_type_ids)
        pooled_output = self.uniter.pooler(sequence_output)
        # concat CLS of the pair
        n_pair = pooled_output.size(0) // 2
        reshaped_output = pooled_output.contiguous().view(n_pair, -1)
        answer_scores = self.nlvr2_output(reshaped_output)

        if compute_loss:
            targets = batch['targets']
            nlvr2_loss = F.cross_entropy(
                answer_scores, targets, reduction='none')
            return nlvr2_loss
        else:
            return answer_scores


class UniterForNlvr2Triplet(UniterPreTrainedModel):
    """ Finetune UNITER for NLVR2 (triplet format)
    """
    def __init__(self, config, img_dim):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim)
        self.nlvr2_output = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_weights)

    def init_type_embedding(self):
        new_emb = nn.Embedding(3, self.uniter.config.hidden_size)
        new_emb.apply(self.init_weights)
        for i in [0, 1]:
            emb = self.uniter.embeddings.token_type_embeddings\
                .weight.data[i, :]
            new_emb.weight.data[i, :].copy_(emb)
        new_emb.weight.data[2, :].copy_(emb)
        self.uniter.embeddings.token_type_embeddings = new_emb

    def forward(self, batch, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attn_masks = batch['attn_masks']
        gather_index = batch['gather_index']
        img_type_ids = batch['img_type_ids']
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, gather_index,
                                      output_all_encoded_layers=False,
                                      img_type_ids=img_type_ids)
        pooled_output = self.uniter.pooler(sequence_output)
        answer_scores = self.nlvr2_output(pooled_output)

        if compute_loss:
            targets = batch['targets']
            nlvr2_loss = F.cross_entropy(
                answer_scores, targets, reduction='none')
            return nlvr2_loss
        else:
            return answer_scores


class AttentionPool(nn.Module):
    """ attention pooling layer """
    def __init__(self, hidden_size, drop=0.0):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(hidden_size, 1), nn.ReLU())
        self.dropout = nn.Dropout(drop)

    def forward(self, input_, mask=None):
        """input: [B, T, D], mask = [B, T]"""
        score = self.fc(input_).squeeze(-1)
        if mask is not None:
            mask = mask.to(dtype=input_.dtype) * -1e4
            score = score + mask
        norm_score = self.dropout(F.softmax(score, dim=1))
        output = norm_score.unsqueeze(1).matmul(input_).squeeze(1)
        return output


class UniterForNlvr2PairedAttn(UniterPreTrainedModel):
    """ Finetune UNITER for NLVR2
        (paired format with additional attention layer)
    """
    def __init__(self, config, img_dim):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim)
        self.attn1 = MultiheadAttention(config.hidden_size,
                                        config.num_attention_heads,
                                        config.attention_probs_dropout_prob)
        self.attn2 = MultiheadAttention(config.hidden_size,
                                        config.num_attention_heads,
                                        config.attention_probs_dropout_prob)
        self.fc = nn.Sequential(
            nn.Linear(2*config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob))
        self.attn_pool = AttentionPool(config.hidden_size,
                                       config.attention_probs_dropout_prob)
        self.nlvr2_output = nn.Linear(2*config.hidden_size, 2)
        self.apply(self.init_weights)

    def init_type_embedding(self):
        new_emb = nn.Embedding(3, self.uniter.config.hidden_size)
        new_emb.apply(self.init_weights)
        for i in [0, 1]:
            emb = self.uniter.embeddings.token_type_embeddings\
                .weight.data[i, :]
            new_emb.weight.data[i, :].copy_(emb)
        new_emb.weight.data[2, :].copy_(emb)
        self.uniter.embeddings.token_type_embeddings = new_emb

    def forward(self, batch, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attn_masks = batch['attn_masks']
        gather_index = batch['gather_index']
        img_type_ids = batch['img_type_ids']
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, gather_index,
                                      output_all_encoded_layers=False,
                                      img_type_ids=img_type_ids)
        # separate left image and right image
        bs, tl, d = sequence_output.size()
        left_out, right_out = sequence_output.contiguous().view(
            bs//2, tl*2, d).chunk(2, dim=1)
        # bidirectional attention
        mask = attn_masks == 0
        left_mask, right_mask = mask.contiguous().view(bs//2, tl*2
                                                       ).chunk(2, dim=1)
        left_out = left_out.transpose(0, 1)
        right_out = right_out.transpose(0, 1)
        l2r_attn, _ = self.attn1(left_out, right_out, right_out,
                                 key_padding_mask=right_mask)
        r2l_attn, _ = self.attn2(right_out, left_out, left_out,
                                 key_padding_mask=left_mask)
        left_out = self.fc(torch.cat([l2r_attn, left_out], dim=-1)
                           ).transpose(0, 1)
        right_out = self.fc(torch.cat([r2l_attn, right_out], dim=-1)
                            ).transpose(0, 1)
        # attention pooling and final prediction
        left_out = self.attn_pool(left_out, left_mask)
        right_out = self.attn_pool(right_out, right_mask)
        answer_scores = self.nlvr2_output(
            torch.cat([left_out, right_out], dim=-1))

        if compute_loss:
            targets = batch['targets']
            nlvr2_loss = F.cross_entropy(
                answer_scores, targets, reduction='none')
            return nlvr2_loss
        else:
            return answer_scores
