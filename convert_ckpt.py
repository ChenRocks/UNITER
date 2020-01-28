import sys
from collections import OrderedDict

import torch

bert_ckpt, output_ckpt = sys.argv[1:]

bert = torch.load(bert_ckpt)
uniter = OrderedDict()
for k, v in bert.items():
    uniter[k.replace('bert', 'uniter')] = v

torch.save(uniter, output_ckpt)
