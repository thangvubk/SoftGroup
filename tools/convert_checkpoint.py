# Convert spconv1 checkpoint to spconv2 checkpoint
import argparse
from collections import OrderedDict

import torch

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint', type=str, help='spconv1 checkpoint')
args = parser.parse_args()

checkpoint = torch.load(args.checkpoint)
model = checkpoint['net']
new_model = OrderedDict()

for k, v in model.items():
    new_k, new_v = k, v
    if 'weight' in k and len(v.size()) == 5:
        # KKKIO to OKKKI (0, 1, 2, 3, 4) -> (4, 0, 1, 2, 3)
        new_v = v.permute(4, 0, 1, 2, 3)
    if 'intra_ins_unet' in k:
        new_k = k.replace('intra_ins_unet', 'tiny_unet')
    elif 'score_linear' in new_k:
        new_k = k.replace('score_linear', 'iou_score_linear')
    elif 'intra_ins_outputlayer' in k:
        new_k = k.replace('intra_ins_outputlayer', 'tiny_unet_outputlayer')
    new_model[new_k] = new_v

checkpoint['net'] = new_model
torch.save(checkpoint, args.checkpoint.replace('.pth', '_spconv2.pth'))
