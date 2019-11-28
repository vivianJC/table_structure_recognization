import torch
import numpy as np
import torch.nn as nn


class segLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, masks_probs_v, masks_probs_h, true_masks_v, true_masks_h):
        self.mask_v = masks_probs_v
        self.mask_h = masks_probs_h
        self.true_v = true_masks_v
        self.true_h = true_masks_h

        masks_probs_v = masks_probs_v.permute(1,0)
        masks_probs_h = masks_probs_h.permute(1,0)
        
        true_masks_v = true_masks_v.squeeze().long()
        true_masks_h = true_masks_h.squeeze().long()
        
        criterion = nn.CrossEntropyLoss(weight = torch.tensor([0.5, 1, 0]))
        criterion = criterion.cuda(1)
        
        
        loss = (0.0*criterion(masks_probs_v, true_masks_v) + 1.0*criterion(masks_probs_h, true_masks_h))
        
        return loss

