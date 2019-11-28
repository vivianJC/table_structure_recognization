import torch
import torch.nn.functional as F

from utils import segLoss
from torch.autograd import Variable


def eval_net(net, dataset, args):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    loss = 0
    
    gpu = args.gpu
    gpu_id = args.gpu_id
    
    criterion = segLoss()
    
    for i, data in enumerate(dataset):
        imgs, vmasks, hmasks = data

        
        imgs = Variable(imgs)
        vmasks = Variable(vmasks)
        hmasks = Variable(hmasks)
        
        if gpu:
            imgs = imgs.cuda(gpu_id)
            vmasks = vmasks.cuda(gpu_id)
            hmasks = hmasks.cuda(gpu_id)
            
        vmasks_pred, hmasks_pred = net(imgs)
        vmasks_pred.detach_()
        hmasks_pred.detach_()

        vmasks_pred = vmasks_pred.permute(1,0,2,3)
        vmasks_pred = vmasks_pred.contiguous()
        vmasks_pred = vmasks_pred.view(vmasks_pred.shape[0],-1)
        hmasks_pred = hmasks_pred.permute(1,0,2,3)
        hmasks_pred = hmasks_pred.contiguous()
        hmasks_pred = hmasks_pred.view(hmasks_pred.shape[0],-1)

        vmasks = vmasks.permute(1,0,2,3)
        vmasks = vmasks.contiguous()
        true_masks_v = vmasks.view(vmasks.shape[0],-1)
        hmasks = hmasks.permute(1,0,2,3)
        hmasks = hmasks.contiguous()
        true_masks_h = hmasks.view(hmasks.shape[0],-1)
                
        loss+= criterion(vmasks_pred, hmasks_pred, true_masks_v, true_masks_h)
    return loss / (i + 1)
