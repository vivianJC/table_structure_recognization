import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from eval import eval_net
from unet import UNet
from utils import readname, cTDaR19_Dataset, segLoss

def train_net(net, args):
    epochs=args.epochs
    batch_size=args.batchsize
    lr=args.lr
    gpu=args.gpu
    img_scale=args.scale
    save_cp = args.save_cp
    gpu_device = torch.device('cuda:'+str(args.gpu_id) if torch.cuda.is_available() else "cpu")
    
    img_path = os.path.expanduser('~/table/preprocess')
    filename_path = os.path.expanduser('~/table/code/data')
    dir_checkpoint = os.path.expanduser('~/table/code/ckpt/')
    
    ############################# dataset ##########################
    

    
    train_data = cTDaR19_Dataset(img_path,os.path.join(filename_path,'train.txt'))
    valid_data = cTDaR19_Dataset(img_path,os.path.join(filename_path,'test.txt'))

    # 构建DataLoder
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size)
    
    

    
    ########################### loss & optim  #######################
#    optimizer = optim.SGD(net.parameters(),
#                          lr=lr,
#                          momentum=0.9,
#                          weight_decay=0.0005)
    optimizer = optim.Adam(net.parameters(),
                          lr = lr,
                          betas = (0.9, 0.99))
    criterion = segLoss()
    
    
    #############################  train  ##########################
    
    
    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(train_data),
               len(valid_loader), str(save_cp), str(gpu))) 

    N_train = len(train_data)

    for epoch in range(epochs):
        if epoch>=10:
            optimizer = optim.Adam(net.parameters(),
                          lr = lr/10,
                          betas = (0.9, 0.99))
        
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()

        epoch_loss = 0

        for i, data in enumerate(train_loader):
            imgs, vmasks, hmasks = data
            imgs, vmasks, hmasks = Variable(imgs), Variable(vmasks), Variable(hmasks)

            if gpu:
                imgs = imgs.cuda(gpu_device)
                vmasks = vmasks.cuda(gpu_device)
                hmasks = hmasks.cuda(gpu_device)
            
            
            vmasks_pred, hmasks_pred = net(imgs)
            
            vmasks_pred = vmasks_pred.permute(1,0,2,3)
            masks_probs_v = vmasks_pred.contiguous()
            masks_probs_v = masks_probs_v.view(masks_probs_v.shape[0],-1)
            hmasks_pred = hmasks_pred.permute(1,0,2,3)
            masks_probs_h = hmasks_pred.contiguous()
            masks_probs_h = masks_probs_h.view(masks_probs_h.shape[0],-1)
    
            vmasks = vmasks.permute(1,0,2,3)
            vmasks = vmasks.contiguous()
            true_masks_v = vmasks.view(vmasks.shape[0],-1)
            hmasks = hmasks.permute(1,0,2,3)
            hmasks = hmasks.contiguous()
            true_masks_h = hmasks.view(hmasks.shape[0],-1)

            
            loss = criterion(masks_probs_v, masks_probs_h, true_masks_v, true_masks_h)
            epoch_loss += loss.item()
            
            if i%10==0:
                print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch finished ! Loss: {}'.format(epoch_loss / i))

        if 1:
            val_dice = eval_net(net, valid_loader, args)
            print('Validation Dice Coeff: {}'.format(val_dice))
            with open('val_result.txt','a') as f:
                f.write('%f\n'%(val_dice))

        if save_cp and epoch%10==0:
            torch.save(net.state_dict(),dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))
        

def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=10,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-d', '--gpu_id', dest='gpu_id', default=0,
                     type='int', help='gpu id')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')
    parser.add_option('-m', '--save_cp', dest='save_cp',
                      default=True, help='save checkpoints')
    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()
    dir_checkpoint = os.path.expanduser('~/table/code/ckpt/')
    #############################   net  ##########################
    net = UNet(n_channels=3, n_classes=3)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda(args.gpu_id)
        # cudnn.benchmark = True # faster convolutions, but more memory


    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

        train_net(net, args)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), dir_checkpoint + 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
