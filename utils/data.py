import os
import cv2
from PIL import Image
import numpy as np

import torch.utils.data as data
import torch
import torchvision.transforms as transforms

def readname(file_path):
    file = open(file_path)
    filename = []
    for line in file.readlines():
        line = line.strip('\n')
        filename.append(line) 
    return filename


class cTDaR19_Dataset(torch.utils.data.Dataset):
    def __init__(self, file_path, filename_path, train = None):
        self.filename = readname(filename_path)
        self.file_path = file_path
        
    def __getitem__(self, idx):
        name = self.filename[idx]
        img_path = os.path.join(self.file_path, name+'.jpg')
        vmask_path = os.path.join(self.file_path, name+'_maskv.png')
        hmask_path = os.path.join(self.file_path, name+'_maskh.png')
        
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)        
        vmask = cv2.imread(vmask_path, cv2.IMREAD_GRAYSCALE)
        hmask = cv2.imread(hmask_path, cv2.IMREAD_GRAYSCALE)
        
        height, width = img.shape[:2]
        size = (800,500)#(min(500, int(width/4)), min(500,int(height/4)))
        normMean = [0.4948052, 0.48568845, 0.44682974]
        normStd = [0.24580306, 0.24236229, 0.2603115]
        normTransform = transforms.Normalize(normMean, normStd)
        Transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            normTransform
        ])
        Transform_mask = transforms.Compose([
            transforms.Resize(size, interpolation=cv2.INTER_NEAREST),
            transforms.ToTensor()
        ])
        
        
        img = Image.fromarray(img) 
        vmask = Image.fromarray(vmask) 
        hmask = Image.fromarray(hmask)   
        
        
        img = Transform(img)
        vmask = Transform_mask(vmask) 
        hmask = Transform_mask(hmask) 

        vmask = vmask*255
        hmask = hmask*255
        
        return img, vmask, hmask
        
    def __len__(self):
        return len(self.filename)

