

import torch
import torch.nn.functional as F
import torchvision.datasets as td
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import os

#import utils as ut

device='cuda'
dtype=torch.float


from inception import InceptionV3
import fid_score as fs

import numpy as np
def calc_FID(net,M,z):
    


    fid_vals = []
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device)
    batch_size=50
    val = td.CIFAR10('cifar10',train= False,transform=transforms.ToTensor(),download=False)

    vd = DataLoader(dataset=val,batch_size=M, shuffle=True)
    data_v= next(iter(vd))
    gt = data_v[0].to(device)
    label = data_v[1].to(device)
    gt = gt.permute(0,2,3,1).cpu().numpy()
    if gt.shape[-1] == 1:
        gt = np.concatenate([gt, gt, gt], axis=-1)
    gt = np.transpose(gt, axes=(0,3,1,2))
    m1, s1 = fs.calculate_activation_statistics(gt, model, batch_size, 2048, device)

    
    x= sample(net,M,label,z)
    gen = torch.clip(x.permute(0, 2, 3, 1), 0, 1).cpu().numpy()
    if gen.shape[-1] == 1:
        gen = np.concatenate([gen, gen, gen], axis=-1)
    gen = np.transpose(gen, axes=(0, 3, 1, 2))
    m2, s2 = fs.calculate_activation_statistics(gen, model, batch_size, 2048, device)
    fid_value = fs.calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value
        

    

def sample(net,M,label,z):
    with torch.no_grad():
 
 
    
  
  
        x= z
     
     
        steps = 100
        for j in range(steps):
            t=torch.ones((100,),device=device)*(j/steps)
            for p in range(M//100):
                x[p*100:(p+1)*100]= x[p*100:(p+1)*100]- (1/steps)*net(x[p*100:(p+1)*100],t,label[p*100:(p+1)*100])
        

        return x[:,:3,:,:].detach()

    


