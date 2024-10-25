import numpy as np
import pandas as pd

import torch
from torch import nn


class DiceCoef(nn.Module):
    def __init__(self):
        super(DiceCoef,self).__init__()

    def forward(self,pred,target,smooth=1.):
        if isinstance(pred,np.ndarray):
            pred =torch.from_numpy(pred)
        if isinstance(target,np.ndarray):
            target = torch.from_numpy(target)
            
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        score =(2. * intersection + smooth) / (A_sum + B_sum + smooth)
        return(score)
    


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss,self).__init__()

    def forward(self,pred,target,smooth=1.):
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        score =(2. * intersection + smooth) / (A_sum + B_sum + smooth)
        loss = 1-score
        return(loss)
    
