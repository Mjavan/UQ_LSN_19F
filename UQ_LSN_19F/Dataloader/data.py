import os, glob 
import numpy as np
import pandas as pd
import random
import torch
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Subset
import torchvision
import torchvision.transforms as T
from torchvision import transforms, datasets
import scipy.io as spio
import nibabel as nib
from scipy import ndimage
from scipy.ndimage import label
from PIL import Image
from skimage.io import imread
from matplotlib import cm
from sklearn.model_selection import ShuffleSplit

import warnings 
warnings.filterwarnings('ignore')


def binerize(mask):
   
    mask_b = np.ones_like(mask)
    mask_b = (mask!=0)
    
    return(mask_b)


class Resize(object):
    
    def __init__(self,in_size):
        
        in_size= tuple(in_size) if isinstance(in_size,list) else (in_size,in_size,in_size)
        self.in_D,self.in_H,self.in_W = in_size
        
        assert not self.in_D%2 and not self.in_H%2 and not self.in_W%2, "Input size must be divided by 2!"
    
    def __call__(self,vol):
        
        [depth, height, width] = vol.shape
        pad_d = int((self.in_D-depth)/2)
        pad_h = int((self.in_H-height)/2)
        pad_w = int((self.in_W-width)/2)
        
        vol = np.pad(vol, ((pad_d,pad_d), (pad_h,pad_h), (pad_w, pad_w)), 'constant')
        return(vol)

class Normalize(object):
    def __init__(self):  
        pass
    def __call__(self,vol):
        vol = (vol-vol.mean())/vol.std()
        return(vol) 

class RandomCropResize(object):
    
    def __init__(self,pad=4):
        self.pad = pad
        
    def __call__(self,vol):
        
        tot_pad = 2*self.pad
        
        x = np.random.randint(0,tot_pad)
        y = np.random.randint(0,tot_pad)
        z = np.random.randint(0,tot_pad)
        
        vol= vol.squeeze()
        
        h,w,d = vol.shape
        
        vol = np.pad(vol,((self.pad,self.pad), (self.pad,self.pad), (self.pad, self.pad)), 'constant')
        
        vol = vol[x:x+h,y:y+w,z:z+d]
        
        vol = np.expand_dims(vol,0)
        
        return(vol)  


class RandFlip(object):
    def __init__(self,flip_rate):
        
        self.flip_rate = flip_rate
        
    def __call__(self,vol):
        
        if np.random.random_sample() < self.flip_rate:
            return(np.fliplr(vol))
        
        return(vol)
    

def get_transform(pad=0,fl_rate=None):

    if pad and not fl_rate:
            
        transform = transforms.Compose([RandomCropResize(pad)])
    
    if fl_rate and not pad:
        
        transform = transforms.Compose([RandFlip(fl_rate)])  
        
    if pad and fl_rate:
        
        transform = transforms.Compose([RandomCropResize(pad),RandFlip(fl_rate)]) 
        
    return(transform)



class NoisedData(Dataset):
    
    def __init__(self,root_dir,rm_out=True,nr=False,in_size=None,syn=0):
        
        self.root_dir = root_dir
        self.rm_out = rm_out
        self.nr = nr
        self.in_size = in_size
        
        
        self.dict_list_real = os.listdir(self.root_dir/'realData')
        
        if '.ipynb_checkpoints' in self.dict_list_real:
            
            idx_ch = self.dict_list_real.index('.ipynb_checkpoints')
            del self.dict_list_real[idx_ch]
        
        dict_imgs_real = [spio.loadmat(os.path.join(self.root_dir/'realData',img)) for img in self.dict_list_real]
            
        self.data = [dic['testData'] for dic in dict_imgs_real]
        self.gt = [dic['refData_thresholded'] for dic in dict_imgs_real]
        self.id = [dic['imageID'][0][:-10] for dic in dict_imgs_real]
        
        if syn:
            
            self.dict_list_syn = os.listdir(self.root_dir/'artificialData')
            
            if '.ipynb_checkpoints' in self.dict_list_syn:
                idx_ch = self.dict_list_syn.index('.ipynb_checkpoints')
                del self.dict_list_syn[idx_ch]
            
            dict_imgs_syn = [spio.loadmat(os.path.join(self.root_dir/'artificialData',img)) \
                             for img in self.dict_list_syn[:syn]]
            
            self.data_syn = [dic['testData'] for dic in dict_imgs_syn]
            self.gt_syn = [dic['refData_thresholded'] for dic in dict_imgs_syn]
            self.id_syn = [dic['imageID'][0] for dic in dict_imgs_syn]
            
            
            self.data.extend(self.data_syn)
            self.gt.extend(self.gt_syn)
            self.id.extend(self.id_syn)
                    
        self.len = len(self.data)
        assert len(self.data)==len(self.gt)
        
        
    def __len__(self):
        
        return(self.len)
    
    
    def rem_outliers(self,vox,nvox=3):
        
        labels, num = label(vox)
    
        for i in range(1, num+1):
        
            nb_vox = np.sum(vox[labels==i])
            
            if nb_vox < nvox:
                
                vox[labels==i]=0     
        return(vox)
    
    
    def __getitem__(self,index):
        
        self.vol = self.data[index]        
        self.mask = self.gt[index]
        self.ID = self.id[index]
       
        self.mask = binerize(self.mask)
        
        if self.rm_out:
            
            self.mask = self.rem_outliers(self.mask)
             
        self.vol  = Resize(self.in_size)(self.vol)
        self.mask = Resize(self.in_size)(self.mask)
                
        if self.nr:
            self.vol = Normalize()(self.vol) 
        
        assert len(self.vol.shape)==len(self.mask.shape)==3
        
        self.vol = self.vol[np.newaxis,:]
        self.mask = self.mask[np.newaxis,:]
    
        self.vol = self.vol.astype('float32')
        self.mask = self.mask.astype('float32')
            
        return(self.vol,self.mask,self.ID)      


    
def get_train_val_set(dataset,seed=42,val_size=1,transform=None):
           
    split = ShuffleSplit(n_splits=1,test_size=val_size,random_state=seed)
        
    indices = range(len(dataset))
        
    for train_index, val_index in split.split(indices):    
        train_ind = train_index
        val_ind = val_index
                
    train_set = Subset(dataset,train_ind)
    val_set = Subset(dataset,val_ind)
    
    if transform:
        
        train_set.transform = transform
                
    return(train_set, val_set)


