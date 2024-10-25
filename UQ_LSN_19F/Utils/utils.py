import numpy as np
import pandas as pd


## Uncertainty estimation

def Binary_Entropy(p):
    p = p.cpu().numpy()
    p_f = p
    p_b = 1-p
    p_b = np.where(p_b == 0, 0.0001, p_b)
    H_f = -(p_f *np.log(p_f))
    H_b = -(p_b*np.log(p_b))
    H = -(p_f *np.log(p_f)+p_b*np.log(p_b))
    return(H, H_f, H_b)

min_v = 0 
def update_lr(lr0,batch_idx,cycle_batch_length,n_sam_per_cycle,optimizer):       
    is_end_of_cycle = False
    prop = batch_idx % cycle_batch_length
    pfriction = prop/cycle_batch_length
    lr = lr0 * (min_v +(1.0-min_v)*0.5*(np.cos(np.pi * pfriction)+1.0))
    if prop >= cycle_batch_length-n_sam_per_cycle:
        is_end_of_cycle = True
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_lr(optimizer):
    for group in optimizer.param_groups:
        return(group['lr'])

# remove zero padding
def unpad(img):
    assert len(img.shape)==5, "Invalid shape for unpadd"
    img = img[:,:,8:-8,12:-12,12:-12]
    return(img)
