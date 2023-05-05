import os, glob 
import numpy as np
import pandas as pd
import random
import copy
import argparse
import json
import time
import datetime
from pathlib import Path


import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.optimizer import Optimizer, required
from torch.utils.data import DataLoader


try:
    import cPickle as pickle
except:
    import pickle


import warnings 
warnings.filterwarnings('ignore')



from UQ_LSN_19F.optimizer import SGLD, SGHM
from UQ_LSN_19F.Utils.utils import update_lr
from UQ_LSN_19F.Utils.plot import plotCurves
import UQ_LSN_19F.unet as unet
from UQ_LSN_19F.dice import DiceCoef, DiceLoss
from UQ_LSN_19F.Dataloader.data import NoisedData,get_train_val_set,get_transform


parser = argparse.ArgumentParser(description='TRAINING SG_MCMC FOR F19-LSN')


parser.add_argument('--seed',type=int,default=42,
                        help='the seed for experiments!')

parser.add_argument('--exp',type=int,default=900,
                        help='ID of this expriment!')

parser.add_argument('--dts',type=str, default='noise')

parser.add_argument('--epochs',type=int,default=10,
                        help='Number of epochs that we want to train the model.')

# loss function
parser.add_argument('--crit',type=str,default='BCrsent', choices=('dice','BCrsent','comb'),
                        help='The cost function that we want to train the model.')

parser.add_argument('--scale',type=bool, default=True,
                    help = 'if we want to scale loss or not.')

# optimizer
parser.add_argument('--opt',type=str,default='sghm', choices=('sgd','sghm','sgld'),
                        help='The optimizer that we want to train the model with.')

parser.add_argument('--mom',type=float, default =0.99, 
                        help = 'momentum')

parser.add_argument('--weight_decay',type=float, default=0.0)

parser.add_argument('--prior',type =str, default =None,choices=(None,'norm'), 
                        help = 'if we wnat to use prior or not.')

parser.add_argument('--addnoise',type=int, default=0,
                    help = 'if we want to add noise or not.')

parser.add_argument('--temp',type=float, default=1.0, 
                        help = 'temprature for cold posterior.')

parser.add_argument('--epoch-inject',type =int, default=0, 
                        help = 'The epoch that we want to inject the noise to take samples!')

parser.add_argument('--save_sample',type=int, default=1,
                    help = 'if we want to save checkpoints.')

parser.add_argument('--sampling_start',type=int, default=0,
                    help = 'the epoch that we want to start sampling.')

parser.add_argument('--samples',type=int,default=7,
                        help='Number of ensembles that we want to take during training.')

# lr
parser.add_argument('--lr0',type =float, default =1e-1, 
                        help = 'initial learning rate.')

parser.add_argument('--lr-sch',type =str, default ='cyclic',choices=(None,'fixed','cyclic'), 
                        help = 'Type of learning rate schedule.')

parser.add_argument('--n-sam-cycle',type =int, default =1, 
                        help = 'Number of samples that we wnat to take in ecah cycle!')

parser.add_argument('--cycle-length',type=int, default=50,
                     help='cycle length in sghm experiment.')

# Unet model
parser.add_argument('--dr',type=float,default=0.02,
                        help='Dropout rate.')

parser.add_argument('--n_filter',type=int, default=8, 
                        help = 'number of filters in U-net model.')

parser.add_argument('--bil',type=int, default=1,
                   help = 'if we want to use bilinear or upsampling.')

parser.add_argument('--activation',type=str, default="relu", 
                        help = 'activation funstion in U-net model.')

# Preprocess
parser.add_argument('--tr',type=str,default='padd', choices=('inter','padd'),
                        help='If we want to use interpolation or padding to resize the imagse.')

parser.add_argument('--in_size',type=tuple, default =[128,64,64], 
                        help = 'input size for training Unet.')

parser.add_argument('--rm_out',type=int, default=1,
                   help = 'if we want to remove outliers or not.')

parser.add_argument('--nr',type=int, default=0,
                   help = 'if we want to normalize data or not.')

# Dataset
parser.add_argument('--test_size',type =float, default =10, 
                        help = 'the portion of data that we want to use as test set.')

parser.add_argument('--val_size',type =int, default =4, 
                        help = 'the number of images that we want to use as val set.')

parser.add_argument('--b_size_tr',type =int, default=2, 
                        help = 'batch size in training set.')

parser.add_argument('--b_size_val',type =int, default=2,
                        help = 'batch size in val set.')

parser.add_argument('--n_workers',type=int, default =4, 
                        help = 'number of workers.')

parser.add_argument('--save_tr_vl',type=bool, default=True,
                   help = 'save training and validation sets.')

parser.add_argument('--tr-evl',type=bool,default=True,
                        help='If we want to evaluate model on train and val set.')

parser.add_argument('--syn',type=int,default=30,
                        help='0 if we do not want to use synthetic data, n if we want to use n syn data.')

# Augmentation
parser.add_argument('--trans',type =bool, default =True, 
                        help = 'if we wnat to have augmentations or not.')

parser.add_argument('--pad',type =int, default =10, 
                        help = 'if we wnat to use crop resize or not.')

parser.add_argument('--flip-rate',type =float, default =0.5, 
                        help = 'if we wnat to have random flip or not.')

# Plot
parser.add_argument('--plot',type=bool, default=True, 
                        help = 'If we want to plot lr curve or not.')

args = parser.parse_args()


DEFAULT_ALPHA = 0.99
DEFAULT_EPSILON = 1e-7
CLIP_NORM = 0.25

def main(args):
    
    seed = args.seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    
    root_dir = Path('./UQ_LSN_19F')

    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')    
    
    if device =='gpu':
        torch.cuda.set_device(device)
    
    HPD = vars(args)
    with open(root_dir / 'params' / f'{args.exp}_noise_param.json','w') as file:
        json.dump(HPD,file,indent=4)
       
               
    path_data = root_dir / 'Data'
    save_dir = root_dir / 'ckpts'
    dataset = NoisedData(path_data,rm_out=args.rm_out,nr=args.nr,in_size=args.in_size,syn=args.syn)
        
    
    os.makedirs(save_dir,exist_ok=True)
    
    # Transformations
    if args.trans:
        transform_train = get_transform(pad=args.pad,fl_rate=args.flip_rate)
    else:
        transform_train = None
        
    
    # Datasets                    
    train_set, val_set = get_train_val_set(dataset,seed=args.seed,val_size=args.val_size,transform=transform_train)
        
    
    # Making Dataloaders
    train_loader = DataLoader(train_set, batch_size=args.b_size_tr,num_workers=args.n_workers,drop_last=True,shuffle=True)    
    val_loader = DataLoader(val_set, batch_size=args.b_size_val,num_workers=args.n_workers,drop_last=True,shuffle=True)
    
        
    N_train = len(train_loader.dataset)
    
    if args.weight_decay and args.scale and args.opt=='sghm':
        
        weight_decay = args.weight_decay / N_train
    
    # Model
    model = UNet(n_channels=1,n_classes=1,n_filters=args.n_filter,drop=args.dr,bilinear=args.bil).to(device)
        
    # Optimizer
    if args.opt=='adam': 
        optimizer = optim.Adam(model.parameters(),lr=args.lr0,weight_decay=weight_decay)
             
    elif args.opt=='sgd':
        optimizer = optim.SGD(model.parameters(),lr=args.lr0,momentum=args.mom,weight_decay=args.weight_decay)
        
    elif args.opt=='sgld':
        optimizer = SGLD(params=model.parameters(),
                         lr=args.lr0,
                         temp=args.temp,
                         weight_decay=weight_decay,
                         addnoise=args.addnoise,
                         N_train=N_train)
    
    elif args.opt=='sghm':
        optimizer = SGHM(params=model.parameters(),
                         lr=args.lr0,
                         temp=args.temp,
                         weight_decay=weight_decay,
                         addnoise=args.addnoise,
                         momentum=args.mom,
                         dampening=DEFAULT_DAMPENING,
                         N_train=N_train) 
         
    
    if args.crit =='dice':           
        loss = DiceLoss().to(device)
        
    elif args.crit =='BCrsent':
        loss = nn.BCEWithLogitsLoss().to(device)
                
    Dice = DiceCoef()
 
    n_batch = len(train_loader)
    cycle_batch_length = args.cycle_length * n_batch
    batch_idx = 0

    
    weight_set_samples = []
    sampled_epochs = []
    
    best_loss = 1000
    
    loss_total ={'train':[], 'val':[]}
    dice_total={'train':[], 'val':[]}
    dice_total2 = {'train':[], 'val':[]}
    
    for epoch in range(args.epochs): 
        
        tic = time.time()
        
        for phase in ['train','val']:
            
            if phase=='train':
                model.train()
                dataloader = train_loader

            else:
                model.eval()
                dataloader= val_loader
                    
            total_loss =0
            total_dice =0
            total_dice2=0
        
            for j,(vol,mask,_) in enumerate(dataloader):

                vol = vol.to(device)
                mask = mask.to(device)
                
                out = model(vol)
                probs = F.sigmoid(out)
                pred = (probs>0.5).float()
                
                
                dice_t = Dice(pred.squeeze(),mask.squeeze())
                total_dice += dice_t.item()
                
            
                if args.crit=='BCrsent':
                    target= mask
                    loss_t= loss(out,target)                       
                    
                elif args.crit=='dice':
                    target = mask
                    loss_t= loss(probs,target)
                    
                elif args.crit =='comb':
                    target = mask
                    loss_t1= nn.BCEWithLogitsLoss()(out,target)
                    loss_t2= DiceLoss()(probs,target)
                    
                    loss_t = loss_t1+loss_t2
                    
                if args.scale and phase=='train':
                    loss_t = loss_t * N_train
                    total_loss += loss_t.item()*vol.shape[0]/N_train
                    
                elif args.scale and phase=='val':
                    total_loss += loss_t.item()*vol.shape[0]
                    
                else:
                    total_loss += loss_t.item()

                if phase=='train': 
                    
                    optimizer.zero_grad()
                    
                    if args.lr_sch =='cyclic':
                        
                        update_lr(args.lr0,batch_idx,cycle_batch_length,args.n_sam_cycle,optimizer)

                    loss_t.backward()

                    if args.lr_sch =='cyclic':
                        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=CLIP_NORM ,norm_type=2)
                        
                        if (epoch%args.cycle_length)+1 > args.epoch_inject:
                            optimizer.param_groups[0]['epoch_noise']=True
     
                        else:
                            optimizer.param_groups[0]['epoch_noise']=False

                    else:
                        if (epoch+1)> args.epoch_inject:
                            
                            optimizer.param_groups[0]['epoch_noise']=True
                            
                    optimizer.step()
                    batch_idx+=1
                            
            if args.scale:
                loss_total[phase].append(total_loss/len(dataloader.dataset))
            else:
                loss_total[phase].append(total_loss/len(dataloader))
                
            dice_total[phase].append(total_dice/len(dataloader))

            if args.save_sample:

                if epoch>=args.sampling_start and (epoch%args.cycle_length)+1>(args.cycle_length-args.n_sam_cycle) and\
                phase=='train':

                    if len(weight_set_samples) >= args.samples:
                        
                        weight_set_samples.pop(0)
                        sampled_epochs.pop(0)

                    weight_set_samples.append(copy.deepcopy(model.state_dict()))
                    sampled_epochs.append(epoch)
                        
    
        toc = time.time()
        runtime_epoch = toc - tic
        
        print('Epoch:%d, loss_train:%0.4f, loss_val:%0.4f, dice_train:%0.4f, dice_val:%0.4f, time:%0.4f seconds'%\
                  (epoch,loss_total['train'][epoch],loss_total['val'][epoch],\
                   dice_total['train'][epoch],dice_total['val'][epoch],\
                   runtime_epoch)) 
        
        is_best = bool(loss_total['val'][epoch] < best_loss)
        best_loss = loss_total['val'][epoch] if is_best else best_loss
        loss_val = loss_total['val'][epoch]
        
        if is_best:
            
            checkpoints = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': best_loss,
                'val_dice_score': dice_total['val'][epoch]}
        
            torch.save(checkpoints, save_dir / f'{args.opt}_{args.exp}_best_model.pt')
        
    
    state = pd.DataFrame({'train_loss':loss_total['train'], 'valid_loss':loss_total['val'],
             'train_dice':dice_total['train'],'valid_dice':dice_total['val']})
    
    state.to_csv(root_dir / 'loss' / f'{args.opt}_exp_{args.exp}_loss.csv')
    
    # save model at the end of training
    print(f'model saved at end of training!')
    torch.save({'epoch':epoch,
               'lr':args.lr0,
               'model':model.state_dict(),
               'optimizer':optimizer.state_dict(),
               'Loss':state}, save_dir / f'{args.opt}_{args.exp}_seg3d.pt')
    
    if args.save_sample: 
        
        torch.save(weight_set_samples,save_dir / f'{args.opt}_{args.exp}_state_dicts.pt')
        torch.save(sampled_epochs,save_dir / f'{args.opt}_{args.exp}_epochs.pt')
    
    if args.plot:  

        plotCurves(state,root_dir / 'lr_curves' / f'{args.opt}_exp_{args.exp}_loss.png') 
    
    
    print(f'finish training')



if __name__=='__main__':
    
    main(args)
    






