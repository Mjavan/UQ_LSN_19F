
import os, glob 
import numpy as np
import pandas as pd
import json
import time
import datetime
import argparse
import random
import copy
from pathlib import Path


import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import cPickle as pickle
except:
    import pickle


import warnings 
warnings.filterwarnings('ignore')


from Metrics.metrics import recall,TNR,FDR,FPR,precision,fp_fn,AUC_Roc_PR,correct_uncmap_thresholded,roc_uncmap_thresholded,ECE
from Utils.utils import Binary_Entropy,unpad
from unet import unet
from dice import DiceCoef, DiceLoss


parser = argparse.ArgumentParser(description='TESTING SG_MCMC FOR Noise')

parser.add_argument('--exp',type=int,default=850,
                    help='ID of this expriment!')

parser.add_argument('--dts',type=str, default='noise')

parser.add_argument('--b_size',type =int, default =2, 
                    help = 'batch size for test set!')

parser.add_argument('--dr',type=float,default=0.0,help='Dropout rate.')
                    
parser.add_argument('--opt',type=str,default='sgd',
                    help='Optimizer to train the model.')

parser.add_argument('--crit',type=str,default='BCrsent', choices=('dice','BCrsent','comb'),
                    help='The cost function that we want to train the model.')

parser.add_argument('--sampler',type=str,default=None, choices=('sgd','sgmcmc',None),
                    help='Optimizer to train the model.')

parser.add_argument('--write-exp',type=bool,default=True, 
                    help='If we want to write test results in a dataframe.')

parser.add_argument('--Nsamples',type=int, default=0,
                    help = 'the number of samples that we want to use for ensembeling.')

parser.add_argument('--logits',type=bool, default=True,
                    help = 'if we want to take mean over logits or probablities.')

parser.add_argument('--metrics',type =tuple, default =['cor_unc_thr','roc_unc_thr','unc_err_dice','ece','ece_s'], 
                    help = 'metrics that we want to use for evaluation!')

parser.add_argument('--eval',type=str,default='last',choices=('best','last'),
                    help='if we want to evaluate at best checkpont or last.')


args = parser.parse_args()
        
def test(args):
    
    root_dir = Path('./UQ_LSN_19')
     
    with open(os.path.join(root_dir / 'params',f'{args.exp}_noise_param.json')) as file:
        HPT = json.load(file)
    
    seed = HPT['seed']
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
       
    path_data = root_dir / 'Data'
    svd = root_dir / 'ckpts'
        
    with open(os.path.join(root_dir/'test_set','fixed_test_set.pickle'),'rb') as f:
        testset = pickle.load(f)
        
    test_loader = DataLoader(testset,batch_size=2,num_workers=4,drop_last=True,shuffle=False)
       
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

    model = UNet(n_channels=1,n_classes=1,n_filters=HPT['n_filter'],drop=args.dr,bilinear=HPT['bil'])
          
    
    if args.eval == 'last':
        
        load_dir = os.path.join(svd, f'{args.opt}_{args.exp}_seg3d.pt')
        checkpoint = torch.load(load_dir, map_location=device)
        epoch = checkpoint['epoch']
        
    elif args.eval == 'best':
        
        load_dir = os.path.join(svd, f'{args.opt}_{args.exp}_best_model.pt')
        checkpoint = torch.load(load_dir, map_location=device)
        epoch = checkpoint['epoch']

        
    
    if args.sampler=='sgmcmc' or args.sampler=='sgd':
             
        weights_set = torch.load(os.path.join(svd,f'{args.opt}_{args.exp}_state_dicts.pt'),map_location=device)
        sampled_epochs = torch.load(os.path.join(svd,f'{args.opt}_{args.exp}_epochs.pt'),map_location=device)
        
        assert len(weights_set)==len(sampled_epochs),print('The length of sampled weights and sampled epochs are not equal')
            
        for wieght in (weights_set):
            if len(weights_set)> args.Nsamples:
                weights_set.pop(0)
                sampled_epochs.pop(0)
            
        weight_set_samples = weights_set
                   
    else:
        
        model.load_state_dict(checkpoint['model'])
        model = model.to(device)
    
    if args.crit =='dice':      
        loss = DiceLoss().to(device)
        
    elif args.crit =='BCrsent':
        loss = nn.BCEWithLogitsLoss().to(device)
            
    Dice = DiceCoef()
   
                    
    # original dimension of images
    ch,h,w,d = 1,112,40,40
    
    # saving voxel-wsie uncertainty 
    b_entropy_tot = np.zeros((len(testset),h,w,d))

    out_tot= np.zeros((len(testset),h,w,d))
    preds_tot = np.zeros((len(testset),h,w,d))
    probs_tot = np.zeros((len(testset),h,w,d))
    
    
    masks,imgs,indices = [],[],[]
    
    n_samples = 0
    total_loss = 0
    total_dice = 0
    
    
    tic = time.time()

    with torch.no_grad(): 
        
        for j,(vol,mask,idx) in enumerate(test_loader):

            vol = vol.to(device)
            mask = mask.to(device)
            indices.extend(idx)
            
            if args.sampler=='sgmcmc' or args.sampler=='sgd':
                
                # store predictions over sample weights
                out = vol.data.new(args.Nsamples,args.b_size,ch,h,w,d)
                
                for idx, weight_dict in enumerate(weight_set_samples):  
                    
                    model.load_state_dict(weight_dict)
                    model.to(device)
                    
                    out[idx] = unpad(model(vol.float()))
                
                if not args.logits:
                    probs = F.sigmoid(out).data.mean(dim=0)
                     
                mean_out = out.mean(dim=0,keepdim=False)
                out = mean_out
                            
            else:
                out = unpad(model(vol))
                
                if not args.logits:
                    probs = F.sigmoid(out).data
                
            if args.logits:
                probs = F.sigmoid(out).data 
            
            # Binerzing predictions
            preds = (probs>0.5).float()
            
            vol = unpad(vol)
            mask= unpad(mask)

            preds_np = preds.cpu().squeeze().numpy()
            mask_np = mask.cpu().detach().squeeze().numpy()
            vol_np = vol.cpu().detach().squeeze().numpy()
            
            imgs.extend(vol_np)
            masks.extend(mask_np)
            
            if args.crit=='BCrsent':
                target= mask
                loss_t = loss(out,target)
                    
            elif args.crit=='dice':
                target = mask
                loss_t = loss(probs,target)
                
            elif args.crit=='comb':
                target = mask
                loss_t1 = nn.BCEWithLogitsLoss()(out,target)
                loss_t2 = DiceLoss()(probs,target)
                loss_t = loss_t1+loss_t2
                
            total_loss += loss_t.item()
                     
            probs_tot[n_samples:n_samples+len(vol),:] = probs.cpu().squeeze().numpy()            
            out_tot[n_samples:n_samples+len(vol),:] = out.detach().cpu().squeeze().numpy() 
            preds_tot[n_samples:n_samples+len(vol),:]= preds.cpu().squeeze().numpy()

            b_ent, f_unc, b_unc = Binary_Entropy(probs.squeeze())
            b_entropy_tot[n_samples:n_samples+len(vol),:] = b_ent
             
            n_samples += len(vol)
                    
        total_loss /= len(test_loader)
        
        imgs = np.array(imgs)
        masks= np.array(masks)
        indices.extend(['tot'])
        
        correct = (preds_tot==masks)
        err_tot = (preds_tot!=masks)
        
        # compute auc_roc and auc_pr
        auc_roc_score,auc_pr_score,precision,recall,thresholds1,fprm,tprm,thresholds2 = AUC_Roc_PR(masks,probs_tot) # fpr_fnr3
                                     
        df_dice = pd.DataFrame(index=['dice','tps','gts','det_sig_per','tpr','ppv'],columns=indices)
        
        dice_t,tpr_t,ppv_t = 0,0,0
        
        # compute precesion and recall for each image
        for i,(pred,gt) in enumerate(zip(preds_tot,masks)):
            
            pred_b = pred.copy().astype(np.bool)
            gt_b = gt.copy().astype(np.bool)
            
            dice = Dice(pred,gt)
            df_dice.loc['dice'][i]= round(dice.item(),4)
            dice_t += dice.item()
            
            tp,tn,fp,fn = fp_fn(pred_b,gt_b)
            df_dice.loc['tps'][i] = tp
            
            tpr = recall(tp,fn)
            df_dice.loc['tpr'][i] = round(tpr.item(),4)
            
            tnr = TNR(tn,fp)
            fdr = FDR(tp,fp)
            
            ppv = precision(tp,fp)
            df_dice.loc['ppv'][i] = round(ppv.item(),4)
            
            tpr_t += tpr
            ppv_t += ppv
            
            df_dice.loc['gts'][i] = gt_b.sum()
            
            df_dice.loc['det_sig_per'][i] = (df_dice.loc['tps'][i]/df_dice.loc['gts'][i])* 100
             
        total_dice = dice_t/preds_tot.shape[0]
        df_dice.loc['dice']['tot']= total_dice

        df_dice.loc['tpr']['tot']= tpr_t/preds_tot.shape[0]
        df_dice.loc['ppv']['tot']= ppv_t/preds_tot.shape[0]
             
            
        if args.metrics:
            
            if 'corr_unc_thr' in args.metrics:
                
                rm_thr = [l.round(2) for l in list(np.arange(0.1,0.8,0.1))]
                
                df = correct_uncmap_thresholded(preds_tot,b_entropy_tot,masks,rm_thr)                                    
            
            if 'roc_unc_thr' in args.metrics:
                
                thr_roc_unc = [0.4,0.5,0.6,0.69,0.7]    
                
                df_tpr, df_fdr, df_pr, df_fpr = roc_uncmap_thresholded(probs_tot,masks,b_entropy_tot,thr_roc_unc)
            
            
            if 'unc_err_dice' in args.metrics:
                
                thr_unc_err = [l.round(2) for l in list(np.arange(0.05,0.7,0.05))]
                
                dic_tot = dict.fromkeys(thr_unc_err)
                
                df_unc = pd.DataFrame(index=np.arange(len(testset)),columns=thr_unc_err)
                
                for thr in thr_unc_err:
                    
                    unc_thr = (b_entropy_tot>=thr).astype(np.float)

                    Dice_unc = Dice(err_tot,unc_thr)
                    
                    dic_tot[thr] = round(Dice_unc.item(),4)
                    
                    for i,(err_s,unc_s) in enumerate(zip(err_tot,unc_thr)):
                            
                        dice_s = Dice(err_s,unc_s)
   
                        df_unc.loc[i][thr] = dice_s.item()
                
                df_dic = pd.DataFrame(dic_tot,index =["dice"])
                     
            
            if 'ece' in args.metrics:
                
                preds_b = preds_tot.copy().astype(np.bool)
                gts_b = masks.copy().astype(np.bool)
                
                gts = masks.copy()
                
                tps = np.logical_and(preds_b,gts_b)
                prob_tps = np.where(probs_tot>0.5,probs_tot,0)
                
                pred_prob = probs_tot
                
                ece_tps,acc_tps,conf_tps = ECE(pred_prob,gts)
                
                # ece for subjects
                if 'ece_s' in args.metrics:
                    
                    idx =[1,3,7,9]
                    
                    for i in idx:
                        ece_i,acc_i,conf_i = ECE(pred_prob[i],gts[i])
                                

        if args.write_exp:

            data = {
            'exp': [args.exp],
            'in_size':[HPT['in_size']],
            'opt':[args.opt],
            'dr': [args.dr],
            'filter':[HPT['n_filter']],
            'n_ensembel': [args.Nsamples],
            'nll':[round(total_loss,4)],
            'dice':[round(total_dice,4)]}

            csv_path = root_dir/'results'/'nll'/ 'run_sweeps_test.csv'

            if os.path.exists(csv_path):

                sweeps_df = pd.read_csv(csv_path)
                sweeps_df = sweeps_df.append(
                pd.DataFrame.from_dict(data), ignore_index=True).set_index('exp')

            else:
                
                sweeps_df = pd.DataFrame.from_dict(data).set_index('exp')

            sweeps_df.to_csv(csv_path)               
           
                
if __name__=='__main__':
    
    test(args)   
            
        
                
                           
                           


       
