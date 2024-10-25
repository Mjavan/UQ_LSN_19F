import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score,precision_recall_curve, precision_score, recall_score,f1_score, average_precision_score, auc 


### ECE 
def ECE(conf,acc,n_bins=15):
    acc_list = []
    conf_list= []
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece =  0.0
    bin_counter = 0
    avg_confs_in_bins = []
    for bin_lower, bin_upper in zip(bin_lowers,bin_uppers):
        in_bin = np.logical_and(conf > bin_lower, conf <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(acc[in_bin])
            avg_conf_in_bin = np.mean(conf[in_bin])
            delta = avg_conf_in_bin - acc_in_bin
            avg_confs_in_bins.append(delta)
            acc_list.append(acc_in_bin)
            conf_list.append(avg_conf_in_bin)
            ece += np.abs(delta) * prop_in_bin
        else:
            avg_confs_in_bins.append(None)
        bin_counter+=1 
    return(ece,acc_list,conf_list)


### Improving AUC by filtering most uncertain voxels
def roc_uncmap_thresholded(probs,gts,uncs,thersholds):
    """ This code computes auc for uncertainty maps thresholded at different uncertainty thresholds """    
    b_thrs = [l.round(2) for l in list(np.arange(0.0,1.1,0.1))]
    b_thrs.append('retrained_voxels')
    df_tpr = pd.DataFrame(columns=b_thrs, index=thersholds)
    df_fdr = pd.DataFrame(columns=b_thrs, index=thersholds)
    df_pr = pd.DataFrame(columns=b_thrs, index=thersholds)
    df_fpr = pd.DataFrame(columns=b_thrs, index=thersholds)
    for u_thr in thersholds:
        uncs_c = uncs.copy()
        mask = (uncs_c>= u_thr)
        for b_thr in b_thrs[:-1]:
            pred_temp = np.zeros(probs.shape)
            probs_c = probs.copy()
            pred_temp = np.where(probs_c>=b_thr,1.0,0.0)
            pred_thr = pred_temp.copy()
            gts_thr = gts.copy()
            gt_sum = gts_thr.sum()
            pred_thr[mask]=0
            gts_thr[mask]=0
            gt_mask_sum = gts_thr.sum()
            pred_thr_b = pred_thr.astype(np.bool)
            gts_thr_b = gts_thr.astype(np.bool)
            tp,tn,fp,fn = fp_fn(pred_thr_b,gts_thr_b)
            tpr = recall(tp, fn)
            fdr = FDR(tp,fp)
            fpr = FPR(tn,fp)
            ppv = precision(tp,fp)
            df_tpr.loc[u_thr][b_thr] = tpr
            df_fdr.loc[u_thr][b_thr] = fdr
            df_pr.loc[u_thr][b_thr] = ppv
            df_fpr.loc[u_thr][b_thr]= fpr
        df_tpr.loc[u_thr]['retained_voxels'] = (gt_mask_sum/gt_sum)*100
    return(df_tpr, df_fdr, df_pr, df_fpr)        
            
### Benefit of uncertainties to correct segmentation failures 
def correct_uncmap_thresholded(preds, uncs,gts, thersholds):
    """This code corrects unecrtain voxels to the refrence labels at different uncertainty thresholds"""
    df = pd.DataFrame(index = thersholds,columns=['dice_mask','prd_mean','mask_prd_mean','ftpr',\
                                                  'ftnr', 'tpu','tnu','dice_fp','dice_fn',\
                                                  'dice_fp_fn','ref_vox','ref_mis_vox','ref_fp',\
                                                  'ref_fn','dice_add','dice_rm'])                                              
    for thr in thersholds:
        uncs = uncs.copy()
        preds_b = preds.copy().astype(np.bool)
        gts_b = gts.copy().astype(np.bool)
        thersholded_unc = (uncs>=thr) 
        tpu,tnu,fpu,fnu,tpu_s,tnu_s,fpu_s,fnu_s,tp,tn,fp,fn = fp_fn(preds_b,gts_b, thersholded_unc,unc=True)
        preds_mean = preds.copy().mean()
        masked_preds = preds.copy()
        masked_gts = gts.copy()
        masked_preds[thersholded_unc]=0
        masked_gts[thersholded_unc]=0
        masked_preds_mean = masked_preds.mean()
        masked_dice = DiceCoef()(masked_preds,masked_gts)
        
        df.loc[thr]['dice_mask']= masked_dice.item()
        df.loc[thr]['prd_mean'] = preds_mean.item()
        df.loc[thr]['mask_prd_mean']= masked_preds_mean.item()
        
        ftp = (tp-tpu_s)/tp
        ftn = (tn-tnu_s)/tn
        df.loc[thr]['ftpr']= ftp
        df.loc[thr]['ftnr']= ftn
        df.loc[thr]['tpu'] = tpu_s
        df.loc[thr]['tnu'] = tnu_s
        
        corrected_preds = preds.copy()
        corrected_preds[fpu] =0
        corrected_dice_fp = DiceCoef()(corrected_preds,gts)

        df.loc[thr]['dice_fp']= corrected_dice_fp.item() 
        corrected_preds = preds.copy()
        corrected_preds[fnu] =1
        corrected_dice_fn = DiceCoef()(corrected_preds,gts)
        df.loc[thr]['dice_fn']= corrected_dice_fn.item()
        
        # correct (fp to tn) and (fn to tp)
        corrected_preds = preds.copy()
        corrected_preds[fpu] =0
        corrected_preds[fnu] =1
        corrected_dice_fp_fn = DiceCoef()(corrected_preds,gts)
        
        df.loc[thr]['dice_fp_fn'] = corrected_dice_fp_fn.item()
        # correct to foreground :
        corrected_preds = preds.copy()
        corrected_preds[thersholded_unc] = 1
        corrected_add_dice = DiceCoef()(corrected_preds,gts)
        df.loc[thr]['dice_add']= corrected_add_dice.item()
        
        # correct to background : 
        corrected_preds = preds.copy() 
        corrected_preds[thersholded_unc] = 0
        corrected_dice = DiceCoef()(corrected_preds,gts)
        df.loc[thr]['dice_rm'] = corrected_dice.item()
        
        # ratio of voxels that we thershold
        df.loc[thr]['ref_vox'] = (fpu_s+fnu_s) / (tp+tn+fp+fn) 
        # ratio of wrong voxels that we went for correction
        df.loc[thr]['ref_mis_vox'] = (fpu_s+fnu_s) / (fp+fn)
        # ratio of referred fp 
        df.loc[thr]['ref_fp']= fpu_s/fp
        # ratio of referred fn
        df.loc[thr]['ref_fn']= fnu_s/fn         
    return(df)

def recall(tp, fn):
    actual_positives = tp + fn
    if actual_positives <= 0:
        return(0)    
    return(tp / actual_positives)

def TNR(tn, fp):    
    actual_negatives = tn + fp
    if actual_negatives <= 0:
        return(0)
    return(tn / actual_negatives)
     
def FDR(tp,fp):
    predicted_positives = tp + fp
    if predicted_positives <=0: 
        return(0)
    return(fp/predicted_positives)


def FPR(tn,fp):
    actual_negatives = fp + tn
    if actual_negatives <=0:
        return(0)
    return(fp/actual_negatives)


def precision(tp,fp):
    predicted_positives = tp + fp
    if predicted_positives <=0: 
        return(0)
    return(tp/predicted_positives)
    

def fp_fn(preds,gts,uncs_thr=None,unc=False):    
    tps = np.logical_and(gts,preds)
    tns = np.logical_and(~gts,~preds)
    fps = np.logical_and(~gts,preds)
    fns = np.logical_and(gts,~preds)
    
    tp = tps.sum()
    tn = tns.sum()
    fp = fps.sum()
    fn = fns.sum()
    
    if unc:
        tpu = np.logical_and(tps,uncs_thr)
        tnu = np.logical_and(tns,uncs_thr)
        fpu = np.logical_and(fps,uncs_thr)
        fnu = np.logical_and(fns,uncs_thr)
        
        tpu_s = tpu.sum()
        tnu_s = tnu.sum()
        fpu_s = fpu.sum()
        fnu_s = fnu.sum()
        return(tpu,tnu,fpu, fnu,tpu_s,tnu_s,fpu_s,fnu_s,tp,tn,fp,fn)
    else:
        return(tp,tn,fp,fn)

### Compute AUC pr and ROC
def AUC_Roc_PR(masks_t, probs_t):               
    gt = masks_t.flatten()
    prob = probs_t.flatten()
    fpr, tpr, thresholds1 = roc_curve(gt,prob)
    auc_score = roc_auc_score(gt,prob)
    precision, recall, thresholds2 = precision_recall_curve(gt,prob)
    auc_precision_recall = auc(recall, precision)
    return(auc_score,auc_precision_recall,precision,recall,thresholds2,fpr,tpr,thresholds1)
