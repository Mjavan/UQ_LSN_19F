import matplotlib as mlp
import matplotlib.pyplot as plt
import scipy.io as spio
import nibabel as nib


def plotCurves(stats,results_dir=None):
    
    fig = plt.figure(figsize=(12, 6))

    plt.subplot(1,2,1)

    plt.plot(stats['train_loss'], label='train_loss')
    plt.plot(stats['valid_loss'], label='valid_loss')
        
    textsize = 12
    marker=5
        
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('NLL')
        
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='--')
    lgd = plt.legend(['train', 'validation'], markerscale=marker,prop={'size': textsize, 'weight': 'normal'}) 
                 
    ax = plt.gca()    

    plt.subplot(1,2,2)

    plt.plot(stats['train_dice'], label='train')
    plt.plot(stats['valid_dice'], label='validation')

    plt.xlabel('Epochs')
    plt.ylabel('Dice')
    plt.title('Dice coefficient')
        
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='--')
    lgd = plt.legend(['train', 'validation'], markerscale=marker,prop={'size': textsize, 'weight': 'normal'}) 
                 
    fig.tight_layout(pad=3.0)
    plt.savefig(results_dir , bbox_extra_artists=(lgd,), bbox_inches='tight')
    
    plt.show()