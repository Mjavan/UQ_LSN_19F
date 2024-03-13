import torch
from torch import nn
import torch.nn.functional as F

import warnings 
warnings.filterwarnings('ignore')



class Conv(nn.Module):

    def __init__(self,in_channel,out_channel,drop=None,bn=True,padding=1,kernel=3,activation=True):

        super(Conv,self).__init__()
        
        self.conv = nn.Sequential()

        self.conv.add_module('conv',nn.Conv3d(in_channel,out_channel,kernel_size=kernel, padding = padding))

        if drop is not None:
            
            self.conv.add_module('dropout', nn.Dropout3d(p=drop))
        if bn:
            self.conv.add_module('bn', nn.BatchNorm3d(out_channel))

        if activation:
            self.conv.add_module('relu', nn.ReLU(inplace=True))

    def forward(self,x):

        x= self.conv(x)
        return(x)


class DoubleConv(nn.Module):

    def __init__(self,in_channel,out_channel,mid_channel=None,drop=None,drop_mode='all',bn=True,repetitions=2):

        super(DoubleConv,self).__init__()
        
        if not mid_channel:

            mid_channel = out_channel

        convs = []

        in_ch_temp = in_channel

        for i in range(repetitions):

            do = _get_dropout(drop, drop_mode, i, repetitions)
            
            convs.append(Conv(in_ch_temp,mid_channel,do,bn))

            in_ch_temp = mid_channel
            mid_channel = out_channel

        self.block = nn.Sequential(*convs)

    def forward(self,x):
        return(self.block(x))


def _get_dropout(drop, drop_mode, i, repetitions):

    if drop_mode == 'all':
        return(drop)

    if drop_mode == 'first' and i == 0:
        return(drop)

    if drop_mode == 'last' and i == repetitions - 1:
        return(drop)

    if drop_mode == 'no':

        return(None)

    return(None)


def _get_dropout_mode(drop_center, curr_depth, depth, is_down):

    if drop_center is None:
        return 'all'

    if curr_depth == depth:
        return 'no'

    if curr_depth + drop_center >= depth:
        return 'last' if is_down else 'first'
    return 'no'

class Down(nn.Module):

    def __init__(self,in_channel,out_channel,
                 drop=None,drop_center = 'all',curr_depth=0,depth=4,bn=True):
        super().__init__()

        do_mode = _get_dropout_mode(drop_center, curr_depth, depth, True)
        self.maxpool_conv= nn.Sequential(
        nn.MaxPool3d(2),
        DoubleConv(in_channel,out_channel,drop=drop,drop_mode=do_mode,bn=bn))

    def forward(self,x):
        return(self.maxpool_conv(x))


class Up(nn.Module):

    def __init__(self, in_channel, out_channel,
                 drop=None,drop_center='all',curr_depth=0,depth=4,bn=True,bilinear=True):

        super(Up,self).__init__()
        do_mode = _get_dropout_mode(drop_center, curr_depth, depth, False)

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channel, out_channel, in_channel // 2,drop,do_mode,bn)
        else:
            self.up = nn.ConvTranspose3d(in_channel,in_channel // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channel, out_channel,drop=drop,drop_mode= do_mode,bn=bn)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        diffZ = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,diffY // 2, diffY - diffY // 2,diffZ // 2,diffZ - diffZ // 2 ])
        x = torch.cat([x2, x1], dim=1)
        return(self.conv(x))

class OutConv(nn.Module):

    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.conv= nn.Conv3d(in_channel,out_channel, kernel_size=1)

    def forward(self,x):
        return(self.conv(x))
    
class UNet(nn.Module):

    DEFAULT_DEPTH = 4
    DEFAULT_DROPOUT = 0.2
    DEFAULT_FILTERS =64

    def __init__(self,n_channels,n_classes,n_filters=DEFAULT_FILTERS,depth=DEFAULT_DEPTH,drop=DEFAULT_DROPOUT,
                 drop_center=None,bn=True,bilinear=True):

        super(UNet,self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_filters =n_filters
        self.bilinear = bilinear
        self.drop = drop
        self.drop_center = drop_center
        self.bn = bn
        
        curr_depth = 0
        do_mode = _get_dropout_mode(drop_center, curr_depth, depth, True)
        self.inc = DoubleConv(n_channels,n_filters,drop=drop,drop_mode=do_mode,bn=bn)
        curr_depth +=1
        self.down1 = Down(n_filters,n_filters*2,drop,drop_center,curr_depth,depth,bn)
        curr_depth +=1
        self.down2 = Down(n_filters*2,n_filters*4,drop,drop_center,curr_depth,depth,bn)
        curr_depth +=1
        self.down3 = Down(n_filters*4,n_filters*8, drop,drop_center,curr_depth,depth,bn)

        factor =2 if self.bilinear else 1
        
        
        self.down4 =Down(n_filters*8,n_filters*16 //factor,drop, drop_center,depth,depth,bn)
        curr_depth = 3
        self.up1 = Up(n_filters*16, n_filters*8 // factor,drop,drop_center,curr_depth,depth,bn,bilinear)
        curr_depth = 2
        self.up2 =Up(n_filters*8, n_filters*4 // factor, drop, drop_center,curr_depth,depth,bn,bilinear)
        curr_depth = 1
        self.up3 = Up(n_filters*4, n_filters*2 //factor,drop, drop_center,curr_depth,depth,bn,bilinear)
        curr_depth = 0
        self.up4 = Up(n_filters*2,n_filters, drop, drop_center,curr_depth,depth,bn,bilinear)

        self.outc = OutConv(n_filters,n_classes)
                     
    def forward(self,x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5,x4)

        x = self.up2(x,x3)
        x = self.up3(x,x2)
        x = self.up4(x,x1)
        logits = self.outc(x)

        return(logits)

    def sample_predict(self,x, Nsamples,classes=None):
        
        b,ch,h,w,d = x.size()
        predictions = x.data.new(Nsamples,b,ch,h,w,d)

        for i in range(Nsamples):
            y = self.forward(x.float())
            predictions[i] = y
        return(predictions)
    
    
if __name__=="__main__":
    
    model = UNet(n_channels=1,n_classes=1,n_filters=4,drop=0,bilinear=1)

    print(model)
