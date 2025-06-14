from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from .network_utils import (
    Classifier, 
    ResBlock, 
    ConvNormAct, 
    convert_to_rpm_matrix_v9,
    convert_to_rpm_matrix_v6
)

class SelfAttention(nn.Module):
    def __init__(
        self,
        in_planes,
        dropout = 0.1,
        num_heads = 8
    ): 
        super().__init__()
        self.q = nn.Linear(in_planes, in_planes)
        self.kv = nn.Linear(in_planes, in_planes*2)
        self.num_heads=num_heads
        self.head_dim=in_planes//num_heads
        self.m = nn.Linear(in_planes, in_planes)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
        b,t,l,c = x.shape
        shortcut = x
        q = F.normalize(self.q(x).reshape(b, t, l, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4), dim=-1)
        k, v = self.kv(x).reshape(b, t, l, self.num_heads*2, self.head_dim).permute(0, 1, 3, 2, 4).chunk(2, dim=2)
        k, v = F.normalize(k, dim=-1), F.normalize(v, dim=-1)

        atten = self.drop(q @ k.transpose(-2, -1))
        atten = F.softmax(atten / math.sqrt(self.head_dim), dim=-1)
        x = (atten @ v)

        x = self.m(x.permute(0,1,3,2,4).reshape(b,t,l,c))+shortcut
        return x

class PredictiveReasoningBlock(nn.Module):

    def __init__(
        self, 
        in_planes, 
        ou_planes,
        downsample,
        stride = 1, 
        dropout = 0.1, 
        num_contexts = 8
    ):

        super().__init__()

        self.stride = stride

        md_planes = ou_planes*4
        self.pconv = ConvNormAct(in_planes, in_planes, (num_contexts, 1))
        self.conv1 = ConvNormAct(in_planes, md_planes, 3, 1)
        self.conv2 = ConvNormAct(md_planes, ou_planes, 3, 1)
        self.conv = nn.Sequential(ConvNormAct((num_contexts+1), (num_contexts+1)*4, 3, 1), ConvNormAct((num_contexts+1)*4, (num_contexts+1), 3, 1))
        self.drop = nn.Dropout(dropout) if dropout > .0 else nn.Identity()
        self.lp = nn.Linear(in_planes, in_planes*2)
        self.m = nn.Linear(in_planes, in_planes)
        self.m1 = nn.Linear(in_planes, in_planes)

        self.downsample = downsample

    def forward(self, x):
        
        b, c, t, l = x.size()
        identity = self.downsample(x)
        g, x = self.lp(x.permute(0,2,3,1)).chunk(2, dim=-1)
        g = self.m(self.conv(g.contiguous()))
        x = x.permute(0,3,1,2).contiguous()
        contexts, choices = x[:,:,:t-1], x[:,:,t-1:]
        predictions = self.pconv(contexts)
        prediction_errors = F.relu(choices) - predictions
        
        out = torch.cat((contexts, prediction_errors), dim=2)
        out = self.m1(out.permute(0,2,3,1)*F.gelu(g)).permute(0,3,1,2).contiguous()
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.drop(out)
        out = out + identity
        
        return out


class PredRNet(nn.Module):

    def __init__(self, num_filters=32, block_drop=0.0, classifier_drop=0.0, 
                 classifier_hidreduce=1.0, in_channels=1, num_classes=8, 
                 num_extra_stages=1, reasoning_block=PredictiveReasoningBlock,
                 num_contexts=8):

        super().__init__()

        channels = [num_filters, num_filters*2, num_filters*3, num_filters*4]
        strides = [2, 2, 2, 2]

        # -------------------------------------------------------------------
        # frame encoder 

        self.in_planes = in_channels

        for l in range(len(strides)):
            setattr(
                self, "res"+str(l), 
                self._make_layer(
                    channels[l], stride=strides[l], 
                    block=ResBlock, dropout=block_drop,
                )
            )
        # -------------------------------------------------------------------



        # -------------------------------------------------------------------
        # predictive coding 
        self.num_extra_stages = num_extra_stages
        self.num_contexts = num_contexts
        self.in_planes = 32
        self.channel_reducer = ConvNormAct(channels[-1], self.in_planes, 1, 0, activate=False)    

        for l in range(num_extra_stages):
            setattr(
                self, "prb"+str(l), 
                self._make_layer(
                    self.in_planes, stride = 1, 
                    block = reasoning_block, 
                    dropout = block_drop
                )
            )
        # -------------------------------------------------------------------

        self.featr_dims = 1024

        self.classifier = Classifier(
            self.featr_dims, 1, 
            norm_layer = nn.BatchNorm1d, 
            dropout = classifier_drop, 
            hidreduce = classifier_hidreduce
        )


        
        self.in_channels = in_channels
        self.ou_channels = num_classes


    def _make_layer(self, planes, stride, dropout, block, downsample=True):
        if downsample and block == ResBlock:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size = 2, stride = stride) if stride != 1 else nn.Identity(),
                ConvNormAct(self.in_planes, planes, 1, 0, activate = False, stride=1),
            )
        elif downsample and (block == PredictiveReasoningBlock or type(block) == partial):
            downsample = ConvNormAct(self.in_planes, planes, 1, 0, activate = False)
        else:
            downsample = nn.Identity()

        if block == PredictiveReasoningBlock or type(block) == partial:
            stage = block(self.in_planes, planes, downsample, stride = stride, 
                          dropout = dropout, num_contexts = self.num_contexts)
        elif block == ResBlock:
            stage = block(self.in_planes, planes, downsample, stride = stride, dropout = dropout)

        self.in_planes = planes

        return stage

    def forward(self, x):

        if self.in_channels == 1:
            b, n, h, w = x.size()
            x = x.reshape(b*n, 1, h, w)
        elif self.in_channels == 3:
            b, n, _, h, w = x.size()
            x = x.reshape(b*n, 3, h, w)

        for l in range(4):
            x = getattr(self, "res"+str(l))(x)

        x = self.channel_reducer(x)

        _, c, h, w = x.size()

        if self.num_contexts == 8:
            x = convert_to_rpm_matrix_v9(x, b, h, w)
        else:
            x = convert_to_rpm_matrix_v6(x, b, h, w)

        x = x.reshape(b * self.ou_channels, self.num_contexts + 1, -1, h * w)
        # e.g. [b,9,c,l] -> [b,c,9,l] (l=h*w)
        x = x.permute(0,2,1,3)

        for l in range(0, self.num_extra_stages):
            x = getattr(self, "prb"+str(l))(x)

        x = x.reshape(b, self.ou_channels, -1)
        x = F.adaptive_avg_pool1d(x, 1024)

        x = x.reshape(b * self.ou_channels, self.featr_dims)

        out = self.classifier(x)

        return out.view(b, self.ou_channels)
    

def predrnet_raven(**kwargs):
    return PredRNet(**kwargs, num_contexts=8)


def predrnet_analogy(**kwargs):
    return PredRNet(**kwargs, num_contexts=5, num_classes=4)