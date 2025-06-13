import torch
import torch.nn as nn
import torch.nn.functional as F

def convert_to_rpm_matrix_v9(input, b):
    # b: batch
    # h: height
    # w: width
    output = input.reshape(b, 16, -1)
    output = torch.stack(
        [torch.cat((output[:,:8], output[:,i].unsqueeze(1)), dim=1) for i in range(8, 16)], 
        dim=1
    )
    output = output.reshape(b*8, 9, -1)

    return output


def convert_to_rpm_matrix_v6(input, b, h, w):
    # b: batch
    # h: height
    # w: width
    output = input.reshape(b, 9, -1, h, w)
    output = torch.stack(
        [torch.cat((output[:,:5], output[:,i].unsqueeze(1)), dim=1) for i in range(5, 9)], 
        dim=1
    )

    return output


def ConvNormAct(
        inplanes, ouplanes, kernel_size=3, 
        padding=0, stride=1, activate=True
    ):

    block = [nn.Conv2d(inplanes, ouplanes, kernel_size, padding=padding, bias=False, stride=stride)]
    block += [nn.BatchNorm2d(ouplanes)]
    if activate:
        block += [nn.ReLU()]
    
    return nn.Sequential(*block)


def ConvNormAct1D(
        inplanes, ouplanes, kernel_size=3,
        padding=0, stride=1, activate=True
):
    block = [nn.Conv1d(inplanes, ouplanes, kernel_size, padding=padding, bias=False, stride=stride)]
    block += [nn.BatchNorm1d(ouplanes)]
    if activate:
        block += [nn.ReLU()]

    return nn.Sequential(*block)


class ResBlock(nn.Module):

    def __init__(self, inplanes, ouplanes, downsample, stride=1, dropout=0.0):
        super().__init__()

        mdplanes = ouplanes

        self.conv1 = ConvNormAct(inplanes, mdplanes, 3, 1, stride=stride)
        self.conv2 = ConvNormAct(mdplanes, mdplanes, 3, 1)
        self.conv3 = ConvNormAct(mdplanes, ouplanes, 3, 1)

        self.downsample = downsample
        self.drop = nn.Dropout(p=dropout) if dropout > 0. else nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.drop(out)
        identity = self.downsample(x)
        out = out + identity
        return out

class Classifier(nn.Module):

    def __init__(self, inplanes, ouplanes, norm_layer=nn.BatchNorm2d, dropout=0.0, hidreduce=1.0):
        super().__init__()

        midplanes = inplanes // hidreduce

        self.mlp = nn.Sequential(
            nn.Linear(inplanes, midplanes, bias=False),
            norm_layer(midplanes),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(midplanes, ouplanes)
        )

    def forward(self, x):
        return self.mlp(x)


class PredictiveReasoningBlock(nn.Module):

    def __init__(
            self,
            in_planes
    ):
        super().__init__()
        md_planes = in_planes * 4
        num_contexts = 8
        dropout = 0.1
        self.pconv = ConvNormAct1D(in_planes, in_planes, num_contexts)
        self.conv1 = ConvNormAct1D(in_planes, md_planes, 3, 1)
        self.conv2 = ConvNormAct1D(md_planes, in_planes, 3, 1)
        self.conv = nn.Sequential(ConvNormAct1D((num_contexts + 1), (num_contexts + 1) * 4, 3, 1),
                                  ConvNormAct1D((num_contexts + 1) * 4, (num_contexts + 1), 3, 1))
        self.drop = nn.Dropout(dropout) if dropout > .0 else nn.Identity()
        self.lp = nn.Linear(in_planes, in_planes * 2)
        self.m = nn.Linear(in_planes, in_planes)
        self.m1 = nn.Linear(in_planes, in_planes)


    def forward(self, x):
        b, t, c = x.size()
        identity = x
        g, x = self.lp(x).chunk(2, dim=-1)
        g = self.m(self.conv(g))
        x = x.permute(0, 2, 1).contiguous()
        contexts, choices = x[:, :, :t - 1], x[:, :, t - 1:]
        predictions = self.pconv(contexts)
        # prediction_errors = F.relu(choices) - predictions

        out = torch.cat((contexts, predictions), dim=2)
        out = self.m1(out.permute(0, 2, 1).contiguous() * F.gelu(g)).permute(0, 2, 1).contiguous()
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.drop(out)
        out = out.permute(0, 2, 1).contiguous() + identity

        return out

class ReshapeBlock(nn.Module):
    """
    Reshapes a tensor to a given size. Copied from RAISE.
    """
    def __init__(self, size):
        super(ReshapeBlock, self).__init__()
        self.size = size

    def forward(self, x):
        shape = x.size()[:1] + tuple(self.size)
        return x.reshape(shape)

class CNNDecoder(nn.Module):
    """
    The Convolutional Neural Network Decoder from RAISE.
    """
    def __init__(self, input_dim=64, output_dim=1, hidden=None, inner_dim=64):
        super(CNNDecoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        hidden = [64, 64, 32, 32] if hidden is None else hidden
        self.net = nn.ModuleList()
        self.net.append(nn.Sequential(
            ReshapeBlock([input_dim, 1, 1]),
            nn.ConvTranspose2d(input_dim, inner_dim, 1, 1, 0),
            nn.BatchNorm2d(inner_dim),
            nn.LeakyReLU()
        ))
        self.net.append(nn.Sequential(
            nn.ConvTranspose2d(inner_dim, hidden[0], 4, 1, 0),
            nn.BatchNorm2d(hidden[0]),
            nn.LeakyReLU()
        ))
        for in_dim, out_dim in zip(hidden[:-1], hidden[1:]):
            self.net.append(nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 4, 2, 1),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU()
            ))
        self.net.append(nn.Sequential(
            nn.ConvTranspose2d(hidden[-1], output_dim, 4, 2, 1),
            nn.Sigmoid()
        ))

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x

class Adapter(nn.Module):
    """
    The bridge to connect the PredRNet encoder and the RAISE decoder.
    """
    def __init__(self, in_channels=32, in_size=5, out_features=64):
        super().__init__()
        self.in_features = in_channels * in_size * in_size
        self.adapter = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.in_features, self.in_features // 4),
            nn.ReLU(),
            nn.Linear(self.in_features // 4, out_features)
        )

    def forward(self, x):
        return self.adapter(x)
