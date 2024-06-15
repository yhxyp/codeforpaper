import numpy as np
import torch.nn as nn
import torchpwl
from scipy.io import loadmat
from os.path import join
import os
from utils.fftc import *
import torch
import torch.nn.functional as F


class ADMMESINetLayer(nn.Module):
    def __init__(
        self,
        L,
        in_channels: int = 1,
        out_channels: int = 64,
        kernel_size: int = 5
    ):
        """
        Args:

        """
        super(ADMMESINetLayer, self).__init__()

        self.rho = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.gamma = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.L = L.to('cuda')
        self.re_org_layer = ReconstructionOriginalLayer(self.rho, self.L)
        self.conv1_layer = ConvolutionLayer1(in_channels, out_channels, kernel_size)
        self.nonlinear_layer = NonlinearLayer()
        self.conv2_Ori_layer = ConvolutionLayer2Ori(out_channels, in_channels, kernel_size)
        self.conv2_Mid_layer = ConvolutionLayer2Mid(out_channels, in_channels, kernel_size)
        self.min_layer = MinusLayer()
        self.multiple_org_layer = MultipleOriginalLayer(self.gamma)
        self.re_update_layer = ReconstructionUpdateLayer(self.rho, self.L)
        self.add_layer = AdditionalLayer()
        self.multiple_update_layer = MultipleUpdateLayer(self.gamma)
        self.re_final_layer = ReconstructionFinalLayer(self.rho, self.L)
        layers = []

        layers.append(self.re_org_layer)
        layers.append(self.conv1_layer)
        layers.append(self.nonlinear_layer)
        layers.append(self.conv2_Ori_layer)
        layers.append(self.min_layer)
        layers.append(self.multiple_org_layer)

        for i in range(5):
            layers.append(self.re_update_layer)
            layers.append(self.add_layer)
            layers.append(self.conv1_layer)
            layers.append(self.nonlinear_layer)
            layers.append(self.conv2_Mid_layer)
            layers.append(self.min_layer)
            layers.append(self.multiple_update_layer)

        layers.append(self.re_update_layer)
        layers.append(self.add_layer)
        layers.append(self.conv1_layer)
        layers.append(self.nonlinear_layer)
        layers.append(self.conv2_Mid_layer)
        layers.append(self.min_layer)
        layers.append(self.multiple_update_layer)

        layers.append(self.re_final_layer)

        self.cs_net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1_layer.conv.weight = torch.nn.init.normal_(self.conv1_layer.conv.weight, mean=0, std=1)
        self.conv2_Ori_layer.conv.weight = torch.nn.init.normal_(self.conv2_Ori_layer.conv.weight, mean=0, std=1)
        self.conv2_Mid_layer.conv.weight = torch.nn.init.normal_(self.conv2_Mid_layer.conv.weight, mean=0, std=1)
        self.conv1_layer.conv.weight.data = self.conv1_layer.conv.weight.data * 0.025
        self.conv2_Ori_layer.conv.weight.data = self.conv2_Ori_layer.conv.weight.data * 0.025
        self.conv2_Mid_layer.conv.weight.data = self.conv2_Mid_layer.conv.weight.data * 0.025


    def forward(self, x):
        x = self.cs_net(x)
        return x


# reconstruction original layers
class ReconstructionOriginalLayer(nn.Module):
    def __init__(self, rho, L):
        super(ReconstructionOriginalLayer, self).__init__()
        self.rho = rho
        self.L = L

    def forward(self, x):

        L = self.L
        rho = self.rho
        orig_output1 = woodbury_inv(L, rho)

        orig_output2 = torch.matmul(L.t(), x)
        orig_output3 = torch.matmul(orig_output1, orig_output2)

        # define data dict
        eeg_data = dict()
        eeg_data['input'] = x
        eeg_data['conv1_input'] = orig_output3
        return eeg_data


# reconstruction middle layers
class ReconstructionUpdateLayer(nn.Module):
    def __init__(self, rho, L):
        super(ReconstructionUpdateLayer, self).__init__()
        self.rho = rho
        self.L = L

    def forward(self, x):

        minus_output = x['minus_output']  # Z
        multiple_output = x['multi_output']  # M
        input = x['input']  # B
        L = self.L
        orig_output1 = woodbury_inv(L, self.rho)
        orig_output2 = torch.matmul(L.t(), input)
        orig_output3 = torch.mul(self.rho, torch.sub(minus_output, multiple_output))
        orig_output4 = torch.add(orig_output2, orig_output3)
        orig_output5 = torch.matmul(orig_output1, orig_output4)
        x['re_mid_output'] = orig_output5
        return x


# reconstruction middle layers
class ReconstructionFinalLayer(nn.Module):
    def __init__(self, rho, L):
        super(ReconstructionFinalLayer, self).__init__()
        self.rho = rho
        self.L = L

    def forward(self, x):

        minus_output = x['minus_output']
        multiple_output = x['multi_output']
        input = x['input']
        L = self.L
        orig_output1 = woodbury_inv(L, self.rho)
        orig_output2 = torch.matmul(L.t(), input)
        orig_output3 = torch.mul(self.rho, torch.sub(minus_output, multiple_output))
        orig_output4 = torch.add(orig_output2, orig_output3)
        orig_output5 = torch.matmul(orig_output1, orig_output4)

        x['re_final_output'] = orig_output5

        return x['re_final_output']


# multiple original layer
class MultipleOriginalLayer(nn.Module):
    def __init__(self, gamma):
        super(MultipleOriginalLayer, self).__init__()
        self.gamma = gamma

    def forward(self, x):

        org_output = x['conv1_input']
        minus_output = x['minus_output']
        output = torch.mul(self.gamma, torch.sub(org_output, minus_output))
        x['multi_output'] = output  # M
        return x


# multiple middle layer
class MultipleUpdateLayer(nn.Module):
    def __init__(self, gamma):
        super(MultipleUpdateLayer, self).__init__()
        self.gamma = gamma

    def forward(self, x):

        multiple_output = x['multi_output']
        re_mid_output = x['re_mid_output']
        minus_output = x['minus_output']
        output = torch.add(multiple_output, torch.mul(self.gamma, torch.sub(re_mid_output, minus_output)))
        x['multi_output'] = output
        return x


# convolution layer
class ConvolutionLayer1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(ConvolutionLayer1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=int((kernel_size - 1) / 2),
                              stride=1, dilation=1, bias=True)

    def forward(self, x):

        conv1_input = x['conv1_input']
        output = self.conv(conv1_input)
        # output = F.relu(output)
        x['conv1_output'] = output
        return x


# convolution layer
class ConvolutionLayer2Ori(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(ConvolutionLayer2Ori, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=int((kernel_size - 1) / 2),
                              stride=1, dilation=1, bias=True)

    def forward(self, x):

        Z = x['conv1_input']
        nonlinear_output = x['nonlinear_output']
        output1 = self.conv(nonlinear_output)
        x['conv2_output'] = output1
        return x

class ConvolutionLayer2Mid(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(ConvolutionLayer2Mid, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=int((kernel_size - 1) / 2),
                              stride=1, dilation=1, bias=True)

    def forward(self, x):

        Z = x['minus_output']
        nonlinear_output = x['nonlinear_output']
        output1 = self.conv(nonlinear_output)
        x['conv2_output'] = output1
        return x


# nonlinear layer
class NonlinearLayer(nn.Module):
    def __init__(self):
        super(NonlinearLayer, self).__init__()
        self.pwl = torchpwl.PWL(num_channels=128, num_breakpoints=101)

    def forward(self, x):

        conv1_output = x['conv1_output']
        output = self.pwl(conv1_output)
        x['nonlinear_output'] = output

        return x


# minus layer   # Z
class MinusLayer(nn.Module):
    def __init__(self):
        super(MinusLayer, self).__init__()

    def forward(self, x):

        minus_input = x['conv1_input']  # addtion layer output   X
        conv2_output = x['conv2_output']
        output = torch.sub(minus_input, conv2_output)
        x['minus_output'] = output  # Z
        return x


# addtional layer
class AdditionalLayer(nn.Module):
    def __init__(self):
        super(AdditionalLayer, self).__init__()

    def forward(self, x):

        mid_output = x['re_mid_output']  # X
        multi_output = x['multi_output']  # M
        output = torch.add(mid_output, multi_output)  # X+M
        x['conv1_input'] = output
        return x

def woodbury_inv(L, rho):
   LT_L = torch.matmul(torch.transpose(L, 0, 1), L)
   rho_I = rho * torch.eye(LT_L.shape[0], device=L.device)
   M = torch.add(LT_L, rho_I)
   result = torch.inverse(M)

   return result

