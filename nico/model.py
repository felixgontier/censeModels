import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(input.shape[0], *self.shape)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, input):
        return input.view(input.shape[0], -1)

class PresPredCNN(nn.Module):
    def __init__(self, nClasses):
        super(PresPredCNN, self).__init__()
        self.nClasses = nClasses
        self.layers = []

        self.layers.append(nn.Linear(128, 128))
        self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.Linear(128, self.nClasses))

        self.main = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.main(x)

    def parameter_count(self):
        par = list(self.parameters())
        s = sum([np.prod(list(d.size())) for d in par])
        return s

class PresPredRNN(nn.Module):
    def __init__(self, nClasses, dType=torch.FloatTensor):
        super(PresPredRNN, self).__init__()

        self.nClasses = nClasses
        self.dType = dType
        self.GRULayer = nn.GRU(input_size=128, hidden_size=128, batch_first=True)
        self.FCLayer = nn.Linear(128, nClasses)


    def forward(self, x):
        # x in dimension (batch,1,totalFrames,Freq)
        seqLen = x.size(1)
        xpred = torch.zeros((x.size(0), seqLen, self.nClasses)).type(self.dType) # batch x seq_len x self.nClasses

        xseq,_ = self.GRULayer(x)
        for iF in range(seqLen):
            xpred[:, iF, :] = self.FCLayer(F.leaky_relu(xseq[:, iF, :]))

        return xpred

    def parameter_count(self):
        par = list(self.parameters())
        s = sum([np.prod(list(d.size())) for d in par])
        return s
