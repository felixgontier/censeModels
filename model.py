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


class VectorLatentEncoder(nn.Module):
    def __init__(self, settings):
        super().__init__()
        
        nbBlock = settings['model']['encoder']['nb_blocks']
        nbChannels = settings['model']['encoder']['nb_channels']
        textureSize = settings['data']['texture_length']
        embeddingSize = settings['model']['encoder']['embedding_size']
        kernelSize = tuple(settings['model']['encoder']['kernel_size'])
        maxpoolSize = tuple(settings['model']['encoder']['maxpool_size'])
        padding = tuple(settings['model']['encoder']['padding'])
        
        self.layers = nn.ModuleList()
        inChannels = 1
        for b in range(nbBlock):
            outChannels = nbChannels*(2**b)
            self.layers.append(nn.Conv2d(inChannels, outChannels, kernel_size=kernelSize, padding=padding))
            self.layers.append(nn.BatchNorm2d(outChannels))
            self.layers.append(nn.ReLU())
            
            self.layers.append(nn.Conv2d(outChannels, outChannels, kernel_size=kernelSize, padding=padding))
            self.layers.append(nn.BatchNorm2d(outChannels))
            self.layers.append(nn.ReLU())
            if b<nbBlock-1:
                self.layers.append(nn.MaxPool2d(maxpoolSize, stride=maxpoolSize))
            inChannels = outChannels
            
        self.layers.append(nn.MaxPool2d((int(np.maximum(textureSize/2**(nbBlock-1), 1)), int(32/2**(nbBlock-1)))))
        self.outLayer = nn.Linear(inChannels, embeddingSize)
        
    def forward(self, data):
        data = data.unsqueeze(1)
        if data.dim() < 4:
            data = data.unsqueeze(1)
        for layer in self.layers:
            data = layer(data)
        data = self.outLayer(data.squeeze(3).squeeze(2))
        data = F.relu(data)
        return data
        
    def parameter_count(self):
        par = list(self.parameters())
        s = sum([np.prod(list(d.size())) for d in par])
        return s
        
class PresPredCNN(nn.Module):
    def __init__(self, settings):
        super(PresPredCNN, self).__init__()
        self.nClasses = len(settings['data']['classes'])
        
        self.layers = []
        prev_hDim = settings['model']['encoder']['embedding_size']
        for hDim in settings['model']['classifier']['hidden_dimensions']:
            self.layers.append(nn.Linear(prev_hDim, hDim))
            self.layers.append(nn.LeakyReLU())
            prev_hDim = hDim
        self.layers.append(nn.Linear(prev_hDim, self.nClasses))
        
        self.main = nn.Sequential(*self.layers)
        
    def forward(self, x):
        return self.main(x)
        
    def parameter_count(self):
        par = list(self.parameters())
        s = sum([np.prod(list(d.size())) for d in par])
        return s
        
class PresPredRNN(nn.Module):
    def __init__(self, settings, dtype=torch.FloatTensor):
        super(PresPredRNN, self).__init__()
        
        self.nClasses = len(settings['data']['classes'])
        self.dtype = dtype
        self.GRULayer = nn.GRU(input_size=settings['model']['encoder']['embedding_size'], hidden_size=settings['model']['classifier']['hidden_dimensions'], batch_first=True)
        self.FCLayer = nn.Linear(settings['model']['classifier']['hidden_dimensions'], self.nClasses)
        
    def forward(self, x):
        # x in dimension (batch,1,totalFrames,Freq)
        seqLen = x.size(1)
        xpred = torch.zeros((x.size(0), seqLen, self.nClasses)).type(self.dtype) # batch x seq_len x self.nClasses
        
        xseq,_ = self.GRULayer(x)
        for iF in range(seqLen):
            xpred[:, iF, :] = self.FCLayer(F.leaky_relu(xseq[:, iF, :]))
        
        return xpred
        
    def parameter_count(self):
        par = list(self.parameters())
        s = sum([np.prod(list(d.size())) for d in par])
        return s
        
