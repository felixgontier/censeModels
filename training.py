import torch
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm
import math
import csv
import numpy as np
import os

class PresPredTrainer:
    def __init__(self, settings, enc, dec, modelName, devDataset, valDataset=None, evalDataset=None, dtype=torch.FloatTensor, ltype=torch.LongTensor):
        self.dtype = dtype
        self.ltype = ltype
        
        self.enc = enc
        self.dec = dec
        self.devDataset = devDataset
        self.valDataset = valDataset
        self.evalDataset = evalDataset
        self.tLen = settings['data']['texture_length']
        self.lr = settings['training']['lr']
        self.encOptimizer = optim.Adam(params=self.enc.parameters(), lr=self.lr)
        self.decOptimizer = optim.Adam(params=self.dec.parameters(), lr=self.lr)
        self.modelName = modelName
        self.seqInput = settings['data']['seq_length'] != 1
        
        self.do_validate = settings['workflow']['validate']
        self.optEnc = settings['model']['encoder']['finetune']
        
        self.checkpointDir = settings['model']['checkpoint_dir']
        self.loggingDir = settings['model']['logging_dir']
        self.encEmbeddingSize = settings['model']['encoder']['embedding_size']
        
    def train(self, batchSize=32, epochs=10):
        self.trainDataloader = torch.utils.data.DataLoader(self.devDataset, batch_size=batchSize, shuffle=True, num_workers=8, pin_memory=False, drop_last=False)
        if self.valDataset is not None:
            self.valDataloader = torch.utils.data.DataLoader(self.valDataset, batch_size=batchSize, shuffle=True, num_workers=8, pin_memory=False, drop_last=False)
        losses = []
        lossesVal = []
        
        if epochs>0 and self.do_validate:
            lossVal = self.validate(-1)
            lossesVal.append(lossVal.cpu().numpy())
        
        if self.optEnc:
            self.enc.train()
        else:
            self.enc.eval() # If not optimized, encoder should be in eval mode for BatchNorms
        self.dec.train()
        cur_loss = 0
        for currentEpoch in range(epochs):
            # Training
            with tqdm(self.trainDataloader, desc='Epoch {}, loss: {:.4f}'.format(currentEpoch+1, cur_loss)) as t:
                for currentBatch, (x, p) in enumerate(t):
                    x = x.type(self.dtype)
                    p = p.type(self.dtype)
                    
                    if self.seqInput:
                        r = torch.zeros((x.size(0), x.size(2)-(self.tLen-1), self.encEmbeddingSize)).type(self.dtype) # batch x seq_len x embedding_size
                        for iSeq in range(x.size(2)-(self.tLen-1)):
                            r[:, iSeq, :] = self.enc(x[:, :, iSeq:iSeq+self.tLen, :].squeeze(1))
                    else:
                        r = self.enc(x.squeeze(1))
                    o = self.dec(r)
                    
                    loss = F.binary_cross_entropy_with_logits(o, p.squeeze(1))
                    
                    if self.optEnc:
                        self.encOptimizer.zero_grad()
                    self.decOptimizer.zero_grad()
                    loss.backward()
                    
                    #tqdm.write('Loss is {:.4f}'.format(loss.data))
                    cur_loss = loss.data.cpu().numpy()
                    t.set_description('Epoch {}, loss: {:.4f}'.format(currentEpoch+1, cur_loss))
                    losses.append(cur_loss)
                    
                    if self.optEnc:
                        self.encOptimizer.step()
                    self.decOptimizer.step()
            
            # Validation
            if self.do_validate:
                lossVal = self.validate(currentEpoch)
                lossesVal.append(lossVal.cpu().numpy())
            
            # Save model state
            if not os.path.exists(self.checkpointDir):
                os.makedirs(self.checkpointDir)
            torch.save(self.enc.state_dict(), os.path.join(self.checkpointDir, 'model_' + self.modelName + '_enc_Epoch' + str(currentEpoch+1) + '.pt'))
            torch.save(self.dec.state_dict(), os.path.join(self.checkpointDir, 'model_' + self.modelName + '_dec_Epoch' + str(currentEpoch+1) + '.pt'))
            
        if epochs>0:
            if not os.path.exists(self.loggingDir):
                os.makedirs(self.loggingDir)
            with open(os.path.join(self.loggingDir, 'loss_'+self.modelName+'.txt'), 'w') as lf:
                writer = csv.writer(lf)
                for l in losses:
                    writer.writerow([np.round(l*10000)/10000])
            with open(os.path.join(self.loggingDir, 'lossVal_'+self.modelName+'.txt'), 'w') as lf:
                writer = csv.writer(lf)
                for l in lossesVal:
                    writer.writerow([np.round(l*10000)/10000])
    
    def validate(self, currentEpoch):
        self.enc.eval()
        self.dec.eval()
        lossVal = 0
        for currentBatch, (x, p) in enumerate(tqdm(self.valDataloader, desc='Epoch {}'.format(currentEpoch+1))):
            x = x.type(self.dtype)
            p = p.type(self.dtype)
            
            if self.seqInput:
                r = torch.zeros((x.size(0), x.size(2)-(self.tLen-1), self.encEmbeddingSize)).type(self.dtype) # batch x seq_len x embedding_size
                for iSeq in range(x.size(2)-(self.tLen-1)):
                    r[:, iSeq, :] = self.enc(x[:, :, iSeq:iSeq+self.tLen, :].squeeze(1))
            else:
                r = self.enc(x.squeeze(1))
            o = self.dec(r)
            
            loss = F.binary_cross_entropy_with_logits(o, p.squeeze(1))
            
            lossVal = lossVal + loss.data
        lossVal = lossVal/len(self.valDataloader)
        print(" => Validation loss at epoch {} is {:.4f}".format(currentEpoch+1, lossVal))
        
        if self.optEnc:
            self.enc.train()
        self.dec.train()
        
        return lossVal
        
    def evaluate(self, batchSize=32, classes=None):
        self.enc.eval()
        self.dec.eval()
        self.evalDataloader = torch.utils.data.DataLoader(self.evalDataset, batch_size=batchSize, shuffle=False, num_workers=8, pin_memory=False)
        
        # TODO auto classes
        nClasses = len(classes)
        
        mse_t_pres_val = 0
        # All sources
        pres_acc = float(0)
        n_ex = float(0)
        tp = float(0)
        tn = float(0)
        fp = float(0)
        fn = float(0)
        n_p = float(0)
        n_n = float(0)
        # Source specific
        pres_acc_s = float(0)
        n_ex_s = float(0)
        tp_s = float(0)
        tn_s = float(0)
        fp_s = float(0)
        fn_s = float(0)
        n_p_s = float(0)
        n_n_s = float(0)
        
        for current_batch, (x, p) in enumerate(tqdm(self.evalDataloader, desc='Evaluation')):
            x = x.type(self.dtype)
            p = p.type(self.dtype)
            
            if self.seqInput:
                r = torch.zeros((x.size(0), x.size(2)-(self.tLen-1), self.encEmbeddingSize)).type(self.dtype) # batch x seq_len x embedding_size
                for iSeq in range(x.size(2)-(self.tLen-1)):
                    r[:, iSeq, :] = self.enc(x[:, :, iSeq:iSeq+self.tLen, :].squeeze(1))
            else:
                r = self.enc(x.squeeze())
            o = self.dec(r)
            o = torch.sigmoid(o)
            pres_pred = o[:,:,:nClasses].squeeze().round().data.cpu()
            pres_target = p.squeeze().cpu()
            
            t_pres_pred = torch.mean(pres_pred, dim=0)
            t_pres_target = torch.mean(pres_target, dim=0)
            mse_t_pres = (t_pres_pred-t_pres_target)**2
            mse_t_pres_val = mse_t_pres_val + mse_t_pres
            
            # All sources
            tp = tp + torch.sum((pres_pred==1) & (pres_target==1))
            tn = tn + torch.sum((pres_pred==0) & (pres_target==0))
            fp = fp + torch.sum((pres_pred==1) & (pres_target==0))
            fn = fn + torch.sum((pres_pred==0) & (pres_target==1))
            pres_acc = pres_acc + torch.sum(pres_pred == pres_target)
            n_p = n_p + torch.sum(pres_target==1)
            n_n = n_n + torch.sum(pres_target==0)
            n_ex = n_ex + torch.numel(pres_target)
            
            # Source specific
            tp_s = tp_s + torch.sum((pres_pred==1) & (pres_target==1), dim=0).type(torch.FloatTensor)
            tn_s = tn_s + torch.sum((pres_pred==0) & (pres_target==0), dim=0).type(torch.FloatTensor)
            fp_s = fp_s + torch.sum((pres_pred==1) & (pres_target==0), dim=0).type(torch.FloatTensor)
            fn_s = fn_s + torch.sum((pres_pred==0) & (pres_target==1), dim=0).type(torch.FloatTensor)
            pres_acc_s = pres_acc_s + torch.sum(pres_pred == pres_target, dim=0).type(torch.FloatTensor)
            n_p_s = n_p_s + torch.sum(pres_target==1, dim=0).type(torch.FloatTensor)
            n_n_s = n_n_s + torch.sum(pres_target==0, dim=0).type(torch.FloatTensor)
            n_ex_s = n_ex_s + pres_target.size(0)
            
        mse_t_pres_val = mse_t_pres_val/len(self.evalDataloader)
        print(" => All sources estimated PTP MSE is {:.4f} (RMSE is {:.4f})".format(torch.mean(mse_t_pres_val), math.sqrt(torch.mean(mse_t_pres_val))))
        for iS in range(nClasses):
            print(" => {} estimated PTP MSE is {:.4f} (RMSE is {:.4f})".format(classes[iS], mse_t_pres_val[iS], math.sqrt(mse_t_pres_val[iS])))
        
        # All sources
        pres_acc = pres_acc/n_ex
        
        tp = tp/n_p
        tn = tn/n_n
        fp = fp/n_n
        fn = fn/n_p
        print(" => All sources presence accuracy is {:.2f}%".format(100*pres_acc))
        print(" => All sources tp: {:.2f}%, tn: {:.2f}%, fp: {:.2f}%, fn: {:.2f}%".format(100*tp, 100*tn, 100*fp, 100*fn))
        # Source specific
        pres_acc_s = pres_acc_s/n_ex_s
        tp_s = tp_s/n_p_s
        tn_s = tn_s/n_n_s
        fp_s = fp_s/n_n_s
        fn_s = fn_s/n_p_s
        for iS in range(nClasses):
            print(" => {} presence accuracy is {:.2f}%".format(classes[iS], 100*pres_acc_s[iS]))
            print(" => {} tp: {:.2f}%, tn: {:.2f}%, fp: {:.2f}%, fn: {:.2f}%".format(classes[iS], 100*tp_s[iS], 100*tn_s[iS], 100*fp_s[iS], 100*fn_s[iS]))
        
