import os
import argparse
import sys
import numpy as np


def fast_to_slow(config):
    dataSpec = np.load(os.path.join(config.data_path, config.dataset+'_spectralData.npy'), mmap_mode='r')
    dataPres = np.load(os.path.join(config.data_path, config.dataset+'_presence.npy'), mmap_mode='r')
    dataTimePres = np.load(os.path.join(config.data_path, config.dataset+'_time_of_presence.npy'), mmap_mode='r')
    
    nSlowFrames = int(np.floor(dataPres.shape[1]/8))
    dataSpecSlow = np.zeros((dataSpec.shape[0], nSlowFrames, dataSpec.shape[2]))
    dataPresSlow = np.zeros((dataPres.shape[0], nSlowFrames, dataPres.shape[2]))
    for iF in range(nSlowFrames):
        dataSpecSlow[:,iF,:] = 10*np.log10(np.mean(10**(dataSpec[:,iF*8:(iF+1)*8,:]/10),1)) + 33.96 - 15.90 # Level correction fast to slow, allows use of preset lvl_offset_db in exp. settings
        dataPresSlow[:,iF,:] = np.mean(dataPres[:,iF*8:(iF+1)*8,:],1)>=0.5 # Majority vote
    
    np.save(os.path.join(config.data_path, config.dataset+'_Slow_spectralData.npy'), dataSpecSlow)
    np.save(os.path.join(config.data_path, config.dataset+'_Slow_presence.npy'), dataPresSlow)
    np.save(os.path.join(config.data_path, config.dataset+'_Slow_time_of_presence.npy'), dataTimePres) # Copy
    
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Lorient-1k', help='Evaluation dataset')
    parser.add_argument('--data_path', type=str, default='data', help='Evaluation data path')
    parser.add_argument('-force_recompute', action='store_true')
    config = parser.parse_args()
    
    fast_to_slow(config)


