import sys
import numpy as np

class ThirdOctaveTransform():
    def __init__(self, sr, fLen, hLen):
        # Constants: process parameters
        self.sr = sr
        self.fLen = fLen
        self.hLen = hLen
        
        # Third-octave band analysis weights
        self.f = []
        self.H = []
        with open('tob_'+str(self.fLen)+'.txt') as w_file:
            for line in w_file: # = For each band
                line = line.strip()
                f_temp = line.split(',')
                # Weight array (variable length)
                f_temp = [float(i) for i in f_temp]
                self.H.append(f_temp[2:])
                # Beginning and end indices
                f_temp = [int(i) for i in f_temp]
                self.f.append(f_temp[:2])
                
        # Declarations/Initialisations
        self.w = np.ones(self.fLen)
        self.fft_norm = np.sum(np.square(self.w))/self.fLen
        print([np.sum(h) for h in self.H])
        self.corr_global = 20*np.log10(self.fLen/np.sqrt(2)) # This should be deducted from outputs in wave_to_third_octave, but is instead covered by lvl_offset_db in data_loader

    def wave_to_third_octave(self, x, zeroPad):
        if (x.shape[0]-self.fLen)%self.hLen != 0:
            if zeroPad:
                x = np.append(x, np.zeros(self.hLen-(x.shape[0]-self.fLen)%self.hLen))
            else:
                x = x[:-((x.shape[0]-self.fLen)%self.hLen)]
        
        nFrames = int(np.floor((x.shape[0]-self.fLen)/self.hLen+1));
        
        X_tob = np.zeros((len(self.f), nFrames))
        
        # Process
        for iFrame in range(5,6):#nFrames):
            # Squared magnitude of RFFT
            X = np.fft.rfft(x[iFrame*self.hLen:iFrame*self.hLen+self.fLen]*self.w)
            X = np.square(np.absolute(X))/self.fft_norm
            
            # Third-octave band analysis
            for iBand in range(len(self.f)):
                X_tob[iBand, iFrame] = np.dot(X[self.f[iBand][0]-1:self.f[iBand][1]], self.H[iBand])
                if X_tob[iBand, iFrame] == 0:
                    X_tob[iBand, iFrame] = 1e-15
            # dB, # - self.corr_global
            X_tob[:, iFrame] = 10*np.log10(X_tob[:, iFrame]) - self.corr_global
        return X_tob
        
