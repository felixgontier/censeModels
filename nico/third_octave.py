import sys
import numpy as np

class ThirdOctaveTransform():
    def __init__(self):
        # Constants: process parameters
        self.sr = 32000
        # 125ms, fast
        #self.l_frame = 4096
        #self.l_hop = int(self.l_frame)
        # 1s, slow
        self.l_frame = 32768
        self.l_hop = 32000 # 4000 for train (data aug.), 32000 for eval
        
        # Third-octave band analysis weights
        self.f = []
        self.H = []
        #with open("tob_4096.txt") as w_file:
        with open('tob_'+str(self.l_frame)+'.txt') as w_file:
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
        self.w = np.ones(self.l_frame)
        self.fft_norm = np.sum(np.square(self.w))/self.l_frame
        
        #f = np.linspace(0,16384, 16385)
        #print(10**((np.log10(f[self.f[0][0]])+np.log10(f[self.f[0][1]]))/2))
        
        # ADDED 07/09/2021: Band dependant correction
        self.corr_global = 20*np.log10(self.l_frame/np.sqrt(2)) # TODO TODO TODO THIS WORKS !!! TODO TODO TODO
        print('Leq correction from window size: {:.4f}dB.'.format(self.corr_global))
        
        
    def wave_to_third_octave(self, x):
        if (x.shape[0]-self.l_frame)%self.l_hop != 0:
            x = np.append(x, np.zeros(self.l_hop-(x.shape[0]-self.l_frame)%self.l_hop))
        
        n_frames = int(np.floor((x.shape[0]-self.l_frame)/self.l_hop+1));
        
        X_tob = np.zeros((len(self.f), n_frames))
        # Process
        for ind_frame in range(n_frames):
            # Squared magnitude of RFFT
            #plt.plot(x[ind_frame*self.l_hop:ind_frame*self.l_hop+self.l_frame])
            X = np.fft.rfft(x[ind_frame*self.l_hop:ind_frame*self.l_hop+self.l_frame]*self.w)
            X = np.square(np.absolute(X))/self.fft_norm
            # Third-octave band analysis
            for ind_band in range(len(self.f)):
                X_tob[ind_band, ind_frame] = 0
                X_tob[ind_band, ind_frame] = X_tob[ind_band, ind_frame] + np.dot(X[self.f[ind_band][0]-1:self.f[ind_band][1]], self.H[ind_band])
                if X_tob[ind_band, ind_frame] == 0:
                    X_tob[ind_band, ind_frame] = 1e-15
            # dB
            X_tob[:, ind_frame] = 10*np.log10(X_tob[:, ind_frame])
            
            # ADDED 07/09/2021: Band dependant correction
            X_tob[:, ind_frame] = X_tob[:, ind_frame]-self.corr_global
        return X_tob
        
