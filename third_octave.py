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
        #with open("tob_4096.txt") as w_file:
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
        
        #ascale = [0.58365, 0.734975, 0.9242703000000001, 1.1694339999999999, 1.4709270968, 1.8521412247002582, 2.3312225506999997, 2.9415900080735775, 3.7076479767054016, 4.66801717014583, 5.878790866480005, 7.410143942727, 9.335937275677011, 11.761727755082788, 14.82030789747725, 18.671879276857887, 23.527326782529876, 29.64061304725454, 37.343764388048584, 47.050789093774796, 59.281217924104396, 74.68748099932147, 94.10157552172242, 118.5586026958666, 149.37890492064477, 188.2031011341576, 237.12115575518948, 298.7539711491687, 376.40627121318505]
        #bscale = [4.656804116366291, 5.874077400553002, 7.406313182356, 9.343694979256009, 11.781264374058813, 14.81250665645844, 18.65626255772545, 23.5312592017711, 29.65625033447279, 37.343740097618294, 47.03127296963517, 59.281271267263136, 74.68752057973502, 94.09373929593846, 118.56253023891097, 149.37499595793605, 188.21876798191647, 237.12498534301284, 298.7499920821549, 376.4063522727558, 474.24999964217704, 597.5000156967055, 752.8124949874943, 948.4687546697835, 1195.0312688090885, 1505.6251096534006, 1896.9687656085348, 2390.03147364318, 3011.2502430031886]
        
    def wave_to_third_octave(self, x, zeroPad):
        if (x.shape[0]-self.fLen)%self.hLen != 0:
            if zeroPad:
                x = np.append(x, np.zeros(self.hLen-(x.shape[0]-self.fLen)%self.hLen))
            else:
                x = x[:-((x.shape[0]-self.fLen)%self.hLen)]
        
        nFrames = int(np.floor((x.shape[0]-self.fLen)/self.hLen+1));
        #x = x[:, np.newaxis]
        X_tob = np.zeros((len(self.f), nFrames))
        X_tob2 = np.zeros((len(self.f), nFrames))
        
        
        #H_temp = np.zeros((int(self.fLen/2+1), len(self.f)))
        #for iBand in range(len(self.f)):
        #    H_temp[self.f[iBand][0]-1:self.f[iBand][1], iBand] = self.H[iBand]
        #print(np.sum(H_temp, axis=1))
        
        # Process
        for iFrame in range(5,6):#nFrames):
            #print(10*np.log10(np.sum((x[iFrame*self.hLen:iFrame*self.hLen+self.fLen,:]**2)/self.fLen)))
            # Squared magnitude of RFFT
            #X = np.fft.rfft((x[iFrame*self.hLen:iFrame*self.hLen+self.fLen,:].T*self.w).T)
            print(x.shape)
            X = np.fft.rfft(x[iFrame*self.hLen:iFrame*self.hLen+self.fLen]*self.w)
            X = np.square(np.absolute(X))/self.fft_norm
            print(X.shape)
            #X = np.squeeze(np.square(np.abs(X)))/self.fft_norm/self.fLen # ADDED
            #print(10*np.log10(np.sum(X)))
            
            for iBand in range(len(self.f)):
                X_tob[iBand, iFrame] = 0
                X_tob[iBand, iFrame] = X_tob[iBand, iFrame] + np.dot(X[self.f[iBand][0]-1:self.f[iBand][1]], self.H[iBand])
                if X_tob[iBand, iFrame] == 0:
                    X_tob[iBand, iFrame] = 1e-15
                
            
            # Third-octave band analysis
            for iBand in range(len(self.f)):
                X_tob2[iBand, iFrame] = np.dot(X[self.f[iBand][0]-1:self.f[iBand][1]], self.H[iBand])
                if X_tob2[iBand, iFrame] == 0:
                    X_tob2[iBand, iFrame] = 1e-15
            # dB
            print(10*np.log10(X_tob[:,iFrame]))
            print(10*np.log10(X_tob2[:,iFrame]))
            #print(10*np.log10(np.sum(X_tob[:,iFrame])))
            
            X_tob[:, iFrame] = 10*np.log10(X_tob[:, iFrame])
        #print(10*np.log10(np.mean(np.sum(np.power(10, X_tob/10), axis=0))))
        return X_tob
        
