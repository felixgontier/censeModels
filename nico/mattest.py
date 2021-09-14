import numpy as np
from scipy.io import loadmat
import librosa as lr
from third_octave import ThirdOctaveTransform

def mattest():
    tobt = ThirdOctaveTransform()
    
    mat_levels = loadmat('TVBCense_dev_levels.mat')['sceneLeq']
    x, sr = lr.core.load('noisyStreet01.wav', sr=32000, mono=True)
    
    ref_level = mat_levels[0][0][1][0][0]
    tob = tobt.wave_to_third_octave(x)
    tob1 = 10*np.log10((10**(tob/10)).mean(1))
    
    print(tob1)
    print('Computed Leq: {:.4f}dB'.format(10*np.log10((10**(tob1/10)).sum())))
    print('GT Leq: {:.4f}dB'.format(ref_level))
    print('Difference: {:.4f}dB'.format(10*np.log10((10**(tob1/10)).sum())-ref_level))
    
    x, sr = lr.core.load('noisyStreet02.wav', sr=32000, mono=True)
    
    ref_level = mat_levels[0][0][1][0][1]
    tob = tobt.wave_to_third_octave(x)
    tob1 = 10*np.log10((10**(tob/10)).mean(1))
    
    print(tob1)
    print('Computed Leq: {:.4f}dB'.format(10*np.log10((10**(tob1/10)).sum())))
    print('GT Leq: {:.4f}dB'.format(ref_level))
    print('Difference: {:.4f}dB'.format(10*np.log10((10**(tob1/10)).sum())-ref_level))
    
    x, sr = lr.core.load('noisyStreet03.wav', sr=32000, mono=True)
    
    ref_level = mat_levels[0][0][1][0][2]
    tob = tobt.wave_to_third_octave(x)
    tob1 = 10*np.log10((10**(tob/10)).mean(1))
    
    print(tob1)
    print('Computed Leq: {:.4f}dB'.format(10*np.log10((10**(tob1/10)).sum())))
    print('GT Leq: {:.4f}dB'.format(ref_level))
    print('Difference: {:.4f}dB'.format(10*np.log10((10**(tob1/10)).sum())-ref_level))
    


if __name__=='__main__':
    mattest()


