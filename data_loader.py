import os
import os.path
import math
import threading
import torch
import torch.utils.data
import numpy as np
import csv
import torch.nn.functional as F
import librosa as lr
from third_octave import ThirdOctaveTransform
import sys
np.set_printoptions(threshold=sys.maxsize)

class PresPredDataset(torch.utils.data.Dataset):
    def __init__(self, settings, evalSet=False, subset='train'):
        self.datasetName = settings['eval_dataset_name'] if evalSet else settings['dataset_name']
        self.datasetPath = os.path.join(settings['root_dir'], self.datasetName)
        self.sr = settings['sr']
        self.tLen = settings['texture_length']
        self.fLen = settings['frame_length']
        self.hLen = settings['eval_hop_length'] if evalSet else settings['hop_length']
        self.lvlOffset = settings['level_offset_db']
        #self.valSplit = 0 if evalSet else settings['val_split']
        self.seqLen = settings['eval_seq_length'] if evalSet else settings['seq_length']
        self.classes = settings['classes']
        self.zero_bands_lower = None
        if 'zero_bands_lower' in settings.keys():
            self.zero_bands_lower = settings['zero_bands_lower']
        self.nClasses = len(self.classes)

        if not os.path.isfile(self.datasetPath+'_'+subset+'_tob.npy'):
            assert settings['audio_dir'] is not None, 'No location for dataset audio files specified.'
            if not os.path.exists(settings['root_dir']):
                os.makedirs(settings['root_dir'])

            self.create_dataset(os.path.join(settings['audio_dir'], self.datasetName+'_'+subset), settings['root_dir'], subset=subset, padExamples=settings['pad_examples'])

        self.data_tob = np.load(self.datasetPath+'_'+subset+'_tob.npy', mmap_mode='r')
        self.data_pres = np.load(self.datasetPath+'_'+subset+'_pres.npy', mmap_mode='r')

        self.nFrames = self.data_pres.shape[1]
        self.allowExampleOverlap = settings['allow_example_overlap']

        print('{} dataset {} split length: {}'.format(self.datasetName, subset, self.__len__()))

    def create_dataset(self, location, outputDir, subset='train', padExamples=False):
        print('create dataset from audio files at', location)
        files, pres_files = list_all_audio_files(os.path.join(location, 'sound'), os.path.join(location, 'pres_profile'))

        tob_transform = ThirdOctaveTransform(self.sr, self.fLen, self.hLen)

        # Dummy extraction for dimensions
        x, _ = lr.load(path=files[0], sr=self.sr, mono=True)
        if padExamples:
            nFrames = int(np.ceil(((x.shape[0]-(self.tLen*self.fLen))/self.hLen)+1)) # assume equal for all files
        else:
            nFrames = int(np.floor(((x.shape[0]-(self.tLen*self.fLen))/self.hLen)+1)) # assume equal for all files
        nBands = 29
        nFiles = len(files)

        x_tob = np.zeros((nFiles, nFrames+(self.tLen-1), nBands)) # nFrames+(self.tLen-1) = number of 125ms frames
        x_pres = np.zeros((nFiles, nFrames, self.nClasses))

        for iF, file in enumerate(files):
            print(' -> Processed ' + str(iF) + ' of ' + str(nFiles) + ' files')
            print(pres_files[iF])
            x, _ = lr.load(path=file, sr=self.sr, mono=True)
            x_pres_temp = np.zeros((self.nClasses, nFrames))
            # Presence profile
            for iC in range(self.nClasses):
                with open(pres_files[iF]+'_pp_'+self.classes[iC]+'.csv', newline='') as csvfile:
                    csvreader = csv.reader(csvfile, delimiter=',')
                    for row in csvreader:
                        if len(row) == nFrames:
                            x_pres_temp[iC, :] = np.asarray([[float(l) for l in row]])

            x_tob[iF, :, :] = tob_transform.wave_to_third_octave(x, padExamples).T
            x_pres[iF, :, :] = x_pres_temp.T

        np.save(os.path.join(outputDir, self.datasetName+'_'+subset+'_tob.npy'), x_tob)
        np.save(os.path.join(outputDir, self.datasetName+'_'+subset+'_pres.npy'), x_pres)

    def __getitem__(self, idx):
        if self.seqLen == 1: # CNN
            iFile = int(np.floor(idx/self.nFrames)) # Index of file
            iEx = np.mod(idx, self.nFrames) # Index of exemple within file
            input_x = torch.unsqueeze(torch.from_numpy(self.data_tob[iFile, iEx:iEx+self.tLen, :]), 0)
            pres = torch.unsqueeze(torch.from_numpy(self.data_pres[iFile, iEx, :self.nClasses]), 0)

            return F.pad(input_x+self.lvlOffset, (0, 3)), pres # Pad last dimension (freq) from 29 to 32 for the encoder, plus level correction
        else: # RNN
            if self.allowExampleOverlap:
                iFile = int(np.floor(idx/(self.nFrames-int(np.round(self.fLen/self.hLen))*(self.seqLen-1))))
                iEx = np.mod(idx, self.nFrames-int(np.round(self.fLen/self.hLen))*(self.seqLen-1)) # Index of exemple within file
                input_x = torch.unsqueeze(torch.from_numpy(self.data_tob[iFile, iEx:iEx+int(np.round(self.fLen/self.hLen))*(self.seqLen-1)+1:int(np.round(self.fLen/self.hLen)), :]), 0)
                pres = torch.unsqueeze(torch.from_numpy(self.data_pres[iFile, iEx:iEx+int(np.round(self.fLen/self.hLen))*(self.seqLen-1)+1:int(np.round(self.fLen/self.hLen)), :self.nClasses]), 0)
            else:
                iFile = int(np.floor(idx/int(np.floor(self.nFrames/self.seqLen))))
                iEx = np.mod(idx, int(np.floor(self.nFrames/self.seqLen))) # Index of exemple within file
                input_x = torch.unsqueeze(torch.from_numpy(self.data_tob[iFile, iEx*self.seqLen:(iEx+1)*self.seqLen+(self.tLen-1), :]), 0)
                pres = torch.unsqueeze(torch.from_numpy(self.data_pres[iFile, iEx*self.seqLen:(iEx+1)*self.seqLen, :self.nClasses]), 0)
            #print(input_x)
            # ----- ZEROS ON BANDS LOWER THAN 100Hz ----- TODO
            # input_x = input_x.type(torch.FloatTensor)
            # input_x[:,:,:self.zero_bands_lower] = -self.lvlOffset
            # ----- ZEROS ON BANDS LOWER THAN 100Hz ----- TODO
            return F.pad(input_x+self.lvlOffset, (0, 3)), pres # Pad last dimension (freq) from 29 to 32 for the encoder, plus level correction

    def __len__(self):
        if self.seqLen == 1: # CNN
            return self.data_pres.shape[0]*self.nFrames
        else: # RNN
            if self.allowExampleOverlap:
                return self.data_pres.shape[0]*(self.nFrames-int(np.round(self.fLen/self.hLen))*(self.seqLen-1))
            else:
                return self.data_pres.shape[0]*int(np.floor(self.nFrames/self.seqLen))

def wav_to_npy_no_labels(settings, dataPath, datasetName):
    files, _ = list_all_audio_files(os.path.join(dataPath, datasetName, 'sound'))
    tob_transform = ThirdOctaveTransform(settings['sr'], settings['frame_length'], settings['eval_hop_length'])
    x_tob = []
    for iF, file in enumerate(files):
        print(files[iF])
        x, _ = lr.load(path=file, sr=settings['sr'], mono=True)
        print(x.shape)
        x_tob.append(tob_transform.wave_to_third_octave(x, True).T)
        #print(x_tob[-1])
        #print(x_tob[-1].shape)
        #print(x_tob[-1][5,:]+101)
        print(' -> Processed ' + str(iF+1) + ' of ' + str(len(files)) + ' files')
    np.save(os.path.join(dataPath, datasetName+'_spectralData.npy'), x_tob, allow_pickle=True)

def list_all_audio_files(audio_location, pres_location=None):
    pres_files = []
    audio_files = []
    for dirpath, dirnames, filenames in os.walk(audio_location):
        for filename in [f for f in sorted(filenames) if f.endswith(('.mp3', '.wav', '.aif', 'aiff')) and 'channel' not in f]:
            if pres_location is not None:
                pres_files.append(os.path.join(pres_location, filename[:-4]))
            audio_files.append(os.path.join(dirpath, filename))

    if len(audio_files) == 0:
        print('found no audio files in ' + audio_location)
    return audio_files, pres_files
