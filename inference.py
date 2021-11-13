import os
import argparse
import torch
import torch.nn as nn

from model import *
from util import *
import sys
from tqdm import tqdm
from data_loader import wav_to_npy_no_labels

def main(config):
    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)
    
    settings = load_settings(Path('./exp_settings/', config.exp+'.yaml'))
    print(config.dataset+'_spectralData.npy')
    # Load datasets
    if not os.path.exists(os.path.join(config.data_path, config.dataset+'_spectralData.npy')):
        wav_to_npy_no_labels(settings['data'], config.data_path, config.dataset)
    dataSpec = np.load(os.path.join(config.data_path, config.dataset+'_spectralData.npy'), allow_pickle=True)
    try:
        dataPres = np.load(os.path.join(config.data_path, config.dataset+'_presence.npy'), mmap_mode='r')
        dataTimePres = np.load(os.path.join(config.data_path, config.dataset+'_time_of_presence.npy'), mmap_mode='r')
    except FileNotFoundError:
        dataPres = None
        dataTimePres = None

    modelName = get_model_name(settings)
    print('Model: ', modelName)

    presencePath = os.path.join(config.output_path, config.dataset+'_'+modelName+'_presence.npy')
    scoresPath = os.path.join(config.output_path, config.dataset+'_'+modelName+'_scores.npy')
    print(presencePath)
    if not os.path.exists(presencePath) or config.force_recompute:
        useCuda = torch.cuda.is_available() and not settings['training']['force_cpu']
        if useCuda:
            print('Using CUDA.')
            dtype = torch.cuda.FloatTensor
            ltype = torch.cuda.LongTensor
        else:
            print('No CUDA available.')
            dtype = torch.FloatTensor
            ltype = torch.LongTensor

        # Model init.
        enc = VectorLatentEncoder(settings)
        dec = PresPredRNN(settings, dtype=dtype)
        if useCuda:
            enc = nn.DataParallel(enc).cuda()
            dec = nn.DataParallel(dec).cuda()

        # Pretrained state dict. loading
        enc.load_state_dict(load_latest_model_from(settings['model']['checkpoint_dir'], modelName+'_enc', useCuda=useCuda))
        dec.load_state_dict(load_latest_model_from(settings['model']['checkpoint_dir'], modelName+'_dec', useCuda=useCuda))

        print('Encoder: ', enc)
        print('Decoder: ', dec)
        print('Encoder parameter count: ', enc.module.parameter_count() if useCuda else enc.parameter_count())
        print('Decoder parameter count: ', dec.module.parameter_count() if useCuda else dec.parameter_count())
        print('Total parameter count: ', enc.module.parameter_count()+dec.module.parameter_count() if useCuda else enc.parameter_count()+dec.parameter_count())

        enc.eval()
        dec.eval()

        # presence = np.zeros((dataSpec.shape[0], (dataSpec.shape[1] if 'Slow' in config.exp else dataSpec.shape[1]-7), len(settings['data']['classes'])))
        # scores = np.zeros(presence.shape)
        presence = []
        scores = []
        for k in tqdm(range(len(dataSpec))):
            x = torch.Tensor(dataSpec[k]).type(dtype)
            x = F.pad(x.unsqueeze(0).unsqueeze(0)+settings['data']['level_offset_db'], (0, 3))
            if useCuda:
                x = x.cuda()
            if 'Slow' in config.exp:
                encData = torch.zeros((x.size(0), x.size(2), 128)).type(dtype) # batch x seq_len x embedding_size
                for iSeq in range(x.size(2)):
                    encData[:, iSeq, :] = enc(x[:, :, iSeq, :].squeeze(1))
                score = torch.sigmoid(dec(encData))
                #print(torch.sum(score))
            else:
                encData = torch.zeros((x.size(0), x.size(2)-7, 128)).type(dtype) # batch x seq_len x embedding_size
                for iSeq in range(x.size(2)-7):
                    encData[:, iSeq, :] = enc(x[:, :, iSeq:iSeq+8, :].squeeze(1))
                score = torch.sigmoid(dec(encData))
            scores.append(np.array(score.squeeze().cpu().data))
            presence.append(np.array(score.squeeze().round().cpu().data)) # TODO threshold
        np.save(presencePath, np.array(presence, dtype=object), allow_pickle=True)
        np.save(scoresPath, np.array(scores, dtype=object), allow_pickle=True)
    else:
        presence = np.load(presencePath, allow_pickle=True)

    if not config.no_metrics and dataPres is not None:
        presence=np.stack(presence)
        if 'Ext' in config.exp:
            prediction[k, :, 0] = presence[:,0] + presence[:,1] + presence[:,2]
            prediction[k, :, 1] = presence[:,3]
            prediction[k, :, 2] = presence[:,4] + presence[:,5]
            prediction = prediction >= 1
        else:
            prediction = presence[:,:,:dataPres.shape[2]]
        metricsPath = os.path.join(config.output_path, modelName)
        # Metrics
        reference = dataPres
        np.save(metricsPath+'_tppSe.npy', np.mean((np.mean(prediction, axis=1)-dataTimePres).flatten()**2))

        np.save(metricsPath+'_accuracy.npy', (prediction==reference).flatten())

        np.save(metricsPath+'_truePositive.npy', np.sum((prediction==1) & (reference==1))/np.sum(reference==1))
        np.save(metricsPath+'_trueNegative.npy', np.sum((prediction==0) & (reference==0))/np.sum(reference==0))
        np.save(metricsPath+'_falsePositive.npy', np.sum((prediction==1) & (reference==0))/np.sum(reference==0))
        np.save(metricsPath+'_falseNegative.npy', np.sum((prediction==0) & (reference==1))/np.sum(reference==1))
        print('Overall metrics')
        print(' - Accuracy:             {:.4f}'.format(np.mean((prediction==reference).flatten())))
        print(' - True positive rate:   {:.4f}'.format(np.sum((prediction==1) & (reference==1))/np.sum(reference==1)))
        print(' - True negative rate:   {:.4f}'.format(np.sum((prediction==0) & (reference==0))/np.sum(reference==0)))
        print(' - False positive rate:  {:.4f}'.format(np.sum((prediction==1) & (reference==0))/np.sum(reference==0)))
        print(' - False negative rate:  {:.4f}'.format(np.sum((prediction==0) & (reference==1))/np.sum(reference==1)))
        print(' - Time of presence MSE: {:.4f}'.format(np.mean((np.mean(prediction, axis=1)-dataTimePres).flatten()**2)))
        sources = ['traffic', 'voice', 'bird']

        tp_s = np.sum(np.sum((prediction==1) & (reference==1), axis=0), axis=0)/np.sum(np.sum(reference==1, axis=0), axis=0)
        tn_s = np.sum(np.sum((prediction==0) & (reference==0), axis=0), axis=0)/np.sum(np.sum(reference==0, axis=0), axis=0)
        fp_s = np.sum(np.sum((prediction==1) & (reference==0), axis=0), axis=0)/np.sum(np.sum(reference==0, axis=0), axis=0)
        fn_s = np.sum(np.sum((prediction==0) & (reference==1), axis=0), axis=0)/np.sum(np.sum(reference==1, axis=0), axis=0)
        act_s = np.mean(np.mean(reference==1, axis=0), axis=0)

        for si, s in enumerate(sources):
            print('Metrics for source {}'.format(s))
            np.save(metricsPath+'_'+s+'Accuracy.npy', (prediction[:, :, si]==reference[:, :, si]).flatten()) # Normal accuracy
            np.save(metricsPath+'_'+s+'TppSe.npy', (np.mean(prediction[:, :, si], axis=1)-dataTimePres[:, si])**2)
            # Source specific accuracies, no conf. weighting implemented
            np.save(metricsPath+'_'+s+'TruePositive.npy',tp_s[si])
            np.save(metricsPath+'_'+s+'TrueNegative.npy',tn_s[si])
            np.save(metricsPath+'_'+s+'FalsePositive.npy',fp_s[si])
            np.save(metricsPath+'_'+s+'FalseNegative.npy',fn_s[si])
            print(' - Accuracy:             {:.4f}'.format(np.mean((prediction[:, :, si]==reference[:, :, si]).flatten())))
            print(' - True positive rate:   {:.4f}'.format(tp_s[si]))
            print(' - True negative rate:   {:.4f}'.format(tn_s[si]))
            print(' - False positive rate:  {:.4f}'.format(fp_s[si]))
            print(' - False negative rate:  {:.4f}'.format(fn_s[si]))
            print(' - Time of presence MSE: {:.4f}'.format(np.mean((np.mean(prediction[:, :, si], axis=1)-dataTimePres[:, si])**2)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default=None, help='Experience settings YAML, or oracle, chance, null, yamnet')
    parser.add_argument('--dataset', type=str, default='Lorient-1k', help='Evaluation dataset')
    parser.add_argument('--data_path', type=str, default='data', help='Evaluation data path')
    parser.add_argument('--output_path', type=str, default='eval_outputs', help='Evaluation output path')
    parser.add_argument('-force_recompute', action='store_true')
    parser.add_argument('-no_metrics', action='store_true')
    config = parser.parse_args()

    main(config)
