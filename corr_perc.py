import numpy as np
import numpy.ma as ma
import csv

def main():
    # Grafic perc.
    g_names = []
    g_tt = []
    g_tv = []
    g_tb = []
    with open('FinalNotesWithWav.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        for r in reader:
            if r[22] != 'NA':
                g_names.append(r[22])
                g_tt.append(float(r[9]) if r[9]!='NA' else np.nan)
                g_tv.append(float(r[10]) if r[10]!='NA' else np.nan)
                g_tb.append(float(r[12]) if r[12]!='NA' else np.nan)
    '''
    print(g_names)
    print(g_tt)
    print(g_tv)
    print(g_tb)
    '''
    # Nantes perc.
    n_names = []
    n_road = []
    n_exvo = []
    n_cavo = []
    n_chvo = []
    n_bird = []
    with open('NantesWithNoiseCapturev2.csv', 'r') as f:
        reader = csv.reader(f, delimiter=';')
        next(reader)
        for r in reader:
            if r[39] != 'NA':
                n_names.append(r[39].split('.')[0])
                n_road.append(float(r[12]) if r[12]!='NA' else np.nan)
                n_exvo.append(float(r[19]) if r[19]!='NA' else np.nan)
                n_cavo.append(float(r[20]) if r[20]!='NA' else np.nan)
                n_chvo.append(float(r[21]) if r[21]!='NA' else np.nan)
                n_bird.append(float(r[24]) if r[24]!='NA' else np.nan)
    '''
    print(n_names)
    print(n_road)
    print(n_exvo)
    print(n_cavo)
    print(n_chvo)
    print(n_bird)
    '''
    # Grafic res
    g_modelnames = []
    with open('graficList.txt', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for r in reader:
            g_modelnames.append(r[0])
    print([n1==n2 for n1,n2 in zip(g_names,g_modelnames)])
    g_tt = np.array(g_tt)
    g_tv = np.array(g_tv)
    g_tb = np.array(g_tb)
    
    res_files = ['Grafic_Fast_TVBCense_Fast_rnn_enc_scratch', 'Grafic_Fast_TVBExtCense_Fast_rnn_enc_scratch', 'Grafic_Fast_TVBNCense_Fast_rnn_enc_scratch', 'Grafic_Slow_TVBCense_Slow_rnn_enc_scratch', 'Grafic_Slow_TVBExtCense_Slow_rnn_enc_scratch', 'Grafic_Slow_TVBNCense_Slow_rnn_enc_scratch']
    for r_file in res_files:
        print(r_file)
        print('--- Presence ---')
        g_res = np.load('eval_outputs/'+r_file+'_presence.npy', allow_pickle=True)
        
        g_mt = []
        g_mv = []
        g_mb = []
        for r in g_res:
            if 'TVBExtCense' in r_file:
                g_mt.append(np.mean((r[:,0]==1)|(r[:,1]==1)|(r[:,2]==1)))
                g_mv.append(np.mean(r[:,3]==1))
                g_mb.append(np.mean((r[:,4]==1)|(r[:,5]==1)))
            else:
                res = np.mean(r, 0)
                g_mt.append(res[0])
                g_mv.append(res[1])
                g_mb.append(res[2])
        g_mt = np.array(g_mt)
        g_mv = np.array(g_mv)
        g_mb = np.array(g_mb)
        
        print(ma.corrcoef(ma.masked_invalid(g_tt), ma.masked_invalid(g_mt)))
        print(ma.corrcoef(ma.masked_invalid(g_tv), ma.masked_invalid(g_mv)))
        print(ma.corrcoef(ma.masked_invalid(g_tb), ma.masked_invalid(g_mb)))
        print('--- Scores ---')
        g_res = np.load('eval_outputs/'+r_file+'_scores.npy', allow_pickle=True)
        
        g_mt = []
        g_mv = []
        g_mb = []
        for r in g_res:
            if 'TVBExtCense' in r_file:
                g_mt.append(np.mean(r[:,0]+r[:,1]+r[:,2]))
                g_mv.append(np.mean(r[:,3]))
                g_mb.append(np.mean(r[:,4]+r[:,5]))
            else:
                res = np.mean(r, 0)
                g_mt.append(res[0])
                g_mv.append(res[1])
                g_mb.append(res[2])
        g_mt = np.array(g_mt)
        g_mv = np.array(g_mv)
        g_mb = np.array(g_mb)
        
        print(ma.corrcoef(ma.masked_invalid(g_tt), ma.masked_invalid(g_mt)))
        print(ma.corrcoef(ma.masked_invalid(g_tv), ma.masked_invalid(g_mv)))
        print(ma.corrcoef(ma.masked_invalid(g_tb), ma.masked_invalid(g_mb)))
        
    
    
    
    # Nantes res
    n_modelnames = []
    with open('fileNames.txt', 'r') as f:
        reader = csv.reader(f, delimiter='.')
        for r in reader:
            n_modelnames.append(r[0])
    id_model = [] # Reorg perc. annotations according to these indices
    id_perc = [] # Reorg model outputs according to these indices
    for iname, name in enumerate(n_modelnames):
        if name in n_names:
            id_model.append(iname)
            id_perc.append(n_names.index(name))
    n_road = [n_road[i] for i in id_perc]
    n_exvo = [n_exvo[i] for i in id_perc]
    n_cavo = [n_cavo[i] for i in id_perc]
    n_chvo = [n_chvo[i] for i in id_perc]
    n_bird = [n_bird[i] for i in id_perc]
    print(len(id_model))
    n_road = np.array(n_road)
    n_exvo = np.array(n_exvo)
    n_cavo = np.array(n_cavo)
    n_chvo = np.array(n_chvo)
    n_bird = np.array(n_bird)
    
    res_files = ['nantes_SlowZeros_TVBCense_SlowZeros100_rnn_enc_scratch', 'nantes_SlowZeros_TVBExtCense_SlowZeros100_rnn_enc_scratch', 'nantes_SlowZeros_TVBNCense_SlowZeros100_rnn_enc_scratch']
    for r_file in res_files:
        print(r_file)
        print('--- Presence ---')
        n_res = np.load('eval_outputs/'+r_file+'_presence.npy', allow_pickle=True)
        
        n_mt = []
        n_mv = []
        n_mb = []
        for r in n_res:
            if 'TVBExtCense' in r_file:
                n_mt.append(np.mean((r[:,0]==1)|(r[:,1]==1)|(r[:,2]==1)))
                n_mv.append(np.mean(r[:,3]==1))
                n_mb.append(np.mean((r[:,4]==1)|(r[:,5]==1)))
            else:
                res = np.mean(r, 0)
                n_mt.append(res[0])
                n_mv.append(res[1])
                n_mb.append(res[2])
        n_mt = np.array([n_mt[i] for i in id_model])
        n_mv = np.array([n_mv[i] for i in id_model])
        n_mb = np.array([n_mb[i] for i in id_model])
        
        print(ma.corrcoef(ma.masked_invalid(n_road), ma.masked_invalid(n_mt)))
        print(ma.corrcoef(ma.masked_invalid(n_exvo), ma.masked_invalid(n_mv)))
        print(ma.corrcoef(ma.masked_invalid(n_cavo), ma.masked_invalid(n_mv)))
        print(ma.corrcoef(ma.masked_invalid(n_chvo), ma.masked_invalid(n_mv)))
        print(ma.corrcoef(ma.masked_invalid(n_bird), ma.masked_invalid(n_mb)))
        print('--- Scores ---')
        n_res = np.load('eval_outputs/'+r_file+'_scores.npy', allow_pickle=True)
        
        n_mt = []
        n_mv = []
        n_mb = []
        for r in n_res:
            if 'TVBExtCense' in r_file:
                n_mt.append(np.mean(r[:,0]+r[:,1]+r[:,2]))
                n_mv.append(np.mean(r[:,3]))
                n_mb.append(np.mean(r[:,4]+r[:,5]))
            else:
                res = np.mean(r, 0)
                n_mt.append(res[0])
                n_mv.append(res[1])
                n_mb.append(res[2])
        n_mt = np.array([n_mt[i] for i in id_model])
        n_mv = np.array([n_mv[i] for i in id_model])
        n_mb = np.array([n_mb[i] for i in id_model])
        
        print(ma.corrcoef(ma.masked_invalid(n_road), ma.masked_invalid(n_mt)))
        print(ma.corrcoef(ma.masked_invalid(n_exvo), ma.masked_invalid(n_mv)))
        print(ma.corrcoef(ma.masked_invalid(n_cavo), ma.masked_invalid(n_mv)))
        print(ma.corrcoef(ma.masked_invalid(n_chvo), ma.masked_invalid(n_mv)))
        print(ma.corrcoef(ma.masked_invalid(n_bird), ma.masked_invalid(n_mb)))
        
    
    
if __name__=='__main__':
    main()



