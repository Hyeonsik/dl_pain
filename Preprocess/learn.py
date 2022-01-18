import itertools as it
import os
import time
from datetime import datetime
import pandas as pd
import numpy as np
import pandas as pd
import random
import vitaldb
from pyvital2 import arr
import pickle
import matplotlib.pyplot as plt
import scipy.stats


BATCH_SIZE = 1024
MAX_CASES = 2000
SEGLEN_IN_SEC = 20
SRATE = 100
LEN_INPUT = 20
OVERLAP = 10
LEN_PER_PRE = 60
LEN_PER_POST = 120


def load_vital_data(file_path):
    print('loading vital data...')
    # tracks to extract / VENT_SET_TV -> VENT_INSP_TM, SET_INSP_TM
    track_names = ["SNUADC/ECG_II", "SNUADC/PLETH", "Solar8000/VENT_INSP_TM", "Primus/SET_INSP_TM", "Orchestra/PPF20_CE", "Orchestra/RFTN20_CE", "Solar8000/NIBP_MBP", "Solar8000/ART_MBP", "Solar8000/HR"]


    # create saving folder
    #file_path = "vital_to_np"
    if not os.path.exists(file_path):
        os.mkdir(file_path)


    # dataframe of patient information    
    df = pd.read_csv("https://api.vitaldb.net/cases")

    
    # target patients' caseids
    caseids = list(vitaldb.caseids_tiva & set(df.loc[df['ane_type'] == 'General', 'caseid']))


    cnt = 0
    for caseid in caseids:
        cnt = cnt + 1
        print(f'{cnt}/{len(caseids)}({caseid})', end='...')

        # check if file is already existing
        filename = f'{file_path}/{caseid}.npz'
        if os.path.isfile(filename):
            print('already existing')
            continue


        # get vital file and save as numpy
        vf = vitaldb.VitalFile(caseid, track_names)
        vals = vf.to_numpy(track_names, interval=1/SRATE)

        # intubation time - find the first t which satisfies vent_set_tm != nan & ppf_ce != nan
        t_intu = np.where(~np.isnan(vals[:,5]))[0][0]

        if not np.mean(~np.isnan(vals[:,2])):
            if not np.mean(~np.isnan(vals[:,3])):
                print(f'no valid data for insp_tm')
                continue
            intu = vals[:,3]
            intv = 850 # maximum interval for "Primus/SET_INSP_TM"
        else:
            intu = vals[:,2]
            intv = 250 # maximum interval for "Solar8000/VENT_INSP_TM"
        
        idc_intu = np.where(~np.isnan(intu))[0]
        while True:
            # vent_insp_tm이 nan이 아닌 경우
            if not np.isnan(intu[t_intu]):
                print(t_intu, end=' ')
                idx = np.where(idc_intu==t_intu)[0][0]
                prev = t_intu

                switch = True
                for i in range(1,11):
                    if idc_intu[idx+i] - prev > intv:
                        switch = False
                        t_intu = t_intu + 1
                    prev = idc_intu[idx+i]
                if switch:
                    break
            else:
                t_intu = t_intu + 1
            
        # MBP value
        if not np.mean(~np.isnan(vals[:,6])):
            if not np.mean(~np.isnan(vals[:,7])):
                print(f'no valid data for MBP')
        mbp = np.array([art[i] if art[i]>30 else nibp[i] for i in range(len(nibp))])
                   
        # HR
        if not np.mean(~np.isnan(vals[:,8])):
            print('no valid data for HR')
        hr = vals[:,8]
                        
        # non-event data : extract vital from previous 120s-60s from intubation
        ppg = vals[:,1]
        prev_ppg = ppg[t_intu - SRATE*120:t_intu - SRATE*60]

        ecg = vals[:,0]
        prev_ecg = ecg[t_intu - SRATE*120:t_intu - SRATE*60]

        nmbp = mbp[t_intu - SRATE*120:t_intu - SRATE*60]
        nhr = hr[t_intu - SRATE*120:t_intu - SRATE*60]

        # after intubation, pain calculated using TSS, CISA
        post_ppg = ppg[t_intu:t_intu + SRATE*LEN_PER_POST]
        post_ecg = ecg[t_intu:t_intu + SRATE*LEN_PER_POST]
        
        ppf = vals[:,4]
        ppf = ppf[t_intu:t_intu + SRATE*LEN_PER_POST]

        rftn = vals[:,5]
        rftn = rftn[t_intu:t_intu + SRATE*LEN_PER_POST]
    
        embp = mbp[t_intu:t_intu + SRATE*LEN_PER_POST]
        ehr = hr[t_intu:t_intu + SRATE*LEN_PER_POST]
    
        np.savez(filename, nECG=prev_ecg, nPPG=prev_ppg, ECG=post_ecg, PPG=post_ppg, PPF=ppf, RFTN=rftn, nMBP=nmbp, MBP=embp, nHR=nhr, HR=ehr)
        print('  completed')
    
    
 # 피크 사이 wave를 모두 같은 length로 만들기 위한 함수
def linear_connection(list, idx):
    int_idx = int(idx)
    return list[int_idx] + (list[int_idx+1] - list[int_idx]) * (idx - int_idx)


def preprocess(file_path):
    ### file_path : path for inputs extracted from vital file ###
    ### LEN_INPUT : length of input, OVERLAP : overlap of each input, SRATE : sampling rate from vital data ###

    # path for cache
    if not os.path.exists('./cache'):
        os.mkdir('./cache')
    if not os.path.exists('./cache/peaks'):
        os.mkdir('./cache/peaks')
    if not os.path.exists(f"cache/peaks/PPG_{SRATE}Hz_1min_seg"):
        os.mkdir(f"cache/peaks/PPG_{SRATE}Hz_1min_seg")
    if not os.path.exists(f"cache/peaks/ECG_{SRATE}Hz_1min_seg"):
        os.mkdir(f"cache/peaks/ECG_{SRATE}Hz_1min_seg")        
    if not os.path.exists('./cache/preprocess'):
        os.mkdir('./cache/preprocess')
    
    
    # dataframe to save preprocessing info
    n_aug = int((LEN_PER_PRE-LEN_INPUT)/OVERLAP) + 1   # number of data augmentation
    n_aug2 = int((LEN_PER_POST-LEN_INPUT)/OVERLAP) + 1
    column_list = ['caseid'] + [str(i+1) for i in range(n_aug+n_aug2)]
    df_preprocess = pd.DataFrame(columns = column_list)


    # set variables
    caseids = os.listdir(file_path)
    error_list = []
    f_num = 0
    initial = f_num
    interval = len(caseids)

   
    for caseid in caseids[initial:initial+interval]:
        caseid = caseid[:-4]
        f_num += 1
        print('\n###Input', f_num,'/ '+str(len(caseids))+': '+caseid+'###')


        # vital data 불러오기    
        vals = np.load(f'{file_path}/{caseid}.npz')


        #dataframe에 새로운 행 만들기
        df_preprocess.loc[f_num-1,'caseid'] = caseid

        ppg_cache = f"cache/peaks/PPG_{SRATE}Hz_1min_seg/" + caseid
        ecg_cache = f"cache/peaks/ECG_{SRATE}Hz_1min_seg/" + caseid    
        ecg_cache2 = f"cache/peaks/ECG_{SRATE}Hz_1min_seg/" + caseid


        # 20초 단위로 끊기
        for i in range(n_aug):
            print('  segment', i+1, end='')
            start_idx = i*OVERLAP*SRATE # 500i
            end_idx = (i*OVERLAP + LEN_INPUT)*SRATE # 500i + 1000


            ### non-event input ###
            seg_ppg = vals['nPPG'][start_idx:end_idx]
            seg_ecg = vals['nECG'][start_idx:end_idx]


            ## 1. 결측치 처리 ##             
            # df.isnull().sum() 하면 더 간단하게 가능하나 애초에 NRS에 해당하는 vital data가 120초 보다 짧은 경우
            nan_ppg_list = np.isnan(seg_ppg)
            nan_ecg_list = np.isnan(seg_ecg)
            nan_ppg_perc = np.sum(nan_ppg_list) / LEN_INPUT / SRATE
            nan_ecg_perc = np.sum(nan_ecg_list) / LEN_INPUT / SRATE

            # ECG, PPG 둘다 결측치인 부분
            nan_both_perc = 0
            for j in range(len(seg_ppg)):
                if nan_ppg_list[j] and  nan_ecg_list[j]:
                    nan_both_perc += 1
            nan_both_perc /= (LEN_INPUT*SRATE)

            # segment의 결측치 비율 정보
            nan_info = [nan_ppg_perc, nan_ecg_perc, nan_both_perc]

            # 결측치가 많은 경우, noise 확인할 것도 없이 False -  이 경우의 noise_info는 -1로 처리
            if nan_ppg_perc > 0.05 or nan_ecg_perc > 0.05 or nan_both_perc > 0.05:
                df_preprocess.loc[f_num-1,str(i+1)] = (False, nan_info, [-1, -1])
                print(' too much missing data', end='...')
                continue


            ## 2. Noise 처리 ##
            # peak detection
            if os.path.exists(ppg_cache+'_n{}'.format(i+1)):
                _, ppg_peak = pickle.load(open(ppg_cache+'_n{}'.format(i+1), 'rb'))
                ecg_peak = pickle.load(open(ecg_cache+'_n{}'.format(i+1), 'rb'))
                print('...loaded peak...', end='')


            else:
                try:
                    min_peak, ppg_peak = arr.detect_peaks(pd.DataFrame(seg_ppg).fillna(method='ffill', axis=0).fillna(method='bfill', axis=0).values.flatten(), SRATE)
                    ecg_peak = arr.detect_qrs(pd.DataFrame(seg_ecg).fillna(method='ffill', axis=0).fillna(method='bfill', axis=0).values.flatten(), SRATE)


                except Exception as e:
                    print('error of', e)
                    error_list.append(caseid)
                    df_preprocess.loc[f_num-1,str(i+1)] = (False, nan_info, [-3, -3])
                    continue


                if len(ppg_peak)==0:
                    print('no peak')


                pickle.dump((min_peak, ppg_peak), open(ppg_cache+'_n{}'.format(i+1), 'wb'))
                pickle.dump(ecg_peak, open(ecg_cache+'_n{}'.format(i+1), 'wb'))
                print('...saved peak...', end='')


            # 10초 segment 내의 ppg, ecg peak idx
            #seg_ppg_min = ppg_min[(start_idx<=np.array(ppg_min)) & (np.array(ppg_min)<end_idx)]
            idx_ppg_peak = ppg_peak
            idx_ecg_peak = ecg_peak


            # peak가 HR 30~150 -> 20s - min 10 peaks(HR30)
            # peak 개수가 기준 미달이면 noise 계산 자세히 할 필요없이 False - 이 경우의 noise_info는 -2로 처리
            if len(idx_ppg_peak)<5/10*LEN_INPUT or len(idx_ecg_peak)<5/10*LEN_INPUT:
                df_preprocess.loc[f_num-1,str(i+1)] = (False, nan_info, [-2, -2])
                print(' too less peaks', end='')
                continue


            # 20초 segment 내의 ppg, ecg peak value
            #print(len(seg_ppg), idx_ppg_peak)
            val_ppg_peak = [seg_ppg[k] for k in idx_ppg_peak]
            val_ecg_peak = [seg_ecg[k] for k in idx_ecg_peak]

            # peak와 peak 사이 interval에 대한 noise 여부 -> 따라서 길이는 peak - 1
            bool_noise_ppg = [False for k in range(len(idx_ppg_peak)-1)]
            bool_noise_ecg = [False for k in range(len(idx_ecg_peak)-1)]


            #  2.1 peak 간격 이상한 noise (HR 30~150 -> HBI 0.4s ~ 2s로 SRATE 곱해주면 40~200)
            for k in range(len(bool_noise_ppg)):
                if not 0.4*SRATE < idx_ppg_peak[k+1] - idx_ppg_peak[k] < 2*SRATE:
                    bool_noise_ppg[k] = True
            for k in range(len(bool_noise_ecg)):
                if not 0.4*SRATE < idx_ecg_peak[k+1] - idx_ecg_peak[k] < 2*SRATE:
                    bool_noise_ecg[k] = True


            # 2.2 모양 이상한 noise
            # wave interval into same length(2s(200))
            len_wave = 2*SRATE
            norm_seg_ppg, norm_seg_ecg = [], []

            for k in range(len(bool_noise_ppg)):
                len_interval_ppg = idx_ppg_peak[k+1] - idx_ppg_peak[k]

                # peak 사이 wave를 모두 같은 길이로 변환
                norm_seg_ppg.append([linear_connection(seg_ppg[idx_ppg_peak[k]:idx_ppg_peak[k+1]+1], n/len_wave*len_interval_ppg) for n in range(len_wave)])

            for k in range(len(bool_noise_ecg)):
                len_interval_ecg = idx_ecg_peak[k+1] - idx_ecg_peak[k]

                # peak 사이 wave를 모두 같은 길이로 변환
                norm_seg_ecg.append([linear_connection(seg_ecg[idx_ecg_peak[k]:idx_ecg_peak[k+1]+1], n/len_wave*len_interval_ecg) for n in range(len_wave)])


            # wave interval 사이 correlation 계산 - PPG
            mean_wave_ppg = np.nanmean(norm_seg_ppg, axis = 0)
            mean_wave_ppg = pd.DataFrame(mean_wave_ppg).fillna(method='ffill', axis=0).fillna(method='bfill', axis=0).values.flatten()
            norm_seg_ppg = pd.DataFrame(norm_seg_ppg).fillna(method='ffill', axis=1).fillna(method='bfill', axis=1).values
            for k in range(len(bool_noise_ppg)):
                if np.corrcoef(norm_seg_ppg[k], mean_wave_ppg)[0,1] < 0.9:
                    bool_noise_ppg[k] = True
            noise_ppg_perc = np.sum(bool_noise_ppg) / len(bool_noise_ppg)

            # wave interval 사이 correlation 계산 - ECG                
            mean_wave_ecg = np.nanmean(norm_seg_ecg, axis = 0)
            mean_wave_ecg = pd.DataFrame(mean_wave_ecg).fillna(method='ffill', axis=0).fillna(method='bfill', axis=0).values.flatten()
            norm_seg_ecg = pd.DataFrame(norm_seg_ecg).fillna(method='ffill', axis=1).fillna(method='bfill', axis=1).values
            for k in range(len(bool_noise_ecg)):
                if np.corrcoef(norm_seg_ecg[k], mean_wave_ecg)[0,1] < 0.9:
                    bool_noise_ecg[k] = True
            noise_ecg_perc = np.sum(bool_noise_ecg) / len(bool_noise_ecg)

            # segment의 noise 비율 정보
            noise_info = [noise_ppg_perc, noise_ecg_perc]

            # segment를 input으로 써도 되는지
            if nan_ppg_perc < 0.05 and nan_ecg_perc < 0.05 and nan_both_perc < 0.05 and noise_ppg_perc < 0.1 and noise_ecg_perc < 0.1:
                bool_pass = True
            else:
                bool_pass = False

            # 이 segment의 정보를 dataframe에 저장 - (전처리 성공여부, 전처리 nan 비율, 전처리 noise 비율, 통증 점수)
            arry = np.empty(1, dtype=object)
            arry[0] = [bool_pass, nan_info, noise_info, 0, 0]
            df_preprocess.loc[f_num-1,f'{i+1}'] = arry[0] #{'pass':bool_pass, 'nan_perc':nan_info, 'noise_perc':noise_info, 'tss':0, 'cisa':0}        
            print('preprocessing done...', end='')
            ##########################################################################

            
        for i in range(n_aug2):
            print('  segment', i+1, end='')
            start_idx = i*OVERLAP*SRATE # 500i
            end_idx = (i*OVERLAP + LEN_INPUT)*SRATE # 500i + 1000

            
            ### event input ###
            seg_ppg = vals['PPG'][start_idx:end_idx]
            seg_ecg = vals['ECG'][start_idx:end_idx]


            ## 1. 결측치 처리 ##              
            # df.isnull().sum() 하면 더 간단하게 가능하나 애초에 NRS에 해당하는 vital data가 120초 보다 짧은 경우
            nan_ppg_list = np.isnan(seg_ppg)
            nan_ecg_list = np.isnan(seg_ecg)
            nan_ppg_perc = np.sum(nan_ppg_list) / LEN_INPUT / SRATE
            nan_ecg_perc = np.sum(nan_ecg_list) / LEN_INPUT / SRATE

            # ECG, PPG 둘다 결측치인 부분
            nan_both_perc = 0
            for j in range(len(seg_ppg)):
                if nan_ppg_list[j] and  nan_ecg_list[j]:
                    nan_both_perc += 1
            nan_both_perc /= (LEN_INPUT*SRATE)

            # segment의 결측치 비율 정보
            nan_info = [nan_ppg_perc, nan_ecg_perc, nan_both_perc]

            # 결측치가 많은 경우, noise 확인할 것도 없이 False -  이 경우의 noise_info는 -1로 처리
            if nan_ppg_perc > 0.05 or nan_ecg_perc > 0.05 or nan_both_perc > 0.05:
                df_preprocess.loc[f_num-1,str(i+n_aug+1)] = (False, nan_info, [-1, -1])
                print(' too much missing data', end='...')
                continue


            ## 2. Noise 처리 ##
            # peak detection
            if os.path.exists(ppg_cache+'_{}'.format(i+1)):
                _, ppg_peak = pickle.load(open(ppg_cache+'_{}'.format(i+1), 'rb'))
                ecg_peak = pickle.load(open(ecg_cache+'_{}'.format(i+1), 'rb'))
                print('...loaded peak...', end='')


            else:
                try:
                    min_peak, ppg_peak = arr.detect_peaks(pd.DataFrame(seg_ppg).fillna(method='ffill', axis=0).fillna(method='bfill', axis=0).values.flatten(), SRATE)
                    ecg_peak = arr.detect_qrs(pd.DataFrame(seg_ecg).fillna(method='ffill', axis=0).fillna(method='bfill', axis=0).values.flatten(), SRATE)


                except Exception as e:
                    print('error of', e)
                    error_list.append(caseid)
                    df_preprocess.loc[f_num-1,str(i+n_aug+1)] = (False, nan_info, [-3, -3])
                    continue


                if len(ppg_peak)==0:
                    print('no peak')


                pickle.dump((min_peak, ppg_peak), open(ppg_cache+'_{}'.format(i+1), 'wb'))
                pickle.dump(ecg_peak, open(ecg_cache+'_{}'.format(i+1), 'wb'))
                print('...saved peak...', end='')


            # 10초 segment 내의 ppg, ecg peak idx
            #seg_ppg_min = ppg_min[(start_idx<=np.array(ppg_min)) & (np.array(ppg_min)<end_idx)]
            idx_ppg_peak = ppg_peak
            idx_ecg_peak = ecg_peak


            # peak가 HR 30~150 -> 20s - min 10 peaks(HR30)
            # peak 개수가 기준 미달이면 noise 계산 자세히 할 필요없이 False - 이 경우의 noise_info는 -2로 처리
            if len(idx_ppg_peak)<5/10*LEN_INPUT or len(idx_ecg_peak)<5/10*LEN_INPUT:
                df_preprocess.loc[f_num-1,str(i+n_aug+1)] = (False, nan_info, [-2, -2])
                print(' too less peaks', end='...')
                continue


            # 20초 segment 내의 ppg, ecg peak value
            #print(len(seg_ppg), idx_ppg_peak)
            val_ppg_peak = [seg_ppg[k] for k in idx_ppg_peak]
            val_ecg_peak = [seg_ecg[k] for k in idx_ecg_peak]

            # peak와 peak 사이 interval에 대한 noise 여부 -> 따라서 길이는 peak - 1
            bool_noise_ppg = [False for k in range(len(idx_ppg_peak)-1)]
            bool_noise_ecg = [False for k in range(len(idx_ecg_peak)-1)]


            #  2.1 peak 간격 이상한 noise (HR 30~150 -> HBI 0.4s ~ 2s로 SRATE 곱해주면 40~200)
            for k in range(len(bool_noise_ppg)):
                if not 0.4*SRATE < idx_ppg_peak[k+1] - idx_ppg_peak[k] < 2*SRATE:
                    bool_noise_ppg[k] = True
            for k in range(len(bool_noise_ecg)):
                if not 0.4*SRATE < idx_ecg_peak[k+1] - idx_ecg_peak[k] < 2*SRATE:
                    bool_noise_ecg[k] = True


            # 2.2 모양 이상한 noise
            # wave interval into same length(2s(200))
            len_wave = 2*SRATE
            norm_seg_ppg, norm_seg_ecg = [], []

            for k in range(len(bool_noise_ppg)):
                len_interval_ppg = idx_ppg_peak[k+1] - idx_ppg_peak[k]

                # peak 사이 wave를 모두 같은 길이로 변환
                norm_seg_ppg.append([linear_connection(seg_ppg[idx_ppg_peak[k]:idx_ppg_peak[k+1]+1], n/len_wave*len_interval_ppg) for n in range(len_wave)])

            for k in range(len(bool_noise_ecg)):
                len_interval_ecg = idx_ecg_peak[k+1] - idx_ecg_peak[k]

                # peak 사이 wave를 모두 같은 길이로 변환
                norm_seg_ecg.append([linear_connection(seg_ecg[idx_ecg_peak[k]:idx_ecg_peak[k+1]+1], n/len_wave*len_interval_ecg) for n in range(len_wave)])


            # wave interval 사이 correlation 계산 - PPG
            mean_wave_ppg = np.nanmean(norm_seg_ppg, axis = 0)
            mean_wave_ppg = pd.DataFrame(mean_wave_ppg).fillna(method='ffill', axis=0).fillna(method='bfill', axis=0).values.flatten()
            norm_seg_ppg = pd.DataFrame(norm_seg_ppg).fillna(method='ffill', axis=1).fillna(method='bfill', axis=1).values
            for k in range(len(bool_noise_ppg)):
                if np.corrcoef(norm_seg_ppg[k], mean_wave_ppg)[0,1] < 0.9:
                    bool_noise_ppg[k] = True
            noise_ppg_perc = np.sum(bool_noise_ppg) / len(bool_noise_ppg)


            # wave interval 사이 correlation 계산 - ECG                
            mean_wave_ecg = np.nanmean(norm_seg_ecg, axis = 0)
            mean_wave_ecg = pd.DataFrame(mean_wave_ecg).fillna(method='ffill', axis=0).fillna(method='bfill', axis=0).values.flatten()
            norm_seg_ecg = pd.DataFrame(norm_seg_ecg).fillna(method='ffill', axis=1).fillna(method='bfill', axis=1).values
            for k in range(len(bool_noise_ecg)):
                if np.corrcoef(norm_seg_ecg[k], mean_wave_ecg)[0,1] < 0.9:
                    bool_noise_ecg[k] = True
            noise_ecg_perc = np.sum(bool_noise_ecg) / len(bool_noise_ecg)


            # segment의 noise 비율 정보
            noise_info = [noise_ppg_perc, noise_ecg_perc]

            # segment를 input으로 써도 되는지
            if nan_ppg_perc < 0.05 and nan_ecg_perc < 0.05 and nan_both_perc < 0.05 and noise_ppg_perc < 0.1 and noise_ecg_perc < 0.1:
                bool_pass = True
            else:
                bool_pass = False


            # 통증 점수 계산
            ### TSS(total surgical stimulation) = 1.57 - rftn20_ce / 3
            ### CISA(combined index of stimulus and analgesia) = stim_intensity - beta * ce + gamma, beta = 1/8, gamma = 1.5, stim_intensity = 5.5 
            rftn = vals['RFTN'][start_idx:end_idx]
            rftn = np.mean(rftn[~np.isnan(rftn)])
            tss = 1.57 - rftn / 3
            if tss < 0:
                tss = 0
            cisa = 7 - rftn / 8

            # 이 segment의 정보를 dataframe에 저장
            arry = np.empty(1, dtype=object)
            arry[0] = [bool_pass, nan_info, noise_info, tss, cisa]
            df_preprocess.loc[f_num-1,f'{i+n_aug+1}'] = arry[0] #{'pass':bool_pass, 'nan_perc':nan_info, 'noise_perc':noise_info, 'tss':0, 'cisa':0}            
            print('preprocessing done...', end='')            

    print(f'\ndumping cache of df_preprocess {f_num}/{len(caseids)}', end='...')
    
    # df_preprocess에 demographs(age, gender) 추가
    df_demograph = pd.read_csv("https://api.vitaldb.net/cases")
    df_preprocess['age'] = np.nan
    df_preprocess['gender'] = np.nan

    for idx, row in df_preprocess.iterrows():     
        row_demo = df_demograph[df_demograph['caseid']==int(row['caseid'])]

        df_preprocess.loc[idx, 'age'] = row_demo['age'].values[0]
        df_preprocess.loc[idx, 'gender'] = row_demo['sex'].values[0]

    df_preprocess.reset_index(drop=True, inplace=True)    
    pickle.dump(df_preprocess, open('cache/preprocess/df_preprocess', 'wb'))
    print('dumping success')
    
    # 전처리 통과 비율 출력
    ne_pass, e_pass = 0, 0

    for _, row in df_preprocess.iterrows():   
        for i in range(n_aug):
            if row[str(i+1)][0]:
                ne_pass = ne_pass + 1

        for i in range(n_aug, n_aug+n_aug2):
            if row[str(i+1)][0]:
                e_pass = e_pass + 1

    print(f'non-event seg pass: {ne_pass/n_aug/2684*100:.2f}%, event seg pass: {e_pass/n_aug2/2684*100:.2f}%')
    print(f'passed segments : {ne_pass+e_pass}')


# loading ~ preprocessing
file_path = f'vital_to_np_{LEN_PER_PRE}s-{LEN_PER_POST}s'
load_vital_data(file_path)
preprocess(file_path)
df_preprocess = pickle.load(open('cache/preprocess/df_preprocess', 'rb'))

# shuffle caseids which has survived preprocessing
p_caseids = []
for _, row in df_preprocess.iterrows():
    for i in range(0,10):
        if row[str(i+1)][0]:
            p_caseids.append(row['caseid'])
            break
print(f'survived caseids : {len(p_caseids)} cases / {len(df_preprocess)} cases')

# caseid 단위로 train, val, test set로 나눔
caseids = list(np.unique(p_caseids))
random.shuffle(caseids)

ntest = max(1, int(len(caseids) * 0.1))
nval = max(1, int(len(caseids) * (1 - 0.1) * 0.1))
ntrain = len(caseids) - ntest - nval

caseid_train = caseids[ntest + nval:]
caseid_val = caseids[ntest:ntest + nval]
caseid_test = caseids[:ntest]

print('전체 caseid 수: {}'.format(len(p_caseids)))
print('train caseid 수: {}, val caseid 수: {}, test caseid 수: {}'.format(len(caseid_train), len(caseid_val), len(caseid_test)))

pickle.dump(caseid_train, open('../DL_model/caseid_train','wb'))
pickle.dump(caseid_val, open('../DL_model/caseid_val','wb'))
pickle.dump(caseid_test, open('../DL_model/caseid_test','wb'))


# input - filtering, saving
input_path = f"../DL_model/dataset/ne{LEN_PER_PRE}s-e{LEN_PER_POST}s-len{LEN_INPUT}-{OVERLAP}/"
if not os.path.exists('../DL_model/dataset'):
    os.mkdir('../DL_model/dataset')
if not os.path.exists(input_path[:-1]):
    os.mkdir(input_path[:-1])
    
    
# variables
non_lis = []
x_train, tss_train, cisa_train = [], [], []
x_test, tss_test, cisa_test = [], [], []
x_val, tss_val, cisa_val = [], [], []
age_train, gender_train = [], []
age_test, gender_test = [], []
age_val, gender_val = [], []
     

for f_num, row in df_preprocess.iterrows():
    caseid = row['caseid']
    print(f'\n###Input{f_num}/{len(df_preprocess)}: {caseid}###')
    

    # vital data 불러오기    
    vals = np.load(f'{file_path}/{caseid}.npz')


    # 20초 단위로 끊기
    for i in range(n_aug):
        print('  n_segment', i+1, end='')
        start_idx = i*OVERLAP*SRATE # 500i
        end_idx = (i*OVERLAP + LEN_INPUT)*SRATE # 500i + 1000
        
        # non-event data
        if row[str(i+1)][0]:
            print(' passed...lowess filtering...', end='')
            
            #save_path = f'cache/lowess_filtered/intu120s-input20s-10s/{caseid}_n{i+1}.npz'

            ppg_inp = vals['nPPG'][start_idx:end_idx]
            ecg_inp = vals['nECG'][start_idx:end_idx]
            
            ppg_inp = pd.DataFrame(ppg_inp).fillna(method='ffill', axis=0).fillna(method='bfill', axis=0).values.flatten()
            ecg_inp = pd.DataFrame(ecg_inp).fillna(method='ffill', axis=0).fillna(method='bfill', axis=0).values.flatten()
            
            # lowess filter 적용
            ppg_input = ppg_inp - lowess(ppg_inp)
            ecg_input = ecg_inp - lowess(ecg_inp)
            
            ppg_input = ppg_input - np.nanmean(ppg_input)
            ecg_input = ecg_input - np.nanmean(ecg_input)
            #ecg_input = (ecg_input - min(ecg_input)) / (max(ecg_input) - min(ecg_input))
            

            # 해당 caseid가 test set에 속하는 경우
            if row['caseid'] in caseid_test:
                age_test.append(int(row['age']))
                if row['gender']=='F':
                    gender_test.append(1)
                else:
                    gender_test.append(0)
                x_test.append([ppg_input, ecg_input])
                tss_test.append(row[str(i+1)][3])
                cisa_test.append(row[str(i+1)][4])

            # 해당 caseid가 val set에 해당하는 경우
            elif row['caseid'] in caseid_val:
                age_val.append(int(row['age']))
                if row['gender']=='F':
                    gender_val.append(1)
                else:
                    gender_val.append(0)                    
                x_val.append([ppg_input, ecg_input])
                tss_val.append(row[str(i+1)][3])
                cisa_val.append(row[str(i+1)][4])

            # 해당 caseid가 train set에 해당하는 경우
            elif row['caseid'] in caseid_train:
                age_train.append(int(row['age']))
                if row['gender']=='F':
                    gender_train.append(1)
                else:
                    gender_train.append(0)                    
                x_train.append([ppg_input, ecg_input])
                tss_train.append(row[str(i+1)][3])
                cisa_train.append(row[str(i+1)][4])

            else:
                print('no case%$')
                non_lis.append(row['caseid'])

            #np.savez(save_path, ECG = ecg_input, PPG = ppg_input)
            print('done', end=' ')
    
    print('')
    for i in range(n_aug2):
        print('  segment', i+1, end='')
        start_idx = i*OVERLAP*SRATE # 500i
        end_idx = (i*OVERLAP + LEN_INPUT)*SRATE # 500i + 1000
            
        # event data
        if row[str(i+n_aug+1)][0]:
            print(' passed...lowess filtering...', end='')
            
            #save_path = f'cache/lowess_filtered/input20s-10s/{caseid}_{i+1}.npz'
            ppg_inp = vals['PPG'][start_idx:end_idx]
            ecg_inp = vals['ECG'][start_idx:end_idx]
            
            ppg_inp = pd.DataFrame(ppg_inp).fillna(method='ffill', axis=0).fillna(method='bfill', axis=0).values.flatten()
            ecg_inp = pd.DataFrame(ecg_inp).fillna(method='ffill', axis=0).fillna(method='bfill', axis=0).values.flatten()
            
            # lowess filter 적용
            ppg_input = ppg_inp - lowess(ppg_inp)
            ecg_input = ecg_inp - lowess(ecg_inp)
            
            ppg_input = ppg_input - np.nanmean(ppg_input)
            ecg_input = (ecg_input - min(ecg_input)) / (max(ecg_input) - min(ecg_input))


            # 해당 caseid가 test set에 속하는 경우
            if row['caseid'] in caseid_test:
                age_test.append(int(row['age']))
                if row['gender']=='F':
                    gender_test.append(1)
                else:
                    gender_test.append(0)
                x_test.append([ppg_input, ecg_input])
                tss_test.append(row[str(i+n_aug+1)][3])
                cisa_test.append(row[str(i+n_aug+1)][4])

            # 해당 caseid가 val set에 해당하는 경우
            elif row['caseid'] in caseid_val:
                age_val.append(int(row['age']))
                if row['gender']=='F':
                    gender_val.append(1)
                else:
                    gender_val.append(0)                    
                x_val.append([ppg_input, ecg_input])
                tss_val.append(row[str(i+n_aug+1)][3])
                cisa_val.append(row[str(i+n_aug+1)][4])

            # 해당 caseid가 train set에 해당하는 경우
            elif row['caseid'] in caseid_train:
                age_train.append(int(row['age']))
                if row['gender']=='F':
                    gender_train.append(1)
                else:
                    gender_train.append(0)                    
                x_train.append([ppg_input, ecg_input])
                tss_train.append(row[str(i+n_aug+1)][3])
                cisa_train.append(row[str(i+n_aug+1)][4])

            else:
                print('no case%$')
                non_lis.append(row['caseid'])                 
            
            #np.savez(save_path, ECG = ecg_input, PPG = ppg_input) 
            print('done', end=' ')

x_train = np.array(x_train, np.float32)
x_test = np.array(x_test, np.float32)
x_val = np.array(x_val, np.float32)
tss_train = np.array(tss_train, np.float32)
tss_test = np.array(tss_test, np.float32)
tss_val = np.array(tss_val, np.float32)
cisa_train = np.array(cisa_train, np.float32)
cisa_test = np.array(cisa_test, np.float32)
cisa_val = np.array(cisa_val, np.float32)

age_train = np.array(age_train, int)
age_test = np.array(age_test, int)
age_val = np.array(age_val, int)
gender_train = np.array(gender_train, int)
gender_test = np.array(gender_test, int)
gender_val = np.array(gender_val, int)


# 알맞게 input 변환
x_train = np.transpose(x_train, [0,2,1])
x_val = np.transpose(x_val, [0,2,1])
x_test = np.transpose(x_test, [0,2,1])

print('after concatenate + transpose')
print('x_train shape:', x_train.shape)
print('x_val shape:', x_val.shape)
print('x_test shape:', x_test.shape)


# 저장하기
print('saving...', end='', flush=True)
np.savez_compressed(input_path+'x_train.npz', x_train)
np.savez_compressed(input_path+'x_test.npz', x_test)
np.savez_compressed(input_path+'x_val.npz', x_val)
np.savez_compressed(input_path+'tss_train.npz', tss_train)
np.savez_compressed(input_path+'tss_test.npz', tss_test)
np.savez_compressed(input_path+'tss_val.npz', tss_val)
np.savez_compressed(input_path+'cisa_train.npz', cisa_train)
np.savez_compressed(input_path+'cisa_test.npz', cisa_test)
np.savez_compressed(input_path+'cisa_val.npz', cisa_val)

np.savez_compressed(input_path+'age_train.npz', age_train)
np.savez_compressed(input_path+'age_test.npz', age_test)
np.savez_compressed(input_path+'age_val.npz', age_val)    
np.savez_compressed(input_path+'gender_train.npz', gender_train)
np.savez_compressed(input_path+'gender_test.npz', gender_test)
np.savez_compressed(input_path+'gender_val.npz', gender_val)    

print('done', flush=True)
print('size of training set(pacu):', len(x_train))
print('size of validation set(pacu):', len(x_val))
print('size of test set(pacu):', len(x_test))