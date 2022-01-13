import os, sys
import pickle
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pyvital2 import arr


def smooth(y):
    #return savitzky_golay(y, window_size=2001, order=3)
    return lowess(y)


# 0.2가 제일 잘 없앴음
def lowess(y, f=0.2):
    x = np.arange(0, len(y))
    return sm.nonparametric.lowess(y, x, frac=f, it=0)[:, 1].T


# 피크 사이 wave를 모두 같은 length로 만들기 위한 함수
def linear_connection(list, idx):
    int_idx = int(idx)
    return list[int_idx] + (list[int_idx+1] - list[int_idx]) * (idx - int_idx)


def preprocess(file_path, LEN_INPUT = 20, OVERLAP = 10, SRATE = 100):
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
    n_aug = int((60-LEN_INPUT)/OVERLAP) + 1   # number of data augmentation
    column_list = ['caseid'] + [str(i+1) for i in range(n_aug*2)]
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
        df_preprocess.loc[f_num-1,'file_path'] = caseid

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
                print('too much missing data')
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
                print('too less peaks')
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
            df_preprocess.loc[f_num-1,f'{i+1}'] = [bool_pass, nan_info, noise_info, 0, 0] #{'pass':bool_pass, 'nan_perc':nan_info, 'noise_perc':noise_info, 'tss':0, 'cisa':0}        
            print('preprocessing done...', end='')
            ##########################################################################

            
        for i in range(n_aug):
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
                print('too much missing data')
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
                print('too less peaks')
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
            df_preprocess.loc[f_num-1,f'{i+n_aug+1}'] = [bool_pass, nan_info, noise_info, tss, cisa] #{'pass':bool_pass, 'nan_perc':nan_info, 'noise_perc':noise_info, 'tss':tss, 'cisa':cisa}       
            print('preprocessing done...', end='')            



    print(f'\ndumping cache of d_preprocess {f_num}/{len(caseids)}')
    #pickle.dump(df_preprocess, open(f'cache/preprocess/df_preprocess_{initial}-{initial+interval}', 'wb'))
    pickle.dump(df_preprocess, open('cache/preprocess/df_preprocess', 'wb'))