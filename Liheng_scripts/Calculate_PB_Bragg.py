## load dose script
import glob, os
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
import datetime
from Liheng_PB_dose import *
# !python ./Liheng_scripts/mount_server.py
# %run Liheng_PB_dose.py
ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
## for office computer, mount the cluster

import argparse
import os

# python Calculate_PB_Bragg.py --CT_block_path /mnt/p/data/SynthRAD2023_KD/crop/save_test/ --save_path /mnt/p/data/SynthRAD2023_KD/crop/ --tune_f 1.0

# inplant 1PA086, 1PA168
# actually have prostate: 1PA049, 1PA083, 1PA093, 1PA100, 1PA134, 1PA150, 1PA165, 1PA187
# have no prostate: '1PA054', '1PA109', '1PA019', '1PA154', '1PA116', '1PA028','1PA115', '1PA110'
patient_id = ['1PA054', '1PA109', '1PA019', '1PA154', '1PA116', '1PA028','1PA115', '1PA110']
desired_blocks = np.zeros((48,48,300))

parser = argparse.ArgumentParser(
    description="Need paths revised images"
)
parser.add_argument("--CT_block_path", required=True, type=str)
parser.add_argument("--save_path", required=True, type=str)
parser.add_argument("--tune_f", required=True, type=float)

args = parser.parse_args()

CT_block_path = args.CT_block_path
save_path = args.save_path+ 'Range/'
f_i = args.tune_f


print(datetime.datetime.now())
#### patient

keyword_i = 'CT'
# ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
print(ts)

if not os.path.isdir(save_path):
    os.makedirs(save_path)

save_file = save_path + keyword_i+ '_'+ts+'.csv'
ene_min = 70.0
ene_max = 220.0
CT_full = glob.glob(CT_block_path+'*_'+keyword_i+'*')
CT_full[0]

result_this = pd.DataFrame(columns=["name", "energy", "range"])

for CT_file_i in CT_full:
    name_i = os.path.basename(CT_file_i)
    ## need to check if already calculated

    CT_i = np.rint(np.transpose(np.load(CT_file_i), (2, 0, 1))).astype(np.int64)
    if_overshoot = 1
    counter_i = 0
    energy_i = round(np.random.uniform(ene_min, ene_max), 2) ## generate a random energy, penaltrate, reduce
    while if_overshoot == 1:
        # print(energy_i)
        dose_i = predict_pencil_beam_dose(CT_i,energy_mev=energy_i, if_check=0)
        Bragg_i = np.sum(np.sum(dose_i,axis=-1),axis=-1)
        if np.mean(Bragg_i[-5:])<0.1*np.max(Bragg_i): ## no overshooting
            if_overshoot = 0
            break
        else:
            energy_i=np.mean([energy_i,ene_min])
            counter_i = counter_i + 1
        if counter_i >= 10:
            break
    if if_overshoot == 0: ## valid result
        range_i = distal_falloff_80(Bragg_i)
        ## calculate 80% distal fall off
    else:
        range_i = -1
        energy_i = -1
    new_row = pd.DataFrame(
        [[name_i, energy_i, range_i]],
        columns=["name", "energy", "range"]
        )
    result_this = pd.concat([result_this, new_row], ignore_index=True)
    # break # CT blocks
result_this.to_csv(save_file, index=False)

save_file_sCT = save_path + 'sCT_'+str(f_i)+'_'+ts+'.csv'
result_sCT = pd.DataFrame(columns=["name", "energy", "range_p", "range_m"])
CT_info = result_this
key_sCT = 'sCT'
key_un = 'un'
counter_i = 0
for i in range(len(CT_info)):
    name_i = CT_info['name'].iloc[i].split('CT')[0]
    energy_i = CT_info['energy'].iloc[i]
    if energy_i >=70:
        sCT_i = np.transpose(np.load(CT_block_path+name_i+key_sCT+'.npy'), (2, 0, 1))
        un_i = np.transpose(np.load(CT_block_path+name_i+key_un+'.npy'), (2, 0, 1))

        sCT_p = np.rint(sCT_i+un_i*f_i).astype(np.int64)
        sCT_m = np.rint(sCT_i-un_i*f_i).astype(np.int64)

        sCT_m[sCT_m<-1000] = -1000 ## we did sCT - f*un here
        # sCT_p[sCT_p<-900] = -1000 ## ???


        dose_p_i = predict_pencil_beam_dose(sCT_p,energy_mev=energy_i, if_check=0)
        dose_m_i = predict_pencil_beam_dose(sCT_m,energy_mev=energy_i, if_check=0)

        Bragg_p_i = np.sum(np.sum(dose_p_i,axis=-1),axis=-1)
        Bragg_m_i = np.sum(np.sum(dose_m_i,axis=-1),axis=-1)

        range_p_i = distal_falloff_80(Bragg_p_i)
        range_m_i = distal_falloff_80(Bragg_m_i)
    else: 
        energy_i = -1
        range_p_i = -1
        range_m_i = -1
    new_row = pd.DataFrame(
            [[name_i, energy_i, range_p_i, range_m_i]],
            columns=["name", "energy", "range_p", "range_m"]
            )
    result_sCT = pd.concat([result_sCT, new_row], ignore_index=True)
    if min(range_m_i, range_p_i) <= CT_info['range'].iloc[i] <= max(range_m_i, range_p_i):
        counter_i = counter_i+1
    # if i % 100 == 0:
    #     print(i)
result_sCT.to_csv(save_file_sCT, index=False)
print('f = ',f_i)
print(counter_i/len(CT_info))
    # break # CT blocks