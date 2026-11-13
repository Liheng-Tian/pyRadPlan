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

save_file = save_path + keyword_i+'.csv'
ene_min = 70.0
ene_max = 220.0
CT_full = glob.glob(CT_block_path+'*_'+keyword_i+'*')
# CT_full[0]

if os.path.isfile(save_file):
    df_existing = pd.read_csv(save_file)
    print(f"Found existing file: {save_file} – {len(df_existing)} rows loaded.")
    # print(df_existing.head(5))
else:
    df_existing = pd.DataFrame(columns=["name", "energy", "range"])
    print(f"No existing result file – starting fresh.")


result_this = pd.DataFrame(columns=["name", "energy", "range"])
# print(CT_full[:5])
for CT_file_i in CT_full:
    name_i = os.path.basename(CT_file_i)

    if name_i in df_existing["name"].values:
        # Already in the CSV → skip all heavy calculations
        # print(f"Skipping {name_i} – already processed.")
        continue

    ## need to check if already calculated
    if int(name_i.split('Angle_')[-1].split('_')[0]) <= 180: # only for non-bed situation
        CT_i = np.rint(np.transpose(np.load(CT_file_i), (2, 0, 1))).astype(np.int64)
        if_overshoot = 1
        counter_i = 0
        energy_i = round(np.random.uniform(ene_min, ene_max), 2) ## generate a random energy, penaltrate, reduce
        while if_overshoot == 1:
            # print(energy_i)
            dose_i = predict_pencil_beam_dose(CT_i,energy_mev=energy_i)
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

result_this = pd.concat([df_existing, result_this], ignore_index=True)
result_this = result_this.drop_duplicates(subset="name", keep="first")
result_this.to_csv(save_file, index=False)
print('CT calculation finished')

save_file_sCT = save_path + 'sCT_'+str(f_i)+'.csv'
result_sCT = pd.DataFrame(columns=["name", "energy", "range_p", "range_m"])


# If the file already exists, read it so we can skip rows that were
# processed in a previous run.
if os.path.isfile(save_file_sCT):
    df_sCT_existing = pd.read_csv(save_file_sCT)
    print(f"Found existing sCT interval file: {save_file_sCT} – {len(df_sCT_existing)} rows loaded.")
else:
    df_sCT_existing = pd.DataFrame(columns=["name", "energy", "range_p", "range_m"])
    print("No existing sCT interval result file – starting fresh.")


CT_info = result_this
key_sCT = 'sCT'
key_un = 'un'
counter_i = 0

for i in range(len(CT_info)):
    name_i = CT_info['name'].iloc[i].split('CT')[0]

    # Skip this patient if we already have a row for it in the sCT CSV.
    if name_i in df_sCT_existing["name"].values:
        # print(f"Skipping sCT for {name_i} – already in CSV.")
        continue


    energy_i = CT_info['energy'].iloc[i]
    if energy_i >=70:
        sCT_i = np.transpose(np.load(CT_block_path+name_i+key_sCT+'.npy'), (2, 0, 1))
        un_i = np.transpose(np.load(CT_block_path+name_i+key_un+'.npy'), (2, 0, 1))

        sCT_p = np.rint(sCT_i+un_i*f_i).astype(np.int64)
        sCT_m = np.rint(sCT_i-un_i*f_i).astype(np.int64)

        sCT_m[sCT_m<-1000] = -1000 ## we did sCT - f*un here
        # sCT_p[sCT_p<-900] = -1000 ## ???


        dose_p_i = predict_pencil_beam_dose(sCT_p,energy_mev=energy_i)
        dose_m_i = predict_pencil_beam_dose(sCT_m,energy_mev=energy_i)

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
result_sCT = pd.concat([df_sCT_existing, result_sCT], ignore_index=True)
result_sCT = result_sCT.drop_duplicates(subset="name", keep="first")
result_sCT.to_csv(save_file_sCT, index=False)
print('dose interval calculation finished')
# print('f = ',f_i)
# print(counter_i/len(CT_info))
    # break # CT blocks


save_file_sCT = save_path + 'sCT'+'.csv'
result_sCT = pd.DataFrame(columns=["name", "energy", "range"])


# If the file already exists, read it so we can skip rows that were
# processed in a previous run.
if os.path.isfile(save_file_sCT):
    df_sCT_existing = pd.read_csv(save_file_sCT)
    print(f"Found existing sCT file: {save_file_sCT} – {len(df_sCT_existing)} rows loaded.")
else:
    df_sCT_existing = pd.DataFrame(columns=["name", "energy", "range"])
    print("No existing sCT result file – starting fresh.")


CT_info = result_this
key_sCT = 'sCT'
# key_un = 'un'
counter_i = 0

for i in range(len(CT_info)):
    name_i = CT_info['name'].iloc[i].split('CT')[0]

    # Skip this patient if we already have a row for it in the sCT CSV.
    if name_i in df_sCT_existing["name"].values:
        # print(f"Skipping sCT for {name_i} – already in CSV.")
        continue


    energy_i = CT_info['energy'].iloc[i]
    if energy_i >=70:
        sCT_i = np.transpose(np.load(CT_block_path+name_i+key_sCT+'.npy').astype(np.int64), (2, 0, 1))
        # un_i = np.transpose(np.load(CT_block_path+name_i+key_un+'.npy'), (2, 0, 1))

        # sCT_p = np.rint(sCT_i+un_i*f_i).astype(np.int64)
        # sCT_m = np.rint(sCT_i-un_i*f_i).astype(np.int64)

        sCT_i[sCT_i<-1000] = -1000 ## we did sCT - f*un here
        # sCT_p[sCT_p<-900] = -1000 ## ???


        dose_i = predict_pencil_beam_dose(sCT_i,energy_mev=energy_i)
        # dose_m_i = predict_pencil_beam_dose(sCT_m,energy_mev=energy_i)

        Bragg_i = np.sum(np.sum(dose_i,axis=-1),axis=-1)
        # Bragg_m_i = np.sum(np.sum(dose_m_i,axis=-1),axis=-1)

        range_i = distal_falloff_80(Bragg_i)
        # range_m_i = distal_falloff_80(Bragg_m_i)
    else: 
        energy_i = -1
        range_i = -1

    new_row = pd.DataFrame(
            [[name_i, energy_i, range_i]],
            columns=["name", "energy", "range"]
            )
    result_sCT = pd.concat([result_sCT, new_row], ignore_index=True)
    # if i % 100 == 0:
    #     print(i)
result_sCT = pd.concat([df_sCT_existing, result_sCT], ignore_index=True)
result_sCT = result_sCT.drop_duplicates(subset="name", keep="first")
result_sCT.to_csv(save_file_sCT, index=False)
print('sCT dose calculation finished')



##### entrance, [15:34,15:34], this area
ct_csv_path = os.path.join(save_path, "CT.csv")          # <-- same file you wrote before
if not os.path.isfile(ct_csv_path):
    raise FileNotFoundError(f"Cannot find CT‑summary file: {ct_csv_path}")

df_ct = pd.read_csv(ct_csv_path)
if "entrance" in df_ct.columns:
    print("Column 'entrance' already present – nothing to recompute.")
else:
    print("Column 'entrance' missing → start surface‑detection calculations...")
    # --------------------------------------------------------------
    # 3️⃣  Prepare a new column (filled with NaN for now)
    # --------------------------------------------------------------
    df_ct["entrance"] = np.nan

    # --------------------------------------------------------------
    # 4️⃣  Loop over every CT file that appears in the CSV
    # --------------------------------------------------------------
    for idx, row in df_ct.iterrows():
        # ----- name in the CSV already contains the “CT” suffix -----
        ct_name = row["name"]               # e.g. 1PA054_Angle_132_…_CT.npy
        ct_file = os.path.join(CT_block_path, ct_name)

        # ----- safety check – file must exist (you may have removed some) -----
        if not os.path.isfile(ct_file):
            print(f"  ⚠️  CT file not found, skip: {ct_file}")
            continue

        # ----- load the CT volume exactly as you did before -----
        CT_i = np.rint(
            np.transpose(np.load(ct_file), (2, 0, 1))
        ).astype(np.int64)

        # ----- surface detection (you already have the function) -----
        surface_i = surface_detection(CT_i)          # returns a 2‑D surface map

        # ----- average the central 19×19 region (indices 15:34) -----
        entrance_i = surface_i[15:34, 15:34].mean()

        # ----- store the value back into the DataFrame -----
        df_ct.at[idx, "entrance"] = entrance_i

        # optional progress printout
        if (idx + 1) % 500 == 0 or (idx + 1) == len(df_ct):
            print(f"  processed {idx + 1}/{len(df_ct)} rows")

    # --------------------------------------------------------------
    # 5️⃣  Save the updated CSV (overwrites the old one)
    # --------------------------------------------------------------
    df_ct.to_csv(ct_csv_path, index=False)
    print(f"\nFinished – column 'entrance' added and saved to {ct_csv_path}")


save_file_sCT = os.path.join(save_path, 'sCT.csv')   # the file you just created
if not os.path.isfile(save_file_sCT):
    raise FileNotFoundError(f"Could not find {save_file_sCT}")

df_sCT = pd.read_csv(save_file_sCT)
if 'entrance' in df_sCT.columns:
    print("Column 'entrance' already present in sCT CSV – skipping surface calculation.")
else:
    print("Adding column 'entrance' to the sCT CSV …")
    # -------------------------------------------------------------
    # 3️⃣  Prepare an empty column (filled with NaN)
    # -------------------------------------------------------------
    df_sCT['entrance'] = np.nan
    # -------------------------------------------------------------
    # 4️⃣  Loop over every row that is **not** yet in the CSV
    # -------------------------------------------------------------
    # We reuse the same list of names you already have in `result_sCT`
    # (the CSV contains exactly the same rows you processed for dose).
    key_sCT = 'sCT'          # same token you used when loading the volume
    for idx, row in df_sCT.iterrows():
        # ----- the original `name` column still contains the “_CT.npy” suffix.
        # ----- Strip that suffix to obtain the base name that is used for the sCT file.
        base_name = row['name'].split('CT')[0]          # e.g. 1PA054_Angle_…
        sCT_path  = os.path.join(CT_block_path, f"{base_name}{key_sCT}.npy")

        if not os.path.isfile(sCT_path):
            print(f"  ⚠️  sCT file not found – skip {base_name}")
            continue

        # ----- load the sCT volume exactly as you did for the dose calculation
        sCT_i = np.transpose(
                    np.load(sCT_path).astype(np.int64),   # keep int64 like before
                    (2, 0, 1)
                )

        # ----- surface detection
        surface_i = surface_detection(sCT_i)          # returns a 2‑D map

        # ----- average the central 19×19 region (indices 15:34)
        entrance_i = surface_i[15:34, 15:34].mean()

        # ----- store the value
        df_sCT.at[idx, 'entrance'] = entrance_i

        # optional progress report
        if (idx + 1) % 500 == 0 or (idx + 1) == len(df_sCT):
            print(f"  processed {idx + 1}/{len(df_sCT)} rows")

    # -------------------------------------------------------------
    # 5️⃣  Save the updated CSV (overwrites the previous file)
    # -------------------------------------------------------------
    df_sCT.to_csv(save_file_sCT, index=False)
    print(f"\nFinished – column 'entrance' added and saved to {save_file_sCT}")