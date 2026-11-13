#!/usr/bin/env python
# -*- coding: utf-8 -*-
# by AI, not tested yet
# --------------------------------------------------------------
# 0️⃣  Imports & helpers
# --------------------------------------------------------------
import os, glob, argparse, datetime
import numpy as np
import pandas as pd
from Liheng_PB_dose import (
    predict_pencil_beam_dose,
    distal_falloff_80,
    surface_detection,
)

# --------------------------------------------------------------
# 1️⃣  Tiny in‑memory cache – each .npy file is read only once
# --------------------------------------------------------------
_volume_cache = {}          # path → np.ndarray (already transposed)

def _load_volume(path: str, dtype=np.float32) -> np.ndarray:
    """Load a .npy file, transpose to (z,x,y) and return as *dtype*."""
    if path not in _volume_cache:
        # original CT loading used rint → int64, but for sCT we need float first
        arr = np.transpose(np.load(path), (2, 0, 1)).astype(dtype)
        _volume_cache[path] = arr
    return _volume_cache[path]


def _base_name(ct_name: str) -> str:
    """'…_CT.npy' → the base part that is common to CT, sCT and un."""
    return ct_name.split('CT')[0]


# --------------------------------------------------------------
# 2️⃣  Argument parsing (unchanged)
# --------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Calculate proton‑beam range, dose and skin entrance for CT / sCT blocks"
)
parser.add_argument("--CT_block_path", required=True, type=str,
                    help="folder that contains *_CT.npy, *_sCT.npy, *_un.npy")
parser.add_argument("--save_path", required=True, type=str,
                    help="folder where the CSV summary will be stored")
parser.add_argument("--tune_f", required=True, type=float,
                    help="scaling factor f_i used for the sCT±f·un calculation")
args = parser.parse_args()

CT_block_path = args.CT_block_path
save_path      = os.path.join(args.save_path, "Range")   # keep the original sub‑folder
f_i            = args.tune_f

os.makedirs(save_path, exist_ok=True)

# --------------------------------------------------------------
# 3️⃣  -----------------------   CT PART   -----------------------
# --------------------------------------------------------------
ct_csv_path = os.path.join(save_path, "CT.csv")
if os.path.isfile(ct_csv_path):
    df_ct = pd.read_csv(ct_csv_path)
else:
    df_ct = pd.DataFrame(columns=["name", "energy", "range", "entrance"])

CT_files = sorted(glob.glob(os.path.join(CT_block_path, "*_CT*")))

for ct_file in CT_files:
    name_ct = os.path.basename(ct_file)   # e.g. 1PA054_…_CT.npy
    if name_ct in df_ct["name"].values:
        continue

    # ---------- load CT volume (int64 as you originally did) ----------
    CT_i = _load_volume(ct_file, dtype=np.int64)   # already int64 after cache

    # ---------- entrance (skin surface) ----------
    surface_i = surface_detection(CT_i)
    entrance_i = surface_i[15:34, 15:34].mean()

    # ---------- energy & overshoot loop ----------
    ene_min, ene_max = 70.0, 220.0
    # keep only blocks that are NOT the “bed” situation
    if int(name_ct.split('Angle_')[-1].split('_')[0]) > 180:
        continue

    energy_i = round(np.random.uniform(ene_min, ene_max), 2)
    overshoot = True
    cnt = 0
    while overshoot and cnt < 10:
        dose_i = predict_pencil_beam_dose(CT_i, energy_mev=energy_i)
        Bragg_i = np.sum(np.sum(dose_i, axis=-1), axis=-1)
        if np.mean(Bragg_i[-5:]) < 0.1 * np.max(Bragg_i):   # no overshoot
            overshoot = False
        else:
            energy_i = np.mean([energy_i, ene_min])
            cnt += 1

    if overshoot:                     # gave up after 10 attempts
        range_i = -1
        energy_i = -1
    else:
        range_i = distal_falloff_80(Bragg_i)

    # ---------- append ----------
    new_row = pd.DataFrame(
        [[name_ct, energy_i, range_i, entrance_i]],
        columns=["name", "energy", "range", "entrance"]
    )
    df_ct = pd.concat([df_ct, new_row], ignore_index=True)

# save CT summary
df_ct = df_ct.drop_duplicates(subset="name", keep="first")
df_ct.to_csv(ct_csv_path, index=False)
print(f"[CT] finished – {len(df_ct)} rows stored in {ct_csv_path}")

# --------------------------------------------------------------
# 4️⃣  -----------------------   sCT PART   -----------------------
# --------------------------------------------------------------
sct_csv_path = os.path.join(save_path, "sCT.csv")
if os.path.isfile(sct_csv_path):
    df_sct = pd.read_csv(sct_csv_path)
else:
    df_sct = pd.DataFrame(
        columns=["name", "energy",
                 "range_sCT", "range_sCT_p", "range_sCT_m",
                 "entrance"]
    )

key_sCT = "sCT"
key_un  = "un"

for _, ct_row in df_ct.iterrows():
    name_ct   = ct_row["name"]                     # includes the "_CT.npy" suffix
    base_name = _base_name(name_ct)               # e.g. 1PA054_…_
    if base_name in df_sct["name"].values:
        continue                                 # already processed → skip

    energy_i  = ct_row["energy"]
    if energy_i < 70:                            # the same filter you used for dose
        # we still keep a row (so the CSV stays aligned with CT)
        new_row = pd.DataFrame(
            [[base_name, -1, -1, -1, -1, np.nan]],
            columns=["name", "energy",
                     "range_sCT", "range_sCT_p", "range_sCT_m",
                     "entrance"]
        )
        df_sct = pd.concat([df_sct, new_row], ignore_index=True)
        continue

    # ---------- load sCT (float) and uncertainty ----------
    sCT_path = os.path.join(CT_block_path, f"{base_name}{key_sCT}.npy")
    un_path  = os.path.join(CT_block_path, f"{base_name}{key_un}.npy")
    if not (os.path.isfile(sCT_path) and os.path.isfile(un_path)):
        print(f"⚠️  Missing sCT / un for {base_name} – skip")
        continue

    sCT_i = _load_volume(sCT_path, dtype=np.float32)   # keep as float
    un_i  = _load_volume(un_path,  dtype=np.float32)

    # ---------- entrance on the original float sCT ----------
    surface_sCT = surface_detection(sCT_i)             # surface_detection works on float too
    entrance_i  = surface_sCT[15:34, 15:34].mean()

    # ---------- build the three volumes (only round+cast AFTER the '+/- f·un') ----------
    # plain sCT  (clip negative values before rounding)
    sCT_plain = sCT_i.copy()
    sCT_plain[sCT_plain < -1000] = -1000
    sCT_plain = np.rint(sCT_plain).astype(np.int64)

    # sCT + f·un  (upper)
    sCT_p = sCT_i + un_i * f_i
    sCT_p[sCT_p < -1000] = -1000
    sCT_p = np.rint(sCT_p).astype(np.int64)

    # sCT - f·un  (lower)
    sCT_m = sCT_i - un_i * f_i
    sCT_m[sCT_m < -1000] = -1000
    sCT_m = np.rint(sCT_m).astype(np.int64)

    # ---------- predict dose and extract range for the three volumes ----------
    # plain sCT
    dose_sCT = predict_pencil_beam_dose(sCT_plain, energy_mev=energy_i)
    Bragg_sCT = np.sum(np.sum(dose_sCT, axis=-1), axis=-1)
    range_sCT = distal_falloff_80(Bragg_sCT)

    # sCT + f·un
    dose_sCT_p = predict_pencil_beam_dose(sCT_p, energy_mev=energy_i)
    Bragg_sCT_p = np.sum(np.sum(dose_sCT_p, axis=-1), axis=-1)
    range_sCT_p = distal_falloff_80(Bragg_sCT_p)

    # sCT - f·un
    dose_sCT_m = predict_pencil_beam_dose(sCT_m, energy_mev=energy_i)
    Bragg_sCT_m = np.sum(np.sum(dose_sCT_m, axis=-1), axis=-1)
    range_sCT_m = distal_falloff_80(Bragg_sCT_m)

    # ---------- store the row ----------
    new_row = pd.DataFrame(
        [[base_name, energy_i,
          range_sCT, range_sCT_p, range_sCT_m,
          entrance_i]],
        columns=["name", "energy",
                 "range_sCT", "range_sCT_p", "range_sCT_m",
                 "entrance"]
    )
    df_sct = pd.concat([df_sct, new_row], ignore_index=True)

    # optional progress output
    if (len(df_sct) % 100) == 0:
        print(f"  processed {len(df_sct)} sCT blocks …")

# --------------------------------------------------------------
# 5️⃣  Save the sCT summary (only the columns you asked for)
# --------------------------------------------------------------
df_sct = df_sct.drop_duplicates(subset="name", keep="first")
df_sct.to_csv(sct_csv_path, index=False)
print(f"[sCT] finished – {len(df_sct)} rows stored in {sct_csv_path}")

# --------------------------------------------------------------
# 6️⃣  Quick final report (optional)
# --------------------------------------------------------------
print("\n---  Summary  ------------------------------------------------")
print(f"CT blocks processed   : {len(df_ct)}")
print(f"sCT blocks processed  : {len(df_sct)}")
print(f"Scaling factor f_i   : {f_i}")