# DN_vf_tools.py – Visual Field utility functions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle
import math
import os
import json
import random
from typing import List, Optional, Dict, Any, cast

def explore_df(df, df_name="DataFrame"):
    print(f"\n=== Exploration {df_name} ===")
    print(f"Nombre de lignes : {df.shape[0]}")
    print(f"Nombre de colonnes : {df.shape[1]}")
    print(f"Colonnes : {df.columns.tolist()}")
    if 'PatientID' in df.columns:
        print(f"Patients uniques : {df['PatientID'].nunique()}")
    if 'ExamID' in df.columns:
        print(f"Examens uniques : {df['ExamID'].nunique()}")

def plot_column_distribution(df, column, bins=50, kde=True):
    plt.figure(figsize=(8, 4))
    sns.histplot(df[column].dropna(), bins=bins, kde=kde)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_vf(tds, vmin=None, vmax=None, show_value=True, show_colorbar=True, title=None, color='black'):
    vf_type = 24
    if len(tds) == 68:
        vf_type = 10
    mat = np.zeros([8, 9])
    nulls = [0,1,2,7,8,9,10,17,18,34,43,45,54,
             55,62,63,64,65,70,71] # 8x9, gray:[43,45]
    mat[3][7] = np.nan
    mat[4][7] = np.nan
    if vf_type == 10:
        mat = np.zeros([10, 10])
        nulls = [0,1,2,3,6,7,8,9,10,11,18,19,20,29,30,39,60,69,70,
                 79,80,81,88,89,90,91,92,93,96,97,98,99] # 10x10
    k = 0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            pos = i * mat.shape[1] + j
            if pos not in nulls:
                mat[i][j] = tds[k]
                k += 1
    # Si vmin/vmax ne sont pas fournis OU ne sont pas finis (NaN/inf), on recalcule.
    if vmin is None or not np.isfinite(vmin):            
        vmin = np.nanmin(tds)
        vmin = 0.0 if not np.isfinite(vmin) else math.floor(vmin * 100) / 100
    if vmax is None or not np.isfinite(vmax):            
        vmax = np.nanmax(tds)
        vmax = 0.0 if not np.isfinite(vmax) else math.ceil(vmax * 100) / 100

    # Si l'étendue est nulle (toutes les valeurs identiques), on élargit
    if np.isclose(vmin, vmax):
        # Choisit un petit écart symétrique autour de la valeur unique
        delta = 1e-3 if np.isclose(vmin, 0) else abs(vmin) * 0.01 + 1e-3
        vmin -= delta
        vmax += delta
    if show_value:
        fig, ax = plt.subplots(1, figsize=(6,5))
        aspect = 0.042
    else:
        fig, ax = plt.subplots(1, figsize=(4,4))
        aspect = 0.05
    # --- Choose colormap & normalization strategy ---------------------------------
    # 1) Range entirely ≥ 0  ->  sequential blues
    # 2) Range entirely ≤ 0  ->  sequential reds (reversed so that lower values are darker)
    # 3) Mixed sign          ->  diverging red-white-blue centred at 0
    if vmin >= 0:
        cmap = plt.get_cmap('Blues').copy()
        divnorm = colors.Normalize(vmin=vmin, vmax=vmax)
        cbar_ticks = [vmin, (vmin + vmax) / 2, vmax]
    elif vmax <= 0:
        cmap = plt.get_cmap('Reds_r').copy()
        divnorm = colors.Normalize(vmin=vmin, vmax=vmax)
        cbar_ticks = [vmin, (vmin + vmax) / 2, vmax]
    else:
        cmap = plt.get_cmap('bwr_r').copy()
        divnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        cbar_ticks = [vmin, 0, vmax]
    cmap.set_bad('gray')
        
    im = plt.imshow(mat, cmap=cmap, norm=divnorm, aspect='auto')
    ax.axis('off')
    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.1)
        cbar = plt.colorbar(orientation="horizontal", cax=cax, ticks=cbar_ticks)
        cax.set_aspect(aspect)
        cbar.ax.tick_params(labelsize=12)
    if vf_type == 24:
        ax.add_patch(Rectangle((6.48, 2.48), 1, 2, fill=False, edgecolor='black', lw=0.5))
    if show_value:
        for (i, j), z in np.ndenumerate(mat):
            pos = i * mat.shape[1] + j
            if pos not in nulls:
                ax.text(j, i, round(z, 2), ha='center', va='center', size=11, color=color)
    if title != None:
        ax.set_title(title, size=12)
    plt.show()




def cal_min(tds_combine):
    """Renvoie le min global en ignorant les NaN dans chaque carte."""
    mins = [np.nanmin(t) for t in tds_combine if np.isfinite(np.nanmin(t))]
    return min(mins) if mins else 0.0

def cal_max(tds_combine):
    """Renvoie le max global en ignorant les NaN dans chaque carte."""
    maxs = [np.nanmax(t) for t in tds_combine if np.isfinite(np.nanmax(t))]
    return max(maxs) if maxs else 0.0
def gen_vfmat(tds, mark_blind=False):
    vf_type = 24
    if len(tds) == 68:
        vf_type = 10
    mat = np.zeros([8, 9])
    nulls = [0,1,2,7,8,9,10,17,18,34,43,45,54,
             55,62,63,64,65,70,71] # 8x9, gray:[43,45]
    if vf_type == 10:
        mat = np.zeros([10, 10])
        nulls = [0,1,2,3,6,7,8,9,10,11,18,19,20,29,30,39,60,69,70,
                 79,80,81,88,89,90,91,92,93,96,97,98,99] # 10x10
    else:
        if mark_blind:
            mat[3][7] = np.nan
            mat[4][7] = np.nan
    k = 0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            pos = i * mat.shape[1] + j
            if pos not in nulls:
                mat[i][j] = tds[k]
                k += 1
    return mat
# def visualize(imgs, sizes=(9,3), vmin=None, vmax=None, show_colorbar=False, show_value=False, title=None, aspect=0.042, cb_shrink=1.0, cbar_label=None):
def visualize(imgs, sizes=(9,3), vmin=-38, vmax=26, show_colorbar=False, show_value=False, title=None, aspect=0.042, cb_shrink=1.0, cbar_label="TD (dB)"):
    if vmin == None:            
        vmin = math.floor(cal_min(imgs)*100)/100
    if vmax == None:            
        vmax = math.ceil(cal_max(imgs)*100)/100
    if vmin >= 0:
        divnorm= colors.TwoSlopeNorm(vmin=-0.001, vcenter=0, vmax=vmax)
    elif vmax <= 0:
        divnorm= colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=0.001)
    else:
        divnorm= colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        
    fig = plt.figure(figsize=sizes, constrained_layout=True)
    for i, tds in enumerate(imgs):
        vf_type = 24
        mat = gen_vfmat(tds)
        nulls = [0,1,2,7,8,9,10,17,18,34,43,45,54,
             55,62,63,64,65,70,71] # 8x9, gray:[43,45]
        if len(tds) == 68:
            vf_type = 10
            nulls = [0,1,2,3,6,7,8,9,10,11,18,19,20,29,30,39,60,69,70,
                 79,80,81,88,89,90,91,92,93,96,97,98,99] # 10x10
        else:
            mat[3][7] = np.nan
            mat[4][7] = np.nan
        plt_idx = i + 1
        ax = plt.subplot(1, len(imgs), plt_idx)
        ax.axis('off')
        
        cmap = plt.get_cmap('bwr_r').copy()
        cmap.set_bad('gray')
        im = ax.imshow(mat, cmap=cmap, norm=divnorm, aspect='auto')
        if show_colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("bottom", size="5%", pad=0.1)
            if cbar_label is None:
                cbar = fig.colorbar(im, orientation="horizontal", cax=cax, ticks=[vmin, 0, vmax])
            else:
                cbar = fig.colorbar(im, orientation="horizontal", cax=cax, ticks=[vmin, 0, vmax], label=cbar_label)
            cax.set_aspect(aspect)
            cbar.ax.tick_params(labelsize=12)

            # ---- Optionally shrink the horizontal length of the colorbar ----
            # cb_shrink < 1.0 => reduce width proportionally and keep it centered
            try:
                if 0 < cb_shrink <= 1:
                    bbox = cax.get_position()
                    new_w = bbox.width * cb_shrink
                    x0 = bbox.x0 + (bbox.width - new_w) / 2
                    cax.set_position([x0, bbox.y0, new_w, bbox.height])
                    # Disable constrained layout to keep manual position
                    cax.figure.set_constrained_layout(False)
            except Exception:
                # Fail silently so visualize never crashes because of weird bbox operations
                pass
        if vf_type == 24:
            ax.add_patch(Rectangle((6.5, 2.5), 1, 2, fill=False, edgecolor='black', lw=0.5))
        if show_value:
            for (p, q), z in np.ndenumerate(mat):
                pos = p * mat.shape[1] + q
                if pos not in nulls and not np.isnan(z):
                    ax.text(
                        q,
                        p,
                        f"{round(float(z), 1)}",
                        ha="center",
                        va="center",
                        size=6,
                    )

        # Gestion sécurisée des titres
        if title and i < len(title) and title[i] is not None:
            ax.set_title(cast(str, title[i]), size=12)
        
    plt.show()
    
    
def plot_longitudinal_vf(df, min_tests=5, max_eyes=5):
    import matplotlib.pyplot as plt
    from datetime import datetime
    import numpy as np

    td_cols = [f'td{i}' for i in range(1, 55) if i not in (26, 35)]
    
    # Filtrer les yeux avec suffisamment de tests
    eye_counts = df.groupby('eyeid').size()
    selected_eyes = eye_counts[eye_counts >= min_tests].index[:max_eyes]
    df_filtered = df[df['eyeid'].isin(selected_eyes)].copy()
    df_filtered['parsed_date'] = pd.to_datetime(df_filtered['parsed_date'])

    for eyeid in selected_eyes:
        eye_data = df_filtered[df_filtered['eyeid'] == eyeid].sort_values('parsed_date')
        patient_id = eye_data['id'].iloc[0]  # récupère le patient associé à cet eyeid
        tds_list = []
        titles = []
        prev_date = None
        for i, (_, row) in enumerate(eye_data.iterrows()):
            tds = row[td_cols].values.astype(float)
            date = row['parsed_date']
            if prev_date is None:
                delta = 0
            else:
                delta = (date - prev_date).days
            titles.append(f"{i+1}  {date.date()} ({delta}d)") #({delta}d)
            tds_list.append(tds)
            prev_date = date

        print(f"Patient ID: {patient_id} | EyeID: {eyeid} | {len(tds_list)} tests")
        visualize(tds_list, sizes=(2.2 * len(tds_list), 3), show_value=True, show_colorbar=True, title=titles)


    ###########         RNFLT function

def display_multiple_npz(npz_dir: str, n: int = 5):
    """
    Display multiple random examples from a directory of .npz files.

    Args:
        npz_dir (str): Path to directory containing paired npz files.
        n (int, optional): Number of samples to display. Defaults to 5.
    """


    all_files = [f for f in os.listdir(npz_dir) if f.endswith(".npz")]
    samples = random.sample(all_files, k=min(n, len(all_files)))

    for fname in samples:
        print(f"\nFile: {fname}")
        data = np.load(os.path.join(npz_dir, fname), allow_pickle=True)
        td = data['td']
        rnflt = data['rnflt']
        meta = json.loads(data['meta'].item())
        print(f"Meta: ID={meta['id']} | Eye={meta['eye']} | Δt={meta['delta_days']:.1f} days | MD={meta['md']:.2f} | VFI={meta['vfi']:.1f}")
        plot_rnflt_and_vf(rnflt, td, meta=meta)
        
def plot_rnflt_and_vf(
    rnflt: np.ndarray,
    tds: np.ndarray,
    meta: Optional[Dict[str, Any]] = None,
    figsize: tuple = (6, 5),
) -> None:
    """
    Plot RNFLT map and Visual Field side by side (in two separate plots, no subplot trick).

    Args:
        rnflt (np.ndarray): 2D RNFLT map.
        tds (np.ndarray): 1D vector of TD values (length 52).
        meta (dict, optional): Optional metadata to display in titles.
        figsize (tuple, optional): Figure size for RNFLT plot.
    """
    # RNFLT plot
    plt.figure(figsize=figsize)
    plt.imshow(rnflt, cmap='jet')
    plt.colorbar(label="RNFLT (μm)")
    title1 = "RNFLT Map"
    if meta and "delta_days" in meta:
        title1 += f"\nΔt: {meta['delta_days']:.1f} days"
    plt.title(title1)
    plt.axis('off')
    plt.show()

    # VF plot (calls default plot_vf with its own layout)
    title2 = "Visual Field (TD)"
    if meta and "md" in meta:
        title2 += f"\nMD: {meta['md']:.2f} dB"
    plot_vf(tds, title=title2,vmin=-38,vmax=26)





    ################## CONVERTION TO IMAGE FOR CNN##################
def convertvf2image(vf):
    temp = np.nan
    # découpage original
    r0, r1 = vf[:4].copy(), vf[4:10].copy()
    r2, r3 = vf[10:18].copy(), vf[18:26].copy()
    r4, r5 = vf[26:34].copy(), vf[34:42].copy()
    r6, r7 = vf[42:48].copy(), vf[48:52].copy()

    # inserts/appends pour obtenir longueur 12
    r3 = np.insert(r3, len(r3)-1, temp); r4 = np.insert(r4, len(r4)-1, temp)
    r0 = np.insert(r0, 0, temp); r0 = np.insert(r0, 1, temp); r0 = np.insert(r0, 2, temp)
    r0 = np.append(r0, temp);    r0 = np.append(r0, temp)
    r1 = np.insert(r1, 0, temp); r1 = np.insert(r1, 1, temp); r1 = np.append(r1, temp)
    r2 = np.insert(r2, 0, temp)
    r5 = np.insert(r5, 0, temp)
    r6 = np.insert(r6, 0, temp); r6 = np.insert(r6, 1, temp); r6 = np.append(r6, temp)
    r7 = np.insert(r7, 0, temp); r7 = np.insert(r7, 1, temp); r7 = np.insert(r7, 2, temp)
    r7 = np.append(r7, temp);    r7 = np.append(r7, temp)

    rows = [r0,r1,r2,r3,r4,r5,r6,r7]
    # append col10 & col11
    rows = [np.append(np.append(r, temp), temp) for r in rows]
    # insert col12
    rows = [np.insert(r, 0, temp) for r in rows]

    full_nan = np.full(rows[0].shape, temp)
    img = np.vstack([full_nan, full_nan] + rows + [full_nan, full_nan])
    return img

#  DON T USE THIS FUNCTION NOT GOOD TO PAD WITH ZERO-> USE convertvf2image instead, !!!not used in the project!!!
def convertvf2image_zero_padding(vf):
    temp = 0.0  # mettre 0 au lieu de NaN

    r0, r1 = vf[:4], vf[4:10]
    r2, r3 = vf[10:18], vf[18:26]
    r4, r5 = vf[26:34], vf[34:42]
    r6, r7 = vf[42:48], vf[48:52]

    r3 = np.insert(r3, len(r3)-1, temp); r4 = np.insert(r4, len(r4)-1, temp)
    r0 = np.insert(r0, 0, temp); r0 = np.insert(r0, 1, temp); r0 = np.insert(r0, 2, temp)
    r0 = np.append(r0, temp);    r0 = np.append(r0, temp)
    r1 = np.insert(r1, 0, temp); r1 = np.insert(r1, 1, temp); r1 = np.append(r1, temp)
    r2 = np.insert(r2, 0, temp)
    r5 = np.insert(r5, 0, temp)
    r6 = np.insert(r6, 0, temp); r6 = np.insert(r6, 1, temp); r6 = np.append(r6, temp)
    r7 = np.insert(r7, 0, temp); r7 = np.insert(r7, 1, temp); r7 = np.insert(r7, 2, temp)
    r7 = np.append(r7, temp);    r7 = np.append(r7, temp)

    rows = [r0, r1, r2, r3, r4, r5, r6, r7]
    rows = [np.append(np.append(r, temp), temp) for r in rows]
    rows = [np.insert(r, 0, temp) for r in rows]

    full_zero = np.full(rows[0].shape, temp)
    img = np.vstack([full_zero, full_zero] + rows + [full_zero, full_zero])
    return img.astype(np.float32)  # (12, 12)
