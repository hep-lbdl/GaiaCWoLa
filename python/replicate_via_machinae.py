### Generic imports 
import os
import numpy as np
import pandas as pd
import matplotlib 
matplotlib.use('Agg') # to not generate plots via X11
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import stats
from glob import glob
import argparse
import json
import time 
from random import randrange
import tensorflow as tf
# import wandb
# from wandb.keras import WandbCallback

### Custom imports
import sys
sys.path.append('./python')
from functions import *
from models import *

### GPU Setup
os.environ["CUDA_VISIBLE_DEVICES"] = '2' # pick a number < 4 on ML4HEP; < 3 on Voltan 
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

### Plot setup
plt.rcParams.update({
    'figure.dpi': 150,
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.family": "serif",
    "font.size": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11
})

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_label", default='test', type=str, help="Folder name for saving training outputs & plots.")
    parser.add_argument("--n_patches", default=21, type=int, help="Number of patches to train over.")
    parser.add_argument("--layer_size", default=128, type=int, help="Number of nodes per layer.")
    parser.add_argument("--patience", default=30, type=int, help="How many epochs of no val_loss improvement before the training is stopped.")
    parser.add_argument("--epochs", default=1000, type=int, help="Number of training epochs.")
    parser.add_argument("--batch_size", default=1000, type=int, help="Batch size during training.")
    parser.add_argument("--dropout", default=0.2, type=float, help="Dropout probability.")
    parser.add_argument("--l2_reg", default=0, type=float, help="L2 regularization.")
    parser.add_argument("--n_folds", default=1, type=int, help="Number of k-folds.")
    parser.add_argument("--sample_weight", default=1, type=float, help="If not equal to 1, adds an additional weight to each star in the stream.")
    parser.add_argument("--best_of_n_loops", default=1, type=int, help="Repeats the training N times and picks the best weights.")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    t0 = time.time()
    
    save_label = args.save_label
    save_folder = os.path.join("./trained_models",save_label)
    os.makedirs(save_folder, exist_ok=True)
    
    ### Save arguments
    with open(os.path.join(save_folder,'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    ### Load file & preprocess
    df_all = pd.read_hdf("./gaia_data/gd1/gd1_allpatches.h5")
    visualize_stream(df_all, save_folder=save_folder)
    
#     limits = pd.DataFrame([
#                       [0, -15, -13.5, -11, -10],
#                       [1, -14, -13, -12, -11],
#                       [2, -14, -13, -11.5, -10],
#                       [3, -14, -13, -11, -10],
#                       [4, -15, -14, -12, -11],
#                       [5, -15, -14, -12, -11],
#                       [6, -5, -4, -2.5, 2],
#                       [7, -10, -9.5, -8, -7],
#                       [8, -9, -8, -7, -6],
#                       [9, -9, -8, -7, -6],
#                       [10, -12.5, -12, -11, -10],
#                       [11, -14, -13, -11, -10],
#                       [12, -14, -13, -12, -11],
#                       [13, -6, -5, -4, -2],
#                       [14, -9, -8, -7, -6],
#                       [15, -9, -8, -7, -6],
#                       [16, -14, -13, -11, -10],
#                       [17, -14, -13, -12, -11],
#                       [18, -6, -5, -3, -2],
#                       [19, -9, -8, -6, -5],
#                       [20, -9, -8.5, -6, -5.5],
#                       ],
#                       columns=["patch_id","sb_min","sr_min","sr_max","sb_max"])
    
    target_stream = []
    top_stars = []

    ## Scan over patches
    n_patches = args.n_patches
    alphas = np.linspace(df_all[df_all.stream].α.min(), df_all[df_all.stream].α.max(), n_patches)
    deltas = np.array([df_all[(df_all.stream & (np.abs(df_all.α - alpha) < 5))].δ.mean() for alpha in alphas])
    limits = pd.DataFrame(zip(np.arange(len(alphas)),alphas,deltas), columns=["patch_id","α_center", "δ_center"])

    for patch_id in tqdm(limits.patch_id.unique()):
        α_min = limits.iloc[patch_id]["α_center"]-10
        α_max = limits.iloc[patch_id]["α_center"]+10
        δ_min = limits.iloc[patch_id]["δ_center"]-10
        δ_max = limits.iloc[patch_id]["δ_center"]+10
        df = (df_all[(α_min < df_all.α) & (df_all.α < α_max) & 
                     (δ_min < df_all.δ) & (df_all.δ < δ_max)])        
        if np.sum(df.stream)/len(df) < 0.00001: # skip patches with hardly any stream stars
            continue
        else:
            visualize_stream(df, save_folder=save_folder+"/patches/patch{}".format(str(patch_id)))
            target_stream.append(df[df.stream])                
            df_train = signal_sideband(df, save_folder=save_folder+"/patches/patch{}".format(str(patch_id)),
                            sb_min = df[df.stream].μ_δ.min(),
                            sr_min = df[df.stream].μ_δ.min()+1,
                            sr_max = df[df.stream].μ_δ.max()-1,
                            sb_max = df[df.stream].μ_δ.max()
                            )
            tf.keras.backend.clear_session()
            test = train(df_train, 
              n_folds = args.n_folds, 
              best_of_n_loops = args.best_of_n_loops,
              layer_size = args.layer_size, 
              batch_size = args.batch_size, 
              dropout = args.dropout, 
              epochs = args.epochs, 
              patience = args.patience,
              save_folder=save_folder+"/patches/patch{}".format(str(patch_id)),
            )

            # Grab top 100 stars
            patch_top_stars = test.sort_values('nn_score',ascending=False)[:100]
            patch_top_stars.to_hdf(save_folder+"/patches/patch{}/top_stars.h5".format(str(patch_id)), "df")
            top_stars.append(patch_top_stars)
            
        ### Do this at each iteration in case the training stops early
        all_gd1_stars = pd.concat([df for df in target_stream])
        cwola_stars = pd.concat([df for df in top_stars])
        all_gd1_stars.to_hdf(save_folder+"/patches/patch{}/all_gd1_stars.h5".format(str(patch_id)), "df")
        cwola_stars.to_hdf(save_folder+"/patches/patch{}/cwola_stars.h5".format(str(patch_id)), "df")
                     
    ### Make Via Machinae plot
    plt.figure(dpi=200, figsize=(12,4), tight_layout=True)
    plt.scatter(all_gd1_stars.α, all_gd1_stars.δ, marker='.', s=2, 
                color="lightgray", label="GD1")
    plt.scatter(cwola_stars[cwola_stars.stream == False].α, cwola_stars[cwola_stars.stream == False].δ, marker='.', s=2, 
                color="darkorange", label="CWoLa (Non-Match)")
    plt.scatter(cwola_stars[cwola_stars.stream].α, cwola_stars[cwola_stars.stream].δ, marker='.', s=2, 
                color="crimson", label="CWoLa (Match)")
    plt.legend()
    plt.xlabel(r"$\alpha$ [\textdegree]");
    plt.ylabel(r"$\delta$ [\textdegree]");
    plt.xlim(115,230);
    plt.savefig(os.path.join(save_folder, "via_machinae_plot.png"))
                     
    print("CWoLa-identified stars:", cwola_stars.stream.value_counts())
                
    print("Finished in {:,.1f} seconds.".format(time.time() - t0))
          