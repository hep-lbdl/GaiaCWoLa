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

### ML imports
from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential
from keras import callbacks, regularizers
from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback

### Custom imports
from functions import *
from models import *

### GPU Setup
os.environ["CUDA_VISIBLE_DEVICES"] = str(randrange(4)) # pick a number < 4 on ML4HEP; < 3 on Voltan 
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
    parser.add_argument("--layer_size", default=64, type=int, help="Number of nodes per layer.")
    parser.add_argument("--patience", default=30, type=int, help="How many epochs of no val_loss improvement before the training is stopped.")
    parser.add_argument("--epochs", default=200, type=int, help="Number of training epochs.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size during training.")
    parser.add_argument("--dropout", default=0.2, type=float, help="Dropout probability.")
    parser.add_argument("--l2_reg", default=0, type=float, help="L2 regularization.")
    parser.add_argument("--n_folds", default=1, type=int, help="Number of k-folds.")
    parser.add_argument("--sample_weight", default=1, type=float, help="If not equal to 1, adds an additional weight to each star in the stream.")
    parser.add_argument("--best_of_n_loops", default=3, type=int, help="Repeats the training N times and picks the best weights.")


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
    df_all = pd.read_hdf("gd1_allpatches.h5")
    visualize_stream(df_all, save_folder=save_folder)
    
    target_stream = []
    top_stars = []
    
    for patch_id in tqdm(df_all.patch_id.unique(), desc="Patches"):
        df = df_all[(df_all.patch_id == patch_id)]
        patch_folder = save_folder+"/patches/patch{}".format(str(patch_id))
        os.makedirs(patch_folder, exist_ok=True)
        visualize_stream(df, save_folder=patch_folder)
        target_stream.append(df[df.stream])
        df_train = signal_sideband(df,
                        sb_min = df[df.stream].μ_δ.quantile(0.01), 
                        sr_min = df[df.stream].μ_δ.quantile(0.2), 
                        sr_max = df[df.stream].μ_δ.quantile(0.8),
                        sb_max = df[df.stream].μ_δ.quantile(0.99),
                        save_folder=patch_folder, 
                            )
        ### Add stream weights
        stream_weight = args.sample_weight
        df_train["weight"] = np.where(df_train['stream'] == True, stream_weight, 1)

        tf.keras.backend.clear_session()
        
        ### wandb setup
        wandb.init(project="my-test-project", entity="mpettee")
        wandb.config = {
          "epochs": args.epochs,
          "batch_size": args.batch_size,
        }
        
        test = train(df_train, 
          n_folds = args.n_folds, 
          best_of_n_loops = args.best_of_n_loops,
          layer_size = args.layer_size, 
          batch_size = args.batch_size, 
          dropout = args.dropout, 
          epochs = args.epochs, 
          patience = args.patience,
          save_folder=save_folder+"/patches/patch{}".format(str(patch_id)),
          callbacks=[WandbCallback()]
                    )
        # Grab top 100 stars
        patch_top_stars = test.sort_values('nn_score',ascending=False)[:100]
        patch_top_stars.to_hdf(patch_folder+"/top_stars.h5", "df")
        top_stars.append(patch_top_stars)
    
    all_gd1_stars = pd.concat([df for df in target_stream])
    cwola_stars = pd.concat([df for df in top_stars])
    
    ### Make Via Machinae plot
    plt.figure(dpi=200, figsize=(12,4))
    plt.scatter(all_gd1_stars.α, all_gd1_stars.δ, marker='.', s=1, 
                color="lightgray", label="GD1")
    plt.scatter(cwola_stars[cwola_stars.stream == False].α, cwola_stars[cwola_stars.stream == False].δ, marker='.', s=1, 
                color="lightpink", label="CWoLa (Non-Match)")
    plt.scatter(cwola_stars[cwola_stars.stream].α, cwola_stars[cwola_stars.stream].δ, marker='.', s=1, 
                color="crimson", label="CWoLa (Match)")
    plt.xlabel(r"$\alpha$ [\textdegree]")
    plt.ylabel(r"$\delta$ [\textdegree]")
    plt.legend();
    plt.savefig(os.path.join(save_folder, "via_machinae_plot.png"))
                
    print("Finished in {:,.1f} seconds.".format(time.time() - t0))
          