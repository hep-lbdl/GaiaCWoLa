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
    parser.add_argument("stream", default=None, choices = ["gd1", "gd1_tail", "mock"], help="Choose which stream to analyze.")
    parser.add_argument("--data_folder", default="../gaia_data/", help="Specify the folder where the stream data lies.")
    parser.add_argument("--save_label", default='test', type=str, help="Folder name for saving training outputs & plots.")
    parser.add_argument("--percent_bkg", default=100, type=int, help="Percent of background to train on.")
    parser.add_argument("--layer_size", default=128, type=int, help="Number of nodes per layer.")
    parser.add_argument("--patience", default=20, type=int, help="How many epochs of no val_loss improvement before the training is stopped.")
    parser.add_argument("--epochs", default=200, type=int, help="Number of training epochs.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size during training.")
    parser.add_argument("--dropout", default=0.2, type=float, help="Dropout probability.")
    parser.add_argument("--l2_reg", default=0, type=float, help="L2 regularization.")
    parser.add_argument("--n_folds", default=1, type=int, help="Number of k-folds.")
    parser.add_argument("--sample_weight", default=1, type=float, help="If not equal to 1, adds an additional weight to each star in the stream.")
    parser.add_argument('--remove_stream_sb', action='store_true', help="Use this if you want to remove stream stars from the sideband region.")
    parser.add_argument("--best_of_n_loops", default=1, type=int, help="Repeats the training N times and picks the best weights.")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    t0 = time.time()
    
#     wandb.init(project="gd1-tail", entity="mpettee")
#     wandb.config = {
#       "epochs": args.epochs,
#       "batch_size": args.batch_size,
#       "layer_size": args.layer_size, 
#     }
    
    save_label = args.save_label
    save_folder = os.path.join("../trained_models",save_label)
    os.makedirs(save_folder, exist_ok=True)
    
    ### Save arguments
    with open(os.path.join(save_folder,'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    ### Load file & preprocess
    df, file = load_file(stream = args.stream, folder = args.data_folder, percent_bkg = args.percent_bkg)
        
    visualize_stream(df, save_folder = save_folder)
    
    ### Define signal & sideband regions 
    df = signal_sideband(df, stream = args.stream, save_folder = save_folder)
    
    ### Remove stream stars from sideband
    if args.remove_stream_sb:
        df = df[((df.label == 0) & (df.stream == False)) | (df.label == 1)]
    
    ### Add sample weights to stream stars
    df["weight"] = np.where(df['stream']==True, args.sample_weight, 1)
    
    ### Train
    train(df, layer_size=args.layer_size, dropout=args.dropout, l2_reg=args.l2_reg, epochs=args.epochs, patience=args.patience, n_folds=args.n_folds, best_of_n_loops=args.best_of_n_loops, save_folder=save_folder, 
#           other_callbacks=[WandbCallback()]
         )
    
    print("Finished in {:,.1f} seconds.".format(time.time() - t0))
          