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

### ML imports
from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential
from keras import callbacks, regularizers
from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf

### Custom imports
from functions import *
from models import *

### GPU Setup
os.environ["CUDA_VISIBLE_DEVICES"] = "2" # pick a number < 4 on ML4HEP; < 3 on Voltan 
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
    parser.add_argument("--save_label", default='test', type=str, help="Folder name for saving training outputs & plots.")
    parser.add_argument("--percent_bkg", default=100, type=int, help="Percent of background to train on.")
    parser.add_argument("--layer_size", default=128, type=int, help="Number of nodes per layer.")
    parser.add_argument("--patience", default=30, type=int, help="How many epochs of no val_loss improvement before the training is stopped.")
    parser.add_argument("--epochs", default=200, type=int, help="Number of training epochs.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size during training.")
    parser.add_argument("--dropout", default=0, type=float, help="Dropout probability.")
    parser.add_argument("--l2_reg", default=0, type=float, help="L2 regularization.")
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
    df, file = load_file(stream = args.stream, percent_bkg = args.percent_bkg)
        
    visualize_stream(df, save_folder = save_folder)
    
    ### Define signal & sideband regions 
    df_slice = signal_sideband(df, stream = args.stream, save_folder = save_folder)
    
    ### Train
    train(df_slice, layer_size=args.layer_size, dropout=args.dropout, l2_reg=args.l2_reg, epochs=args.epochs, patience=args.patience, n_folds=1, save_folder=save_folder)
    
    print("Finished in {:,.1f} seconds.".format(time.time() - t0))
          