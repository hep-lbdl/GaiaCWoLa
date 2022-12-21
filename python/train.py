### Generic imports 
import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
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
# import wandb
# from wandb.keras import WandbCallback

### Custom imports
import sys
sys.path.append('../')
from functions import *
from models import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("stream", default=None, choices = ["gd1", "gd1_tail", "mock"], help="Choose which stream to analyze.")
    parser.add_argument("--data_folder", default="gaia_data", help="Specify the folder where the stream data lies.")
    parser.add_argument("--save_label", default='test', type=str, help="Folder name for saving training outputs & plots.")
    parser.add_argument("--layer_size", default=200, type=int, help="Number of nodes per layer.")
    parser.add_argument("--patience", default=30, type=int, help="How many epochs of no val_loss improvement before the training is stopped.")
    parser.add_argument("--sr_factor", default=1, type=float, help="Multiplicative factor for sigma to define signal region.")
    parser.add_argument("--sb_factor", default=3, type=float, help="Multiplicative factor for sigma to define sideband region.")
    parser.add_argument("--epochs", default=100, type=int, help="Number of training epochs.")
    parser.add_argument("--batch_size", default=10000, type=int, help="Batch size during training.")
    parser.add_argument("--dropout", default=0.2, type=float, help="Dropout probability.")
    parser.add_argument("--n_folds", default=5, type=int, help="Number of k-folds.")
    parser.add_argument("--sample_weight", default=1, type=float, help="If not equal to 1, adds an additional weight to each star in the stream.")
    parser.add_argument("--best_of_n_loops", default=3, type=int, help="Repeats the training N times and picks the best weights.")
    parser.add_argument("--gpu_id", default=-1, type=int, help="Specify GPU to use. (Use -1 for CPU only)")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    t0 = time.time()
    
    ### GPU Setup
    if args.gpu_id > -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id) # pick a number < 4 on ML4HEP; < 3 on Voltan; use -1 for CPU only
        physical_devices = tf.config.list_physical_devices('GPU') 
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        print("Using CPU only")
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"

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
    
#     wandb.init(project="gd1-tail", entity="mpettee")
#     wandb.config = {
#       "epochs": args.epochs,
#       "batch_size": args.batch_size,
#       "layer_size": args.layer_size, 
#     }
    
    from pathlib import Path
    path = Path(__file__)
    base_dir = path.parent.parent.absolute() # main working directory
    
    save_label = args.save_label
    save_folder = os.path.join(base_dir, "trained_models", save_label)
    os.makedirs(save_folder, exist_ok=True)
    
    ### Setup logging
    import logging
    logging.basicConfig(filename=save_folder+'/log.log', level=logging.INFO)
    logging.info("Stream = "+str(args.stream))
    
    ### Save arguments
    with open(os.path.join(save_folder,'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    ### Load file & preprocess
    df, file = load_file(stream = args.stream, folder = args.data_folder)
        
    ### Make plots
    plot_coords(df, save_folder = save_folder)
    
    ### Define signal & sideband regions 
    logging.info("Signal Region scale factor = "+str(args.sr_factor))
    logging.info("Sideband Region scale factor = "+str(args.sb_factor))
    df = signal_sideband(df, sr_factor = args.sr_factor, sb_factor=args.sb_factor, stream = args.stream, save_folder = save_folder)

    print(args)
    
    ### Train
    test = train(df, 
                 layer_size=args.layer_size, 
                 batch_size=args.batch_size, 
                 dropout=args.dropout, 
                 epochs=args.epochs, 
                 patience=args.patience, 
                 n_folds=args.n_folds, 
                 best_of_n_loops=args.best_of_n_loops, 
                 save_folder=save_folder, 
                 verbose=False,
        #           other_callbacks=[WandbCallback()]
         )

    print("======== SUMMARY ========)
    for x in os.listdir(save_folder):
        if x.startswith("kfold"):
            for y in os.listdir(x+"/after_fiducial_cuts/"):
                if y.startswith("top_"):
                    print(x, y)
        elif x.startswith("before_"):
            for y in os.listdir(x):
                if y.startswith("top_"):
                    print(x, y)
        elif x.startswith("after_"):
            for y in os.listdir(x):
                if y.startswith("top_"):
                    print(x, y)
    print("Finished in {:,.1f} seconds.".format(time.time() - t0))
          