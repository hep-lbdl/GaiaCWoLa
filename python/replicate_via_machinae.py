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
import tensorflow as tf
from multiprocessing import Pool, cpu_count
# import wandb
# from wandb.keras import WandbCallback

### Custom imports
import sys
sys.path.append('./python')
from functions import *
from models import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_label", default='test', type=str, help="Folder name for saving training outputs & plots.")
    parser.add_argument("--n_patches", default=21, type=int, help="Number of patches to train over.")
    parser.add_argument("--layer_size", default=256, type=int, help="Number of nodes per layer.")
    parser.add_argument("--patience", default=20, type=int, help="How many epochs of no val_loss improvement before the training is stopped.")
    parser.add_argument("--epochs", default=100, type=int, help="Number of training epochs.")
    parser.add_argument("--batch_size", default=10000, type=int, help="Batch size during training.")
    parser.add_argument("--dropout", default=0.2, type=float, help="Dropout probability.")
    parser.add_argument("--n_folds", default=5, type=int, help="Number of k-folds.")
    parser.add_argument("--sample_weight", default=1, type=float, help="If not equal to 1, adds an additional weight to each star in the stream.")
    parser.add_argument("--sr_factor", default=1, type=float, help="Multiplicative factor for sigma to define signal region.")
    parser.add_argument("--sb_factor", default=3, type=float, help="Multiplicative factor for sigma to define sideband region.")
    parser.add_argument("--best_of_n_loops", default=3, type=int, help="Repeats the training N times and picks the best weights.")
    parser.add_argument("--gpu_id", default=-1, type=int, help="Choose a GPU to run over (or -1 if you want to use CPU only).")
    parser.add_argument("--scan_over_mu_phi", action="store_true")
    parser.add_argument("--train_after_cuts", action="store_true")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    t0 = time.time()
    
    ### GPU Setup
    if args.gpu_id != -1: 
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        physical_devices = tf.config.list_physical_devices('GPU') 
        print(physical_devices)
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else: 
        print("Using CPU only")
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    
    save_label = args.save_label
    save_folder = os.path.join("./trained_models",save_label)
    os.makedirs(save_folder, exist_ok=True)
    
    print("Saving to",save_folder)
    
    ### Save arguments
    with open(os.path.join(save_folder,'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    import os
    print(os.getcwd())
    
    patch_list = [
     # b = 33.7 
     './gaia_data/gd1/gaiascan_l195.0_b33.7_ra128.4_dec28.8.npy',
     './gaia_data/gd1/gaiascan_l210.0_b33.7_ra132.6_dec16.9.npy',
     './gaia_data/gd1/gaiascan_l225.0_b33.7_ra138.1_dec5.7.npy', 
     # b = 41.8 
     './gaia_data/gd1/gaiascan_l187.5_b41.8_ra136.5_dec36.1.npy',
     './gaia_data/gd1/gaiascan_l202.5_b41.8_ra138.8_dec25.1.npy',
     './gaia_data/gd1/gaiascan_l217.5_b41.8_ra142.7_dec14.5.npy', 
     # b = 50.2 
     './gaia_data/gd1/gaiascan_l99.0_b50.2_ra224.7_dec60.6.npy',
     './gaia_data/gd1/gaiascan_l117.0_b50.2_ra202.4_dec66.5.npy',
     './gaia_data/gd1/gaiascan_l135.0_b50.2_ra174.3_dec65.1.npy',
     './gaia_data/gd1/gaiascan_l153.0_b50.2_ra156.2_dec57.5.npy',
     './gaia_data/gd1/gaiascan_l171.0_b50.2_ra148.6_dec47.0.npy',
     './gaia_data/gd1/gaiascan_l189.0_b50.2_ra146.9_dec35.6.npy',
     './gaia_data/gd1/gaiascan_l207.0_b50.2_ra148.6_dec24.2.npy',
     # b = 58.4 
     './gaia_data/gd1/gaiascan_l101.2_b58.4_ra212.7_dec55.2.npy',
     './gaia_data/gd1/gaiascan_l123.8_b58.4_ra192.0_dec58.7.npy',
     './gaia_data/gd1/gaiascan_l146.2_b58.4_ra171.8_dec54.7.npy',
     './gaia_data/gd1/gaiascan_l168.8_b58.4_ra160.5_dec45.5.npy',
     './gaia_data/gd1/gaiascan_l191.2_b58.4_ra156.9_dec34.1.npy',
     # b = 66.4 
     './gaia_data/gd1/gaiascan_l105.0_b66.4_ra203.7_dec49.1.npy',
     './gaia_data/gd1/gaiascan_l135.0_b66.4_ra185.4_dec50.0.npy',
     './gaia_data/gd1/gaiascan_l165.0_b66.4_ra171.4_dec43.0.npy',    
    ]

    ## Scan over patches
    target_stream = []
    top_stars = []
    n_patches = args.n_patches

    def train_on_patch(patch_id):
        df = pd.read_hdf(patch_list[patch_id][:-4]+".h5")
        
        if args.train_after_cuts:
            ### Apply fiducial cuts BEFORE training
            df = fiducial_cuts(df)
        
        os.makedirs(save_folder+"/patches/patch{}".format(str(patch_id)), exist_ok=True)
        make_plots(df, save_folder=save_folder+"/patches/patch{}".format(str(patch_id)))
        df_train = signal_sideband(df, save_folder=save_folder+"/patches/patch{}".format(str(patch_id)),
                      sr_factor = args.sr_factor,
                      sb_factor = args.sb_factor,
                        verbose=False,
                        scan_over_mu_phi=args.scan_over_mu_phi
                        )
        tf.keras.backend.clear_session()
        test = train(df_train, 
          n_folds = args.n_folds, 
          best_of_n_loops = args.best_of_n_loops,
          layer_size = args.layer_size, 
          batch_size = args.batch_size, 
          scan_over_mu_phi = args.scan_over_mu_phi,
          dropout = args.dropout, 
          epochs = args.epochs, 
          patience = args.patience,
          save_folder=save_folder+"/patches/patch{}".format(str(patch_id)),
          verbose = False,
        )
        print("Finished Patch #{}".format(str(patch_id)))
        return test

    if args.gpu_id == -1: ### multiprocessing on CPU
        pool = Pool(processes=8) # max = cpu_count()
        results = pool.map(train_on_patch, np.arange(21)) # for same 21 patches as Via Machinae
        pool.close()
        pool.join()    

    else: ### if using a GPU, just run the patches in order
        results = []
        for patch_id in tqdm(np.arange(21), desc="Patches"):
            results.append(train_on_patch(patch_id))
    
    all_gd1_stars = []
    cwola_stars = []

    for test in results:
        n_top_stars = np.min([len(test[test.stream]),100]) # whichever's smaller: 100, or the number of stars in the patch
        patch_top_stars = test.sort_values('nn_score',ascending=False)[:n_top_stars]
        all_gd1_stars.append(test[test.stream])
        cwola_stars.append(patch_top_stars)

    all_gd1_stars = pd.concat([df for df in all_gd1_stars])
    cwola_stars = pd.concat([df for df in cwola_stars])
    
    all_gd1_stars.reset_index(inplace=True)
    cwola_stars.reset_index(inplace=True)
    
    all_gd1_stars.drop_duplicates(subset = 'index')
    cwola_stars.drop_duplicates(subset = 'index')

    plt.figure(dpi=200, figsize=(12,4))
    plt.scatter(all_gd1_stars.α_wrapped-360, all_gd1_stars.δ, marker='.', s=2, 
                color="lightgray", label="GD1")
    plt.scatter(cwola_stars[cwola_stars.stream == False].α_wrapped-360, cwola_stars[cwola_stars.stream == False].δ, marker='.', s=2, 
                color="darkorange", label="CWoLa (Non-Match)")
    plt.scatter(cwola_stars[cwola_stars.stream].α_wrapped-360, cwola_stars[cwola_stars.stream].δ, marker='.', s=2, 
                color="crimson", label="CWoLa (Match)")
    plt.legend()
    plt.xlabel(r"$\alpha$ [\textdegree]");
    plt.xlim(-241,-135);
    plt.savefig(os.path.join(save_folder, "via_machinae_plot.png"))

    print("CWoLa-identified stars:", cwola_stars.stream.value_counts())
    print("Purity = {:.0f}% in CWoLa-identified stars".format(100*len(cwola_stars[cwola_stars.stream])/len(cwola_stars)))
    print("Purity = {:.0f}% vs. all of GD-1".format(100*len(cwola_stars[cwola_stars.stream])/len(all_gd1_stars)))
    print("Finished in {:,.1f} hours.".format((time.time() - t0)/60/60))
          