### Generic imports 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import stats
from tqdm import tqdm 
from glob import glob
import random

### ML imports
from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential
from keras import callbacks, regularizers
from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from livelossplot.keras import PlotLossesCallback

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

def get_random_file():
    file = random.choice(glob("/data0/mpettee/gaia_data/mock_streams/*.npy"))
    return(file)

def load_file(stream = None, percent_bkg = 100):
    ### Stream options: ["mock", "gd1", "fjorm"]
    if stream == "mock": 
        file = get_random_file()
        df = pd.DataFrame(np.load(file), columns=['μ_δ','μ_α','δ','α','mag','color','a','b','c','d','stream'])
        df['stream'] = df['stream']/100
        df['stream'] = df['stream'].astype(bool)

    elif stream == "gd1": 
        print("Load GD1")
        df = pd.DataFrame()
        
    elif stream == "gd1_tail":
        file = "/data0/mpettee/gaia_data/gd1_tail/gd1_tail.h5"
        df = pd.read_hdf(file)
        
    ### Drop any rows containing a NaN value
    df.dropna(inplace = True)

    ### Restrict data to a radius of 15
    center_α = 0.5*(df.α.min()+df.α.max())
    center_δ = 0.5*(df.δ.min()+df.δ.max())
    df = df[np.sqrt((df.δ-center_δ)**2+(df.α-center_α)**2) < 15]        
    
    if percent_bkg != 100:
        ### Optional: reduce background
        n_sig = len(df[df.stream == True])
        n_bkg = len(df[df.stream == False])
        print("Before reduction, stream stars make up {:.3f}% of the dataset.".format(100*n_sig/len(df)))

        events_to_drop = np.random.choice(df[df.stream == False].index, int((1-percent_bkg/100)*n_bkg), replace=False)
        df = df.drop(events_to_drop)
        print("After reduction, stream stars make up {:.3f}% of the dataset.".format(100*n_sig/len(df)))
    
    return df

def visualize_stream(df, save_folder=None):
    plt.figure(figsize=(3,3), tight_layout=True) 
    plt.hist2d(df.α,df.δ,bins=100)
    if "stream" in df.keys():
        plt.scatter(df[df.stream == True].α,df[df.stream == True].δ,color='white',s=0.2, label="Stream")
    plt.xlabel(r"$\alpha$ [\textdegree]")
    plt.ylabel(r"$\delta$ [\textdegree]");
#     plt.legend();
    if save_folder is not None:
        plt.savefig(os.path.join(save_folder,"stream_position.png"))
    
    bins = np.linspace(-25,10,100)
    if "stream" in df.keys():
         fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6,3), tight_layout=True)
    else:
         fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3,3), tight_layout=True)
    
    if "stream" in df.keys(): ax = axs[0]
    ax.hist2d(df.μ_α*np.cos(df.δ),df.μ_δ, bins=[bins,bins])
    ax.set_xlabel(r"$\mu_\alpha\cos(\delta)$ [$\mu$as/year]", fontsize=11)
    ax.set_ylabel(r"$\mu_\delta$ [$\mu$as/year]", fontsize=11);
    ax.set_title("Background")
    
    if "stream" in df.keys():
        ax = axs[1]
        ax.hist2d(df[df.stream == True].μ_α*np.cos(df[df.stream == True].δ),df[df.stream == True].μ_δ, bins=[bins,bins])
        ax.set_xlabel(r"$\mu_\alpha\cos(\delta)$ [$\mu$as/year]", fontsize=11)
        ax.set_ylabel(r"$\mu_\delta$ [$\mu$as/year]", fontsize=11);
        ax.set_title("Stream");
        
    if save_folder is not None:
        plt.savefig(os.path.join(save_folder,"stream_velocities.png"))
        
def signal_sideband(df, stream, save_folder=None):
    if stream == "gd1_tail":
        sb_min = -6
        sb_max = 0
        sr_min = -4.5
        sr_max = -1.7
        
    elif stream == "mock":
        sb_min = df[df.stream].μ_δ.mean()-df[df.stream].μ_δ.std()/2
        sb_max = df[df.stream].μ_δ.mean()+df[df.stream].μ_δ.std()/2
        sr_min = df[df.stream].μ_δ.mean()-df[df.stream].μ_δ.std()/4
        sr_max = df[df.stream].μ_δ.mean()+df[df.stream].μ_δ.std()/4

    df_slice = df[(df.μ_δ > sb_min) & (df.μ_δ < sb_max)]
    df_slice['label'] = np.where(((df_slice.μ_δ > sr_min) & (df_slice.μ_δ < sr_max)), 1, 0)
    
    plt.figure(figsize=(4,3), tight_layout=True)
    bins = np.linspace(sb_min,sb_max,100)
    plt.hist(df_slice[df_slice.label == 1].μ_δ,bins=bins,color="dodgerblue",label="Signal Region")
    plt.hist(df_slice[df_slice.label == 0].μ_δ,bins=bins,color="orange",label="Sideband Region")
    plt.legend(frameon=False)
    plt.xlabel(r"$\mu_\delta$ [$\mu$as/year]")
    plt.ylabel("Counts")
    if save_folder is not None:
        plt.savefig(os.path.join(save_folder,"signal_sideband.png"))

    sr = df_slice[df_slice.label == 1]
    sb = df_slice[df_slice.label == 0]

    print("Signal region has {:,} stream and {:,} bkg events.".format(sr.stream.value_counts()[True], sr.stream.value_counts()[False]))
    print("Sideband region has {:,} stream and {:,} bkg events.".format(sb.stream.value_counts()[True], sb.stream.value_counts()[False]))
    print("Total counts: SR = {:,}, SB = {:,}".format(len(sr), len(sb)))
    return df_slice

def plot_results(test, save_folder=None):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8,3), constrained_layout=True)
    bins=np.linspace(0,1,10)

    ax = axs[0]
    ax.tick_params(labelsize=12)
    ax.hist(test[test.label == 1].nn_score, bins=bins, histtype='step', linewidth=2, color="dodgerblue", label="Signal Region")
    ax.hist(test[test.label == 0].nn_score, bins=bins, histtype='step', linewidth=2, color="orange", label="Sideband Region")
    ax.legend(fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_title("Test Set (15\% of full dataset)")
    ax.set_xlabel("NN Score", size=12)
    ax.set_ylabel("Events", size=12)

    ax = axs[1]
    ax.tick_params(labelsize=12)
    ax.hist(test[test.stream == False].nn_score, bins=bins, histtype='step', linewidth=2, color="grey", label="Not Stream")
    ax.hist(test[test.stream == True].nn_score, 
            bins=bins, histtype='step', linewidth=2, color="crimson", label="Stream")
    ax.legend(fontsize=12)
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_title("Test Set (15\% of full dataset)")
    ax.set_xlabel("NN Score", size=12)
    ax.set_ylabel("Events", size=12);
    if save_folder is not None: 
        plt.savefig(os.path.join(save_folder,"nn_scores.png"))
    
    ### Plot purities
    # Scan for optimal percentage
    cuts = np.linspace(0.1, 50, 100)
    efficiencies = []
    purities = []
    for x in cuts:
        top_stars = test[(test['nn_score'] >= test['nn_score'].quantile((100-x)/100))]
        if True in top_stars.stream.unique():
            n_perfect_matches = top_stars.stream.value_counts()[True]
            stream_stars_in_test_set = test[test.stream == True]
            efficiencies.append(100*n_perfect_matches/len(stream_stars_in_test_set))
            purities.append(n_perfect_matches/len(top_stars)*100)
        else:
            efficiencies.append(np.nan)
            purities.append(np.nan)

    ### Choose a cut to optimize purity
    print("Maximum purity of {:.1f}% at {:.1f}%".format(np.nanmax(purities),cuts[np.nanargmax(purities)]))
    cut = cuts[np.nanargmax(purities)]
    plt.figure()
    plt.plot(cuts, purities, label="Signal Purity")
    plt.xlabel("Top \% Stars, ranked by NN score")
    plt.legend()    
    if save_folder is not None: 
        plt.savefig(os.path.join(save_folder,"purities.png"))

    ### Plot highest-ranked stars
    for x in [0.1, 1, 5, 10, 20]: # percentages
        top_stars = test[(test['nn_score'] >= test['nn_score'].quantile((100-x)/100))]
        stream_stars_in_test_set = test[test.stream == True]

        if True in top_stars.stream.unique(): 
            n_perfect_matches = top_stars.stream.value_counts()[True] 
        else: 
            n_perfect_matches = 0 
            print("No stream matches identified in top star selection.")
        
        print("Cut at {}%...".format(x))
        print("Efficiency: {:.1f}%".format(100*n_perfect_matches/len(stream_stars_in_test_set)))
        print("Purity: {:.1f}%".format(n_perfect_matches/len(top_stars)*100))

        plt.figure(figsize=(5,3), tight_layout=True) 
        plt.title('Top {:.1f}\% NN Scores'.format(x))
#         plt.title("Purity = {:.0f}\%".format(n_perfect_matches/len(top_stars)*100), fontsize=11)
        plt.scatter(stream_stars_in_test_set.α, stream_stars_in_test_set.δ, marker='.', 
                    color = "lightgray",
                    label='Stream')
        plt.scatter(top_stars.α, top_stars.δ, marker='.', 
                    color = "lightpink",
                    label="Top Stars\n(Purity = {:.0f}\%)".format(n_perfect_matches/len(top_stars)*100))
        if True in top_stars.stream.unique(): 
            plt.scatter(top_stars[top_stars.stream].α, top_stars[top_stars.stream].δ, marker='.', 
                    color = "crimson",
                    label='Matches')
        plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
        plt.xlim(-15,15)
        plt.ylim(-15,15)
        plt.xlabel(r"$\alpha$ [\textdegree]")
        plt.ylabel(r"$\delta$ [\textdegree]")
        if save_folder is not None: 
            plt.savefig(os.path.join(save_folder,"top_{}%_stars.png".format(x)))

    