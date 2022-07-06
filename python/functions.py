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
    'figure.dpi': 100,
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.family": "serif",
    "font.size": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.max_open_warning": False,
})

def fiducial_cuts(df):
    δ_mid = 0.5*(df.δ.max() + df.δ.min())
    α_mid = 0.5*(df.α.max() + df.α.min())
    df = df[((df.δ-δ_mid)**2 + (df.α-α_mid)**2) < 10**2] # central 10 degree circle, to avoid edge effects
    df = df[df.mag < 20.2]
    df = df[(0.5 < df.color) & (df.color < 1)]
    return(df)

#function from David's file via_machinae.py
def angular_distance(angle1,angle2):
    # inputs are np arrays of [ra,dec]
    deltara=np.minimum(np.minimum(np.abs(angle1[:,0]-angle2[:,0]+360),np.abs(angle1[:,0]-angle2[:,0])),\
                          np.abs(angle1[:,0]-angle2[:,0]-360))
    deltadec=np.abs(angle1[:,1]-angle2[:,1])
    return np.sqrt(deltara**2+deltadec**2)

def FilterGD1(stars, gd1_stars):
    gd1stars=np.zeros(len(stars))
    for x in gd1_stars:
        ra=x[0]
        dec=x[1]
        pmra=x[2]
        pmdec=x[3]
        foundlist=angular_distance(np.dstack((stars[:,3],stars[:,2]))[0],np.array([[ra,dec]]))
        foundlist=np.sqrt(foundlist**2+(stars[:,0]-pmdec)**2+(stars[:,1]-pmra)**2)   
        foundlist=foundlist<.0001
        if len(np.argwhere(foundlist))>1:
            print(foundlist)
        if len(np.argwhere(foundlist))==1:
            gd1stars+=foundlist
    gd1stars=gd1stars.astype('bool')
    return gd1stars,stars[gd1stars]

def get_random_file(glob_path):
    file = random.choice(glob(glob_path))
    return(file)

def load_file(stream = None, folder = "../gaia_data/", percent_bkg = 100):
    ### Stream options: ["gd1", "gd1_tail", "mock", "jhelum"]
    if stream == "mock": 
        file = "../gaia_data/mock_streams/gaiamock_ra156.2_dec57.5_stream_feh-1.6_v3_863.npy"
#         file = get_random_file(os.path.join(folder,"mock_streams/*.npy"))
        print(file)
        df = pd.DataFrame(np.load(file), columns=['μ_δ','μ_α','δ','α','mag','color','a','b','c','d','stream'])
        df['stream'] = df['stream']/100
        df['stream'] = df['stream'].astype(bool)

    elif stream == "gd1": 
        file = os.path.join(folder,"gd1/GD1-circle-140-30-15.pkl")
        df = np.load(file, allow_pickle = True)

        ### Select columns
        columns = ['pmdec','pmra','dec','ra','phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag','streammask']
        df = df[columns]

        ### Create b-r & g columns; rename others
        df["b-r"] = df.phot_bp_mean_mag - df.phot_rp_mean_mag
        df.drop(columns = ['phot_bp_mean_mag','phot_rp_mean_mag'], inplace=True)
        df.rename(columns={'phot_g_mean_mag': 'g', 
                           'ra': 'α',
                           'dec': 'δ',
                           'pmra': 'μ_α',
                           'pmdec': 'μ_δ',
                           'streammask': 'stream'}, inplace=True)

    elif stream == "gd1_tail":
#         file = os.path.join(folder,"gd1_tail/gd1_tail.h5")
        file = os.path.join(folder,"gd1_tail/gd1_tail_optimized_patch.h5")
        df = pd.read_hdf(file)

    elif stream == "gaia3":
        ### Note that we don't have stream labels here
        file = get_random_file(os.path.join(folder,"gaia3/*.npy"))
        print(file)
        df = pd.DataFrame(np.load(file)[:,[9,8,6,7,4,5]], columns=['μ_δ','μ_α','δ','α','mag','color'])        
    elif stream == "jhelum":
        ### Note that we don't have stream labels here
        file = get_random_file(os.path.join(folder,"jhelum/*.npy"))
#         file = "../GaiaCWoLa/gaia_data/jhelum/gaiascan_l303.8_b58.4_ra193.3_dec-4.5.npy"
        print(file)
        df = pd.DataFrame(np.load(file)[:,[9,8,6,7,4,5]], columns=['μ_δ','μ_α','δ','α','mag','color'])
    else:
        print("Stream not recognized.")
        
    ### Drop any rows containing a NaN value
    df.dropna(inplace = True)

    ### Restrict data to a radius of 15
    center_α = 0.5*(df.α.min()+df.α.max())
    center_δ = 0.5*(df.δ.min()+df.δ.max())
    df = df[np.sqrt((df.δ-center_δ)**2+(df.α-center_α)**2) < 15]
    
    if percent_bkg != 100 and "stream" in df.keys():
        ### Optional: reduce background
        n_sig = len(df[df.stream == True])
        n_bkg = len(df[df.stream == False])
        print("Before reduction, stream stars make up {:.3f}% of the dataset.".format(100*n_sig/len(df)))

        events_to_drop = np.random.choice(df[df.stream == False].index, int((1-percent_bkg/100)*n_bkg), replace=False)
        df = df.drop(events_to_drop)
        print("After reduction, stream stars make up {:.3f}% of the dataset.".format(100*n_sig/len(df)))
    
    df.reset_index(inplace=True)
    return df, file

def plot_coords(df, save_folder=None):
    fig, axs = plt.subplots(nrows=2, ncols=3, dpi=200, figsize=(10,7), tight_layout=True)
    
    bins_α=np.linspace(df.α.min(),df.α.max(),100)
    bins_δ=np.linspace(df.δ.min(),df.δ.max(),100)
    
    cmap="binary"
    labelsize=14
    
    ax = axs[0,0]
    ax.hist2d(df.α,df.δ, bins=[bins_α,bins_δ], cmap=cmap)
    ax.set_xlabel(r"$\alpha$ [\textdegree]", fontsize=labelsize)
    ax.set_ylabel(r"$\delta$ [\textdegree]", fontsize=labelsize);
    ax.set_title("Full Dataset", fontsize=16)
    
    ax = axs[1,0]
    ax.hist2d(df[df.stream == True].α,df[df.stream == True].δ, bins=[bins_α,bins_δ], cmap=cmap)
    ax.set_xlabel(r"$\alpha$ [\textdegree]", fontsize=labelsize)
    ax.set_ylabel(r"$\delta$ [\textdegree]", fontsize=labelsize);
    ax.set_title("Stream Only", fontsize=16);
    
    bins = np.linspace(-25,10,100)
    ax = axs[0,1]
    ax.hist2d(df.μ_α*np.cos(df.δ),df.μ_δ, bins=[bins,bins], cmap=cmap)
    ax.set_xlabel(r"$\mu_\alpha\cos(\delta)$ [mas/year]", fontsize=labelsize)
    ax.set_ylabel(r"$\mu_\delta$ [mas/year]", fontsize=labelsize);
    ax.set_title("Full Dataset", fontsize=16)
    
    ax = axs[1,1]
    ax.hist2d(df[df.stream == True].μ_α*np.cos(df[df.stream == True].δ), df[df.stream == True].μ_δ, bins=[bins,bins], cmap=cmap)
    ax.set_xlabel(r"$\mu_\alpha\cos(\delta)$ [mas/year]", fontsize=labelsize)
    ax.set_ylabel(r"$\mu_\delta$ [mas/year]", fontsize=labelsize);
    ax.set_title("Stream Only", fontsize=16);
    
    bins_color=np.linspace(df.color.min(),df.color.max(),100)
    bins_mag=np.linspace(df.mag.min(),df.mag.max(),100)
    
    ax = axs[0,2]
    ax.hist2d(df.color,df.mag, bins=[bins_color,bins_mag], cmap=cmap)
    ax.set_ylim(ax.get_ylim()[::-1]) # reverse y axis to match Via Machinae plot
    ax.set_xlabel(r"$b-r$", fontsize=labelsize)
    ax.set_ylabel(r"$g$", fontsize=labelsize);
    ax.set_title("Full Dataset", fontsize=16)
    
    ax = axs[1,2]
    ax.hist2d(df[df.stream == True].color,df[df.stream == True].mag, bins=[bins_color,bins_mag], cmap=cmap)
    ax.set_ylim(ax.get_ylim()[::-1]) # reverse y axis to match Via Machinae plot
    ax.set_xlabel(r"$b-r$", fontsize=labelsize)
    ax.set_ylabel(r"$g$", fontsize=labelsize);
    ax.set_title("Stream Only", fontsize=16);
    
    if save_folder is not None:
        plt.savefig(os.path.join(save_folder,"input_variables.png"))
        plt.savefig(os.path.join(save_folder,"input_variables.pdf"))
    
def visualize_stream(df, save_folder=None):
    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)
    plot_coords(df, save_folder)

    plt.figure(dpi=150) 
    bins = np.linspace(df.μ_δ.min(),df.μ_δ.max(),100) 
    plt.hist(df[df.stream == False].μ_δ, density=False, color="lightgray", histtype="stepfilled", linewidth=2, bins=bins, label="Background");
    plt.hist(df[df.stream].μ_δ, density=False, color="crimson", histtype="stepfilled", linewidth=2, bins=bins, label="GD-1")
    plt.title('GD-1 Proper Velocity')
    plt.xlabel(r'$\mu_\delta$')
    plt.ylabel('Counts')
    plt.yscale('log')
    plt.legend();
    if save_folder is not None:
        plt.savefig(os.path.join(save_folder,"mu_delta_zoomed_in.png"))
        
def signal_sideband(df, stream=None, save_folder=None, sb_min=None, sb_max=None, sr_min=None, sr_max=None, verbose=True):
    if sb_min is not None:
        sb_min = sb_min
        sb_max = sb_max
        sr_min = sr_min
        sr_max = sr_max
        
    # elif stream == "gd1_tail":
        ### Optimized GD1 tail w/ overlapping patches 
        # sb_min = -7
        # sr_min = -6
        # sr_max = -3.1
        # sb_max = -3
        
    elif stream == "mock":
        sb_min = df[df.stream].μ_δ.mean()-df[df.stream].μ_δ.std()/2
        sb_max = df[df.stream].μ_δ.mean()+df[df.stream].μ_δ.std()/2
        sr_min = df[df.stream].μ_δ.mean()-df[df.stream].μ_δ.std()/4
        sr_max = df[df.stream].μ_δ.mean()+df[df.stream].μ_δ.std()/4

    elif stream == "gd1": 
        sb_min = -18
        sr_min = -15
        sr_max = -11
        sb_max = -9.5
        
    elif stream == "jhelum":
        sb_min = df.μ_δ.mean()-df.μ_δ.std()/2
        sb_max = df.μ_δ.mean()+df.μ_δ.std()/2
        sr_min = df.μ_δ.mean()-df.μ_δ.std()/4
        sr_max = df.μ_δ.mean()+df.μ_δ.std()/4        
        
    else: 
        sb_min = df[df.stream].μ_δ.min()
        sb_max = df[df.stream].μ_δ.max()
        sr_min = sb_min+1
        sr_max = sb_max-1
        
    # plt.figure(dpi=150)
    # bins=np.linspace(df.μ_δ.min(),df.μ_δ.max(),50)
    # plt.hist(df.μ_δ, bins=bins, color="lightgray", label="Background");
    # if "stream" in df.keys(): plt.hist(df[df.stream == True].μ_δ, bins=bins, color="crimson", label="Stream");
    # plt.yscale('log')
    # plt.legend(frameon=False)
    # plt.xlabel(r"$\mu_\delta$ [$\mu$as/year]")
    # plt.ylabel("Counts")
    # if save_folder is not None:
    #     os.makedirs(save_folder, exist_ok=True)
    #     plt.savefig(os.path.join(save_folder,"mu_delta.png"))
    
    if verbose:
        print("Sideband region: [{:.1f},{:.1f}]".format(sb_min,sb_max))
        print("Signal region: [{:.1f},{:.1f}]".format(sr_min,sr_max))
    
    df_slice = df[(df.μ_δ > sb_min) & (df.μ_δ < sb_max)]
    df_slice['label'] = np.where(((df_slice.μ_δ > sr_min) & (df_slice.μ_δ < sr_max)), 1, 0)
    
    sr = df_slice[df_slice.label == 1]
    sb = df_slice[df_slice.label == 0]
    if verbose: print("Total counts: SR = {:,}, SB = {:,}".format(len(sr), len(sb)))
    
    plt.figure(figsize=(4,3), dpi=150, tight_layout=True)
    bins = np.linspace(sb_min,sb_max,50)
    plt.hist(sr.μ_δ,bins=bins,color="dodgerblue",label="Signal Region")
    plt.hist(sb.μ_δ,bins=bins,color="orange",label="Sideband Region")
    plt.legend(frameon=False)
    plt.xlabel(r"$\mu_\delta$ [$\mu$as/year]")
    plt.ylabel("Counts")
    if save_folder is not None:
        plt.savefig(os.path.join(save_folder,"signal_sideband.png"))
        
    if "stream" in df.keys():
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4,3), dpi=150, tight_layout=True)
        bins = np.linspace(sb_min,sb_max,30)
        ax.hist([df[df.stream].μ_δ, df[df.stream == False].μ_δ], bins=bins, stacked=True, color=["crimson","lightgray"], label=["Stream", "Background"])
        ax.set_yscale('log')
        ax.set_xlabel(r"$\mu_\delta$ [$\mu$as/year]")
        ax.set_ylabel("Counts")
        ax.axvline(x=sr_min, color="black")
        ax.axvline(x=sr_max, color="black")
        ax.legend(frameon=False)
        if save_folder is not None:
            plt.savefig(os.path.join(save_folder,"signal_sideband_composition.png"))

    if "stream" in df.keys():
        try: n_sig_stream_stars = sr.stream.value_counts()[True]
        except: n_sig_stream_stars = 0
        try: n_sideband_stream_stars = sb.stream.value_counts()[True]
        except: n_sideband_stream_stars = 0
        try: n_sig_bkg_stars = sr.stream.value_counts()[False]
        except: n_sig_bkg_stars = 0
        try: n_sideband_bkg_stars = sb.stream.value_counts()[False]
        except: n_sideband_bkg_stars = 0
          
        if verbose:
            print("Signal region has {:,} stream and {:,} bkg events ({:.2f}%).".format(n_sig_stream_stars, n_sig_bkg_stars,100*n_sig_stream_stars/n_sig_bkg_stars))
            print("Sideband region has {:,} stream and {:,} bkg events ({:.2f}%).".format(n_sideband_stream_stars, n_sideband_bkg_stars, 100*n_sideband_stream_stars/n_sideband_bkg_stars))
            print("f_sig = {:.1f}X f_sideband.".format(n_sig_stream_stars/n_sig_bkg_stars/(n_sideband_stream_stars/n_sideband_bkg_stars)))
    return df_slice

def plot_results(test, top_n = [50], save_folder=None, verbose=True):
    if save_folder is not None: 
        os.makedirs(save_folder, exist_ok=True)
    fig, axs = plt.subplots(nrows=1, ncols=2, dpi=150, figsize=(8,3), constrained_layout=True)
    bins=np.linspace(0,1,50)
    ax = axs[0]
    ax.tick_params(labelsize=12)
    ax.hist(test[test.label == 1].nn_score, bins=bins, histtype='step', linewidth=2, color="dodgerblue", label="Signal Region")
    ax.hist(test[test.label == 0].nn_score, bins=bins, histtype='step', linewidth=2, color="orange", label="Sideband Region")
    ax.legend(fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_title("Test Set")
    ax.set_xlabel("NN Score", size=12)
    ax.set_ylabel("Events", size=12)

    if "stream" in test.keys():
        ax = axs[1]
        ax.tick_params(labelsize=12)
        ax.hist(test[test.stream == False].nn_score, bins=bins, histtype='step', linewidth=2, color="grey", label="Not Stream")
        ax.hist(test[test.stream == True].nn_score, 
                bins=bins, histtype='step', linewidth=2, color="crimson", label="Stream")
        ax.legend(fontsize=12)
        ax.set_yscale("log")
        ax.set_xlim(0, 1)
        ax.set_title("Test Set")
        ax.set_xlabel("NN Score", size=12)
        ax.set_ylabel("Events", size=12);
    if save_folder is not None: 
        plt.savefig(os.path.join(save_folder,"nn_scores.png"))
    
    ### Plot purities
    if "stream" in test.keys():
        # Scan for optimal percentage
        cuts = np.linspace(0.01, 50, 100)
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
        if not np.isnan(purities).all():
#             if verbose: print("Maximum purity of {:.1f}% at {:.2f}%".format(np.nanmax(purities),cuts[np.nanargmax(purities)]))
            cut = cuts[np.nanargmax(purities)]
            plt.figure(dpi=150)
            plt.plot(cuts, purities, label="Signal Purity")
            plt.xlabel("Top \% Stars, ranked by NN score")
            plt.legend()    
            if save_folder is not None: 
                plt.savefig(os.path.join(save_folder,"purities.png"))

    ### Plot highest-ranked stars
    for x in top_n: # top N stars
        top_stars = test.sort_values('nn_score',ascending=False)[:x]
        if "stream" in test.keys():
            stream_stars_in_test_set = test[test.stream == True]
            if True in top_stars.stream.unique(): 
                n_perfect_matches = top_stars.stream.value_counts()[True] 
            else: 
                n_perfect_matches = 0 
        
            if verbose: print("Top {} stars: Purity = {:.1f}% ".format(x,n_perfect_matches/len(top_stars)*100))

        plt.figure(figsize=(5,3), dpi=150, tight_layout=True) 
        plt.title('Top {} Stars'.format(x))
        if "stream" in test.keys():
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
        else:
            plt.scatter(top_stars.α, top_stars.δ, marker='.', 
                    color = "crimson",
                    label="Top Stars") 
        plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
        plt.xlim(test.α.min(),test.α.max())
        plt.ylim(test.δ.min(),test.δ.max())
        plt.xlabel(r"$\alpha$ [\textdegree]")
        plt.ylabel(r"$\delta$ [\textdegree]")
        if save_folder is not None: 
            plt.savefig(os.path.join(save_folder,"top_{}_stars.png".format(x)))
            plt.savefig(os.path.join(save_folder,"top_{}_stars.pdf".format(x)))
            
#         plt.figure(figsize=(5,3), dpi=150, tight_layout=True) 
#         plt.title('Top {} Stars'.format(x))
#         plt.scatter(stream_stars_in_test_set.α, stream_stars_in_test_set.δ, marker='.', 
#                     color = "lightgray",
#                     label='Stream')
#         plt.scatter(top_stars.α, top_stars.δ, marker='.', 
#         c = top_stars.nn_score,
#         cmap = "Blues",
#         vmin=top_stars.nn_score.min(),
#         vmax=top_stars.nn_score.max(),
#         label="Top Stars") 
#         plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
#         plt.xlim(test.α.min(),test.α.max())
#         plt.ylim(test.δ.min(),test.δ.max())
#         plt.xlabel(r"$\alpha$ [\textdegree]")
#         plt.ylabel(r"$\delta$ [\textdegree]")
#         if save_folder is not None: 
#             plt.savefig(os.path.join(save_folder,"top_{}_stars_ranked.png".format(x)))
