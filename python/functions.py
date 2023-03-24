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

### Plot setup
plt.rcParams.update({
    'figure.dpi': 200,
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.family": "serif",
    "font.size": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.max_open_warning": False,
})

def load_file(filename = "../gaia_data/gd1/gaiascan_l207.0_b50.2_ra148.6_dec24.2.npy"): # default patch from Via Machinae
    column_names = ["μ_δ", "μ_α", "δ", "α", "b-r", "g", "ϕ", "λ", "μ_ϕcosλ", "μ_λ"]
    gd1_stars = np.load('../gaia_data/gd1/gd1_stars.npy')
    df = pd.DataFrame(np.load(filename), columns = column_names)

    ### Label stream stars 
    is_stream, stream = FilterGD1(np.array(df), gd1_stars)
    df["stream"] = is_stream
    
    ### Wrap around alpha
    df['α_wrapped'] = df['α'].apply(lambda x: x if x > 100 else x + 360)
    return df

def angular_distance(angle1,angle2): # function from David's file via_machinae.py, needed for FilterGD1 function
    # inputs are np arrays of [ra,dec]
    deltara=np.minimum(np.minimum(np.abs(angle1[:,0]-angle2[:,0]+360),np.abs(angle1[:,0]-angle2[:,0])),\
                          np.abs(angle1[:,0]-angle2[:,0]-360))
    deltadec=np.abs(angle1[:,1]-angle2[:,1])
    return np.sqrt(deltara**2+deltadec**2)

def FilterGD1(stars, gd1_stars):
    gd1stars=np.zeros(len(stars))
    for x in tqdm(gd1_stars):
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

def fiducial_cuts(df):
#     center_ϕ = 0.5*(np.max(df['ϕ']) + np.min(df['ϕ']))
#     center_λ = 0.5*(np.max(df['λ']) + np.min(df['λ']))
#     df = df[np.sqrt((df['ϕ'] - center_ϕ)**2 + (df['λ'] - center_λ)**2) < 10] # avoid edge effects
    df = df[df.g < 20.2] # reduces streaking 
#     df = df[(np.abs(df['μ_λ']) > 2) | (np.abs(df['μ_ϕcosλ']) > 2)] # exclude stars near 0 proper motion
#     df = df[(df['μ_λ']**2 + df['μ_ϕcosλ']**2) > 4] # exclude stars near 0 proper motion
    df = df[(0.5 <= df['b-r']) & (df['b-r'] <= 1)] # cold stellar streams in particular
    return df

def make_plots(df, save_folder = "../plots"): 
#     fig = plt.figure(figsize=(12,4), dpi=200, tight_layout=True)
#     ax = fig.add_subplot(131)
#     ax.hexbin(df['ϕ'], df['λ'], bins=200, cmap="Greys")
#     ax.scatter(df[df.stream]['ϕ'], df[df.stream]['λ'], marker='.', s=5, color='crimson')
# #     circle = plt.Circle((0, 0), 10, color='k',lw=1,fill=False)
# #     ax.add_patch(circle)
#     ax.set_xlabel(r'$\phi~(^\circ)$',fontsize=20)
#     ax.set_ylabel(r'$\lambda~(^\circ)$',fontsize=20)
# #     ax.set_xlim(-11,11);
# #     ax.set_ylim(-11,11);

#     ax = fig.add_subplot(132)
#     ax.hexbin(df['μ_ϕcosλ'], df['μ_λ'], cmap='Greys', bins='log', gridsize=400, mincnt=1)
#     ax.scatter(df[df.stream]['μ_ϕcosλ'], df[df.stream]['μ_λ'], marker='.', s=5, color='crimson')
#     ax.set_xlim(-30,15)
#     ax.set_ylim(-30,15)
#     ax.set_xlabel(r'$\mu_\phi^*$ (mas/yr)',fontsize=20)
#     ax.set_ylabel(r'$\mu_\lambda$ (mas/yr)',fontsize=20)

#     ax = fig.add_subplot(133)
#     ax.hexbin(df['b-r'], df['g'], cmap='Greys', bins='log', gridsize=400, mincnt=1)
#     ax.scatter(df[df.stream]['b-r'], df[df.stream]['g'], marker='.', s=5, color='crimson')
#     ax.set_xlabel(r'$b-r$',fontsize=20)
#     ax.set_ylabel(r'$g$',fontsize=20)
#     ax.set_xlim(0,3)
#     ax.set_ylim(9,20.2)
#     ax.invert_yaxis()
#     plt.savefig(os.path.join(save_folder, "coords.pdf"))

    fig = plt.figure(figsize=(13,8), dpi=200, tight_layout=True)

    cmap = 'Greys'
    bins_0 = (np.linspace(-15,15,100), np.linspace(-15,15,100))
    bins_1 = (np.linspace(-20,10,100), np.linspace(-20,10,100))
    bins_2 = (np.linspace(0,3,100),np.linspace(9,20.2,100))

    ax = fig.add_subplot(231)
    h = ax.hist2d(df['ϕ'], df['λ'], cmap=cmap, cmin=1, vmax=250, bins=bins_0)
    # ax.scatter(df[df.stream]['ϕ'], df[df.stream]['λ'], marker='.', s=5, color='deeppink')
    ax.set_xlabel(r'$\phi~[^\circ]$',fontsize=20)
    ax.set_ylabel(r'$\lambda~[^\circ]$',fontsize=20)
    ax.set_xlim(-15,15);
    ax.set_ylim(-15,15);
    fig.colorbar(h[3], ax=ax)

    ax = fig.add_subplot(232)
    h = ax.hist2d(df['μ_ϕcosλ'], df['μ_λ'], cmap=cmap, cmin=1, bins=bins_1)
    ax.set_xlim(-20,10)
    ax.set_ylim(-20,10)
    ax.set_xlabel(r'$\mu_\phi^*$ [mas/yr]',fontsize=20)
    ax.set_ylabel(r'$\mu_\lambda$ [mas/yr]',fontsize=20)
    fig.colorbar(h[3], ax=ax)
    ax.set_title('Full Patch', fontsize=25, pad=15)

    ax = fig.add_subplot(233)
    h = ax.hist2d(df['b-r'], df['g'], cmap=cmap, cmin=1, bins=bins_2)
    ax.set_xlabel(r'$b-r$',fontsize=20)
    ax.set_ylabel(r'$g$',fontsize=20)
    ax.set_xlim(0,3)
    # ax.set_ylim(9,20.2)
    ax.invert_yaxis()
    fig.colorbar(h[3], ax=ax)

    ax = fig.add_subplot(234)
    h = ax.hist2d(df[df.stream]['ϕ'], df[df.stream]['λ'], cmap='Reds', bins=bins_0, cmin=1)
    # circle = plt.Circle((0, 0), 10, color='k',lw=1, linestyle='solid', fill=False)
    # ax.add_patch(circle)
    ax.set_xlabel(r'$\phi~[^\circ]$',fontsize=20)
    ax.set_ylabel(r'$\lambda~[^\circ]$',fontsize=20)
    ax.set_xlim(-15,15);
    ax.set_ylim(-15,15);
    fig.colorbar(h[3], ax=ax)

    ax = fig.add_subplot(235)
    h = ax.hist2d(df[df.stream]['μ_ϕcosλ'], df[df.stream]['μ_λ'], cmap='Reds',cmin=1, bins=bins_1)
    ax.set_xlim(-20,10)
    ax.set_ylim(-20,10)
    ax.set_xlabel(r'$\mu_\phi^*$ [mas/yr]',fontsize=20)
    ax.set_ylabel(r'$\mu_\lambda$ [mas/yr]',fontsize=20)
    fig.colorbar(h[3], ax=ax)
    ax.set_title('Labeled Stream Stars', fontsize=25, pad=15)

    ax = fig.add_subplot(236)
    h = ax.hist2d(df[df.stream]['b-r'], df[df.stream]['g'], cmap='Reds', cmin=1, bins=bins_2)#(np.linspace(0.5,1,10),np.linspace(17,20.2,14)))
    ax.set_xlabel(r'$b-r$',fontsize=20)
    ax.set_ylabel(r'$g$',fontsize=20)
    ax.set_xlim(0,3)
    ax.set_ylim(9,20.2)
    ax.invert_yaxis()
    fig.colorbar(h[3], ax=ax);

    plt.savefig(os.path.join(save_folder, "coords.pdf"))

# def get_random_file(glob_path):
#     file = random.choice(glob(glob_path))
#     return(file)

# def load_file(file = None, stream = None, folder = "../gaia_data/", percent_bkg = 100):
#     ### Stream options: ["gd1", "gd1_tail", "mock", "jhelum"]
#     if stream == "mock": 
#         if file is not None:
#             file = file
#         else:
#             file = os.path.join(folder,"mock_streams/gaiamock_ra156.2_dec57.5_stream_feh-1.6_v3_863.npy")
# #             file = get_random_file(os.path.join(folder,"mock_streams/*.npy"))
#         print(file)
#         df = pd.DataFrame(np.load(file), columns=['μ_δ','μ_α','δ','α','mag','color','a','b','c','d','stream'])
#         df['stream'] = df['stream']/100
#         df['stream'] = df['stream'].astype(bool)

#     elif stream == "gd1": 
#         file = os.path.join(folder,"gd1/GD1-circle-140-30-15.pkl")
#         df = np.load(file, allow_pickle = True)

#         ### Select columns
#         columns = ['pmdec','pmra','dec','ra','phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag','streammask']
#         df = df[columns]

#         ### Create b-r & g columns; rename others
#         df["b-r"] = df.phot_bp_mean_mag - df.phot_rp_mean_mag
#         df.drop(columns = ['phot_bp_mean_mag','phot_rp_mean_mag'], inplace=True)
#         df.rename(columns={'phot_g_mean_mag': 'g', 
#                            'ra': 'α',
#                            'dec': 'δ',
#                            'pmra': 'μ_α',
#                            'pmdec': 'μ_δ',
#                            'streammask': 'stream'}, inplace=True)
#         df["mag"] = df.g
#         df["color"] = df["b-r"]
#         df["weight"] = 1

#     elif stream == "gd1_tail":
# #         file = os.path.join(folder,"gd1_tail/gd1_tail.h5")
#         file = os.path.join(folder,"gd1_tail/gd1_tail_optimized_patch.h5")
#         df = pd.read_hdf(file)
#         df = df.drop_duplicates(subset=['α','δ','μ_α','μ_δ','color','mag'])
#         weight = 1 
#         df["weight"] = np.where(df['stream']==True, weight, 1)

#     elif stream == "gaia3":
#         ### Note that we don't have stream labels here
#         file = get_random_file(os.path.join(folder,"gaia3/*.npy"))
#         print(file)
#         df = pd.DataFrame(np.load(file)[:,[9,8,6,7,4,5]], columns=['μ_δ','μ_α','δ','α','mag','color'])        
#     elif stream == "jhelum":
#         ### Note that we don't have stream labels here
#         file = get_random_file(os.path.join(folder,"jhelum/*.npy"))
# #         file = "../GaiaCWoLa/gaia_data/jhelum/gaiascan_l303.8_b58.4_ra193.3_dec-4.5.npy"
#         print(file)
#         df = pd.DataFrame(np.load(file)[:,[9,8,6,7,4,5]], columns=['μ_δ','μ_α','δ','α','mag','color'])
#     else:
#         print("Stream not recognized.")
        
        
#     ### Drop duplicate stars 
#     if 'color' in df.keys(): 
#         variables = ['μ_α','μ_δ','δ','α','color','mag']
#     elif 'b-r' in df.keys():
#         variables = ['μ_α','μ_δ','δ','α','g','b-r']
#     df = df.drop_duplicates(subset=variables)
    
#     ### Drop any rows containing a NaN value
#     df.dropna(inplace = True)

#     ### Restrict data to a radius of 15
#     center_α = 0.5*(df.α.min()+df.α.max())
#     center_δ = 0.5*(df.δ.min()+df.δ.max())
#     df = df[np.sqrt((df.δ-center_δ)**2+(df.α-center_α)**2) < 15]
    
#     if percent_bkg != 100 and "stream" in df.keys():
#         ### Optional: reduce background
#         n_sig = len(df[df.stream == True])
#         n_bkg = len(df[df.stream == False])
#         print("Before reduction, stream stars make up {:.3f}% of the dataset.".format(100*n_sig/len(df)))

#         events_to_drop = np.random.choice(df[df.stream == False].index, int((1-percent_bkg/100)*n_bkg), replace=False)
#         df = df.drop(events_to_drop)
#         print("After reduction, stream stars make up {:.3f}% of the dataset.".format(100*n_sig/len(df)))
    
#     df.reset_index(inplace=True)
#     return df, file

def plot_coords(df, save_folder=None):
    fig, axs = plt.subplots(nrows=2, ncols=3, dpi=200, figsize=(10,7), tight_layout=True)
    
    print(df.keys())
    bins_ϕ=np.linspace(df['ϕ'].min(),df['ϕ'].max(),100)
    bins_λ=np.linspace(df.λ.min(),df.λ.max(),100)
    
    cmap="binary"
    labelsize=14
    
    ax = axs[0,0]
    ax.hist2d(df.lon,df.lat, bins=[bins_ϕ,bins_λ], cmap=cmap)
    ax.set_xlabel(r"$\phi$ [\textdegree]", fontsize=labelsize)
    ax.set_ylabel(r"$\lambda$ [\textdegree]", fontsize=labelsize);
    ax.set_title("Full Dataset", fontsize=16)
    
    ax = axs[1,0]
    ax.hist2d(df[df.stream == True].ϕ,df[df.stream == True].λ, bins=[bins_ϕ,bins_λ], cmap=cmap)
    ax.set_xlabel(r"$\phi$ [\textdegree]", fontsize=labelsize)
    ax.set_ylabel(r"$\lambda$ [\textdegree]", fontsize=labelsize);
    ax.set_title("Stream Only", fontsize=16);
    
    bins = np.linspace(-25,10,100)
    ax = axs[0,1]
    ax.hist2d(df.μ_α*np.cos(df.λ),df.μ_λ, bins=[bins,bins], cmap=cmap)
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
#         plt.savefig(os.path.join(save_folder,"input_variables.png"))
        plt.savefig(os.path.join(save_folder,"input_variables.pdf"))
    
def signal_sideband(df, sr_factor = 1, sb_factor = 3, save_folder=None, sb_min=None, sb_max=None, sr_min=None, sr_max=None, verbose=True, scan_over_mu_phi=False):
    
    print("SR factor:", sr_factor)
    print("SB factor:", sb_factor)
    
    if scan_over_mu_phi:
        var = "μ_ϕcosλ"
    else:
        var = "μ_λ"
        
    print("Scanning over "+str(var))
    
    if sb_min is not None:
        sb_min = sb_min
        sb_max = sb_max
        sr_min = sr_min
        sr_max = sr_max    
    else: 
        sb_min = df[df.stream][var].median()-sb_factor*df[df.stream][var].std()
        sb_max = df[df.stream][var].median()+sb_factor*df[df.stream][var].std()
        sr_min = df[df.stream][var].median()-sr_factor*df[df.stream][var].std()
        sr_max = df[df.stream][var].median()+sr_factor*df[df.stream][var].std()
        
    if verbose:
        print("Sideband region: [{:.1f},{:.1f}) & ({:.1f},{:.1f}]".format(sb_min, sr_min, sr_max, sb_max))
        print("Signal region: [{:.1f},{:.1f}]".format(sr_min,sr_max))
    
    df_slice = df[(df[var] >= sb_min) & (df[var] <= sb_max)]
    df_slice['label'] = np.where(((df_slice[var] >= sr_min) & (df_slice[var] <= sr_max)), 1, 0)
    
    sr = df_slice[df_slice.label == 1]
    sb = df_slice[df_slice.label == 0]
    if verbose: print("Total counts: SR = {:,}, SB = {:,}".format(len(sr), len(sb)))

    outer_region = df[(df[var] < sb_min) | (df[var] > sb_max)]
    sb = df[(df[var] >= sb_min) & (df[var] <= sb_max)]
    sr = df[(df[var] >= sr_min) & (df[var] <= sr_max)]    
        
    bins = np.linspace(sb_min - (sr_min - sb_min), sb_max + (sb_max - sr_max), 40)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,5), dpi=300, tight_layout=True) 
    ax = axs[0]
    ax.hist(outer_region[outer_region.stream == False][var], density=False, color="lightgray", alpha=0.3, histtype="stepfilled", linewidth=2, bins=bins, label="Outer Region");
    ax.hist(sb[sb.stream == False][var], density=False, color="lightgray", alpha=1, histtype="stepfilled", linewidth=2, bins=bins, label="Sideband Region");
    ax.hist(sr[sr.stream == False][var], density=False, color="gray", histtype="stepfilled", linewidth=2, bins=bins, label="Signal Region");
    ax.set_title('Background Stars', fontsize=23)
    if var == "μ_ϕcosλ": 
        ax.set_xlabel(r'$\mu_\phi^*$ [mas/year]', fontsize=20)
    else:
        ax.set_xlabel(r'$\mu_\lambda$ [mas/year]', fontsize=20)
    ax.set_ylabel('Number of Stars', fontsize=20)
#     ax.set_yscale('log')
    ax.legend(loc="upper left", frameon=False);
    
    ax = axs[1]
    ax.hist(outer_region[outer_region.stream][var], density=False, color="crimson", histtype="stepfilled", alpha=0.25, linewidth=2, bins=bins, label="Outer Region")
    ax.hist(sb[sb.stream][var], color="crimson", density=False, histtype="stepfilled", alpha=0.4, linewidth=2, bins=bins, label="Sideband Region")
    ax.hist(sr[sr.stream][var], color="crimson", density=False, histtype="stepfilled", linewidth=2, bins=bins, label="Signal Region")
    
    ax.set_title('Stream Stars', fontsize=23)
    if var == "μ_ϕcosλ": 
        ax.set_xlabel(r'$\mu_\phi^*$ [mas/year]', fontsize=20)
    else:
        ax.set_xlabel(r'$\mu_\lambda$ [mas/year]', fontsize=20)
    ax.set_ylabel('Number of Stars', fontsize=20)
#     ax.set_yscale('log')
    ax.legend(frameon=False);
    if save_folder is not None:
        plt.savefig(os.path.join(save_folder,"mu_lambda.pdf"))    
    
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

def plot_results(test, top_n = [50, 100], save_folder=None, verbose=True, show=True):
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
    if show: plt.show()
    plt.close()
    
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
            if show: plt.show()
            plt.close()

    ### Plot highest-ranked stars
    for x in top_n: # top N stars
        top_stars = test.sort_values('nn_score',ascending=False)[:x]
        if "stream" in test.keys():
            stream_stars_in_test_set = test[test.stream == True]
            if True in top_stars.stream.unique(): 
                n_perfect_matches = top_stars.stream.value_counts()[True] 
            else: 
                n_perfect_matches = 0 
        
            if verbose and show: print("Top {} stars: Purity = {:.1f}% ".format(x,n_perfect_matches/len(top_stars)*100))

        plt.figure(figsize=(5,3), dpi=150, tight_layout=True) 
        plt.title('Top {} Stars'.format(x))
        if "stream" in test.keys():
            plt.scatter(stream_stars_in_test_set.α_wrapped - 360, stream_stars_in_test_set.δ, marker='.', 
                    color = "lightgray",
                    label='Stream')
            plt.scatter(top_stars.α_wrapped - 360, top_stars.δ, marker='.', 
                    color = "lightpink",
                    label="Top Stars\n(Purity = {:.0f}\%)".format(n_perfect_matches/len(top_stars)*100))
            if True in top_stars.stream.unique(): 
                plt.scatter(top_stars[top_stars.stream].α_wrapped - 360, top_stars[top_stars.stream].δ, marker='.', 
                        color = "crimson",
                        label='Matches')
        else:
            plt.scatter(top_stars.α_wrapped - 360, top_stars.δ, marker='.', 
                    color = "crimson",
                    label="Top Stars") 
        plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
        plt.xlim(test.α_wrapped.min() - 360,test.α_wrapped.max()- 360)
        plt.ylim(test.δ.min(),test.δ.max())
        plt.xlabel(r"$\alpha$ [\textdegree]")
        plt.ylabel(r"$\delta$ [\textdegree]")
        if save_folder is not None: 
            plt.savefig(os.path.join(save_folder,"top_{}_stars_purity_{}.pdf".format(x, int(n_perfect_matches/len(top_stars)*100))))
        if show: plt.show()
        plt.close()
    