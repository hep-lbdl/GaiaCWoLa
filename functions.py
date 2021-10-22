import os
import random 
from glob import glob
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 

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

def angular_distance(angle1,angle2):
    # inputs are np arrays of [ra,dec]
    deltara=np.minimum(np.minimum(np.abs(angle1[:,0]-angle2[:,0]+360),np.abs(angle1[:,0]-angle2[:,0])),np.abs(angle1[:,0]-angle2[:,0]-360))
    deltadec=np.abs(angle1[:,1]-angle2[:,1])
    return np.sqrt(deltara**2+deltadec**2)

#function from David's file via_machinae.py
def FilterGD1(stars, gd1):
    gd1stars=np.zeros(len(stars))
    for x in tqdm(gd1):
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
    return gd1stars

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
        file = "/data0/mpettee/gaia_data/gd1_tail/gaiascan_l101.2_b58.4_ra212.7_dec55.2.npy"
        gd1 = np.load('/data0/mpettee/gaia_data/gd1/gd1_stars.npy')
        df = pd.DataFrame(np.load(file)[:,[9,8,6,7,4,5]], columns=['μ_δ','μ_α','δ','α','color','mag'])
        df['stream'] = FilterGD1(np.load(file), gd1)
        
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

def visualize_stream(df, save_label=None):
    plt.figure(figsize=(3,3), tight_layout=True) 
    plt.hist2d(df.α,df.δ,bins=100)
    if "stream" in df.keys():
        plt.scatter(df[df.stream == True].α,df[df.stream == True].δ,color='white',s=0.2, label="Stream")
    plt.xlabel(r"$\alpha$ [\textdegree]")
    plt.ylabel(r"$\delta$ [\textdegree]");
#     plt.legend();
    if save_label is not None:
        os.makedirs(os.path.join("./plots",save_label), exist_ok=True)
        plt.savefig(os.path.join("./plots",save_label,"stream_position.png"))
    
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
        
    if save_label is not None:
        plt.savefig(os.path.join("./plots",save_label,"stream_velocities.png"))
        
    