import random 
from glob import glob
import pandas as pd 
import numpy as np

def get_random_file():
    file = random.choice(glob("/data0/mpettee/gaia_data/mock_streams/*.npy"))
    return(file)

def load_file(file, stream = None, percent_bkg = 100):
    ### Stream options: ["mock", "gd1", "fjorm"]
    if stream == "mock": 
        df = pd.DataFrame(np.load(file), columns=['μ_δ','μ_α','δ','α','mag','color','a','b','c','d','stream'])
        df['stream'] = df['stream']/100
        df['stream'] = df['stream'].astype(bool)

        ### Drop any rows containing a NaN value
        df.dropna(inplace = True)

        ### Restrict data to a radius of 15
        center_α = 0.5*(df.α.min()+df.α.max())
        center_δ = 0.5*(df.δ.min()+df.δ.max())
        df = df[np.sqrt((df.δ-center_δ)**2+(df.α-center_α)**2) < 15]

        ### Construct stream DataFrame
        stream = df[df.stream == True]
        
    elif stream == "gd1": 
        print("Load GD1")
        df = pd.DataFrame()
        
    if percent_bkg != 100:
        ### Optional: reduce background
        n_sig = len(df[df.stream == True])
        n_bkg = len(df[df.stream == False])
        print("Before reduction, stream stars make up {:.3f}% of the dataset.".format(100*n_sig/len(df)))

        events_to_drop = np.random.choice(df[df.stream == False].index, int((1-percent_bkg/100)*n_bkg), replace=False)
        df = df.drop(events_to_drop)
        print("After reduction, stream stars make up {:.3f}% of the dataset.".format(100*n_sig/len(df)))
        
    return df