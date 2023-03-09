import sys
sys.path.append('../python')
from functions import *
from models import *
import tensorflow as tf
from livelossplot import PlotLossesKeras
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # pick a number < 4 on ML4HEP; < 3 on Voltan 
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

n_streams = 69

import random
mock_streams = glob("gaia_data/mock_streams/gaiamock_*.npy")
selected_streams = random.sample(mock_streams, n_streams)

for file in tqdm(selected_streams):
    save_folder = "trained_models/mocks/mock_"+file.split('_')[-1][:3]
    
    if save_folder is not None: 
        os.makedirs(save_folder, exist_ok=True)

    df = pd.DataFrame(np.load(file), columns = ["μ_δ", "μ_α", "δ", "α", "b-r", "g", "ϕ", "λ", "μ_ϕcosλ", "μ_λ", 'stream'])
    df['α_wrapped'] = df['α'].apply(lambda x: x if x > 100 else x + 360)
    df['stream'] = df['stream']/100
    df['stream'] = df['stream'].astype(bool)

    make_plots(df, save_folder = save_folder)

    df_slice = signal_sideband(df, save_folder = save_folder, sr_factor=0.25, sb_factor=0.5)

    tf.keras.backend.clear_session()
    test = train(df_slice, verbose=False, save_folder = save_folder)
    
outputs = glob("trained_models/mocks/mock_*/after_fiducial_cuts/top_50_stars_purity*.pdf")
purities = [int(file.split('_')[-1][:-4]) for file in outputs]
print("Average purity among top 50 stars: {}".format(np.mean(purities)))