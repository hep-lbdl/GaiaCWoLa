import sys
sys.path.append('../python')
from functions import *
from models import *
import tensorflow as tf
from livelossplot import PlotLossesKeras
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # pick a number < 4 on ML4HEP; < 3 on Voltan 
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

indices = [625, 825, 260, 636, 791, 995, 76]

for i in indices:
    print("Running mock stream {}".format(i))
    save_folder = "./trained_models/mocks/mock_{}".format(i)
    file = "./gaia_data/mock_streams/gaiamock_ra156.2_dec57.5_stream_feh-1.6_v3_{}.npy".format(i)
    df = pd.DataFrame(np.load(file), columns = ["μ_δ", "μ_α", "δ", "α", "b-r", "g", "ϕ", "λ", "μ_ϕcosλ", "μ_λ", 'stream'])
    df['α_wrapped'] = df['α'].apply(lambda x: x if x > 100 else x + 360)
    df['stream'] = df['stream']/100
    df['stream'] = df['stream'].astype(bool)

    make_plots(df, save_folder = save_folder)
    df_slice = signal_sideband(df, save_folder = save_folder, sr_factor=1, sb_factor=3)
    test = train(df_slice, verbose=True, save_folder = save_folder)
