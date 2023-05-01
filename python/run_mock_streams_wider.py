import sys
sys.path.append('../python')
from functions import *
from models import *
import tensorflow as tf
from livelossplot import PlotLossesKeras
os.environ["CUDA_VISIBLE_DEVICES"] = "2" # pick a number < 4 on ML4HEP; < 3 on Voltan 
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

files = glob("./trained_models/mocks/*/df_test.h5")
n_top_stars = 250
mock_purities = []

plt.figure(dpi=300, figsize=(5,5))

for file in tqdm(files):
    test = pd.read_hdf(file)
    top_stars = test.sort_values('nn_score',ascending=False)[:n_top_stars]
    try: 
        n_perfect_matches = top_stars.stream.value_counts()[True] 
    except:
        n_perfect_matches = 0
    purity = n_perfect_matches/len(top_stars)*100
    mock_purities.append(purity)

print("All mock purities:", mock_purities)
print("Mean = {:.2f}, Median = {:2f}".format(np.mean(mock_purities), np.median(mock_purities)))
      
mask = np.array(mock_purities) < 10
from itertools import compress
print("Low purities:", list(compress(mock_purities, mask)))
print("Low purity files:", list(compress(files, mask)))

mock_streams_low_purity = ["gaia_data/mock_streams/gaiamock_ra156.2_dec57.5_stream_feh-1.6_v3_"+file.split('/')[-2][5:]+".npy" for is_low_purity, file in zip(mask, files) if is_low_purity]

print(mock_streams_low_purity)

for file in tqdm(mock_streams_low_purity):
    save_folder = "trained_models/mocks/mock_"+file.split('_')[-1][:3]
    print("\n"+save_folder)
    
    df = pd.DataFrame(np.load(file), columns = ["μ_δ", "μ_α", "δ", "α", "b-r", "g", "ϕ", "λ", "μ_ϕcosλ", "μ_λ", 'stream'])
    df['α_wrapped'] = df['α'].apply(lambda x: x if x > 100 else x + 360)
    df['stream'] = df['stream']/100
    df['stream'] = df['stream'].astype(bool)

    make_plots(df, save_folder = save_folder)

    df_slice = signal_sideband(df, save_folder = save_folder, sr_factor=1, sb_factor=3)

    tf.keras.backend.clear_session()
    test = train(df_slice, verbose=False, save_folder = save_folder)
    
outputs = glob("trained_models/mocks/mock_*/after_fiducial_cuts/top_50_stars_purity*.pdf")
purities = [int(file.split('_')[-1][:-4]) for file in outputs]
print("Average purity among top 50 stars: {}".format(np.mean(purities)))