### Generic imports 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import stats
from glob import glob
import argparse

### ML imports
from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential
from keras import callbacks, regularizers
from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf

### Custom imports
from functions import *

### GPU Setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # pick a number < 4 on ML4HEP; < 3 on Voltan 
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("stream", default=None, choices = ["gd1", "gd1_tail", "mock"], help="Choose which stream to analyze.")
    parser.add_argument("--percent_bkg", default=100, type=int, help="Percent of background to train on.")
    parser.add_argument("--layer_size", default=128, type=int, help="Number of nodes per layer.")
    parser.add_argument("--epochs", default=200, type=int, help="Number of training epochs.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size during training.")
    parser.add_argument("--dropout", default=0, type=float, help="Dropout probability.")
    parser.add_argument("--l2_reg", default=0, type=float, help="L2 regularization.")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    
    save_label = args.stream
    
    ### Load file & preprocess
    df = load_file(stream = args.stream, percent_bkg = args.percent_bkg)
    visualize_stream(df, save_label = save_label)
    
    ### Define signal & sideband regions 
    if args.stream == "gd1_tail":
        sb_min = -6
        sb_max = 0
        sr_min = -4.5
        sr_max = -1.7
        
    elif args.stream == "mock":
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
    plt.savefig(os.path.join("./plots",save_label,"signal_sideband.png"))

    sr = df_slice[df_slice.label == 1]
    sb = df_slice[df_slice.label == 0]

    print("Signal region has {:,} stream and {:,} bkg events.".format(sr.stream.value_counts()[True], sr.stream.value_counts()[False]))
    print("Sideband region has {:,} stream and {:,} bkg events.".format(sb.stream.value_counts()[True], sb.stream.value_counts()[False]))
    print("Total counts: SR = {:,}, SB = {:,}".format(len(sr), len(sb)))

    ### Prepare datasets for training
    training_vars = ['μ_α','δ','α','color','mag']
    train, validate, test = np.split(df_slice.sample(frac=1), [int(.7*len(df_slice)), int(.85*len(df_slice))]) # 70/15/15 train/validate/test split

    x_train, x_val, x_test = [train[training_vars], validate[training_vars], test[training_vars]]
    y_train, y_val, y_test = [train.label, validate.label, test.label]

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    x_val = sc.transform(x_val)
    
    ### Define model architecture 
    save_name = "best_weights"+args.stream

    model = Sequential()
    
    if args.l2_reg == 0: 
        reg = None
    else:
        reg = regularizers.l2(args.l2_reg)
    
    model.add(Dense(args.layer_size, input_dim=len(training_vars), activation='relu',
                   activity_regularizer=reg
                   )) 
    if args.dropout != 0: 
        model.add(Dropout(args.dropout))
    model.add(Dense(args.layer_size, activation='relu',
                   activity_regularizer=reg
                   ))
    if args.dropout != 0: 
        model.add(Dropout(args.dropout))
    model.add(Dense(args.layer_size, activation='relu',
                   activity_regularizer=reg
                   ))
    if args.dropout != 0: 
        model.add(Dropout(args.dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    model.summary()
    
    # stops if val_loss doesn't improve for [patience] straight epochs
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', 
                                             patience=20, 
                                             verbose=1) 

    # saves weights from the epoch with lowest val_loss 
    os.makedirs("./weights", exist_ok=True)
    checkpoint = callbacks.ModelCheckpoint("./weights/"+save_name+".h5", 
                                           monitor='val_loss', 
                                           mode='auto', 
                                           verbose=1, 
                                           save_best_only=True, 
                                           save_weights_only=True)

    ### Train!
    history = model.fit(x_train, y_train, 
                        epochs=args.epochs, 
                        batch_size=args.batch_size,
                        validation_data=(x_val,y_val),
                        callbacks = [
                                    checkpoint, 
                                    early_stopping],
                        verbose = 2,
                       )
    
    ### Evaluate training
    model.load_weights("./weights/"+save_name+".h5")

    ### Add the NN prediction score to the test set: 
    test["nn_score"] = model.predict(x_test)
    fake_eff_baseline, real_eff_baseline, thresholds = roc_curve(np.asarray(y_test), test.nn_score)
    auc_baseline = auc(fake_eff_baseline, real_eff_baseline)
    print("AUC: {}".format(auc_baseline))

    ### Plot scores:
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
    plt.savefig(os.path.join("./plots",save_label,"nn_scores.png"))
    
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
    plt.plot(cuts, purities, label="Signal Purity")
    plt.xlabel("Top \% Stars, ranked by NN score")
    plt.legend()    
    
    ### Plot highest-ranked stars
    x = 1 # desired percentage
    top_stars = test[(test['nn_score'] >= test['nn_score'].quantile((100-x)/100))]
    
#     N = 20
#     top_stars = test.sort_values(by=["nn_score"],ascending=False)[:N]
    
    n_perfect_matches = top_stars.stream.value_counts()[True]
    stream_stars_in_test_set = test[test.stream == True]

    print("Efficiency: {:.1f}%".format(100*n_perfect_matches/len(stream_stars_in_test_set)))
    print("Purity: {:.1f}%".format(n_perfect_matches/len(top_stars)*100))
    
    plt.figure(figsize=(3,3), tight_layout=True) 
    plt.scatter(stream_stars_in_test_set.α, stream_stars_in_test_set.δ, marker='.', 
                color = "lightgray",
                label='GD1 Tail')
    plt.scatter(top_stars.α, top_stars.δ, marker='.', 
                color = "lightpink",
#                 label = 'Top {} stars'.format(N))
                label='Top {:.0f}\% NN Scores'.format(x))
    plt.scatter(top_stars[top_stars.stream].α, top_stars[top_stars.stream].δ, marker='.', 
                color = "crimson",
                label='Matches')
    plt.legend()
    plt.xlim(-15,15)
    plt.ylim(-15,15)
    plt.xlabel(r"$\alpha$ [\textdegree]")
    plt.ylabel(r"$\delta$ [\textdegree]")
    plt.savefig(os.path.join("./plots",save_label,"top_{}\%_stars.png".format(x)))

    
