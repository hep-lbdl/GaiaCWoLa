### Generic imports 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import stats
from glob import glob
import argparse
import json
import time 

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
os.environ["CUDA_VISIBLE_DEVICES"] = "2" # pick a number < 4 on ML4HEP; < 3 on Voltan 
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
    parser.add_argument("--save_label", default=None, type=str, help="Folder name for saving training outputs & plots. If not specified, plots will not be saved.")
    parser.add_argument("--percent_bkg", default=100, type=int, help="Percent of background to train on.")
    parser.add_argument("--layer_size", default=128, type=int, help="Number of nodes per layer.")
    parser.add_argument("--patience", default=30, type=int, help="How many epochs of no val_loss improvement before the training is stopped.")
    parser.add_argument("--epochs", default=200, type=int, help="Number of training epochs.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size during training.")
    parser.add_argument("--dropout", default=0, type=float, help="Dropout probability.")
    parser.add_argument("--l2_reg", default=0, type=float, help="L2 regularization.")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    t0 = time.time()
    
    save_label = args.save_label
    save_folder = os.path.join("./trained_models",save_label)
    os.makedirs(save_folder, exist_ok=True)
    
    ### Save arguments
    with open(os.path.join(save_folder,'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    ### Load file & preprocess
    df = load_file(stream = args.stream, percent_bkg = args.percent_bkg)
    visualize_stream(df, save_folder = save_folder)
    
    ### Define signal & sideband regions 
    df_slice = signal_sideband(df, stream = args.stream, save_folder = save_folder)
    
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
    model = Sequential()
    
    if args.l2_reg == 0: 
        reg = None
    else:
        reg = regularizers.l2(args.l2_reg)
    
    model.add(Dense(args.layer_size, input_dim=len(training_vars), activation='relu', activity_regularizer=reg)) 
    if args.dropout != 0: 
        model.add(Dropout(args.dropout))
    model.add(Dense(args.layer_size, activation='relu', activity_regularizer=reg))
    if args.dropout != 0: 
        model.add(Dropout(args.dropout))
    model.add(Dense(args.layer_size, activation='relu', activity_regularizer=reg))
    if args.dropout != 0: 
        model.add(Dropout(args.dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    model.summary()
    
    # stops if val_loss doesn't improve for [patience] straight epochs
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', 
                                             patience=args.patience, 
                                             verbose=1) 

    # saves weights from the epoch with lowest val_loss 
    checkpoint = callbacks.ModelCheckpoint(os.path.join(save_folder,"weights.h5"), 
                                           monitor='val_loss', 
                                           mode='auto', 
                                           verbose=1, 
                                           save_best_only=True, 
                                           save_weights_only=True)

    ### Save model architecture 
    with open(os.path.join(save_folder,"model.json"), "w") as json_file:
        json_file.write(model.to_json())
    
    ### Train!
    history = model.fit(x_train, y_train, 
                        epochs=args.epochs, 
                        batch_size=args.batch_size,
                        validation_data=(x_val,y_val),
                        callbacks = [checkpoint,early_stopping],
                        verbose = 2,
                       )
    
    ### Save training losses & accuracies
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize = (12,6))
    ax = axs[0]
    ax.plot(history.history["accuracy"], label="Training Accuracy")
    ax.plot(history.history["val_accuracy"], label="Validation Accuracy")
    ax.set_title("Accuracy")
    ax.set_xlabel("Epochs")
    ax.legend()

    ax = axs[1]
    ax.plot(history.history["loss"], label="Training Loss")
    ax.plot(history.history["val_loss"], label="Validation Loss")
    ax.set_title("Loss")
    ax.set_xlabel("Epochs")
    ax.legend()
    plt.savefig(os.path.join(save_folder,"loss_curves.png"))
    
    ### Load best weights
    model.load_weights(os.path.join(save_folder,"weights.h5"))

    ### Add the NN prediction score to the test set: 
    test["nn_score"] = model.predict(x_test)
    fake_eff_baseline, real_eff_baseline, thresholds = roc_curve(np.asarray(y_test), test.nn_score)
    auc_baseline = auc(fake_eff_baseline, real_eff_baseline)
    print("AUC: {:.3f}".format(auc_baseline))

    ### Plot scores:
    plot_results(test, save_folder)
    
    ### Save test DataFrame for future plotting
    test.to_hdf(os.path.join(save_folder,"df_test.h5"), "df")
    
    print("Finished in {:,} seconds.".format(time.time() - t0))
          