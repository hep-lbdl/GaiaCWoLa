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
import shutil

### ML imports
from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential
from keras import callbacks, regularizers
from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

### Custom imports
from functions import *

def train(df_slice, layer_size, batch_size, dropout, l2_reg, epochs, patience, n_folds, best_of_n_loops, save_folder, other_callbacks=None, verbose=True):
    os.makedirs(save_folder, exist_ok=True)
    if 'color' in df_slice.keys(): 
        training_vars = ['μ_α','δ','α','color','mag']
    elif 'b-r' in df_slice.keys():
        training_vars = ['μ_α','δ','α','g','b-r']
        
    loop_purities = []
    for loop in tqdm(np.arange(best_of_n_loops), desc="Loop"):
        train, validate, test = np.split(df_slice.sample(frac=1), [int(.7*len(df_slice)), int(.85*len(df_slice))]) # 70/15/15 train/validate/test split

        x_train, x_val, x_test = [train[training_vars], validate[training_vars], test[training_vars]]
        y_train, y_val, y_test = [train.label, validate.label, test.label]

        sample_weight = None

        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)
        x_val = sc.transform(x_val)

        ### Define model architecture 
        reg = regularizers.l2(l2_reg)
        model = Sequential()
        model.add(Dense(layer_size, input_dim=len(training_vars), 
                        activation='relu', activity_regularizer=reg)) 
        if dropout != 0: model.add(Dropout(dropout))
        model.add(Dense(layer_size, activation='relu', activity_regularizer=reg))
        if dropout != 0: model.add(Dropout(dropout))
        model.add(Dense(layer_size, activation='relu', activity_regularizer=reg))
        if dropout != 0: model.add(Dropout(dropout))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#             model.summary()

        # stops if val_loss doesn't improve for [patience] straight epochs
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', 
                                                 patience=patience, 
                                                 verbose=0) 

        # saves weights from the epoch with lowest val_loss 
        weights_path = os.path.join(save_folder,"weights_loop{}.h5".format(loop))
        checkpoint = callbacks.ModelCheckpoint(weights_path, 
                                               monitor='val_loss', 
                                               mode='auto', 
                                               verbose=0, 
                                               save_best_only=True, 
                                               save_weights_only=True)

        callbacks_list = [checkpoint,early_stopping]
        if other_callbacks is not None:
            callbacks_list = callbacks_list + [other_callbacks]

        ### Train!
        history = model.fit(x_train, y_train, 
                    epochs=epochs, 
                    sample_weight=sample_weight,
                    batch_size=batch_size,
                    validation_data=(x_val,y_val),
                    callbacks = callbacks_list,
                    verbose = 0,#int(verbose),
                   )

#         ### Save training losses & accuracies
#         fig, axs = plt.subplots(nrows=1, ncols=2, figsize = (12,6))
#         ax = axs[0]
#         ax.plot(history.history["accuracy"], label="Training Accuracy")
#         ax.plot(history.history["val_accuracy"], label="Validation Accuracy")
#         ax.set_title("Accuracy")
#         ax.set_xlabel("Epochs")
#         ax.legend()

#         ax = axs[1]
#         ax.plot(history.history["loss"], label="Training Loss")
#         ax.plot(history.history["val_loss"], label="Validation Loss")
#         ax.set_title("Loss")
#         ax.set_xlabel("Epochs")
#         ax.legend()
#         plt.savefig(os.path.join(save_folder,"loss_curve_loop{}.png".format(loop)))

        ### Add the NN prediction score to the test set: 
        test["nn_score"] = model.predict(x_test)
        fake_eff_baseline, real_eff_baseline, thresholds = roc_curve(np.asarray(y_test), test.nn_score)
        auc_baseline = auc(fake_eff_baseline, real_eff_baseline)

        if "stream" in test.keys():
            # Scan for optimal percentage
            efficiencies = []
            purities = []
            top_stars = test.sort_values('nn_score',ascending=False)[:50] # top 50 stars
            stream_stars_in_test_set = test[test.stream == True]
            if True in top_stars.stream.unique(): 
                n_perfect_matches = top_stars.stream.value_counts()[True] 
                stream_stars_in_test_set = test[test.stream == True]
                efficiencies.append(100*n_perfect_matches/len(stream_stars_in_test_set))
                purities.append(n_perfect_matches/len(top_stars)*100)
            else: 
                n_perfect_matches = 0 
                efficiencies.append(np.nan)
                purities.append(np.nan)
        
        loop_purities.append(purities)

    max_loop_purities = [np.nanmax(loop_purity) for loop_purity in loop_purities]
    print("Max loop purities:", max_loop_purities)
    print("Best loop = {}, with a purity of {:.2f}%.".format(np.argmax(max_loop_purities), np.nanmax(max_loop_purities)))
    print("Loading weights from loop {}...".format(np.argmax(max_loop_purities)))
    best_weights_path = os.path.join(save_folder,"weights_loop{}.h5".format(np.argmax(max_loop_purities)))
    model.load_weights(best_weights_path)
    
    ### Plot scores:
    plot_results(test, save_folder=save_folder, verbose=verbose)

    ### Save test DataFrame for future plotting
    test.to_hdf(os.path.join(save_folder,"df_test.h5"), "df")

    return(test)

    

