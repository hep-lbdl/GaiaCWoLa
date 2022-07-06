### Generic imports 
import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
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
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn import preprocessing

### Custom imports
from functions import *

def train(df, layer_size, batch_size, dropout, l2_reg, epochs, patience, n_folds, best_of_n_loops, save_folder, other_callbacks=None, verbose=True):
    os.makedirs(save_folder, exist_ok=True)
    if 'color' in df.keys(): 
        training_vars = ['μ_α','δ','α','color','mag']
    elif 'b-r' in df.keys():
        training_vars = ['μ_α','δ','α','g','b-r']
   
    ### Explicitly get indices of stars for each k-fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=15)
    fold_stars = []
    for fold, (train_index, test_index) in enumerate(skf.split(
        df[training_vars], df.label)):
        fold_stars.append(test_index)
    
    ### Loop through the k-folds
    fold_labels = np.arange(len(fold_stars))
    test_dataframes = []
    for fold in tqdm(fold_labels, desc="Folds"):
        save_folder_fold = os.path.join(save_folder,"kfold_{}".format(fold))
        
        ### Define test set
        test_stars = fold_stars[fold]
        
        ### Loop through all remaining val sets
        val_losses = []
        test_scores = []
        for val_set in tqdm(np.delete(fold_labels, fold), desc="Validation sets"):
            ### Make save folder
            save_folder_val = os.path.join(save_folder,"kfold_{}".format(fold),"val_set_{}".format(val_set))
            os.makedirs(save_folder_val, exist_ok=True)
        
            ### Define val set
            val_stars = fold_stars[val_set]
            train_stars = np.concatenate([fold_stars[i] for i in np.delete(fold_labels, [fold, val_set])])
            
            ### Define datasets
            train = df.iloc[train_stars]
            val = df.iloc[val_stars]
            test = df.iloc[test_stars]
        
            ### Standardize the inputs (x) and create the array of labels (y)
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            train_x = sc.fit_transform(train[training_vars])
            train_y = train.label.to_numpy()

            val_x = sc.transform(val[training_vars])
            val_y = val.label.to_numpy()

            test_x = sc.transform(test[training_vars])
            test_y = test.label.to_numpy()

            ### Temporary -- apply an extra weight to the signal region
            sample_weight = train.weight.to_numpy()

            ### Define model architecture 
            model = Sequential()
            model.add(Dense(layer_size, input_dim=len(training_vars), activation='relu')) 
            if dropout != 0: model.add(Dropout(dropout))
            model.add(Dense(layer_size, activation='relu'))
            if dropout != 0: model.add(Dropout(dropout))
            model.add(Dense(layer_size, activation='relu'))
            if dropout != 0: model.add(Dropout(dropout))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            ### Early stopping (stops training if loss doesn't improve for [patience] straight epochs)
            early_stopping = callbacks.EarlyStopping(monitor='loss', 
                                                     patience=patience, 
                                                     verbose=0) 

            ### Save the best weights
            weights_path = os.path.join(save_folder_val,"weights.h5")
            checkpoint = callbacks.ModelCheckpoint(weights_path, 
                                                   monitor='loss', 
                                                   mode='auto', 
                                                   verbose=0, 
                                                   save_best_only=True, 
                                                   save_weights_only=True)

            ### Add any additional callbacks for training
            callbacks_list = [checkpoint,early_stopping]
            if other_callbacks is not None:
                callbacks_list = callbacks_list + [other_callbacks]

            ### Train!
            history = model.fit(train_x, train_y, 
                        epochs=epochs, 
                        sample_weight=sample_weight,
                        batch_size=batch_size,
                        validation_data=(val_x, val_y),
                        callbacks = callbacks_list,
                        verbose = int(verbose),
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
            plt.savefig(os.path.join(save_folder_val,"loss_curve.png"))
            
            val_losses.append(np.min(history.history["val_loss"]))

            ### Add the NN prediction score to the test set: 
            test["nn_score"] = model.predict(test_x)
            test_scores.append(np.array(test.nn_score))

            ### Plot scores:
            plot_results(test, save_folder=save_folder_val, verbose=verbose)

        ### For each of the best models per validation set (measured by lowest val_loss), evaluate on the test set and take the average score for each star
        test["nn_score"] = np.mean(test_scores, axis=0) ### use AVERAGE score from all val sets
        test.to_hdf(os.path.join(save_folder_fold,"df_test.h5"), "df")
        test_dataframes.append(test)
        
        plot_results(test, save_folder=os.path.join(save_folder_fold, "before_fiducial_cuts"))
        plot_results(fiducial_cuts(test), save_folder=os.path.join(save_folder_fold, "after_fiducial_cuts"))

    ### Stitch all the test sets into a mega-Frankenstein-test set of the entire dataset
    test_full = pd.concat([df for df in test_dataframes])
    plot_results(test_full, save_folder=os.path.join(save_folder, "before_fiducial_cuts"))
    plot_results(fiducial_cuts(test_full), save_folder=os.path.join(save_folder, "after_fiducial_cuts"))
    return(test_full)

    

