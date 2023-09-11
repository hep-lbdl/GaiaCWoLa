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
import keras.backend as K
from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

### Custom imports
from functions import *

def prepare_data(df, vars_list):
    """Prepare data for training/testing/validation."""
    sc = StandardScaler()
    x = sc.fit_transform(df[vars_list])
    y = df.label.to_numpy()
    return x, y

def build_model(input_dim, layer_size=200, dropout=0.2):
    """Create and compile a model."""
    model = Sequential()
    model.add(Dense(layer_size, input_dim=input_dim, activation='relu')) 
    if dropout != 0: model.add(Dropout(dropout))
    model.add(Dense(layer_size, activation='relu'))
    if dropout != 0: model.add(Dropout(dropout))
    model.add(Dense(layer_size, activation='relu'))
    if dropout != 0: model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def reset_weights(model):
    session = K.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer') and layer.kernel.initializer is not None:
            layer.kernel.initializer.run(session=session)
            

def get_callbacks(patience, weights_path):
    """Generate a list of callbacks for Keras model."""
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=0)
    checkpoint = callbacks.ModelCheckpoint(weights_path, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True)
    return [checkpoint, early_stopping]




def train(df, layer_size=200, batch_size=10000, dropout=0.2, epochs=100, patience=30, n_folds=5, best_of_n_loops=3, save_folder=None, other_callbacks=None, verbose=True, scan_over_mu_phi=False, apply_cuts=True):
    os.makedirs(save_folder, exist_ok=True)
    if scan_over_mu_phi:
        training_vars = ['ϕ', 'λ', 'μ_λ', 'b-r', 'g']
    else:
        training_vars = ['ϕ', 'λ', 'μ_ϕcosλ', 'b-r', 'g']
    
    training_vars = ['p1', 'p2', 'p3', 'p4', 'p5']

    # Build the model
    model=build_model(len(training_vars))
   
    ### Explicitly get indices of stars for each k-fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=15)
    fold_stars = []
    for fold, (train_index, test_index) in enumerate(skf.split(df[training_vars], df.label)):
        fold_stars.append(test_index)
    
    ### Loop through the k-folds
    fold_labels = np.arange(len(fold_stars))
    test_dataframes = []
    for fold in fold_labels:
        save_folder_fold = os.path.join(save_folder,"kfold_{}".format(fold))
        
        ### Define test set
        test_stars = fold_stars[fold]
        
        ### Loop through all remaining val sets
        test_scores = []
        for val_set in np.delete(fold_labels, fold):
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
            train_x, train_y= prepare_data(train, training_vars)
            val_x, val_y= prepare_data(val, training_vars)
            test_x, test_y= prepare_data(test, training_vars)
            
            ### Temporary -- apply an extra weight to the signal region
            if "weight" not in train.keys(): 
                sample_weight = None
            elif len(train.weight.unique()) == 1: 
                sample_weight = None
            else:
                sample_weight = train.weight.to_numpy()
            
            ### Repeat this training multiple times to find the one with the lowest loss 
            val_losses = []
            for n in range(best_of_n_loops): 
                os.makedirs(os.path.join(save_folder_val, "loop_{}".format(n)), exist_ok=True)
                
                ### Reset model weights
                reset_weights(model)

                ### Early stopping (stops training if val_loss doesn't improve for [patience] straight epochs)
                early_stopping = callbacks.EarlyStopping(monitor='val_loss', 
                                                         patience=patience, 
                                                         verbose=0) 

                ### Save the best weights
                weights_path = os.path.join(save_folder_val,"loop_{}".format(n),"weights.h5")
                checkpoint = callbacks.ModelCheckpoint(weights_path, 
                                                       monitor='val_loss', 
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
                            verbose = 0,
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
                plt.savefig(os.path.join(save_folder_val, "loop_{}".format(n), "loss_curve.png"))
                plt.close()

                val_losses.append(np.min(history.history["val_loss"]))

            ### Choose the loop with the lowest val_loss 
            model.load_weights(os.path.join(save_folder_val,"loop_{}".format(np.argmin(val_losses)),"weights.h5"))
            
            ### Add the NN prediction score to the test set: 
            test["nn_score"] = model.predict(test_x)
            test_scores.append(np.array(test.nn_score))

            ### Plot scores:
            plot_results(test, save_folder=save_folder_val, verbose=verbose, show=False)

        ### For each of the best models per validation set (measured by lowest val_loss), evaluate on the test set and take the average score for each star
        test["nn_score"] = np.mean(test_scores, axis=0) ### use AVERAGE score from all val sets
        test.to_hdf(os.path.join(save_folder_fold,"df_test.h5"), "df")
        test_dataframes.append(test)
        
        print("Plotting results before fiducial cuts...")
        plot_results(test, show=False, save_folder=os.path.join(save_folder_fold, "before_fiducial_cuts"))
        if apply_cuts and len(fiducial_cuts(test) > 0):
            print("Plotting results after fiducial cuts...")
            plot_results(fiducial_cuts(test), show=False, save_folder=os.path.join(save_folder_fold, "after_fiducial_cuts"))

    ### Stitch all the test sets into a mega-Frankenstein-test set of the entire dataset
    test_full = pd.concat([df for df in test_dataframes])
    plot_results(test_full, save_folder=os.path.join(save_folder, "before_fiducial_cuts"))
    test_full.to_hdf(os.path.join(save_folder,"df_test.h5"), "df")

    if apply_cuts and len(fiducial_cuts(test) > 0):
        plot_results(fiducial_cuts(test_full), save_folder=os.path.join(save_folder, "after_fiducial_cuts"))
        
    return(test_full)
    