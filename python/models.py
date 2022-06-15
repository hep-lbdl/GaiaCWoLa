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
        
    loop_purities = []
    for loop in tqdm(np.arange(best_of_n_loops), desc="Loop"):
        
        ### Standardize the inputs (x) and create the array of labels (y)
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        inputs = sc.fit_transform(df[training_vars])
        labels = df.label.to_numpy()
        
        ### Temporary -- apply an extra weight to the signal region
        sample_weight = df.weight.to_numpy()
        
        ### Stratified K-Folding will create folds with equal ratios of signal vs. sideband stars
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
        fold_purities = []
        for fold, (train_index, test_index) in enumerate(skf.split(inputs, labels)):
            save_folder_fold = os.path.join(save_folder,"kfold_{}".format(fold))
            os.makedirs(save_folder_fold, exist_ok=True)
            print("Training k-fold {}...".format(fold))
            x_train, x_test = inputs[train_index], inputs[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            test = df.iloc[test_index] # hang onto the dataframe for plotting etc. 
            
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
            weights_path = os.path.join(save_folder_fold,"weights.h5")
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
            history = model.fit(x_train, y_train, 
                        epochs=epochs, 
                        sample_weight=sample_weight[train_index],
                        batch_size=batch_size,
#                         validation_data=(x_val,y_val),
                        callbacks = callbacks_list,
                        verbose = int(verbose),
                       )

            ### Save training losses & accuracies
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize = (12,6))
            ax = axs[0]
            ax.plot(history.history["accuracy"], label="Training Accuracy")
#             ax.plot(history.history["val_accuracy"], label="Validation Accuracy")
            ax.set_title("Accuracy")
            ax.set_xlabel("Epochs")
            ax.legend()

            ax = axs[1]
            ax.plot(history.history["loss"], label="Training Loss")
#             ax.plot(history.history["val_loss"], label="Validation Loss")
            ax.set_title("Loss")
            ax.set_xlabel("Epochs")
            ax.legend()
            plt.savefig(os.path.join(save_folder_fold,"loss_curve.png"))

            ### Add the NN prediction score to the test set: 
            test["nn_score"] = model.predict(x_test)
            test.to_hdf(os.path.join(save_folder_fold,"df_test.h5"), "df")

            if "stream" in test.keys():
                # Scan for optimal percentage
                top_stars = test.sort_values('nn_score',ascending=False)[:50] # top 50 stars
                stream_stars_in_test_set = test[test.stream == True]
                if True in top_stars.stream.unique(): 
                    n_perfect_matches = top_stars.stream.value_counts()[True] 
                    stream_stars_in_test_set = test[test.stream == True]
                    efficiency = 100*n_perfect_matches/len(stream_stars_in_test_set)
                    purity = n_perfect_matches/len(top_stars)*100
                else: 
                    n_perfect_matches = 0 
                    efficiency = 0
                    purity = 0

            fold_purities.append(np.round(purity, decimals=2))

#         max_loop_purities = [np.nanmax(loop_purity) for loop_purity in loop_purities]
#         print("Max loop purities:", max_loop_purities)

            ### Plot scores:
            plot_results(test, save_folder=save_folder_fold, verbose=verbose)

        ### Load the weights from the best k-fold (measured by purity) 
        print("Fold purities:", fold_purities)
        print("Average purity = {}".format(np.mean(fold_purities)))
        print("Best k-fold = {}, with a purity of {:.2f}%.".format(np.argmax(fold_purities), np.nanmax(fold_purities)))
        print("Loading weights from k-fold {}...".format(np.argmax(fold_purities)))
        best_fold_folder = os.path.join(save_folder,"kfold_{}".format(np.argmax(fold_purities)))
        model.load_weights(os.path.join(best_fold_folder,"weights.h5"))      
        test = pd.read_hdf(os.path.join(best_fold_folder,"df_test.h5"))
        
        ### Save best model & final performance plots
        os.makedirs(os.path.join(save_folder, "before_fiducial_cuts"), exist_ok=True)
        os.makedirs(os.path.join(save_folder, "after_fiducial_cuts"), exist_ok=True)
        test.to_hdf(os.path.join(save_folder, "before_fiducial_cuts", "df_test.h5"), "df")
        model.save_weights(os.path.join(save_folder, "before_fiducial_cuts", "weights.h5"))
        plot_results(test, save_folder=os.path.join(save_folder, "before_fiducial_cuts"))
        plot_results(fiducial_cuts(test), save_folder=os.path.join(save_folder, "after_fiducial_cuts"))
    return(test)

    

