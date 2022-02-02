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
from livelossplot.keras import PlotLossesCallback
from livelossplot import PlotLossesKeras

### Custom imports
from functions import *

def train(df_slice, save_folder="test", n_folds=5, epochs=100, batch_size=32, layer_size=10, dropout=0, l2_reg=0, patience=10, best_of_n_loops=1, other_callbacks=None):
    os.makedirs(save_folder, exist_ok=True)
    if 'color' in df_slice.keys(): 
        training_vars = ['μ_α','δ','α','color','mag']
    elif 'b-r' in df_slice.keys():
        training_vars = ['μ_α','δ','α','g','b-r']
    train, validate, test = np.split(df_slice.sample(frac=1), [int(.7*len(df_slice)), int(.85*len(df_slice))]) # 70/15/15 train/validate/test split

    x_train, x_val, x_test = [train[training_vars], validate[training_vars], test[training_vars]]
    y_train, y_val, y_test = [train.label, validate.label, test.label]
    
    if 'weight' in df_slice.keys():
        sample_weight = train.weight
        print("Using stream weight = {}".format(train.weight.unique().max()))
    else:
        print("Not using sample weights")
        sample_weight = None

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    x_val = sc.transform(x_val)

    print("Training on {:,} events.".format(len(train)))

    if n_folds <= 1:  # train without k-folding
        best_losses = []
        for loop in range(best_of_n_loops): 
            ### Define model architecture 
            reg = regularizers.l2(l2_reg)
            model = Sequential()
            model.add(Dense(layer_size, input_dim=len(training_vars), activation='relu', activity_regularizer=reg)) 
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

            callbacks_list = [PlotLossesKeras(),checkpoint,early_stopping]
            if other_callbacks is not None:
                callbacks_list = callbacks_list + other_callbacks
                            
            ### Train!
            history = model.fit(x_train, y_train, 
                        epochs=epochs, 
                        sample_weight=sample_weight,
                        batch_size=batch_size,
                        validation_data=(x_val,y_val),
                        callbacks = callbacks_list,
                        verbose = 0,
                       )
            best_losses.append(np.min(history.history['loss']))
        
        ### Load best weights
        print("Best losses:", best_losses)
        print("Loading weights from best loop, i.e. loop #{}.".format(np.argmin(best_losses)))
        best_weights_path = os.path.join(save_folder,"weights_loop{}.h5".format(np.argmin(best_losses)))
        model.load_weights(best_weights_path)

    elif n_folds > 1: 
        # Define per-fold score containers
        acc_per_fold = []
        loss_per_fold = []

        inputs = np.concatenate((x_train,x_val), axis=0)
        targets = np.concatenate((y_train,y_val), axis=0)
    
        # Define the K-fold Cross Validator
        from sklearn.model_selection import KFold
        kfold = KFold(n_splits=n_folds, shuffle=True)
        fold_number = 0

        for train, validate in kfold.split(inputs, targets):
            print("\nTraining fold #{}...".format(fold_number))
            best_losses = []
            for loop in range(best_of_n_loops): 
                ### Define model architecture 
                reg = regularizers.l2(l2_reg)
                model = Sequential()
                model.add(Dense(layer_size, input_dim=len(training_vars), activation='relu', activity_regularizer=reg)) 
                if dropout != 0: model.add(Dropout(dropout))
                model.add(Dense(layer_size, activation='relu', activity_regularizer=reg))
                if dropout != 0: model.add(Dropout(dropout))
                model.add(Dense(layer_size, activation='relu', activity_regularizer=reg))
                if dropout != 0: model.add(Dropout(dropout))
                model.add(Dense(1, activation='sigmoid'))
                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#                 model.summary()

                # stops if val_loss doesn't improve for [patience] straight epochs
                early_stopping = callbacks.EarlyStopping(monitor='val_loss', 
                                                         patience=patience, 
                                                         verbose=1) 

                # saves weights from the epoch with lowest val_loss 
                checkpoint = callbacks.ModelCheckpoint(os.path.join(save_folder,"kfold{}_loop{}_weights.h5".format(fold_number,loop)), 
                                                       monitor='val_loss', 
                                                       mode='auto', 
                                                       verbose=1, 
                                                       save_best_only=True, 
                                                       save_weights_only=True)

                history = model.fit(inputs[train], targets[train], 
                                epochs=epochs, 
                                batch_size=batch_size,
                                validation_data = (inputs[validate], targets[validate]),
                                callbacks = [
                                             # PlotLossesKeras(),
                                             checkpoint,early_stopping],
                                verbose = 1,
                               )
                best_losses.append(np.min(history.history['loss']))

            ### Load best weights
            print("Best losses:", best_losses)
            print("Loading weights from best loop, i.e. loop #{}.".format(np.argmin(best_losses)))
            weights_path = os.path.join(save_folder,"kfold{}_loop{}_weights.h5".format(fold_number, np.argmin(best_losses)))
            model.load_weights(weights_path)
            shutil.copy(os.path.join(save_folder,"kfold{}_loop{}_weights.h5".format(fold_number, np.argmin(best_losses))),os.path.join(save_folder,"kfold{}_best_weights.h5".format(fold_number)))
                
            # Evaluate trained model
            scores = model.evaluate(inputs[validate], targets[validate], verbose=0)
            y_pred = model.predict(inputs[validate]).ravel()
            print('Score for fold {}: {} of {:.2f}; {} of {:.2f}%'.format(fold_number,model.metrics_names[0], scores[0],model.metrics_names[1],scores[1]*100))
            acc_per_fold.append(scores[1] * 100)
            loss_per_fold.append(scores[0])
            fold_number += 1

        # == Provide average scores ==
        print('------------------------------------------------------------------------')
        print('Score per fold')
        for i in range(0, len(acc_per_fold)):
            print('------------------------------------------------------------------------')
            print('> Fold {} - Loss: {:.2f} - Accuracy: {:.2f}%'.format(i+1, loss_per_fold[i], acc_per_fold[i]))
        print('------------------------------------------------------------------------')
        print('Average scores for all folds:')
        print('> Accuracy: {:.2f} (+- {:.2f})'.format(np.mean(acc_per_fold),np.std(acc_per_fold)))
        print('> Loss: {:.2f}'.format(np.mean(loss_per_fold)))
        print('------------------------------------------------------------------------')
        print('Best fold number (lowest loss): {}'.format(np.argmin(loss_per_fold)))

        best_fold_number = np.argmin(loss_per_fold)
        model.load_weights(os.path.join(save_folder,"kfold{}_best_weights.h5".format(best_fold_number)))
        
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
    
    ### Add the NN prediction score to the test set: 
    test["nn_score"] = model.predict(x_test)
    fake_eff_baseline, real_eff_baseline, thresholds = roc_curve(np.asarray(y_test), test.nn_score)
    auc_baseline = auc(fake_eff_baseline, real_eff_baseline)
    print("AUC: {:.3f}".format(auc_baseline))

    ### Plot scores:
    plot_results(test, save_folder=save_folder)
    
    ### Save test DataFrame for future plotting
    test.to_hdf(os.path.join(save_folder,"df_test.h5"), "df")
    
    return(test)
    
    

