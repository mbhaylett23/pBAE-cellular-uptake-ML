# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 19:11:05 2023

@author: mbruy
"""
import os
import pandas as pd
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import  mean_absolute_error
import datetime
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from tensorflow import keras
from keras import models
from keras import layers 
import kerastuner as kt


df = pd.read_csv('D:\gene_expression\Final-Year-Project-Machine-Learning-main\Final-Year-Project-Machine-Learning-main\GeneDataforModel.csv')
saveDIR=('D:\gene_expression\Final-Year-Project-Machine-Learning-main\Final-Year-Project-Machine-Learning-main\GenesSINcelltype')
print(df)

# split the data in 80:10:10 for train:valid:test dataset
train_size=0.8

# X = df[['C6H', 'C6R', 'C6K', 'C32H', 'C32R', 'C32K', 'CellType', 'Size', 'Polydispersity','ACTN1','ACTN2','ACTN3','ACTN4','AGRIN','BDNF','CAV1','CAV2','CAV3','CD44','CFL2','COL4A1','COL6A1','COL6A2','COL9A2','CSF1','EGFR','ETS1','FGF1','FGF18','FGF6','FGF9','FGFR2','FLNA','FSCN1','FZD2','GNAI2','GNG12','IL7R','ITGA3','ITGA5','ITGA6','ITGB1','ITGB6','ITGB8','MET','MSN','MYH9','MYLK','NGF','NOTCH2','NOTCH3','OSMR','PDGFC','PLAU','RRAS','RRAS2','SDC4','TGFB1','THBS1','VEGFC','VIM','ZYX']]
X = df[['C6H', 'C6R', 'C6K', 'C32H', 'C32R', 'C32K', 'Size', 'Polydispersity','ACTN1','ACTN2','ACTN3','ACTN4','AGRIN','BDNF','CAV1','CAV2','CAV3','CD44','CFL2','COL4A1','COL6A1','COL6A2','COL9A2','CSF1','EGFR','ETS1','FGF1','FGF18','FGF6','FGF9','FGFR2','FLNA','FSCN1','FZD2','GNAI2','GNG12','IL7R','ITGA3','ITGA5','ITGA6','ITGB1','ITGB6','ITGB8','MET','MSN','MYH9','MYLK','NGF','NOTCH2','NOTCH3','OSMR','PDGFC','PLAU','RRAS','RRAS2','SDC4','TGFB1','THBS1','VEGFC','VIM','ZYX']]
cellIDX = df['CellType']
cellUNIQUE=np.unique(cellIDX)
shuffle(cellUNIQUE)
print(cellUNIQUE)
y = df['Uptake']

print(X.columns)
print(y)

Xa=X.values
ya=y.values
           
Xa                                   

# First, we split the data into the training and the remaining dataset
X_train, X_rem, y_train, y_rem = train_test_split(Xa,ya, train_size=0.8, random_state = 7)

# Next, we want the validation and test set to be each 10% of the overall data 
# We define valid_size=0.5 and test_size=0.5 so they are split equally
test_size = 0.5
X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5, random_state = 7)


# cT_train=X_train[:,6].copy()
# cT_test=X_test[:,6].copy()
# cT_valid=X_valid[:,6].copy()

# print(cT_train)
# print(cT_test)

print(X_train.shape), print(y_train.shape)
print(X_test.shape), print(y_test.shape)
print(X_train)

#data normalisation
mean = X_train.mean(axis=0)
print(mean.max(0))
X_train -= mean
std = X_train.std(axis=0)
print(std)
X_train /= std


# print(X_train[:,6])
# X_train[:,6]=cT_train.copy()
# print(X_train[:,6])

X_valid -=mean
X_valid/=std

X_test -=mean
X_test/=std
# print(X_test[:,6])
# print(X_valid[:,6])
# X_test[:,6]=cT_test.copy()
# X_valid[:,6]=cT_valid.copy()
# print(X_test[:,6])
# print(X_valid[:,6])

#Training Model
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)


#Calculating error of model on test set
prediction_linear = np.zeros(len(y_test))
for i in range(len(y_test)):
    prediction_linear[i] = regr.predict([(X_test)[i]])
mae = mean_absolute_error(y_test, prediction_linear)
print(mae)


#Random Forest Model
#Grid search on random forest model to find optimal number of trees and depth of each tree
## Define Grid 
grid = { 
    'n_estimators': [50,75, 100, 130, 140, 160, 150, 200, 250, 300],
    'max_depth' : [5,10,13,14,15,16,17,18, 20,25],
}
## show start time
print(datetime.datetime.now())
## Grid Search function
CV_rfr = GridSearchCV(estimator=RandomForestRegressor(), param_grid=grid, cv= 5)
CV_rfr.fit(X_train, y_train)
## show end time
print(datetime.datetime.now())
CV_rfr.best_params_

#Training model with optimal parameters
model_RF = RandomForestRegressor(n_estimators = 150, 
                            max_features = 'sqrt', max_depth = 18)
model_RF.fit(X_train, y_train)

#Calculating error of model
prediction_RF = model_RF.predict(X_test)
mae = mean_absolute_error(y_test, prediction_RF)
print(mae)

#Gradient Boosting Trees Model
#Grid search on Gradient Boosting Trees model to find optimal number of trees and depth of each tree
## Define Grid 
grid = { 
    'n_estimators': [50,75, 100, 130, 140, 160, 150, 200, 250, 300],
    'max_depth' : [5,10,13,14,15,16,17,18, 20,25],
}
## show start time
print(datetime.datetime.now())
## Grid Search function
CV_rfr = GridSearchCV(estimator=GradientBoostingRegressor(), param_grid=grid, cv= 5)
CV_rfr.fit(X_train, y_train)
## show end time
print(datetime.datetime.now())
CV_rfr.best_params_

#Training model with optimal parameters
model_boost = GradientBoostingRegressor(n_estimators = 50, max_depth = 5, )
model_boost.fit(X_train,y_train)

#Calculating error of model
prediction_boost = model_boost.predict(X_test)
mae = mean_absolute_error(y_test, prediction_boost)
print(mae)

#Neural Network
#Define function to create neural network with different parameters to be tuned
def model_builder(hp):
  '''
  Args:
    hp - Keras tuner object
  '''
  # Initialize the Sequential API and start stacking the layers
  model = keras.Sequential()
  model.add(keras.layers.Dense(40, activation='relu',
                            input_shape=(X_train.shape[1],)))
  # Tune the number of units in the first Dense layer
  # Choose an optimal value between 1-100
  hp_units = hp.Int('units', min_value=1, max_value=100, step=2)
  model.add(keras.layers.Dense(units=hp_units, activation='relu'))
  # Add next layers
  model.add(keras.layers.Dense(units=hp_units, activation='relu'))
  #model.add(keras.layers.Dense(units=hp_units, activation='relu'))
  model.add(keras.layers.Dense(1))
  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2])
  model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=hp_learning_rate),
                loss='mse',
                metrics=[keras.metrics.MeanAbsoluteError()])
  return model
#keras.optimizers.Adam(learning_rate=hp_learning_rate)
#metrics=[keras.metrics.MeanAbsoluteError()]

#Use Keras Tuner to find the optimal parameters resulting in the lowest error
tuner = kt.RandomSearch(model_builder, objective = kt.Objective("val_mean_absolute_error",direction="min"), max_trials=200,overwrite = True, directory='my_dir')
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(X_train, y_train, epochs=100, validation_data = (X_valid, y_valid), callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")


tuner.results_summary()

#Build neural network with optimal parameters found - 2 hidden layers, each with 41 nodes
#Model has two hidden layers each with 41 units 
def build_model():
    # Since we will need to instantiate
    # the same model multiple times,
    # we use a function to construct it.
    model = models.Sequential()
    model.add(layers.Dense(41, activation='relu',
                            input_shape=(X_train.shape[1],)))
    
    model.add(layers.Dense(41, activation='relu'))
    model.add(layers.Dense(1))
    #model.compile(optimizer='RMSprop', loss='mse', metrics=['mae'])
    #optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate)
    model.compile(optimizer='RMSprop', loss='mse', metrics=['mae'])
    return model

#Train the model over 100 epochs on the training set, and calculate error with validation set
num_epochs = 100
#all_scores = []
all_mae_histories = []

#Build the Keras model
model = build_model()
# Train the model (in silent mode, verbose=0) and evaluate on validation data 
history = model.fit(X_train, y_train, validation_data = (X_valid, y_valid), epochs=num_epochs, batch_size=5, verbose=0)
# Evaluate the model on the validation data
#val_mse, val_mae = model.evaluate(X_valid, y_valid, verbose=0)
#all_scores.append(val_mae)
mae_history = history.history['val_mae']
all_mae_histories.append(mae_history)

#Observe validation set error
#Print the validation set error of model
print(history.history.keys())
val_mse, val_mae = model.evaluate(X_valid, y_valid, verbose=0)
print(val_mae)
print(min(mae_history))

#Observe training and test set error
#Print the training set and test set error
train_mse, train_mae = model.evaluate(X_train, y_train, verbose=0)
print(train_mae)

test_mse, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(test_mae)

#Plot the validation error over number of epochs to observe learning of the model
#Plot the validation set error as a function of the number of epochs of the model 
plt.plot(mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.title('Validation MAE During Training - 2 Hidden Layers')
plt.show()

#SHAP Analysis on Gradient Boosting Trees Model
import shap
explainer = shap.TreeExplainer(model_boost)
print(explainer)
shap_values = shap.TreeExplainer(model_boost).shap_values(X_train)

shap.summary_plot(shap_values, X_train, plot_type="bar",feature_names= X.columns,  max_display=100,show=False)
plt.title('Global Feature Importance')
dirname = os.path.dirname(saveDIR)
results_dir = os.path.join(dirname, 'Results/')
print(results_dir)
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)
plt.savefig(results_dir + "GlobalFeatureImportance.svg")
plt.savefig(results_dir + "GlobalFeatureImportance.png")
plt.show()


shap.summary_plot(shap_values, X_train,feature_names= X.columns, max_display=100, show=False)
plt.gcf().axes[-1].set_aspect(100)
plt.gcf().axes[-1].set_box_aspect(100)
plt.title('Local Explanation Summary', fontsize = 12)
plt.gcf()
plt.savefig(results_dir + "LocalExplanationSummary.svg")
plt.savefig(results_dir + "LocalExplanationSummary.png")
plt.show()

# shap.initjs()

# shap.force_plot(explainer.expected_value[0],shap_values[3],X.iloc[[3]])

# shap.force_plot(explainer.expected_value,shap_values,X_train)


shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0], shap_values[1],feature_names= X.columns,max_display=100,show=False)
plt.gcf()
plt.savefig(results_dir + "WaterfallPLOTsingle.svg")
plt.savefig(results_dir + "WaterfallPLOTsingle.png")
plt.show()


shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0], shap_values[1],feature_names= X.columns,max_display=100,show=False)
plt.savefig(results_dir + "WaterfallPLOT.svg")
plt.savefig(results_dir + "WaterfallPLOT.png")
plt.show()


shap.decision_plot(explainer.expected_value[0], shap_values[0],feature_names= X.columns.tolist(),show=False)
plt.savefig(results_dir + "decision_plotSINGLE.svg")
plt.savefig(results_dir + "decision_plotSINGLE.png")
plt.show()

shap.decision_plot(explainer.expected_value, shap_values,feature_names= X.columns.tolist(),show=False)
plt.savefig(results_dir + "decision_plot.svg")
plt.savefig(results_dir + "decision_plot.png")
plt.show()

# plt.figure()
# shap.dependence_plot("Size", shap_values,X_train, interaction_index = "CellType",feature_names= X.columns, dot_size = 70, show=False)
# plt.title('Partial Dependence Plot: Size and Cell Type', fontsize = 14)
# plt.figure()
# shap.dependence_plot("C32H", shap_values,X_train, interaction_index = "CellType",feature_names= X.columns, dot_size = 70, show=False)
# plt.title('Partial Dependence Plot: C32H and Cell Type', fontsize = 14)
# plt.figure()
# shap.dependence_plot("C32K", shap_values,X_train, interaction_index = "CellType",feature_names= X.columns, dot_size = 70, show=False)
# plt.title('Partial Dependence Plot: C32K and Cell Type', fontsize = 14)
# plt.figure()
# shap.dependence_plot("C32R", shap_values,X_train, interaction_index = "CellType",feature_names= X.columns, dot_size = 70, show=False)
# plt.title('Partial Dependence Plot: C32R and Cell Type', fontsize = 14)
# plt.figure()
# shap.dependence_plot("C6H", shap_values,X_train, interaction_index = "CellType",feature_names= X.columns, dot_size = 70, show=False)
# plt.title('Partial Dependence Plot: C6H and Cell Type', fontsize = 14)
# plt.figure()
# shap.dependence_plot("C6K", shap_values,X_train, interaction_index = "CellType",feature_names= X.columns, dot_size = 70, show=False)
# plt.title('Partial Dependence Plot: C6K and Cell Type', fontsize = 14)
# plt.figure()
# shap.dependence_plot("C6R", shap_values,X_train, interaction_index = "CellType",feature_names= X.columns, dot_size = 70, show=False)
# plt.title('Partial Dependence Plot: C6R and Cell Type', fontsize = 14)
# plt.figure()
# shap.dependence_plot("Polydispersity", shap_values,X_train, interaction_index = "CellType",feature_names= X.columns, dot_size = 70, show=False)
# plt.title('Partial Dependence Plot: Polydispersity and Cell Type', fontsize = 14)
