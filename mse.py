import sys

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import math
import seaborn as sns
import matplotlib.pyplot as plt

import os
from os import listdir
from os.path import join, isfile

# get the gene feature columns
features = pd.read_csv('/account/tli/tggates/data/special/nieh_genes.csv').PROBEID.unique()
print(len(features))
# true value
liver_kidney = pd.read_csv('/account/tli/tggates/data/liver_kidney_train_test_31099.csv', low_memory=False)


# calculate rmse value
def generated_rmse(predicted, filename, org=liver_kidney, features=features):
    predicted = predicted.reset_index(drop=True)
    result = predicted.iloc[:, :-len(features)]
    rmsevalues = []
    for i in range(len(result)):
        # true values
        targetSample = result.loc[i, 'targetId']
        true_value = org[org['BARCODE'] == targetSample].loc[:, features].values[0]
        # predicted values
        predicted_value = predicted.loc[i, features].values
        mse = mean_squared_error(true_value, predicted_value)
        rmse = math.sqrt(mse)
        rmsevalues.append(rmse)
    result['rmse'] = rmsevalues
    result.to_csv(filename)
    
def control_rmse(result, filename, org=liver_kidney, features=features):
    result = result.reset_index(drop=True)
    rmsevalues = []
    for i in range(len(result)):        
        # true values
        targetSample = result.loc[i, 'targetId']
        true_value = org[org['BARCODE'] == targetSample].loc[:, features].values[0]
        # predicted values
        sourceSample = result.loc[i, 'BARCODE']
        predicted_value = org[org['BARCODE'] == sourceSample].loc[:, features].values[0]
        
        mse = mean_squared_error(true_value, predicted_value)
        rmse = math.sqrt(mse)
        rmsevalues.append(rmse)
        
    result['rmse'] = rmsevalues
    result.to_csv(filename)
   

# path
path = '/compute01/tli/tggates/results/hpc3_s1500/github'
# final model number
number='1526000'


### test
dataPath = path + '/predictions_decoded/test'
rmsePath = path + '/performance/rmse/test'
# kidney
testKidney = pd.read_csv(dataPath + '/generator1_encoded_prediction_' + number + '_KidenyGenerator_test.csv')
generated_rmse(testKidney, rmsePath+'/' + number + '_KidenyGenerator.csv')
# liver
testLiver = pd.read_csv(dataPath + '/generator1_encoded_prediction_' + number + '_LiverGenerator_test.csv')
generated_rmse(testLiver, rmsePath+'/' + number + '_LiverGenerator.csv')
# control
control_rmse(testKidney.iloc[:, :-len(features)], rmsePath+'/test_control_cosine.csv')

### train
dataPath = path + '/predictions_decoded/train'
rmsePath = path + '/performance/rmse/train'
# kidney
trainKidney = pd.read_csv(dataPath + '/generator1_encoded_prediction_' + number + '_KidenyGenerator.csv')
generated_rmse(trainKidney, rmsePath+'/' + number + '_KidenyGenerator.csv')
# liver
trainLiver = pd.read_csv(dataPath + '/generator1_encoded_prediction_' + number + '_LiverGenerator.csv')
generated_rmse(trainLiver, rmsePath+'/' + number + '_LiverGenerator.csv')
# control
control_rmse(trainKidney.iloc[:, :-len(features)], rmsePath+'/train_control_cosine.csv')
