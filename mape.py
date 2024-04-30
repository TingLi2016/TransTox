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
# true value
liver_kidney = pd.read_csv('/account/tli/tggates/data/liver_kidney_train_test_31099.csv', low_memory=False)

# calculate rmse value
def generated_mape(predicted, filepath, org=liver_kidney, features=features):
    predicted = predicted.reset_index(drop=True)
    result = predicted.iloc[:, :-len(features)]
    mapevalues = []
    for i in range(len(result)):
        # true values
        targetSample = result.loc[i, 'targetId']
        true_value = org[org['BARCODE'] == targetSample].loc[:, features].values[0]
        # predicted values
        predicted_value = predicted.loc[i, features].values
        
        error = (true_value - predicted_value)/true_value
        mapevalues.append(abs(error).mean())
        
    result['mape'] = mapevalues
    result.to_csv(filepath)
    
def control_mape(result, filename, org=liver_kidney, features=features):
    result = result.reset_index(drop=True)
    mapevalues = []
    for i in range(len(result)):        
        # true values
        targetSample = result.loc[i, 'targetId']
        true_value = org[org['BARCODE'] == targetSample].loc[:, features].values[0]
        # predicted values
        sourceSample = result.loc[i, 'BARCODE']
        predicted_value = org[org['BARCODE'] == sourceSample].loc[:, features].values[0]
        
        error = (true_value - predicted_value)/true_value
        mapevalues.append(abs(error).mean())
        
    result['mape'] = mapevalues
    result.to_csv(filename)
    
    
# path
path = '/compute01/tli/tggates/results/hpc3_s1500/github'
# final model number
number='1526000'


### test
dataPath = path + '/predictions_decoded/test'
mapePath = path + '/performance/mape/test'
# kidney
testKidney = pd.read_csv(dataPath + '/generator1_encoded_prediction_' + number + '_KidenyGenerator_test.csv')
generated_mape(testKidney, mapePath+'/' + number + '_KidenyGenerator.csv')
# liver
testLiver = pd.read_csv(dataPath + '/generator1_encoded_prediction_' + number + '_LiverGenerator_test.csv')
generated_mape(testLiver, mapePath+'/' + number + '_LiverGenerator.csv')
# control
control_mape(testKidney.iloc[:, :-len(features)], mapePath+'/test_control_mape.csv')

### train
dataPath = path + '/predictions_decoded/train'
mapePath = path + '/performance/mape/train'
# kidney
trainKidney = pd.read_csv(dataPath + '/generator1_encoded_prediction_' + number + '_KidenyGenerator.csv')
generated_mape(trainKidney, mapePath+'/' + number + '_KidenyGenerator.csv')
# liver
trainLiver = pd.read_csv(dataPath + '/generator1_encoded_prediction_' + number + '_LiverGenerator.csv')
generated_mape(trainLiver, mapePath+'/' + number + '_LiverGenerator.csv')
# control
control_mape(trainKidney.iloc[:, :-len(features)], mapePath+'/train_control_mape.csv')