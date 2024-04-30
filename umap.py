# import sys

import pandas as pd
import numpy as np
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import umap

import os
from os import listdir
from os.path import join, isfile

def plot_helper(dataset, cols):
    scaler = StandardScaler()
    scaler.fit(dataset.loc[:, cols])
    scaled_data = scaler.transform(dataset.loc[:, cols])

    reducer = umap.UMAP(random_state=11)
    reducer.fit(scaled_data)
    embedding = reducer.transform(scaled_data)

    
    plot_data = dataset.iloc[:, :-31099]
    plot_data['vec1'] = embedding[:, 0]
    plot_data['vec2'] = embedding[:, 1]
    return scaler, reducer, plot_data  


def data_umap(data, cols, filename):
    dataUmap = data.iloc[:, :-len(cols)]
    # scale and umap embeding the generated genes
    X_scaler = scaler.transform(data[cols])
    X_embedding = reducer.transform(X_scaler)
    # construct the dataframe for the predicted umap vectors
    dataUmap['vec1'] = X_embedding[:, 0]
    dataUmap['vec2'] = X_embedding[:, 1]
    # save the dataUmap
    dataUmap.to_csv(filename)

# resultPath
path = '/compute01/tli/tggates/results/hpc3_s1500/github'
    
# get the gene feature columns
features = pd.read_csv('/account/tli/tggates/data/special/nieh_genes.csv').PROBEID.unique()
print(len(features))
# true value
liver_kidney = pd.read_csv('/account/tli/tggates/data/liver_kidney_train_test_31099.csv', low_memory=False)
# UMAP helper
scaler, reducer, plot_data = plot_helper(liver_kidney, features)
plot_data.to_csv(path + '/performance/umap/umap_org.csv')

kidney_train = plot_data[(plot_data.ORGAN == 'Kidney') & (plot_data.usage == 'train')].reset_index(drop=True).iloc[:, 1:].to_csv(path + '/performance/umap/train/control_train_kidney.csv')
liver_train = plot_data[(plot_data.ORGAN == 'Liver') & (plot_data.usage == 'train')].reset_index(drop=True).iloc[:, 1:].to_csv(path + '/performance/umap/train/control_train_liver.csv')

kidney_test = plot_data[(plot_data.ORGAN == 'Kidney') & (plot_data.usage == 'test')].reset_index(drop=True).iloc[:, 1:].to_csv(path + '/performance/umap/test/control_test_kidney.csv')
liver_test = plot_data[(plot_data.ORGAN == 'Liver') & (plot_data.usage == 'test')].reset_index(drop=True).iloc[:, 1:].to_csv(path + '/performance/umap/test/control_test_liver.csv')



### test
dataPath = path + '/predictions_decoded/test'
umapPath = path + '/performance/umap/test'
number='1526000'
# kidney
testKidney = pd.read_csv(dataPath + '/generator1_encoded_prediction_' + number + '_KidenyGenerator_test.csv')
data_umap(testKidney, features, umapPath+'/' + number + '_KidenyGenerator.csv')
# liver
testLiver = pd.read_csv(dataPath + '/generator1_encoded_prediction_' + number + '_LiverGenerator_test.csv')
data_umap(testLiver, features, umapPath+'/' + number + '_LiverGenerator.csv')


### train
dataPath = path + '/predictions_decoded/train'
umapPath = path + '/performance/umap/train'
# kidney
trainKidney = pd.read_csv(dataPath + '/generator1_encoded_prediction_' + number + '_KidenyGenerator.csv')
data_umap(trainKidney, features, umapPath+'/' + number + '_KidenyGenerator.csv')
# liver
trainLiver = pd.read_csv(dataPath + '/generator1_encoded_prediction_' + number + '_LiverGenerator.csv')
data_umap(trainLiver, features, umapPath+'/' + number + '_LiverGenerator.csv')

