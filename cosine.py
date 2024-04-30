import sys
#number=sys.argv[1]
#path=sys.argv[2]


import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization, ActivityRegularization
from tensorflow.keras import activations
from tensorflow import keras
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
import tensorflow as tf
from keras import regularizers
from sklearn.metrics.pairwise import cosine_similarity


# get the gene feature columns
features = pd.read_csv('/account/tli/tggates/data/special/nieh_genes.csv').PROBEID.unique()
print(len(features))
# true value
liver_kidney = pd.read_csv('/account/tli/tggates/data/liver_kidney_train_test_31099.csv', low_memory=False)

# cosine function for control and generated 
def control_cosine(result, filename, features=features, org=liver_kidney):
    values = []
    for i in range(len(result)):        
        # true values
        targetSample = result.loc[i, 'targetId']
        true_value = org[org['BARCODE'] == targetSample].loc[:, features].values[0]
        # predicted values
        sourceSample = result.loc[i, 'BARCODE']
        predicted_value = org[org['BARCODE'] == sourceSample].loc[:, features].values[0]
        cos=cosine_similarity(true_value.reshape(1,-1), predicted_value.reshape(1,-1))[0][0]
        values.append(cos)
    result['cosine'] = values
    result.to_csv(filename)

def generated_cosine(predicted, filename, features=features, org=liver_kidney):
    result = predicted.iloc[:, :-len(features)]
    values = []
    for i in range(len(result)):        
        # true values
        targetSample = result.loc[i, 'targetId']
        true_value = org[org['BARCODE'] == targetSample].loc[:, features].values[0]
        # predicted values
        predicted_value = predicted.loc[i, features].values
        cos=cosine_similarity(true_value.reshape(1,-1), predicted_value.reshape(1,-1))[0][0]
        values.append(cos)
    result['cosine'] = values
    result.to_csv(filename)

    
# resultPath
path = '/compute01/tli/tggates/results/hpc3_s1500/github'
# final model number
number='1526000'

### test
dataPath = path + '/predictions_decoded/test'
cosinePath = path + '/performance/cosine/test'
# kidney
testKidney = pd.read_csv(dataPath + '/generator1_encoded_prediction_' + number + '_KidenyGenerator_test.csv')
generated_cosine(testKidney, cosinePath+'/' + number + '_KidenyGenerator.csv')
# liver
testLiver = pd.read_csv(dataPath + '/generator1_encoded_prediction_' + number + '_LiverGenerator_test.csv')
generated_cosine(testLiver, cosinePath+'/' + number + '_LiverGenerator.csv')
# control
control_cosine(testKidney.iloc[:, :-len(features)], cosinePath+'/test_control_cosine.csv')

### train
dataPath = path + '/predictions_decoded/train'
cosinePath = path + '/performance/cosine/train'
# kidney
trainKidney = pd.read_csv(dataPath + '/generator1_encoded_prediction_' + number + '_KidenyGenerator.csv')
generated_cosine(trainKidney, cosinePath+'/' + number + '_KidenyGenerator.csv')
# liver
trainLiver = pd.read_csv(dataPath + '/generator1_encoded_prediction_' + number + '_LiverGenerator.csv')
generated_cosine(trainLiver, cosinePath+'/' + number + '_LiverGenerator.csv')
# control
control_cosine(trainKidney.iloc[:, :-len(features)], cosinePath+'/train_control_cosine.csv')


