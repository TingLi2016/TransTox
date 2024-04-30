import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam, SGD
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import BatchNormalization
from keras.layers import GaussianNoise
from keras.layers import Dropout
from keras.constraints import MaxNorm
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
#from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.preprocessing import LabelBinarizer
from random import randint
import random

# path
dataPath = '/account/tli/tggates/data'
resultPath = '/compute01/tli/tggates/results/hpc3_s1500/github/predictions_decoded/test'
# load the Kidney generator 
g_kidney = keras.models.load_model('/compute01/tli/tggates/results/hpc3_s1500/cycleGAN_dropout08/model2/g_model2_1526000.h5')
# load the Liver generator 
g_liver = keras.models.load_model('/compute01/tli/tggates/results/hpc3_s1500/cycleGAN_dropout08/model1/g_model1_1526000.h5')


def read_data(path):
    data = pd.read_csv(path)
    data = data.iloc[:, 1:]
    return data

def binarizer(data):
    organBinarizer = LabelBinarizer().fit(data["ORGAN"])
    doseBinarizer = LabelBinarizer().fit(data["DOSE_LEVEL"])
    stageBinarizer = LabelBinarizer().fit(data["SACRIFICE_PERIOD"])
    bioCopyBinarizer = LabelBinarizer().fit(data["INDIVIDUAL_ID"])
    return organBinarizer, doseBinarizer, stageBinarizer, bioCopyBinarizer

# read data set
liverTrain = read_data(dataPath + '/liver_train_31099.csv')
liverTest = read_data(dataPath + '/liver_test_31099.csv')
kidneyTrain = read_data(dataPath + '/kidney_train_31099.csv')
kidneyTest = read_data(dataPath + '/kidney_test_31099.csv')


# concatenate data set
dataset = pd.concat([liverTrain, liverTest, kidneyTrain, kidneyTest], axis = 0)
dataset = dataset.sort_values('EXP_ID', ascending = True)

# initialize binarizer to transfer category data (EXP_TEST_TYPE, SACRIFICE_PERIOD, DOSE_LEVEL) to binary data
organBinarizer, doseBinarizer, stageBinarizer, bioCopyBinarizer = binarizer(dataset)

# get the gene feature columns
cols = pd.read_csv('/account/tli/tggates/data/special/nieh_genes.csv').PROBEID.unique()
print(len(cols))



def scale(df1, df2, cols):
    X1 = df1[cols]
    X2 = df2[cols]
    # scale data
    scaler = MinMaxScaler()
    scaler.fit(X1)
    X2 = scaler.transform(X2)
    scaledf = pd.DataFrame(X2, columns = cols)
    resultdf = pd.concat([df2.iloc[:, :-31099], scaledf], axis = 1)
    return resultdf, scaler

#liverTrain, liverScaler = scale(liverTrain, liverTrain, cols=cols)
#kidneyTrain, kidneyScaler  = scale(kidneyTrain, kidneyTrain, cols=cols)

liverTest, _  = scale(liverTrain, liverTest, cols=cols)
liverTrain, liverScaler = scale(liverTrain, liverTrain, cols=cols)
kidneyTest, _  = scale(kidneyTrain, kidneyTest, cols=cols)
kidneyTrain, kidneyScaler = scale(kidneyTrain, kidneyTrain, cols=cols)



# select a batch of random samples, returns Xs and target
def generate_test_real_samples(liverTest, kidneyTest, cols=cols, organBinarizer=organBinarizer, doseBinarizer=doseBinarizer, stageBinarizer=stageBinarizer, bioCopyBinarizer=bioCopyBinarizer):
    dataset = pd.concat([liverTest, kidneyTest])
    dataset = dataset.sort_values(['COMPOUND_NAME', 'DOSE_LEVEL'])
    dataset = dataset.reset_index(drop=True)
    df = pd.DataFrame()
    
    def pairs(subFrom):
        tmpDf = pd.DataFrame()
        tmpDf = tmpDf.append([subFrom]*len(subFrom), ignore_index=True)
        tmpDf = tmpDf.reset_index(drop=True)
        subTo = tmpDf[['BARCODE', 'ORGAN', 'SACRIFICE_PERIOD', 'DOSE_LEVEL', 'INDIVIDUAL_ID']]
        subTo = subTo.sort_values('BARCODE')
        subTo.columns = ['targetId', 'targetOrgan', 'targetTime', 'targetDose', 'targetBioCopy']
        subTo = subTo.reset_index(drop = True)
        tmpDf = pd.concat([tmpDf, subTo], axis = 1)
        return tmpDf     
    
    for compound in dataset.COMPOUND_NAME.unique():
        subFrom = dataset[(dataset.COMPOUND_NAME == compound) & (dataset.DOSE_LEVEL == 'Control')]
        tmpDf = pairs(subFrom)
        df = df.append(tmpDf, ignore_index=True)
        subFrom = dataset[(dataset.COMPOUND_NAME == compound) & (dataset.DOSE_LEVEL != 'Control')]
        tmpDf = pairs(subFrom)
        df = df.append(tmpDf, ignore_index=True)
        
    def binaryRepresentation(data):
        data = data.reset_index(drop=True)
        # reform sample label in source
        organ = organBinarizer.transform(data['ORGAN'])
        dose = doseBinarizer.transform(data['DOSE_LEVEL'])
        time = stageBinarizer.transform(data['SACRIFICE_PERIOD'])
        bioCopy = bioCopyBinarizer.transform(data['INDIVIDUAL_ID'])

        # reform sample label in target
        targetOrgan = organBinarizer.transform(data['targetOrgan'])
        targetDose = doseBinarizer.transform(data['targetDose'])
        targetTime = stageBinarizer.transform(data['targetTime'])
        targetBioCopy = bioCopyBinarizer.transform(data['targetBioCopy'])
        
        input_noise = np.random.normal(0, 0.01, [len(np.array(data[cols])),len(np.array(data[cols])[0])])
        
        return data.drop([*cols], axis=1), np.array(data[cols]), np.hstack([organ, dose, time, bioCopy]), np.hstack([targetOrgan, targetDose, targetTime, targetBioCopy]), input_noise
        
        
    # drop the targetOrgan and Organ is the same pairs
    toKidney = df[(df.ORGAN != df.targetOrgan) & (df.ORGAN == 'Liver')]
    toLiver = df[(df.ORGAN != df.targetOrgan) & (df.ORGAN == 'Kidney')]
    
    # get the input for generator Liver
    masterKidney, X_kidney, X_kidney_Label, X_kidney_target, kidney_input_noise = binaryRepresentation(toLiver)
    # get the input for generator Kidney
    masterLiver, X_liver, X_liver_Label, X_liver_target, liver_input_noise = binaryRepresentation(toKidney)

    return masterKidney, X_kidney, X_kidney_Label, X_kidney_target, kidney_input_noise, masterLiver, X_liver, X_liver_Label, X_liver_target, liver_input_noise


def summarize_performance(step, g_model, input_gene, input_label, input_target, input_noise, orgDf, name, organScaler, features=cols, resultPath=resultPath):
    # make prediction
    X_out = g_model.predict([input_gene, input_label, input_target, input_noise])
    X_out = organScaler.inverse_transform(X_out)
    
    # save prediction
    X_out_df = pd.DataFrame(data=X_out, columns=features)
    X_out_df = pd.concat([orgDf, X_out_df], axis = 1)
    filename = resultPath + '/generator1_encoded_prediction_%06d_%s.csv' %(step, name)
    X_out_df.to_csv(filename)
    

    
### for train set
masterKidney, X_kidney, X_kidney_Label, X_kidney_target, kidney_input_noise, masterLiver, X_liver, X_liver_Label, X_liver_target, liver_input_noise = generate_test_real_samples(liverTrain, kidneyTrain)
# make Kidney predictions
summarize_performance(1526000, g_kidney, X_liver, X_liver_Label, X_liver_target, liver_input_noise, masterLiver, 'KidenyGenerator', kidneyScaler)
# make Liver predictions
summarize_performance(1526000, g_liver, X_kidney, X_kidney_Label, X_kidney_target, kidney_input_noise, masterKidney, 'LiverGenerator', liverScaler)


### for test set
masterKidney, X_kidney, X_kidney_Label, X_kidney_target, kidney_input_noise, masterLiver, X_liver, X_liver_Label, X_liver_target, liver_input_noise = generate_test_real_samples(liverTest, kidneyTest)
# make Kidney predictions
summarize_performance(1526000, g_kidney, X_liver, X_liver_Label, X_liver_target, liver_input_noise, masterLiver, 'KidenyGenerator_test', kidneyScaler)
# make Liver predictions
summarize_performance(1526000, g_liver, X_kidney, X_kidney_Label, X_kidney_target, kidney_input_noise, masterKidney, 'LiverGenerator_test', liverScaler)
