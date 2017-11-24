'''
Created on 22 Nov. 2017

@author: Alvin UTS
'''
from sklearn import preprocessing

def normalization(data):
    x = data.values #returns a numpy array
    #print type(x)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    print 'Data Normalization Done', '\n'
    #print type(x_scaled)
    #print x_scaled.shape
    #print x_scaled[:, 0: 5].shape
    return x_scaled