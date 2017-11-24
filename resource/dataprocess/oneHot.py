'''
Created on 20 Nov. 2017

@author: Alvin UTS
'''
import pandas as pd
def oneHotData(data):
    print 'One Hot Encoding Done', '\n'
    return pd.get_dummies(data)