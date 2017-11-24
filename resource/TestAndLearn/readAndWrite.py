'''
Created on 24 Nov. 2017

@author: Alvin UTS
'''
import pandas as pd

data = pd.read_csv('../dataprocess/data/testRead.csv')

print data

data.to_csv('../dataprocess/data/toTestWrite.csv', index=False)