'''

Created on 22 Nov. 2017

@author: Alvin UTS
'''

from readData import readData
from Clustering import kmeans_al
from Vi import showCluster
from dataNomalization import normalization
from dimReduction import DimReduction
from oneHot import oneHotData
from topicModelling import topic
from metricLearning import metricLearning
#import pandas as pd




if __name__ == "__main__":
    fileName = '../../demographic+Data+From+mimic.csv'
    toRows = 20
    
    data = readData(fileName, toRows)
    onehotdata = oneHotData(data)
    scaledData = normalization(onehotdata) 
    
    #df_scaledData = pd.DataFrame(scaledData) 
    df_OriData = topic(scaledData, 10)
    
    df_NewData = metricLearning(df_OriData)
    
    
    
    OriKmeansresult = kmeans_al(df_OriData)
    NewKmeansresult = kmeans_al(df_NewData)
    #print kmeansresult.labels_
    TwoDOriData = DimReduction(df_OriData)
    TwoDNewData = DimReduction(df_NewData)
    
    showCluster(TwoDOriData, OriKmeansresult.labels_)  
    showCluster(TwoDNewData, NewKmeansresult.labels_)  
    