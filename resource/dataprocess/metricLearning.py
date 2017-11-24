'''
Created on 24 Nov. 2017

@author: Alvin UTS
'''
import numpy as np
import pandas as pd
from metric_learn import SDML

def metricLearning (data):
    df_label = pd.read_csv('../TestAndLearn/data/outcome_labels.csv')
    #print("df_label", '\n', df_label)
    
    #get unique row ids
    rowIDLIst = pd.concat([df_label.id1,df_label.id2],axis = 0).unique().tolist()
    print "rowIDLIst",'\n', rowIDLIst
    
    #connectivity graph
    cmatrix = np.zeros([len(rowIDLIst),len(rowIDLIst)])
    
    #print("as_Matrix", '\n', df_label.as_matrix)
    for lbl in df_label.as_matrix():
       
        #print ("rowIDLIst.index(lbl[0])", rowIDLIst.index(lbl[0]),"rowIDLIst.index(lbl[1])",rowIDLIst.index(lbl[1]))
        cmatrix[rowIDLIst.index(lbl[0])][rowIDLIst.index(lbl[1])] = int(lbl[2])
        cmatrix[rowIDLIst.index(lbl[1])][rowIDLIst.index(lbl[0])] = int(lbl[2])
    
    print "cmatrix.shape", '\n', cmatrix.shape
    
    trainedData = []
    
    for rid in rowIDLIst:
        row = data.iloc[[rid]]
        #print "row","\n",row
        #print "rowType","\n",type(row)
        trainedData.append(row)
    
    
    #print "LentrainedData","\n", len(trainedData)
        
    #print "typetrainedData1", '\n', len(trainedData)
    
    trainedData = pd.concat(trainedData,axis = 0).as_matrix()   
    print "trainedData.shape","\n", trainedData.shape
    #print "trainedData2", "\n", trainedData
    
    metric = SDML().fit(trainedData, cmatrix)  
    
    newData = metric.transform(data) 
    return newData