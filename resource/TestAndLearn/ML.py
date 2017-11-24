'''
Created on 23 Nov. 2017

@author: Alvin UTS
'''
from dataprocess.readData import readData
from dataprocess.Clustering import kmeans_al
from dataprocess.Vi import showCluster
from dataprocess.dataNomalization import normalization
from dataprocess.dimReduction import DimReduction
from dataprocess.oneHot import oneHotData
from metric_learn import SDML
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation

def topic(df, num_topics=5):
    """
    Represent the topics features of original features
    :param df: pandas DataFrame
    :param num_topics: the number of topics, default=5
    :return: the probability vectors of each topics the entry belongs to
    """
#     X, y = df[df.columns[:-1]], df[df.columns[-1]]
    lda = LatentDirichletAllocation(n_topics=num_topics,
                                    max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    return lda.fit_transform(df)




if __name__ == "__main__":
    fileName = 'C:\Users\Alvin UTS\eclipse-workspace\ML_Alvin\demographic+Data+From+mimic.csv'
    toRows = 50
    
    data = readData(fileName, toRows)
    onehotdata = oneHotData(data)
    scaledData = normalization(onehotdata) 
    df_data = pd.DataFrame(scaledData) 
    mdl = topic(df_data, 8)
    cols = ['topic_{}'.format(i) for i in range(len(mdl[0]))]
    df_reperent = pd.DataFrame(mdl, columns=cols)
    
    #df_data = pd.read_csv('data/features_rep.csv')
    df_reperent.to_csv('data/alvin_rep.csv', index=False) 
    
    df_label = pd.read_csv('data/outcome_labels.csv')
    #print("df_label", '\n', df_label)
    
    #get unique row ids
    rowIDLIst = pd.concat([df_label.id1,df_label.id2],axis = 0).unique().tolist()
    #rowIDLIst2 = pd.concat([df_label.id1,df_label.id2],axis = 1)
    #print("rowIDLIst",'\n', rowIDLIst)
    #print("rowIDLIst2",'\n', rowIDLIst2)

    #connectivity graph
    cmatrix = np.zeros([len(rowIDLIst),len(rowIDLIst)])
   
    #print("as_Matrix", '\n', df_label.as_matrix)
    for lbl in df_label.as_matrix():
        #print ("lbl",lbl)
        #print ("lbl[0]",lbl[0])
        #print ("lbl[1]",lbl[1])
        #print ("lbl[2]",lbl[2])
        #print ("rowIDLIst.index(lbl[0])", rowIDLIst.index(lbl[0]),"rowIDLIst.index(lbl[1])",rowIDLIst.index(lbl[1]))
        cmatrix[rowIDLIst.index(lbl[0])][rowIDLIst.index(lbl[1])] = int(lbl[2])
        cmatrix[rowIDLIst.index(lbl[1])][rowIDLIst.index(lbl[0])] = int(lbl[2])
   
    print "cmatrix.shape", '\n', cmatrix.shape
    
    trainedData = []
    
    for rid in rowIDLIst:
        row = df_reperent.iloc[[rid]]
        #print "row","\n",row
        #print "rowType","\n",type(row)
        trainedData.append(row)
    
    
    #print "LentrainedData","\n", len(trainedData)
        
    #print "typetrainedData1", '\n', len(trainedData)
    
    trainedData = pd.concat(trainedData,axis = 0).as_matrix()   
    print "trainedData.shape","\n", trainedData.shape
    #print "trainedData2", "\n", trainedData

    metric = SDML().fit(trainedData, cmatrix)  
    
    newData = metric.transform(df_reperent) 
    print type(newData)
    print newData.shape
    

    
    
    
    