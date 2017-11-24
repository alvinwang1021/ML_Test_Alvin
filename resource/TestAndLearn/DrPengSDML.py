'''
Created on 23 Nov. 2017

@author: Alvin UTS
'''

from metric_learn import SDML
from sklearn.manifold import TSNE
import pandas as pd
import sklearn.cluster as cluster
import numpy as np


# do clustering algorithm
def clusteringAndTSNE(X):
   
    #get Cluster ID for each record
    y_pred = cluster.KMeans(n_clusters=3).fit_predict(X)
    
    #add cluster into a new column
    X['cluster'] = y_pred  
    #write to a csv file  
    X.to_csv('data/clustering_results.csv', index=False) 
  
    #PCA processing
    del X['cluster']
    # export json file and transform num_classes to 2 dim by pca
    pca = TSNE(n_components=2)
    X_trans = pca.fit_transform(X)
    
    x_trans = pd.DataFrame(data=X_trans,columns=['x','y'])
    
    diag_dict = pd.read_csv('data/outcome_diags_desc.csv')
    frames = [x_trans,  diag_dict ]  
    visualData =  pd.concat(frames, axis=1)
    visualData['cluster'] = y_pred  
    
    visualData = visualData.rename(columns = {'Unnamed: 0':'id'})    
    visualData.to_csv('data/outcome_visual.csv', index=False) 

def update():
    
    df_label = pd.read_csv('data/outcome_labels.csv')
    print "df_label", '\n', df_label
    
    df_data = pd.read_csv('data/features_rep.csv')
    #df_data = pd.read_csv('data/alvin_rep.csv')
    print "df_data",'\n', df_data 
    
    #print("df_data", df_data)
    
    #get unique row ids
    rowIDLIst = pd.concat([df_label.id1,df_label.id2],axis = 0).unique().tolist()
    #rowIDLIst2 = pd.concat([df_label.id1,df_label.id2],axis = 1)
    print "rowIDLIst",'\n', rowIDLIst
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
   
    print "cmatrixShape", '\n', cmatrix.shape 
    
    trainedData = []
    
    for rid in rowIDLIst:
        row = df_data.iloc[[rid]]
        #print "row","\n",row
        #print "rowType","\n",type(row)
        trainedData.append(row)
        #print "trainedData","\n", trainedData
        
    #print "typetrainedData1", '\n', len(trainedData)
    
    trainedData = pd.concat(trainedData,axis = 0).as_matrix()   
    
    #print "trainedData2", "\n", trainedData
    print "trainedData.shape",'\n', trainedData.shape
    metric = SDML().fit(trainedData, cmatrix)  
    
    newData = metric.transform(df_data) 
    
    clusteringAndTSNE(newData)
    
    
if __name__ == "__main__":
    update()
