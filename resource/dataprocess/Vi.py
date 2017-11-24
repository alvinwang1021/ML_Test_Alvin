'''
Created on 22 Nov. 2017

@author: Alvin UTS
'''
import matplotlib.pyplot as plt 

def showCluster(dataSet, labels):  
      
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']  

    # draw all samples  
    for i in range(len(dataSet)):  
        markIndex = int(labels[i])
 
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])  
        plt.annotate(i, (dataSet[i, 0], dataSet[i, 1]),
            ) 
    plt.show()  
