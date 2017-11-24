'''
Created on 22 Nov. 2017

@author: Alvin UTS
'''

from sklearn.cluster import KMeans

def kmeans_al(data):
    kmeansResult = KMeans(n_clusters=3, random_state=0).fit(data)
    print 'Kmeans Clustering Done', '\n'
    return kmeansResult