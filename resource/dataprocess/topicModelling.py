'''
Created on 24 Nov. 2017

@author: Alvin UTS
'''
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd

def topic(data, num_topics=5):
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
    result = lda.fit_transform(data)
    
    cols = ['topic_{}'.format(i) for i in range(len(result[0]))]
    df_reperent = pd.DataFrame(result, columns=cols)
    return df_reperent
    
    #df_data = pd.read_csv('data/features_rep.csv')
    df_reperent.to_csv('../TestAndLearn/data/alvin_rep.csv', index=False) 