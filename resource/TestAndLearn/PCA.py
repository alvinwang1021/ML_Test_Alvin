'''
Created on 21 Nov. 2017

@author: Alvin UTS
'''

import numpy as np
from sklearn.manifold import TSNE
X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
print type(X)
print X, '\n'
X_embedded = TSNE(n_components=2).fit_transform(X)
print type(X_embedded)
print X_embedded, '\n'

print X_embedded.shape, '\n'

