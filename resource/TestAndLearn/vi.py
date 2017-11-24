'''
Created on 23 Nov. 2017

@author: Alvin UTS
'''
import numpy as np
import matplotlib.pyplot as plt

"""ax = plt.subplot(111)

t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2*np.pi*t)
line, = plt.plot(t, s, lw=2)

plt.annotate('local max', xy=(0, 1), xytext=(3, 1.5),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )

plt.ylim(-2,2)
plt.show()"""

x=[0.15, 0.3, 0.45, 0.6, 0.75]
y=[2.56422, 3.77284,3.52623,3.51468,3.02199]
n=[58,651,393,203,123]

fig, ax = plt.subplots()
ax.scatter(x, y)

for i, txt in enumerate(n):
    ax.annotate(txt, (x[i],y[i]))
plt.show()